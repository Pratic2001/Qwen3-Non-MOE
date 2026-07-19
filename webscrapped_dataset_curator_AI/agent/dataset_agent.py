#!/usr/bin/env python3
"""
dataset_agent.py

A self-directed agent that never runs out of data: it uses a local Ollama
model to plan search queries per category, calls the web_scraper MCP server
to search + fetch + clean pages, filters/dedupes with the same rules as
build_dataset.py, and writes JSONL shards + a manifest.json in the exact
format your existing pack_dataset.py already consumes.

Two output modes:
    --mode pretrain   -> {"text": ..., "source": ..., "category": ...}
                         (matches build_dataset.py / pack_dataset.py)
    --mode sft        -> {"prompt": ..., "thinking": "", "answer": ...,
                          "source": ..., "category": ...}
                         (matches download_sft_data.py / pack_sft_data.py;
                          "thinking" is left empty since raw web pages don't
                          contain a CoT trace -- see README for how to
                          backfill it with an Ollama-generated rationale)

Usage:
    ollama pull llama3.1                     # or any instruct model you like
    python dataset_agent.py --target-size 500MB --mode pretrain \
        --categories web,knowledge,reasoning --out-dir ./data

    python dataset_agent.py --target-size 200MB --mode sft \
        --categories math,code,reasoning,science --out-dir ./sft_data

Requires the web_scraper_mcp server (see ../web_scraper_mcp/server.py) and
an Ollama daemon running locally (default http://localhost:11434).
"""

import argparse
import asyncio
import itertools
import json
import logging
import os
import sys
import time
import urllib.parse
from contextlib import AsyncExitStack
from typing import Optional

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dataset_agent")

# Monotonic id shared by every Ollama call (batcher submit, per-doc
# generate) so concurrent/serialized requests can be told apart in the
# logs when tracking down a hang -- e.g. "call #7 still waiting" tells you
# exactly which in-flight request is stuck, instead of a wall of
# identical-looking log lines.
_ollama_call_seq = itertools.count(1)

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

sys.path.insert(0, os.path.dirname(__file__))
from quality import (
    ExactDedup, NearDedup, RunState, ShardWriter,
    passes_prose_quality_filter, passes_code_quality_filter,
    passes_transcript_quality_filter, passes_sft_pair_quality_filter,
)
from topics import TOPIC_SEEDS, HUB_SEARCH_KEYWORDS
import public_sources

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")

# When True, judge_quality returns True and extract_sft_pair returns None
# without ever invoking Ollama. Toggled by --fast-heuristics on the CLI.
# Reads of this flag are unconditional, so callers don't need to know
# whether the bypass is on -- the same code path that handles a
# failed/empty Ollama response also handles the bypass.
LLM_BYPASS: bool = False

# Length floor (in chars, combined prompt+answer) used ONLY for rows that
# already carry their own labeled SFT pair (mode=="sft" + extra.prompt/
# extra.answer present -- see _process_article). Deliberately separate from
# --min-doc-chars: that flag feeds passes_prose_quality_filter, which is
# tuned for scraped web articles (default 500 chars) and will reject
# essentially 100% of a short-form Q&A dataset (math problems, one-line
# code fixes, ...) before a single row ever reaches the LLM judge. Set via
# --min-sft-pair-chars.
MIN_SFT_PAIR_CHARS: int = 20


# ---------------------------------------------------------------------------
# Ollama batching layer
# ---------------------------------------------------------------------------
# A local Ollama daemon processes one request at a time on a single GPU, so
# naively fanning N concurrent judge/SFT-extract calls at it just queues them
# and the throughput ceiling is ~1 doc/s. OllamaBatcher coalesces a small
# window of incoming requests into a single prompt that asks the model for
# a JSON array of verdicts (one per doc) and fans the response back out to
# the per-doc futures. With batch_size=8, throughput on a local 8B model
# jumps ~5-8x on the judge phase.

# ---------------------------------------------------------------------------
# Defensive JSON parsing for Ollama responses
# ---------------------------------------------------------------------------
# Ollama's format="json" constrains decoding but doesn't guarantee the
# model emits ONLY the JSON object -- smaller/quantized models in
# particular routinely wrap it in ```json fences, prefix it with "Here is
# the mapping:", or occasionally emit nothing at all for a turn. A bare
# json.loads(raw) on that is what produced the opaque "Expecting value:
# line 1 column 1 (char 0)" report -- which tells you json.loads failed
# but nothing about what Ollama actually sent back. _extract_json_object
# strips the common wrapping patterns before giving up, and every caller
# now logs a preview of the raw response on failure instead of just the
# exception text.

def _strip_json_fences(text: str) -> str:
    """Strip a wrapping ```json ... ``` or ``` ... ``` markdown fence, if
    present. Returns text unchanged (just whitespace-stripped) if no
    fence is found."""
    text = text.strip()
    if not text.startswith("```"):
        return text
    lines = text.split("\n")
    lines = [ln for ln in lines if not ln.strip().startswith("```")]
    return "\n".join(lines).strip()


def _extract_json_object(raw: Optional[str]) -> Optional[dict]:
    """Best-effort parse of a single JSON object out of a raw Ollama
    response. Tries, in order: (1) direct parse after fence-stripping,
    (2) slicing out the outermost {...} span in case the model added
    prose before/after it. Returns None (never raises) if nothing
    parseable is found -- callers already have a fallback path for that,
    they just need to know it happened."""
    if not raw:
        return None
    text = _strip_json_fences(raw)
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
        return data if isinstance(data, dict) else None
    except Exception:
        return None


class OllamaBatcher:
    """Async batching wrapper for ollama_generate.

    Callers `submit(prompt, system)` and get back the model's text response
    as if they'd called ollama_generate directly; the batcher hides the
    coalescing. One batcher per (system-prompt flavor), because the system
    prompt is fixed for the lifetime of a batcher's run (judge, SFT extract,
    etc.) and varies across them.

    Flushing policy: a batch flushes as soon as EITHER `batch_size` requests
    have arrived OR `flush_interval` seconds have elapsed since the first
    request in the current batch. This balances "fill the batch to amortize
    the model call" against "don't artificially wait when there's a steady
    trickle of requests."

    If batch_size == 1 the batcher degenerates to a passthrough (one
    ollama_generate call per submit). That's intentional -- it lets users
    --no-judge-batching without any code path divergence."""

    def __init__(self, batch_size: int = 8, flush_interval: float = 1.0,
                 submit_timeout: float = 180.0, name: str = "batcher"):
        self.name = name
        self.batch_size = max(1, batch_size)
        self.flush_interval = max(0.05, flush_interval)
        # Upper bound on how long submit() will wait for a result before
        # giving up and returning None. Without this, a hung Ollama call
        # (e.g. model never finishes loading, network stalls) makes every
        # caller of judge_quality() block forever -- which is what the
        # "hanging while ollama judge is supposed to work" symptom was.
        self.submit_timeout = max(1.0, submit_timeout)
        self._queue: asyncio.Queue = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None
        self._shutdown = False
        self._all_batches: list[asyncio.Task] = []
        self._submit_seq = itertools.count(1)

    def _ensure_running(self):
        if self._task is None or self._task.done():
            log.info(f"[{self.name}] starting background _run task "
                     f"(was {'never started' if self._task is None else 'done/dead'})")
            self._task = asyncio.create_task(self._run())
        else:
            log.debug(f"[{self.name}] _run task already alive, not restarting")

    def _resolve_drain_queue(self):
        """Resolve every future still sitting in the queue to None. Called
        on shutdown and when _run exits for any reason, so a caller awaiting
        a future that arrived after shutdown was signalled doesn't block
        forever. Idempotent. Sentinels (fut is None) are silently dropped
        -- they're just wake-up tokens, not real pending requests."""
        while True:
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            _, _, fut = item
            if fut is not None and not fut.done():
                fut.set_result(None)

    async def submit(self, prompt: str, system: str) -> Optional[str]:
        """Enqueue a (prompt, system) and await the model's text response.
        Returns None if the batcher is shutting down, the request times out,
        or the underlying ollama call fails (callers fall back to their own
        parse-failure path, which is the right thing to do)."""
        sid = next(self._submit_seq)
        if self._shutdown:
            log.info(f"[{self.name} #{sid}] submit() called after shutdown; returning None")
            return None
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        await self._queue.put((prompt, system, fut))
        log.info(f"[{self.name} #{sid}] queued (qsize={self._queue.qsize()}); ensuring _run task is alive")
        self._ensure_running()
        t0 = time.time()
        try:
            result = await asyncio.wait_for(fut, timeout=self.submit_timeout)
            log.info(f"[{self.name} #{sid}] resolved after {time.time()-t0:.1f}s "
                     f"({'empty/None' if result is None else f'{len(result)} chars'})")
            return result
        except asyncio.TimeoutError:
            log.warning(f"[{self.name} #{sid}] submit timed out after "
                        f"{self.submit_timeout:.0f}s; resolving to None "
                        f"so the caller can fall through")
            if not fut.done():
                fut.set_result(None)
            return None

    async def _run(self):
        """Background coroutine: drain the queue into batches and dispatch
        each batch as one ollama_generate call. Crashes in one batch
        (exception in ollama_generate, malformed response) resolve the
        per-doc futures to None so the caller's existing try/except
        fallbacks fire -- no doc silently disappears."""
        log.info(f"[{self.name}] _run task started")
        pending: list[tuple[str, str, asyncio.Future]] = []
        window_deadline: Optional[float] = None  # absolute time at which to flush whatever we have
        try:
            while not (self._shutdown and self._queue.empty() and not pending):
                # Compute how long to wait for the next item. While the queue
                # is empty AND pending is empty, idle forever (caller will
                # wake us by submitting). Once we have at least one pending
                # item, race a deadline (set when the first pending item
                # arrived) against "queue has another item ready now."
                if not pending and self._queue.empty():
                    # Park until something arrives or shutdown is signalled.
                    if self._shutdown:
                        break
                    log.debug(f"[{self.name}] idle, parked on queue.get()")
                    item = await self._queue.get()
                    if self._is_sentinel(item):
                        continue
                    pending.append(item)
                    window_deadline = asyncio.get_running_loop().time() + self.flush_interval
                    # Fast path: if the batch is already full, flush now.
                    if len(pending) >= self.batch_size:
                        log.debug(f"[{self.name}] batch full ({len(pending)}/{self.batch_size}), flushing immediately")
                        await self._flush(pending)
                        pending = []
                        window_deadline = None
                    continue

                # We have at least one pending item. Decide whether to flush:
                # - batch is full
                # - shutdown signalled (drain whatever's left)
                # - flush_interval has elapsed since the first pending item arrived
                now = asyncio.get_running_loop().time()
                interval_elapsed = window_deadline is not None and now >= window_deadline
                should_flush = (
                    len(pending) >= self.batch_size
                    or (self._shutdown and pending)
                    or interval_elapsed
                )
                if should_flush:
                    log.debug(f"[{self.name}] flushing {len(pending)} item(s) "
                              f"(full={len(pending) >= self.batch_size}, "
                              f"shutdown={self._shutdown}, interval_elapsed={interval_elapsed})")
                    await self._flush(pending)
                    pending = []
                    window_deadline = None
                    continue

                # Otherwise race the deadline against the next queue item.
                if not self._queue.empty():
                    item = self._queue.get_nowait()
                    if self._is_sentinel(item):
                        continue
                    was_empty = not pending
                    pending.append(item)
                    if was_empty:
                        # This is the first item of a new window. The other
                        # branch that can add a "first" item (the idle/parked
                        # queue.get() above) sets window_deadline when it
                        # does so; this fast path must do the same, or
                        # window_deadline stays None forever, should_flush
                        # is permanently False, and the final "wait on
                        # deadline" branch below is skipped too (it only
                        # runs when window_deadline is not None) -- the loop
                        # then spins with no await at all, pinning a CPU
                        # core and hanging silently forever.
                        window_deadline = asyncio.get_running_loop().time() + self.flush_interval
                    continue
                # Wait up to (deadline - now) for an item, whichever first.
                if window_deadline is not None:
                    remaining = max(0.0, window_deadline - now)
                    try:
                        item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                    except asyncio.TimeoutError:
                        continue
                    if self._is_sentinel(item):
                        continue
                    pending.append(item)
        except asyncio.CancelledError:
            # shutdown() cancelled us mid-flush (drain_timeout elapsed).
            # Make sure the in-flight flush's futures aren't left dangling.
            log.info(f"[{self.name}] _run task cancelled ({len(pending)} pending resolved to None)")
            for _, _, fut in pending:
                if not fut.done():
                    fut.set_result(None)
            raise
        except Exception as e:
            # Anything else escaping the loop above (a bug, not a
            # transient per-batch failure -- those are already caught
            # inside _flush) used to kill this task silently: asyncio
            # just stores the exception on the Task and moves on unless
            # someone awaits/retrieves it, which nobody here does. Every
            # submit() after that point would enqueue into a queue that
            # NOTHING is reading anymore and sit waiting the full
            # submit_timeout before giving up -- which looks exactly like
            # a permanent hang with zero log output. Log it loudly here so
            # that failure mode is no longer silent.
            log.error(f"[{self.name}] _run task CRASHED: {type(e).__name__}: {e}", exc_info=True)
            for _, _, fut in pending:
                if not fut.done():
                    fut.set_result(None)
            self._resolve_drain_queue()
            raise
        finally:
            # Always drain whatever's in the queue on exit, so any submit()
            # that landed between shutdown's _shutdown=True and this point
            # gets its future resolved instead of waiting forever.
            for _, _, fut in pending:
                if not fut.done():
                    fut.set_result(None)
            self._resolve_drain_queue()
            log.info(f"[{self.name}] _run task exiting")

    @staticmethod
    def _is_sentinel(item: tuple) -> bool:
        """shutdown() pushes a (prompt="", _, fut=None) sentinel to wake a
        parked _run loop. Real submits always carry a non-empty prompt AND
        a non-None future, so either condition identifies a sentinel."""
        prompt, _, fut = item
        return prompt == "" or fut is None

    async def _flush(self, items: list[tuple[str, str, asyncio.Future]]):
        """Send one batched prompt to ollama_generate, parse the response,
        and resolve each per-doc future with its own slot's text. Items
        that don't have a parseable slot get None -- the caller's existing
        except path picks that up."""
        prompts = [p for p, _, _ in items]
        # All items in one batch share the same system prompt in practice
        # (each batcher is bound to one role), but we use the first item's
        # system as the canonical one and warn loudly if any other item
        # disagrees.
        systems = {s for _, s, _ in items}
        if len(systems) > 1:
            log.warning(f"[ollama-batcher] batch had {len(systems)} distinct "
                        f"system prompts; using the first. This shouldn't "
                        f"happen if each batcher is bound to one role.")
        system = items[0][1]
        n = len(prompts)
        log.info(f"[{self.name}] _flush: dispatching batch of {n} "
                 f"(mode={'passthrough' if self.batch_size == 1 or n == 1 else 'batched'})")

        if self.batch_size == 1 or n == 1:
            # Passthrough: no batched prompt, just dispatch one at a time
            # to preserve the original per-doc semantics. (batch_size==1
            # is the "I turned batching off" knob.)
            for i, (prompt, _, fut) in enumerate(items):
                if fut.done():
                    continue
                log.debug(f"[{self.name}] _flush: passthrough item {i+1}/{n} -> ollama_generate")
                try:
                    resp = await ollama_generate(prompt, system=system, json_mode=True)
                    fut.set_result(resp)
                except Exception as e:
                    log.warning(f"[ollama-batcher] per-doc call failed: {e}")
                    fut.set_result(None)
            log.debug(f"[{self.name}] _flush: passthrough batch of {n} complete")
            return

        try:
            batched_prompt = self._build_batched_prompt(prompts)
            raw = await ollama_generate(batched_prompt, system=system, json_mode=True)
            slots = self._parse_batched_response(raw, n)
        except Exception as e:
            log.warning(f"[ollama-batcher] batched call failed ({e}); "
                        f"falling back to per-doc dispatch for this batch of {n}")
            slots = None

        if slots is None:
            # Either the call itself failed or the response was unparseable.
            # Fall back to per-doc dispatch so a transient batch failure
            # doesn't lose the whole window -- this preserves the original
            # behavior of judge_quality/extract_sft_pair where each doc got
            # its own try/except.
            log.info(f"[{self.name}] _flush: falling back to per-doc dispatch for {n} item(s)")
            for prompt, _, fut in items:
                if fut.done():
                    continue
                try:
                    resp = await ollama_generate(prompt, system=system, json_mode=True)
                    fut.set_result(resp)
                except Exception as e:
                    log.debug(f"[ollama-batcher] per-doc fallback failed: {e}")
                    fut.set_result(None)
            return

        for (_, _, fut), slot in zip(items, slots):
            if fut.done():
                continue
            fut.set_result(slot)
        log.debug(f"[{self.name}] _flush: batched call complete, {n} slots resolved")

    def _build_batched_prompt(self, prompts: list[str]) -> str:
        parts = [f"Documents: {len(prompts)} total. "
                 f"Respond with a JSON array of length {len(prompts)}, one "
                 f"entry per document in order, using the same JSON shape "
                 f"you would for a single document. Do not merge or skip entries."]
        for i, p in enumerate(prompts):
            parts.append(f"---DOC {i}---\n{p}\n---END {i}---")
        parts.append(f"\nRespond with a JSON array of length {len(prompts)}.")
        return "\n".join(parts)

    def _parse_batched_response(self, raw: str, n: int) -> Optional[list[Optional[str]]]:
        """Try to extract a length-N JSON array from `raw`. The model may
        wrap it in markdown fences (```json ... ```) or chatter around it;
        we strip those and try json.loads. On success returns a list of
        length N (entries coerced to str; missing entries become None).
        On any failure returns None so the caller falls back to per-doc
        dispatch."""
        if not raw:
            return None
        text = _strip_json_fences(raw)
        # Find the outermost JSON array -- the model occasionally prefixes
        # with prose like "Here is the array: [...]".
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1 or end <= start:
            return None
        text = text[start:end + 1]
        try:
            data = json.loads(text)
        except Exception:
            return None
        if not isinstance(data, list):
            return None
        slots: list[Optional[str]] = []
        for i in range(n):
            if i < len(data) and data[i] is not None:
                slots.append(data[i] if isinstance(data[i], str) else json.dumps(data[i]))
            else:
                slots.append(None)
        return slots

    async def shutdown(self, drain_timeout: float = 5.0):
        """Stop accepting new requests, flush whatever's already in the
        queue (best effort, bounded by `drain_timeout`), and wait for the
        background _run coroutine to finish. Safe to call multiple times."""
        self._shutdown = True
        # Push a sentinel so _run wakes from any pending get() even if the
        # queue is empty. fut=None is the explicit "this is a wake-up
        # token, not a real pending request" marker -- _resolve_drain_queue
        # and the _is_sentinel check both treat None as such.
        await self._queue.put(("", "", None))
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=drain_timeout)
            except asyncio.TimeoutError:
                log.warning("[ollama-batcher] shutdown timed out; cancelling run loop")
                self._task.cancel()
                try:
                    await self._task
                except (asyncio.CancelledError, Exception):
                    pass
        # Belt-and-suspenders: even if _run already drained via its finally
        # block, sweep the queue once more so a submit() that landed during
        # the shutdown race above doesn't end up holding a future that
        # nothing will ever resolve.
        self._resolve_drain_queue()
        self._task = None


async def _shutdown_all_batchers(drain_timeout: float = 5.0):
    for b in _BATCHERS:
        try:
            await b.shutdown(drain_timeout=drain_timeout)
        except Exception as e:
            log.warning(f"[ollama-batcher] shutdown error: {e}")


_BATCHERS: list[OllamaBatcher] = []


def _register_batcher(b: OllamaBatcher) -> OllamaBatcher:
    _BATCHERS.append(b)
    return b


# Singleton batchers. Created with placeholder settings; main_async()
# applies the real CLI values at startup. batch_size=1 is the safe
# default (passthrough) -- if startup is skipped for any reason, we
# behave exactly like the pre-batching code.
JUDGE_BATCHER = _register_batcher(OllamaBatcher(batch_size=1, flush_interval=1.0, name="judge-batcher"))
SFT_BATCHER = _register_batcher(OllamaBatcher(batch_size=1, flush_interval=1.0, name="sft-batcher"))


def configure_batcher_settings(batch_size: int, flush_interval: float):
    """Called from main_async() after argparse to apply --judge-batch-size
    and --judge-batch-flush-seconds to both singleton batchers. batch_size
    of 0 or 1 disables batching (passthrough)."""
    for b in (JUDGE_BATCHER, SFT_BATCHER):
        b.batch_size = max(1, batch_size)
        b.flush_interval = max(0.05, flush_interval)


DEFAULT_MIX = {
    "web": 0.35,
    "knowledge": 0.20,
    "reasoning": 0.20,
    "code": 0.15,
    "math": 0.10,
}


# ---------------------------------------------------------------------------
# Ollama calls
# ---------------------------------------------------------------------------

class OllamaModelNotFoundError(Exception):
    """Raised when the configured OLLAMA_MODEL is not in the local Ollama
    install. Caught by callers (judge_quality, extract_sft_pair, plan_queries)
    so they can return a fallback instead of a malformed result. The user
    sees a single actionable error at startup (via _prewarm_ollama), not a
    silent quality drop where the judge always returns True and the
    corpus is built with no LLM gate at all."""


async def _heartbeat(call_id: int, stage: str, interval: float = 15.0):
    """Background task: log a 'still waiting' line every `interval`
    seconds until cancelled. Started right before a potentially slow
    await (e.g. the Ollama HTTP call) and cancelled as soon as it
    returns, so a genuine hang shows up as a repeating log line naming
    the exact call-id and stage instead of just going silent."""
    waited = 0.0
    try:
        while True:
            await asyncio.sleep(interval)
            waited += interval
            log.info(f"[ollama #{call_id}] still waiting on {stage} after {waited:.0f}s...")
    except asyncio.CancelledError:
        pass


async def ollama_generate(prompt: str, system: Optional[str] = None, json_mode: bool = False) -> str:
    call_id = next(_ollama_call_seq)
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        # Without this, Ollama falls back to its default keep_alive (5m of
        # idle time before the model is evicted from VRAM). Phase 1 of a
        # public-source run (raw download to byte budget) makes zero Ollama
        # calls and can easily run for well over 5 minutes on a multi-GB
        # pull, so by the time phase 2 (the filter pass) makes its first
        # real judge call, the model _prewarm_ollama() loaded at startup has
        # already been evicted. -1 means "never unload on idle"; the model
        # only leaves VRAM if this process exits or something else asks
        # Ollama to load a different model.
        "keep_alive": -1,
    }
    if system:
        payload["system"] = system
    if json_mode:
        payload["format"] = "json"
        # Reasoning-capable models (qwen3, deepseek-r1, gpt-oss, ...)
        # default to an internal "thinking" pass. Observed failure modes
        # without this: (1) the model narrates its reasoning in plain
        # prose ("We have a dataset schema with columns... So we need to
        # output JSON...") and the generation budget runs out before it
        # ever emits the actual JSON, so `response` is truncated prose,
        # not JSON; (2) on some Ollama versions the entire completion
        # routes into a separate `thinking` field and `response` comes
        # back completely empty despite HTTP 200. Every json_mode caller
        # here wants machine-parseable structured output, never a
        # reasoning trace, so disable it explicitly. Ollama silently
        # ignores unknown request fields, so this is a harmless no-op on
        # models that don't support extended thinking at all.
        payload["think"] = False
    log.info(f"[ollama #{call_id}] -> POST {OLLAMA_URL}/api/generate model={OLLAMA_MODEL} "
             f"prompt_chars={len(prompt)} json_mode={json_mode}")
    log.debug(f"[ollama #{call_id}] prompt={prompt[:120]!r}...")
    start = time.time()
    # 300s, not 120s: if the model was evicted from VRAM for any reason
    # (Ollama restarted, another model was loaded in between, etc.) this
    # call also has to pay the full cold-load cost documented in
    # _prewarm_ollama (30s-5min), on top of actual inference time. A
    # shorter client-side timeout would abort the HTTP request before
    # Ollama finishes loading, which looks identical to a real hang from
    # the caller's side -- and since every retry pays the same cold-load
    # cost and gets cut off the same way, it can appear to "never load
    # back" even though the server would have finished given the time.
    hb = asyncio.create_task(_heartbeat(call_id, "POST /api/generate (client-side, waiting on Ollama)"))
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            log.debug(f"[ollama #{call_id}] httpx client opened, sending request now")
            resp = await client.post(f"{OLLAMA_URL}/api/generate", json=payload)
            log.info(f"[ollama #{call_id}] <- HTTP {resp.status_code} after "
                     f"{time.time()-start:.1f}s")
            # Detect model-not-found explicitly. Without this branch, the 404
            # body `{"error": "model 'X' not found"}` gets parsed as if it
            # were a success response, the caller gets "" back, and
            # `json.loads("")` raises -- caught and silently swallowed in
            # judge_quality's except, which returns True. So a missing model
            # looks identical to a working one and the quality gate is
            # effectively disabled. The explicit 404 check makes the failure
            # mode loud and actionable.
            if resp.status_code == 404:
                raise OllamaModelNotFoundError(
                    f"Ollama model {OLLAMA_MODEL!r} not found at {OLLAMA_URL}. "
                    f"Run `ollama list` to see installed models, or set "
                    f"OLLAMA_MODEL=... to one of them. To skip the LLM judge "
                    f"entirely, re-run with --no-llm-judge or --fast-heuristics."
                )
            resp.raise_for_status()
            data = resp.json()
            text = data.get("response", "")
            if not text and data.get("thinking"):
                # Belt-and-suspenders: think=False didn't fully suppress it (or
                # this Ollama version always splits response/thinking
                # regardless) and the model's actual answer ended up in the
                # thinking trace. Better to try recovering JSON from it than
                # return nothing and force every caller to fall back.
                log.debug(f"[ollama #{call_id}] response empty but thinking field present "
                          f"({len(data['thinking'])} chars); using it as fallback")
                text = data["thinking"]
    except Exception as e:
        log.warning(f"[ollama #{call_id}] FAILED after {time.time()-start:.1f}s: "
                    f"{type(e).__name__}: {e}")
        raise
    finally:
        hb.cancel()
        try:
            await hb
        except asyncio.CancelledError:
            pass
    log.info(f"[ollama #{call_id}] done in {time.time()-start:.1f}s, "
             f"response_chars={len(text)}")
    log.debug(f"[ollama #{call_id}] response={text[:120]!r}...")
    return text


async def _prewarm_ollama(timeout: float = 300.0) -> bool:
    """Make one tiny inference call to load OLLAMA_MODEL into VRAM, so the
    first real filter-pass call returns in <1s instead of paying the full
    model-load cost (30s-5min depending on model size, which is what was
    causing the filter pass to silently hang for minutes after the first
    heuristic-rejected rows were logged). The prewarm is a no-op if the
    model is already loaded. Returns True on success, False on any error
    (logged, never raised -- callers decide whether a failed prewarm
    should abort the run)."""
    if not OLLAMA_MODEL:
        log.info("prewarm: OLLAMA_MODEL not set; skipping prewarm")
        return False
    log.info(f"prewarm: loading {OLLAMA_MODEL!r} into Ollama (up to {timeout:.0f}s)...")
    t0 = time.time()
    call_id = next(_ollama_call_seq)
    hb = asyncio.create_task(_heartbeat(call_id, f"prewarm POST /api/generate (loading {OLLAMA_MODEL!r})"))
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": OLLAMA_MODEL, "prompt": "ok", "stream": False, "keep_alive": -1},
            )
            if resp.status_code == 404:
                log.error(
                    f"prewarm FAILED: Ollama model {OLLAMA_MODEL!r} is not "
                    f"installed at {OLLAMA_URL}. Run `ollama list` to see "
                    f"installed models, or set OLLAMA_MODEL=... to one of "
                    f"them. To skip the LLM judge entirely, re-run with "
                    f"--no-llm-judge or --fast-heuristics."
                )
                return False
            resp.raise_for_status()
        log.info(f"prewarm: {OLLAMA_MODEL!r} loaded in {time.time()-t0:.1f}s")
        return True
    except OllamaModelNotFoundError as e:
        # Raised by ollama_generate on 404; we bypass ollama_generate here
        # but keep the same exception class for the caller's except list.
        log.error(f"prewarm FAILED: {e}")
        return False
    except Exception as e:
        log.warning(
            f"prewarm failed ({type(e).__name__}: {e}); "
            f"continuing -- first real call may be slow"
        )
        return False
    finally:
        hb.cancel()
        try:
            await hb
        except asyncio.CancelledError:
            pass


async def plan_queries(category: str, recent_topics: list, n: int = 8) -> list:
    """Ask the local model for fresh, specific search queries for a category,
    steering away from topics already covered so the corpus keeps expanding
    instead of circling the same few queries."""
    avoid = ", ".join(recent_topics[-30:]) if recent_topics else "(none yet)"
    system = (
        "You generate web search queries for building a language-model "
        "training corpus. Return ONLY a JSON object: "
        '{"queries": ["...", "..."]}. Queries must be short (3-8 words), '
        "specific, and diverse -- avoid vague single-word queries. Do not "
        "explain your reasoning or add any text before/after the JSON -- "
        "the JSON object must be your entire response."
    )
    prompt = (
        f"Category: {category}\n"
        f"Seed topics for this category: {', '.join(TOPIC_SEEDS.get(category, [category]))}\n"
        f"Recently used queries (avoid repeating/near-duplicating these): {avoid}\n"
        f"Generate {n} new, specific search queries for this category."
    )
    try:
        raw = await ollama_generate(prompt, system=system, json_mode=True)
        data = _extract_json_object(raw)
        if data is None:
            raise ValueError(f"no parseable JSON object in response: {raw[:300]!r}")
        queries = data.get("queries", [])
        queries = [q.strip() for q in queries if isinstance(q, str) and q.strip()][:n]
        log.info(f"[plan:{category}] planner produced {len(queries)} queries: {queries}")
        return queries
    except Exception as e:
        fallback = TOPIC_SEEDS.get(category, [category])[:n]
        log.warning(f"[plan:{category}] plan_queries failed ({e}), falling back to seed topics: {fallback}")
        return fallback


async def judge_quality(text: str, category: str, pair_mode: bool = False) -> bool:
    """LLM-based quality gate, applied AFTER the cheap heuristic filters
    (which catch the obvious junk for free). Only invoked on documents that
    already passed the heuristics, to keep the number of LLM calls bounded.
    Returns True if the model says this is usable training data.

    The actual Ollama call goes through JUDGE_BATCHER, which coalesces a
    small window of concurrent judge calls into a single batched prompt.
    The per-doc contract is unchanged: we get back a JSON object string
    (one per document) and parse {"keep": ...} from it.

    `pair_mode` must be True when `text` is a labeled (prompt, answer) pair
    from a dataset hub, NOT scraped prose. The original system prompt was
    written entirely in terms of "scraped web document" quality (boilerplate,
    ads/nav menus, listicles, machine-translated prose) -- criteria that
    don't apply to a short, already-correct math/code Q&A pair, and were
    rejecting good terse answers ("42", "2x", a one-line fix) as if they
    were thin web content. pair_mode swaps in a judge that only asks whether
    the pair is a valid, coherent, on-topic training example.

    With --fast-heuristics (LLM_BYPASS=True), this short-circuits to True
    without making any Ollama call. The heuristic filters above already
    caught the obvious junk; this just trusts them entirely."""
    if LLM_BYPASS:
        return True
    if pair_mode:
        system = (
            "You judge whether a (prompt, answer) pair is usable instruction-"
            "tuning data. The pair may be very short -- a math problem with a "
            "one-line or purely numeric/symbolic answer, or a one-line code "
            "fix, is normal and GOOD, not low quality. Judge only: (1) is the "
            "answer a correct, on-topic response to the prompt, (2) is the "
            "pair coherent (not garbled, not truncated mid-thought, not "
            "boilerplate/placeholder text). Do NOT reject for brevity alone. "
            f"Category: {category}. Do not explain your reasoning -- respond "
            'with ONLY this JSON object, nothing before or after it: '
            '{"keep": true} or {"keep": false}.'
        )
    else:
        system = (
            "You judge whether a scraped web document is high-quality training "
            "data for a language model. Reject: boilerplate, ads/nav menus, "
            "listicles with no substance, spam, incoherent machine-translated "
            "text, or content that's mostly links/references with little prose. "
            f"Accept substantive {category} content. Do not explain your "
            "reasoning -- respond with ONLY this JSON object, nothing before "
            'or after it: {"keep": true} or {"keep": false}.'
        )
    snippet = text[:3000]
    log.info(f"[judge:{category}] calling judge_quality (pair_mode={pair_mode}, "
             f"snippet_chars={len(snippet)})")
    t0 = time.time()
    try:
        raw = await JUDGE_BATCHER.submit(snippet, system)
        if raw is None:
            # Batcher returned no per-doc slot (parse failure or shutdown);
            # treat it like any other judge-call failure and default to keep.
            raise RuntimeError("judge batcher returned no slot for this document")
        data = _extract_json_object(raw)
        if data is None:
            raise ValueError(f"no parseable JSON object in judge response: {raw[:300]!r}")
        keep = bool(data.get("keep", False))
        log.info(f"[judge:{category}] keep={keep} ({time.time()-t0:.1f}s)")
        return keep
    except Exception as e:
        # If the judge fails/times out, don't block the pipeline on it --
        # fall back to trusting the heuristic filters alone.
        log.warning(f"[judge:{category}] judge call failed after {time.time()-t0:.1f}s "
                    f"({e}), defaulting to keep=True")
        return True


async def extract_sft_pair(text: str, category: str) -> Optional[dict]:
    """For --mode sft: turn a scraped article into a (prompt, answer) pair
    by having the model pose a question the article answers, and produce a
    concise answer grounded in the text. Returns None if the article doesn't
    cleanly support this (e.g. pure narrative with no answerable question).

    Like judge_quality, this goes through SFT_BATCHER for amortized
    throughput on local Ollama. The per-doc contract is unchanged: we get
    back a single JSON object string.

    With --fast-heuristics (LLM_BYPASS=True), this short-circuits to None
    and the row is rejected as "no sft pair extractable" -- the heuristic
    filters aren't going to invent a Q/A pair for us. This is the same
    behavior as if the model had returned {"prompt": null}."""
    if LLM_BYPASS:
        return None
    system = (
        "You convert a source article into ONE high-quality instruction-"
        "tuning example. Ask a specific, well-posed question that the "
        "article answers, then answer it accurately using ONLY information "
        "in the article, in your own words (do not quote the article "
        "verbatim). Do not explain your reasoning -- respond with ONLY "
        'this JSON object, nothing before or after it: '
        '{"prompt": "...", "answer": "..."} or {"prompt": null} if no good '
        "question/answer pair exists in this text."
    )
    try:
        raw = await SFT_BATCHER.submit(text[:6000], system)
        if raw is None:
            log.debug(f"[sft:{category}] batcher returned no slot for this doc")
            return None
        data = _extract_json_object(raw)
        if data is None:
            log.warning(f"[sft:{category}] no parseable JSON object in extraction "
                         f"response, rejecting row: {raw[:300]!r}")
            return None
        if not data.get("prompt") or not data.get("answer"):
            log.debug(f"[sft:{category}] no usable Q/A pair in article")
            return None
        log.debug(f"[sft:{category}] extracted pair, prompt={data['prompt'][:80]!r}...")
        return {"prompt": data["prompt"].strip(), "thinking": "", "answer": data["answer"].strip()}
    except Exception as e:
        log.warning(f"[sft:{category}] extraction failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Dataset schema inference (LLM-driven column mapping)
# ---------------------------------------------------------------------------
# The static column-name heuristics in public_sources.py (_PROMPT_COLUMNS,
# _ANSWER_COLUMNS, ...) cover common cases but miss anything with an
# unfamiliar schema -- that's exactly what happened with MATH-500's
# "problem" column not being in the candidate list. Rather than keep
# growing that list by hand every time a new dataset trips it up, ask
# Ollama to look at one sample row per dataset/config and name the actual
# columns itself. This runs ONCE per (dataset_id, config) -- not once per
# row, which would make column mapping another per-row Ollama bottleneck
# on top of the judge/SFT-extract calls -- and is cached for the life of
# the process. It's a hint, never the only path: row_to_record always
# falls back to the static heuristics if the hint is missing, wrong, or
# doesn't resolve to real content in a given row.

_COLUMN_MAPPING_CACHE: dict = {}  # (dataset_id, config) -> mapping dict | None


def _truncate_val(v, n: int = 300) -> str:
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, default=str)
    s = s.strip()
    return s[:n] + ("..." if len(s) > n else "")


async def infer_column_mapping(dataset_id: str, config: Optional[str],
                                columns: list, sample_row: dict) -> Optional[dict]:
    """Ask Ollama which columns in a dataset row hold the prompt, the
    answer, a chat-format conversation, or generic free-text -- shown ONE
    concrete example row (not just the bare column names) so the model is
    mapping against actual content, not guessing from names alone (a
    column named "output" could be code, a numeric answer, or a full
    article depending on the dataset; only the value tells you). Returns a
    dict like {"prompt_col": "problem", "answer_col": "answer",
    "conversation_col": None, "text_col": None} using EXACT column names
    from `columns` (case-normalized against the real column list so a
    slightly-off-case model answer still resolves), or None if inference
    isn't possible/enabled/useful. Results are cached per (dataset_id,
    config) so a dataset with many rows or configs only gets asked once."""
    cache_key = (dataset_id, config)
    if cache_key in _COLUMN_MAPPING_CACHE:
        return _COLUMN_MAPPING_CACHE[cache_key]
    if LLM_BYPASS:
        _COLUMN_MAPPING_CACHE[cache_key] = None
        return None

    sample_preview = {k: _truncate_val(v) for k, v in list(sample_row.items())[:20]}
    has_content = any(
        v is not None and str(v).strip() not in ("", "nan", "NaN")
        for v in sample_row.values()
    )
    if not has_content:
        # First row was entirely empty/null across every column -- nothing
        # for the model to look at, and an empty example is worse than no
        # example (it invites guessing from column names alone, which is
        # exactly what the static heuristics already do for free). Skip
        # the call rather than waste it.
        log.info(f"[schema:{dataset_id}] sample row is empty across all columns, "
                  f"skipping LLM mapping -- static heuristics only")
        _COLUMN_MAPPING_CACHE[cache_key] = None
        return None

    cfg_label = f":{config}" if config else ""
    log.info(f"[schema:{dataset_id}{cfg_label}] inferring column mapping from example row: "
             f"{json.dumps(sample_preview, ensure_ascii=False)}")

    system = (
        "You analyze the schema of a machine-learning training dataset by "
        "looking at ONE concrete example row from it (not just column "
        "names -- the actual values matter, since a column's real content "
        "is the only reliable signal for what it holds). Identify which "
        "column (if any) holds the question/instruction/prompt, which "
        "holds the answer/response/solution, which holds a chat-format "
        "conversation (a list of turn objects with role/from + content/"
        "value keys), and which holds generic free-form document text "
        "(only relevant if the row is NOT a Q&A pair, e.g. a plain "
        "article/passage). Use EXACT column names from the list given, or "
        "null if none fits -- never invent a name that isn't in the list. "
        "Do not explain your reasoning or narrate your analysis -- respond "
        'with ONLY this JSON object, nothing before or after it: '
        '{"prompt_col": "..."|null, '
        '"answer_col": "..."|null, "conversation_col": "..."|null, '
        '"text_col": "..."|null}'
    )
    prompt = (
        f"Columns: {columns}\n"
        f"Here is one example row from this dataset (values truncated for length):\n"
        f"{json.dumps(sample_preview, ensure_ascii=False, indent=2)}"
    )
    mapping = None
    try:
        raw = await ollama_generate(prompt, system=system, json_mode=True)
        data = _extract_json_object(raw)
        if data is None:
            raise ValueError(f"no parseable JSON object in response: {raw[:300]!r}")
        col_lookup = {c.lower(): c for c in columns}
        resolved = {}
        for key in ("prompt_col", "answer_col", "conversation_col", "text_col"):
            val = data.get(key)
            if isinstance(val, str) and val.strip().lower() in col_lookup:
                # Snap to the real column casing so downstream
                # case-insensitive lookups always have something concrete
                # even if the model echoed the name with different case.
                resolved[key] = col_lookup[val.strip().lower()]
            else:
                resolved[key] = None
        mapping = resolved if any(resolved.values()) else None
        log.info(f"[schema:{dataset_id}{cfg_label}] inferred column mapping: {mapping}")
    except Exception as e:
        log.warning(f"[schema:{dataset_id}] column mapping inference failed "
                     f"({e}); falling back to static column heuristics")
        mapping = None

    _COLUMN_MAPPING_CACHE[cache_key] = mapping
    return mapping


def _infer_column_mapping_sync(dataset_id: str, config: Optional[str],
                                columns: list, sample_row: dict) -> Optional[dict]:
    """Sync wrapper for infer_column_mapping, passed as the column_mapper
    callback into public_sources.stream_hf_dataset / fetch_kaggle_dataset_
    rows. Those generators run inside asyncio.to_thread (see
    _drain_dataset_rows in run_public_sources_for_category) -- a plain OS
    thread with no event loop of its own -- so asyncio.run() here is
    safe; this is never called from inside the main event loop."""
    try:
        return asyncio.run(infer_column_mapping(dataset_id, config, columns, sample_row))
    except Exception as e:
        log.warning(f"[schema:{dataset_id}] column mapping sync wrapper failed: {e}")
        return None


# ---------------------------------------------------------------------------
# MCP tool wrappers
# ---------------------------------------------------------------------------

class ScraperClient:
    def __init__(self, session: ClientSession):
        self.session = session

    async def search(self, query: str, max_results: int = 8) -> list:
        result = await self.session.call_tool("web_search", {"query": query, "max_results": max_results})
        if getattr(result, "isError", False):
            # MCP-level tool error (e.g. the server raised before it could
            # even build a response) -- surface it instead of letting the
            # caller mistake this for "no results."
            raise RuntimeError(f"web_search tool error: {_first_json(result)}")
        parsed = _first_json(result)
        # Defensive: if structuredContent wasn't available for some reason
        # and we fell back to reconstructing from content blocks, a single
        # search hit collapses indistinguishably from a bare dict (same as
        # extract_article's return shape). web_search always means "a list
        # of hits," so re-wrap a lone dict rather than let it be mistaken
        # for a malformed response downstream.
        if isinstance(parsed, dict):
            parsed = [parsed]
        return parsed

    async def extract(self, url: str) -> dict:
        """Format-agnostic extraction: HTML, PDF, DOCX/PPTX/XLSX, images,
        and video/audio (transcripts) all come back through this one call
        now, dispatched server-side by extract_content."""
        result = await self.session.call_tool("extract_content", {"url": url})
        return _first_json(result)

    async def deep_crawl(self, seed_url: str, max_pages: int = 10, max_depth: int = 1,
                          keywords: str = "") -> list:
        """Harvest multiple same-domain pages from one seed URL via the
        crawl4ai-backed deep_crawl tool -- one call yields many already-
        extracted documents instead of one extract() per URL. Use for
        domains already known to be relevant (a hit's own domain looked
        promising, a docs site, a wiki), not for open-ended discovery --
        that's what search() is for.
        Returns a list of extract_content-shaped dicts (never raises for
        "crawl4ai not installed" -- that comes back as a single-item list
        with an "error" key, same convention as search())."""
        result = await self.session.call_tool(
            "deep_crawl",
            {"seed_url": seed_url, "max_pages": max_pages, "max_depth": max_depth,
             "keywords": keywords},
        )
        if getattr(result, "isError", False):
            raise RuntimeError(f"deep_crawl tool error: {_first_json(result)}")
        parsed = _first_json(result)
        if isinstance(parsed, dict):
            parsed = [parsed]
        return parsed or []


def _first_json(tool_result):
    """Reconstruct a tool's actual return value from a CallToolResult.

    IMPORTANT: the MCP SDK's default content serialization
    (_convert_to_content) splits a list-returning tool's output into ONE
    SEPARATE CONTENT BLOCK PER LIST ITEM, not a single block containing the
    whole JSON array. Reading only content[0] -- what this function used to
    do -- silently truncates every multi-result list down to just its first
    element, with no error anywhere. That's why `web_search` hits kept
    showing up client-side as a single dict instead of a list of 8.

    `structuredContent` doesn't have that problem: FastMCP auto-generates
    an output schema from the tool's return-type annotation (e.g.
    `-> list[dict]`) and stores the real structured value there as
    {"result": <value>}, fully intact. Prefer that; only fall back to
    reassembling content blocks (which works fine for single-dict-return
    tools like extract_article, since those never get split) if
    structuredContent isn't present.
    """
    sc = getattr(tool_result, "structuredContent", None)
    if isinstance(sc, dict) and "result" in sc:
        return sc["result"]

    parsed = []
    for block in tool_result.content:
        if not hasattr(block, "text"):
            continue
        try:
            parsed.append(json.loads(block.text))
        except Exception:
            parsed.append(block.text)
    if not parsed:
        return None
    return parsed[0] if len(parsed) == 1 else parsed


# ---------------------------------------------------------------------------
# Main crawl loop
# ---------------------------------------------------------------------------

def _quality_filter_for(content_type: Optional[str], text: str, url: str, min_doc_chars: int) -> tuple:
    """Dispatch to the right quality bar for what extract_content actually
    returned. content_type comes back from the server (html/pdf/docx/pptx/
    xlsx/csv/image/video/audio/text); category alone isn't enough to know
    this anymore since a "code" category doc might legitimately be a PDF
    spec or a transcript of a talk, not just a source file. Returns
    (passed, reason) -- reason is None on pass."""
    if content_type in ("video", "audio"):
        return passes_transcript_quality_filter(text, min_doc_chars)
    source_ext = os.path.splitext(url.split("?")[0].split("#")[0])[1].lower()
    if source_ext in {".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java",
                       ".c", ".h", ".cpp", ".hpp", ".rb", ".php", ".sh", ".sql"}:
        return passes_code_quality_filter(text, url, min_doc_chars)
    return passes_prose_quality_filter(text, min_doc_chars)


async def _process_article(article: dict, url: str, category: str, mode: str,
                            min_doc_chars: int, use_llm_judge: bool, exact_dedup: ExactDedup,
                            near_dedup, writer: ShardWriter, byte_budget: int,
                            write_lock: asyncio.Lock, counters: dict) -> bool:
    """Filter + (maybe) write one already-extracted article/row. Returns
    True if a record was written. Shared by both the live-scrape path
    (_process_hit, article comes from extract_content) and the public
    dataset-hub path (_process_public_row, article comes from a HF/Kaggle
    row already normalized to the same shape) -- everything downstream of
    "I have text and a content_type" is identical regardless of where the
    text came from."""
    log.debug(f"[process:{category}] enter _process_article for {url}")
    if not article or article.get("error") or not article.get("text"):
        reason = article.get("error") if article else "no response"
        reason_str = str(reason)
        if "robots.txt" in reason_str or "blocked:" in reason_str:
            counters["robots_blocked"] += 1
        elif "video/media" in reason_str or "duration" in reason_str:
            counters["video_skipped"] += 1
        elif reason_str.startswith("HTTP "):
            counters["http_error"] += 1
        elif "not installed" in reason_str or "deps missing" in reason_str:
            counters["missing_dependency"] += 1
        else:
            counters["other_extract_fail"] += 1
        log.info(f"[extract:{category}] SKIP {url} -- {reason}")
        return False

    text = article["text"]
    content_type = article.get("content_type")
    log.debug(f"[extract:{category}] OK {url} -- {len(text)} chars, type={content_type}")

    # If the source already carries its own prompt/answer labels (a HF
    # instruction dataset, a Kaggle Q&A CSV, ...), trust those over an LLM
    # guess -- they're the dataset author's ground truth, not a
    # hallucinated question. Resolved up front (not just inside the
    # mode=="sft" record-build block below) because it also changes which
    # quality bar applies: a labeled pair is judged on its own terms via
    # passes_sft_pair_quality_filter, NOT passes_prose_quality_filter's
    # min_doc_chars floor (500 by default), which is tuned for scraped web
    # articles and would reject nearly every short-but-legitimate Q&A pair
    # (a math problem + a one-line answer, a one-line code fix) before it
    # ever reached the LLM judge.
    extra = article.get("extra") or {}
    given_prompt, given_answer = extra.get("prompt"), extra.get("answer")
    has_given_pair = bool(mode == "sft" and given_prompt and given_answer)

    if has_given_pair:
        ok, reason = passes_sft_pair_quality_filter(given_prompt, given_answer, MIN_SFT_PAIR_CHARS)
    else:
        ok, reason = _quality_filter_for(content_type, text, url, min_doc_chars)
    if not ok:
        counters["filtered_quality"] += 1
        counters["filtered_quality_reasons"][reason] = counters["filtered_quality_reasons"].get(reason, 0) + 1
        log.info(f"[filter:{category}] REJECT (quality heuristics: {reason}) {url}")
        return False
    log.debug(f"[filter:{category}] quality heuristics passed for {url}, "
              f"use_llm_judge={use_llm_judge}")

    judge_text = f"{given_prompt}\n\n{given_answer}" if has_given_pair else text
    if use_llm_judge:
        log.debug(f"[filter:{category}] entering judge_quality for {url}")
        keep = await judge_quality(judge_text, category, pair_mode=has_given_pair)
        log.debug(f"[filter:{category}] returned from judge_quality for {url}, keep={keep}")
        if not keep:
            counters["llm_rejected"] += 1
            log.info(f"[filter:{category}] REJECT (llm judge) {url}")
            return False

    if mode == "sft":
        if has_given_pair:
            pair = {"prompt": given_prompt, "thinking": "", "answer": given_answer}
        else:
            # Row genuinely doesn't carry a labeled pair -- fall back to
            # Ollama inventing one from raw prose. Under --fast-heuristics
            # (LLM_BYPASS), extract_sft_pair always returns None here --
            # there's no way to synthesize a question from raw prose
            # without an LLM, full stop. That's expected, but it means
            # --fast-heuristics + --mode sft can ONLY ever keep rows whose
            # prompt/answer columns were detected by the static heuristics
            # in public_sources.py (no LLM schema inference either, same
            # reason) -- any dataset with an unrecognized schema and no
            # labeled pair will reject 100% of its rows in this combination.
            log.debug(f"[filter:{category}] entering extract_sft_pair for {url}")
            pair = await extract_sft_pair(text, category)
            log.debug(f"[filter:{category}] returned from extract_sft_pair for {url}")
        if pair is None:
            counters["no_sft_pair"] += 1
            if LLM_BYPASS:
                log.info(f"[filter:{category}] REJECT (no sft pair extractable -- row has no "
                          f"detected prompt/answer columns, and --fast-heuristics disables the "
                          f"LLM fallback that would otherwise synthesize one) {url}")
            else:
                log.info(f"[filter:{category}] REJECT (no sft pair extractable) {url}")
            return False
        record = {**pair, "source": url, "category": category, "content_type": content_type}
    else:
        record = {"text": text, "source": url, "category": category, "content_type": content_type}

    log.debug(f"[write:{category}] waiting on write_lock for {url}")
    async with write_lock:
        log.debug(f"[write:{category}] acquired write_lock for {url}")
        if writer.total_bytes >= byte_budget:
            return False  # another concurrent task already hit budget
        if exact_dedup.is_duplicate(text) or near_dedup.is_near_duplicate(text):
            counters["filtered_dup"] += 1
            log.info(f"[filter:{category}] REJECT (duplicate) {url}")
            return False
        writer.write(record)
        log.info(f"[write:{category}] KEPT {url} ({len(text)} chars, type={content_type}) "
                 f"-- total {writer.total_bytes/1024**2:.2f} MB / "
                 f"{byte_budget/1024**2:.1f} MB, {writer.total_docs} docs")
        return True


async def _process_hit(scraper: ScraperClient, url: str, category: str, mode: str,
                        min_doc_chars: int, use_llm_judge: bool, exact_dedup: ExactDedup,
                        near_dedup, writer: ShardWriter, byte_budget: int,
                        write_lock: asyncio.Lock, counters: dict) -> bool:
    """Extract + filter + (maybe) write one URL. Returns True if a record
    was written. Runs concurrently across many URLs; `write_lock` serializes
    only the dedup-check-and-write step so budget/duplicate checks stay
    correct under concurrency without serializing the slow network/ASR work."""
    try:
        article = await scraper.extract(url)
    except Exception as e:
        log.warning(f"[extract:{category}] failed for {url}: {e}")
        counters["other_extract_fail"] += 1
        return False

    return await _process_article(article, url, category, mode, min_doc_chars, use_llm_judge,
                                   exact_dedup, near_dedup, writer, byte_budget, write_lock,
                                   counters)


async def _process_public_row(article: dict, category: str, mode: str, min_doc_chars: int,
                               use_llm_judge: bool, exact_dedup: ExactDedup, near_dedup,
                               writer: ShardWriter, byte_budget: int, write_lock: asyncio.Lock,
                               counters: dict) -> bool:
    """Same as _process_hit but for a row that's already been fetched from
    a public dataset hub (no network extract step needed -- public_sources
    already normalized it to the extract_content shape)."""
    url = article.get("url", "public-dataset-row")
    return await _process_article(article, url, category, mode, min_doc_chars, use_llm_judge,
                                   exact_dedup, near_dedup, writer, byte_budget, write_lock,
                                   counters)


async def _process_deep_crawl_page(page: dict, category: str, mode: str, min_doc_chars: int,
                                    use_llm_judge: bool, exact_dedup: ExactDedup, near_dedup,
                                    writer: ShardWriter, byte_budget: int, write_lock: asyncio.Lock,
                                    counters: dict) -> bool:
    """Same as _process_hit but for a page that deep_crawl already fetched
    and extracted server-side (no separate extract_content round-trip
    needed -- deep_crawl's pages are already in the extract_content
    shape)."""
    url = page.get("url", "deep-crawl-page")
    return await _process_article(page, url, category, mode, min_doc_chars, use_llm_judge,
                                   exact_dedup, near_dedup, writer, byte_budget, write_lock,
                                   counters)


async def run_public_sources_for_category(category: str, byte_budget: int, mode: str,
                                           min_doc_chars: int, use_llm_judge: bool,
                                           exact_dedup: ExactDedup, near_dedup,
                                           writer: ShardWriter, write_lock: asyncio.Lock,
                                           counters: dict, public_cfg: dict,
                                           concurrency: int = 5) -> None:
    """Drains Hugging Face / Kaggle datasets configured for this category
    before falling back to live web scraping. Public sources are cheaper
    and more reliable than scraping (no robots.txt, no rate limiting, no
    HTML boilerplate to strip), so this runs first and only cedes the
    remaining budget to the search+scrape loop.

    One dataset at a time, in order:

    1. **Download** -- pull all rows (up to `max_rows_per_dataset`) from
       one configured/discovered dataset. If `keep_raw_staging` is set,
       each raw row is also echoed to
       `<out-dir>/<category>/.public_raw_staging.jsonl` for audit/debug
       purposes, but nothing is quality-filtered, judged, deduped, or
       written to the real output shards at this point.
    2. **Filter + write** -- that dataset's rows are run through the same
       heuristic quality filters + (if enabled) the Ollama LLM judge as
       any other source, and passing rows are written to the category's
       shards.
    3. **Check quota** -- if `writer.total_bytes >= byte_budget`, stop
       entirely: no further dataset (Hugging Face or Kaggle) is even
       downloaded. Otherwise move on to the next configured/discovered
       dataset and repeat.

    This means a run only ever downloads as many datasets as it actually
    needs to hit budget, rather than always pulling every configured/
    discovered dataset up front. The tradeoff (vs. the old download-
    everything-then-filter design) is that you no longer see the full raw
    total available across every dataset before filtering starts -- but
    for a --public-only run that's the right tradeoff: no point in paying
    for (or waiting on) a multi-GB download of a dataset you'll never
    need because an earlier one already filled the budget.

    Because filtering only ever shrinks what a dataset contributes, if
    you exhaust every configured/discovered dataset and still haven't
    hit budget, raise `--public-max-rows` / `--public-discover-limit`, or
    pass more explicit `--hf-datasets`/`--kaggle-datasets`, for more raw
    material to draw from next time.

    `public_cfg` shape: {
        "sources": {"huggingface", "kaggle"},       # which backends are on
        "hf_datasets": {category: [dataset_id, ...]},   # explicit ids, optional
        "kaggle_datasets": {category: [ref, ...]},      # explicit refs, optional
        "blacklist_datasets": {category: [dataset_id_or_ref, ...]},  # excluded ids/refs, optional
        "max_rows_per_dataset": int,
        "discover_limit": int,                          # datasets to auto-discover per category
        "keep_raw_staging": bool,                       # keep an audit copy of every raw row pulled
    }
    If no explicit dataset ids/refs are given for a category, it falls back
    to auto-discovery: searching the hub with each of that category's
    HUB_SEARCH_KEYWORDS (topics.py) in turn and merging/deduping the hits,
    up to discover_limit total. These are short (1-3 word) hub-search
    terms, deliberately NOT the same as the long natural-language
    TOPIC_SEEDS sentences used to steer the web-search query planner --
    HF/Kaggle's dataset search does simple keyword matching against
    dataset names/tags, so a full sentence like "calculus integration by
    parts examples" matches nothing and silently discovers zero datasets.


    `blacklist_datasets` (same {category: [...]} / bare-list-applies-to-
    every-category shape as hf_datasets/kaggle_datasets, matched
    case-insensitively) excludes dataset ids/refs everywhere they could
    otherwise enter a run: filtered out of an explicit --hf-datasets/
    --kaggle-datasets list, AND skipped during auto-discovery so a
    blacklisted hit doesn't consume one of that category's
    discover_limit slots -- discovery keeps trying further keywords to
    backfill instead. Use this to permanently exclude a dataset you've
    found to be low-quality, mislabeled, or otherwise unwanted, without
    having to touch --hf-datasets/--kaggle-datasets for every category
    that happens to auto-discover it.
    """
    sem = asyncio.Semaphore(concurrency)
    _row_seq = itertools.count(1)

    async def bounded(article: dict) -> bool:
        row_id = next(_row_seq)
        url = article.get("url", "public-dataset-row")
        log.debug(f"[public:{category}] row #{row_id} ({url}) waiting on semaphore "
                  f"(concurrency={concurrency})")
        async with sem:
            log.debug(f"[public:{category}] row #{row_id} ({url}) acquired semaphore, processing")
            try:
                result = await _process_public_row(article, category, mode, min_doc_chars,
                                                    use_llm_judge, exact_dedup, near_dedup,
                                                    writer, byte_budget, write_lock, counters)
                log.debug(f"[public:{category}] row #{row_id} ({url}) done, kept={result}")
                return result
            except Exception as e:
                # asyncio.gather(..., return_exceptions=True) swallows this
                # into the results list without a trace -- log it here so a
                # per-row bug doesn't look identical to a silent hang.
                log.error(f"[public:{category}] row #{row_id} ({url}) RAISED: "
                          f"{type(e).__name__}: {e}", exc_info=True)
                raise

    max_rows = public_cfg.get("max_rows_per_dataset", 500)
    discover_limit = public_cfg.get("discover_limit", 3)
    keep_raw_staging = public_cfg.get("keep_raw_staging", False)
    search_keywords = HUB_SEARCH_KEYWORDS.get(category, [category])

    def _lookup(cat_map: dict) -> Optional[list]:
        return cat_map.get(category) or cat_map.get("__all__")

    blacklist = {i.strip().lower() for i in (_lookup(public_cfg.get("blacklist_datasets", {})) or [])}
    if blacklist:
        log.info(f"[public:{category}] blacklist active ({len(blacklist)} entries): {sorted(blacklist)}")

    def _drop_blacklisted(ids: list, label: str) -> list:
        kept = [i for i in ids if i.lower() not in blacklist]
        skipped = [i for i in ids if i.lower() in blacklist]
        if skipped:
            log.info(f"[{label}:{category}] blacklisted, skipping: {skipped}")
        return kept

    async def _discover(discover_fn, label: str) -> list:
        """Try each short keyword in turn, merging/deduping hits (in
        discovery order) until discover_limit distinct ids/refs are found
        or every keyword's been tried. Logs what each individual keyword
        turned up so a bad keyword is visible in the log instead of just
        silently contributing nothing. Blacklisted hits are dropped
        before counting toward discover_limit, so a blacklisted dataset
        doesn't waste a discovery slot -- more keywords get tried instead."""
        found: list = []
        for kw in search_keywords:
            if len(found) >= discover_limit:
                break
            hits = await asyncio.to_thread(discover_fn, kw, discover_limit - len(found))
            log.info(f"[{label}:{category}] keyword {kw!r} -> {hits}")
            hits = _drop_blacklisted(hits, label)
            for h in hits:
                if h not in found:
                    found.append(h)
        return found[:discover_limit]

    # -----------------------------------------------------------------
    # Per-dataset interleaved loop: download one dataset -> filter+judge+
    # write it -> check quota -> move to the next dataset, stopping as
    # soon as `byte_budget` (post-filter, written bytes) is met. This
    # trades the old two-phase design's "always know the full raw total
    # before filtering starts" property for the more common-sense
    # behavior of not downloading datasets you'll never need: once
    # writer.total_bytes hits budget, no further dataset is even
    # requested. `_process_article` (reached via `bounded()`) already
    # re-checks writer.total_bytes under write_lock before every write, so
    # quota is still enforced precisely even with concurrent in-flight
    # judge calls for the current dataset's rows.
    # -----------------------------------------------------------------
    staging_path = os.path.join(writer.dir, ".public_raw_staging.jsonl")
    raw_fh = open(staging_path, "w") if keep_raw_staging else None
    total_raw_rows = 0

    # Ollama prewarm happens once, up front, rather than "after phase 1" --
    # there's no separate download-everything phase anymore, so the first
    # dataset's filter pass is the first real judge call.
    if use_llm_judge:
        await _prewarm_ollama()

    def _drain_dataset_rows(row_iter) -> list:
        """Blocking helper: pull all rows out of a (streaming, network-
        backed) generator for ONE dataset into a list, optionally echoing
        each raw row to the audit staging file. Runs inside
        asyncio.to_thread since dataset/network iteration blocks. Bounded
        by max_rows already, via the generator itself (stream_hf_dataset /
        fetch_kaggle_dataset_rows are both called with max_rows=max_rows)."""
        rows = []
        for row in row_iter:
            rows.append(row)
            if raw_fh is not None:
                raw_fh.write(json.dumps(row) + "\n")
        if raw_fh is not None:
            raw_fh.flush()
        return rows

    async def _process_dataset(label: str, ref: str, rows: list) -> None:
        """Run the existing filter + (LLM judge) + dedup + write pipeline
        over one dataset's already-downloaded rows, same batching/flush
        policy as before (gather in JUDGE_BATCHER-sized chunks so Ollama
        batching still coalesces requests)."""
        print(f"[public-{label}:{category}] processing {len(rows)} rows from {ref}"
              + (" (LLM judge on)" if use_llm_judge else " (heuristics only, --no-llm-judge)"))
        tasks = []
        batch_flush = max(1, JUDGE_BATCHER.batch_size) * 2
        log.info(f"[public-{label}:{category}] batch_flush size = {batch_flush} "
                 f"(JUDGE_BATCHER.batch_size={JUDGE_BATCHER.batch_size})")
        rows_queued = 0
        for row in rows:
            if writer.total_bytes >= byte_budget:
                log.debug(f"[public-{label}:{category}] quota met mid-dataset, "
                          f"stopping after queuing {rows_queued}/{len(rows)} rows")
                break
            tasks.append(asyncio.create_task(bounded(row)))
            rows_queued += 1
            if len(tasks) >= batch_flush:
                log.info(f"[public-{label}:{category}] gathering batch of {len(tasks)} task(s) "
                         f"({rows_queued}/{len(rows)} rows queued so far)")
                t0 = time.time()
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for r in results:
                    if isinstance(r, Exception):
                        log.error(f"[public-{label}:{category}] task in batch raised: "
                                  f"{type(r).__name__}: {r}")
                log.info(f"[public-{label}:{category}] batch of {len(tasks)} gathered "
                         f"in {time.time()-t0:.1f}s")
                tasks = []
        if tasks:
            log.info(f"[public-{label}:{category}] gathering final batch of {len(tasks)} task(s)")
            t0 = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, Exception):
                    log.error(f"[public-{label}:{category}] task in final batch raised: "
                              f"{type(r).__name__}: {r}")
            log.info(f"[public-{label}:{category}] final batch of {len(tasks)} gathered "
                     f"in {time.time()-t0:.1f}s")
        print(f"[public-{label}:{category}] after {ref}: "
              f"{writer.total_bytes/1024**2:.2f}/{byte_budget/1024**2:.1f} MB written, "
              f"{writer.total_docs} docs")

    async def _run_source(label: str, ds_ids: list, stream_fn, discover_fn) -> bool:
        """Iterate one source's dataset list (already explicit or about to
        be discovered), download+process+check-quota one dataset at a
        time. Returns True if quota was met (caller can skip the next
        source entirely) else False."""
        nonlocal total_raw_rows
        if ds_ids:
            ds_ids = _drop_blacklisted(ds_ids, f"public-{label}")
        else:
            ds_ids = await _discover(discover_fn, f"public-{label}")
            log.info(f"[public-{label}:{category}] discovered datasets: {ds_ids}")
        for i, ref in enumerate(ds_ids):
            if writer.total_bytes >= byte_budget:
                log.info(f"[public-{label}:{category}] quota already met "
                          f"({writer.total_bytes/1024**2:.2f}/{byte_budget/1024**2:.1f} MB) -- "
                          f"skipping remaining datasets: {ds_ids[i:]}")
                return True
            log.info(f"[public-{label}:{category}] downloading {ref} (max {max_rows} rows)")
            gen = stream_fn(ref, max_rows=max_rows, column_mapper=_infer_column_mapping_sync)
            rows = await asyncio.to_thread(_drain_dataset_rows, gen)
            total_raw_rows += len(rows)
            print(f"[public-{label}:{category}] {ref} downloaded: {len(rows)} rows")
            await _process_dataset(label, ref, rows)
            if writer.total_bytes >= byte_budget:
                log.info(f"[public-{label}:{category}] quota met after {ref} "
                          f"({writer.total_bytes/1024**2:.2f}/{byte_budget/1024**2:.1f} MB)")
                return True
        return False

    sources = public_cfg.get("sources", set())
    quota_met = False
    if "huggingface" in sources:
        hf_ids = _lookup(public_cfg.get("hf_datasets", {}))
        quota_met = await _run_source("hf", hf_ids, public_sources.stream_hf_dataset,
                                       public_sources.discover_hf_datasets)

    if "kaggle" in sources and not quota_met:
        kg_refs = _lookup(public_cfg.get("kaggle_datasets", {}))
        quota_met = await _run_source("kaggle", kg_refs, public_sources.fetch_kaggle_dataset_rows,
                                       public_sources.discover_kaggle_datasets)

    if raw_fh is not None:
        raw_fh.close()
        log.info(f"[public:{category}] kept raw staging file at {staging_path} "
                  f"(--public-keep-raw-staging, {total_raw_rows} rows total)")

    if not quota_met:
        log.warning(f"[public:{category}] finished every discovered/configured dataset without "
                     f"hitting budget ({writer.total_bytes/1024**2:.2f}/{byte_budget/1024**2:.1f} MB). "
                     f"Raise --public-max-rows and/or --public-discover-limit, or pass explicit "
                     f"--hf-datasets/--kaggle-datasets, for more raw material next time.")

    print(f"[public:{category}] after public-source top-up: "
          f"{writer.total_bytes/1024**2:.2f} MB / {byte_budget/1024**2:.1f} MB, "
          f"{writer.total_docs} docs")


async def run_category(scraper: ScraperClient, category: str, byte_budget: int, out_dir: str,
                        mode: str, min_doc_chars: int, use_llm_judge: bool, concurrency: int = 5,
                        public_cfg: Optional[dict] = None, deep_crawl_per_domain: int = 0,
                        deep_crawl_max_pages: int = 10):
    writer = ShardWriter(out_dir, category)
    exact_dedup = ExactDedup(persist_path=os.path.join(out_dir, category, ".seen_hashes"))
    near_dedup = NearDedup()
    state = RunState(out_dir, category)  # resumable across process runs
    write_lock = asyncio.Lock()
    sem = asyncio.Semaphore(concurrency)

    counters = {
        "filtered_quality": 0, "filtered_dup": 0, "llm_rejected": 0,
        "robots_blocked": 0, "http_error": 0, "video_skipped": 0,
        "missing_dependency": 0, "other_extract_fail": 0, "no_sft_pair": 0,
        "filtered_quality_reasons": {},
    }

    resumed = f" (resumed: {len(state.used_queries)} prior queries, {len(state.seen_urls)} prior URLs)" \
        if state.used_queries or state.seen_urls else ""
    print(f"\n=== [{category}] target: {byte_budget / 1024**2:.1f} MB "
          f"(live web scraping, concurrency={concurrency}){resumed} ===")

    if public_cfg and public_cfg.get("sources"):
        await run_public_sources_for_category(
            category, byte_budget, mode, min_doc_chars, use_llm_judge,
            exact_dedup, near_dedup, writer, write_lock, counters, public_cfg,
            concurrency=concurrency,
        )
        if public_cfg.get("public_only"):
            state.save()
            print(f"\n[{category}] public-sources-only run done: "
                  f"{writer.total_bytes / 1024**2:.2f} MB, {writer.total_docs} docs, "
                  f"filtered {counters['filtered_quality']} low-quality + "
                  f"{counters['filtered_dup']} duplicate + {counters['llm_rejected']} llm-rejected + "
                  f"{counters['no_sft_pair']} no-labeled-pair"
                  f" -- quality reject breakdown: {counters['filtered_quality_reasons']}")
            writer.close()
            exact_dedup.close()
            return writer.total_bytes, writer.total_docs

    async def bounded_process(url: str):
        async with sem:
            return await _process_hit(scraper, url, category, mode, min_doc_chars,
                                       use_llm_judge, exact_dedup, near_dedup, writer,
                                       byte_budget, write_lock, counters)

    async def bounded_deep_crawl(seed_url: str):
        async with sem:
            try:
                pages = await scraper.deep_crawl(seed_url, max_pages=deep_crawl_max_pages,
                                                   max_depth=1, keywords=category)
            except Exception as e:
                log.warning(f"[deep_crawl:{category}] failed for {seed_url}: {e}")
                return 0
            if len(pages) == 1 and pages[0].get("error"):
                log.info(f"[deep_crawl:{category}] {seed_url} -- {pages[0]['error']}")
                return 0
            written = 0
            for page in pages:
                page_url = page.get("url", seed_url)
                if page_url in state.seen_urls:
                    continue
                state.seen_urls.add(page_url)
                if await _process_deep_crawl_page(page, category, mode, min_doc_chars,
                                                   use_llm_judge, exact_dedup, near_dedup,
                                                   writer, byte_budget, write_lock, counters):
                    written += 1
            log.info(f"[deep_crawl:{category}] {seed_url} -> {len(pages)} pages harvested, "
                     f"{written} kept")
            return written

    deep_crawled_domains: set = set()

    stall_rounds = 0
    while writer.total_bytes < byte_budget:
        queries = await plan_queries(category, state.used_queries, n=6)
        state.used_queries.extend(queries)
        progressed_this_round = False
        round_urls = []

        for query in queries:
            log.info(f"[search:{category}] query={query!r}")
            try:
                hits = await scraper.search(query, max_results=8)
            except Exception as e:
                log.warning(f"[search:{category}] search failed for {query!r}: {e}")
                continue

            if not isinstance(hits, list):
                log.warning(f"[search:{category}] {query!r} returned unexpected "
                            f"non-list response, treating as failure: {hits!r}")
                continue
            if len(hits) == 1 and isinstance(hits[0], dict) and "error" in hits[0]:
                log.warning(f"[search:{category}] backend error for {query!r}: {hits[0]['error']}")
                continue

            log.info(f"[search:{category}] {query!r} -> {len(hits)} hits: "
                      f"{[h.get('url') for h in hits if isinstance(h, dict)]}")

            for hit in hits:
                url = hit.get("url") if isinstance(hit, dict) else None
                if not url or url in state.seen_urls:
                    continue
                state.seen_urls.add(url)
                round_urls.append(url)

        # Extract this round's URLs concurrently (bounded by --concurrency)
        # instead of one at a time -- this is where wall-clock time actually
        # goes (network fetch + optional ASR/OCR), so serializing it was the
        # single biggest throughput bottleneck in the original loop.
        if round_urls:
            results = await asyncio.gather(*(bounded_process(u) for u in round_urls),
                                             return_exceptions=True)
            for r in results:
                if r is True:
                    progressed_this_round = True
                elif isinstance(r, Exception):
                    log.warning(f"[{category}] worker task raised: {r}")

        # Deep-crawl top-up: this round's search hits already told us which
        # domains are relevant to this category -- rather than only taking
        # the one page web_search pointed at, spend a few crawl4ai calls
        # harvesting more pages from the SAME domains (its docs/blog/wiki
        # neighbors), which is usually much cheaper per-document than
        # planning + running more search queries. Each domain is only
        # deep-crawled once per category run (deep_crawled_domains), and
        # only up to --deep-crawl-per-domain new domains get this treatment
        # per round, to keep it a top-up rather than the primary path.
        if deep_crawl_per_domain > 0 and round_urls:
            candidate_domains = []
            for u in round_urls:
                host = urllib.parse.urlparse(u).netloc.lower()
                if host and host not in deep_crawled_domains:
                    deep_crawled_domains.add(host)
                    candidate_domains.append(u)  # crawl from the hit itself as the seed
                if len(candidate_domains) >= deep_crawl_per_domain:
                    break
            if candidate_domains:
                dc_results = await asyncio.gather(
                    *(bounded_deep_crawl(u) for u in candidate_domains),
                    return_exceptions=True,
                )
                for r in dc_results:
                    if isinstance(r, int) and r > 0:
                        progressed_this_round = True
                    elif isinstance(r, Exception):
                        log.warning(f"[deep_crawl:{category}] worker task raised: {r}")

        state.save()

        if not progressed_this_round:
            stall_rounds += 1
            log.warning(f"[{category}] no docs written this round "
                        f"(stall_rounds={stall_rounds}/5)")
            if stall_rounds >= 5:
                log.warning(f"[{category}] no progress after 5 query rounds -- stopping early "
                            f"at {writer.total_bytes / 1024**2:.1f} MB")
                break
        else:
            stall_rounds = 0

    print(f"\n[{category}] done: {writer.total_bytes / 1024**2:.2f} MB, "
          f"{writer.total_docs} docs, {writer.shard_idx + 1} shard(s), "
          f"filtered {counters['filtered_quality']} low-quality + {counters['filtered_dup']} duplicate + "
          f"{counters['llm_rejected']} llm-rejected + {counters['no_sft_pair']} no-labeled-pair\n"
          f"[{category}] quality reject breakdown: {counters['filtered_quality_reasons']}\n"
          f"[{category}] fetch failures: {counters['robots_blocked']} robots.txt/domain-blocked, "
          f"{counters['http_error']} HTTP error (403/etc.), {counters['video_skipped']} video/duration-skipped, "
          f"{counters['missing_dependency']} missing optional dependency, "
          f"{counters['other_extract_fail']} other extraction failures")
    writer.close()
    exact_dedup.close()
    return writer.total_bytes, writer.total_docs


def _find_server_path() -> str:
    """Locate server.py. Tries the README's suggested `web_scraper_mcp/`
    sibling-directory layout first, then falls back to flatter layouts
    (server.py next to dataset_agent.py, or one level up), since a mismatch
    here silently breaks the MCP subprocess launch with zero error output --
    `session.initialize()` just hangs waiting for a handshake that will
    never come from a process that never started."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "..", "web_scraper_mcp", "server.py"),
        os.path.join(here, "server.py"),
        os.path.join(here, "..", "server.py"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c)
    tried = "\n  ".join(candidates)
    raise FileNotFoundError(
        f"Could not find server.py. Tried:\n  {tried}\n"
        f"Set the MCP_SERVER_PATH env var to the exact path if your layout "
        f"differs from all of these."
    )


async def main_async(args):
    # Apply the CLI-controlled Ollama batcher settings before any
    # judge_quality / extract_sft_pair call can happen. Default is
    # batch_size=8 (a real speedup on local Ollama); --no-judge-batching
    # forces the legacy per-doc path. The batchers live for the duration
    # of the process and are torn down in main() after main_async returns
    # so any in-flight requests get a clean shutdown drain.
    batch_size = 1 if getattr(args, "no_judge_batching", False) else max(1, args.judge_batch_size)
    flush_interval = max(0.05, args.judge_batch_flush_seconds)
    configure_batcher_settings(batch_size, flush_interval)
    log.info(f"Ollama batcher: batch_size={batch_size}, "
             f"flush_interval={flush_interval}s "
             f"({'disabled' if batch_size == 1 else 'enabled'})")

    # --fast-heuristics bypasses ALL Ollama calls (judge + SFT pair extract).
    # --no-llm-judge only skips the judge; SFT pair extraction still runs.
    # When bypass is on, we ALSO pass use_llm_judge=False into run_category,
    # so the gate around judge_quality inside _process_article is skipped
    # entirely (no point calling a function that returns True unconditionally).
    global LLM_BYPASS
    if getattr(args, "fast_heuristics", False):
        LLM_BYPASS = True
        log.info("--fast-heuristics: bypassing ALL Ollama calls (judge + SFT pair extract + "
                  "schema/column-mapping inference)")
        if args.mode == "sft":
            log.warning(
                "--fast-heuristics + --mode sft: rows can ONLY be kept if their prompt/answer "
                "columns are recognized by the static heuristics in public_sources.py "
                "(_PROMPT_COLUMNS/_ANSWER_COLUMNS/conversation-format detection) -- there is no "
                "LLM available in this mode to synthesize a Q/A pair from unlabeled prose, or to "
                "infer a column mapping for a dataset with an unfamiliar schema. If a dataset's "
                "reject log is dominated by 'no sft pair extractable', that dataset's columns "
                "aren't matching the static list; either extend it or drop --fast-heuristics for "
                "that run so the LLM-based schema inference and pair extraction can run.")
    else:
        LLM_BYPASS = False
    use_llm_judge_flag = (not args.no_llm_judge) and (not LLM_BYPASS)

    global MIN_SFT_PAIR_CHARS
    MIN_SFT_PAIR_CHARS = args.min_sft_pair_chars
    log.info(f"min_sft_pair_chars={MIN_SFT_PAIR_CHARS} (labeled-pair rows only; "
             f"--min-doc-chars={args.min_doc_chars} still applies to unlabeled/prose rows)")

    target_bytes = _parse_size(args.target_size)
    categories = args.categories.split(",") if args.categories else list(DEFAULT_MIX.keys())
    mix = {c: DEFAULT_MIX.get(c, 1.0 / len(categories)) for c in categories}
    total_frac = sum(mix.values())
    mix = {c: f / total_frac for c, f in mix.items()}

    os.makedirs(args.out_dir, exist_ok=True)

    manifest_path = os.path.join(args.out_dir, "manifest.json")
    manifest = {"target_bytes": target_bytes, "mix": mix, "categories": {}, "mode": args.mode}

    public_cfg = None
    public_source_set = {s.strip().lower() for s in (args.public_sources or "").split(",") if s.strip()}
    if public_source_set:
        public_cfg = {
            "sources": public_source_set,
            "hf_datasets": _parse_category_map(args.hf_datasets),
            "kaggle_datasets": _parse_category_map(args.kaggle_datasets),
            "blacklist_datasets": _parse_category_map(args.blacklist_datasets),
            "max_rows_per_dataset": args.public_max_rows,
            "discover_limit": args.public_discover_limit,
            "public_only": args.public_only,
            "keep_raw_staging": args.public_keep_raw_staging,
        }
        log.info(f"Public dataset sources enabled: {public_source_set} "
                 f"(public_only={args.public_only})")

    # Pre-warm Ollama once at startup, before any per-category work. Without
    # this, the first row that passes the heuristic on each category would
    # trigger the full model-load inside the filter pass -- a 30s-5min wait
    # that silently stalls the entire pipeline (the OllamaBatcher coalesces
    # 10+ in-flight requests, all of which queue behind the single cold
    # load). Failing fast here also saves a multi-GB HuggingFace download
    # when the user has OLLAMA_MODEL pointed at a non-existent model.
    if use_llm_judge_flag and not getattr(args, "fast_heuristics", False):
        prewarm_ok = await _prewarm_ollama()
        if not prewarm_ok:
            log.error(
                "LLM judge prewarm failed. Aborting before wasting a multi-GB "
                "HuggingFace download on a pipeline whose quality gate can't "
                "run. Fix the Ollama model (or rerun with --no-llm-judge / "
                "--fast-heuristics) and try again."
            )
            return manifest

    # Categories used to run strictly one after another (`for category: await
    # run_category(...)`), so total wall-clock time was the SUM across
    # categories rather than roughly the time of the slowest one. Nothing
    # about run_category's internals actually requires that: each call gets
    # its own ShardWriter/ExactDedup/RunState/write_lock rooted at
    # <out-dir>/<category>/, so there's no shared mutable state between
    # categories to race on. The MCP session (shared across categories in
    # the non-public-only branch) already supports concurrent calls -- a
    # single category's own --concurrency workers already prove that, since
    # they fire concurrent tool calls down the same session. The Ollama
    # judge batchers are module-level singletons *designed* to coalesce
    # concurrent callers, so more concurrent categories means better
    # batching, not worse. --category-concurrency (0 = unbounded, i.e. all
    # categories at once) caps how many run in parallel if that's ever
    # worth limiting independently of --concurrency.
    category_sem = asyncio.Semaphore(args.category_concurrency) if args.category_concurrency > 0 \
        else None

    async def _bounded(coro):
        if category_sem is None:
            return await coro
        async with category_sem:
            return await coro

    # Computed once and reused for both branches below -- avoids re-deriving
    # "which categories have nonzero budget" twice (once to build the task
    # list, once to line results back up with category names) and risking
    # the two derivations drifting apart.
    runnable = [(category, int(target_bytes * frac))
                for category, frac in mix.items() if int(target_bytes * frac) > 0]

    if public_cfg and public_cfg.get("public_only"):
        # No live scraping requested at all -- skip spinning up the MCP
        # subprocess entirely, since ScraperClient/scraper is never touched
        # on the public-sources-only return path in run_category.
        log.info("public_only=True: skipping MCP scraper subprocess launch.")

        async def _run_one(category: str, budget: int):
            return await run_category(
                None, category, budget, args.out_dir, args.mode,
                args.min_doc_chars, use_llm_judge=use_llm_judge_flag,
                concurrency=args.concurrency, public_cfg=public_cfg,
                deep_crawl_per_domain=0,  # no MCP session in the public-only path -- N/A
            )

        results = await asyncio.gather(
            *(_bounded(_run_one(category, budget)) for category, budget in runnable),
            return_exceptions=True,
        )
        for (category, budget), r in zip(runnable, results):
            if isinstance(r, Exception):
                log.error(f"[{category}] run failed: {r}")
                continue
            actual_bytes, docs = r
            manifest["categories"][category] = {
                "target_bytes": budget, "actual_bytes": actual_bytes, "docs": docs,
            }
    else:
        server_path = os.environ.get("MCP_SERVER_PATH") or _find_server_path()
        log.info(f"Launching MCP server: {sys.executable} {server_path}")
        server_params = StdioServerParameters(command=sys.executable, args=[server_path])
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                try:
                    await asyncio.wait_for(session.initialize(), timeout=30)
                except asyncio.TimeoutError:
                    log.error(
                        "Timed out after 30s waiting for the MCP server to respond to "
                        "initialize(). The subprocess likely crashed on startup (missing "
                        "dependency, import error) or is hanging before it even prints "
                        "anything. Run it directly to see the real error:\n"
                        f"    {sys.executable} {server_path}"
                    )
                    raise
                log.info("MCP server initialized OK.")
                scraper = ScraperClient(session)

                async def _run_one(category: str, budget: int):
                    return await run_category(
                        scraper, category, budget, args.out_dir, args.mode,
                        args.min_doc_chars, use_llm_judge=use_llm_judge_flag,
                        concurrency=args.concurrency, public_cfg=public_cfg,
                        deep_crawl_per_domain=args.deep_crawl_per_domain,
                        deep_crawl_max_pages=args.deep_crawl_max_pages,
                    )

                results = await asyncio.gather(
                    *(_bounded(_run_one(category, budget)) for category, budget in runnable),
                    return_exceptions=True,
                )
                for (category, budget), r in zip(runnable, results):
                    if isinstance(r, Exception):
                        log.error(f"[{category}] run failed: {r}")
                        continue
                    actual_bytes, docs = r
                    manifest["categories"][category] = {
                        "target_bytes": budget, "actual_bytes": actual_bytes, "docs": docs,
                    }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    total_actual = sum(c["actual_bytes"] for c in manifest["categories"].values())
    print(f"\n=== Done. Total: {total_actual / 1024**2:.2f} MB across "
          f"{len(manifest['categories'])} categories ===")
    print(f"Manifest written to {manifest_path}")


def _parse_category_map(spec: Optional[str]) -> dict:
    """Parses `category=id1,id2;category2=id3,id4` into
    {"category": ["id1", "id2"], "category2": ["id3", "id4"]}. A bare
    comma list with no `category=` prefix (e.g. just `id1,id2`) is applied
    to every category -- convenient when you want the same dataset(s)
    pulled regardless of which category bucket they land in."""
    if not spec:
        return {}
    result: dict = {}
    segments = [s.strip() for s in spec.split(";") if s.strip()]
    bare_ids = []
    for seg in segments:
        if "=" in seg:
            cat, ids = seg.split("=", 1)
            result[cat.strip()] = [i.strip() for i in ids.split(",") if i.strip()]
        else:
            bare_ids.extend(i.strip() for i in seg.split(",") if i.strip())
    if bare_ids:
        result["__all__"] = bare_ids
    return result


def _parse_size(size_str: str) -> int:
    size_str = size_str.strip().upper()
    units = {"GB": 1024**3, "MB": 1024**2, "KB": 1024, "B": 1}
    for unit, mult in units.items():
        if size_str.endswith(unit):
            return int(float(size_str[: -len(unit)]) * mult)
    return int(float(size_str))


def main():
    parser = argparse.ArgumentParser(description="Live-scraping infinite dataset agent (Ollama + MCP).")
    parser.add_argument("--target-size", required=True, help="e.g. 500MB, 2GB")
    parser.add_argument("--out-dir", default="./data")
    parser.add_argument("--categories", default=None,
                         help="Comma-separated, e.g. web,knowledge,reasoning,code,math")
    parser.add_argument("--mode", choices=["pretrain", "sft"], default="pretrain")
    parser.add_argument("--min-doc-chars", type=int, default=500)
    parser.add_argument("--min-sft-pair-chars", type=int, default=20,
                         help="Length floor (combined prompt+answer chars) for rows that "
                              "already carry a labeled SFT pair (mode=sft only). Separate "
                              "from --min-doc-chars, which is tuned for scraped prose and "
                              "would reject nearly all short Q&A pairs (default: 20).")
    parser.add_argument("--no-llm-judge", action="store_true",
                         help="Skip the Ollama quality-judging pass, keep only heuristic filters (faster)")
    parser.add_argument("--fast-heuristics", action="store_true",
                         help="Bypass ALL Ollama calls -- skip both the quality judge AND the SFT "
                              "pair extraction. Documents are kept/rejected purely on the "
                              "heuristic filters (length, language, content-type patterns). "
                              "For --mode sft, rows without a pre-supplied prompt/answer are "
                              "rejected (the heuristic filters can't invent a Q/A pair). "
                              "Mutually exclusive with --no-llm-judge (which only skips the "
                              "judge and still runs SFT pair extraction).")
    parser.add_argument("--concurrency", type=int, default=5,
                         help="Max concurrent extract+filter tasks per category round (default 5). "
                              "Raise for I/O-bound HTML/PDF-heavy runs; keep low if ASR transcription "
                              "is in play (each faster-whisper call is CPU/GPU-heavy) or the LLM judge "
                              "is on (concurrent calls just queue behind a single local Ollama model).")
    parser.add_argument("--category-concurrency", type=int, default=0,
                         help="Max number of categories to run concurrently (default 0 = all of "
                              "them at once, bounded only by --concurrency within each). Categories "
                              "used to run one at a time, so total wall-clock time was the SUM "
                              "across categories instead of roughly the slowest one -- the single "
                              "biggest end-to-end throughput fix at multi-category, multi-hundred-GB "
                              "scale. All categories share the same MCP server subprocess (now "
                              "backed by a process pool for CPU-bound extraction, see "
                              "SCRAPER_EXTRACT_WORKERS/SCRAPER_MEDIA_WORKERS) and the same Ollama "
                              "judge batcher, both of which are designed for exactly this kind of "
                              "concurrent, cross-category load. Lower this only if you have very "
                              "many categories and want to cap total in-flight work more tightly "
                              "than --concurrency alone does.")
    parser.add_argument("--judge-batch-size", type=int, default=8,
                         help="Number of documents to coalesce into a single Ollama judge / SFT-"
                              "extract call (default 8). Local Ollama processes one request at a "
                              "time, so batching N concurrent requests into a single prompt is "
                              "the main throughput lever on the LLM-gated phases. Set to 1 to "
                              "disable batching (passthrough -- one Ollama call per document, "
                              "original behavior).")
    parser.add_argument("--judge-batch-flush-seconds", type=float, default=1.0,
                         help="Max time the judge batcher waits for a batch to fill before "
                              "flushing whatever it has (default 1.0s). Lower = lower latency, "
                              "less amortization. 0 disables the timer flush; only batch_size "
                              "fill will flush.")
    parser.add_argument("--no-judge-batching", action="store_true",
                         help="Disable Ollama batching entirely (equivalent to "
                              "--judge-batch-size 1). Useful for debugging or when the model is "
                              "too small to handle the combined context.")
    parser.add_argument("--deep-crawl-per-domain", type=int, default=0,
                         help="Per query round, deep-crawl up to N new domains seen among that "
                              "round's search hits (via crawl4ai's deep_crawl MCP tool), harvesting "
                              "several same-domain pages per call instead of just the one page "
                              "web_search pointed at. 0 (default) disables this top-up entirely. "
                              "Each domain is only deep-crawled once per category run. Requires "
                              "crawl4ai to be installed server-side (see requirements.txt); silently "
                              "yields 0 extra docs per domain if it isn't, same as any other missing "
                              "optional dependency.")
    parser.add_argument("--deep-crawl-max-pages", type=int, default=10,
                         help="Max pages to extract per deep_crawl call (default 10). Only relevant "
                              "when --deep-crawl-per-domain > 0.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                         help="DEBUG shows every Ollama call, skipped/seen URLs, and full text lengths; "
                              "INFO (default) shows every query, hit list, extract skip reason, filter "
                              "rejection reason, and write, without the very-verbose stuff.")

    public = parser.add_argument_group(
        "public dataset sources",
        "Pull rows from Hugging Face / Kaggle as a faster, more reliable top-up before (or "
        "instead of) live web scraping. For each category: first ALL configured/discovered "
        "datasets are downloaded (raw, unfiltered) to a staging file until that category's "
        "byte budget worth of raw text is pulled or datasets/rows run out; only THEN does the "
        "filter pass run -- the exact same heuristic quality filters, LLM judge, and (in "
        "--mode sft) the same built-in-AI Q/A extraction as scraped pages -- over everything "
        "staged, writing passing rows to the category's shards. Final written size is usually "
        "smaller than the byte budget, since filtering discards some of what was downloaded.")
    public.add_argument("--public-sources", default=None,
                         help="Comma list of public hub backends to pull from: huggingface,kaggle. "
                              "Unset (default) disables this feature entirely, matching prior behavior.")
    public.add_argument("--hf-datasets", default=None,
                         help="Explicit Hugging Face dataset ids to stream. Syntax: "
                              "'category=id1,id2;category2=id3', or a bare 'id1,id2' to apply to "
                              "every category. If omitted, ids are auto-discovered per category via "
                              "the Hub search API using short curated keywords (see topics.py's "
                              "HUB_SEARCH_KEYWORDS), not the long natural-language web-search topics.")
    public.add_argument("--kaggle-datasets", default=None,
                         help="Explicit Kaggle dataset refs (owner/dataset-slug), same syntax as "
                              "--hf-datasets. Requires KAGGLE_USERNAME/KAGGLE_KEY env vars (or "
                              "~/.kaggle/kaggle.json). If omitted, refs are auto-discovered per "
                              "category via Kaggle's search API.")
    public.add_argument("--blacklist-datasets", default=None,
                         help="Dataset ids/refs to exclude, same 'category=id1,id2;category2=id3' "
                              "syntax as --hf-datasets (or a bare 'id1,id2' to blacklist everywhere). "
                              "Matched case-insensitively against both HF dataset ids and Kaggle "
                              "owner/dataset-slug refs. Applies to explicit --hf-datasets/"
                              "--kaggle-datasets entries AND auto-discovered ones -- a blacklisted "
                              "hit is dropped before it counts toward --public-discover-limit, so "
                              "discovery tries further keywords to backfill instead of just running "
                              "one slot short. Use this to permanently exclude a dataset you've found "
                              "to be low-quality, mislabeled, gated/inaccessible, or otherwise "
                              "unwanted, without editing --hf-datasets/--kaggle-datasets by hand for "
                              "every category that happens to auto-discover it.")
    public.add_argument("--public-max-rows", type=int, default=500,
                         help="Max rows to pull per dataset during the raw-download phase (default "
                              "500) -- a per-dataset safety ceiling, separate from the overall "
                              "per-category byte budget the download phase stops at. For large "
                              "target sizes (multi-GB+ per category), raise this substantially, since "
                              "the default 500 rows/dataset x a handful of discovered datasets won't "
                              "get remotely close to a multi-GB raw pull on its own.")
    public.add_argument("--public-discover-limit", type=int, default=3,
                         help="When dataset ids/refs aren't given explicitly, how many datasets to "
                              "auto-discover per category via hub search (default 3).")
    public.add_argument("--public-keep-raw-staging", action="store_true",
                         help="Keep each category's raw (pre-filter) staged JSONL "
                              "(<out-dir>/<category>/.public_raw_staging.jsonl) after the filter pass "
                              "instead of deleting it -- useful for inspecting exactly what was "
                              "downloaded before filtering, or for re-running the filter pass "
                              "yourself without re-downloading.")
    public.add_argument("--public-only", action="store_true",
                         help="Skip live web search/scraping entirely -- fill each category's "
                              "budget purely from the configured public dataset hubs, and don't "
                              "even launch the MCP scraper subprocess. Useful when you just want a "
                              "fast Kaggle/HF-only top-up run.")

    args = parser.parse_args()
    logging.getLogger("dataset_agent").setLevel(args.log_level)
    try:
        asyncio.run(main_async(args))
    finally:
        # Always drain the Ollama batchers, even on exception, so a
        # half-flushed batch doesn't strand pending futures. Uses a fresh
        # event loop if asyncio.run tore the main one down already.
        try:
            asyncio.run(_shutdown_all_batchers(drain_timeout=5.0))
        except RuntimeError:
            # No running loop available -- best effort: nothing to do.
            pass
        except Exception as e:
            log.warning(f"[ollama-batcher] shutdown error during exit: {e}")


if __name__ == "__main__":
    main()
