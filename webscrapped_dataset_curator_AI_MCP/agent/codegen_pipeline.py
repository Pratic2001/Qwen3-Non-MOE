#!/usr/bin/env python3
"""
codegen_pipeline.py

A deliberately simpler alternative to dataset_agent.py's per-row async
pipeline (search -> extract -> LLM-judge -> dedup -> write, one row at a
time, for every single document). That approach is thorough but slow and
complex: an Ollama call per document, an async governor, a dedup layer
that has to stay bounded across the whole run, etc.

This module does the same job build_dataset.py / download_sft_data.py /
download_grpo_data.py already do by hand -- map a dataset's specific
columns onto the target schema -- but has an LLM write that mapping once
per dataset instead of a human hand-coding an entry in a SOURCES dict.

Architecture (the "AI developer" loop): plan -> generate -> TEST -> run.

The single most important design choice here is HOW MUCH the LLM is
trusted with. The previous version of this pipeline asked Ollama to write
an entire standalone script per dataset: argparse, the full streaming
loop, quality filtering, dedup, shard writing, progress printing, error
handling -- everything. That is a lot of surface area for a small local
model to get right, and the bugs that actually happened in this codebase's
history were exactly the structural kind a whole-script generation
invites (a naive regex silently corrupting extracted answers, a mismatched
prompt template producing off-distribution rollouts, a missing exhaustion
check silently duplicating a dataset, a double-shift in a training loop).
None of those are "the model picked the wrong column" -- they're "the
model got the surrounding plumbing subtly wrong," and a whole-script
diff is a bad place to catch that.

So the LLM's job is now shrunk to the smallest possible unit: ONE pure
function, `map_row(row: dict) -> Optional[dict]`, that maps one source row
to the target schema (or returns None to skip it). Everything else --
streaming, quality filtering, exact-dedup, shard writing, progress
reporting, and the final manifest line -- is `agent/harness.py`: hand-
written once, reviewed once, unit-tested once (agent/tests/test_harness.py),
and reused identically for every dataset and every category, forever.

Four phases, used the same way for both public datasets and live web
crawling:

    PUBLIC DATASETS (--public-only)
        1. discover  -- reuse public_sources.discover_hf_datasets() to find
           candidate datasets for a category (same as dataset_agent.py).
        2. sample    -- stream the first N rows, note the columns.
        3. codegen   -- ask Ollama for map_row(row) ONLY: no loop, no I/O,
           no argparse, no filtering logic (the harness does that).
        4. TEST      -- py_compile it, then run an auto-generated pytest
           suite (agent/test_gen.py) against map_row() using the SAME real
           sample rows the model was shown -- catches "wrong column name",
           "always returns None", and "crashes on a null field" in well
           under a second, before any full-scale download starts. Failures
           (compile OR test) are fed back to Ollama as a targeted repair
           prompt, up to a bounded number of attempts.
        5. run       -- agent/harness.py imports the now-tested map_row()
           and does the real streaming/filtering/dedup/writing, with its
           own runtime safety net (aborts fast on a systemic error rate
           instead of silently limping through a broken run).

    LIVE WEB CRAWL (default, no --public-only)
        1. crawl-raw -- gather a batch of raw {url, text} pages via the
           MCP scraper (search + extract, no LLM judge, no per-row work)
           into one JSONL file, up to a byte budget with headroom for
           later filtering losses.
        2-5. codegen -> TEST -> run, same as above, with map_row(row) taking
           a {"url", "text"} raw-crawl row instead of an HF dataset row.

Usage:
    python codegen_pipeline.py --target-size 500MB --public-only \\
        --categories web,knowledge,math --out-dir ./data --mode pretrain

    python codegen_pipeline.py --target-size 200MB \\
        --categories web,knowledge --out-dir ./data --mode pretrain

    # Tune how aggressively --public-only discovers candidate datasets:
    python codegen_pipeline.py --target-size 5GB --public-only \\
        --categories web,math --discover-limit 20 \\
        --max-candidates-to-try 8 --max-total-considered 100
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import itertools
import json
import os
import re
import subprocess
import sys
import time
from typing import Optional

import httpx

sys.path.insert(0, os.path.dirname(__file__))
from public_sources import discover_hf_datasets, discover_hf_configs, schema_is_suitable
from topics import HUB_SEARCH_KEYWORDS, TOPIC_SEEDS
import test_gen

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")
AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
HARNESS_PY_PATH = os.path.join(AGENT_DIR, "harness.py")

# ---------------------------------------------------------------------------
# Unlike the previous version of this pipeline, generated map_row() modules
# import NOTHING local -- they're pure functions with no quality-filtering
# or dedup responsibility, so there's no "stage quality.py next to the
# generated script" problem anymore: agent/harness.py (which does own that
# logic) always runs from its real location in this repo and finds
# quality.py/schema_check.py next to itself, regardless of where --out-dir
# points. One less place for a path bug to hide.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Interpreter / environment -- every subprocess this module launches (the
# generated extractor scripts, py_compile, the MCP server) is spawned with
# sys.executable, i.e. THIS process's own interpreter, so they all share
# whatever venv/conda env is currently active and see the same installed
# requirements. env=os.environ.copy() is passed explicitly too, so PATH,
# VIRTUAL_ENV, HF_TOKEN, OLLAMA_URL, etc. all carry through rather than
# relying on subprocess's default inherit-parent-env behavior.
# ---------------------------------------------------------------------------

SUBPROCESS_ENV = os.environ.copy()


def _env_banner() -> None:
    """Prints which interpreter/env is about to be used, and does a quick,
    non-fatal check for the packages the pipeline (and the scripts it will
    generate) rely on, so a missing-dependency problem shows up immediately
    instead of surfacing later as a cryptic runtime failure in a generated
    script's own [FATAL ERROR CAUGHT] line."""
    venv = os.environ.get("VIRTUAL_ENV") or os.environ.get("CONDA_DEFAULT_ENV")
    log_header("Environment")
    log_info(f"interpreter : {sys.executable}")
    log_info(f"python      : {sys.version.split()[0]}")
    log_info(f"active env  : {venv or '(none detected -- system/base interpreter)'}")
    for pkg in ("datasets", "httpx", "mcp", "pytest"):
        found = importlib.util.find_spec(pkg) is not None
        (log_success if found else log_warn)(
            f"{pkg:<10} {'available' if found else 'NOT FOUND -- pip install ' + pkg} "
            f"(in {os.path.basename(sys.executable)}'s env)"
        )


# ---------------------------------------------------------------------------
# Colorized console logging -- plain ANSI, no external dependency. Auto-
# disabled when stdout isn't a real terminal or NO_COLOR is set, so piping
# to a file or CI log never ends up full of escape codes. Subprocess output
# streamed from generated scripts is colorized for the console the same way
# but always written PLAIN to the per-dataset log file (see
# run_generated_script) so log files stay grep-friendly.
# ---------------------------------------------------------------------------

class _Ansi:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    YELLOW = "\033[33m"
    GREEN = "\033[32m"
    RED = "\033[31m"
    GRAY = "\033[90m"
    B_RED = "\033[91m"
    B_GREEN = "\033[92m"
    B_YELLOW = "\033[93m"
    B_BLUE = "\033[94m"
    B_MAGENTA = "\033[95m"
    B_CYAN = "\033[96m"


_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR") and os.environ.get("TERM") != "dumb"


def _paint(text: str, *codes: str) -> str:
    if not _COLOR or not text:
        return text
    return "".join(codes) + text + _Ansi.RESET


def log_header(text: str) -> None:
    bar = "─" * max(20, min(78, len(text) + 4))
    print(_paint(f"\n{bar}", _Ansi.B_CYAN, _Ansi.BOLD))
    print(_paint(f"  {text}", _Ansi.B_CYAN, _Ansi.BOLD))
    print(_paint(bar, _Ansi.B_CYAN, _Ansi.BOLD))


def log_section(category: str, text: str) -> None:
    print(_paint(f"[{category}] ", _Ansi.B_MAGENTA, _Ansi.BOLD) + text)


def log_info(text: str) -> None:
    print(_paint("  ℹ ", _Ansi.B_BLUE) + text)


def log_step(text: str) -> None:
    print(_paint("  ▸ ", _Ansi.B_CYAN) + text)


def log_success(text: str) -> None:
    print(_paint("  ✔ ", _Ansi.B_GREEN, _Ansi.BOLD) + _paint(text, _Ansi.GREEN))


def log_warn(text: str) -> None:
    print(_paint("  ⚠ ", _Ansi.B_YELLOW, _Ansi.BOLD) + _paint(text, _Ansi.YELLOW))


def log_error(text: str) -> None:
    print(_paint("  ✘ ", _Ansi.B_RED, _Ansi.BOLD) + _paint(text, _Ansi.RED))


def log_dim(text: str) -> None:
    print(_paint(text, _Ansi.GRAY))


def _colorize_passthrough(line: str) -> str:
    """Colorizes one line of a generated script's stdout for console
    display only -- the caller writes the original, uncolored `line` to
    the log file separately."""
    if not _COLOR:
        return line
    stripped = line.rstrip("\n")
    if stripped.startswith("[FATAL ERROR CAUGHT]"):
        return _paint(stripped, _Ansi.B_RED, _Ansi.BOLD) + "\n"
    if stripped.startswith("RESULT_JSON:"):
        return _paint(stripped, _Ansi.B_GREEN, _Ansi.BOLD) + "\n"
    low = stripped.lower()
    if "warn" in low[:20]:
        return _paint(stripped, _Ansi.B_YELLOW) + "\n"
    if "error" in low[:20] or "traceback" in low[:20]:
        return _paint(stripped, _Ansi.RED) + "\n"
    return _paint(stripped, _Ansi.GRAY) + "\n"


# ---------------------------------------------------------------------------
# Target schemas -- same three the hand-written reference scripts use.
# ---------------------------------------------------------------------------

MODE_SCHEMAS = {
    "pretrain": {
        "record_shape": '{"text": str, "source": str, "category": str}',
        "explanation": (
            "text is the full document body (article text, a Q&A pair joined "
            "into one block, whatever prose the dataset provides). No prompt/"
            "answer split needed."
        ),
    },
    "sft": {
        "record_shape": '{"prompt": str, "thinking": str, "answer": str, "source": str, "category": str}',
        "explanation": (
            "prompt is the question/instruction, answer is the final answer/"
            "solution, thinking is the chain-of-thought/derivation if the "
            "dataset has one (else an empty string \"\" -- never fabricate one). "
            "If the dataset has a single combined solution field with both "
            "reasoning and a final answer, put the reasoning portion in "
            "thinking and just the final answer/result in answer."
        ),
    },
    "grpo": {
        "record_shape": '{"prompt": str, "answer": str, "source": str, "category": str}',
        "explanation": (
            "prompt is the question, answer is ONLY the canonical ground-truth "
            "answer (numeric, boxed, or a short comparable string) -- no "
            "reasoning trace at all, since GRPO generates its own rollout."
        ),
    },
}


# ---------------------------------------------------------------------------
# Ollama call -- plain synchronous HTTP, no batching/async machinery needed
# since this is one call per *dataset*, not per row.
# ---------------------------------------------------------------------------

def call_ollama(prompt: str, timeout: float = 240.0, log_path: Optional[str] = None,
                 label: str = "") -> str:
    """Streams the generation from Ollama (stream=True) instead of waiting
    for the whole response, so the prompt and the model's output can both
    be shown live -- lets you watch/validate what's actually being sent
    and generated instead of staring at a blank screen for up to
    `timeout` seconds.

    The full prompt is printed to the console (boxed, dimmed) before the
    call, and every token Ollama streams back is echoed live as it
    arrives. If `log_path` is given, the full prompt AND the full
    streamed response are also appended to `<log_path>.ollama.log`
    (plain text, no ANSI codes) so you can review a run afterward even if
    you weren't watching the console at the time."""
    header = f"Ollama call{f' -- {label}' if label else ''} (model={OLLAMA_MODEL})"
    log_header(header)

    log_fh = None
    if log_path:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        log_fh = open(f"{log_path}.ollama.log", "a", encoding="utf-8")
        log_fh.write(f"\n{'=' * 80}\n"
                      f"# {label or 'ollama call'} @ {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                      f"{'=' * 80}\n## PROMPT\n{prompt}\n\n## RESPONSE (streamed)\n")
        log_dim(f"  full transcript -> {log_path}.ollama.log")

    print(_paint("  ┌─ PROMPT " + "─" * 60, _Ansi.B_BLUE, _Ansi.BOLD))
    for pline in prompt.splitlines() or [""]:
        print(_paint("  │ ", _Ansi.B_BLUE) + _paint(pline, _Ansi.DIM))
    print(_paint("  └" + "─" * 70, _Ansi.B_BLUE, _Ansi.BOLD))

    print(_paint("  ┌─ OLLAMA OUTPUT (live) " + "─" * 46, _Ansi.B_GREEN, _Ansi.BOLD))
    sys.stdout.write(_paint("  │ ", _Ansi.B_GREEN))
    sys.stdout.flush()

    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": True}
    chunks = []
    try:
        with httpx.Client(timeout=timeout) as client:
            with client.stream("POST", f"{OLLAMA_URL}/api/generate", json=payload) as resp:
                resp.raise_for_status()
                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue
                    try:
                        obj = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue
                    piece = obj.get("response", "")
                    if piece:
                        chunks.append(piece)
                        # keep the left margin lined up across the wrapped/streamed text
                        sys.stdout.write(_paint(piece.replace("\n", "\n  │ "), _Ansi.GREEN))
                        sys.stdout.flush()
                        if log_fh:
                            log_fh.write(piece)
                            log_fh.flush()
                    if obj.get("done"):
                        break
    finally:
        print()
        print(_paint("  └" + "─" * 70, _Ansi.B_GREEN, _Ansi.BOLD))
        if log_fh:
            log_fh.close()

    return "".join(chunks)


def _extract_code_block(text: str) -> str:
    """Ollama almost always wraps code in ```python fences even when told
    not to. Pull the largest fenced block out if present, else assume the
    whole response is code."""
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return max(blocks, key=len).strip() + "\n"
    return text.strip() + "\n"


# ---------------------------------------------------------------------------
# PHASE 1a (public datasets): discover + sample
# ---------------------------------------------------------------------------

_UNSUPPORTED_FEATURE_TYPES = {
    "Image", "Audio", "Video", "Array2D", "Array3D", "Array4D", "Array5D",
}


def _unsupported_feature_column(features) -> Optional[str]:
    """Inspects a `datasets.Features` mapping (available on a streaming
    dataset without pulling any rows) and returns the name of the first
    column whose type is unsupported for this text-only pipeline, or None
    if every column looks safe. Also looks one level into Sequence(...)
    wrappers, e.g. Sequence(Image())."""
    if not features:
        return None
    for col_name, feat in features.items():
        feat_type = type(feat).__name__
        if feat_type in _UNSUPPORTED_FEATURE_TYPES:
            return col_name
        inner = getattr(feat, "feature", None)
        if inner is not None and type(inner).__name__ in _UNSUPPORTED_FEATURE_TYPES:
            return col_name
    return None


def _row_has_unsupported_value(value) -> bool:
    """Belt-and-suspenders fallback for the (rarer) case where a dataset's
    declared Features don't reveal a binary column -- e.g. custom loading
    scripts -- but an actual sampled row still contains a non-JSON-safe
    object such as a PIL image, raw bytes, or a numpy array. Recurses into
    dicts/lists since HF rows are often nested."""
    if isinstance(value, dict):
        return any(_row_has_unsupported_value(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_row_has_unsupported_value(v) for v in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return False
    # Anything else -- PIL.Image.Image, bytes/bytearray, numpy arrays,
    # torch tensors, etc. -- is not something this text pipeline supports.
    return True


def sample_hf_dataset(dataset_id: str, config: Optional[str], split: str = "train",
                       n: int = 12) -> Optional[dict]:
    """Streams the first n rows of one dataset/config/split and returns
    {"columns": [...], "rows": [...]}, or None if it can't be opened at all
    (gated, doesn't exist, no matching split -- all treated as skip-this-
    candidate rather than fatal, same convention as public_sources.py) OR
    if it carries an unsupported dtype (images/audio/video/tensors) that
    this text-only pipeline can never turn into JSONL -- such datasets are
    skipped here, before any full download is ever kicked off."""
    try:
        from datasets import load_dataset
    except ImportError:
        log_error("`datasets` package not installed -- pip install datasets")
        return None
    for try_split in (split, "train", "validation", "test"):
        try:
            ds = load_dataset(dataset_id, config, split=try_split, streaming=True)

            bad_col = _unsupported_feature_column(getattr(ds, "features", None))
            if bad_col:
                log_warn(f"skipping {dataset_id} (config={config}): column "
                         f"'{bad_col}' has an unsupported dtype (image/audio/"
                         f"video/tensor) for this text-only pipeline")
                return None

            rows = list(itertools.islice(ds, n))
            if not rows:
                continue

            for row in rows:
                if any(_row_has_unsupported_value(v) for v in row.values()):
                    log_warn(f"skipping {dataset_id} (config={config}): sampled "
                             f"rows contain an unsupported (non-JSON-serializable) "
                             f"value for this text-only pipeline")
                    return None

            columns = sorted(rows[0].keys())

            if not schema_is_suitable(columns):
                # Same gate dataset_agent.py's public_sources path uses --
                # applied HERE, before codegen, is the expensive half of
                # the fix: without this, a dataset with no usable text/
                # prompt/answer/conversation column still gets a full
                # Ollama codegen call (which will hallucinate SOME mapping
                # rather than refuse -- it was never told "no" is an
                # option) followed by a full-dataset streaming download,
                # and only shows up as wasted time/junk output much later.
                # Rejecting on the columns alone, from the cheap n-row
                # sample already in hand, means bad datasets cost one
                # `sample_hf_dataset` call and nothing more.
                log_warn(f"skipping {dataset_id} (config={config}): columns {columns} "
                         f"don't match any known prompt/answer/text/conversation "
                         f"pattern -- rejecting before codegen, not worth an LLM call "
                         f"or a full download")
                return None

            return {"columns": columns, "rows": rows, "split": try_split}
        except Exception:
            continue
    return None


def discover_candidates(category: str, limit: int = 5) -> list:
    """Same discovery keywords dataset_agent.py's --public-only path uses
    (topics.HUB_SEARCH_KEYWORDS), so results are consistent between the two
    pipelines."""
    candidates = []
    seen = set()
    for kw in HUB_SEARCH_KEYWORDS.get(category, [category]):
        for dataset_id in discover_hf_datasets(kw, limit=limit):
            if dataset_id not in seen:
                seen.add(dataset_id)
                candidates.append(dataset_id)
    return candidates


# ---------------------------------------------------------------------------
# PHASE 1b (live web crawl): gather a raw batch, no LLM calls yet
# ---------------------------------------------------------------------------

async def _crawl_raw_batch_async(category: str, byte_budget: int, raw_path: str,
                                  max_results_per_query: int = 8) -> int:
    """Runs the MCP scraper's search()+extract() over this category's seed
    queries (topics.TOPIC_SEEDS) and appends {"url","text"} lines to
    raw_path until byte_budget is reached. No filtering/dedup/LLM calls
    here at all -- purely "get raw material fast", same spirit as a normal
    web crawler. Filtering happens once, afterward, in the generated
    cleanup script (phase 2)."""
    # Reused from dataset_agent.py rather than duplicated: MCP session
    # bootstrap + ScraperClient are already implemented there.
    from dataset_agent import ScraperClient, _find_server_path
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    server_path = os.environ.get("MCP_SERVER_PATH") or _find_server_path()
    # command=sys.executable alone guarantees the right interpreter binary,
    # but the MCP stdio transport doesn't necessarily forward the parent's
    # full environment by default -- pass it explicitly so HF_TOKEN,
    # KAGGLE_*, OLLAMA_URL etc. from the active env reach the server too.
    server_params = StdioServerParameters(command=sys.executable, args=[server_path], env=SUBPROCESS_ENV)
    written_bytes = 0
    seen_urls = set()
    queries = TOPIC_SEEDS.get(category, [category])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await asyncio.wait_for(session.initialize(), timeout=30)
            scraper = ScraperClient(session)
            with open(raw_path, "a", encoding="utf-8") as raw_fh:
                for query in itertools.cycle(queries):
                    if written_bytes >= byte_budget:
                        break
                    try:
                        hits = await scraper.search(query, max_results=max_results_per_query)
                    except Exception as e:
                        log_warn(f"search failed for {query!r}: {e}")
                        continue
                    for hit in hits:
                        if written_bytes >= byte_budget:
                            break
                        url = hit.get("url") if isinstance(hit, dict) else None
                        if not url or url in seen_urls:
                            continue
                        seen_urls.add(url)
                        try:
                            doc = await scraper.extract(url)
                        except Exception as e:
                            log_warn(f"extract failed for {url}: {e}")
                            continue
                        text = (doc or {}).get("text") or (doc or {}).get("content") or ""
                        if not text or len(text) < 50:
                            continue
                        line = json.dumps({"url": url, "text": text}, ensure_ascii=False) + "\n"
                        raw_fh.write(line)
                        written_bytes += len(line.encode("utf-8"))
                        if written_bytes % (1024 * 256) < len(line):
                            msg = _paint(
                                f"[{category}] raw crawl: {written_bytes/1024**2:.1f} MB "
                                f"/ {byte_budget/1024**2:.1f} MB", _Ansi.B_CYAN)
                            print(msg, end="\r")
                    # itertools.cycle over a small finite list forever would
                    # spin if every query keeps failing/returning nothing;
                    # bail once we've gone through all queries with zero
                    # progress rather than looping forever.
                    if len(seen_urls) == 0 and query == queries[-1]:
                        break
    print()
    return written_bytes


def crawl_raw_batch(category: str, byte_budget: int, raw_path: str) -> int:
    """Sync wrapper -- the rest of this module (and its CLI) stays plain
    synchronous code, matching build_dataset.py's style; only this one
    function dips into asyncio, because the MCP scraper protocol requires
    it, and that's hidden here."""
    os.makedirs(os.path.dirname(raw_path) or ".", exist_ok=True)
    log_section(category, f"crawling raw batch -> {raw_path} "
                f"(target {byte_budget/1024**2:.1f} MB)")
    return asyncio.run(_crawl_raw_batch_async(category, byte_budget, raw_path))


def sample_raw_batch(raw_path: str, n: int = 8) -> list:
    rows = []
    with open(raw_path, "r", encoding="utf-8") as f:
        for line in itertools.islice(f, n):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


# ---------------------------------------------------------------------------
# PHASE 2: codegen -- ask Ollama to write the extractor/cleanup script
# ---------------------------------------------------------------------------

_MAP_ROW_API_SUMMARY = """\
Do NOT apply any quality/length heuristics, dedup, or file writing yourself
-- a trusted harness (agent/harness.py) already does all of that AFTER your
function returns. Your ONLY job is correct column -> target-field mapping
(and, for sft/grpo, splitting a combined field into prompt/thinking/answer
when the dataset only provides one blob). Returning a row that's short,
boilerplate-y, or low-quality is fine -- the harness's quality filter will
reject it; that is not your responsibility and adding your own heuristics
here just duplicates (and can conflict with) a filter you can't see.
"""


def build_hf_map_row_prompt(category: str, mode: str, dataset_id: str, config: Optional[str],
                             split: str, columns: list, rows: list,
                             prior_error: Optional[str] = None) -> str:
    schema = MODE_SCHEMAS[mode]
    # rows should already be pure JSON-safe types -- sample_hf_dataset()
    # filters out anything with an unsupported dtype (images/audio/etc.)
    # before it ever gets here. default=str is only a last-ditch safety net
    # so a stray non-serializable value degrades to a string instead of
    # crashing the whole pipeline.
    sample_json = json.dumps(rows[:5], ensure_ascii=False, indent=2, default=str)[:4000]
    repair_note = ""
    if prior_error:
        repair_note = (
            f"\nThe previous version of this function failed like this -- fix it, keep "
            f"everything else the same. This may be a compile error, an exception raised "
            f"while actually calling map_row() on real sample rows, or a pytest assertion "
            f"failure (read it carefully, it names the exact problem):\n{prior_error}\n"
        )
    return f"""You write ONE pure Python function, nothing else. No explanation, no
markdown outside a single ```python code fence.

TASK: write `def map_row(row: dict) -> Optional[dict]:` that maps ONE row
from the Hugging Face dataset "{dataset_id}" (config={config!r}, split="{split}")
to this target record shape for mode="{mode}":
    {schema['record_shape']}
    {schema['explanation']}
    Always set "source" to the literal string "{dataset_id}" and "category"
    to the literal string "{category}".

This dataset's actual columns: {columns}
First few real rows (use these to figure out which column(s) map to which
target field -- do NOT guess generic column names like "text" if they aren't
in the list above -- this dataset's real schema is authoritative, not your
prior assumptions about what a dataset like this "usually" looks like):
{sample_json}

{_MAP_ROW_API_SUMMARY}
Function requirements:
- Signature exactly `def map_row(row: dict) -> Optional[dict]:`.
- Use `row.get(...)` with sensible defaults, never bare `row[...]` indexing
  a column name that might be missing/None on some rows -- a KeyError or
  AttributeError here aborts the whole run, it is not treated as "just
  skip this row".
- Return None (not an empty dict, not a dict with empty string fields) if
  this particular row genuinely can't be mapped (e.g. every candidate
  source column is missing or None on this row).
- Every value in the returned dict must be a plain `str`. Cast/`str(...)`
  non-string values (numbers, etc.) rather than leaving them as-is.
- Pure function: no network calls, no file I/O, no `import datasets`, no
  loops over a dataset, no argparse, no top-level executable code besides
  imports and this function (a couple of tiny module-level regex/constant
  helpers are fine if map_row calls them).
{repair_note}
Output ONLY the function (and any small helpers it calls) in a single
```python fence.
"""


def build_web_map_row_prompt(category: str, mode: str, raw_path: str, sample_rows: list,
                              prior_error: Optional[str] = None) -> str:
    schema = MODE_SCHEMAS[mode]
    sample_preview = json.dumps(
        [{"url": r.get("url"), "text": (r.get("text") or "")[:800]} for r in sample_rows[:4]],
        ensure_ascii=False, indent=2,
    )[:4000]
    repair_note = ""
    if prior_error:
        repair_note = (
            f"\nThe previous version of this function failed like this -- fix it, keep "
            f"everything else the same. This may be a compile error, an exception raised "
            f"while actually calling map_row() on real sample rows, or a pytest assertion "
            f"failure (read it carefully, it names the exact problem):\n{prior_error}\n"
        )
    return f"""You write ONE pure Python function, nothing else. No explanation, no
markdown outside a single ```python code fence.

TASK: write `def map_row(row: dict) -> Optional[dict]:` where `row` is one
{{"url": str, "text": str}} object from a raw scraped-web-page batch (the raw
markdown/text extracted from each page, likely still containing some nav/
boilerplate). Map it to this target record shape for mode="{mode}":
    {schema['record_shape']}
    {schema['explanation']}
    Set "source" to row["url"] and "category" to the literal string "{category}".

A few real sample rows (look for recurring boilerplate patterns across them
-- nav menus, cookie notices, "subscribe" prompts, etc -- and strip anything
you notice; it's fine if some slips through, the harness's own junk-marker
filter is a second line of defense, not your only one):
{sample_preview}

{_MAP_ROW_API_SUMMARY}
Function requirements:
- Signature exactly `def map_row(row: dict) -> Optional[dict]:`.
- {"For pretrain mode: use row['text'] mostly as-is after stripping obvious boilerplate lines you can identify (nav/cookie/subscribe banners) -- do not try to be clever about quality, just clean and pass through." if mode == "pretrain" else "This raw batch is unstructured scraped prose with no natural prompt/answer split -- only map pages that look like a coherent Q&A, tutorial, or worked-example (return None for everything else), splitting into prompt/answer where a clear question/answer structure is visible in the text."}
- Use `row.get(...)` defensively; never let a missing/None field raise --
  return None for that row instead.
- Every value in the returned dict must be a plain `str`.
- Pure function: no network calls, no file I/O, no loops over a file, no
  argparse, no top-level executable code besides imports and this function.
{repair_note}
Output ONLY the function (and any small helpers it calls) in a single
```python fence.
"""


# ---------------------------------------------------------------------------
# Validate + repair loop, then run
# ---------------------------------------------------------------------------

def _py_compile_check(script_path: str) -> Optional[str]:
    """Returns None if the module compiles cleanly, else the error text."""
    result = subprocess.run(
        [sys.executable, "-m", "py_compile", script_path],
        capture_output=True, text=True, env=SUBPROCESS_ENV,
    )
    return None if result.returncode == 0 else (result.stderr or result.stdout)


def generate_map_row_module(prompt_fn, script_path: str, prior_error: Optional[str] = None,
                             log_path: Optional[str] = None, attempt_label: str = "") -> Optional[str]:
    """prompt_fn(prior_error) -> prompt string. One generate+compile-check
    attempt (no internal retry loop -- the caller, generate_test_and_run,
    owns the retry budget across compile/test/runtime failures combined).
    Returns the compile error string on failure, or None on success."""
    log_step(f"codegen: writing map_row() for {_paint(os.path.basename(script_path), _Ansi.BOLD)}...")
    response = call_ollama(prompt_fn(prior_error), log_path=log_path,
                            label=f"{os.path.basename(script_path)} -- {attempt_label}")
    code = _extract_code_block(response)
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(code)
    error = _py_compile_check(script_path)
    if error is None:
        log_success(f"codegen: {os.path.basename(script_path)} compiles OK")
    else:
        log_warn("codegen: compile error:")
        log_dim(error[:500])
    return error


def test_map_row_module(script_path: str, sample_rows: list, mode: str, log_path: str) -> Optional[str]:
    """Runs the auto-generated pytest suite (test_gen.py) against map_row()
    using the same real sample rows the model was shown. Returns None if
    every test passed, else the pytest failure output (truncated) to feed
    back into the repair prompt. This is the stage that turns 'wrong column
    name' / 'always returns None' / 'crashes on a null field' into a
    sub-second local failure instead of a wasted full-scale run."""
    test_path = os.path.splitext(script_path)[0] + "_test.py"
    log_step(f"test: running pre-flight pytest for "
             f"{_paint(os.path.basename(script_path), _Ansi.BOLD)} against "
             f"{len(sample_rows)} real sample row(s)...")
    test_gen.write_pytest_file(script_path, sample_rows, mode, test_path, agent_dir=AGENT_DIR)
    passed, output = test_gen.run_pytest_file(test_path, env=SUBPROCESS_ENV)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n--- pytest pre-flight for {os.path.basename(script_path)} ---\n{output}\n")
    if passed:
        log_success(f"test: {os.path.basename(script_path)} passed pre-flight tests "
                    f"-- proceeding to full run")
        return None
    log_warn(f"test: {os.path.basename(script_path)} FAILED pre-flight tests "
             f"(see {test_path} / {log_path}):")
    log_dim(output[-800:])
    return output[-4000:]


# The harness prints "[FATAL ERROR CAUGHT]: ..." for a systemic runtime
# failure (see agent/harness.py's run()); pull it back out of the captured
# output and feed it to Ollama as a repair prompt, same convention the
# compile/test stages already use.
_FATAL_ERROR_RE = re.compile(r"^\[FATAL ERROR CAUGHT\]:\s*(.*)$")


def run_via_harness(script_path: str, target_bytes: int, out_dir: str, category: str, mode: str,
                     min_doc_chars: int, log_path: str, source_args: list) -> dict:
    """Runs agent/harness.py (trusted, hand-written, unit-tested) as a
    subprocess with `--map-module script_path`, streaming output live to
    both the console and a log file, same UX as the old whole-script run
    had. The harness itself owns streaming/filtering/dedup/writing/progress/
    RESULT_JSON -- this function's only jobs are process management and
    parsing that trailing RESULT_JSON line for the manifest."""
    cmd = [sys.executable, HARNESS_PY_PATH,
           "--map-module", script_path, "--mode", mode, "--category", category,
           "--target-size-bytes", str(target_bytes), "--out-dir", out_dir,
           "--min-doc-chars", str(min_doc_chars)] + source_args
    log_step(f"run: {' '.join(cmd)}")
    log_dim(f"    (using {sys.executable}, same env this pipeline is running in)")
    log_dim(f"    logging to {log_path}")
    result_json = {"actual_bytes": 0, "docs": 0}
    fatal_error = None
    tail_lines = []
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log_fh:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, bufsize=1, env=SUBPROCESS_ENV)
        for line in proc.stdout:
            sys.stdout.write(_colorize_passthrough(line))
            log_fh.write(line)
            tail_lines.append(line)
            tail_lines[:] = tail_lines[-40:]
            m = _FATAL_ERROR_RE.match(line.strip())
            if m:
                fatal_error = m.group(1).strip()
            if line.startswith("RESULT_JSON:"):
                try:
                    result_json = json.loads(line[len("RESULT_JSON:"):].strip())
                except json.JSONDecodeError:
                    pass
        proc.wait()
    if proc.returncode != 0:
        log_warn(f"run: harness exited with code {proc.returncode} "
                 f"(see {log_path}) -- using whatever RESULT_JSON it printed, if any")
        if fatal_error is None:
            fatal_error = "".join(tail_lines[-15:]).strip() or f"process exited with code {proc.returncode}"
    result_json["error"] = fatal_error
    return result_json


def generate_test_and_run(prompt_fn, script_path: str, target_bytes: int, out_dir: str,
                           category: str, mode: str, min_doc_chars: int, log_path: str,
                           sample_rows: list, source_args: list, max_repair: int = 2) -> dict:
    """The full plan -> generate -> TEST -> run loop for one dataset/raw
    batch. A failure at ANY stage -- compile error, pytest pre-flight
    failure, or a runtime "[FATAL ERROR CAUGHT]" from the harness -- is fed
    back into prompt_fn as prior_error and map_row() is regenerated, up to
    max_repair times total across all three stages combined. The pytest
    stage in particular is what catches a bad column-name guess or an
    always-None mapping in well under a second, using the exact sample rows
    the model was shown, instead of only finding out after a real run."""
    prior_error = None
    result = {"actual_bytes": 0, "docs": 0, "error": None}
    for attempt in range(max_repair + 1):
        label = f"attempt {attempt + 1}/{max_repair + 1}"

        compile_error = generate_map_row_module(
            prompt_fn, script_path, prior_error=prior_error, log_path=log_path, attempt_label=label)
        if compile_error is not None:
            prior_error = compile_error
            log_warn(f"codegen: {label} failed to compile, "
                     f"{'retrying' if attempt < max_repair else 'giving up'}...")
            continue

        test_error = test_map_row_module(script_path, sample_rows, mode, log_path)
        if test_error is not None:
            prior_error = test_error
            if attempt < max_repair:
                log_step(f"codegen: regenerating {os.path.basename(script_path)} "
                         f"with the pytest failure fed back to Ollama...")
            continue

        result = run_via_harness(
            script_path, target_bytes, out_dir, category, mode, min_doc_chars,
            log_path, source_args,
        )
        if result.get("error") is None:
            return result

        log_warn(f"codegen: harness runtime error on {label}:")
        log_dim(result["error"][:500])
        if attempt < max_repair:
            log_step(f"codegen: regenerating {os.path.basename(script_path)} with this "
                     f"runtime error fed back to Ollama...")
        prior_error = result["error"]

    log_error(f"codegen: giving up on {os.path.basename(script_path)} after "
              f"{max_repair + 1} generate+test+run attempt(s)")
    return result


# ---------------------------------------------------------------------------
# Top-level: one category, either mode
# ---------------------------------------------------------------------------

def run_category_public(category: str, budget_bytes: int, out_dir: str, mode: str,
                        min_doc_chars: int, discover_limit: int = 5,
                        max_candidates_to_try: int = 3,
                        max_total_considered: int = 40) -> dict:
    """Tries candidate datasets one at a time until either budget_bytes is
    filled or max_candidates_to_try datasets have *actually contributed*
    data. A dataset that gets rejected -- whether pre-download (unsupported
    dtype, gated, doesn't exist -- sample_hf_dataset() returns None) or
    post-download (codegen/run produced zero usable bytes) -- does NOT
    consume one of those max_candidates_to_try slots and is not counted
    toward the quota; the loop just moves on to the next candidate. If the
    initially discovered candidate list runs dry before the quota is met,
    more candidates are pulled in (widening the search) rather than giving
    up early. max_total_considered is an absolute safety ceiling on how
    many datasets we'll ever look at for one category, regardless of
    outcome, so a category with nothing but bad candidates can't loop
    forever."""
    scripts_dir = os.path.join(out_dir, "_generated_scripts")
    logs_dir = os.path.join(out_dir, "_logs")
    os.makedirs(scripts_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    candidates = discover_candidates(category, limit=discover_limit)
    log_section(category, f"discovered {_paint(str(len(candidates)), _Ansi.BOLD)} "
                f"candidate dataset(s): {candidates}")

    total_bytes, total_docs = 0, 0
    tried = 0       # datasets that actually contributed data -- gates max_candidates_to_try
    rejected = 0    # datasets skipped/failed and NOT counted toward the quota
    considered = set()
    idx = 0
    widen_limit = discover_limit

    while total_bytes < budget_bytes and tried < max_candidates_to_try:
        if idx >= len(candidates):
            # Ran out of candidates without filling the quota -- widen the
            # discovery search instead of giving up.
            widen_limit *= 2
            more = [c for c in discover_candidates(category, limit=widen_limit)
                     if c not in considered]
            if not more:
                log_warn(f"[{category}] no more candidate datasets left to try "
                         f"(considered {len(considered)}) -- stopping short of quota")
                break
            candidates.extend(more)

        dataset_id = candidates[idx]
        idx += 1
        if dataset_id in considered:
            continue
        considered.add(dataset_id)
        if len(considered) > max_total_considered:
            log_warn(f"[{category}] hit the {max_total_considered}-dataset safety "
                     f"ceiling -- stopping short of quota")
            break

        configs = discover_hf_configs(dataset_id) or [None]
        sample, config = None, None
        for config in configs:
            log_section(category, f"sampling {_paint(dataset_id, _Ansi.BOLD)} (config={config})...")
            sample = sample_hf_dataset(dataset_id, config)
            if sample is not None:
                break
            log_warn(f"[{category}] {dataset_id} config={config} rejected -- "
                     f"trying next config of this dataset, if any")
        if sample is None:
            log_warn(f"[{category}] {dataset_id} rejected before download -- "
                     f"no config ({configs}) had usable columns -- "
                     f"not counted toward quota, trying next candidate")
            rejected += 1
            continue

        safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", f"{category}_{dataset_id}")
        script_path = os.path.join(scripts_dir, f"{safe_name}.py")
        remaining = budget_bytes - total_bytes
        result = generate_test_and_run(
            lambda err: build_hf_map_row_prompt(
                category, mode, dataset_id, config, sample["split"],
                sample["columns"], sample["rows"], prior_error=err),
            script_path, remaining, out_dir, category, mode, min_doc_chars,
            log_path=os.path.join(logs_dir, f"{safe_name}.log"),
            sample_rows=sample["rows"],
            source_args=["--source", "hf", "--dataset-id", dataset_id,
                         "--split", sample["split"]] + (["--config", config] if config else []),
        )
        if result.get("actual_bytes", 0) <= 0:
            log_warn(f"[{category}] {dataset_id} produced no usable data after "
                     f"download -- not counted toward quota, trying next candidate")
            rejected += 1
            continue

        tried += 1
        total_bytes += result.get("actual_bytes", 0)
        total_docs += result.get("docs", 0)
        log_success(f"[{category}] after {dataset_id}: {total_bytes/1024**2:.1f} MB / "
                    f"{budget_bytes/1024**2:.1f} MB, {total_docs} docs total "
                    f"({rejected} rejected so far)")

    return {"target_bytes": budget_bytes, "actual_bytes": total_bytes, "docs": total_docs,
            "candidates_tried": tried, "candidates_rejected": rejected}


def run_category_web(category: str, budget_bytes: int, out_dir: str, mode: str,
                     min_doc_chars: int, raw_headroom: float = 1.4) -> dict:
    scripts_dir = os.path.join(out_dir, "_generated_scripts")
    logs_dir = os.path.join(out_dir, "_logs")
    raw_dir = os.path.join(out_dir, "_raw")
    os.makedirs(scripts_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    raw_path = os.path.join(raw_dir, f"{category}_raw.jsonl")
    raw_budget = int(budget_bytes * raw_headroom)
    if not os.path.exists(raw_path) or os.path.getsize(raw_path) < raw_budget:
        crawl_raw_batch(category, raw_budget, raw_path)
    else:
        log_section(category, f"reusing existing raw batch at {raw_path}")

    sample_rows = sample_raw_batch(raw_path)
    if not sample_rows:
        log_warn(f"[{category}] raw batch is empty -- nothing to process")
        return {"target_bytes": budget_bytes, "actual_bytes": 0, "docs": 0}

    safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", f"{category}_web")
    script_path = os.path.join(scripts_dir, f"{safe_name}.py")
    result = generate_test_and_run(
        lambda err: build_web_map_row_prompt(category, mode, raw_path, sample_rows, prior_error=err),
        script_path, budget_bytes, out_dir, category, mode, min_doc_chars,
        log_path=os.path.join(logs_dir, f"{safe_name}.log"),
        sample_rows=sample_rows,
        source_args=["--source", "raw", "--raw-path", raw_path],
    )
    return {"target_bytes": budget_bytes, "actual_bytes": result.get("actual_bytes", 0),
            "docs": result.get("docs", 0)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_size(size_str: str) -> int:
    size_str = size_str.strip().upper()
    for unit, mult in (("GB", 1024**3), ("MB", 1024**2), ("KB", 1024), ("B", 1)):
        if size_str.endswith(unit):
            return int(float(size_str[: -len(unit)]) * mult)
    return int(float(size_str))


def main():
    parser = argparse.ArgumentParser(
        description="Simple discover -> sample -> codegen -> run pipeline "
                    "(public datasets and/or live web crawling).")
    parser.add_argument("--target-size", required=True)
    parser.add_argument("--out-dir", default="./data")
    parser.add_argument("--categories", default="web,knowledge,reasoning,code,math,science")
    parser.add_argument("--mode", choices=list(MODE_SCHEMAS.keys()), default="pretrain")
    parser.add_argument("--min-doc-chars", type=int, default=500)
    parser.add_argument("--public-only", action="store_true",
                        help="Use public HF datasets only (discover+sample+codegen+run). "
                             "Without this flag, does a live web crawl instead.")
    parser.add_argument("--mix", default=None,
                        help="Comma-separated category=fraction, e.g. web=0.5,math=0.5. "
                             "Defaults to an even split across --categories.")
    parser.add_argument("--discover-limit", type=int, default=5,
                        help="[--public-only] How many HF Hub results to pull per search "
                             "keyword when discovering candidate datasets for a category "
                             "(each category has ~3 keywords, so the initial candidate pool "
                             "is roughly 3x this number). Default: 5.")
    parser.add_argument("--max-candidates-to-try", type=int, default=3,
                        help="[--public-only] How many datasets must actually contribute "
                             "usable data before a category is considered done (rejected/"
                             "empty datasets don't count against this). Default: 3.")
    parser.add_argument("--max-total-considered", type=int, default=40,
                        help="[--public-only] Absolute safety ceiling on how many candidate "
                             "datasets will ever be looked at for one category, including "
                             "rejected ones, before giving up on the quota. Default: 40.")
    args = parser.parse_args()

    _env_banner()

    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    target_bytes = parse_size(args.target_size)
    if args.mix:
        mix = {}
        for part in args.mix.split(","):
            k, v = part.split("=")
            mix[k.strip()] = float(v)
    else:
        mix = {c: 1.0 / len(categories) for c in categories}

    os.makedirs(args.out_dir, exist_ok=True)
    manifest = {"target_bytes": target_bytes, "mode": args.mode, "mix": mix, "categories": {}}

    for category, frac in mix.items():
        budget = int(target_bytes * frac)
        if budget <= 0:
            continue
        log_header(f"[{category}] target {budget/1024**2:.1f} MB "
                   f"({'public datasets' if args.public_only else 'live web crawl'})")
        if args.public_only:
            manifest["categories"][category] = run_category_public(
                category, budget, args.out_dir, args.mode, args.min_doc_chars,
                discover_limit=args.discover_limit,
                max_candidates_to_try=args.max_candidates_to_try,
                max_total_considered=args.max_total_considered)
        else:
            manifest["categories"][category] = run_category_web(
                category, budget, args.out_dir, args.mode, args.min_doc_chars)

    manifest_path = os.path.join(args.out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    total_actual = sum(c["actual_bytes"] for c in manifest["categories"].values())
    log_header(f"Done -- {total_actual/1024**2:.2f} MB across "
               f"{len(manifest['categories'])} categories")
    log_success(f"manifest written to {manifest_path}")
    log_dim(f"generated scripts kept under {os.path.join(args.out_dir, '_generated_scripts')} "
            f"-- re-runs skip regeneration if you rerun the same script directly.")


if __name__ == "__main__":
    main()
