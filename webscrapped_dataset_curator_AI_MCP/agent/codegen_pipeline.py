#!/usr/bin/env python3
"""
codegen_pipeline.py

A deliberately simpler alternative to dataset_agent.py's per-row async
pipeline (search -> extract -> LLM-judge -> dedup -> write, one row at a
time, for every single document). That approach is thorough but slow and
complex: an Ollama call per document, an async governor, a dedup layer
that has to stay bounded across the whole run, etc.

This module does the same job the way build_dataset.py / download_sft_data.py
/ download_grpo_data.py already do it by hand: write ONE plain Python
extractor script per dataset (a `load_dataset(..., streaming=True)` loop +
a small function mapping that dataset's specific columns to the target
schema), then just run it. The only difference from those reference
scripts is *who* writes the extractor function -- instead of a human
hand-coding one entry per dataset in a SOURCES dict, an LLM call writes it
once per dataset, after being shown that dataset's actual columns and a
few real rows. After that one generation step, everything is plain,
synchronous, fast Python with zero further LLM calls -- exactly like the
reference scripts.

Two phases, used the same way for both public datasets and live web
crawling:

    PUBLIC DATASETS (--public-only)
        1. discover  -- reuse public_sources.discover_hf_datasets() to find
           candidate datasets for a category (same as dataset_agent.py).
        2. sample    -- stream the first N rows, note the columns.
        3. codegen   -- ask Ollama to write a standalone script (import
           quality.py's ExactDedup/ShardWriter/quality filters, same as
           build_dataset.py) that streams the FULL dataset, maps this
           dataset's specific columns to the target schema, filters,
           dedups, and writes shards.
        4. validate  -- py_compile it; on failure, send the error back to
           Ollama once or twice to fix, rather than giving up.
        5. run       -- plain `subprocess.run`, output streamed to both the
           console and a per-dataset log file.

    LIVE WEB CRAWL (default, no --public-only)
        1. crawl-raw -- gather a batch of raw {url, text} pages via the
           MCP scraper (search + extract, no LLM judge, no per-row work)
           into one JSONL file, up to a byte budget with headroom for
           later filtering losses.
        2. codegen   -- show Ollama a few raw samples and ask for a script
           that reads the raw batch, cleans/filters/dedups it (via the
           same quality.py functions), and writes final shards.
        3. validate + run, same as above.

Usage:
    python codegen_pipeline.py --target-size 500MB --public-only \\
        --categories web,knowledge,math --out-dir ./data --mode pretrain

    python codegen_pipeline.py --target-size 200MB \\
        --categories web,knowledge --out-dir ./data --mode pretrain
"""

from __future__ import annotations

import argparse
import asyncio
import io
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
from public_sources import discover_hf_datasets, discover_hf_configs
from topics import HUB_SEARCH_KEYWORDS, TOPIC_SEEDS

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")
QUALITY_PY_PATH = os.path.join(os.path.dirname(__file__), "quality.py")

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

def call_ollama(prompt: str, timeout: float = 240.0) -> str:
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(f"{OLLAMA_URL}/api/generate", json=payload)
        resp.raise_for_status()
        return resp.json().get("response", "")


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

def sample_hf_dataset(dataset_id: str, config: Optional[str], split: str = "train",
                       n: int = 12) -> Optional[dict]:
    """Streams the first n rows of one dataset/config/split and returns
    {"columns": [...], "rows": [...]}, or None if it can't be opened at all
    (gated, doesn't exist, no matching split -- all treated as skip-this-
    candidate rather than fatal, same convention as public_sources.py)."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("[error] `datasets` package not installed -- pip install datasets")
        return None
    for try_split in (split, "train", "validation", "test"):
        try:
            ds = load_dataset(dataset_id, config, split=try_split, streaming=True)
            rows = list(itertools.islice(ds, n))
            if not rows:
                continue
            columns = sorted(rows[0].keys())
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
    server_params = StdioServerParameters(command=sys.executable, args=[server_path])
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
                        print(f"[warn] search failed for {query!r}: {e}")
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
                            print(f"[warn] extract failed for {url}: {e}")
                            continue
                        text = (doc or {}).get("text") or (doc or {}).get("content") or ""
                        if not text or len(text) < 50:
                            continue
                        line = json.dumps({"url": url, "text": text}, ensure_ascii=False) + "\n"
                        raw_fh.write(line)
                        written_bytes += len(line.encode("utf-8"))
                        if written_bytes % (1024 * 256) < len(line):
                            print(f"[{category}] raw crawl: {written_bytes/1024**2:.1f} MB "
                                  f"/ {byte_budget/1024**2:.1f} MB", end="\r")
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
    print(f"[{category}] crawling raw batch -> {raw_path} "
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

_QUALITY_API_SUMMARY = """\
A module `quality.py` is importable from the same directory (add
`sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))` before
importing it) with these ALREADY-IMPLEMENTED functions/classes -- use them,
do not reimplement filtering or dedup logic yourself:

    passes_prose_quality_filter(text: str, min_doc_chars: int = 500) -> (bool, reason)
        Use for pretrain-mode "text" fields (or any free-form prose).

    passes_sft_pair_quality_filter(prompt: str, answer: str, min_chars: int = 20) -> (bool, reason)
        Use for sft/grpo-mode (prompt, answer) pairs.

    passes_code_quality_filter(text: str, path: str, min_doc_chars: int = 500) -> (bool, reason)
        Use only if the category is "code".

    ExactDedup(persist_path: Optional[str] = None)
        .is_duplicate(text: str) -> bool   # call once per record on its
                                            # main text content before writing

    ShardWriter(out_dir: str, category: str)
        .write(record: dict)   # pass the full final record dict
        .close()
        .total_bytes / .total_docs   # read after the loop for a summary
"""


def build_hf_codegen_prompt(category: str, mode: str, dataset_id: str, config: Optional[str],
                             split: str, columns: list, rows: list,
                             prior_error: Optional[str] = None) -> str:
    schema = MODE_SCHEMAS[mode]
    sample_json = json.dumps(rows[:5], ensure_ascii=False, indent=2)[:4000]
    repair_note = ""
    if prior_error:
        repair_note = (
            f"\nThe previous version of this script failed with this error -- fix it, "
            f"keep everything else the same:\n{prior_error}\n"
        )
    return f"""You write ONE standalone Python script, nothing else. No explanation, no
markdown outside a single ```python code fence.

TASK: write a complete, runnable Python script that streams the Hugging Face
dataset "{dataset_id}" (config={config!r}, split="{split}") in FULL (not just
the sample below) via `datasets.load_dataset(..., streaming=True)`, maps each
row to this target record shape, filters, dedups, and writes JSONL shards.

Target record shape for mode="{mode}":
    {schema['record_shape']}
    {schema['explanation']}
    Always set "source" to "{dataset_id}" and "category" to the value of a
    --category CLI arg (default "{category}").

This dataset's actual columns: {columns}
First few real rows (use these to figure out which column(s) map to which
target field -- do NOT guess generic column names like "text" if they aren't
in the list above):
{sample_json}

{_QUALITY_API_SUMMARY}
Script requirements:
- `import argparse` with `--target-size` (e.g. "500MB", parse with a helper
  you write: supports GB/MB/KB/B suffixes, case-insensitive) and `--out-dir`
  (default "./data") and `--category` (default "{category}") and
  `--min-doc-chars` (default 500).
- Loop over the streamed dataset, map columns -> target schema fields using
  this dataset's REAL column names from above.
- Skip rows that fail the appropriate quality.py filter or are exact
  duplicates (ExactDedup.is_duplicate on the main text/answer content).
- Write passing rows via ShardWriter(out_dir, category).write(record).
- Print progress roughly every 5 seconds: bytes written / target, docs
  written, filtered counts -- use plain print(), this runs as a subprocess
  whose stdout is captured to a log file.
- Stop once ShardWriter.total_bytes >= target_bytes, or the stream is
  exhausted (print a clear message either way).
- At the very end print one line starting with exactly "RESULT_JSON:"
  followed by a single-line JSON object: {{"actual_bytes": int, "docs": int}}
  -- the caller parses this line to build a manifest, so it must be the
  last line printed and must be valid JSON on that one line.
- Wrap the whole streaming loop in a broad try/except that prints the error
  and still calls ShardWriter.close() and prints the RESULT_JSON line with
  whatever was written so far, rather than crashing with a traceback and no
  usable output.
{repair_note}
Output ONLY the script in a single ```python fence.
"""


def build_web_codegen_prompt(category: str, mode: str, raw_path: str, sample_rows: list,
                              prior_error: Optional[str] = None) -> str:
    schema = MODE_SCHEMAS[mode]
    sample_preview = json.dumps(
        [{"url": r.get("url"), "text": (r.get("text") or "")[:800]} for r in sample_rows[:4]],
        ensure_ascii=False, indent=2,
    )[:4000]
    repair_note = ""
    if prior_error:
        repair_note = (
            f"\nThe previous version of this script failed with this error -- fix it, "
            f"keep everything else the same:\n{prior_error}\n"
        )
    return f"""You write ONE standalone Python script, nothing else. No explanation, no
markdown outside a single ```python code fence.

TASK: write a complete, runnable Python script that reads a raw JSONL file of
scraped web pages (one {{"url": str, "text": str}} object per line -- the raw
markdown/text extracted from each page, likely still containing some nav/
boilerplate) and produces cleaned, filtered, deduped JSONL shards in this
target record shape for mode="{mode}":
    {schema['record_shape']}
    {schema['explanation']}
    Set "source" to the row's "url" and "category" to "{category}".

A few real sample rows from the raw file (look for recurring boilerplate
patterns across them -- nav menus, cookie notices, "subscribe" prompts, etc
-- and strip anything you notice that quality.py's junk-marker check
wouldn't already catch):
{sample_preview}

{_QUALITY_API_SUMMARY}
Script requirements:
- `import argparse` with `--raw-path` (default "{raw_path}"), `--target-size`
  (parse GB/MB/KB/B suffixes), `--out-dir` (default "./data"), `--category`
  (default "{category}"), `--min-doc-chars` (default 500).
- Read the raw JSONL file line by line (it can be large -- do NOT load it
  all into memory at once, iterate the file object).
- {"For pretrain mode: use the page text mostly as-is after cleanup (strip obvious boilerplate lines), pass through passes_prose_quality_filter." if mode == "pretrain" else "This raw batch is unstructured scraped prose with no natural prompt/answer split -- do your best to only keep pages that look like a coherent Q&A, tutorial, or worked-example (skip everything else), splitting into prompt/answer where a clear question/answer structure is visible in the text, else skip the page."}
- Apply ExactDedup on the main text content before writing.
- Write passing rows via ShardWriter(out_dir, category).write(record).
- Print progress roughly every 5 seconds like a normal long-running script.
- Stop once ShardWriter.total_bytes >= target_bytes, or the raw file is
  exhausted.
- At the very end print one line starting with exactly "RESULT_JSON:"
  followed by a single-line JSON object: {{"actual_bytes": int, "docs": int}}
  as the LAST line printed.
- Wrap the main loop in a broad try/except that still closes the writer and
  prints RESULT_JSON with whatever was produced so far, rather than crashing
  with no usable output.
{repair_note}
Output ONLY the script in a single ```python fence.
"""


# ---------------------------------------------------------------------------
# Validate + repair loop, then run
# ---------------------------------------------------------------------------

def _py_compile_check(script_path: str) -> Optional[str]:
    """Returns None if the script compiles cleanly, else the error text."""
    result = subprocess.run(
        [sys.executable, "-m", "py_compile", script_path],
        capture_output=True, text=True,
    )
    return None if result.returncode == 0 else (result.stderr or result.stdout)


def generate_and_validate_script(prompt_fn, script_path: str, max_repair: int = 2,
                                  prior_error: Optional[str] = None) -> bool:
    """prompt_fn(prior_error) -> prompt string. Generates, compiles, and on
    failure re-prompts with the exact error up to max_repair times. Returns
    True if a compiling script now exists at script_path.

    `prior_error` can be seeded from outside (e.g. a previous *runtime*
    failure passed in by generate_validate_and_run below) so the very first
    generation attempt already includes it, not just compile errors found
    here."""
    for attempt in range(max_repair + 1):
        label = "generating" if attempt == 0 else f"repairing (attempt {attempt})"
        print(f"  [codegen] {label} {os.path.basename(script_path)}...")
        response = call_ollama(prompt_fn(prior_error))
        code = _extract_code_block(response)
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(code)
        error = _py_compile_check(script_path)
        if error is None:
            print(f"  [codegen] {os.path.basename(script_path)} compiles OK "
                  f"({attempt + 1} attempt(s))")
            return True
        print(f"  [codegen] compile error, will retry:\n{error[:500]}")
        prior_error = error
    print(f"  [codegen] giving up on {os.path.basename(script_path)} after "
          f"{max_repair + 1} attempt(s) -- skipping this source")
    return False


# Generated scripts are instructed to wrap their whole run in a broad
# try/except and print the caught exception instead of dying with a bare
# traceback (see _QUALITY_API_SUMMARY / build_*_codegen_prompt). Match that
# convention here so a runtime failure can be pulled back out of the
# captured output and fed to Ollama as a repair prompt, the same way a
# compile error already is.
_FATAL_ERROR_RE = re.compile(r"^\[FATAL ERROR CAUGHT\]:\s*(.*)$")


def run_generated_script(script_path: str, target_size: str, out_dir: str, category: str,
                          min_doc_chars: int, log_path: str, extra_args: Optional[list] = None) -> dict:
    """Runs the generated script as a plain subprocess, streaming its output
    live to both the console and a log file (so a long run can be tailed),
    and parses the trailing RESULT_JSON line for a manifest entry. Also
    watches for the script's own "[FATAL ERROR CAUGHT]: ..." line (or a
    non-zero exit with no RESULT_JSON at all, i.e. it crashed before even
    reaching its own try/except) and returns that under "error" so callers
    can decide whether to trigger a repair-and-rerun."""
    cmd = [sys.executable, script_path,
           "--target-size", target_size, "--out-dir", out_dir,
           "--category", category, "--min-doc-chars", str(min_doc_chars)]
    if extra_args:
        cmd.extend(extra_args)
    print(f"  [run] {' '.join(cmd)}")
    print(f"  [run] logging to {log_path}")
    result_json = {"actual_bytes": 0, "docs": 0}
    fatal_error = None
    tail_lines = []  # last few lines, in case it crashes with no FATAL marker at all
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log_fh:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
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
        print(f"  [run] WARNING: script exited with code {proc.returncode} "
              f"(see {log_path}) -- using whatever RESULT_JSON it printed, if any")
        if fatal_error is None:
            # Crashed hard enough it never even hit its own try/except
            # (e.g. an ImportError before the loop starts) -- fall back to
            # the raw tail of output so there's still something to repair with.
            fatal_error = "".join(tail_lines[-15:]).strip() or f"process exited with code {proc.returncode}"
    elif fatal_error is None and result_json.get("docs", 0) == 0 and result_json.get("actual_bytes", 0) == 0:
        # Exited cleanly but wrote nothing at all -- often the same class of
        # bug (e.g. it caught its own error and printed RESULT_JSON with
        # zeros) without matching our marker regex exactly. Surface the tail
        # of output so a human/Ollama can see why nothing was produced.
        fatal_error = "".join(tail_lines[-15:]).strip() or None
    result_json["error"] = fatal_error
    return result_json


def generate_validate_and_run(prompt_fn, script_path: str, target_size: str, out_dir: str,
                               category: str, min_doc_chars: int, log_path: str,
                               extra_args: Optional[list] = None, max_repair: int = 2) -> dict:
    """Ties generate -> compile-validate -> run into one repair loop. A
    failure at EITHER stage (compile error, or a runtime "[FATAL ERROR
    CAUGHT]"/crash/zero-output from the actual run) is fed back into
    prompt_fn as prior_error and the script is regenerated, up to
    max_repair times total across both stages combined. This is what
    closes the gap that let a runtime bug like a bad load_dataset() kwarg
    slip through silently before: previously only compile errors triggered
    a rewrite, so a script that compiled fine but blew up at runtime just
    printed its FATAL line and was left alone."""
    prior_error = None
    result = {"actual_bytes": 0, "docs": 0, "error": None}
    for attempt in range(max_repair + 1):
        ok = generate_and_validate_script(
            prompt_fn, script_path, max_repair=0, prior_error=prior_error)
        if not ok:
            # generate_and_validate_script with max_repair=0 only fails on a
            # compile error with no more compile-retries left; treat that
            # compile error as this attempt's failure and loop again here.
            error = _py_compile_check(script_path)
            prior_error = error or prior_error
            print(f"  [codegen] attempt {attempt + 1}/{max_repair + 1} failed to compile, "
                  f"{'retrying' if attempt < max_repair else 'giving up'}...")
            continue

        result = run_generated_script(
            script_path, target_size, out_dir, category, min_doc_chars,
            log_path=log_path, extra_args=extra_args,
        )
        if result.get("error") is None:
            return result

        print(f"  [codegen] runtime error on attempt {attempt + 1}/{max_repair + 1}:\n"
              f"{result['error'][:500]}")
        if attempt < max_repair:
            print(f"  [codegen] regenerating {os.path.basename(script_path)} with this "
                  f"error fed back to Ollama...")
        prior_error = result["error"]

    print(f"  [codegen] giving up on {os.path.basename(script_path)} after "
          f"{max_repair + 1} generate+run attempt(s)")
    return result


# ---------------------------------------------------------------------------
# Top-level: one category, either mode
# ---------------------------------------------------------------------------

def run_category_public(category: str, budget_bytes: int, out_dir: str, mode: str,
                        min_doc_chars: int, discover_limit: int = 5,
                        max_candidates_to_try: int = 3) -> dict:
    scripts_dir = os.path.join(out_dir, "_generated_scripts")
    logs_dir = os.path.join(out_dir, "_logs")
    os.makedirs(scripts_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    candidates = discover_candidates(category, limit=discover_limit)
    print(f"[{category}] discovered {len(candidates)} candidate dataset(s): {candidates}")

    total_bytes, total_docs = 0, 0
    tried = 0
    for dataset_id in candidates:
        if total_bytes >= budget_bytes or tried >= max_candidates_to_try:
            break
        tried += 1
        configs = discover_hf_configs(dataset_id) or [None]
        config = configs[0]
        print(f"[{category}] sampling {dataset_id} (config={config})...")
        sample = sample_hf_dataset(dataset_id, config)
        if sample is None:
            print(f"[{category}] could not sample {dataset_id} -- skipping")
            continue

        safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", f"{category}_{dataset_id}")
        script_path = os.path.join(scripts_dir, f"{safe_name}.py")
        remaining = budget_bytes - total_bytes
        result = generate_validate_and_run(
            lambda err: build_hf_codegen_prompt(
                category, mode, dataset_id, config, sample["split"],
                sample["columns"], sample["rows"], prior_error=err),
            script_path, f"{remaining}B", out_dir, category, min_doc_chars,
            log_path=os.path.join(logs_dir, f"{safe_name}.log"),
        )
        total_bytes += result.get("actual_bytes", 0)
        total_docs += result.get("docs", 0)
        print(f"[{category}] after {dataset_id}: {total_bytes/1024**2:.1f} MB / "
              f"{budget_bytes/1024**2:.1f} MB, {total_docs} docs total")

    return {"target_bytes": budget_bytes, "actual_bytes": total_bytes, "docs": total_docs,
            "candidates_tried": tried}


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
        print(f"[{category}] reusing existing raw batch at {raw_path}")

    sample_rows = sample_raw_batch(raw_path)
    if not sample_rows:
        print(f"[{category}] raw batch is empty -- nothing to process")
        return {"target_bytes": budget_bytes, "actual_bytes": 0, "docs": 0}

    safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", f"{category}_web")
    script_path = os.path.join(scripts_dir, f"{safe_name}.py")
    result = generate_validate_and_run(
        lambda err: build_web_codegen_prompt(category, mode, raw_path, sample_rows, prior_error=err),
        script_path, f"{budget_bytes}B", out_dir, category, min_doc_chars,
        log_path=os.path.join(logs_dir, f"{safe_name}.log"),
        extra_args=["--raw-path", raw_path],
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
    args = parser.parse_args()

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
        print(f"\n=== [{category}] target {budget/1024**2:.1f} MB "
              f"({'public datasets' if args.public_only else 'live web crawl'}) ===")
        if args.public_only:
            manifest["categories"][category] = run_category_public(
                category, budget, args.out_dir, args.mode, args.min_doc_chars)
        else:
            manifest["categories"][category] = run_category_web(
                category, budget, args.out_dir, args.mode, args.min_doc_chars)

    manifest_path = os.path.join(args.out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    total_actual = sum(c["actual_bytes"] for c in manifest["categories"].values())
    print(f"\n=== Done. Total: {total_actual/1024**2:.2f} MB across "
          f"{len(manifest['categories'])} categories ===")
    print(f"Manifest written to {manifest_path}")
    print(f"Generated scripts kept under {os.path.join(args.out_dir, '_generated_scripts')} "
          f"-- re-runs skip regeneration if you rerun the same script directly.")


if __name__ == "__main__":
    main()
