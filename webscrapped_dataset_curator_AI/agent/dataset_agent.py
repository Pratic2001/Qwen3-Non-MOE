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
import json
import logging
import os
import sys
import time
from contextlib import AsyncExitStack
from typing import Optional

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dataset_agent")

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

sys.path.insert(0, os.path.dirname(__file__))
from quality import (
    ExactDedup, NearDedup, ShardWriter,
    passes_prose_quality_filter, passes_code_quality_filter,
)
from topics import TOPIC_SEEDS

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")

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

async def ollama_generate(prompt: str, system: Optional[str] = None, json_mode: bool = False) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    if system:
        payload["system"] = system
    if json_mode:
        payload["format"] = "json"
    log.debug(f"[ollama] -> model={OLLAMA_MODEL} prompt={prompt[:120]!r}...")
    start = time.time()
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(f"{OLLAMA_URL}/api/generate", json=payload)
        resp.raise_for_status()
        text = resp.json().get("response", "")
    log.debug(f"[ollama] <- ({time.time()-start:.1f}s) {text[:120]!r}...")
    return text


async def plan_queries(category: str, recent_topics: list, n: int = 8) -> list:
    """Ask the local model for fresh, specific search queries for a category,
    steering away from topics already covered so the corpus keeps expanding
    instead of circling the same few queries."""
    avoid = ", ".join(recent_topics[-30:]) if recent_topics else "(none yet)"
    system = (
        "You generate web search queries for building a language-model "
        "training corpus. Return ONLY a JSON object: "
        '{"queries": ["...", "..."]}. Queries must be short (3-8 words), '
        "specific, and diverse -- avoid vague single-word queries."
    )
    prompt = (
        f"Category: {category}\n"
        f"Seed topics for this category: {', '.join(TOPIC_SEEDS.get(category, [category]))}\n"
        f"Recently used queries (avoid repeating/near-duplicating these): {avoid}\n"
        f"Generate {n} new, specific search queries for this category."
    )
    try:
        raw = await ollama_generate(prompt, system=system, json_mode=True)
        data = json.loads(raw)
        queries = data.get("queries", [])
        queries = [q.strip() for q in queries if isinstance(q, str) and q.strip()][:n]
        log.info(f"[plan:{category}] planner produced {len(queries)} queries: {queries}")
        return queries
    except Exception as e:
        fallback = TOPIC_SEEDS.get(category, [category])[:n]
        log.warning(f"[plan:{category}] plan_queries failed ({e}), falling back to seed topics: {fallback}")
        return fallback


async def judge_quality(text: str, category: str) -> bool:
    """LLM-based quality gate, applied AFTER the cheap heuristic filters
    (which catch the obvious junk for free). Only invoked on documents that
    already passed the heuristics, to keep the number of LLM calls bounded.
    Returns True if the model says this is usable training data."""
    system = (
        "You judge whether a scraped web document is high-quality training "
        "data for a language model. Reject: boilerplate, ads/nav menus, "
        "listicles with no substance, spam, incoherent machine-translated "
        "text, or content that's mostly links/references with little prose. "
        f"Accept substantive {category} content. "
        'Respond ONLY with JSON: {"keep": true} or {"keep": false}.'
    )
    snippet = text[:3000]
    try:
        raw = await ollama_generate(snippet, system=system, json_mode=True)
        data = json.loads(raw)
        keep = bool(data.get("keep", False))
        log.debug(f"[judge:{category}] keep={keep}")
        return keep
    except Exception as e:
        # If the judge fails/times out, don't block the pipeline on it --
        # fall back to trusting the heuristic filters alone.
        log.warning(f"[judge:{category}] judge call failed ({e}), defaulting to keep=True")
        return True


async def extract_sft_pair(text: str, category: str) -> Optional[dict]:
    """For --mode sft: turn a scraped article into a (prompt, answer) pair
    by having the model pose a question the article answers, and produce a
    concise answer grounded in the text. Returns None if the article doesn't
    cleanly support this (e.g. pure narrative with no answerable question)."""
    system = (
        "You convert a source article into ONE high-quality instruction-"
        "tuning example. Ask a specific, well-posed question that the "
        "article answers, then answer it accurately using ONLY information "
        "in the article, in your own words (do not quote the article "
        'verbatim). Respond ONLY with JSON: '
        '{"prompt": "...", "answer": "..."} or {"prompt": null} if no good '
        "question/answer pair exists in this text."
    )
    try:
        raw = await ollama_generate(text[:6000], system=system, json_mode=True)
        data = json.loads(raw)
        if not data.get("prompt") or not data.get("answer"):
            log.debug(f"[sft:{category}] no usable Q/A pair in article")
            return None
        log.debug(f"[sft:{category}] extracted pair, prompt={data['prompt'][:80]!r}...")
        return {"prompt": data["prompt"].strip(), "thinking": "", "answer": data["answer"].strip()}
    except Exception as e:
        log.debug(f"[sft:{category}] extraction failed: {e}")
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
        result = await self.session.call_tool("extract_article", {"url": url})
        return _first_json(result)


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

async def run_category(scraper: ScraperClient, category: str, byte_budget: int, out_dir: str,
                        mode: str, min_doc_chars: int, use_llm_judge: bool):
    writer = ShardWriter(out_dir, category)
    exact_dedup = ExactDedup(persist_path=os.path.join(out_dir, category, ".seen_hashes"))
    near_dedup = NearDedup()

    seen_urls = set()
    used_queries = []
    n_filtered_quality = 0
    n_filtered_dup = 0
    n_llm_rejected = 0

    print(f"\n=== [{category}] target: {byte_budget / 1024**2:.1f} MB (live web scraping) ===")

    stall_rounds = 0
    while writer.total_bytes < byte_budget:
        queries = await plan_queries(category, used_queries, n=6)
        used_queries.extend(queries)
        progressed_this_round = False

        for query in queries:
            if writer.total_bytes >= byte_budget:
                break
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
                if not url:
                    continue
                if url in seen_urls:
                    log.debug(f"[skip:{category}] already seen: {url}")
                    continue
                seen_urls.add(url)

                try:
                    article = await scraper.extract(url)
                except Exception as e:
                    log.warning(f"[extract:{category}] failed for {url}: {e}")
                    continue
                if not article or article.get("error") or not article.get("text"):
                    reason = article.get("error") if article else "no response"
                    log.info(f"[extract:{category}] SKIP {url} -- {reason}")
                    continue

                text = article["text"]
                log.debug(f"[extract:{category}] OK {url} -- {len(text)} chars")

                if category == "code":
                    ok = passes_code_quality_filter(text, url, min_doc_chars)
                else:
                    ok = passes_prose_quality_filter(text, min_doc_chars)
                if not ok:
                    n_filtered_quality += 1
                    log.info(f"[filter:{category}] REJECT (quality heuristics) {url}")
                    continue

                if exact_dedup.is_duplicate(text) or near_dedup.is_near_duplicate(text):
                    n_filtered_dup += 1
                    log.info(f"[filter:{category}] REJECT (duplicate) {url}")
                    continue

                if use_llm_judge:
                    keep = await judge_quality(text, category)
                    if not keep:
                        n_llm_rejected += 1
                        log.info(f"[filter:{category}] REJECT (llm judge) {url}")
                        continue

                if mode == "sft":
                    pair = await extract_sft_pair(text, category)
                    if pair is None:
                        log.info(f"[filter:{category}] REJECT (no sft pair extractable) {url}")
                        continue
                    record = {**pair, "source": url, "category": category}
                else:
                    record = {"text": text, "source": url, "category": category}

                writer.write(record)
                log.info(f"[write:{category}] KEPT {url} ({len(text)} chars) "
                         f"-- total {writer.total_bytes/1024**2:.2f} MB / "
                         f"{byte_budget/1024**2:.1f} MB, {writer.total_docs} docs")
                progressed_this_round = True

                if writer.total_bytes >= byte_budget:
                    break

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
          f"filtered {n_filtered_quality} low-quality + {n_filtered_dup} duplicate + "
          f"{n_llm_rejected} llm-rejected")
    writer.close()
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
    target_bytes = _parse_size(args.target_size)
    categories = args.categories.split(",") if args.categories else list(DEFAULT_MIX.keys())
    mix = {c: DEFAULT_MIX.get(c, 1.0 / len(categories)) for c in categories}
    total_frac = sum(mix.values())
    mix = {c: f / total_frac for c, f in mix.items()}

    os.makedirs(args.out_dir, exist_ok=True)
    server_path = os.environ.get("MCP_SERVER_PATH") or _find_server_path()
    log.info(f"Launching MCP server: {sys.executable} {server_path}")
    server_params = StdioServerParameters(command=sys.executable, args=[server_path])

    manifest_path = os.path.join(args.out_dir, "manifest.json")
    manifest = {"target_bytes": target_bytes, "mix": mix, "categories": {}, "mode": args.mode}

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

            for category, frac in mix.items():
                budget = int(target_bytes * frac)
                if budget <= 0:
                    continue
                actual_bytes, docs = await run_category(
                    scraper, category, budget, args.out_dir, args.mode,
                    args.min_doc_chars, use_llm_judge=not args.no_llm_judge,
                )
                manifest["categories"][category] = {
                    "target_bytes": budget, "actual_bytes": actual_bytes, "docs": docs,
                }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    total_actual = sum(c["actual_bytes"] for c in manifest["categories"].values())
    print(f"\n=== Done. Total: {total_actual / 1024**2:.2f} MB across "
          f"{len(manifest['categories'])} categories ===")
    print(f"Manifest written to {manifest_path}")


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
    parser.add_argument("--no-llm-judge", action="store_true",
                         help="Skip the Ollama quality-judging pass, keep only heuristic filters (faster)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                         help="DEBUG shows every Ollama call, skipped/seen URLs, and full text lengths; "
                              "INFO (default) shows every query, hit list, extract skip reason, filter "
                              "rejection reason, and write, without the very-verbose stuff.")
    args = parser.parse_args()
    logging.getLogger("dataset_agent").setLevel(args.log_level)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
