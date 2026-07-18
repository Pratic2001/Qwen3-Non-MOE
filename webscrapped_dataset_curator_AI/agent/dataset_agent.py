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
import urllib.parse
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
    ExactDedup, NearDedup, RunState, ShardWriter,
    passes_prose_quality_filter, passes_code_quality_filter,
    passes_transcript_quality_filter,
)
from topics import TOPIC_SEEDS, HUB_SEARCH_KEYWORDS
import public_sources

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

def _quality_filter_for(content_type: Optional[str], text: str, url: str, min_doc_chars: int) -> bool:
    """Dispatch to the right quality bar for what extract_content actually
    returned. content_type comes back from the server (html/pdf/docx/pptx/
    xlsx/csv/image/video/audio/text); category alone isn't enough to know
    this anymore since a "code" category doc might legitimately be a PDF
    spec or a transcript of a talk, not just a source file."""
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

    ok = _quality_filter_for(content_type, text, url, min_doc_chars)
    if not ok:
        counters["filtered_quality"] += 1
        log.info(f"[filter:{category}] REJECT (quality heuristics) {url}")
        return False

    if use_llm_judge:
        keep = await judge_quality(text, category)
        if not keep:
            counters["llm_rejected"] += 1
            log.info(f"[filter:{category}] REJECT (llm judge) {url}")
            return False

    if mode == "sft":
        # If the source already carries its own prompt/answer labels (a HF
        # instruction dataset, a Kaggle Q&A CSV, ...), trust those over an
        # LLM guess -- they're the dataset author's ground truth, not a
        # hallucinated question. Only fall back to the built-in-AI
        # extraction (Ollama inventing a Q/A pair from raw prose) when the
        # row genuinely doesn't carry one.
        extra = article.get("extra") or {}
        given_prompt, given_answer = extra.get("prompt"), extra.get("answer")
        if given_prompt and given_answer:
            pair = {"prompt": given_prompt, "thinking": "", "answer": given_answer}
        else:
            pair = await extract_sft_pair(text, category)
        if pair is None:
            log.info(f"[filter:{category}] REJECT (no sft pair extractable) {url}")
            return False
        record = {**pair, "source": url, "category": category, "content_type": content_type}
    else:
        record = {"text": text, "source": url, "category": category, "content_type": content_type}

    async with write_lock:
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

    `public_cfg` shape: {
        "sources": {"huggingface", "kaggle"},       # which backends are on
        "hf_datasets": {category: [dataset_id, ...]},   # explicit ids, optional
        "kaggle_datasets": {category: [ref, ...]},      # explicit refs, optional
        "max_rows_per_dataset": int,
        "discover_limit": int,                          # datasets to auto-discover per category
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
    """
    sem = asyncio.Semaphore(concurrency)

    async def bounded(article: dict) -> bool:
        async with sem:
            return await _process_public_row(article, category, mode, min_doc_chars,
                                               use_llm_judge, exact_dedup, near_dedup,
                                               writer, byte_budget, write_lock, counters)

    max_rows = public_cfg.get("max_rows_per_dataset", 500)
    discover_limit = public_cfg.get("discover_limit", 3)
    search_keywords = HUB_SEARCH_KEYWORDS.get(category, [category])

    def _lookup(cat_map: dict) -> Optional[list]:
        return cat_map.get(category) or cat_map.get("__all__")

    async def _discover(discover_fn, label: str) -> list:
        """Try each short keyword in turn, merging/deduping hits (in
        discovery order) until discover_limit distinct ids/refs are found
        or every keyword's been tried. Logs what each individual keyword
        turned up so a bad keyword is visible in the log instead of just
        silently contributing nothing."""
        found: list = []
        for kw in search_keywords:
            if len(found) >= discover_limit:
                break
            hits = await asyncio.to_thread(discover_fn, kw, discover_limit - len(found))
            log.info(f"[{label}:{category}] keyword {kw!r} -> {hits}")
            for h in hits:
                if h not in found:
                    found.append(h)
        return found[:discover_limit]

    if "huggingface" in public_cfg.get("sources", set()):
        hf_ids = _lookup(public_cfg.get("hf_datasets", {}))
        if not hf_ids:
            hf_ids = await _discover(public_sources.discover_hf_datasets, "public-hf")
            log.info(f"[public-hf:{category}] discovered datasets: {hf_ids}")
        for ds_id in hf_ids:
            if writer.total_bytes >= byte_budget:
                break
            log.info(f"[public-hf:{category}] streaming {ds_id} (max {max_rows} rows)")
            batch, tasks = [], []
            for row in await asyncio.to_thread(
                    lambda ds=ds_id: list(public_sources.stream_hf_dataset(ds, max_rows=max_rows))):
                batch.append(row)
            for row in batch:
                if writer.total_bytes >= byte_budget:
                    break
                tasks.append(asyncio.create_task(bounded(row)))
                if len(tasks) >= concurrency * 2:
                    await asyncio.gather(*tasks, return_exceptions=True)
                    tasks = []
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    if "kaggle" in public_cfg.get("sources", set()):
        kg_refs = _lookup(public_cfg.get("kaggle_datasets", {}))
        if not kg_refs:
            kg_refs = await _discover(public_sources.discover_kaggle_datasets, "public-kaggle")
            log.info(f"[public-kaggle:{category}] discovered datasets: {kg_refs}")
        for ref in kg_refs:
            if writer.total_bytes >= byte_budget:
                break
            log.info(f"[public-kaggle:{category}] downloading {ref} (max {max_rows} rows)")
            rows = await asyncio.to_thread(
                lambda r=ref: list(public_sources.fetch_kaggle_dataset_rows(r, max_rows=max_rows)))
            tasks = []
            for row in rows:
                if writer.total_bytes >= byte_budget:
                    break
                tasks.append(asyncio.create_task(bounded(row)))
                if len(tasks) >= concurrency * 2:
                    await asyncio.gather(*tasks, return_exceptions=True)
                    tasks = []
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

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
        "missing_dependency": 0, "other_extract_fail": 0,
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
                  f"{counters['filtered_dup']} duplicate + {counters['llm_rejected']} llm-rejected")
            writer.close()
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
          f"{counters['llm_rejected']} llm-rejected\n"
          f"[{category}] fetch failures: {counters['robots_blocked']} robots.txt/domain-blocked, "
          f"{counters['http_error']} HTTP error (403/etc.), {counters['video_skipped']} video/duration-skipped, "
          f"{counters['missing_dependency']} missing optional dependency, "
          f"{counters['other_extract_fail']} other extraction failures")
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

    manifest_path = os.path.join(args.out_dir, "manifest.json")
    manifest = {"target_bytes": target_bytes, "mix": mix, "categories": {}, "mode": args.mode}

    public_cfg = None
    public_source_set = {s.strip().lower() for s in (args.public_sources or "").split(",") if s.strip()}
    if public_source_set:
        public_cfg = {
            "sources": public_source_set,
            "hf_datasets": _parse_category_map(args.hf_datasets),
            "kaggle_datasets": _parse_category_map(args.kaggle_datasets),
            "max_rows_per_dataset": args.public_max_rows,
            "discover_limit": args.public_discover_limit,
            "public_only": args.public_only,
        }
        log.info(f"Public dataset sources enabled: {public_source_set} "
                 f"(public_only={args.public_only})")

    if public_cfg and public_cfg.get("public_only"):
        # No live scraping requested at all -- skip spinning up the MCP
        # subprocess entirely, since ScraperClient/scraper is never touched
        # on the public-sources-only return path in run_category.
        log.info("public_only=True: skipping MCP scraper subprocess launch.")
        for category, frac in mix.items():
            budget = int(target_bytes * frac)
            if budget <= 0:
                continue
            actual_bytes, docs = await run_category(
                None, category, budget, args.out_dir, args.mode,
                args.min_doc_chars, use_llm_judge=not args.no_llm_judge,
                concurrency=args.concurrency, public_cfg=public_cfg,
                deep_crawl_per_domain=0,  # no MCP session in the public-only path -- N/A
            )
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

                for category, frac in mix.items():
                    budget = int(target_bytes * frac)
                    if budget <= 0:
                        continue
                    actual_bytes, docs = await run_category(
                        scraper, category, budget, args.out_dir, args.mode,
                        args.min_doc_chars, use_llm_judge=not args.no_llm_judge,
                        concurrency=args.concurrency, public_cfg=public_cfg,
                        deep_crawl_per_domain=args.deep_crawl_per_domain,
                        deep_crawl_max_pages=args.deep_crawl_max_pages,
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
    parser.add_argument("--no-llm-judge", action="store_true",
                         help="Skip the Ollama quality-judging pass, keep only heuristic filters (faster)")
    parser.add_argument("--concurrency", type=int, default=5,
                         help="Max concurrent extract+filter tasks per category round (default 5). "
                              "Raise for I/O-bound HTML/PDF-heavy runs; keep low if ASR transcription "
                              "is in play (each faster-whisper call is CPU/GPU-heavy) or the LLM judge "
                              "is on (concurrent calls just queue behind a single local Ollama model).")
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
        "instead of) live web scraping. Rows go through the exact same quality filters, "
        "LLM judge, and (in --mode sft) the same built-in-AI Q/A extraction as scraped pages.")
    public.add_argument("--public-sources", default=None,
                         help="Comma list of public hub backends to pull from: huggingface,kaggle. "
                              "Unset (default) disables this feature entirely, matching prior behavior.")
    public.add_argument("--hf-datasets", default=None,
                         help="Explicit Hugging Face dataset ids to stream. Syntax: "
                              "'category=id1,id2;category2=id3', or a bare 'id1,id2' to apply to "
                              "every category. If omitted, ids are auto-discovered per category via "
                              "the Hub search API using that category's topic seeds as the query.")
    public.add_argument("--kaggle-datasets", default=None,
                         help="Explicit Kaggle dataset refs (owner/dataset-slug), same syntax as "
                              "--hf-datasets. Requires KAGGLE_USERNAME/KAGGLE_KEY env vars (or "
                              "~/.kaggle/kaggle.json). If omitted, refs are auto-discovered per "
                              "category via Kaggle's search API.")
    public.add_argument("--public-max-rows", type=int, default=500,
                         help="Max rows to pull per dataset (default 500). Still passes through "
                              "every quality filter / dedup check / LLM judge, so this is a ceiling "
                              "on how much is *considered*, not how much is written.")
    public.add_argument("--public-discover-limit", type=int, default=3,
                         help="When dataset ids/refs aren't given explicitly, how many datasets to "
                              "auto-discover per category via hub search (default 3).")
    public.add_argument("--public-only", action="store_true",
                         help="Skip live web search/scraping entirely -- fill each category's "
                              "budget purely from the configured public dataset hubs, and don't "
                              "even launch the MCP scraper subprocess. Useful when you just want a "
                              "fast Kaggle/HF-only top-up run.")

    args = parser.parse_args()
    logging.getLogger("dataset_agent").setLevel(args.log_level)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
