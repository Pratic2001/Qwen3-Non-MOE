#!/usr/bin/env python3
"""
web_scraper_mcp/server.py

An MCP server that gives an agent three tools for turning "I need more data
about X" into clean text:

    web_search(query, max_results)  -> list of {title, url, snippet}
    fetch_page(url)                 -> raw HTML (truncated) + status
    extract_article(url)            -> cleaned main-content text via
                                        trafilatura (falls back to
                                        readability-lxml, then a raw-text
                                        strip) + basic metadata

Design notes:
- Search uses DuckDuckGo's HTML endpoint (via the `ddgs` package) so no API
  key is required. Swap in Bing/Serper/Tavily here if you have a key and
  want higher reliability at scale -- the tool signature doesn't change.
- Everything is read-only / GET-only. This server does not click buttons,
  submit forms, or execute JS, so it can't be used to take actions on the
  scraped sites -- just to read public pages.
- Respects robots.txt via a small in-process cache, and rate-limits by host
  so the agent can't accidentally hammer one domain.

Run with:
    python server.py                # stdio transport, for MCP-aware clients
"""

import asyncio
import os
import sys
import time
import urllib.parse
import urllib.robotparser
from typing import Optional

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "web-scraper",
    host=os.environ.get("MCP_HOST", "127.0.0.1"),
    port=int(os.environ.get("MCP_PORT", "8000")),
)

USER_AGENT = "Mozilla/5.0 (compatible; DatasetResearchBot/1.0; +https://example.com/bot)"

_robots_cache: dict = {}
_last_request_time: dict = {}
MIN_SECONDS_BETWEEN_REQUESTS_PER_HOST = 2.0


def _host(url: str) -> str:
    return urllib.parse.urlparse(url).netloc


def _robots_allowed(url: str) -> bool:
    host = _host(url)
    if host not in _robots_cache:
        rp = urllib.robotparser.RobotFileParser()
        robots_url = f"{urllib.parse.urlparse(url).scheme}://{host}/robots.txt"
        try:
            resp = httpx.get(robots_url, timeout=5, headers={"User-Agent": USER_AGENT})
            rp.parse(resp.text.splitlines())
        except Exception:
            # If robots.txt can't be fetched, default to allow (matches
            # standard crawler behavior) but still respect rate limiting.
            _robots_cache[host] = None
            return True
        _robots_cache[host] = rp
    rp = _robots_cache[host]
    if rp is None:
        return True
    return rp.can_fetch(USER_AGENT, url)


async def _rate_limit(url: str):
    host = _host(url)
    now = time.monotonic()
    last = _last_request_time.get(host, 0)
    wait = MIN_SECONDS_BETWEEN_REQUESTS_PER_HOST - (now - last)
    if wait > 0:
        await asyncio.sleep(wait)
    _last_request_time[host] = time.monotonic()


@mcp.tool()
def web_search(query: str, max_results: int = 10) -> list[dict]:
    """Search the public web and return title/url/snippet hits.

    Use this to discover candidate pages for a topic before fetching them.
    Keep queries short and specific (3-8 words), the same way you'd type
    into a search box.

    IMPORTANT: never fails silently. If every backend errors out or
    returns zero hits, this returns a single-item list containing
    {"error": "...", "query": "..."} instead of an empty list or a raised
    exception -- so the caller can distinguish "genuinely no results" from
    "the search backend is broken/blocked."
    """
    from ddgs import DDGS  # imported lazily so the server still boots if the
                            # dependency isn't installed yet, with a clear error
    try:
        from ddgs.exceptions import DDGSException
    except ImportError:
        DDGSException = Exception

    # "duckduckgo" is frequently hard-blocked outright on cloud/datacenter
    # IPs (returns "No results found" regardless of query -- confirmed via
    # diagnose_search.py). "auto" fans out to every engine at once, which
    # is slow and more likely to trip rate limits under sustained load.
    # Default to a short list of backends that have tested clean (no ad
    # links mixed into results) and fall through in order. Override with
    # DDGS_BACKENDS="brave,yandex" (comma-separated, tried in order) if
    # your network blocks a different subset.
    backends_env = os.environ.get("DDGS_BACKENDS")
    backends = [b.strip() for b in backends_env.split(",")] if backends_env else \
        ["brave", "yandex", "bing", "auto"]

    last_err = None
    for backend in backends:
        for attempt in range(2):
            try:
                results = []
                with DDGS() as ddgs:
                    for r in ddgs.text(query, max_results=max_results, backend=backend):
                        url = r.get("href", "")
                        # Bing mixes sponsored redirect links (bing.com/aclick)
                        # into organic results -- drop those, they're ads,
                        # not content, and the redirect won't extract cleanly.
                        if "bing.com/aclick" in url:
                            continue
                        results.append({"title": r.get("title", ""), "url": url,
                                         "snippet": r.get("body", "")})
                if results:
                    print(f"[web_search] {query!r} via backend={backend} -> {len(results)} hits",
                          file=sys.stderr, flush=True)
                    return results
                last_err = f"backend={backend} returned zero results (likely rate-limited/blocked)"
            except DDGSException as e:
                last_err = f"backend={backend}: {type(e).__name__}: {e}"
            except Exception as e:
                last_err = f"backend={backend}: {type(e).__name__}: {e}"
            time.sleep(1)  # brief backoff before retrying same backend once

    print(f"[web_search] {query!r} FAILED all backends -- {last_err}", file=sys.stderr, flush=True)
    return [{"error": last_err or "unknown search failure", "query": query}]


@mcp.tool()
async def fetch_page(url: str, max_chars: int = 200_000) -> dict:
    """Fetch the raw content at a URL (GET only). Honors robots.txt and a
    per-host rate limit. Returns {status, content_type, html, error}."""
    if not _robots_allowed(url):
        return {"status": None, "content_type": None, "html": "", "error": "disallowed by robots.txt"}

    await _rate_limit(url)
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=15,
                                      headers={"User-Agent": USER_AGENT}) as client:
            resp = await client.get(url)
        content_type = resp.headers.get("content-type", "")
        if "text" not in content_type and "html" not in content_type:
            return {"status": resp.status_code, "content_type": content_type,
                     "html": "", "error": "non-text content-type, skipped"}
        return {"status": resp.status_code, "content_type": content_type,
                 "html": resp.text[:max_chars], "error": None}
    except Exception as e:
        return {"status": None, "content_type": None, "html": "", "error": str(e)}


@mcp.tool()
async def extract_article(url: str) -> dict:
    """Fetch a URL and extract clean main-content text (nav/ads/boilerplate
    stripped), using trafilatura with a readability-lxml fallback.

    Returns {title, text, author, date, url, error}. `text` is empty and
    `error` is set if extraction fails or robots.txt disallows the fetch.
    """
    page = await fetch_page(url)
    if page["error"] or not page["html"]:
        return {"title": None, "text": "", "author": None, "date": None,
                "url": url, "error": page["error"] or "empty page"}

    html = page["html"]

    try:
        import trafilatura
        text = trafilatura.extract(html, url=url, favor_recall=True,
                                    include_comments=False, include_tables=True)
        meta = trafilatura.extract_metadata(html)
        title = meta.title if meta else None
        author = meta.author if meta else None
        date = meta.date if meta else None
        if text and len(text.strip()) > 200:
            return {"title": title, "text": text.strip(), "author": author,
                    "date": date, "url": url, "error": None}
    except Exception:
        pass

    # Fallback: readability-lxml gets the main content div even when
    # trafilatura's heuristics miss (common on forum / doc-site layouts).
    try:
        from readability import Document
        import re as _re
        doc = Document(html)
        title = doc.title()
        summary_html = doc.summary()
        text = _re.sub(r"<[^>]+>", " ", summary_html)
        text = _re.sub(r"\s+", " ", text).strip()
        if text and len(text) > 200:
            return {"title": title, "text": text, "author": None, "date": None,
                     "url": url, "error": None}
    except Exception:
        pass

    return {"title": None, "text": "", "author": None, "date": None,
            "url": url, "error": "extraction failed"}


@mcp.tool()
def healthcheck() -> dict:
    """Cheap connectivity/config probe -- call this from an orchestrator
    (n8n Cron job, OpenClaw `probe`, a monitoring workflow, etc.) before
    kicking off a real scrape, or on a schedule to catch a broken search
    backend before it silently eats a whole run. Does NOT do a full search;
    just reports whether the search dependency imports and which backend
    is configured, and whether robots.txt fetching works against a known-
    reachable host.
    """
    import importlib
    report = {"ddgs_importable": False, "ddgs_backend": os.environ.get("DDGS_BACKEND", "duckduckgo"),
               "robots_check_ok": False, "error": None}
    try:
        importlib.import_module("ddgs")
        report["ddgs_importable"] = True
    except Exception as e:
        report["error"] = f"ddgs import failed: {e}"
        return report
    try:
        report["robots_check_ok"] = _robots_allowed("https://example.com/")
    except Exception as e:
        report["error"] = f"robots check failed: {e}"
    return report


if __name__ == "__main__":
    # Default stdio keeps `dataset_agent.py`'s stdio_client working unchanged.
    # Set MCP_TRANSPORT=streamable-http (plus optionally MCP_HOST/MCP_PORT)
    # to expose this as an HTTP endpoint instead, e.g. for n8n's MCP Client
    # Tool node or OpenClaw's `openclaw mcp` bridge, which both speak HTTP/SSE
    # rather than spawning the process themselves over stdio.
    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    mcp.run(transport=transport)
