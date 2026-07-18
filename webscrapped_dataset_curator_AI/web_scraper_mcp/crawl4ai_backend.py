"""
crawl4ai_backend.py

Optional, higher-fidelity HTML fetch + extraction backend built on
crawl4ai (https://github.com/unclecode/crawl4ai), a Playwright-driven
crawler purpose-built for LLM training-data pipelines. Two things this
buys over the plain httpx + trafilatura path in server.py/extractors.py:

1. **JS rendering.** httpx just GETs raw bytes -- any site that renders
   its article body client-side (a lot of modern blogs/docs/news SPAs)
   comes back nearly empty no matter how good trafilatura is, because
   there's no content in the initial HTML to extract from. crawl4ai runs
   a real (headless) browser, so it sees the DOM after JS has run.
2. **Multi-page crawling in one call.** `deep_crawl` follows in-page links
   outward from a seed URL (BFS, optionally keyword-scored) and extracts
   every page it visits in a single tool call -- one docs site or blog
   index becomes dozens of documents instead of needing one web_search +
   one extract_content round-trip per page.

Design principles (matching extractors.py):
- crawl4ai itself is imported LAZILY inside each function, so the MCP
  server still boots, and the httpx+trafilatura path still works, if
  crawl4ai (or its Playwright browser) isn't installed. Every function
  here returns the same {title, text, author, date, content_type, url,
  error, extra} shape as extractors.py's extractors, so callers never
  need to know which backend produced a given result.
- A single AsyncWebCrawler (i.e. one headless browser process) is started
  lazily on first use and reused across calls -- starting a fresh browser
  per request would dominate wall-clock time on anything but a handful of
  URLs. Call `shutdown()` once at process exit to close it cleanly
  (server.py registers this via atexit).
- Nothing here does its own robots.txt/rate-limiting bookkeeping beyond
  what crawl4ai's `check_robots_txt` config flag gives per-request --
  server.py still applies its own domain allow/deny list and per-host
  rate limiter around every call into this module, same as it does for
  the httpx path, so operator-configured scraping policy is consistent
  regardless of which backend actually fetched a given page.

Not exercised against live internet traffic in this sandbox (no general
network egress here) -- syntax/import-checked only. Run `healthcheck()`
in server.py (reports `crawl4ai_html` under `formats`) and, in your own
environment, `crawl4ai-doctor` after `pip install crawl4ai && crawl4ai-setup`
to confirm the Playwright browser install is actually working before a
real run.
"""

from __future__ import annotations

import asyncio
import urllib.parse
from typing import Optional

# ---------------------------------------------------------------------------
# Shared result shape (mirrors extractors._result so callers never branch on
# which module produced a given dict)
# ---------------------------------------------------------------------------

def _result(text="", title=None, author=None, date=None, content_type="html",
            url="", error=None, extra=None) -> dict:
    return {
        "title": title, "text": (text or "").strip(), "author": author,
        "date": date, "content_type": content_type, "url": url,
        "error": error, "extra": extra or {},
    }


# Minimum extracted-text length below which we treat a crawl4ai result as
# "didn't really get the article" (nav-only shell, cookie wall, etc.) rather
# than a genuine success -- same threshold extractors.extract_html uses.
_MIN_TEXT_CHARS = 200

# ---------------------------------------------------------------------------
# Shared browser instance
# ---------------------------------------------------------------------------

_crawler = None
_crawler_lock: Optional[asyncio.Lock] = None


def _lock() -> asyncio.Lock:
    global _crawler_lock
    if _crawler_lock is None:
        _crawler_lock = asyncio.Lock()
    return _crawler_lock


async def _get_crawler():
    """Lazily start (once) and return the shared AsyncWebCrawler. Raises
    ImportError if crawl4ai isn't installed -- callers catch this and
    return a normal {"error": ...} result rather than letting it propagate
    as a raw exception."""
    global _crawler
    async with _lock():
        if _crawler is None:
            from crawl4ai import AsyncWebCrawler, BrowserConfig
            import net_utils
            browser_cfg = BrowserConfig(
                headless=True,
                verbose=False,
                user_agent=net_utils.user_agents.get(),
            )
            crawler = AsyncWebCrawler(config=browser_cfg)
            await crawler.start()
            _crawler = crawler
    return _crawler


async def shutdown():
    """Close the shared browser if it was ever started. Safe to call even
    if it never was (no-op then). server.py calls this once at process
    exit via atexit -- crawl4ai's Playwright subprocess otherwise tends to
    linger past the parent process exiting."""
    global _crawler
    if _crawler is not None:
        try:
            await _crawler.close()
        except Exception:
            pass
        _crawler = None


def _extract_markdown(result) -> str:
    """crawl4ai's `result.markdown` is a MarkdownGenerationResult with both
    a raw and a "fit" (boilerplate-pruned) variant when a content filter is
    configured. Prefer fit_markdown -- that's the whole point of running
    PruningContentFilter -- falling back to raw_markdown/str() for older
    crawl4ai versions that return a plain string instead."""
    md = getattr(result, "markdown", None)
    if md is None:
        return ""
    text = getattr(md, "fit_markdown", None)
    if not text:
        text = getattr(md, "raw_markdown", None)
    if not text and isinstance(md, str):
        text = md
    return (text or "").strip()


def _make_markdown_generator():
    from crawl4ai.content_filter_strategy import PruningContentFilter
    from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
    return DefaultMarkdownGenerator(
        content_filter=PruningContentFilter(threshold=0.45, threshold_type="dynamic"),
        options={"ignore_links": True, "ignore_images": True, "escape_html": False},
    )


# ---------------------------------------------------------------------------
# Single-page fetch + extract
# ---------------------------------------------------------------------------

async def fetch_and_extract(url: str, timeout: float = 30.0,
                             wait_for: Optional[str] = None,
                             css_selector: Optional[str] = None) -> dict:
    """Fetch `url` with a real headless browser and extract clean, LLM-
    ready markdown, pruned of nav/ad/boilerplate. Drop-in replacement for
    extractors.extract_html for any HTML URL -- same return shape.

    wait_for: optional crawl4ai wait condition (CSS selector prefixed with
    `css:`, or a JS expression) for pages that render content after an XHR
    -- leave None for ordinary pages.
    css_selector: optional CSS selector to scope extraction to (e.g. the
    main article container) if the page has a lot of surrounding chrome
    the pruning filter doesn't fully strip.
    """
    try:
        from crawl4ai import CrawlerRunConfig, CacheMode
    except ImportError as e:
        return _result(url=url, error=f"crawl4ai not installed (pip install crawl4ai && "
                                       f"crawl4ai-setup): {e}")

    try:
        run_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            markdown_generator=_make_markdown_generator(),
            page_timeout=int(timeout * 1000),
            wait_for=wait_for,
            css_selector=css_selector,
            magic=True,           # best-effort auto-handling of cookie banners/overlays
            simulate_user=True,   # small randomized mouse/scroll jitter, reduces bot-detection blocks
            check_robots_txt=True,
            verbose=False,
        )
        crawler = await _get_crawler()
        result = await crawler.arun(url=url, config=run_config)
    except ImportError as e:
        return _result(url=url, error=f"crawl4ai not installed: {e}")
    except Exception as e:
        return _result(url=url, error=f"crawl4ai fetch failed: {type(e).__name__}: {e}")

    if not result or not getattr(result, "success", False):
        err = getattr(result, "error_message", None) if result else None
        return _result(url=url, error=f"crawl4ai: {err or 'fetch unsuccessful'}")

    text = _extract_markdown(result)
    if len(text) < _MIN_TEXT_CHARS:
        return _result(url=url, error=f"crawl4ai: extracted content too short "
                                       f"({len(text)} chars) -- likely a login/cookie wall "
                                       f"or a page with no real article body")

    meta = getattr(result, "metadata", None) or {}
    title = meta.get("title")
    author = meta.get("author")
    date = meta.get("article:published_time") or meta.get("date") or meta.get("datePublished")

    return _result(
        text=text, title=title, author=author, date=date, url=url,
        extra={"backend": "crawl4ai", "status_code": getattr(result, "status_code", None)},
    )


# ---------------------------------------------------------------------------
# Multi-page deep crawl from a seed URL
# ---------------------------------------------------------------------------

def _host(url: str) -> str:
    h = urllib.parse.urlparse(url).netloc.lower()
    return h[4:] if h.startswith("www.") else h


async def deep_crawl(seed_url: str, max_pages: int = 20, max_depth: int = 2,
                      keywords: Optional[list[str]] = None,
                      same_domain_only: bool = True) -> list:
    """BFS-crawl outward from `seed_url`, extracting every page visited (up
    to max_pages, up to max_depth link-hops away). Returns a list of dicts
    in the same shape as fetch_and_extract/extract_html -- ready to feed
    straight into the same filtering/dedup/writing pipeline as any other
    URL, one entry per successfully extracted page.

    keywords: if given, discovered links are prioritized by relevance to
    these terms (crawl4ai's KeywordRelevanceScorer) rather than plain BFS
    order -- useful when a domain has far more pages than max_pages allows
    and you want the on-topic ones visited first.
    same_domain_only: restrict crawling to seed_url's own host. Recommended
    on almost always -- without it a single seed can wander arbitrarily far
    off-topic across linked domains.

    Returns [{"error": "...", "url": seed_url}] (single-item list, same
    convention web_search uses) if crawl4ai isn't installed or the crawl
    fails outright; individual page failures within a successful crawl are
    just omitted from the result list rather than erroring the whole call.
    """
    try:
        from crawl4ai import CrawlerRunConfig, CacheMode
        from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
        from crawl4ai.deep_crawling.filters import FilterChain, DomainFilter
    except ImportError as e:
        return [{"error": f"crawl4ai (deep-crawl) not installed (pip install crawl4ai "
                           f"&& crawl4ai-setup): {e}", "url": seed_url}]

    filters = []
    if same_domain_only:
        filters.append(DomainFilter(allowed_domains=[_host(seed_url)]))
    filter_chain = FilterChain(filters) if filters else None

    scorer = None
    if keywords:
        try:
            from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer
            scorer = KeywordRelevanceScorer(keywords=keywords, weight=1.0)
        except ImportError:
            scorer = None  # older crawl4ai without this scorer -- fall back to plain BFS order

    try:
        strategy = BFSDeepCrawlStrategy(
            max_depth=max_depth,
            max_pages=max_pages,
            include_external=not same_domain_only,
            filter_chain=filter_chain,
            url_scorer=scorer,
        )
        run_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            deep_crawl_strategy=strategy,
            markdown_generator=_make_markdown_generator(),
            check_robots_txt=True,
            stream=False,
            verbose=False,
        )
        crawler = await _get_crawler()
        results = await crawler.arun(url=seed_url, config=run_config)
    except ImportError as e:
        return [{"error": f"crawl4ai not installed: {e}", "url": seed_url}]
    except Exception as e:
        return [{"error": f"crawl4ai deep_crawl failed: {type(e).__name__}: {e}", "url": seed_url}]

    if results is None:
        return [{"error": "crawl4ai deep_crawl returned no results", "url": seed_url}]
    if not isinstance(results, list):
        results = [results]

    out = []
    for r in results:
        if not getattr(r, "success", False):
            continue
        text = _extract_markdown(r)
        if len(text) < _MIN_TEXT_CHARS:
            continue
        meta = getattr(r, "metadata", None) or {}
        page_url = getattr(r, "url", None) or seed_url
        out.append(_result(
            text=text, title=meta.get("title"), author=meta.get("author"),
            date=meta.get("article:published_time") or meta.get("date"),
            url=page_url,
            extra={"backend": "crawl4ai", "depth": meta.get("depth"),
                   "status_code": getattr(r, "status_code", None)},
        ))
    return out
