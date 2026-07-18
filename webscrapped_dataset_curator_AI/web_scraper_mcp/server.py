#!/usr/bin/env python3
"""
web_scraper_mcp/server.py

An MCP server that gives an agent tools for turning "I need more data about
X" into clean text -- from HTML pages, PDFs, Office docs, images, and
video/audio.

Tools:
    web_search(query, max_results)  -> list of {title, url, snippet}
    fetch_page(url)                 -> raw HTML text (truncated) + status
    fetch_binary(url)               -> base64 bytes + content-type + status
                                        (for PDFs, images, docx, etc.)
    extract_content(url)            -> format-agnostic: detects HTML / PDF /
                                        DOCX / PPTX / XLSX / CSV / image /
                                        video / audio and returns clean text
                                        + metadata. HTML pages go through
                                        crawl4ai_backend (real headless-
                                        browser fetch + pruned markdown)
                                        first, falling back to the legacy
                                        httpx + trafilatura/readability path
                                        in web_scraper_mcp.extractors --
                                        see SCRAPER_HTML_BACKEND below. This
                                        is the tool most agents should call.
    deep_crawl(seed_url, ...)       -> crawl4ai-powered: BFS-crawl outward
                                        from a seed URL, extracting every
                                        page visited (up to max_pages) in
                                        ONE call -- much higher yield per
                                        round than one web_search hit ->
                                        one extract_content call, for
                                        domains you already know are
                                        relevant (docs sites, blogs, wikis).
    extract_article(url)            -> HTML-only alias kept for backward
                                        compatibility with existing callers;
                                        internally now just extract_content
                                        restricted to the html/text path.
    transcribe_media(url)           -> dedicated video/audio -> transcript
                                        tool (captions preferred, local ASR
                                        fallback via faster-whisper).
    healthcheck()                   -> connectivity/config probe.

Design notes:
- Search uses DuckDuckGo's HTML endpoint (via `ddgs`), no API key required.
- Everything is read-only / GET-only: no clicking, no forms -- crawl4ai
  does run a real browser (so it executes page JS to render content), but
  never fills in forms or performs write actions; it's still just reading
  public pages.
- Respects robots.txt (in-process cache for the httpx path; crawl4ai's own
  `check_robots_txt` flag for the crawl4ai path) and rate-limits per host.
- Retries transient failures (timeouts, 429, 5xx) with exponential backoff
  + jitter; does NOT retry permanent failures (403/404/401) since that just
  burns the rate-limit budget for nothing.
- Rotates User-Agent per request from a small realistic pool, and can
  round-robin across proxies via PROXY_LIST, to reduce single-fingerprint
  WAF blocks on long scraping runs.
- Domain allow/deny lists (SCRAPER_ALLOWED_DOMAINS / SCRAPER_BLOCKED_DOMAINS,
  comma-separated) let an operator hard-exclude sites (paywalled news,
  social platforms whose ToS forbid scraping) independent of robots.txt --
  applied uniformly regardless of which HTML backend actually fetches a
  page.
- SCRAPER_HTML_BACKEND (default "auto") picks the HTML fetch/extract
  backend: "auto" tries crawl4ai first and falls back per-URL to the
  httpx+trafilatura path if crawl4ai isn't installed or a given page's
  crawl4ai fetch fails; "crawl4ai" requires it (surfaces its errors
  instead of silently falling back); "httpx" disables crawl4ai entirely
  and always uses the original path. crawl4ai adds real JS rendering
  (catches content on pages that render client-side, which httpx simply
  can't see) and a boilerplate-pruning content filter tuned for
  training-data extraction -- see web_scraper_mcp/crawl4ai_backend.py.

Run with:
    python server.py                # stdio transport, for MCP-aware clients
"""

import asyncio
import atexit
import base64
import os
import sys
import time
import urllib.parse
import urllib.robotparser
from typing import Optional

import httpx
from mcp.server.fastmcp import FastMCP

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extractors
import net_utils
import crawl4ai_backend

mcp = FastMCP(
    "web-scraper",
    host=os.environ.get("MCP_HOST", "127.0.0.1"),
    port=int(os.environ.get("MCP_PORT", "8000")),
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MAX_TEXT_BYTES = int(os.environ.get("SCRAPER_MAX_TEXT_MB", "10")) * 1024 * 1024
MAX_BINARY_BYTES = int(os.environ.get("SCRAPER_MAX_BINARY_MB", "50")) * 1024 * 1024
FETCH_TIMEOUT = float(os.environ.get("SCRAPER_FETCH_TIMEOUT", "20"))
RETRY_ATTEMPTS = int(os.environ.get("SCRAPER_RETRY_ATTEMPTS", "3"))

# "auto" (default): try crawl4ai (real headless-browser fetch + pruned
# markdown) first for every HTML URL, falling back per-URL to the legacy
# httpx+trafilatura/readability path if crawl4ai isn't installed or fails
# on a given page. "crawl4ai": require it, surface its errors instead of
# falling back. "httpx": disable crawl4ai entirely, always use the
# original path. See crawl4ai_backend.py for what crawl4ai adds.
HTML_BACKEND = os.environ.get("SCRAPER_HTML_BACKEND", "auto").strip().lower()
if HTML_BACKEND not in ("auto", "crawl4ai", "httpx"):
    print(f"[config] SCRAPER_HTML_BACKEND={HTML_BACKEND!r} not recognized "
          f"(expected auto/crawl4ai/httpx) -- defaulting to 'auto'", file=sys.stderr)
    HTML_BACKEND = "auto"

_allowed_env = os.environ.get("SCRAPER_ALLOWED_DOMAINS", "")
_blocked_env = os.environ.get("SCRAPER_BLOCKED_DOMAINS", "")
ALLOWED_DOMAINS = {d.strip().lower() for d in _allowed_env.split(",") if d.strip()}
BLOCKED_DOMAINS = {d.strip().lower() for d in _blocked_env.split(",") if d.strip()}

# Video/streaming/social platforms whose player pages are JS-rendered with
# no extractable article prose -- these get routed to extract_video instead
# of the HTML article path, never treated as "extraction failed."
VIDEO_MEDIA_DOMAINS = {
    "youtube.com", "youtu.be", "vimeo.com", "dailymotion.com", "twitch.tv",
    "tiktok.com", "soundcloud.com", "podcasts.apple.com",
}
# Platforms that are neither extractable articles nor extractable media --
# login-walled or feed-only, filtered out at search-result time so no
# fetch/extract round-trip (and rate-limit slot) is wasted on them.
UNSUPPORTED_DOMAINS = {
    "instagram.com", "facebook.com", "x.com", "twitter.com",
}


def _host(url: str) -> str:
    h = urllib.parse.urlparse(url).netloc.lower()
    return h[4:] if h.startswith("www.") else h


def _domain_allowed(url: str) -> Optional[str]:
    """Returns None if allowed, else a human-readable reason it's blocked."""
    host = _host(url)
    if ALLOWED_DOMAINS and not any(host == d or host.endswith("." + d) for d in ALLOWED_DOMAINS):
        return f"{host} not in SCRAPER_ALLOWED_DOMAINS allow-list"
    if any(host == d or host.endswith("." + d) for d in BLOCKED_DOMAINS):
        return f"{host} is in SCRAPER_BLOCKED_DOMAINS"
    if any(host == d or host.endswith("." + d) for d in UNSUPPORTED_DOMAINS):
        return f"{host} is a login-walled/feed-only platform, not supported"
    return None


def _is_video_or_media_url(url: str) -> bool:
    host = _host(url)
    return any(host == d or host.endswith("." + d) for d in VIDEO_MEDIA_DOMAINS)


def _default_headers() -> dict:
    return {
        "User-Agent": net_utils.user_agents.get(),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }


def _client_kwargs() -> dict:
    kwargs = {"follow_redirects": True, "timeout": FETCH_TIMEOUT}
    proxy = net_utils.proxies.get()
    if proxy:
        kwargs["proxy"] = proxy
    return kwargs


_robots_cache: dict = {}
_last_request_time: dict = {}
MIN_SECONDS_BETWEEN_REQUESTS_PER_HOST = float(os.environ.get("SCRAPER_MIN_HOST_INTERVAL", "2.0"))


def _robots_allowed(url: str) -> bool:
    host = _host(url)
    if host not in _robots_cache:
        rp = urllib.robotparser.RobotFileParser()
        robots_url = f"{urllib.parse.urlparse(url).scheme}://{host}/robots.txt"
        try:
            resp = httpx.get(robots_url, timeout=5, headers=_default_headers())
            rp.parse(resp.text.splitlines())
        except Exception:
            _robots_cache[host] = None
            return True
        _robots_cache[host] = rp
    rp = _robots_cache[host]
    if rp is None:
        return True
    return rp.can_fetch(net_utils.user_agents.get(), url)


async def _rate_limit(url: str):
    host = _host(url)
    now = time.monotonic()
    last = _last_request_time.get(host, 0)
    wait = MIN_SECONDS_BETWEEN_REQUESTS_PER_HOST - (now - last)
    if wait > 0:
        await asyncio.sleep(wait)
    _last_request_time[host] = time.monotonic()


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

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
    from ddgs import DDGS
    try:
        from ddgs.exceptions import DDGSException
    except ImportError:
        DDGSException = Exception

    backends_env = os.environ.get("DDGS_BACKENDS")
    backends = [b.strip() for b in backends_env.split(",")] if backends_env else \
        ["brave", "yandex", "auto"]

    last_err = None
    for backend in backends:
        for attempt in range(2):
            try:
                results = []
                skipped = 0
                with DDGS() as ddgs:
                    for r in ddgs.text(query, max_results=max_results, backend=backend):
                        url = r.get("href", "")
                        if "bing.com/aclick" in url:
                            continue
                        reason = _domain_allowed(url)
                        if reason:
                            skipped += 1
                            continue
                        results.append({"title": r.get("title", ""), "url": url,
                                         "snippet": r.get("body", "")})
                if results:
                    print(f"[web_search] {query!r} via backend={backend} -> {len(results)} hits"
                          + (f" ({skipped} filtered by domain rules)" if skipped else ""),
                          file=sys.stderr, flush=True)
                    return results
                last_err = f"backend={backend} returned zero results (likely rate-limited/blocked)"
            except DDGSException as e:
                last_err = f"backend={backend}: {type(e).__name__}: {e}"
            except Exception as e:
                last_err = f"backend={backend}: {type(e).__name__}: {e}"
            time.sleep(1)

    print(f"[web_search] {query!r} FAILED all backends -- {last_err}", file=sys.stderr, flush=True)
    return [{"error": last_err or "unknown search failure", "query": query}]


# ---------------------------------------------------------------------------
# Fetching (text + binary), retried and rate-limited
# ---------------------------------------------------------------------------

async def _do_fetch(url: str, max_bytes: int, want_text: bool) -> dict:
    reason = _domain_allowed(url)
    if reason:
        return {"status": None, "content_type": None, "error": f"blocked: {reason}"}
    if not _robots_allowed(url):
        return {"status": None, "content_type": None, "error": "disallowed by robots.txt"}

    await _rate_limit(url)

    async def _attempt():
        async with httpx.AsyncClient(headers=_default_headers(), **_client_kwargs()) as client:
            resp = await client.get(url)
        if net_utils.is_retryable_status(resp.status_code):
            resp.raise_for_status()
        return resp

    try:
        def _on_retry(attempt, exc):
            print(f"[fetch] retry {attempt}/{RETRY_ATTEMPTS} for {url}: {exc}",
                  file=sys.stderr, flush=True)
        resp = await net_utils.retry_async(
            _attempt, attempts=RETRY_ATTEMPTS,
            retry_on=(httpx.HTTPStatusError, httpx.TransportError, httpx.TimeoutException),
            on_retry=_on_retry,
        )
    except httpx.HTTPStatusError as e:
        return {"status": e.response.status_code, "content_type": None,
                "error": f"HTTP {e.response.status_code}"}
    except Exception as e:
        return {"status": None, "content_type": None, "error": str(e)}

    content_type = resp.headers.get("content-type", "")
    if resp.status_code >= 400:
        return {"status": resp.status_code, "content_type": content_type,
                "error": f"HTTP {resp.status_code}"}

    body = resp.content[:max_bytes]
    truncated = len(resp.content) > max_bytes
    if want_text:
        return {"status": resp.status_code, "content_type": content_type,
                "text": body.decode(resp.encoding or "utf-8", errors="replace"),
                "truncated": truncated, "error": None}
    return {"status": resp.status_code, "content_type": content_type,
            "data": body, "truncated": truncated, "error": None}


@mcp.tool()
async def fetch_page(url: str, max_chars: int = 200_000) -> dict:
    """Fetch the raw text/HTML content at a URL (GET only). Honors
    robots.txt, domain allow/deny lists, and a per-host rate limit; retries
    transient failures (timeouts, 429, 5xx) with backoff.
    Returns {status, content_type, html, error}."""
    result = await _do_fetch(url, MAX_TEXT_BYTES, want_text=True)
    if result.get("error"):
        return {"status": result.get("status"), "content_type": result.get("content_type"),
                "html": "", "error": result["error"]}
    return {"status": result["status"], "content_type": result["content_type"],
            "html": result["text"][:max_chars], "error": None}


@mcp.tool()
async def fetch_binary(url: str) -> dict:
    """Fetch raw bytes at a URL (GET only) for non-HTML content -- PDFs,
    images, Office docs, etc. Same robots.txt/domain/rate-limit/retry
    behavior as fetch_page. Returns base64-encoded content since MCP tool
    results are JSON.
    Returns {status, content_type, data_base64, size_bytes, truncated, error}.
    Capped at SCRAPER_MAX_BINARY_MB (default 50MB) to avoid a single huge
    file (e.g. an unbounded video) blowing up memory -- for anything larger,
    use transcribe_media, which streams audio-only rather than fetching the
    whole file into memory.
    """
    result = await _do_fetch(url, MAX_BINARY_BYTES, want_text=False)
    if result.get("error"):
        return {"status": result.get("status"), "content_type": result.get("content_type"),
                "data_base64": "", "size_bytes": 0, "truncated": False, "error": result["error"]}
    data = result["data"]
    return {"status": result["status"], "content_type": result["content_type"],
            "data_base64": base64.b64encode(data).decode("ascii"),
            "size_bytes": len(data), "truncated": result["truncated"], "error": None}


# ---------------------------------------------------------------------------
# Format-agnostic extraction
# ---------------------------------------------------------------------------

@mcp.tool()
async def extract_content(url: str) -> dict:
    """Fetch a URL and extract clean text, auto-detecting its format:
    HTML article, PDF (with OCR fallback for scanned pages), DOCX, PPTX,
    XLSX, CSV, image (OCR), or video/audio (captions, else local ASR
    transcription). This is the primary extraction tool -- prefer it over
    fetch_page/fetch_binary unless you specifically need the raw bytes.

    Returns {title, text, author, date, content_type, url, error, extra}.
    `text` is empty and `error` is set if extraction fails, robots.txt/
    domain rules disallow the fetch, or the format has no supported
    extractor yet.
    """
    reason = _domain_allowed(url)
    if reason:
        return {"title": None, "text": "", "author": None, "date": None,
                "content_type": "unknown", "url": url, "error": f"blocked: {reason}", "extra": {}}

    if _is_video_or_media_url(url):
        return await transcribe_media(url)

    # HEAD first to learn content-type cheaply before deciding text vs
    # binary fetch; if HEAD fails/is blocked (some servers reject HEAD),
    # fall back to sniffing the URL extension only.
    content_type_hint = None
    try:
        if _robots_allowed(url):
            async with httpx.AsyncClient(headers=_default_headers(), **_client_kwargs()) as client:
                head = await client.head(url)
                content_type_hint = head.headers.get("content-type")
    except Exception:
        pass

    kind = extractors.detect_content_kind(url, content_type_hint)

    if kind == "html":
        if HTML_BACKEND in ("auto", "crawl4ai"):
            await _rate_limit(url)  # crawl4ai does its own fetch, outside _do_fetch's rate limiter
            c4_result = await crawl4ai_backend.fetch_and_extract(url, timeout=FETCH_TIMEOUT + 10)
            if not c4_result.get("error"):
                return c4_result
            if HTML_BACKEND == "crawl4ai":
                return c4_result  # explicitly required -- surface its error rather than fall back
            print(f"[extract_content] crawl4ai path failed for {url} ({c4_result['error']}) "
                  f"-- falling back to httpx+trafilatura", file=sys.stderr, flush=True)

        page = await fetch_page(url)
        if page["error"] or not page["html"]:
            return {"title": None, "text": "", "author": None, "date": None,
                    "content_type": "html", "url": url,
                    "error": page["error"] or "empty page", "extra": {}}
        return extractors.extract_html(page["html"], url)

    if kind in ("video", "audio"):
        return await transcribe_media(url)

    binary = await fetch_binary(url)
    if binary["error"]:
        return {"title": None, "text": "", "author": None, "date": None,
                "content_type": kind, "url": url, "error": binary["error"], "extra": {}}
    data = base64.b64decode(binary["data_base64"])
    return extractors.extract_from_bytes(kind, data, url)


@mcp.tool()
async def deep_crawl(seed_url: str, max_pages: int = 20, max_depth: int = 2,
                      keywords: str = "", same_domain_only: bool = True) -> list[dict]:
    """Crawl outward from a seed URL (a docs site homepage, a blog index, a
    wiki category page, ...) using crawl4ai's JS-aware headless-browser
    crawler, following in-page links up to max_depth hops and extracting up
    to max_pages pages in ONE call. Much higher yield per round than one
    web_search hit -> one extract_content call each -- reach for this when
    you already know a specific domain is relevant and want to harvest it
    thoroughly, rather than for open-ended topic discovery (use web_search
    for that).

    keywords: optional comma-separated terms (e.g. "gradient descent,
    backpropagation, attention mechanism") used to prioritize which
    discovered links get visited first when the domain has more pages than
    max_pages allows. Leave empty for plain breadth-first order.
    same_domain_only: if True (default), never follows links off the seed
    URL's own host -- keeps "harvest this docs site" from wandering off to
    unrelated linked domains. Set False only if you deliberately want that.

    Returns a list of {title, text, author, date, content_type, url, error,
    extra} dicts -- same shape as extract_content -- one per successfully
    extracted page (pages that fail to extract are silently omitted, not
    errored). Respects the same SCRAPER_ALLOWED_DOMAINS/
    SCRAPER_BLOCKED_DOMAINS rules and robots.txt as every other tool here.

    Requires crawl4ai (pip install crawl4ai && crawl4ai-setup). If it isn't
    installed, or the seed URL itself is blocked/unreachable, returns a
    single-item [{"error": "...", "url": seed_url}] list -- same
    "never fails silently" convention as web_search.
    """
    reason = _domain_allowed(seed_url)
    if reason:
        return [{"error": f"blocked: {reason}", "url": seed_url}]

    kw_list = [k.strip() for k in keywords.split(",") if k.strip()] or None
    await _rate_limit(seed_url)
    pages = await crawl4ai_backend.deep_crawl(
        seed_url, max_pages=max_pages, max_depth=max_depth,
        keywords=kw_list, same_domain_only=same_domain_only,
    )
    if len(pages) == 1 and pages[0].get("error") and set(pages[0].keys()) <= {"error", "url"}:
        return pages  # crawl4ai-not-installed / seed-unreachable -- propagate as-is

    # crawl4ai's own DomainFilter only enforces "stayed on seed_url's host";
    # it doesn't know about this server's SCRAPER_BLOCKED_DOMAINS, so
    # re-apply that here to every discovered page before returning them.
    return [p for p in pages if not _domain_allowed(p.get("url", seed_url))]



    """DEPRECATED alias for extract_content, kept for backward
    compatibility with existing callers. New code should call
    extract_content directly, which also handles PDFs/Office docs/images/
    video without a separate tool call.
    Returns {title, text, author, date, url, error} (no `extra`/`content_type`
    keys, matching the original shape)."""
    result = await extract_content(url)
    return {"title": result["title"], "text": result["text"], "author": result["author"],
            "date": result["date"], "url": result["url"], "error": result["error"]}


@mcp.tool()
async def transcribe_media(url: str, asr_fallback: bool = True,
                            asr_model_size: str = "base",
                            max_duration_seconds: int = 3600) -> dict:
    """Get a transcript for a video or audio URL. Prefers existing
    captions/subtitles (fast, free, usually accurate); falls back to local
    ASR transcription via faster-whisper if none exist and asr_fallback is
    true. Works for YouTube/Vimeo/etc. (via yt-dlp) and direct media file
    URLs. `max_duration_seconds` bounds ASR cost on very long videos --
    content longer than this is skipped with an explanatory error rather
    than silently transcribing for many minutes.

    Requires: yt-dlp (always); faster-whisper + ffmpeg on PATH (only for
    the ASR fallback path -- captions-only usage doesn't need them).
    """
    reason = _domain_allowed(url)
    if reason:
        return {"title": None, "text": "", "author": None, "date": None,
                "content_type": "video", "url": url, "error": f"blocked: {reason}", "extra": {}}
    kind = extractors.detect_content_kind(url)
    if kind == "audio" and not _is_video_or_media_url(url):
        # Direct audio file URL (mp3/wav/etc.), not a streaming platform --
        # no captions possible, go straight to ASR.
        if not asr_fallback:
            return {"title": None, "text": "", "author": None, "date": None,
                    "content_type": "audio", "url": url,
                    "error": "direct audio file with asr_fallback=False -- no transcript possible",
                    "extra": {}}
        return extractors.extract_audio(url, is_local_file=False, model_size=asr_model_size)
    return extractors.extract_video(url, asr_fallback=asr_fallback,
                                     asr_model_size=asr_model_size,
                                     max_duration_seconds=max_duration_seconds)


@mcp.tool()
def healthcheck() -> dict:
    """Cheap connectivity/config probe -- call this from an orchestrator
    before kicking off a real scrape, or on a schedule to catch a broken
    search backend or missing optional dependency before it silently eats
    a whole run. Does NOT do a full search/scrape; reports which optional
    extraction dependencies are importable so you know up front which
    formats are actually usable, plus whether robots.txt fetching works."""
    import importlib
    report = {
        "ddgs_importable": False,
        "robots_check_ok": False,
        "proxy_pool_enabled": net_utils.proxies.enabled,
        "formats": {},
        "error": None,
    }
    try:
        importlib.import_module("ddgs")
        report["ddgs_importable"] = True
    except Exception as e:
        report["error"] = f"ddgs import failed: {e}"

    optional_deps = {
        "pdf": ["pdfplumber"],
        "pdf_ocr": ["pdf2image", "pytesseract"],
        "docx": ["docx"],
        "pptx": ["pptx"],
        "xlsx": ["openpyxl"],
        "image_ocr": ["PIL", "pytesseract"],
        "video_captions": ["yt_dlp"],
        "video_audio_asr": ["yt_dlp", "faster_whisper"],
        "crawl4ai_html": ["crawl4ai"],
    }
    for label, modules in optional_deps.items():
        ok = True
        for m in modules:
            try:
                importlib.import_module(m)
            except Exception:
                ok = False
                break
        report["formats"][label] = ok

    report["html_backend"] = HTML_BACKEND
    if HTML_BACKEND != "httpx" and not report["formats"]["crawl4ai_html"]:
        note = ("SCRAPER_HTML_BACKEND is "
                f"{HTML_BACKEND!r} but crawl4ai isn't importable -- every HTML "
                "extraction will silently fall back to httpx+trafilatura" +
                (" (auto mode, so this is fine, just lower-fidelity)"
                 if HTML_BACKEND == "auto" else
                 " and error out (backend=crawl4ai requires it)"))
        report["error"] = (report["error"] + "; " if report["error"] else "") + note

    try:
        report["robots_check_ok"] = _robots_allowed("https://example.com/")
    except Exception as e:
        report["error"] = (report["error"] + "; " if report["error"] else "") + f"robots check failed: {e}"
    return report


def _cleanup_crawl4ai():
    """Best-effort close of the shared crawl4ai browser (if it was ever
    started) so its Playwright subprocess doesn't linger after this process
    exits. No-op, harmlessly, if crawl4ai was never used this run."""
    try:
        asyncio.run(crawl4ai_backend.shutdown())
    except Exception:
        pass


atexit.register(_cleanup_crawl4ai)


if __name__ == "__main__":
    # Default stdio keeps `dataset_agent.py`'s stdio_client working unchanged.
    # Set MCP_TRANSPORT=streamable-http (plus optionally MCP_HOST/MCP_PORT)
    # to expose this as an HTTP endpoint instead.
    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    mcp.run(transport=transport)
