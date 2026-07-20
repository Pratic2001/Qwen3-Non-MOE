"""
net_utils.py

Small, dependency-free (no `tenacity` needed) resilience helpers shared by
server.py: retry with exponential backoff + jitter, a rotating pool of
realistic browser User-Agent strings, and optional proxy rotation. Kept
separate from server.py so they're independently testable and reusable if
you add more fetch-shaped tools later.
"""

from __future__ import annotations

import asyncio
import itertools
import os
import random
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")

# A small, realistic pool of desktop browser UAs across OS/browser combos.
# Rotating across requests (rather than sending one fixed string for the
# whole run) reduces the chance of a single UA getting fingerprinted and
# rate-limited by a WAF partway through a long scraping job. Override/extend
# via the SCRAPER_USER_AGENTS env var (comma-separated).
_DEFAULT_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 "
    "Firefox/125.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0",
]


def _load_list_env(var: str, default: list) -> list:
    raw = os.environ.get(var)
    if not raw:
        return default
    items = [x.strip() for x in raw.split(",") if x.strip()]
    return items or default


class UserAgentPool:
    def __init__(self):
        self._agents = _load_list_env("SCRAPER_USER_AGENTS", _DEFAULT_USER_AGENTS)

    def get(self) -> str:
        return random.choice(self._agents)


class ProxyPool:
    """Round-robins across proxies from PROXY_LIST (comma-separated
    http(s)://user:pass@host:port entries). Empty/unset means no proxy --
    the pool is a no-op and get() returns None."""

    def __init__(self):
        self._proxies = _load_list_env("PROXY_LIST", [])
        self._cycle = itertools.cycle(self._proxies) if self._proxies else None

    @property
    def enabled(self) -> bool:
        return self._cycle is not None

    def get(self) -> str | None:
        return next(self._cycle) if self._cycle else None


user_agents = UserAgentPool()
proxies = ProxyPool()


async def retry_async(
    fn: Callable[[], Awaitable[T]],
    *,
    attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 20.0,
    retry_on: tuple = (Exception,),
    on_retry: Callable[[int, Exception], None] | None = None,
) -> T:
    """Call an async `fn` with exponential backoff + full jitter.

    Retries on any exception type in `retry_on`. Delay for attempt i (1-indexed)
    is `min(max_delay, base_delay * 2**(i-1))`, randomized uniformly in
    [0, delay] (the "full jitter" strategy) to avoid a thundering herd if
    many requests hit the same transient failure at once (e.g. a host
    briefly rate-limiting a whole scraping run).
    """
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return await fn()
        except retry_on as e:  # type: ignore[misc]
            last_exc = e
            if attempt == attempts:
                break
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            delay = random.uniform(0, delay)
            if on_retry:
                on_retry(attempt, e)
            await asyncio.sleep(delay)
    assert last_exc is not None
    raise last_exc


def is_retryable_status(status_code: int | None) -> bool:
    """HTTP statuses worth retrying: rate limits and transient server
    errors. 404/403/401 etc. are NOT retryable -- retrying those just
    burns the rate-limit budget for no benefit."""
    if status_code is None:
        return True  # connection-level failure, not an HTTP response at all
    return status_code == 429 or 500 <= status_code < 600
