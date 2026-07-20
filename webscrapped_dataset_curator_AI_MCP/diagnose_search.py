#!/usr/bin/env python3
"""
diagnose_search.py

Run this directly (no MCP, no agent, no Ollama) to find out exactly why
web_search is returning nothing. Tries several backends and prints the
real exception for each, since the agent/MCP layer can only report
whatever ddgs itself gives it.

Usage:
    python diagnose_search.py "climate change ocean currents"
"""
import sys
import time
import traceback

query = sys.argv[1] if len(sys.argv) > 1 else "python programming tutorial"

try:
    from ddgs import DDGS
except ImportError:
    print("FAIL: `ddgs` isn't installed. Run: pip install ddgs")
    sys.exit(1)

try:
    from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException
except ImportError:
    DDGSException = RatelimitException = TimeoutException = Exception

backends_to_try = ["duckduckgo", "bing", "brave", "mojeek", "yandex", "auto"]

print(f"Query: {query!r}\n")

for backend in backends_to_try:
    print(f"--- backend={backend} ---")
    start = time.time()
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5, backend=backend))
        elapsed = time.time() - start
        if results:
            print(f"OK  ({elapsed:.1f}s) -- {len(results)} results")
            print(f"    first: {results[0].get('title', '')!r} -> {results[0].get('href', '')}")
        else:
            print(f"EMPTY ({elapsed:.1f}s) -- zero results, no exception raised")
    except RatelimitException as e:
        print(f"RATE-LIMITED ({time.time()-start:.1f}s): {e}")
    except TimeoutException as e:
        print(f"TIMEOUT ({time.time()-start:.1f}s): {e}")
    except DDGSException as e:
        print(f"DDGS ERROR ({time.time()-start:.1f}s): {type(e).__name__}: {e}")
    except Exception as e:
        print(f"UNEXPECTED ERROR ({time.time()-start:.1f}s): {type(e).__name__}: {e}")
        traceback.print_exc()
    print()
    time.sleep(1)

print(
    "If every backend above shows RATE-LIMITED, TIMEOUT, or EMPTY: your "
    "network (likely a cloud/datacenter IP -- AWS/GCP/Azure/Hetzner/etc.) "
    "is probably blocked by these engines' anti-bot systems outright, not "
    "just being unlucky with retries. That's a network-reputation problem, "
    "not a code bug -- the fix is a paid search API (Brave Search API, "
    "Serper, Tavily) or a residential/rotating proxy, not more retries.\n"
    "If one backend (e.g. bing or brave) returns OK: set DDGS_BACKEND to "
    "that value as an env var before running server.py."
)
