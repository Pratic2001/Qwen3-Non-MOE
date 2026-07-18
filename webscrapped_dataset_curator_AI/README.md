# Infinite Data Agent

Live-scraping companion to your `build_dataset.py` / `download_sft_data.py`
pipeline. When your fixed HF sources (FineWeb, the-stack, OpenR1-Math, etc.)
run dry, this agent keeps producing data by searching and scraping the open
web, using a **local Ollama model as the planner + quality judge** and an
**MCP server as the hands** (search/fetch/extract).

It writes output in the **exact same JSONL shard + manifest format** your
existing scripts already use, so `pack_dataset.py` / `pack_sft_data.py` /
`pack_grpo_data.py` need zero changes — just point them at the same
`--out-dir` (or merge directories) and they'll pack the agent's shards
alongside the HF-sourced ones.

## Why this fixes the "data source can run out" flaw

- `build_dataset.py` walks a **fixed list** of HF datasets. Once each is
  exhausted, that category simply stops early (you saw this logged as
  `[category] source ... exhausted at ...`).
- This agent instead has the LLM **generate new search queries on the fly**,
  explicitly avoiding queries it has already used, so the query space keeps
  expanding instead of being a fixed list. As long as the open web has
  content on a topic, it has data.

## Architecture

```
Ollama (planner + judge)  <-->  dataset_agent.py  <-->  MCP server (server.py)
        local model              orchestrator            web_search / fetch_page
                                  + quality.py             / extract_article
                                  (same filters as               |
                                   build_dataset.py)             v
                                        |                  live web pages
                                        v
                              ./data/<category>/*.jsonl + manifest.json
                              (same shape as build_dataset.py output)
```

1. **`web_scraper_mcp/server.py`** — an MCP server with three read-only tools:
   - `web_search(query, max_results)` — DuckDuckGo search, no API key needed.
   - `fetch_page(url)` — raw HTML fetch, respects `robots.txt`, rate-limited
     per host (2s between requests to the same domain).
   - `extract_article(url)` — fetches + runs `trafilatura` (falling back to
     `readability-lxml`) to strip nav/ads/boilerplate down to clean prose.

2. **`agent/dataset_agent.py`** — the orchestrator:
   - Asks Ollama for a batch of specific search queries per category,
     seeded from `agent/topics.py`, explicitly telling it to avoid repeating
     recent queries so coverage keeps growing.
   - Calls the MCP tools to search + extract.
   - Runs the **same heuristic filters** as `build_dataset.py` (alpha ratio,
     repetition ratio, code extension/hygiene checks — see `agent/quality.py`,
     which is a straight extraction of that logic) plus two additions suited
     to live scraping: a junk-phrase filter (cookie banners, CAPTCHAs, "add
     to cart", etc.) and a near-duplicate filter (mirrors/reprints are far
     more common on the open web than in curated HF datasets).
   - Optionally runs an **LLM quality judge** pass (Ollama) on documents that
     already passed the heuristics — catches incoherent/low-substance text
     the cheap filters miss. Disable with `--no-llm-judge` for raw throughput.
   - Writes shards with `ShardWriter`, which **resumes shard numbering** from
     whatever's already in `--out-dir`, so repeated runs append rather than
     overwrite.

3. **`agent/topics.py`** — starter topic lists per category. The agent
   expands past these on its own; edit this file to steer initial direction
   or add new categories.

## Setup

```bash
cd infinite_data_agent
pip install -r requirements.txt

# pull a small, fast instruct model for planning/judging
ollama pull llama3.1          # or qwen2.5:7b-instruct, mistral, etc.
ollama serve                  # if not already running
```

## Usage

Pretraining-style output (matches `build_dataset.py`):
```bash
python agent/dataset_agent.py --target-size 500MB \
    --categories web,knowledge,reasoning,code,math \
    --out-dir ./data --mode pretrain
```

SFT-style output (matches `download_sft_data.py`, `{prompt, thinking, answer}`):
```bash
python agent/dataset_agent.py --target-size 300MB \
    --categories math,code,reasoning \
    --out-dir ./sft_data --mode sft
```

Then pack exactly as before:
```bash
python pack_dataset.py --data-dir ./data ...
python pack_sft_data.py --data-dir ./sft_data ...
```

### Practical usage pattern

Run your existing HF pipeline first (it's higher-quality per-token and
free of scraping overhead), then top up whatever's short with this agent
pointed at the same `--out-dir`:

```bash
python build_dataset.py --target-size 5GB --out-dir ./data
# check ./data/manifest.json for any category that fell short of budget
python agent/dataset_agent.py --target-size 1GB --out-dir ./data \
    --categories web,code   # whichever categories under-filled
```

Because `ShardWriter` resumes shard indices instead of overwriting, this is
safe to run repeatedly against the same output directory as a continuous
top-up job (e.g. a nightly cron/systemd timer).

## Notes, limits, and things worth tuning

- **Search backend**: DuckDuckGo's free endpoint is rate-limited and can be
  flaky at high volume. For serious scale, swap `web_search` in `server.py`
  to a paid API (Bing Search, Serper, Tavily, Brave Search API) — the tool
  signature (`query`, `max_results` → list of `{title, url, snippet}`) is
  the only contract the agent depends on.
- **robots.txt / rate limiting**: the server checks robots.txt per host and
  caps request rate to 1 per 2s per domain. This is intentionally
  conservative; check the terms of service of any site you scrape at volume,
  and consider adding an explicit domain allow/deny list in `server.py` for
  sites you know you don't want to touch (e.g. paywalled news, social media
  ToS that prohibit scraping).
- **`thinking` field in SFT mode**: raw web articles don't contain a
  chain-of-thought trace, so `extract_sft_pair` leaves `"thinking": ""`
  (your `sft.py` training code already handles this by skipping the
  `<think>` block, per the docstring in `download_sft_data.py`). If you
  want synthetic CoT instead of an empty field, extend `extract_sft_pair`'s
  prompt to ask the model to think step-by-step before answering and
  capture that as `thinking`.
- **Near-dup filter** (`NearDedup` in `quality.py`) is a lightweight
  shingle-overlap heuristic, not a real MinHash/LSH index — fine at the
  scale of a single run (thousands–tens of thousands of docs per category)
  but not a substitute for a proper dedup pass (e.g. `datasketch`) if you
  scale this to millions of scraped documents.
- **Cost/speed knob**: `--no-llm-judge` skips the Ollama quality-judging
  call per candidate document, which is the slowest part of the loop (one
  local-model call per doc). Heuristic filters alone are much faster but
  let more borderline text through.
- **This sandbox**: I built and organized this code here, but couldn't
  execute a live end-to-end scrape from this environment (no general
  internet egress). Test it in your own environment where Ollama and open
  web access are available; the MCP tool contracts and shard format are
  the load-bearing parts and match your existing scripts exactly, so
  integration should be a drop-in.
