## `agent/codegen_pipeline.py`: plan -> generate -> TEST -> run

This is the primary way to run this codebase as an "AI developer": for each
category/dataset it discovers what data is available, has an LLM generate
mapping code, **tests that code against real sample rows before trusting it
with a real run**, and only then executes at scale.

The key change from earlier versions: the LLM used to write an entire
standalone script per dataset (argparse, the streaming loop, quality
filtering, dedup, shard writing, error handling -- everything). That is a
lot of surface for a small local model to get right, and it's exactly the
kind of surface where this codebase's actual historical bugs lived (a
regex silently corrupting extracted answers, a mismatched prompt template,
a missing exhaustion check, a double-shift in a training loop) -- none of
which are "picked the wrong column," all of which are "the surrounding
plumbing was subtly wrong."

So now the LLM's job is shrunk to one pure function:

```python
def map_row(row: dict) -> Optional[dict]:
    ...  # column mapping only -- no filtering, no I/O, no loop
```

and everything else lives in `agent/harness.py`: hand-written once,
unit-tested once (`agent/tests/test_harness.py`), reused identically for
every dataset forever. The pipeline for each dataset is:

1. **discover + sample** (unchanged) -- find candidate datasets, stream a
   few real rows, note the actual columns.
2. **codegen** -- ask Ollama for `map_row(row)` only, shown this dataset's
   real columns and sample rows.
3. **test** (new) -- `py_compile` it, then run an auto-generated pytest
   suite (`agent/test_gen.py`) that calls `map_row()` on the SAME sample
   rows and checks every output against the target schema
   (`agent/schema_check.py`). This catches "wrong column name", "always
   returns None", and "crashes on a null field" in well under a second --
   before a multi-hour download/crawl ever starts. Any failure (compile,
   or a specific pytest assertion) is fed back to Ollama as a targeted
   repair prompt, up to a bounded number of attempts.
4. **run** -- `agent/harness.py` imports the now-tested `map_row()` and
   does the real streaming, quality filtering, exact-dedup, shard writing,
   and progress reporting. It also has its own runtime safety net: if
   `map_row()` starts raising or producing schema-invalid records at a
   rate that indicates a systemic bug (not just a few odd source rows), it
   aborts fast with a specific error rather than quietly finishing with
   near-empty output.

Run `pytest agent/tests/test_harness.py -q` any time you touch
`harness.py` or `schema_check.py` -- that's the one piece of this pipeline
meant to be trusted rather than re-verified per dataset.

# Infinite Data Agent

Live-scraping companion to your `build_dataset.py` / `download_sft_data.py`
pipeline. When your fixed HF sources (FineWeb, the-stack, OpenR1-Math, etc.)
run dry, this agent keeps producing data by searching and scraping the open
web, using a **local Ollama model as the planner + quality judge** and an
**MCP server as the hands** (search/fetch/extract) — across HTML, PDFs,
Office documents, images, and video/audio, not just web articles.

It writes output in the **exact same JSONL shard + manifest format** your
existing scripts already use, so `pack_dataset.py` / `pack_sft_data.py` /
`pack_grpo_data.py` need zero changes — just point them at the same
`--out-dir` (or merge directories) and they'll pack the agent's shards
alongside the HF-sourced ones.

## What's new: multi-format extraction

The original version only understood HTML articles. `extract_content` now
auto-detects and extracts:

| Format | Method |
|---|---|
| HTML articles | `trafilatura`, falls back to `readability-lxml` |
| PDF | `pdfplumber` text layer; **OCR fallback** (`pdf2image` + `pytesseract`) for scanned/image-only pages |
| Word (`.docx`) | `python-docx` — paragraphs + tables |
| PowerPoint (`.pptx`) | `python-pptx` — slide text, tables, speaker notes |
| Excel (`.xlsx`) / CSV | `openpyxl` — sheet contents as flattened text |
| Images (`.png`/`.jpg`/etc.) | OCR via `pytesseract` |
| Video (YouTube/Vimeo/direct files) | **Captions/subtitles first** (fast, accurate, free) via `yt-dlp`; **local ASR fallback** via `faster-whisper` if none exist |
| Audio (`.mp3`/`.wav`/etc.) | `faster-whisper` transcription |

Every extractor returns the same shape — `{title, text, author, date,
content_type, url, error, extra}` — so the agent's filtering/dedup/writing
logic never branches on format. Every heavy dependency is imported lazily,
so missing one optional package (say, `faster-whisper`) degrades that one
format to a clear error instead of breaking the whole server. Run
`healthcheck()` to see which formats are actually usable in your
environment before kicking off a real run.

## Architecture

```
Ollama (planner + judge)  <-->  dataset_agent.py  <-->  MCP server (server.py)
        local model              orchestrator            web_search
                                  + quality.py             fetch_page / fetch_binary
                                  (same filters as         extract_content  ---> extractors.py
                                   build_dataset.py)              |            (pdf/docx/pptx/xlsx/
                                        |                         |             image/video/audio)
                                        |                         v
                                        |                 live web pages, PDFs,
                                        |                 office docs, media
                                        v
                              agent/public_sources.py  ---> Hugging Face Hub (datasets, streaming)
                              (HF/Kaggle top-up,             Kaggle (dataset search + download)
                               runs BEFORE web scraping)
                                        |
                                        v
                              ./data/<category>/*.jsonl
                              + manifest.json
                              (same shape as build_dataset.py output, regardless of which
                               path -- live scrape or public dataset hub -- produced a row)
```

### Public dataset sources (Hugging Face / Kaggle top-up)

When `--public-sources` is set, each category is topped up from Hugging
Face and/or Kaggle **before** falling back to live web search+scraping --
no robots.txt, no rate limiting, no HTML boilerplate to strip, so it's
faster and more reliable than scraping wherever a matching public dataset
already exists.

- **`agent/public_sources.py`** streams rows (`datasets.load_dataset(...,
  streaming=True)` for HF; download + pandas read for Kaggle CSV/JSON/TSV/
  text files) and normalizes each row into the exact same shape
  `extract_content` returns, so every downstream step -- heuristic quality
  filters, the Ollama quality judge, dedup, shard writing -- runs
  identically no matter which path a row came from.
- **The agent's built-in AI (Ollama) does the shaping work**, same as it
  already does for scraped pages: `judge_quality` gates every row, and in
  `--mode sft`, `extract_sft_pair` turns raw rows into `{prompt, answer}`
  pairs. If a dataset already has its own instruction/response-style
  columns (`prompt`/`question`/`instruction` + `answer`/`response`/
  `output`), those are used directly instead of asking the model to invent
  a question -- the dataset author's own labels are trusted over an LLM
  guess, and Ollama only fills in the gap for rows that are just raw prose.
- If you don't name specific datasets, `public_sources.py` **auto-discovers**
  a few per category by searching each hub with that category's topic
  seeds (same "seed + let it expand" idea as the web-search query planner).
- Requires the optional deps in `requirements.txt`'s "public dataset
  sources" section, and (for Kaggle) `KAGGLE_USERNAME`/`KAGGLE_KEY` env
  vars or `~/.kaggle/kaggle.json` (get these from your Kaggle account
  settings). Hugging Face gated/private datasets need `HF_TOKEN` set;
  public datasets need no credentials at all.
- Missing a dependency or credential doesn't break the run -- that one
  backend just gets skipped with a logged warning, same graceful-
  degradation pattern as the per-format extractors.

1. **`web_scraper_mcp/server.py`** — MCP server with these tools:
   - `web_search(query, max_results)` — DuckDuckGo search, no API key needed.
   - `fetch_page(url)` / `fetch_binary(url)` — raw text / raw bytes fetch.
     Both respect `robots.txt`, per-host rate limiting, and domain
     allow/deny lists, and retry transient failures (timeouts, 429, 5xx)
     with exponential backoff + jitter.
   - **`extract_content(url)`** — the primary tool. Detects format (via
     `Content-Type` header, then URL extension, then known
     streaming-platform hosts) and dispatches to the right extractor. HTML
     pages go through **crawl4ai** first (see below), falling back to the
     original `httpx` + `trafilatura`/`readability` path in
     `web_scraper_mcp/extractors.py` if crawl4ai isn't installed or a
     given page's crawl4ai fetch fails.
   - **`deep_crawl(seed_url, max_pages, max_depth, keywords, same_domain_only)`**
     — new. BFS-crawls outward from a seed URL with crawl4ai and extracts
     every page it visits (up to `max_pages`) in one call — see below.
   - `transcribe_media(url)` — dedicated video/audio → transcript tool.
   - `extract_article(url)` — HTML-only alias kept for backward
     compatibility with anything still calling the old tool name.
   - `healthcheck()` — reports which optional per-format dependencies
     (including crawl4ai) are importable, so you can tell "PDF OCR isn't
     wired up" from "the network is down" before a long run silently
     underperforms.
   - Rotates User-Agent per request from a small pool (`web_scraper_mcp/
     net_utils.py`) and can round-robin across proxies via `PROXY_LIST`, to
     reduce single-fingerprint WAF blocks on long runs.

### HTML backend: crawl4ai (real headless-browser fetch + pruned markdown)

The original version fetched HTML with plain `httpx` (no JS execution) and
extracted with `trafilatura`/`readability-lxml`. That's fast and fine for
static pages, but two things it structurally can't do:

- **Render JS.** A page that builds its article body client-side (a lot of
  modern blogs, docs sites, and SPAs) comes back nearly empty from `httpx`
  no matter how good the extractor is downstream — there's no content in
  the raw response to extract. `web_scraper_mcp/crawl4ai_backend.py` runs a
  real headless Chromium via [crawl4ai](https://github.com/unclecode/crawl4ai)
  instead, so it sees the DOM after JS has run.
- **Harvest a whole domain in one call.** The new `deep_crawl` tool follows
  in-page links outward from a seed URL (BFS, optionally scored by keyword
  relevance) and extracts every page it visits — one docs-site homepage or
  blog index becomes dozens of documents in a single tool call, instead of
  needing one `web_search` + one `extract_content` round-trip per page.

Both paths use crawl4ai's `PruningContentFilter`, which tends to strip
nav/ad/sidebar boilerplate more aggressively than `trafilatura` alone, and
return **the same result shape** (`{title, text, author, date,
content_type, url, error, extra}`) as every other extractor here, so
nothing downstream (filtering, dedup, shard writing, `dataset_agent.py`)
branches on which backend actually produced a given document.

- `SCRAPER_HTML_BACKEND` (default `auto`): `auto` tries crawl4ai first for
  every HTML URL and falls back per-URL to the original `httpx`+
  `trafilatura` path if crawl4ai isn't installed or a given page's crawl4ai
  fetch fails; `crawl4ai` requires it (surfaces its errors instead of
  silently falling back, useful for confirming it's actually working);
  `httpx` disables crawl4ai entirely.
- Missing crawl4ai doesn't break anything — same graceful-degradation
  pattern as every other optional per-format dependency here. Run
  `healthcheck()` (reports `crawl4ai_html` under `formats`, plus the
  configured `html_backend`) to confirm it's wired up before a real run.
- `agent/dataset_agent.py`'s `--deep-crawl-per-domain N` flag (0/disabled
  by default) opts a run into using `deep_crawl` as a top-up: each query
  round, up to `N` domains newly seen among that round's search hits also
  get deep-crawled for a few more same-domain pages each
  (`--deep-crawl-max-pages`, default 10) — usually a cheaper way to grow a
  category's document count than planning and running more search queries,
  since you already know those domains are relevant. Each domain is only
  deep-crawled once per category run.

2. **`agent/dataset_agent.py`** — the orchestrator:
   - Asks Ollama for a batch of specific search queries per category,
     seeded from `agent/topics.py`, explicitly avoiding recently used
     queries so coverage keeps growing.
   - Calls `extract_content` for every hit — works uniformly whether the
     hit turned out to be an article, a PDF, a slide deck, or a YouTube
     video.
   - **Processes each round's URLs concurrently** (`--concurrency`, default
     5) instead of one at a time — extraction (network fetch, OCR, ASR) is
     the actual bottleneck, not the search step.
   - Runs the **same heuristic filters** as `build_dataset.py` (alpha
     ratio, repetition ratio, code extension/hygiene checks —
     `agent/quality.py`) plus format-aware additions: a junk-phrase filter,
     a transcript-specific filter (catches ASR noise, music-only stretches,
     stuck-caption repeats), and near-duplicate detection.
   - **Near-dedup now uses real MinHash + LSH** (`datasketch`) when
     installed — scales to far more docs than simple pairwise shingle
     comparison — and falls back automatically to the lightweight
     shingle-overlap method if `datasketch` isn't installed.
   - Optionally runs an **LLM quality judge** pass (Ollama). Disable with
     `--no-llm-judge` for raw throughput.
   - **Resumable state**: each category's used queries and seen URLs
     persist to `.run_state.json` and reload on the next invocation, so a
     killed/interrupted run — or a nightly cron job — doesn't re-search or
     re-extract things it already covered.
   - Writes shards with `ShardWriter`, which **resumes shard numbering**
     from whatever's already in `--out-dir`, so repeated runs append rather
     than overwrite.

3. **`agent/topics.py`** — starter topic lists per category. The agent
   expands past these on its own; edit this file to steer initial direction
   or add new categories.

## Setup

```bash
cd webscrapped_dataset_curator_AI
pip install -r requirements.txt        # core deps (includes crawl4ai)

# One-time browser setup for the crawl4ai HTML backend (skip only if you
# intend to run with SCRAPER_HTML_BACKEND=httpx):
crawl4ai-setup                         # installs Playwright's Chromium + OS deps
crawl4ai-doctor                        # sanity-checks the install

# Optional, per format you actually need (see requirements.txt for the
# breakdown) -- e.g. for PDF OCR + video ASR:
#   pip install pdf2image pytesseract faster-whisper
# Plus system packages for OCR/ASR:
#   apt install poppler-utils tesseract-ocr ffmpeg

# pull a small, fast instruct model for planning/judging
ollama pull llama3.1          # or qwen2.5:7b-instruct, mistral, etc.
ollama serve                  # if not already running
```

Sanity-check what's actually usable in your environment before a real run:
```python
# from a Python shell with web_scraper_mcp on the path
import server
print(server.healthcheck())
```

## Usage

Pretraining-style output (matches `build_dataset.py`):
```bash
python agent/dataset_agent.py --target-size 500MB \
    --categories web,knowledge,reasoning,code,math \
    --out-dir ./data --mode pretrain --concurrency 8
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

### Topping up from public dataset hubs (Hugging Face / Kaggle)

Let the agent auto-discover and pull matching datasets per category, then
fall back to live web scraping for whatever's still short of budget:
```bash
python agent/dataset_agent.py --target-size 1GB --mode pretrain \
    --categories web,knowledge,code,math \
    --out-dir ./data \
    --public-sources huggingface,kaggle
```

Name specific datasets instead of auto-discovery (`category=id1,id2;...`,
or a bare comma list to apply to every category):
```bash
export KAGGLE_USERNAME=you KAGGLE_KEY=xxxx   # from kaggle.com/settings
python agent/dataset_agent.py --target-size 500MB --mode sft \
    --categories math,code \
    --public-sources huggingface,kaggle \
    --hf-datasets "math=openai/gsm8k;code=codeparrot/apps" \
    --kaggle-datasets "code=owner/some-code-qa-dataset"
```

Skip live scraping entirely and fill purely from public hubs (no MCP
scraper subprocess is even launched):
```bash
python agent/dataset_agent.py --target-size 2GB --mode pretrain \
    --categories knowledge,science \
    --public-sources huggingface \
    --public-only
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

Because `ShardWriter` resumes shard indices and `RunState` persists
used-queries/seen-URLs per category, this is safe to run repeatedly against
the same output directory as a continuous top-up job (e.g. a nightly
cron/systemd timer) — each run picks up roughly where the last one left off.

## Notes, limits, and things worth tuning

- **Categories now run concurrently, not one after another**: previously
  `dataset_agent.py` processed categories in a strict sequential loop, so
  total wall-clock time was the SUM across categories. They now run in
  parallel via `asyncio.gather` (bounded by `--category-concurrency`,
  default 0 = unbounded -- all categories at once). Each category already
  has its own `ShardWriter`/`ExactDedup`/`RunState` rooted at
  `<out-dir>/<category>/`, so there's no shared mutable state to race on;
  the shared MCP session and Ollama judge batchers are both designed to
  handle concurrent callers (a single category's `--concurrency` workers
  already prove the session handles concurrent tool calls).
- **The MCP server offloads CPU-bound extraction to process pools**:
  trafilatura/readability HTML parsing, pdfplumber + OCR, python-docx/
  pptx, openpyxl, and faster-whisper ASR are all synchronous, CPU-bound
  functions. Running them directly on the server's single asyncio event
  loop (the original design) blocked every other in-flight request --
  across every category and every `--concurrency` worker -- for the
  duration of each one, which capped real throughput at roughly one
  extraction at a time no matter how high `--concurrency` was set.
  `web_scraper_mcp/server.py` now runs fast extractors (HTML/PDF/Office/
  image) in a `ProcessPoolExecutor` sized by `SCRAPER_EXTRACT_WORKERS`
  (default: all CPU cores), and video/audio transcription in a **separate**,
  smaller pool sized by `SCRAPER_MEDIA_WORKERS` (default: half the cores)
  so a few long-running ASR jobs can't occupy worker slots that many fast
  HTML/PDF extractions need. `web_search`'s DDGS calls and robots.txt
  fetches are similarly moved off the event loop thread (a thread pool is
  enough there since they're I/O-, not CPU-, bound). Set either worker
  count to 1 to reproduce the old fully-serial behavior for debugging.
- **`ExactDedup` no longer opens/closes its persistence file per document**:
  it now keeps one line-buffered file handle open for the run. At the
  hundreds-of-thousands-of-docs-per-category scale this pipeline targets,
  the old open+write+close-per-document pattern was a real per-doc syscall
  cost.
- **Search backend**: DuckDuckGo's free endpoint is rate-limited and can be
  flaky at high volume. For serious scale, swap `web_search` in `server.py`
  to a paid API (Bing Search, Serper, Tavily, Brave Search API) — the tool
  signature (`query`, `max_results` → list of `{title, url, snippet}`) is
  the only contract the agent depends on.
- **crawl4ai is heavier than the plain httpx path**: it's a real headless
  browser, so per-page latency and memory are both higher than a bare GET
  request, and the first call in a process pays browser-startup cost
  (subsequent calls reuse the same browser instance). If you're scraping
  mostly static, non-JS pages at very high volume and don't need
  `deep_crawl`, set `SCRAPER_HTML_BACKEND=httpx` to skip it entirely and
  keep the original lightweight path.
- **`deep_crawl` respects the same domain rules as everything else**:
  `SCRAPER_ALLOWED_DOMAINS`/`SCRAPER_BLOCKED_DOMAINS` and `robots.txt` are
  still enforced on every page it visits, not just the seed URL.
- **robots.txt / rate limiting / domain rules**: the server checks
  robots.txt per host and caps request rate to 1 per 2s per domain
  (`SCRAPER_MIN_HOST_INTERVAL`). Set `SCRAPER_ALLOWED_DOMAINS` and/or
  `SCRAPER_BLOCKED_DOMAINS` (comma-separated) for explicit allow/deny
  control independent of robots.txt — e.g. to hard-exclude paywalled news
  or social platforms whose ToS forbid scraping.
- **PDF OCR / image OCR / ASR are genuinely expensive**: OCR-ing a scanned
  PDF or transcribing a 40-minute video is orders of magnitude slower than
  extracting an HTML article. `transcribe_media`'s `max_duration_seconds`
  (default 3600) bounds worst-case cost per video; tune `--concurrency`
  down when these formats dominate a run, since each ASR call is CPU/GPU-
  bound rather than I/O-bound like a plain fetch.
- **Video transcripts favor captions over ASR** specifically because
  existing captions (human-written or the platform's own ASR) are both
  faster to retrieve and typically more accurate than re-transcribing
  audio locally with a small Whisper model. ASR is a fallback, not the
  default path, for any video that already has an English caption track.
- **`thinking` field in SFT mode**: raw scraped content doesn't contain a
  chain-of-thought trace, so `extract_sft_pair` leaves `"thinking": ""`
  (your `sft.py` training code already handles this by skipping the
  `<think>` block). If you want synthetic CoT instead, extend
  `extract_sft_pair`'s prompt to ask the model to think step-by-step and
  capture that as `thinking`.
- **Near-dup filter**: install `datasketch` for real MinHash+LSH dedup at
  scale (tens of thousands+ docs/category); without it, `NearDedup()`
  automatically falls back to a slower O(n) shingle-overlap comparison
  that's fine for smaller runs but not a substitute at millions of docs.
- **Cost/speed knob**: `--no-llm-judge` skips the Ollama quality-judging
  call per candidate document. `--concurrency N` controls how many
  extract+filter tasks run in parallel per query round — raise it for
  I/O-bound HTML/PDF-heavy runs, keep it low when ASR is doing real work.
- **This sandbox**: this code was built and functionally tested here
  (extractors verified against generated HTML/PDF/DOCX/PPTX/XLSX samples;
  full syntax-checked throughout) but not run end-to-end against live
  internet search/scrape traffic, since this environment has no general
  internet egress. Video/audio transcription in particular (`yt-dlp` +
  `faster-whisper`, plus the `ffmpeg` system dependency) needs a real
  end-to-end run in your own environment to confirm the audio pipeline
  works with your installed codecs. **The crawl4ai HTML backend and
  `deep_crawl` (`web_scraper_mcp/crawl4ai_backend.py`) are likewise
  syntax-checked against crawl4ai's documented API but not exercised
  against a live Playwright browser or real web traffic here** — after
  `pip install crawl4ai && crawl4ai-setup`, run `crawl4ai-doctor` and then
  `healthcheck()` to confirm the browser install actually works in your
  environment before relying on it for a production run. Run
  `healthcheck()` first in your environment to confirm which formats are
  wired up correctly.
- **`agent/public_sources.py`** (the Hugging Face/Kaggle top-up) is
  likewise syntax-checked but not exercised against a live Hub API or a
  real Kaggle download in this sandbox -- verify `discover_hf_datasets` /
  `stream_hf_dataset` against a real public dataset id, and
  `discover_kaggle_datasets` / `fetch_kaggle_dataset_rows` against real
  Kaggle credentials, before relying on it for a production run. The
  column-name heuristics in `row_to_record` cover common conventions
  (`text`/`content`/`prompt`+`answer`/etc.) but an unusual schema may need
  a small tweak to `_TEXT_COLUMNS`/`_PROMPT_COLUMNS`/`_ANSWER_COLUMNS`.
