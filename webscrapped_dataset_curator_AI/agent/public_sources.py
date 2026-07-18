"""
public_sources.py

Pulls rows from public dataset hubs (Hugging Face Hub, Kaggle) instead of
live web search + scrape. This is the "faster top-up" path: no robots.txt,
no rate limiting, no HTML noise -- just structured rows that already exist
as public datasets, normalized into the same shape `dataset_agent.py`
already knows how to quality-filter, LLM-judge, optionally turn into SFT
pairs (the agent's "built-in AI"), dedup, and shard-write.

Every heavy/optional dependency (`datasets`, `huggingface_hub`, `kaggle`) is
imported lazily, same convention as web_scraper_mcp/extractors.py, so
missing one doesn't break the other or the rest of the pipeline -- it just
degrades that one source to a clear error.

Public contract, mirroring extract_content()'s shape so downstream code
never has to branch on "did this come from the web or from a dataset hub":

    {
        "title": str | None,
        "text": str,
        "author": None,
        "date": None,
        "content_type": "dataset_row",
        "url": "hf://<dataset_id>#<row_idx>" | "kaggle://<ref>/<file>#<row_idx>",
        "error": str | None,
        "extra": {"source": "huggingface"|"kaggle", "dataset": str,
                   "columns": [...],
                   # present only when the row already looks like an
                   # instruction/response pair -- lets the caller skip the
                   # LLM "invent a Q/A" step and use the hub's own labels:
                   "prompt": str | None, "answer": str | None},
    }
"""

from __future__ import annotations

import glob
import logging
import os
import tempfile
from typing import Iterator, Optional

log = logging.getLogger("dataset_agent.public_sources")

# ---------------------------------------------------------------------------
# Column-name heuristics for turning an arbitrary dataset row into text
# ---------------------------------------------------------------------------

_TEXT_COLUMNS = (
    "text", "content", "document", "article", "body", "passage",
    "abstract", "sentence", "description",
)
_PROMPT_COLUMNS = ("prompt", "question", "instruction", "input", "query")
_ANSWER_COLUMNS = ("answer", "response", "output", "completion", "solution", "answers")
_CODE_COLUMNS = ("code", "solution_code", "func_code", "program")


def _first_str(row: dict, candidates) -> Optional[str]:
    for key in candidates:
        for col in row:
            if col.lower() == key and isinstance(row[col], str) and row[col].strip():
                return row[col].strip()
    return None


def row_to_record(row: dict, source_label: str, dataset_id: str, ref: str) -> dict:
    """Normalize one raw dict from a HF/Kaggle dataset into the shared
    extract_content-shaped record. Prefers an explicit prompt/answer pair
    (the dataset author's own labels are more trustworthy than an LLM
    guessing one), falls back to a generic text column, and as a last
    resort concatenates every string field so nothing usable is dropped
    silently."""
    prompt = _first_str(row, _PROMPT_COLUMNS)
    answer = _first_str(row, _ANSWER_COLUMNS) or _first_str(row, _CODE_COLUMNS)

    text = _first_str(row, _TEXT_COLUMNS)
    if not text:
        if prompt and answer:
            text = f"{prompt}\n\n{answer}"
        else:
            # Last resort: join every non-trivial string field. Better than
            # silently dropping rows whose schema doesn't match any known
            # convention -- the quality filter downstream will reject it if
            # it's actually junk.
            parts = [v.strip() for v in row.values() if isinstance(v, str) and v.strip()]
            text = "\n\n".join(parts)

    return {
        "title": None,
        "text": text or "",
        "author": None,
        "date": None,
        "content_type": "dataset_row",
        "url": ref,
        "error": None if text else "row had no extractable string content",
        "extra": {
            "source": source_label,
            "dataset": dataset_id,
            "columns": list(row.keys()),
            "prompt": prompt,
            "answer": answer,
        },
    }


# ---------------------------------------------------------------------------
# Hugging Face Hub
# ---------------------------------------------------------------------------

def discover_hf_datasets(query: str, limit: int = 5) -> list:
    """Search the Hugging Face Hub for dataset ids matching a query
    (e.g. a category name like 'math' or 'code'). Returns [] (with a
    logged warning) if huggingface_hub isn't installed or the API call
    fails -- callers should treat that as 'no datasets discovered', not a
    fatal error."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        log.warning("huggingface_hub not installed -- `pip install huggingface_hub datasets` "
                     "to enable Hugging Face as a public data source")
        return []
    try:
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        results = api.list_datasets(search=query, limit=limit, sort="downloads", direction=-1)
        return [d.id for d in results]
    except Exception as e:
        log.warning(f"[hf] dataset search failed for {query!r}: {e}")
        return []


def stream_hf_dataset(dataset_id: str, max_rows: int = 200, split: Optional[str] = None,
                       config: Optional[str] = None) -> Iterator[dict]:
    """Yields normalized records from a Hugging Face dataset via streaming
    mode (no full download to disk, no waiting for the whole dataset to
    materialize). Stops after max_rows or when the split is exhausted,
    whichever comes first. Any failure (gated dataset needing HF_TOKEN,
    dataset doesn't exist, no `datasets` package, unknown split/config)
    yields a single {"error": ...} record and returns, rather than raising
    -- consistent with how extract_content reports failures to the agent
    loop, so one bad dataset id doesn't kill a whole category's run."""
    try:
        from datasets import load_dataset
    except ImportError:
        yield {"title": None, "text": "", "author": None, "date": None,
               "content_type": "dataset_row", "url": f"hf://{dataset_id}",
               "error": "`datasets` package not installed -- pip install datasets",
               "extra": {"source": "huggingface", "dataset": dataset_id}}
        return

    token = os.environ.get("HF_TOKEN")
    tried_splits = [split] if split else ["train", "test", "validation"]
    ds = None
    last_err = None
    for cfg_attempt in ([config] if config else [None]):
        for s in tried_splits:
            try:
                ds = load_dataset(dataset_id, cfg_attempt, split=s, streaming=True, token=token)
                split = s
                break
            except Exception as e:
                last_err = e
                continue
        if ds is not None:
            break

    if ds is None:
        yield {"title": None, "text": "", "author": None, "date": None,
               "content_type": "dataset_row", "url": f"hf://{dataset_id}",
               "error": f"could not load any split of {dataset_id}: {last_err}",
               "extra": {"source": "huggingface", "dataset": dataset_id}}
        return

    count = 0
    for row in ds:
        if count >= max_rows:
            break
        if not isinstance(row, dict):
            continue
        ref = f"hf://{dataset_id}#{split}:{count}"
        yield row_to_record(row, "huggingface", dataset_id, ref)
        count += 1


# ---------------------------------------------------------------------------
# Kaggle
# ---------------------------------------------------------------------------

def discover_kaggle_datasets(query: str, limit: int = 5) -> list:
    """Search Kaggle for dataset refs (owner/dataset-slug) matching a
    query. Requires Kaggle API credentials (KAGGLE_USERNAME + KAGGLE_KEY
    env vars, or ~/.kaggle/kaggle.json) -- returns [] with a logged
    warning if the `kaggle` package isn't installed or auth isn't
    configured, same graceful-degradation pattern as the HF path."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        log.warning("kaggle package not installed -- `pip install kaggle` and set "
                     "KAGGLE_USERNAME/KAGGLE_KEY to enable Kaggle as a public data source")
        return []
    try:
        api = KaggleApi()
        api.authenticate()
        results = api.dataset_list(search=query)
        return [d.ref for d in results[:limit]]
    except Exception as e:
        log.warning(f"[kaggle] dataset search failed for {query!r} "
                     f"(check KAGGLE_USERNAME/KAGGLE_KEY): {e}")
        return []


_TABULAR_EXTS = (".csv", ".tsv", ".json", ".jsonl")


def fetch_kaggle_dataset_rows(dataset_ref: str, max_rows: int = 200) -> Iterator[dict]:
    """Downloads a Kaggle dataset (public metadata + files, requires
    credentials) into a temp dir, then reads whatever tabular/text files it
    contains, yielding normalized records row-by-row up to max_rows total
    across all files in the dataset. Falls back to reading plain .txt files
    directly (one record per file) for datasets that are just loose text
    files rather than CSV/JSON tables."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        yield {"title": None, "text": "", "author": None, "date": None,
               "content_type": "dataset_row", "url": f"kaggle://{dataset_ref}",
               "error": "kaggle package not installed -- pip install kaggle",
               "extra": {"source": "kaggle", "dataset": dataset_ref}}
        return

    try:
        import pandas as pd
    except ImportError:
        yield {"title": None, "text": "", "author": None, "date": None,
               "content_type": "dataset_row", "url": f"kaggle://{dataset_ref}",
               "error": "pandas not installed -- required to read Kaggle CSV/JSON files",
               "extra": {"source": "kaggle", "dataset": dataset_ref}}
        return

    tmp_dir = tempfile.mkdtemp(prefix="kaggle_ds_")
    try:
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(dataset_ref, path=tmp_dir, unzip=True, quiet=True)
    except Exception as e:
        yield {"title": None, "text": "", "author": None, "date": None,
               "content_type": "dataset_row", "url": f"kaggle://{dataset_ref}",
               "error": f"download failed (check KAGGLE_USERNAME/KAGGLE_KEY and dataset "
                        f"visibility): {e}",
               "extra": {"source": "kaggle", "dataset": dataset_ref}}
        return

    files = sorted(glob.glob(os.path.join(tmp_dir, "**", "*"), recursive=True))
    count = 0

    for fpath in files:
        if count >= max_rows or not os.path.isfile(fpath):
            continue
        ext = os.path.splitext(fpath)[1].lower()
        rel = os.path.relpath(fpath, tmp_dir)
        try:
            if ext == ".csv":
                df_iter = pd.read_csv(fpath, chunksize=200, on_bad_lines="skip")
            elif ext == ".tsv":
                df_iter = pd.read_csv(fpath, sep="\t", chunksize=200, on_bad_lines="skip")
            elif ext == ".jsonl":
                df_iter = pd.read_json(fpath, lines=True, chunksize=200)
            elif ext == ".json":
                df_iter = [pd.read_json(fpath)]
            elif ext == ".txt":
                with open(fpath, "r", errors="ignore") as f:
                    text = f.read()
                yield row_to_record({"text": text}, "kaggle", dataset_ref,
                                     f"kaggle://{dataset_ref}/{rel}")
                count += 1
                continue
            else:
                continue  # skip images/binaries/etc. inside the dataset archive
        except Exception as e:
            log.warning(f"[kaggle] failed reading {rel} from {dataset_ref}: {e}")
            continue

        for chunk in df_iter:
            for i, row in chunk.iterrows():
                if count >= max_rows:
                    break
                record = row_to_record(row.dropna().to_dict(), "kaggle", dataset_ref,
                                        f"kaggle://{dataset_ref}/{rel}#{i}")
                yield record
                count += 1
            if count >= max_rows:
                break
