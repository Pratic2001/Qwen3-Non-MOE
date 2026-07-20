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
# NOTE: "problem" and "task" were added after HuggingFaceH4/MATH-500 (whose
# real columns are problem/solution/answer) was observed being rejected --
# without "problem" here, prompt comes back None, the row never counts as a
# labeled pair, and it silently falls back to the much stricter prose filter
# regardless of how short-but-valid the actual Q&A content is.
_PROMPT_COLUMNS = ("prompt", "question", "instruction", "input", "query", "problem", "task")
_ANSWER_COLUMNS = ("answer", "response", "output", "completion", "solution", "answers")
_CODE_COLUMNS = ("code", "solution_code", "func_code", "program")

# Chat-format columns hold the whole exchange as a list of turn dicts --
# ShareGPT-style [{"from": "human", "value": "..."}, {"from": "gpt", ...}]
# or OpenAI-style [{"role": "user", "content": "..."}, {"role": "assistant",
# ...}]. These are common across instruct datasets but NEVER match
# _first_str (they're lists, not strings), so without explicit handling
# every row from a conversational dataset falls through to the "join every
# string field" last resort below -- which drops the prompt/answer
# structure entirely and often produces empty or near-empty text, since the
# actual content lives one level down inside the list.
_CONVERSATION_COLUMNS = ("conversations", "messages", "conversation")
_TURN_ROLE_KEYS = ("from", "role")
_TURN_VALUE_KEYS = ("value", "content", "text")
_HUMAN_ROLE_VALUES = {"human", "user", "prompter"}
_ASSISTANT_ROLE_VALUES = {"gpt", "assistant", "bot", "model"}


def _first_str(row: dict, candidates) -> Optional[str]:
    for key in candidates:
        for col in row:
            if col.lower() == key and isinstance(row[col], str) and row[col].strip():
                return row[col].strip()
    return None


def _turn_role_and_value(turn: dict) -> tuple:
    role = None
    for k in _TURN_ROLE_KEYS:
        if isinstance(turn.get(k), str):
            role = turn[k].strip().lower()
            break
    value = None
    for k in _TURN_VALUE_KEYS:
        if isinstance(turn.get(k), str) and turn[k].strip():
            value = turn[k].strip()
            break
    return role, value


def _str_from_col(row: dict, col_name: Optional[str]) -> Optional[str]:
    """Case-insensitive lookup of one specific column name, returned only
    if it's actually a non-empty string. Used for LLM-inferred column
    hints, which name one exact column rather than a candidate list."""
    if not col_name:
        return None
    target = col_name.strip().lower()
    for col, val in row.items():
        if col.lower() == target and isinstance(val, str) and val.strip():
            return val.strip()
    return None


def _prompt_answer_from_conversation(row: dict, column_names=_CONVERSATION_COLUMNS) -> tuple:
    """Pull a (prompt, answer) pair from the first human->assistant turn of
    a chat-format column, if one exists. `column_names` defaults to the
    static candidate list but can be narrowed to one LLM-inferred column
    name. Returns (None, None) if no matching column is present or it
    doesn't parse as expected -- callers fall back to their existing
    heuristics in that case."""
    names = {c.lower() for c in column_names} if column_names else set()
    for col in row:
        if col.lower() not in names:
            continue
        turns = row[col]
        if not isinstance(turns, list) or not turns:
            continue
        prompt, answer = None, None
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            role, value = _turn_role_and_value(turn)
            if not value:
                continue
            if prompt is None and role in _HUMAN_ROLE_VALUES:
                prompt = value
            elif prompt is not None and answer is None and role in _ASSISTANT_ROLE_VALUES:
                answer = value
                break
        if prompt and answer:
            return prompt, answer
    return None, None


def row_to_record(row: dict, source_label: str, dataset_id: str, ref: str,
                   column_hint: Optional[dict] = None) -> dict:
    """Normalize one raw dict from a HF/Kaggle dataset into the shared
    extract_content-shaped record. Prefers an explicit prompt/answer pair
    (the dataset author's own labels are more trustworthy than an LLM
    guessing one) -- checking both flat string columns and chat-format
    conversation columns -- falls back to a generic text column, and as a
    last resort concatenates every string field so nothing usable is
    dropped silently.

    `column_hint`, if given, is a per-dataset schema mapping inferred once
    (via infer_column_mapping in dataset_agent.py, not called from here to
    avoid a circular import) and reused for every row of that dataset/
    config -- {"prompt_col": str|None, "answer_col": str|None,
    "text_col": str|None, "conversation_col": str|None}. It's tried FIRST
    since it's schema-aware (handles arbitrary column names the static
    candidate lists don't know about), but every hinted field is validated
    against the actual row (must resolve to real non-empty content) and
    anything that doesn't resolve falls through to the static heuristics
    below -- a wrong or stale hint degrades to the old behavior instead of
    silently dropping the row."""
    prompt = answer = text = None
    if column_hint:
        prompt = _str_from_col(row, column_hint.get("prompt_col"))
        answer = (_str_from_col(row, column_hint.get("answer_col"))
                  or _str_from_col(row, column_hint.get("code_col")))
        if not (prompt and answer) and column_hint.get("conversation_col"):
            conv_prompt, conv_answer = _prompt_answer_from_conversation(
                row, column_names=[column_hint["conversation_col"]])
            prompt = prompt or conv_prompt
            answer = answer or conv_answer
        text = _str_from_col(row, column_hint.get("text_col"))

    if not (prompt and answer):
        fb_prompt = _first_str(row, _PROMPT_COLUMNS)
        fb_answer = _first_str(row, _ANSWER_COLUMNS) or _first_str(row, _CODE_COLUMNS)
        prompt = prompt or fb_prompt
        answer = answer or fb_answer
    if not (prompt and answer):
        conv_prompt, conv_answer = _prompt_answer_from_conversation(row)
        prompt = prompt or conv_prompt
        answer = answer or conv_answer

    if not text:
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


def discover_hf_configs(dataset_id: str, token: Optional[str] = None) -> list:
    """Returns every config/subset name a HF dataset exposes (e.g.
    hendrycks_math -> ['algebra', 'counting_and_probability', 'geometry',
    ...]), so callers can pull from ALL partitions instead of guessing one.
    Falls back to [None] (meaning "just use the dataset's single/default
    config") if the `datasets` package is missing, the dataset genuinely
    has only one unnamed config, or the listing call fails for any reason
    -- config discovery is a best-effort enrichment, never a hard
    requirement, so a lookup failure degrades to old single-config
    behavior rather than dropping the dataset entirely."""
    try:
        from datasets import get_dataset_config_names
    except ImportError:
        return [None]
    try:
        configs = get_dataset_config_names(dataset_id, token=token)
        return list(configs) if configs else [None]
    except Exception as e:
        log.warning(f"[hf] could not list configs for {dataset_id}: {e} "
                     f"-- falling back to default config only")
        return [None]


def stream_hf_dataset(dataset_id: str, max_rows: int = 200, split: Optional[str] = None,
                       config: Optional[str] = None, column_mapper=None) -> Iterator[dict]:
    """Yields normalized records from a Hugging Face dataset via streaming
    mode (no full download to disk, no waiting for the whole dataset to
    materialize).

    Many datasets (hendrycks_math, mmlu, glue, ...) are split into several
    named configs/subsets, each a *separate* partition that
    `load_dataset(dataset_id, split=...)` alone will NOT enumerate -- you
    have to pass the config name explicitly, e.g.
    `load_dataset("EleutherAI/hendrycks_math", "algebra")`. If `config` is
    not given explicitly, this function first calls `discover_hf_configs()`
    to find every available partition and pulls from each of them
    (fair-sharing `max_rows` across configs) instead of silently loading
    only whichever one happens to work first -- otherwise a category like
    "math" would only ever see one subset's worth of rows even though
    several were available.

    Stops after max_rows (summed across all configs pulled) or when every
    config/split combination has been exhausted or failed, whichever comes
    first. Any total failure (gated dataset needing HF_TOKEN, dataset
    doesn't exist, no `datasets` package, no split/config combination
    loads) yields a single {"error": ...} record and returns, rather than
    raising -- consistent with how extract_content reports failures to the
    agent loop, so one bad dataset id doesn't kill a whole category's
    run.

    `column_mapper`, if given, is called as
    `column_mapper(dataset_id, config, columns, sample_row)` on the FIRST
    row of each config only (not every row -- that would mean one Ollama
    call per row, which doesn't scale) and should return a column_hint
    dict (or None) for row_to_record. The mapping is cached for the rest
    of that config's rows."""
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
    configs_to_try = [config] if config else discover_hf_configs(dataset_id, token)
    if config is None and configs_to_try != [None]:
        log.info(f"[hf] {dataset_id} has {len(configs_to_try)} config(s): {configs_to_try}")

    # Fair-share max_rows across every discovered config so e.g. algebra
    # doesn't eat the whole per-dataset budget before geometry/probability
    # ever get a turn -- each config gets an even slice; a slice a config
    # under-uses (exhausted early / failed to load) is left unclaimed
    # rather than silently reassigned, so the accounting stays simple.
    per_config_cap = max(1, max_rows // len(configs_to_try))

    count = 0
    any_success = False
    last_err = None
    for cfg_attempt in configs_to_try:
        if count >= max_rows:
            break
        ds = None
        used_split = None
        for s in tried_splits:
            try:
                ds = load_dataset(dataset_id, cfg_attempt, split=s, streaming=True, token=token)
                used_split = s
                break
            except Exception as e:
                last_err = e
                continue
        if ds is None:
            log.warning(f"[hf] could not load config {cfg_attempt!r} of {dataset_id}: {last_err}")
            continue

        any_success = True
        cfg_count = 0
        cfg_label = f":{cfg_attempt}" if cfg_attempt else ""
        column_hint = None
        hint_resolved = False
        for row in ds:
            if count >= max_rows or cfg_count >= per_config_cap:
                break
            if not isinstance(row, dict):
                continue
            if not hint_resolved:
                # One inference call per config, using the first row we
                # actually see -- not one per row.
                if column_mapper is not None:
                    try:
                        column_hint = column_mapper(dataset_id, cfg_attempt, list(row.keys()), row)
                    except Exception as e:
                        log.warning(f"[hf] column_mapper failed for {dataset_id}{cfg_label}: {e}")
                        column_hint = None
                hint_resolved = True
            ref = f"hf://{dataset_id}{cfg_label}#{used_split}:{cfg_count}"
            yield row_to_record(row, "huggingface", dataset_id, ref, column_hint=column_hint)
            count += 1
            cfg_count += 1

    if not any_success:
        yield {"title": None, "text": "", "author": None, "date": None,
               "content_type": "dataset_row", "url": f"hf://{dataset_id}",
               "error": f"could not load any split/config of {dataset_id}: {last_err}",
               "extra": {"source": "huggingface", "dataset": dataset_id}}


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


def fetch_kaggle_dataset_rows(dataset_ref: str, max_rows: int = 200, column_mapper=None) -> Iterator[dict]:
    """Downloads a Kaggle dataset (public metadata + files, requires
    credentials) into a temp dir, then reads whatever tabular/text files it
    contains, yielding normalized records row-by-row up to max_rows total
    across all files in the dataset. Falls back to reading plain .txt files
    directly (one record per file) for datasets that are just loose text
    files rather than CSV/JSON tables.

    `column_mapper`, if given, is called once per file (on its first row)
    the same way as in stream_hf_dataset -- see that docstring."""
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
                # Unlike .csv/.tsv/.jsonl above, a plain JSON array can't be
                # read incrementally with pandas -- pd.read_json(fpath) has
                # to parse the whole file into memory in one shot before
                # yielding a single row. That's normally fine (most Kaggle
                # JSON files are small), but an unusually large one (a
                # multi-GB JSON export, say) would spike memory regardless
                # of --concurrency or the MemoryGovernor in dataset_agent.py
                # -- neither can intervene mid-parse. Guard against that
                # case explicitly rather than silently OOMing on it.
                size_mb = os.path.getsize(fpath) / 1024**2
                max_json_mb = float(os.environ.get("KAGGLE_MAX_JSON_FILE_MB", "300"))
                if size_mb > max_json_mb:
                    log.warning(f"[kaggle] skipping {rel} from {dataset_ref}: "
                                f"{size_mb:.0f}MB plain .json file exceeds "
                                f"KAGGLE_MAX_JSON_FILE_MB={max_json_mb:.0f} -- it can't be "
                                f"read incrementally like .csv/.jsonl, so loading it would "
                                f"pull the whole thing into memory at once. Raise "
                                f"KAGGLE_MAX_JSON_FILE_MB if you have the RAM for it, or "
                                f"prefer datasets that ship .jsonl instead.")
                    continue
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

        file_hint = None
        file_hint_resolved = False
        for chunk in df_iter:
            for i, row in chunk.iterrows():
                if count >= max_rows:
                    break
                row_dict = row.dropna().to_dict()
                if not file_hint_resolved:
                    if column_mapper is not None:
                        try:
                            # Use the FULL row (not the dropna'd one used for
                            # the actual record) + the DataFrame's real
                            # column list. row.dropna() can silently omit
                            # whichever column happens to be null in this
                            # particular first row -- if that's the answer
                            # column, the mapper would never even see it as
                            # a candidate. NaN -> None so it's JSON-safe.
                            full_row = {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}
                            file_hint = column_mapper(dataset_ref, rel, list(chunk.columns), full_row)
                        except Exception as e:
                            log.warning(f"[kaggle] column_mapper failed for {dataset_ref}/{rel}: {e}")
                            file_hint = None
                    file_hint_resolved = True
                record = row_to_record(row_dict, "kaggle", dataset_ref,
                                        f"kaggle://{dataset_ref}/{rel}#{i}", column_hint=file_hint)
                yield record
                count += 1
            if count >= max_rows:
                break
