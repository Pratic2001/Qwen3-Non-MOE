#!/usr/bin/env python3
"""
pack_dataset.py

Tokenizes the JSONL shards produced by build_dataset.py (./data/<category>/*.jsonl)
using the tokenizer from train_tokenizer.py, and packs the resulting token IDs
into flat, memory-mapped .bin files for fast random-access reading during
training.

Output layout:
    ./packed/train.bin   -- uint16 or uint32 token IDs, contiguous
    ./packed/val.bin     -- held-out validation slice
    ./packed/meta.json   -- dtype, vocab size, token counts, category breakdown

Memory design
─────────────
OLD (memory-hogging):
    Worker loaded all texts from a 256 MB shard into a Python list,
    called encode_batch() on all of them at once, then built a giant
    ids list — 3-5 GB per worker × N workers.

NEW (streaming):
    Workers encode documents in mini-batches of --mini-batch size and
    write uint32 token IDs directly to a binary chunk file.
    Peak RAM per worker ≈ mini_batch × avg_doc_tokens × 4 bytes ≈ 30–80 MB.

    The main process copies each chunk into the output memmap in slices
    of --copy-chunk-mb bytes so it never loads a full chunk into RAM either.

Usage:
    python pack_dataset.py --data-dir ./data --tokenizer ./tokenizer
    python pack_dataset.py --data-dir ./data --tokenizer ./tokenizer \\
        --workers 4 --mini-batch 32 --copy-chunk-mb 64
"""

import argparse
import glob
import json
import os
import struct
import time
from multiprocessing import Pool

import numpy as np
from tokenizers import Tokenizer


# ---------------------------------------------------------------------------
# Worker  (runs in subprocess — one per shard)
# ---------------------------------------------------------------------------

_TOKENIZER = None


def _init_worker(tokenizer_path: str):
    global _TOKENIZER
    _TOKENIZER = Tokenizer.from_file(tokenizer_path)


def _tokenize_shard(job):
    """
    Stream documents from one JSONL shard, encode in mini-batches, write
    raw uint32 token IDs to a binary chunk file.

    Chunk file: plain sequence of little-endian uint32 integers.
    Returns (category, total_token_count, chunk_path).
    """
    shard_path, tmp_dir, idx, mini_batch = job
    global _TOKENIZER

    out_path    = os.path.join(tmp_dir, f"chunk_{idx:06d}.bin")
    category    = None
    total       = 0
    batch_texts = []

    def _flush(texts, fh):
        if not texts:
            return 0
        encodings = _TOKENIZER.encode_batch(texts)
        n = 0
        for enc in encodings:
            if enc.ids:
                fh.write(struct.pack(f"<{len(enc.ids)}I", *enc.ids))
                n += len(enc.ids)
        return n

    try:
        with open(shard_path, "r", encoding="utf-8") as fin, \
             open(out_path, "wb") as fout:

            for line in fin:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = rec.get("text")
                if not text:
                    continue
                if category is None:
                    category = rec.get("category", "unknown")

                batch_texts.append(text)
                if len(batch_texts) >= mini_batch:
                    total      += _flush(batch_texts, fout)
                    batch_texts = []

            total += _flush(batch_texts, fout)   # flush remainder

    except Exception as e:
        print(f"\n[worker {idx}] error: {e}")

    if total == 0:
        if os.path.exists(out_path):
            os.remove(out_path)
        return category or "unknown", 0, None

    return category or "unknown", total, out_path


# ---------------------------------------------------------------------------
# Streaming chunk-to-memmap copy  (no full chunk loaded at once)
# ---------------------------------------------------------------------------

def _copy_chunk(chunk_path: str,
                val_arr:   np.ndarray, val_ptr:   int, val_remaining: int,
                train_arr: np.ndarray, train_ptr: int,
                out_dtype, slice_tokens: int):
    """
    Read chunk_path in slices of `slice_tokens` uint32s.
    Fill val_arr first (up to val_remaining tokens), then train_arr.
    Deletes chunk_path when done.
    Returns (new_val_ptr, new_val_remaining, new_train_ptr).
    """
    bytes_per_tok = 4   # uint32

    with open(chunk_path, "rb") as fh:
        while True:
            raw = fh.read(slice_tokens * bytes_per_tok)
            if not raw:
                break
            arr = np.frombuffer(raw, dtype=np.uint32)

            # Split between val and train within this slice
            if val_remaining > 0:
                take_val = min(val_remaining, len(arr))
                slice_v  = arr[:take_val].astype(out_dtype, copy=False)
                val_arr[val_ptr : val_ptr + take_val] = slice_v
                val_ptr       += take_val
                val_remaining -= take_val
                arr = arr[take_val:]

            if len(arr) > 0:
                slice_t = arr.astype(out_dtype, copy=False)
                train_arr[train_ptr : train_ptr + len(slice_t)] = slice_t
                train_ptr += len(slice_t)

    os.remove(chunk_path)
    return val_ptr, val_remaining, train_ptr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Tokenize and pack JSONL corpus into memmap .bin files."
    )
    p.add_argument("--data-dir",       default="./data")
    p.add_argument("--tokenizer",      default="./tokenizer")
    p.add_argument("--out-dir",        default="./packed")
    p.add_argument("--val-fraction",   type=float, default=0.0005,
                   help="Fraction of tokens for validation (default 0.05%%)")
    p.add_argument("--workers",        type=int,
                   default=max(1, (os.cpu_count() or 2) - 1),
                   help="Parallel tokenization workers")
    p.add_argument("--mini-batch",     type=int, default=64,
                   help="Docs encoded per mini-batch in each worker. "
                        "Lower = less RAM per worker. Default 64.")
    p.add_argument("--copy-chunk-mb",  type=int, default=64,
                   help="Slice size (MB) when copying chunks to final mmap. "
                        "Controls main-process peak RAM. Default 64 MB.")
    p.add_argument("--shuffle-shards", action="store_true", default=True)
    args = p.parse_args()

    tokenizer_path = os.path.join(args.tokenizer, "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(
            f"{tokenizer_path} not found — run train_tokenizer.py first."
        )

    tmp_tok    = Tokenizer.from_file(tokenizer_path)
    vocab_size = tmp_tok.get_vocab_size()
    out_dtype  = np.uint16 if vocab_size <= 65536 else np.uint32
    del tmp_tok
    print(f"Tokenizer vocab size : {vocab_size}  ->  packing as {out_dtype.__name__}")

    shard_paths = sorted(glob.glob(os.path.join(args.data_dir, "*", "*.jsonl")))
    if not shard_paths:
        raise FileNotFoundError(
            f"No .jsonl shards found under {args.data_dir}/<category>/. "
            f"Run build_dataset.py first."
        )
    print(f"Found {len(shard_paths)} shard file(s)")

    if args.shuffle_shards:
        rng = np.random.default_rng(seed=42)
        shard_paths = list(shard_paths)
        rng.shuffle(shard_paths)

    os.makedirs(args.out_dir, exist_ok=True)
    tmp_dir = os.path.join(args.out_dir, "_tmp_chunks")
    os.makedirs(tmp_dir, exist_ok=True)

    slice_tokens = max(1, (args.copy_chunk_mb * 1024 * 1024) // 4)

    # Print memory estimates so the user can tune before a long run
    avg_doc_tokens    = 600
    peak_per_worker   = args.mini_batch * avg_doc_tokens * 4 / 1024**2
    total_worker_ram  = peak_per_worker * args.workers
    print(f"\nWorkers              : {args.workers}")
    print(f"Mini-batch / worker  : {args.mini_batch} docs  "
          f"(~{peak_per_worker:.0f} MB peak RAM per worker)")
    print(f"Est. total worker RAM: ~{total_worker_ram:.0f} MB  "
          f"(was several GB with old design)")
    print(f"Copy slice           : {args.copy_chunk_mb} MB\n")

    # ---------------------------------------------------------------
    # Stage 1 — tokenize shards in parallel, stream-write chunk files
    # ---------------------------------------------------------------
    t0         = time.time()
    work_items = [(path, tmp_dir, i, args.mini_batch)
                  for i, path in enumerate(shard_paths)]

    cat_counts:    dict = {}
    chunk_records: list = []

    print(f"Stage 1 — tokenizing {len(work_items)} shard(s) …")
    with Pool(
        processes=args.workers,
        initializer=_init_worker,
        initargs=(tokenizer_path,),
    ) as pool:
        for i, (cat, n_tok, chunk_path) in enumerate(
            pool.imap(_tokenize_shard, work_items)
        ):
            chunk_records.append((cat, n_tok, chunk_path))
            cat_counts[cat] = cat_counts.get(cat, 0) + n_tok

            report_every = max(1, len(work_items) // 20)
            if (i + 1) % report_every == 0 or (i + 1) == len(work_items):
                done = sum(r[1] for r in chunk_records)
                print(f"  [{i+1:4d}/{len(work_items)}]  "
                      f"{done:,} tokens  ({time.time()-t0:.1f}s)")

    total_tokens = sum(r[1] for r in chunk_records)
    print(f"\nTotal tokens : {total_tokens:,}")
    for cat, n in sorted(cat_counts.items()):
        pct = 100 * n / total_tokens if total_tokens else 0
        print(f"  {cat:14s}: {n:,}  ({pct:.1f}%)")

    if total_tokens == 0:
        raise RuntimeError(
            "No tokens produced — check that JSONL records have a 'text' field."
        )

    # ---------------------------------------------------------------
    # Stage 2 — allocate output mmaps, copy chunk files slice-by-slice
    # ---------------------------------------------------------------
    val_budget = int(total_tokens * args.val_fraction)
    n_train    = total_tokens - val_budget
    print(f"\nStage 2 — writing train.bin ({n_train:,} tok) "
          f"and val.bin ({val_budget:,} tok) …")

    train_path = os.path.join(args.out_dir, "train.bin")
    val_path   = os.path.join(args.out_dir, "val.bin")

    train_arr = np.memmap(train_path, dtype=out_dtype, mode="w+", shape=(n_train,))
    val_arr   = np.memmap(val_path,   dtype=out_dtype, mode="w+",
                          shape=(max(1, val_budget),))

    train_ptr     = 0
    val_ptr       = 0
    val_remaining = val_budget

    for cat, n_tokens, chunk_path in chunk_records:
        if n_tokens == 0 or chunk_path is None:
            continue
        val_ptr, val_remaining, train_ptr = _copy_chunk(
            chunk_path,
            val_arr,   val_ptr,   val_remaining,
            train_arr, train_ptr,
            out_dtype, slice_tokens,
        )

    train_arr.flush()
    val_arr.flush()

    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass

    # ---------------------------------------------------------------
    # Stage 3 — meta.json
    # ---------------------------------------------------------------
    meta = {
        "vocab_size":            vocab_size,
        "dtype":                 out_dtype.__name__,
        "train_tokens":          int(train_ptr),
        "val_tokens":            int(val_ptr),
        "total_tokens":          int(total_tokens),
        "category_token_counts": cat_counts,
        "tokenizer_dir":         os.path.abspath(args.tokenizer),
    }
    meta_path = os.path.join(args.out_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    elapsed  = time.time() - t0
    sz_train = train_ptr * out_dtype().itemsize / 1024**2
    sz_val   = val_ptr   * out_dtype().itemsize / 1024**2
    print(f"\nWrote {train_ptr:,} tokens -> train.bin  ({sz_train:.1f} MB)")
    print(f"Wrote {val_ptr:,}   tokens -> val.bin    ({sz_val:.1f} MB)")
    print(f"Done in {elapsed:.1f}s   Meta: {meta_path}")


if __name__ == "__main__":
    main()
