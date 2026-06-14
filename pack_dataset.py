#!/usr/bin/env python3
"""
pack_dataset.py

Tokenizes the JSONL shards produced by build_dataset.py (./data/<category>/*.jsonl)
using the tokenizer from train_tokenizer.py, and packs the resulting token IDs
into flat, memory-mapped .bin files for fast random-access reading during
training (nanoGPT / llm.c style).

Output layout:
    ./packed/train.bin   -- uint16 or uint32 token IDs, contiguous
    ./packed/val.bin      -- held-out validation slice
    ./packed/meta.json    -- dtype, vocab size, token counts, category breakdown

Each document is tokenized (with EOS appended by the tokenizer's
post-processor) and concatenated directly -- no padding. The training
dataloader later slices fixed-length windows out of this flat array.

Usage:
    python pack_dataset.py --data-dir ./data --tokenizer ./tokenizer --out-dir ./packed
    python pack_dataset.py --data-dir ./data --tokenizer ./tokenizer --val-fraction 0.0005 --workers 8
"""

import argparse
import glob
import json
import os
import time
from multiprocessing import Pool

import numpy as np
from tokenizers import Tokenizer


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

# Global, set once per worker process via initializer (avoids re-pickling
# the tokenizer for every chunk).
_TOKENIZER = None


def _init_worker(tokenizer_path: str):
    global _TOKENIZER
    _TOKENIZER = Tokenizer.from_file(tokenizer_path)


def _tokenize_shard(args):
    """Tokenize one JSONL shard, return (category, token_count, np.uint32 array path)."""
    shard_path, tmp_dir, idx = args
    global _TOKENIZER

    texts = []
    category = None
    with open(shard_path, "r", encoding="utf-8") as f:
        for line in f:
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
            texts.append(text)
            if category is None:
                category = rec.get("category", "unknown")

    if not texts:
        return category or "unknown", 0, None

    # Batch-encode for speed; each encoding already ends with <|endoftext|>
    # thanks to the tokenizer's post-processor.
    encodings = _TOKENIZER.encode_batch(texts)
    ids = []
    for enc in encodings:
        ids.extend(enc.ids)

    arr = np.array(ids, dtype=np.uint32)
    out_path = os.path.join(tmp_dir, f"chunk_{idx:06d}.npy")
    np.save(out_path, arr)
    return category or "unknown", len(arr), out_path


# ---------------------------------------------------------------------------
# Main packing logic
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Tokenize and pack JSONL corpus into memmap .bin files.")
    parser.add_argument("--data-dir", default="./data", help="Directory with JSONL shards (from build_dataset.py)")
    parser.add_argument("--tokenizer", default="./tokenizer", help="Directory containing tokenizer.json")
    parser.add_argument("--out-dir", default="./packed", help="Output directory for .bin files")
    parser.add_argument("--val-fraction", type=float, default=0.0005,
                         help="Fraction of tokens held out for validation (default 0.05%%)")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1),
                         help="Number of parallel tokenization workers")
    parser.add_argument("--shuffle-shards", action="store_true", default=True,
                         help="Shuffle shard order before splitting train/val (default on)")
    args = parser.parse_args()

    tokenizer_path = os.path.join(args.tokenizer, "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"{tokenizer_path} not found -- run train_tokenizer.py first.")

    # Determine dtype from vocab size: uint16 if vocab fits (<=65536),
    # else uint32. Qwen3-scale (~151K) needs uint32.
    tmp_tok = Tokenizer.from_file(tokenizer_path)
    vocab_size = tmp_tok.get_vocab_size()
    dtype = np.uint16 if vocab_size <= 65536 else np.uint32
    del tmp_tok
    print(f"Tokenizer vocab size: {vocab_size} -> packing as {dtype.__name__}")

    shard_paths = sorted(glob.glob(os.path.join(args.data_dir, "*", "*.jsonl")))
    if not shard_paths:
        raise FileNotFoundError(f"No .jsonl shards found under {args.data_dir}/<category>/")
    print(f"Found {len(shard_paths)} shard file(s)")

    if args.shuffle_shards:
        rng = np.random.default_rng(seed=42)
        shard_paths = list(shard_paths)
        rng.shuffle(shard_paths)

    os.makedirs(args.out_dir, exist_ok=True)
    tmp_dir = os.path.join(args.out_dir, "_tmp_chunks")
    os.makedirs(tmp_dir, exist_ok=True)

    # --- Stage 1: tokenize each shard in parallel, write intermediate .npy chunks ---
    t0 = time.time()
    work_items = [(p, tmp_dir, i) for i, p in enumerate(shard_paths)]

    category_token_counts = {}
    chunk_records = []  # (category, token_count, chunk_path) in original shard order

    print(f"Tokenizing {len(work_items)} shards with {args.workers} worker(s)...")
    with Pool(processes=args.workers, initializer=_init_worker, initargs=(tokenizer_path,)) as pool:
        for i, (category, n_tokens, chunk_path) in enumerate(pool.imap(_tokenize_shard, work_items)):
            chunk_records.append((category, n_tokens, chunk_path))
            category_token_counts[category] = category_token_counts.get(category, 0) + n_tokens
            if (i + 1) % max(1, len(work_items) // 20) == 0 or (i + 1) == len(work_items):
                elapsed = time.time() - t0
                done_tokens = sum(c[1] for c in chunk_records)
                print(f"  [{i+1}/{len(work_items)}] shards tokenized, "
                      f"{done_tokens:,} tokens so far ({elapsed:.1f}s)")

    total_tokens = sum(c[1] for c in chunk_records)
    print(f"\nTotal tokens: {total_tokens:,}")
    for cat, n in category_token_counts.items():
        pct = 100 * n / total_tokens if total_tokens else 0
        print(f"  {cat:12s}: {n:,} tokens ({pct:.1f}%)")

    if total_tokens == 0:
        raise RuntimeError("No tokens produced -- check input data.")

    # --- Stage 2: split chunks into train/val by token-count budget, write final .bin files ---
    val_token_budget = int(total_tokens * args.val_fraction)
    print(f"\nVal budget: {val_token_budget:,} tokens ({args.val_fraction*100:.3f}%)")

    train_path = os.path.join(args.out_dir, "train.bin")
    val_path = os.path.join(args.out_dir, "val.bin")

    # Allocate output memmaps with exact final sizes.
    train_arr = np.memmap(train_path, dtype=dtype, mode="w+", shape=(total_tokens - val_token_budget,))
    val_arr = np.memmap(val_path, dtype=dtype, mode="w+", shape=(val_token_budget,))

    train_ptr = 0
    val_ptr = 0
    val_remaining = val_token_budget

    for category, n_tokens, chunk_path in chunk_records:
        if n_tokens == 0:
            continue
        arr = np.load(chunk_path).astype(dtype, copy=False)

        if val_remaining > 0:
            take_val = min(val_remaining, n_tokens)
            val_arr[val_ptr:val_ptr + take_val] = arr[:take_val]
            val_ptr += take_val
            val_remaining -= take_val
            arr = arr[take_val:]

        if len(arr) > 0:
            train_arr[train_ptr:train_ptr + len(arr)] = arr
            train_ptr += len(arr)

        os.remove(chunk_path)

    train_arr.flush()
    val_arr.flush()
    os.rmdir(tmp_dir)

    print(f"\nWrote {train_ptr:,} tokens -> {train_path} "
          f"({train_ptr * dtype().itemsize / 1024**2:.1f} MB)")
    print(f"Wrote {val_ptr:,} tokens -> {val_path} "
          f"({val_ptr * dtype().itemsize / 1024**2:.1f} MB)")

    # --- Stage 3: write meta.json for the dataloader ---
    meta = {
        "vocab_size": vocab_size,
        "dtype": dtype.__name__,
        "train_tokens": int(train_ptr),
        "val_tokens": int(val_ptr),
        "total_tokens": int(total_tokens),
        "category_token_counts": category_token_counts,
        "tokenizer_dir": os.path.abspath(args.tokenizer),
    }
    meta_path = os.path.join(args.out_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s. Wrote {meta_path}")


if __name__ == "__main__":
    main()
