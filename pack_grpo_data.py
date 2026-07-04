#!/usr/bin/env python3
"""
pack_grpo_data.py

Stage 0 of GRPO: read raw JSONL GRPO records (from download_grpo_data.py),
apply a *single-turn* ChatML + answer template, tokenise, and write packed
memmap .bin files in the same format `pack_sft_data.py` writes. This makes
the cache directly readable by train_grpo.py — `GRPOPromptDataset._init_from_packed`
opens it via the same `SFTDataset` manifest convention.

The key difference from SFT packing:
    - GRPO records only carry `{prompt, answer}` (no `thinking`).
    - The "assistant" turn we pack is therefore *just the canonical answer
      text*. The model is expected to generate its own reasoning at rollout
      time. The loss in the GRPO objective is computed against on-policy
      completions, not against this answer turn — the packed answer is only
      used by `GRPOPromptDataset` to recover the ground-truth string for
      reward scoring.
    - The pack step still writes a contiguous mask=1 region followed by an
      EOS separator, identical in shape to SFT, so `_scan_boundaries` in
      train_grpo.py walks the file the same way.

This script does NOT touch the model, torch.distributed, or any GPU code.
It is pure CPU/IO work and is meant to be run once (or in parallel shards)
before training starts.

Template produced for each record:
    user
    {prompt}
    assistant
    {answer}

Output layout (per worker) — *identical* to pack_sft_data.py output, so
train_grpo.py's default `--cache_dir ./sft_packed` works against either:
    <cache_dir>/sft_train_tokens.w{worker}-of-{num_workers}.bin
    <cache_dir>/sft_train_mask.w{worker}-of-{num_workers}.bin
    <cache_dir>/sft_val_tokens.w{worker}-of-{num_workers}.bin
    <cache_dir>/sft_val_mask.w{worker}-of-{num_workers}.bin
    <cache_dir>/sft_manifest.w{worker}-of-{num_workers}.json

Usage:
    # Single process, packs every shard
    python pack_grpo_data.py --data-dir ./grpo_data --tokenizer ./tokenizer \\
        --cache-dir ./grpo_packed

    # Split across N parallel processes (each packs a disjoint shard subset)
    python pack_grpo_data.py --worker 0 --num-workers 4 \\
        --data-dir ./grpo_data --tokenizer ./tokenizer --cache-dir ./grpo_packed

After packing, point train_grpo.py at the new cache:

    # Default discovery (memmap + JSONL for ground-truth recovery)
    python train_grpo.py --checkpoint ./sft_checkpoints/latest.pt \\
        --tokenizer ./tokenizer \\
        --cache_dir ./grpo_packed --data_dir ./grpo_data \\
        --num_generations 8 --max_new_tokens 512 --max_steps 500

    # --prompt_override flag: skip memmap entirely and consume a flat
    # JSONL of {prompt, answer}. Useful for small eval sets or fast iteration.
    python train_grpo.py --checkpoint ./sft_checkpoints/latest.pt \\
        --tokenizer ./tokenizer \\
        --prompt_override ./eval_prompts.jsonl \\
        --num_generations 8 --max_steps 100
"""

import argparse
import glob
import json
import os
import time
from typing import List, Optional, Tuple

import numpy as np
from tokenizers import Tokenizer


# ---------------------------------------------------------------------------
# Tokenizer helpers
# ---------------------------------------------------------------------------

def load_tokenizer(tokenizer_dir: str) -> Tokenizer:
    path = os.path.join(tokenizer_dir, "tokenizer.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"tokenizer.json not found in {tokenizer_dir}")
    return Tokenizer.from_file(path)


def get_special_token_id(tok: Tokenizer, token: str) -> int:
    tid = tok.token_to_id(token)
    if tid is None:
        raise ValueError(
            f"Token {token!r} not in tokenizer vocab. "
            f"Was the tokenizer trained with train_tokenizer.py?"
        )
    return tid


# ---------------------------------------------------------------------------
# Chat template formatting + tokenisation
# ---------------------------------------------------------------------------

def format_and_tokenise(
    record: dict,
    tokenizer: Tokenizer,
    max_len: int = 2048,
) -> Optional[Tuple[List[int], List[int]]]:
    """
    Format one GRPO record into token ids + loss mask.

    Returns (input_ids, loss_mask) where loss_mask[i] = 1 means position i
    is part of the (canonical-answer) assistant turn, 0 means it is part of
    the prompt or the EOS separator.

    NOTE: in the GRPO loop, the loss mask is *not* used for SFT-style cross-
    entropy; the packed assistant turn is purely a marker so that
    `_scan_boundaries` in train_grpo.py can recover prompt / answer regions
    and `_load_answer_strings` can recover the canonical ground-truth text
    in deterministic order.

    Returns None if the formatted example exceeds max_len tokens.
    """
    prompt = record.get("prompt", "").strip()
    answer = record.get("answer", "").strip()

    if not prompt or not answer:
        return None

    # ---- user turn (prompt — masked out of mask)
    user_text = f"user\n{prompt}\n"

    # ---- assistant turn (canonical answer only — mask = 1 region)
    # No <think> block: GRPO expects the model to reason at rollout time.
    asst_text = f"assistant\n{answer}\n"

    user_ids = tokenizer.encode(user_text, add_special_tokens=False).ids
    asst_ids = tokenizer.encode(asst_text, add_special_tokens=False).ids

    # Truncate if necessary, preserving the answer tail.
    total = len(user_ids) + len(asst_ids)
    if total > max_len:
        budget = max_len - len(user_ids)
        if budget < 32:
            user_ids = user_ids[: max_len - min(32, len(asst_ids))]
            asst_ids = asst_ids[:32]
        else:
            asst_ids = asst_ids[:budget]

    input_ids = user_ids + asst_ids
    loss_mask = [0] * len(user_ids) + [1] * len(asst_ids)

    if len(input_ids) < 4:
        return None

    return input_ids, loss_mask


def _safe_load_line(line: str) -> Optional[dict]:
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def _is_val(record_idx: int, val_fraction: float) -> bool:
    """Deterministic train/val split: every Nth record goes to val."""
    if val_fraction <= 0:
        return False
    period = max(1, round(1.0 / val_fraction))
    return (record_idx % period) == 0


# ---------------------------------------------------------------------------
# Worker shard selection
# ---------------------------------------------------------------------------

def select_shards_for_worker(
    data_dir: str, worker: int, num_workers: int,
) -> List[str]:
    """
    Discover every JSONL shard under <data_dir>/<category>/*.jsonl and
    return the subset assigned to this worker (file index modulo
    num_workers). Sorting first makes the assignment deterministic across
    processes/machines — same convention as pack_sft_data.py, which is
    required so that train_grpo.py's `_load_answer_strings` walks the
    JSONL in the same order the packer processed the records.
    """
    all_shards = sorted(glob.glob(os.path.join(data_dir, "*", "*.jsonl")))
    if not all_shards:
        raise FileNotFoundError(
            f"No .jsonl shards found under {data_dir}/<category>/. "
            f"Run download_grpo_data.py first."
        )
    my_shards = all_shards[worker::num_workers]
    return my_shards


# ---------------------------------------------------------------------------
# Pack: stream JSONL -> tokenise -> write memmap .bin files for this worker
# ---------------------------------------------------------------------------

def pack_worker_shard(
    data_dir: str,
    tokenizer: Tokenizer,
    cache_dir: str,
    max_len_per_example: int,
    val_fraction: float,
    worker: int,
    num_workers: int,
    vocab_size: Optional[int] = None,
) -> dict:
    """
    Stream every JSONL record assigned to this worker, tokenise it, and
    write to this worker's memmap files in the same format as
    pack_sft_data.py. Peak RAM = one record + a couple of small write
    buffers, regardless of dataset size.

    Returns the manifest dict that was also written to disk.
    """
    os.makedirs(cache_dir, exist_ok=True)

    shard_paths = select_shards_for_worker(data_dir, worker, num_workers)
    print(f"[pack worker {worker + 1}/{num_workers}] {len(shard_paths)} shard(s) "
          f"assigned out of {len(glob.glob(os.path.join(data_dir, '*', '*.jsonl')))} total")
    if not shard_paths:
        print(f"[pack worker {worker + 1}/{num_workers}] no shards assigned "
              f"(num_workers > number of input files) — writing empty output")

    eos_id     = get_special_token_id(tokenizer, "")
    vocab_size = vocab_size or tokenizer.get_vocab_size()
    dtype_t    = np.uint16 if vocab_size <= 65536 else np.uint32
    dtype_m    = np.uint8

    # Filenames match pack_sft_data.py exactly so train_grpo.py's default
    # --cache_dir discovery picks them up unchanged.
    suffix = f"w{worker}-of-{num_workers}"
    train_tok_path  = os.path.join(cache_dir, f"sft_train_tokens.{suffix}.bin")
    train_mask_path = os.path.join(cache_dir, f"sft_train_mask.{suffix}.bin")
    val_tok_path    = os.path.join(cache_dir, f"sft_val_tokens.{suffix}.bin")
    val_mask_path   = os.path.join(cache_dir, f"sft_val_mask.{suffix}.bin")
    manifest_path   = os.path.join(cache_dir, f"sft_manifest.{suffix}.json")

    print(f"[pack worker {worker + 1}/{num_workers}] this streams records "
          f"one-by-one (constant RAM)")

    # ---- first pass: count tokens so we can pre-allocate mmaps
    t0          = time.time()
    total_train = 0
    total_val   = 0
    n_records   = 0

    for path in shard_paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                rec = _safe_load_line(line)
                if rec is None:
                    continue
                result = format_and_tokenise(rec, tokenizer, max_len=max_len_per_example)
                if result is None:
                    continue
                ids, _ = result
                n_tok = len(ids) + 1  # +1 for EOS separator
                if _is_val(n_records, val_fraction):
                    total_val += n_tok
                else:
                    total_train += n_tok
                n_records += 1

    print(f"[pack worker {worker + 1}/{num_workers}] counted {n_records:,} records "
          f"in {time.time()-t0:.1f}s")
    print(f"[pack worker {worker + 1}/{num_workers}] train tokens: {total_train:,}  "
          f"val tokens: {total_val:,}")

    # ---- allocate memmap files on disk (no RAM)
    np.memmap(train_tok_path,  dtype=dtype_t, mode="w+", shape=(total_train,))
    np.memmap(train_mask_path, dtype=dtype_m, mode="w+", shape=(total_train,))
    np.memmap(val_tok_path,    dtype=dtype_t, mode="w+", shape=(total_val,))
    np.memmap(val_mask_path,   dtype=dtype_m, mode="w+", shape=(total_val,))

    # Re-open for writing
    train_tok  = np.memmap(train_tok_path,  dtype=dtype_t, mode="r+")
    train_mask = np.memmap(train_mask_path, dtype=dtype_m, mode="r+")
    val_tok    = np.memmap(val_tok_path,    dtype=dtype_t, mode="r+")
    val_mask   = np.memmap(val_mask_path,   dtype=dtype_m, mode="r+")

    # ---- second pass: write tokens + masks directly into the mmaps
    train_ptr  = 0
    val_ptr    = 0
    n_records  = 0
    last_print = time.time()

    for path in shard_paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                rec = _safe_load_line(line)
                if rec is None:
                    continue
                result = format_and_tokenise(rec, tokenizer, max_len=max_len_per_example)
                if result is None:
                    continue

                ids, lmask = result
                tok_arr  = np.array(ids,   dtype=dtype_t)
                mask_arr = np.array(lmask, dtype=dtype_m)

                if _is_val(n_records, val_fraction):
                    n = len(tok_arr)
                    val_tok [val_ptr : val_ptr + n] = tok_arr
                    val_mask[val_ptr : val_ptr + n] = mask_arr
                    val_tok [val_ptr + n]            = eos_id
                    val_mask[val_ptr + n]            = 0
                    val_ptr += n + 1
                else:
                    n = len(tok_arr)
                    train_tok [train_ptr : train_ptr + n] = tok_arr
                    train_mask[train_ptr : train_ptr + n] = mask_arr
                    train_tok [train_ptr + n]              = eos_id
                    train_mask[train_ptr + n]              = 0
                    train_ptr += n + 1

                n_records += 1
                if time.time() - last_print > 5:
                    print(f"[pack worker {worker + 1}/{num_workers}] packing … "
                          f"{n_records:,} records written", end="\r")
                    last_print = time.time()

    train_tok.flush(); train_mask.flush()
    val_tok.flush();   val_mask.flush()
    print(f"\n[pack worker {worker + 1}/{num_workers}] packed {n_records:,} records "
          f"in {time.time()-t0:.1f}s total")

    manifest = {
        "worker":          worker,
        "num_workers":     num_workers,
        "shard_paths":     shard_paths,
        "n_records":       n_records,
        "train_tokens":    total_train,
        "val_tokens":      total_val,
        "dtype_t":         str(np.dtype(dtype_t)),
        "dtype_m":         str(np.dtype(dtype_m)),
        "vocab_size":      vocab_size,
        "max_len_per_example": max_len_per_example,
        "val_fraction":    val_fraction,
        "source_format":   "grpo",
        "train_tokens_file": os.path.basename(train_tok_path),
        "train_mask_file":   os.path.basename(train_mask_path),
        "val_tokens_file":   os.path.basename(val_tok_path),
        "val_mask_file":     os.path.basename(val_mask_path),
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[pack worker {worker + 1}/{num_workers}] wrote manifest to {manifest_path}")

    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Read + pack GRPO JSONL data into memmap .bin files "
                     "for train_grpo.py."
    )
    p.add_argument("--data-dir",  default="./grpo_data",
                   help="GRPO data directory from download_grpo_data.py")
    p.add_argument("--tokenizer", default="./tokenizer",
                   help="Tokenizer directory from train_tokenizer.py")
    p.add_argument("--cache-dir", default="./grpo_packed",
                   help="Where to write the packed memmap files "
                        "(default ./grpo_packed). Point train_grpo.py's "
                        "--cache_dir at this directory.")
    p.add_argument("--max-len-per-example", type=int, default=2048,
                   help="Max tokens per individual GRPO example before truncation")
    p.add_argument("--val-fraction", type=float, default=0.01,
                   help="Fraction of records deterministically routed to val")

    p.add_argument("--worker", type=int, default=0,
                   help="This worker's index (0-indexed). Each worker packs "
                        "a disjoint subset of the input .jsonl shards, "
                        "selected by file index modulo --num-workers. "
                        "Run this script once per worker to parallelise "
                        "packing across CPUs/machines.")
    p.add_argument("--num-workers", type=int, default=1,
                   help="Total number of workers across all invocations. "
                        "Must be the same value for every worker packing "
                        "into the same --cache-dir.")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.worker < 0 or args.worker >= args.num_workers:
        raise ValueError(
            f"--worker must be in [0, {args.num_workers}), got {args.worker}"
        )

    tokenizer = load_tokenizer(args.tokenizer)
    print(f"Tokenizer vocab size: {tokenizer.get_vocab_size():,}")

    pack_worker_shard(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        cache_dir=args.cache_dir,
        max_len_per_example=args.max_len_per_example,
        val_fraction=args.val_fraction,
        worker=args.worker,
        num_workers=args.num_workers,
        vocab_size=tokenizer.get_vocab_size(),
    )

    print(f"\nDone. Worker {args.worker}/{args.num_workers} packed files are in "
          f"{args.cache_dir}")
    if args.num_workers > 1:
        print(f"Run the remaining {args.num_workers - 1} worker(s) before "
              f"training, then point train_grpo.py --cache_dir at the same "
              f"directory — it will discover and concatenate all workers' "
              f"shards automatically.")
    print(f"\nNext step:\n"
          f"  # Default — packed memmap + JSONL for ground-truth recovery\n"
          f"  python train_grpo.py --checkpoint ./sft_checkpoints/latest.pt \\\n"
          f"      --tokenizer ./tokenizer --cache_dir {args.cache_dir} \\\n"
          f"      --data_dir {args.data_dir}\n"
          f"\n  # --prompt_override flag — skip the memmap entirely and\n"
          f"  # consume a flat JSONL of {{prompt, answer}} instead\n"
          f"  python train_grpo.py --checkpoint ./sft_checkpoints/latest.pt \\\n"
          f"      --tokenizer ./tokenizer --prompt_override ./eval_prompts.jsonl")