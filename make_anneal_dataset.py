#!/usr/bin/env python3
"""
make_anneal_dataset.py

Companion to BOTH train_deepspeed.py and train.py. Their `PackedDataLoader`
classes are functionally identical (same shard math, same
`torch.Generator` seed formula `seed*1_000_003 + rank*31 + loader_id`,
same per-worker seed formula `base_seed + worker_id*9973`, same
prefetch-ahead accounting) — this script replays that shared RNG stream
regardless of which of the two you trained with. That sampling is:

    torch.randint(n_pos, (batch_size,), generator=gen)

where `gen` is seeded deterministically as above. Because the seed is
fixed and no external randomness is involved, the exact sequence of
sampled window-start offsets for a given (data-dir, seq-len, batch-size,
grad-accum-steps, world-size, seed, completed-steps, num-workers) is
fully reproducible. This script replays that same sequence (WITHOUT
touching the actual token data — it only needs the random offsets) to
build a "used" coverage mask per rank-shard, then extracts the
complementary "never sampled" token ranges and packs them into a new
train.bin under --out-dir (default ./packed_anneal), suitable for an
annealing phase.

WHICH TRAINING SCRIPT SHOULD MAP TO WHICH FLAG HERE
----------------------------------------------------
--data-dir / --seq-len / --batch-size / --grad-accum-steps / --seed /
--num-workers should just be the SAME values you passed to whichever of
train_deepspeed.py / train.py you actually ran. Two things differ
between the two scripts that are easy to get wrong:
  * --num-workers DEFAULT differs: train_deepspeed.py defaults to 0,
    train.py defaults to 2. Always pass the value you actually used —
    don't rely on this script's default.
  * --world-size is not itself a CLI flag on either training script.
    For train_deepspeed.py it's the DeepSpeed-launcher WORLD_SIZE env
    var (GPU count). For train.py it's torch.distributed's world size
    if launched with torchrun (again, GPU count), or 1 if you ran it as
    a single process. Either way, pass the actual number of ranks that
    were training in parallel.

IMPORTANT ASSUMPTIONS / CAVEATS
--------------------------------
1. Reconstructs the RNG stream for BOTH the single-process path
   (--num-workers 0) and the multi-worker path (--num-workers > 0).
   For the multi-worker path this relies on PyTorch's default
   (non-shuffled, no custom sampler) DataLoader behavior: index i is
   dispatched to worker (i % num_workers), and results are returned to
   the main process in strict index order regardless of which worker
   finishes first. That is the standard/default behavior for a map-style
   Dataset with no sampler/shuffle set (which is what train_deepspeed.py
   uses), so this is exactly reproducible, NOT an approximation.
2. PackedDataLoader's generator is NOT checkpointed/restored across
   --resume. If your real run resumed one or more times, each resumed
   process re-seeds the identical generator from scratch and replays the
   *start* of the same stream again (not a continuation). This script
   assumes NO resumes happened during [0, completed_steps) — i.e. one
   continuous process. If you did resume mid-run, pass --completed-steps
   as the number of *optimizer steps actually executed in the final
   continuous session* and be aware the true coverage may differ.
3. val.bin is untouched by this script — it's copied as-is (it's already
   held out, and evaluated sequentially, not randomly sampled).

USAGE
-----
Pass the SAME --data-dir / --seq-len / --batch-size / --grad-accum-steps /
--seed / --num-workers you used for training, plus --world-size (the
number of ranks/GPUs — this maps to the WORLD_SIZE env var DeepSpeed set,
which isn't a CLI flag in train_deepspeed.py so must be supplied here
explicitly) and --completed-steps (how many optimizer steps you actually
trained, e.g. 40000):

    python make_anneal_dataset.py \\
        --data-dir ./packed \\
        --seq-len 2048 \\
        --batch-size 4 \\
        --grad-accum-steps 8 \\
        --world-size 8 \\
        --seed 42 \\
        --completed-steps 40000 \\
        --out-dir ./packed_anneal
"""

import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract the unsampled portion of a packed training "
                    "dataset for use as an annealing-phase dataset."
    )
    # --- must match the real training run exactly ---
    p.add_argument("--data-dir", default="./packed",
                    help="Same --data-dir used for training (contains "
                         "train.bin, val.bin, meta.json)")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=4,
                    help="Same --batch-size (per-GPU micro batch) used for training")
    p.add_argument("--grad-accum-steps", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0,
                    help="Same --num-workers used for training. Only 0 is "
                         "exactly reproducible (see module docstring).")
    p.add_argument("--world-size", type=int, required=True,
                    help="Number of ranks/GPUs the real run used "
                         "(the WORLD_SIZE env var DeepSpeed set — not a "
                         "CLI flag in train_deepspeed.py, so supply it here)")
    p.add_argument("--completed-steps", type=int, required=True,
                    help="Number of optimizer steps actually executed in "
                         "the (single, continuous) training session whose "
                         "leftover data you want to extract")

    # --- output controls ---
    p.add_argument("--out-dir", default="./packed_anneal")
    p.add_argument("--min-run-tokens", type=int, default=None,
                    help="Discard unused runs shorter than this many "
                         "tokens (default: --seq-len, since a run shorter "
                         "than one context window can't form a full "
                         "training example anyway)")
    p.add_argument("--copy-val", action="store_true", default=True,
                    help="Copy val.bin/meta.json unchanged into --out-dir "
                         "(default: on)")
    return p.parse_args()


def replay_used_intervals_single_process(n_pos: int, batch_size: int,
                                          total_builds: int,
                                          gen: torch.Generator) -> np.ndarray:
    """
    num_workers == 0 path. Replay `total_builds` calls of
    torch.randint(n_pos, (batch_size,), gen) -- identical to
    PackedDataLoader._build() -- and return a flat array of sampled
    local-shard start offsets.
    """
    starts = torch.empty(total_builds * batch_size, dtype=torch.int64)
    pos = 0
    for _ in range(total_builds):
        ix = torch.randint(n_pos, (batch_size,), generator=gen)
        starts[pos:pos + batch_size] = ix
        pos += batch_size
    return starts.numpy()


def replay_used_intervals_workers(n_pos: int, batch_size: int,
                                   num_next_batch_calls: int,
                                   num_workers: int,
                                   base_seed: int) -> np.ndarray:
    """
    num_workers > 0 path (matches PackedDataLoader._start_workers /
    _ShardDataset). PyTorch's default DataLoader (no shuffle, no custom
    sampler, map-style dataset) dispatches item index i to worker
    (i % num_workers) and yields results to the caller in strict index
    order. Each worker's own generator is seeded once, lazily, as
    `base_seed + worker_id * 9973` and then advances across all of that
    worker's calls (persistent_workers=True keeps it alive for the run).
    We replay each worker's stream independently; order across workers
    doesn't matter for coverage purposes, only which offsets were drawn.
    """
    all_starts = []
    for w in range(num_workers):
        calls_w = num_next_batch_calls // num_workers
        if w < (num_next_batch_calls % num_workers):
            calls_w += 1
        if calls_w == 0:
            continue
        gen_w = torch.Generator().manual_seed(base_seed + w * 9973)
        starts_w = torch.empty(calls_w * batch_size, dtype=torch.int64)
        pos = 0
        for _ in range(calls_w):
            ix = torch.randint(n_pos, (batch_size,), generator=gen_w)
            starts_w[pos:pos + batch_size] = ix
            pos += batch_size
        all_starts.append(starts_w.numpy())
    if not all_starts:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(all_starts)


def merge_intervals(starts: np.ndarray, seq_len: int, shard_len: int) -> np.ndarray:
    """Merge (possibly overlapping) [start, start+seq_len) intervals into
    a sorted, non-overlapping (K, 2) array of [start, end) ranges."""
    if len(starts) == 0:
        return np.empty((0, 2), dtype=np.int64)
    starts = np.sort(starts)
    ends = starts + seq_len
    merged_starts = [starts[0]]
    merged_ends = [ends[0]]
    for s, e in zip(starts[1:], ends[1:]):
        if s <= merged_ends[-1]:
            if e > merged_ends[-1]:
                merged_ends[-1] = e
        else:
            merged_starts.append(s)
            merged_ends.append(e)
    merged = np.stack([np.array(merged_starts), np.array(merged_ends)], axis=1)
    np.clip(merged, 0, shard_len, out=merged)
    return merged


def invert_intervals(used: np.ndarray, shard_len: int) -> np.ndarray:
    """Given sorted non-overlapping [start,end) 'used' ranges within
    [0, shard_len), return the complementary 'unused' ranges."""
    gaps = []
    cursor = 0
    for s, e in used:
        if s > cursor:
            gaps.append((cursor, s))
        cursor = max(cursor, e)
    if cursor < shard_len:
        gaps.append((cursor, shard_len))
    return np.array(gaps, dtype=np.int64) if gaps else np.empty((0, 2), dtype=np.int64)


def main():
    args = parse_args()

    min_run_tokens = args.min_run_tokens or args.seq_len

    meta_path = os.path.join(args.data_dir, "meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    dtype_np = np.uint16 if meta["dtype"] == "uint16" else np.uint32

    train_bin = os.path.join(args.data_dir, "train.bin")
    data = np.memmap(train_bin, dtype=dtype_np, mode="r")
    total_len = len(data)

    num_next_batch_calls = args.completed_steps * args.grad_accum_steps
    print(f"Total tokens in train.bin : {total_len:,}")
    print(f"World size                : {args.world_size}")
    print(f"Num workers               : {args.num_workers}")
    print(f"Completed optimizer steps : {args.completed_steps:,}")
    print(f"next_batch() calls / rank : {num_next_batch_calls:,} "
          f"({num_next_batch_calls * args.batch_size:,} windows/rank)")

    os.makedirs(args.out_dir, exist_ok=True)
    tmp_out_path = os.path.join(args.out_dir, "train.bin.tmp")

    total_unused_tokens = 0
    total_shard_tokens = 0

    with open(tmp_out_path, "wb") as out_f:
        for rank in range(args.world_size):
            shard_size = total_len // args.world_size
            start = rank * shard_size
            end = start + shard_size if rank < args.world_size - 1 else total_len
            shard_len = end - start
            n_pos = max(1, shard_len - args.seq_len)

            base_seed = args.seed * 1_000_003 + rank * 31 + 0  # loader_id=0 (train)

            if args.num_workers > 0:
                local_starts = replay_used_intervals_workers(
                    n_pos, args.batch_size, num_next_batch_calls,
                    args.num_workers, base_seed,
                )
            else:
                # single-process path: prime() does one _build(), then
                # each next_batch() call does one more (prefetch-ahead)
                total_builds = num_next_batch_calls + 1
                gen = torch.Generator().manual_seed(base_seed)
                local_starts = replay_used_intervals_single_process(
                    n_pos, args.batch_size, total_builds, gen
                )
            used = merge_intervals(local_starts, args.seq_len, shard_len)
            unused = invert_intervals(used, shard_len)

            # keep only runs long enough to matter, extract from memmap,
            # write to output file in order
            rank_unused_tokens = 0
            for s, e in unused:
                if e - s < min_run_tokens:
                    continue
                chunk = np.asarray(data[start + s:start + e])
                chunk.tofile(out_f)
                rank_unused_tokens += (e - s)

            used_tokens = shard_len - rank_unused_tokens
            print(f"  rank {rank:3d}: shard {shard_len:,} tok | "
                  f"~used {used_tokens:,} ({100*used_tokens/max(1,shard_len):.1f}%) | "
                  f"unused kept {rank_unused_tokens:,} "
                  f"({100*rank_unused_tokens/max(1,shard_len):.1f}%)")

            total_unused_tokens += rank_unused_tokens
            total_shard_tokens += shard_len

    final_out_path = os.path.join(args.out_dir, "train.bin")
    os.replace(tmp_out_path, final_out_path)

    print(f"\nTotal unused tokens extracted: {total_unused_tokens:,} "
          f"({100*total_unused_tokens/max(1,total_shard_tokens):.1f}% of dataset)")
    print(f"Written to: {final_out_path}")

    # meta.json for the anneal set — same vocab/dtype, just note provenance
    anneal_meta = dict(meta)
    anneal_meta["source"] = "anneal-extract"
    anneal_meta["source_data_dir"] = os.path.abspath(args.data_dir)
    anneal_meta["completed_steps_excluded"] = args.completed_steps
    anneal_meta["world_size_assumed"] = args.world_size
    anneal_meta["n_tokens"] = int(total_unused_tokens)
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(anneal_meta, f, indent=2)

    if args.copy_val:
        val_src = os.path.join(args.data_dir, "val.bin")
        if os.path.exists(val_src):
            shutil.copy2(val_src, os.path.join(args.out_dir, "val.bin"))
            print("Copied val.bin unchanged (held-out set is unaffected "
                  "by training sampling).")

    if total_unused_tokens < args.seq_len * args.batch_size * args.grad_accum_steps:
        print("\n[WARN] Very little unused data was found — check that "
              "--world-size / --completed-steps / --seed match the real "
              "training run, or the model may already have swept nearly "
              "the whole dataset.", file=sys.stderr)


if __name__ == "__main__":
    main()
