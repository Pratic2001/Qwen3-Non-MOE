#!/usr/bin/env python3
"""
print_packed_data.py

Sanity-check viewer for the packed memmap outputs produced by this repo's
three packers. Reads the raw .bin files directly (no torch dependency) and
prints token counts, a handful of raw token IDs, and — if a tokenizer is
given — the decoded text for a few examples.

Formats understood
-------------------
1. "pretrain"  — output of pack_dataset.py
       <dir>/train.bin, <dir>/val.bin   (flat uint16/uint32 token stream)
       <dir>/meta.json                  (dtype, vocab_size, token counts, ...)

2. "sft" / "grpo" — output of pack_sft_data.py and pack_grpo_data.py
   (identical on-disk convention, one manifest per worker)
       <dir>/sft_manifest.w{worker}-of-{num_workers}.json
       <dir>/sft_train_tokens.w{worker}-of-{num_workers}.bin
       <dir>/sft_train_mask.w{worker}-of-{num_workers}.bin
       <dir>/sft_val_tokens.w{worker}-of-{num_workers}.bin
       <dir>/sft_val_mask.w{worker}-of-{num_workers}.bin

Usage
-----
    # auto-detect format from what's in the directory
    python print_packed_data.py --dir ./packed

    # force a format
    python print_packed_data.py --dir ./sft_packed --format sft

    # decode token ids back to text (needs the tokenizer dir used to pack)
    python print_packed_data.py --dir ./packed --tokenizer ./tokenizer

    # look at val split, print more examples, more tokens per example
    python print_packed_data.py --dir ./sft_packed --split val \\
        --num-examples 5 --max-tokens 64
"""

import argparse
import glob
import json
import os
import sys

import numpy as np


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_tokenizer(tokenizer_dir):
    if not tokenizer_dir:
        return None
    path = os.path.join(tokenizer_dir, "tokenizer.json")
    if not os.path.exists(path):
        print(f"[warn] tokenizer.json not found at {path} — printing raw ids only")
        return None
    from tokenizers import Tokenizer
    return Tokenizer.from_file(path)


def _decode(tokenizer, ids):
    if tokenizer is None:
        return None
    try:
        return tokenizer.decode([int(i) for i in ids])
    except Exception as e:
        return f"<decode error: {e}>"


def _detect_format(d):
    if os.path.exists(os.path.join(d, "meta.json")) and (
        os.path.exists(os.path.join(d, "train.bin")) or os.path.exists(os.path.join(d, "val.bin"))
    ):
        return "pretrain"
    if glob.glob(os.path.join(d, "sft_manifest.w*-of-*.json")):
        return "sft"
    raise FileNotFoundError(
        f"Couldn't detect a known packed format in {d!r}. Expected either "
        f"meta.json+train.bin/val.bin (pack_dataset.py) or "
        f"sft_manifest.w*.json (pack_sft_data.py / pack_grpo_data.py)."
    )


# ---------------------------------------------------------------------------
# pretrain format (pack_dataset.py)
# ---------------------------------------------------------------------------

def print_pretrain(d, split, num_examples, max_tokens, tokenizer):
    meta_path = os.path.join(d, "meta.json")
    with open(meta_path) as f:
        meta = json.load(f)

    print("=" * 70)
    print(f"pretrain-format packed data : {d}")
    print("=" * 70)
    print(json.dumps(meta, indent=2))

    dtype = np.dtype(meta["dtype"])
    bin_path = os.path.join(d, f"{split}.bin")
    if not os.path.exists(bin_path):
        print(f"\n[warn] {bin_path} does not exist, nothing to print")
        return

    arr = np.memmap(bin_path, dtype=dtype, mode="r")
    n_tok = arr.shape[0]
    print(f"\n{split}.bin : {n_tok:,} tokens, dtype={dtype}")

    window = max_tokens * num_examples
    window = min(window, n_tok)
    ids = arr[:window]

    print(f"\nFirst {len(ids)} raw token ids:")
    print(ids.tolist())

    if tokenizer is not None:
        print(f"\nDecoded in chunks of {max_tokens} tokens:")
        for i in range(0, len(ids), max_tokens):
            chunk = ids[i:i + max_tokens]
            text = _decode(tokenizer, chunk)
            print(f"\n--- tokens [{i}:{i + len(chunk)}] ---")
            print(text)


# ---------------------------------------------------------------------------
# sft / grpo format (pack_sft_data.py / pack_grpo_data.py)
# ---------------------------------------------------------------------------

def _load_sft_manifests(d):
    paths = sorted(glob.glob(os.path.join(d, "sft_manifest.w*-of-*.json")))
    if not paths:
        raise FileNotFoundError(f"No sft_manifest.w*.json files found in {d}")
    manifests = []
    for p in paths:
        with open(p) as f:
            manifests.append(json.load(f))
    return manifests


def print_sft(d, split, num_examples, max_tokens, tokenizer):
    manifests = _load_sft_manifests(d)

    print("=" * 70)
    print(f"sft/grpo-format packed data : {d}  ({len(manifests)} worker shard(s))")
    print("=" * 70)

    total_tokens = sum(m[f"{split}_tokens"] for m in manifests)
    total_records = sum(m["n_records"] for m in manifests)
    print(f"workers        : {len(manifests)}")
    print(f"total records  : {total_records:,}")
    print(f"total {split} tokens : {total_tokens:,}")
    print(f"vocab_size     : {manifests[0].get('vocab_size')}")
    print(f"dtype (tokens) : {manifests[0].get('dtype_t')}")
    print(f"dtype (mask)   : {manifests[0].get('dtype_m')}")

    remaining = num_examples
    for m in manifests:
        if remaining <= 0:
            break
        tok_file = m.get(f"{split}_tokens_file")
        mask_file = m.get(f"{split}_mask_file")
        n_tok = m.get(f"{split}_tokens", 0)
        if not tok_file or n_tok == 0:
            continue

        tok_path = os.path.join(d, tok_file)
        mask_path = os.path.join(d, mask_file)
        dtype_t = np.dtype(m["dtype_t"])
        dtype_m = np.dtype(m["dtype_m"])

        tok_arr = np.memmap(tok_path, dtype=dtype_t, mode="r")
        mask_arr = np.memmap(mask_path, dtype=dtype_m, mode="r")

        print(f"\n--- worker {m['worker']} : {tok_file} ({n_tok:,} tokens) ---")

        window = min(max_tokens * remaining, len(tok_arr))
        ids = tok_arr[:window]
        mask = mask_arr[:window]

        print(f"First {len(ids)} raw token ids:")
        print(ids.tolist())
        print(f"Corresponding loss mask (1 = loss computed, 0 = masked):")
        print(mask.tolist())

        if tokenizer is not None:
            print(f"\nDecoded in chunks of {max_tokens} tokens:")
            for i in range(0, len(ids), max_tokens):
                chunk = ids[i:i + max_tokens]
                cmask = mask[i:i + max_tokens]
                text = _decode(tokenizer, chunk)
                print(f"\n[worker {m['worker']}] tokens [{i}:{i + len(chunk)}] "
                      f"(loss-on-tokens={int(cmask.sum())}/{len(cmask)})")
                print(text)
                remaining -= 1
                if remaining <= 0:
                    break


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Print packed dataset contents for a sanity check.")
    p.add_argument("--dir", required=True, help="Directory containing the packed output.")
    p.add_argument("--format", choices=["auto", "pretrain", "sft", "grpo"], default="auto",
                   help="Which packer produced this data. 'grpo' is handled identically to 'sft'. "
                        "Default: auto-detect from files present in --dir.")
    p.add_argument("--split", choices=["train", "val"], default="train",
                   help="Which split to print.")
    p.add_argument("--tokenizer", default=None,
                   help="Path to the tokenizer directory (containing tokenizer.json) used when "
                        "packing, to decode ids back to text. If omitted, only raw ids are shown.")
    p.add_argument("--num-examples", type=int, default=3,
                   help="How many example windows to print.")
    p.add_argument("--max-tokens", type=int, default=32,
                   help="How many tokens per example window.")
    return p.parse_args()


def main():
    args = parse_args()
    fmt = args.format
    if fmt in ("auto",):
        fmt = _detect_format(args.dir)
    elif fmt == "grpo":
        fmt = "sft"  # identical on-disk format

    tokenizer = _load_tokenizer(args.tokenizer)

    if fmt == "pretrain":
        print_pretrain(args.dir, args.split, args.num_examples, args.max_tokens, tokenizer)
    elif fmt == "sft":
        print_sft(args.dir, args.split, args.num_examples, args.max_tokens, tokenizer)
    else:
        print(f"Unknown format: {fmt}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
