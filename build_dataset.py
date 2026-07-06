#!/usr/bin/env python3
"""
build_dataset.py

Builds a pretraining corpus for a Qwen3-style reasoning LLM by streaming
from large, pre-filtered open datasets (no raw web crawling required) and
writing packed JSONL shards to ./data/<category>/.

Distribution (defaults, configurable via --mix):
    web        : 50%  -> FineWeb (filtered CommonCrawl)
    code       : 20%  -> The Stack v2 (smol / dedup subset)
    math       : 15%  -> FineMath
    knowledge  : 10%  -> Wikipedia
    reasoning  :  5%  -> OpenOrca / CoT-style instruction data

Usage:
    python build_dataset.py --target-size 5GB
    python build_dataset.py --target-size 500MB --mix web=0.6,code=0.2,math=0.1,knowledge=0.05,reasoning=0.05
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Iterator, Optional

from datasets import load_dataset

# ---------------------------------------------------------------------------
# Dataset source definitions
# ---------------------------------------------------------------------------
# Each category maps to a list of (dataset_name, config/subset, split, text_field,
# extra_kwargs) tuples. Multiple sources per category are round-robined so the
# corpus isn't dominated by a single dataset's style.

SOURCES = {
    "web": [
        dict(path="HuggingFaceFW/fineweb", name="sample-10BT", split="train", text_field="text"),
    ],
    "code": [
        dict(path="bigcode/the-stack-smol", name=None, split="train", text_field="content"),
    ],
    "math": [
        dict(path="HuggingFaceTB/finemath", name="finemath-4plus", split="train", text_field="text"),
    ],
    "knowledge": [
        dict(path="wikimedia/wikipedia", name="20231101.en", split="train", text_field="text"),
    ],
    "reasoning": [
        dict(path="Open-Orca/OpenOrca", name=None, split="train", text_field=None),  # special-cased
    ],
}

DEFAULT_MIX = {
    "web": 0.50,
    "code": 0.20,
    "math": 0.15,
    "knowledge": 0.10,
    "reasoning": 0.05,
}

SHARD_MAX_BYTES = 256 * 1024 * 1024  # 256MB per shard file


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_size(size_str: str) -> int:
    """Parse a human size string like '5GB', '500MB', '2.5gb' into bytes."""
    size_str = size_str.strip().upper()
    units = {"GB": 1024**3, "MB": 1024**2, "KB": 1024, "B": 1}
    for unit, mult in units.items():
        if size_str.endswith(unit):
            num = size_str[: -len(unit)]
            return int(float(num) * mult)
    # assume bytes if no unit given
    return int(float(size_str))


def parse_mix(mix_str: Optional[str]) -> dict:
    if mix_str is None:
        return DEFAULT_MIX
    mix = {}
    for part in mix_str.split(","):
        k, v = part.split("=")
        mix[k.strip()] = float(v)
    total = sum(mix.values())
    if abs(total - 1.0) > 1e-6:
        print(f"[warn] mix sums to {total}, normalizing to 1.0")
        mix = {k: v / total for k, v in mix.items()}
    for k in mix:
        if k not in SOURCES:
            raise ValueError(f"Unknown category in mix: {k} (valid: {list(SOURCES.keys())})")
    return mix


def extract_text(example: dict, text_field: Optional[str], category: str) -> Optional[str]:
    """Pull a clean text string out of a dataset example."""
    if text_field is not None:
        txt = example.get(text_field)
        return txt if isinstance(txt, str) and txt.strip() else None

    # Special-cased: OpenOrca style instruction/CoT data -> format as a
    # simple instruction/response document so it's useful for next-token
    # pretraining on reasoning-style text.
    if category == "reasoning":
        instr = example.get("question") or example.get("instruction") or ""
        resp = example.get("response") or example.get("output") or ""
        if not instr or not resp:
            return None
        return f"### Instruction:\n{instr}\n\n### Response:\n{resp}"

    return None


# ---------------------------------------------------------------------------
# Shard writer
# ---------------------------------------------------------------------------

class ShardWriter:
    """Writes JSONL records to size-capped shard files under a category dir."""

    def __init__(self, out_dir: str, category: str, max_shard_bytes: int = SHARD_MAX_BYTES):
        self.dir = os.path.join(out_dir, category)
        os.makedirs(self.dir, exist_ok=True)
        self.category = category
        self.max_shard_bytes = max_shard_bytes
        self.shard_idx = 0
        self.bytes_in_shard = 0
        self.total_bytes = 0
        self.total_docs = 0
        self._fh = self._open_new_shard()

    def _open_new_shard(self):
        path = os.path.join(self.dir, f"{self.category}_{self.shard_idx:05d}.jsonl")
        return open(path, "w", encoding="utf-8")

    def write(self, text: str, meta: dict):
        record = {"text": text, **meta}
        line = json.dumps(record, ensure_ascii=False) + "\n"
        line_bytes = len(line.encode("utf-8"))

        if self.bytes_in_shard + line_bytes > self.max_shard_bytes and self.bytes_in_shard > 0:
            self._fh.close()
            self.shard_idx += 1
            self.bytes_in_shard = 0
            self._fh = self._open_new_shard()

        self._fh.write(line)
        self.bytes_in_shard += line_bytes
        self.total_bytes += line_bytes
        self.total_docs += 1

    def close(self):
        self._fh.close()


# ---------------------------------------------------------------------------
# Per-category crawl/stream loop
# ---------------------------------------------------------------------------

def stream_category(category: str, byte_budget: int, out_dir: str, min_doc_chars: int = 200):
    sources = SOURCES[category]
    writer = ShardWriter(out_dir, category)

    print(f"\n=== [{category}] target: {byte_budget / 1024**2:.1f} MB "
          f"from {len(sources)} source(s) ===")

    last_report = time.time()

    # Walk each source at most once. Re-opening a streaming dataset after it
    # has been exhausted restarts from example 0, which silently writes the
    # same documents again and never reaches the byte budget. Track an offset
    # into the source list and only advance it forward.
    for src in sources:
        if writer.total_bytes >= byte_budget:
            break

        try:
            ds = load_dataset(
                src["path"],
                src.get("name"),
                split=src["split"],
                streaming=True,
            )
        except Exception as e:
            print(f"[error] failed to open {src['path']}: {e}")
            continue

        source_was_exhausted = False
        try:
            for example in ds:
                text = extract_text(example, src["text_field"], category)
                if not text or len(text) < min_doc_chars:
                    continue

                writer.write(text, meta={"source": src["path"], "category": category})

                if writer.total_bytes >= byte_budget:
                    break

                if time.time() - last_report > 5:
                    pct = 100 * writer.total_bytes / byte_budget
                    print(f"[{category}] {writer.total_bytes / 1024**2:8.2f} MB "
                          f"/ {byte_budget / 1024**2:.1f} MB  ({pct:5.1f}%)  "
                          f"docs={writer.total_docs}", end="\r")
                    last_report = time.time()
            else:
                # Inner for-loop completed without `break` -> iterator is
                # exhausted. Mark it so we don't re-open this source.
                source_was_exhausted = True

        except Exception as e:
            print(f"\n[warn] stream interrupted for {src['path']}: {e} -- moving to next source")
            continue
        finally:
            if source_was_exhausted and writer.total_bytes < byte_budget:
                print(f"\n[{category}] source {src['path']} exhausted at "
                      f"{writer.total_bytes / 1024**2:.1f} MB "
                      f"(target was {byte_budget / 1024**2:.1f} MB)")

    print(f"\n[{category}] done: {writer.total_bytes / 1024**2:.2f} MB, "
          f"{writer.total_docs} docs, {writer.shard_idx + 1} shard(s)")
    writer.close()
    return writer.total_bytes, writer.total_docs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build a pretraining corpus from streamed open datasets.")
    parser.add_argument("--target-size", required=True, help="Total dataset size, e.g. 5GB, 500MB")
    parser.add_argument("--out-dir", default="./data", help="Output directory (default ./data)")
    parser.add_argument("--mix", default=None,
                         help="Comma-separated category=fraction, e.g. web=0.5,code=0.2,math=0.15,knowledge=0.1,reasoning=0.05")
    parser.add_argument("--min-doc-chars", type=int, default=200,
                         help="Minimum document length (chars) to keep")
    args = parser.parse_args()

    target_bytes = parse_size(args.target_size)
    mix = parse_mix(args.mix)

    print(f"Target total size : {target_bytes / 1024**2:.1f} MB")
    print(f"Output directory  : {args.out_dir}")
    print(f"Mix               : {mix}")

    os.makedirs(args.out_dir, exist_ok=True)

    manifest = {"target_bytes": target_bytes, "mix": mix, "categories": {}}

    for category, frac in mix.items():
        budget = int(target_bytes * frac)
        if budget <= 0:
            continue
        actual_bytes, docs = stream_category(category, budget, args.out_dir, args.min_doc_chars)
        manifest["categories"][category] = {
            "target_bytes": budget,
            "actual_bytes": actual_bytes,
            "docs": docs,
        }

    manifest_path = os.path.join(args.out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    total_actual = sum(c["actual_bytes"] for c in manifest["categories"].values())
    print(f"\n=== Done. Total: {total_actual / 1024**2:.2f} MB across "
          f"{len(manifest['categories'])} categories ===")
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
