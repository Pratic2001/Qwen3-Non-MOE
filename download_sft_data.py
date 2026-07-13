#!/usr/bin/env python3
"""
download_sft_data.py

Downloads and formats supervised fine-tuning (SFT) data for reasoning
from HuggingFace, writing structured JSONL shards to ./sft_data/<category>/.

Every record is written as:
    {"prompt": str, "thinking": str, "answer": str, "source": str, "category": str}

  prompt   — the user's question / problem statement
  thinking — the chain-of-thought reasoning trace (may be empty string for
             datasets that only have final answers; sft.py will skip the
             <think> block in that case)
  answer   — the final answer / solution

Distribution (defaults, configurable via --mix):
    math      : 40%  -> NuminaMath-CoT, MetaMathQA, GSM8K
    code      : 30%  -> Evol-Instruct-Code, Python-Instructions
    reasoning : 20%  -> OpenThoughts-114k, OpenHermes-2.5
    science   : 10%  -> AI2-ARC, ScienceQA

Usage:
    python download_sft_data.py --target-size 2GB
    python download_sft_data.py --target-size 500MB --mix math=0.5,code=0.3,reasoning=0.2
    python download_sft_data.py --target-size 10GB --out-dir ./sft_data
"""

import argparse
import json
import os
import re
import time
from typing import Iterator, Optional

from datasets import load_dataset

# ---------------------------------------------------------------------------
# Dataset source registry
# Each entry: path, name (subset), split, category, extractor key
# ---------------------------------------------------------------------------

SOURCES = {
    "math": [
        dict(
            path="AI-MO/NuminaMath-CoT",
            name=None,
            split="train",
            extractor="numina_math",
        ),
        dict(
            path="meta-math/MetaMathQA",
            name=None,
            split="train",
            extractor="metamath",
        ),
        dict(
            path="openai/gsm8k",
            name="main",
            split="train",
            extractor="gsm8k",
        ),
    ],
    "code": [
        dict(
            path="nickrosh/Evol-Instruct-Code-80k-v1",
            name=None,
            split="train",
            extractor="instruction_output",
        ),
        dict(
            path="iamtarun/python_code_instructions_18k_alpaca",
            name=None,
            split="train",
            extractor="alpaca_code",
        ),
    ],
    "reasoning": [
        dict(
            path="open-thoughts/OpenThoughts-114k",
            name=None,
            split="train",
            extractor="open_thoughts",
        ),
        dict(
            path="teknium/OpenHermes-2.5",
            name=None,
            split="train",
            extractor="open_hermes",
        ),
    ],
    "science": [
        dict(
            path="allenai/ai2_arc",
            name="ARC-Challenge",
            split="train",
            extractor="arc",
        ),
        dict(
            path="derek-thomas/ScienceQA",
            name=None,
            split="train",
            extractor="science_qa",
        ),
    ],
}

DEFAULT_MIX = {
    "math":      0.40,
    "code":      0.30,
    "reasoning": 0.20,
    "science":   0.10,
}

SHARD_MAX_BYTES = 128 * 1024 * 1024   # 128 MB per shard
MIN_PROMPT_CHARS = 10
MIN_ANSWER_CHARS = 5


# ---------------------------------------------------------------------------
# Per-dataset extractors  ->  {"prompt", "thinking", "answer"} or None
# ---------------------------------------------------------------------------

def _extract_boxed(text: str):
    """Return every \\boxed{...} payload in `text`, respecting nested braces.

    The naive regex pattern boxed\{([^}]+)\} stops at the FIRST closing brace,
    so any nested-brace LaTeX (\\boxed{\\frac{1}{2}}, \\boxed{x^{2}}, etc. --
    extremely common in NuminaMath/OpenThoughts solutions) gets truncated
    into a syntactically broken, wrong answer. This does real brace matching
    instead.
    """
    results = []
    i = 0
    marker = "\\boxed{"
    while True:
        start = text.find(marker, i)
        if start == -1:
            break
        depth = 1
        j = start + len(marker)
        while j < len(text) and depth > 0:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        if depth == 0:
            results.append(text[start + len(marker): j - 1])
        i = j
    return results


def _gsm8k_split_thinking(raw_answer: str):
    """GSM8K answers are 'reasoning\n#### final_num'. Split them."""
    if "####" in raw_answer:
        parts = raw_answer.split("####", 1)
        return parts[0].strip(), parts[1].strip()
    return "", raw_answer.strip()


def extract_record(example: dict, extractor: str) -> Optional[dict]:
    """
    Convert a raw HuggingFace dataset row into a normalised SFT record.
    Returns None if the row should be skipped.
    """
    try:
        # ---- Math ----
        if extractor == "numina_math":
            prompt  = (example.get("problem") or "").strip()
            answer  = (example.get("solution") or "").strip()
            if not prompt or not answer:
                return None
            # NuminaMath solutions are long CoT traces; treat whole thing as
            # thinking, with the last sentence / boxed expression as the answer.
            thinking = answer
            # Try to pull out the final boxed answer as the short answer field
            boxed = _extract_boxed(answer)
            short_ans = boxed[-1] if boxed else answer.split("\n")[-1].strip()
            return {"prompt": prompt, "thinking": thinking, "answer": short_ans}

        if extractor == "metamath":
            prompt  = (example.get("query") or "").strip()
            resp    = (example.get("response") or "").strip()
            if not prompt or not resp:
                return None
            # MetaMath responses include reasoning then "The answer is X."
            m = re.search(r"[Tt]he answer is[:\s]*([^\.\n]+)", resp)
            short_ans = m.group(1).strip() if m else resp.split("\n")[-1].strip()
            return {"prompt": prompt, "thinking": resp, "answer": short_ans}

        if extractor == "gsm8k":
            prompt   = (example.get("question") or "").strip()
            raw      = (example.get("answer") or "").strip()
            thinking, answer = _gsm8k_split_thinking(raw)
            if not prompt or not answer:
                return None
            return {"prompt": prompt, "thinking": thinking, "answer": answer}

        # ---- Code ----
        if extractor == "instruction_output":
            prompt = (example.get("instruction") or "").strip()
            answer = (example.get("output") or "").strip()
            if not prompt or not answer:
                return None
            return {"prompt": prompt, "thinking": "", "answer": answer}

        if extractor == "alpaca_code":
            instr  = (example.get("instruction") or "").strip()
            inp    = (example.get("input") or "").strip()
            output = (example.get("output") or "").strip()
            prompt = f"{instr}\n\n{inp}".strip() if inp else instr
            if not prompt or not output:
                return None
            return {"prompt": prompt, "thinking": "", "answer": output}

        # ---- Reasoning ----
        if extractor == "open_thoughts":
            # OpenThoughts-114k: fields are 'problem' and 'solution'
            # solution usually starts with <think>...</think> block
            prompt   = (example.get("problem") or "").strip()
            solution = (example.get("solution") or "").strip()
            if not prompt or not solution:
                return None
            # Extract <think> block if present
            think_match = re.search(r"<think>(.*?)</think>", solution, re.DOTALL)
            if think_match:
                thinking = think_match.group(1).strip()
                answer   = solution[think_match.end():].strip()
            else:
                thinking = solution
                answer   = solution.split("\n")[-1].strip()
            return {"prompt": prompt, "thinking": thinking, "answer": answer}

        if extractor == "open_hermes":
            # OpenHermes: {"conversations": [{"from": "human", "value": ...},
            #                                {"from": "gpt",   "value": ...}]}
            convs = example.get("conversations") or []
            human_turns = [c["value"] for c in convs if c.get("from") == "human"]
            gpt_turns   = [c["value"] for c in convs if c.get("from") == "gpt"]
            if not human_turns or not gpt_turns:
                return None
            prompt = human_turns[0].strip()
            answer = gpt_turns[0].strip()
            if not prompt or not answer:
                return None
            return {"prompt": prompt, "thinking": "", "answer": answer}

        # ---- Science ----
        if extractor == "arc":
            question = (example.get("question") or "").strip()
            choices  = example.get("choices") or {}
            labels   = choices.get("label", [])
            texts    = choices.get("text",  [])
            ans_key  = (example.get("answerKey") or "").strip()
            if not question or not labels or not texts:
                return None
            # Format choices into the prompt so the model sees the options
            opts = "\n".join(f"  {l}. {t}" for l, t in zip(labels, texts))
            prompt = f"{question}\n{opts}"
            # Find the answer text
            answer = ""
            for l, t in zip(labels, texts):
                if l == ans_key:
                    answer = f"{l}. {t}"
                    break
            if not answer:
                return None
            return {"prompt": prompt, "thinking": "", "answer": answer}

        if extractor == "science_qa":
            question = (example.get("question") or "").strip()
            choices  = example.get("choices") or []
            solution = (example.get("solution") or "").strip()
            ans_idx  = example.get("answer")
            if not question:
                return None
            opts   = "\n".join(f"  {chr(65+i)}. {c}" for i, c in enumerate(choices))
            prompt = f"{question}\n{opts}".strip() if choices else question
            answer = ""
            if ans_idx is not None and isinstance(ans_idx, int) and ans_idx < len(choices):
                answer = f"{chr(65+ans_idx)}. {choices[ans_idx]}"
            if not answer:
                return None
            return {
                "prompt":   prompt,
                "thinking": solution,   # ScienceQA has a lecture/explanation field
                "answer":   answer,
            }

    except Exception:
        return None

    return None


# ---------------------------------------------------------------------------
# Size helpers  (identical interface to build_dataset.py)
# ---------------------------------------------------------------------------

def parse_size(s: str) -> int:
    s = s.strip().upper()
    for unit, mult in [("GB", 1024**3), ("MB", 1024**2), ("KB", 1024), ("B", 1)]:
        if s.endswith(unit):
            return int(float(s[:-len(unit)]) * mult)
    return int(float(s))


def parse_mix(mix_str: Optional[str]) -> dict:
    if mix_str is None:
        return DEFAULT_MIX
    mix = {}
    for part in mix_str.split(","):
        k, v = part.split("=")
        mix[k.strip()] = float(v)
    total = sum(mix.values())
    if abs(total - 1.0) > 1e-6:
        print(f"[warn] mix sums to {total:.4f}, normalising to 1.0")
        mix = {k: v / total for k, v in mix.items()}
    for k in mix:
        if k not in SOURCES:
            raise ValueError(f"Unknown category '{k}'. Valid: {list(SOURCES)}")
    return mix


# ---------------------------------------------------------------------------
# Shard writer  (writes SFT JSONL records, not plain text)
# ---------------------------------------------------------------------------

class ShardWriter:
    def __init__(self, out_dir: str, category: str,
                 max_shard_bytes: int = SHARD_MAX_BYTES):
        self.dir = os.path.join(out_dir, category)
        os.makedirs(self.dir, exist_ok=True)
        self.category      = category
        self.max_shard     = max_shard_bytes
        self.shard_idx     = 0
        self.bytes_in_shard = 0
        self.total_bytes   = 0
        self.total_docs    = 0
        self._fh           = self._open()

    def _open(self):
        p = os.path.join(self.dir, f"{self.category}_{self.shard_idx:05d}.jsonl")
        return open(p, "w", encoding="utf-8")

    def write(self, record: dict):
        line       = json.dumps(record, ensure_ascii=False) + "\n"
        line_bytes = len(line.encode("utf-8"))
        if self.bytes_in_shard + line_bytes > self.max_shard and self.bytes_in_shard > 0:
            self._fh.close()
            self.shard_idx     += 1
            self.bytes_in_shard = 0
            self._fh = self._open()
        self._fh.write(line)
        self.bytes_in_shard += line_bytes
        self.total_bytes    += line_bytes
        self.total_docs     += 1

    def close(self):
        self._fh.close()


# ---------------------------------------------------------------------------
# Per-category streaming loop
# ---------------------------------------------------------------------------

def stream_category(category: str, byte_budget: int, out_dir: str) -> tuple:
    sources    = SOURCES[category]
    writer     = ShardWriter(out_dir, category)
    src_idx    = 0
    last_print = time.time()
    # Track which sources have been fully exhausted so the round-robin loop
    # never reopens (and silently re-streams/duplicates) one that's already
    # been read to the end -- a streaming HF dataset restarts from example 0
    # every time load_dataset() is called on it again.
    exhausted  = set()

    print(f"\n=== [{category}] target {byte_budget / 1024**2:.1f} MB "
          f"from {len(sources)} source(s) ===")

    while writer.total_bytes < byte_budget:
        if len(exhausted) >= len(sources):
            print(f"\n[{category}] all sources exhausted at "
                  f"{writer.total_bytes / 1024**2:.1f} MB "
                  f"(target was {byte_budget / 1024**2:.1f} MB) -- stopping, "
                  f"no data will be duplicated")
            break

        src = sources[src_idx % len(sources)]
        src_idx += 1
        if src["path"] in exhausted:
            continue

        try:
            ds = load_dataset(
                src["path"],
                src.get("name"),
                split=src["split"],
                streaming=True,
            )
        except Exception as e:
            print(f"\n[error] could not open {src['path']}: {e}")
            if src_idx >= len(sources) * 3:
                print(f"[abort] no usable sources for '{category}'")
                break
            continue

        try:
            for example in ds:
                rec = extract_record(example, src["extractor"])
                if rec is None:
                    continue
                if (len(rec["prompt"]) < MIN_PROMPT_CHARS or
                        len(rec["answer"]) < MIN_ANSWER_CHARS):
                    continue
                rec["source"]   = src["path"]
                rec["category"] = category
                writer.write(rec)

                if writer.total_bytes >= byte_budget:
                    break

                if time.time() - last_print > 5:
                    pct = 100 * writer.total_bytes / byte_budget
                    print(f"[{category}] {writer.total_bytes / 1024**2:8.2f} MB"
                          f" / {byte_budget / 1024**2:.1f} MB  ({pct:5.1f}%)"
                          f"  docs={writer.total_docs:,}", end="\r")
                    last_print = time.time()
            else:
                # Inner for-loop ran to completion without `break` -> the
                # stream is exhausted. Never reopen it.
                exhausted.add(src["path"])

        except Exception as e:
            print(f"\n[warn] stream error for {src['path']}: {e} — trying next source")
            exhausted.add(src["path"])
            continue

        if writer.total_bytes >= byte_budget:
            break

    writer.close()
    print(f"\n[{category}] done — {writer.total_bytes / 1024**2:.2f} MB, "
          f"{writer.total_docs:,} records, {writer.shard_idx + 1} shard(s)")
    return writer.total_bytes, writer.total_docs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Download and format SFT reasoning data from HuggingFace."
    )
    p.add_argument("--target-size", required=True,
                   help="Total dataset size, e.g. 2GB, 500MB")
    p.add_argument("--out-dir",  default="./sft_data",
                   help="Output directory (default ./sft_data)")
    p.add_argument("--mix",      default=None,
                   help="Comma-separated category=fraction overrides, "
                        "e.g. math=0.5,code=0.3,reasoning=0.2")
    args = p.parse_args()

    target_bytes = parse_size(args.target_size)
    mix          = parse_mix(args.mix)

    print(f"Target size      : {target_bytes / 1024**2:.1f} MB")
    print(f"Output directory : {args.out_dir}")
    print(f"Mix              : {mix}")

    os.makedirs(args.out_dir, exist_ok=True)

    manifest = {
        "target_bytes": target_bytes,
        "mix":          mix,
        "categories":   {},
        "format": {
            "fields":  ["prompt", "thinking", "answer", "source", "category"],
            "note":    "'thinking' may be empty string for datasets without CoT traces",
        },
    }

    for category, frac in mix.items():
        budget = int(target_bytes * frac)
        if budget <= 0:
            continue
        actual_bytes, docs = stream_category(category, budget, args.out_dir)
        manifest["categories"][category] = {
            "target_bytes": budget,
            "actual_bytes": actual_bytes,
            "docs":         docs,
        }

    manifest_path = os.path.join(args.out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    total = sum(c["actual_bytes"] for c in manifest["categories"].values())
    total_docs = sum(c["docs"] for c in manifest["categories"].values())
    print(f"\n=== Done — {total / 1024**2:.1f} MB, {total_docs:,} records ===")
    print(f"Manifest : {manifest_path}")
    print(f"\nNext step:\n"
          f"  python sft.py --checkpoint ./checkpoints/latest.pt "
          f"--data-dir {args.out_dir} --tokenizer ./tokenizer")


if __name__ == "__main__":
    main()
