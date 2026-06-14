#!/usr/bin/env python3
"""
train_tokenizer.py

Trains a byte-level BPE (BBPE) tokenizer -- the same family as Qwen3's
tokenizer -- on the JSONL shards produced by build_dataset.py (./data/*/*.jsonl).

Includes:
  - Byte-level pre-tokenization (handles any UTF-8 text, no <unk> needed)
  - Configurable vocab size (Qwen3 uses ~151K)
  - Special tokens for chat formatting + reasoning (<think>/</think>)
  - A standalone chat template (Jinja) saved alongside the tokenizer

Usage:
    python train_tokenizer.py --data-dir ./data --vocab-size 32000 --out-dir ./tokenizer
    python train_tokenizer.py --vocab-size 151643   # full Qwen3-scale vocab
"""

import argparse
import glob
import json
import os
from typing import Iterator

from tokenizers import Tokenizer, pre_tokenizers, decoders, processors
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import NFC


# ---------------------------------------------------------------------------
# Special tokens
# ---------------------------------------------------------------------------
# Mirrors the structure Qwen3 uses: ChatML-style role tags plus explicit
# <think>/</think> tags for reasoning traces.

SPECIAL_TOKENS = [
    "<|endoftext|>",     # document separator / EOS
    "<|pad|>",           # padding
    "<|im_start|>",      # ChatML turn start
    "<|im_end|>",        # ChatML turn end
    "<think>",           # reasoning block start
    "</think>",          # reasoning block end
    "<|tool_call|>",     # tool/function call start
    "<|tool_call_end|>", # tool/function call end
    "<|fim_prefix|>",    # fill-in-the-middle (useful for code data)
    "<|fim_middle|>",
    "<|fim_suffix|>",
]

CHAT_TEMPLATE = """{% for message in messages %}\
{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}\
{% endfor %}\
{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"""


# ---------------------------------------------------------------------------
# Corpus iterator
# ---------------------------------------------------------------------------

def iter_corpus(data_dir: str, max_docs: int = None) -> Iterator[str]:
    """Yield the 'text' field from every JSONL shard under data_dir/<category>/."""
    shard_paths = sorted(glob.glob(os.path.join(data_dir, "*", "*.jsonl")))
    if not shard_paths:
        raise FileNotFoundError(
            f"No .jsonl shards found under {data_dir}/<category>/. "
            f"Did you run build_dataset.py first?"
        )

    print(f"Found {len(shard_paths)} shard file(s) under {data_dir}")

    n = 0
    for path in shard_paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = record.get("text")
                if text:
                    yield text
                    n += 1
                    if max_docs and n >= max_docs:
                        return


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_tokenizer(data_dir: str, vocab_size: int, min_frequency: int,
                     max_docs: int, out_dir: str):
    tokenizer = Tokenizer(BPE(unk_token=None, byte_fallback=True))

    # Byte-level pre-tokenizer: splits on a GPT-2-style regex, maps raw bytes
    # to printable unicode -> every byte sequence is representable, no <unk>.
    tokenizer.normalizer = NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    print(f"Training BPE tokenizer: vocab_size={vocab_size}, "
          f"min_frequency={min_frequency}, max_docs={max_docs or 'all'}")

    tokenizer.train_from_iterator(
        iter_corpus(data_dir, max_docs=max_docs),
        trainer=trainer,
    )

    # Post-processor: wrap encoded sequences with EOS so packed pretraining
    # data has clean document boundaries.
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="$A <|endoftext|>",
        pair="$A <|endoftext|> $B <|endoftext|>",
        special_tokens=[("<|endoftext|>", eos_id)],
    )

    os.makedirs(out_dir, exist_ok=True)
    tokenizer_path = os.path.join(out_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    print(f"Saved tokenizer to {tokenizer_path}")

    # Save a HF-style config bundle for easy loading with
    # transformers.PreTrainedTokenizerFast
    tokenizer_config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_max_length": 32768,
        "bos_token": None,
        "eos_token": "<|endoftext|>",
        "pad_token": "<|pad|>",
        "additional_special_tokens": SPECIAL_TOKENS,
        "chat_template": CHAT_TEMPLATE,
    }
    with open(os.path.join(out_dir, "tokenizer_config.json"), "w") as f:
        json.dump(tokenizer_config, f, indent=2)

    special_tokens_map = {
        "eos_token": "<|endoftext|>",
        "pad_token": "<|pad|>",
        "additional_special_tokens": SPECIAL_TOKENS,
    }
    with open(os.path.join(out_dir, "special_tokens_map.json"), "w") as f:
        json.dump(special_tokens_map, f, indent=2)

    print(f"Saved tokenizer_config.json and special_tokens_map.json to {out_dir}")
    return tokenizer


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

def sanity_check(tokenizer: Tokenizer, out_dir: str):
    print("\n=== Sanity check ===")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")

    for tok in ["<|endoftext|>", "<|im_start|>", "<think>", "</think>", "<|pad|>"]:
        tid = tokenizer.token_to_id(tok)
        print(f"  {tok!r:18s} -> id {tid}")

    samples = [
        "Hello, world! This is a test.",
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "<think>Let's break this down step by step. First, 2+2=4.</think>\nThe answer is 4.",
        "The integral of x^2 dx is x^3/3 + C.",
    ]

    for s in samples:
        enc = tokenizer.encode(s)
        # decode with special tokens kept, to verify lossless round-trip
        # including <think>/</think>/<|im_start|> etc.
        decoded_full = tokenizer.decode(enc.ids, skip_special_tokens=False)
        decoded_clean = tokenizer.decode(enc.ids, skip_special_tokens=True)
        n_tokens = len(enc.ids)
        ratio = len(s) / max(n_tokens, 1)
        print(f"\nInput        : {s[:80]!r}")
        print(f"Tokens       : {n_tokens} (chars/token = {ratio:.2f})")
        print(f"Decoded full : {decoded_full[:90]!r}")
        print(f"Decoded clean: {decoded_clean[:80]!r}")

        # The lossless check: input text must be a substring of the
        # full decode (special tokens like <|endoftext|> get appended).
        assert s in decoded_full, "Round-trip mismatch (check byte_fallback / NFC settings)"

    print("\nAll round-trip checks passed.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train a Qwen3-style BBPE tokenizer.")
    parser.add_argument("--data-dir", default="./data", help="Directory with JSONL shards (from build_dataset.py)")
    parser.add_argument("--out-dir", default="./tokenizer", help="Output directory for tokenizer files")
    parser.add_argument("--vocab-size", type=int, default=32000,
                         help="Target vocab size (Qwen3 uses ~151643 for full-scale)")
    parser.add_argument("--min-frequency", type=int, default=2,
                         help="Minimum pair frequency to merge")
    parser.add_argument("--max-docs", type=int, default=None,
                         help="Optional cap on number of documents used for training (for speed on huge corpora)")
    args = parser.parse_args()

    tokenizer = train_tokenizer(
        data_dir=args.data_dir,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        max_docs=args.max_docs,
        out_dir=args.out_dir,
    )
    sanity_check(tokenizer, args.out_dir)


if __name__ == "__main__":
    main()
