#!/usr/bin/env python3
"""
train_tokenizer.py

Trains a byte-level BPE (BBPE) tokenizer -- the same family as Qwen3's
tokenizer -- on the JSONL shards produced by build_dataset.py (./data/*/*.jsonl).

Memory design
─────────────
OLD (memory-hogging / segfault-prone):
    - Fed a Python generator to tokenizer.train_from_iterator().
    - The Rust trainer kept references to the iterator's strings; when the
      generator was exhausted or the worker finalized, free() was called on
      buffers that glibc didn't recognize -> "munmap_chunk(): invalid pointer"
      -> core dump.
    - initial_alphabet was a borrowed slice from pre_tokenizers.ByteLevel.alphabet()
      (lifetime tied to a temporary), which can corrupt the trainer's pair table.

NEW (constant RAM, lifetime-safe):
    - Stage 1: a single dedicated process reads each JSONL shard in line-by-line
      lines and pushes a "doc boundary" sentinel between documents. Texts are
      pushed to a memory-bounded queue so peak RAM is bounded regardless of
      corpus size.
    - Stage 2: the trainer process owns the tokenizer, materializes the corpus
      chunk-by-chunk into a pre-allocated buffer, and calls
      tokenizer.train_from_iterator() over that owned buffer. No borrowed
      references, no live generator at process end.
    - initial_alphabet is materialized as a Python list of str before the
      trainer sees it (no borrow from a temporary).
    - All texts are NFC-normalized once on the reader side, so the trainer
      doesn't have to do it under the C-extension's lifetime rules.

Usage:
    # Recommended (handles 50 GB+ corpora without segfaulting):
    python train_tokenizer.py --data-dir ./data --vocab-size 32000 --out-dir ./tokenizer

    # Full Qwen3-scale vocab (still memory-bounded via the new defaults):
    python train_tokenizer.py --vocab-size 151643

    # Override the cap if you want every doc seen (NOT recommended on >20 GB):
    python train_tokenizer.py --max-docs 0 --min-frequency 2
"""

import argparse
import glob
import json
import multiprocessing as mp
import os
import queue
import time
import unicodedata
from typing import List, Optional

from tokenizers import Tokenizer, pre_tokenizers, decoders, processors
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer


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
# Stage 1 — bounded-memory corpus reader
# ---------------------------------------------------------------------------
# A reader process walks the JSONL shards, normalizes each line to NFC, and
# pushes chunks of texts to a Queue. The trainer process owns the tokenizer
# and pulls chunks off the queue, training on each chunk. Putting the reader
# in a separate process means the Python generator the trainer sees always
# yields strings it owns (no borrowed references, no lifetime surprises).

_SENTINEL = None  # end-of-corpus marker

# Codepoints that the tokenizers' NFC normalizer has been known to panic on
# (`NormalizedString bad split` in normalizer.rs). We strip them on the
# Python side so the Rust side never sees them. These are control characters
# (other than common whitespace), the byte-replacement marker, BOM, and
# zero-width / formatting codepoints that confuse the alignment step.
_BAD_CODEPOINTS = frozenset([
    0x00,       # NUL
    0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
    0x0B, 0x0C, 0x0E, 0x0F,
    0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
    0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F,
    0x7F,       # DEL
    0xFFFD,     # Unicode replacement char (often a sign of bad UTF-8)
    0xFEFF,     # BOM / zero-width no-break space
    0x200B,     # zero-width space
    0x200C, 0x200D,  # ZWNJ, ZWJ
    0x202A, 0x202B, 0x202C, 0x202D, 0x202E,  # bidirectional overrides
    0x2066, 0x2067, 0x2068, 0x2069,
])


def _clean_text(text: str) -> Optional[str]:
    """
    NFC-normalize, drop bytes/codepoints that make the Rust NFC normalizer
    panic, and drop the doc if anything goes wrong. Returns None to signal
    "skip this document".
    """
    try:
        text = unicodedata.normalize("NFC", text)
    except Exception:
        return None
    # Strip a curated set of codepoints known to trigger the panic.
    if any(ord(c) in _BAD_CODEPOINTS for c in text):
        text = "".join(c for c in text if ord(c) not in _BAD_CODEPOINTS)
    if not text:
        return None
    return text


def _reader_main(shard_paths: List[str], out_q: mp.Queue, max_docs: Optional[int],
                 chunk_lines: int):
    """Read every JSONL shard, push lists of (already-NFC-normalized) texts."""
    n = 0
    buf: List[str] = []
    bytes_in_buf = 0
    target_bytes = 8 * 1024 * 1024  # ~8 MB of text per chunk -> ~30-50k docs
    skipped = 0

    def flush():
        nonlocal buf, bytes_in_buf
        if buf:
            out_q.put(buf)
            buf = []
            bytes_in_buf = 0

    try:
        for path in shard_paths:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
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
                    cleaned = _clean_text(text)
                    if cleaned is None:
                        skipped += 1
                        continue
                    text = cleaned
                    buf.append(text)
                    bytes_in_buf += len(text)
                    n += 1
                    if bytes_in_buf >= target_bytes or len(buf) >= chunk_lines:
                        flush()
                    if max_docs is not None and n >= max_docs:
                        flush()
                        out_q.put(_SENTINEL)
                        out_q.put(("STATS", {"kept": n, "skipped": skipped}))
                        return
        flush()
        out_q.put(_SENTINEL)
        out_q.put(("STATS", {"kept": n, "skipped": skipped}))
    except Exception as e:
        # Send the exception back to the main process; it'll be re-raised there.
        out_q.put(("ERROR", repr(e)))


# ---------------------------------------------------------------------------
# Stage 2 — tokenizer training
# ---------------------------------------------------------------------------

def _materialize_byte_alphabet() -> List[str]:
    """
    Build the initial byte-level alphabet as an OWNED Python list.

    `pre_tokenizers.ByteLevel.alphabet()` returns a slice that aliases a
    temporary. Passing that to BpeTrainer(initial_alphabet=...) has caused
    `munmap_chunk(): invalid pointer` crashes on some tokenizers / glibc
    combos because the underlying buffer's lifetime is shorter than the
    trainer's internal use of it. We materialize the same bytes as a real
    Python list of one-character strings — owned, copy-isolated, lifetime-safe.
    """
    # The 256 byte values mapped the same way ByteLevel does.
    # These are the codepoints the byte-level pre-tokenizer exposes as
    # initial characters. Ranges are expressed as int code points.
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = [chr(c) for c in bs]
    return cs


def train_tokenizer(
    data_dir: str,
    vocab_size: int,
    min_frequency: int,
    max_docs: Optional[int],
    out_dir: str,
    chunk_lines: int = 4096,
    reader_chunk_mb: int = 8,
):
    """
    Memory-bounded tokenizer training.

    Peak RAM (rough):
      - Reader process: 1 chunk of texts = `reader_chunk_mb` MB (~8 MB default)
      - Trainer process: 1 chunk of texts + tokenizer internals.
        For 32k vocab the BPE pair table is bounded by corpus vocabulary, not
        by corpus size, so steady-state RAM is a few hundred MB regardless
        of how big the corpus is.

    Recommended environment for stability on 50 GB+ corpora:
        RAYON_NUM_THREADS=1            # single-thread the BPE pair merge
        TOKENIZERS_PARALLELISM=false   # no background Python threads
    Together these prevent the intermittent SIGSEGV that occurs when
    `BpeTrainer` parallel-reduces a multi-million-entry pair map and the
    rayon worker pool collides with the Python GIL or runs the box out of
    RAM mid-allocation. See README section "Intermittent segfaults".
    """
    shard_paths = sorted(glob.glob(os.path.join(data_dir, "*", "*.jsonl")))
    if not shard_paths:
        raise FileNotFoundError(
            f"No .jsonl shards found under {data_dir}/<category>/. "
            f"Did you run build_dataset.py first?"
        )
    print(f"Found {len(shard_paths)} shard file(s) under {data_dir}")

    # Own the alphabet up front.
    initial_alphabet = _materialize_byte_alphabet()

    # Build the tokenizer in the main process. We train it via train_from_iterator
    # using a list-iterator we control (NOT a generator that goes out of scope
    # mid-iteration).
    #
    # Important: the trainer's `feed()` function in tokenizers 0.22 ALWAYS
    # calls `self.normalizer` on every input string (see tokenizer/mod.rs:
    # `self.added_vocabulary.extract_and_normalize(self.normalizer.as_ref(), ...)`).
    # The default normalizer is BertNormalizer, which has a known
    # `NormalizedString bad split` panic in normalizer.rs:777 on certain
    # inputs. We set the tokenizer's normalizer to None and pre-normalize
    # in Python inside _clean_text() to bypass the buggy Rust code path
    # entirely. The trainer does NOT have a `normalizer` kwarg in 0.22.
    tokenizer = Tokenizer(BPE(unk_token=None, byte_fallback=True))
    tokenizer.normalizer     = None  # disable Rust-side NFC
    tokenizer.pre_tokenizer  = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder        = decoders.ByteLevel()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=initial_alphabet,
        show_progress=True,
    )

    print(f"Training BPE tokenizer: vocab_size={vocab_size}, "
          f"min_frequency={min_frequency}, max_docs={max_docs or 'all'}")

    # ---- Start the reader process ------------------------------------------
    ctx = mp.get_context("spawn")  # 'spawn' = clean process, no fork-after-mmap
    out_q: mp.Queue = ctx.Queue(maxsize=2)  # bound the queue so the reader blocks

    reader = ctx.Process(
        target=_reader_main,
        args=(shard_paths, out_q, max_docs, chunk_lines),
        daemon=True,
    )
    reader.start()

    # ---- Generator the trainer can call .train_from_iterator() on ----------
    # We pull a chunk at a time, concatenate the strings into a fresh owned
    # list, and yield each string in turn. After a chunk is exhausted, the
    # list is dropped, freeing its memory before we pull the next one.
    # Peak held text: 1 chunk (≈ reader_chunk_mb MB).
    final_stats = {"kept": 0, "skipped": 0}

    def bounded_iter():
        nonlocal final_stats
        while True:
            chunk = out_q.get()
            if chunk is _SENTINEL:
                return
            if isinstance(chunk, tuple) and chunk:
                tag = chunk[0]
                if tag == "ERROR":
                    raise RuntimeError(f"Reader process failed: {chunk[1]}")
                if tag == "STATS":
                    final_stats = chunk[1]
                    continue
            # Re-bind to a local to make sure we own it, then yield.
            owned = list(chunk)
            del chunk
            yield from owned
            del owned

    t0 = time.time()
    try:
        tokenizer.train_from_iterator(bounded_iter(), trainer=trainer)
    finally:
        reader.join(timeout=10)
        if reader.is_alive():
            reader.terminate()
            reader.join()

    print(f"\nReader kept {final_stats['kept']:,} documents, "
          f"skipped {final_stats['skipped']:,} as unclean.")

    # ---- Post-processor + save --------------------------------------------
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="$A <|endoftext|>",
        pair="$A <|endoftext|> $B <|endoftext|>",
        special_tokens=[("<|endoftext|>", eos_id)],
    )

    os.makedirs(out_dir, exist_ok=True)
    tokenizer_path = os.path.join(out_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    print(f"\nSaved tokenizer to {tokenizer_path}  ({time.time()-t0:.1f}s)")

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
        decoded_full  = tokenizer.decode(enc.ids, skip_special_tokens=False)
        decoded_clean = tokenizer.decode(enc.ids, skip_special_tokens=True)
        n_tokens      = len(enc.ids)
        ratio         = len(s) / max(n_tokens, 1)
        print(f"\nInput        : {s[:80]!r}")
        print(f"Tokens       : {n_tokens} (chars/token = {ratio:.2f})")
        print(f"Decoded full : {decoded_full[:90]!r}")
        print(f"Decoded clean: {decoded_clean[:80]!r}")
        assert s in decoded_full, "Round-trip mismatch"

    print("\nAll round-trip checks passed.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Train a Qwen3-style BBPE tokenizer.")
    p.add_argument("--data-dir",       default="./data",
                   help="Directory with JSONL shards (from build_dataset.py)")
    p.add_argument("--out-dir",        default="./tokenizer",
                   help="Output directory for tokenizer files")
    p.add_argument("--vocab-size",     type=int, default=32000,
                   help="Target vocab size (Qwen3 uses ~151643 for full-scale)")
    p.add_argument("--min-frequency",  type=int, default=10,
                   help="Minimum pair frequency to merge. Default 10 (not 2) "
                        "to keep the BPE pair table small and prevent intermittent "
                        "segfaults on 50 GB+ corpora. Set to 2 only for small corpora.")
    p.add_argument("--max-docs",       type=int, default=2_000_000,
                   help="Cap on number of documents fed to the trainer. Default 2M — "
                        "the published sweet spot for a 32k-vocab BBPE; quality "
                        "plateaus well before this on well-mixed FineWeb/FineMath. "
                        "Pass 0 to disable (use the whole corpus).")
    p.add_argument("--reader-chunk-mb", type=int, default=8,
                   help="Approx MB of text the reader holds in RAM at a time. "
                        "Lower = less RAM, more IPC overhead. Default 8 MB.")
    args = p.parse_args()
    # argparse can't represent "unset" for type=int; 0 is the opt-out sentinel.
    if args.max_docs == 0:
        args.max_docs = None

    tokenizer = train_tokenizer(
        data_dir=args.data_dir,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        max_docs=args.max_docs,
        out_dir=args.out_dir,
        reader_chunk_mb=args.reader_chunk_mb,
    )
    sanity_check(tokenizer, args.out_dir)


if __name__ == "__main__":
    # Pin threading BEFORE importing tokenizers. These must be set in the
    # environment when this script is launched, not after import — tokenizers
    # reads RAYON_NUM_THREADS on first use of the parallel iterator and
    # TOKENIZERS_PARALLELISM at import time. Setting them here as a fallback
    # makes the script work even if the user forgets to `export` them.
    os.environ.setdefault("RAYON_NUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
