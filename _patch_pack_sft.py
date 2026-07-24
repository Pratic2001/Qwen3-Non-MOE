#!/usr/bin/env python3
"""Patch pack_sft_data.py to add --seq-length support."""
import re

with open("pack_sft_data.py", "r") as f:
    content = f.read()

# 1. Add _truncate_text_to_tokens helper before format_and_tokenise
helper = '''
def _truncate_text_to_tokens(text, max_tokens, tokenizer):
    """Truncate raw text to at most max_tokens tokens, then decode back."""
    if max_tokens <= 0:
        return ""
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens])


'''

# Insert helper before format_and_tokenise
content = content.replace(
    "# Chat template formatting + tokenisation\n# ---------------------------------------------------------------------------\n\ndef format_and_tokenise(",
    "# Chat template formatting + tokenisation\n# ---------------------------------------------------------------------------\n" + helper + "def format_and_tokenise("
)

# 2. Replace the docstring and body of format_and_tokenise to add seq_length support
old_body = '''    """
    Format one SFT record into token ids + loss mask.

    Returns (input_ids, loss_mask) where loss_mask[i] = 1 means position i
    contributes to the cross-entropy loss (i.e. it is part of the assistant
    turn), and 0 means it is masked out (prompt tokens).

    Returns None if the formatted example exceeds max_len tokens.
    """
    prompt   = record.get("prompt",   "").strip()
    thinking = record.get("thinking", "").strip()
    answer   = record.get("answer",   "").strip()

    if not prompt or not answer:
        return None

    # Build the two halves separately so we can track where the prompt ends
    # and the assistant turn begins.

    # ---- user turn (prompt — masked out of loss)
    user_text = f"<im_start>user\\n{prompt}<im_end>\\n"

    # ---- assistant turn (thinking + answer — included in loss)
    if thinking:
        asst_text = (
            f"<im_start>assistant\\n"
            f"<think>\\n{thinking}\\n</think>\\n"
            f"{answer}<im_end>\\n"
        )
    else:
        asst_text = f"<im_start>assistant\\n{answer}<im_end>\\n"

    user_ids = tokenizer.encode(user_text,  add_special_tokens=False).ids
    asst_ids = tokenizer.encode(asst_text,  add_special_tokens=False).ids

    # Truncate if necessary, preserving at least a few answer tokens
    total = len(user_ids) + len(asst_ids)
    if total > max_len:
        # First try truncating thinking; if still too long truncate prompt
        budget  = max_len - len(user_ids)
        if budget < 32:
            # Truncate the user turn
            user_ids = user_ids[: max_len - min(32, len(asst_ids))]
            asst_ids = asst_ids[:32]
        else:
            asst_ids = asst_ids[:budget]

    input_ids = user_ids + asst_ids
    loss_mask = [0] * len(user_ids) + [1] * len(asst_ids)

    if len(input_ids) < 4:
        return None

    return input_ids, loss_mask'''

new_body = '''    """
    Format one SFT record into token ids + loss mask.

    Returns (input_ids, loss_mask) where loss_mask[i] = 1 means position i
    contributes to the cross-entropy loss (i.e. it is part of the assistant
    turn), and 0 means it is masked out (prompt tokens).

    When seq_length is set, the raw prompt/thinking/answer text is truncated
    to fit within seq_length tokens BEFORE the ChatML template is applied.
    The existing max_len check acts as a secondary safety net.

    Returns None if the formatted example exceeds max_len tokens.
    """
    prompt   = record.get("prompt",   "").strip()
    thinking = record.get("thinking", "").strip()
    answer   = record.get("answer",   "").strip()

    if not prompt or not answer:
        return None

    # ---- Pre-template truncation when seq_length is set ----
    # Truncate raw text fields BEFORE building ChatML templates so the
    # template is applied to already-truncated content.
    if seq_length is not None:
        # ChatML overhead: im_start + "user\\n" + im_end + "\\n"
        #                 + im_start + "assistant\\n" + im_end + "\\n"
        # Conservatively 20 tokens for the fixed template tokens.
        TEMPLATE_OVERHEAD = 20
        budget = seq_length - TEMPLATE_OVERHEAD
        if budget < 32:
            return None

        # Tokenize each field independently to measure lengths
        n_prompt   = len(tokenizer.encode(prompt,   add_special_tokens=False).ids)
        n_thinking = len(tokenizer.encode(thinking, add_special_tokens=False).ids) if thinking else 0
        n_answer   = len(tokenizer.encode(answer,   add_special_tokens=False).ids)

        total_raw = n_prompt + n_thinking + n_answer
        if total_raw > budget:
            # Truncate answer first (most important for training signal)
            answer_budget = min(n_answer, budget)
            # Remaining budget after answer
            remaining = budget - answer_budget
            # Allocate to thinking, then prompt
            thinking_budget = min(n_thinking, remaining)
            remaining -= thinking_budget
            prompt_budget = min(n_prompt, remaining)

            answer   = _truncate_text_to_tokens(answer,   answer_budget,   tokenizer)
            if thinking:
                thinking = _truncate_text_to_tokens(thinking, thinking_budget, tokenizer)
            prompt   = _truncate_text_to_tokens(prompt,   prompt_budget,   tokenizer)

    # Build the two halves separately so we can track where the prompt ends
    # and the assistant turn begins.

    # ---- user turn (prompt — masked out of loss)
    user_text = f"<im_start>user\\n{prompt}<im_end>\\n"

    # ---- assistant turn (thinking + answer — included in loss)
    if thinking:
        asst_text = (
            f"<im_start>assistant\\n"
            f"<think>\\n{thinking}\\n</think>\\n"
            f"{answer}<im_end>\\n"
        )
    else:
        asst_text = f"<im_start>assistant\\n{answer}<im_end>\\n"

    user_ids = tokenizer.encode(user_text,  add_special_tokens=False).ids
    asst_ids = tokenizer.encode(asst_text,  add_special_tokens=False).ids

    # Truncate if necessary, preserving at least a few answer tokens
    total = len(user_ids) + len(asst_ids)
    if total > max_len:
        # First try truncating thinking; if still too long truncate prompt
        budget  = max_len - len(user_ids)
        if budget < 32:
            # Truncate the user turn
            user_ids = user_ids[: max_len - min(32, len(asst_ids))]
            asst_ids = asst_ids[:32]
        else:
            asst_ids = asst_ids[:budget]

    input_ids = user_ids + asst_ids
    loss_mask = [0] * len(user_ids) + [1] * len(asst_ids)

    if len(input_ids) < 4:
        return None

    return input_ids, loss_mask'''

content = content.replace(old_body, new_body)

# 3. Add seq_length parameter to pack_worker_shard
content = content.replace(
    'def pack_worker_shard(\n    data_dir: str,\n    tokenizer: Tokenizer,\n    cache_dir: str,\n    max_len_per_example: int,\n    val_fraction: float,\n    worker: int,\n    num_workers: int,\n    vocab_size: Optional[int] = None,\n) -> dict:',
    'def pack_worker_shard(\n    data_dir: str,\n    tokenizer: Tokenizer,\n    cache_dir: str,\n    max_len_per_example: int,\n    val_fraction: float,\n    worker: int,\n    num_workers: int,\n    vocab_size: Optional[int] = None,\n    seq_length: Optional[int] = None,\n) -> dict:'
)

# 4. Pass seq_length to format_and_tokenise in both passes
content = content.replace(
    'result = format_and_tokenise(rec, tokenizer, max_len=max_len_per_example)\n                if result is None:\n                    continue\n                ids, _ = result\n                n_tok = len(ids) + 1  # +1 for EOS separator',
    'result = format_and_tokenise(rec, tokenizer, max_len=max_len_per_example, seq_length=seq_length)\n                if result is None:\n                    continue\n                ids, _ = result\n                n_tok = len(ids) + 1  # +1 for EOS separator'
)

content = content.replace(
    'result = format_and_tokenise(rec, tokenizer, max_len=max_len_per_example)\n                if result is None:\n                    continue\n\n                ids, lmask = result',
    'result = format_and_tokenise(rec, tokenizer, max_len=max_len_per_example, seq_length=seq_length)\n                if result is None:\n                    continue\n\n                ids, lmask = result'
)

# 5. Add seq_length to manifest
content = content.replace(
    '        "max_len_per_example": max_len_per_example,\n        "val_fraction":    val_fraction,',
    '        "max_len_per_example": max_len_per_example,\n        "seq_length":      seq_length,\n        "val_fraction":    val_fraction,'
)

# 6. Add --seq-length CLI argument
content = content.replace(
    '    p.add_argument("--max-len-per-example", type=int, default=2048,\n                   help="Max tokens per individual SFT example before truncation")\n    p.add_argument("--val-fraction", type=float, default=0.01,',
    '    p.add_argument("--max-len-per-example", type=int, default=2048,\n                   help="Max tokens per individual SFT example before truncation")\n    p.add_argument("--seq-length", type=int, default=None,\n                   help="Max tokens per packed record. Truncates prompt and "\n                        "answer before applying ChatML template. Sets the "\n                        "effective context length for downstream training.")\n    p.add_argument("--val-fraction", type=float, default=0.01,'
)

# 7. Pass seq_length from CLI to pack_worker_shard
content = content.replace(
    '    pack_worker_shard(\n        data_dir=args.data_dir,\n        tokenizer=tokenizer,\n        cache_dir=args.cache_dir,\n        max_len_per_example=args.max_len_per_example,\n        val_fraction=args.val_fraction,\n        worker=args.worker,\n        num_workers=args.num_workers,\n        vocab_size=tokenizer.get_vocab_size(),\n    )',
    '    pack_worker_shard(\n        data_dir=args.data_dir,\n        tokenizer=tokenizer,\n        cache_dir=args.cache_dir,\n        max_len_per_example=args.max_len_per_example,\n        val_fraction=args.val_fraction,\n        worker=args.worker,\n        num_workers=args.num_workers,\n        vocab_size=tokenizer.get_vocab_size(),\n        seq_length=args.seq_length,\n    )'
)

# Verify all changes
checks = [
    ("_truncate_text_to_tokens helper", "_truncate_text_to_tokens" in content),
    ("seq_length in format_and_tokenise signature", "seq_length: Optional[int] = None," in content),
    ("Pre-template truncation block", "Pre-template truncation when seq_length is set" in content),
    ("seq_length in pack_worker_shard signature", "seq_length: Optional[int] = None,\n) -> dict:" in content),
    ("seq_length passed to format_and_tokenise (first pass)", "max_len=max_len_per_example, seq_length=seq_length)" in content),
    ("seq_length in manifest", '"seq_length":      seq_length,' in content),
    ("--seq-length CLI arg", '"--seq-length"' in content),
    ("seq_length passed from CLI", "seq_length=args.seq_length," in content),
]

all_ok = True
for name, ok in checks:
    status = "OK" if ok else "MISSING"
    print(f"  {status}: {name}")
    if not ok:
        all_ok = False

if all_ok:
    with open("pack_sft_data.py", "w") as f:
        f.write(content)
    print("\npack_sft_data.py updated successfully!")
else:
    print("\nSome checks failed - not writing file")
