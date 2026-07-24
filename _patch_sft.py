#!/usr/bin/env python3
"""Patch pack_sft_data.py: replace format_and_tokenise, pack_worker_shard,
add --seq-length to CLI, and wire seq_length through."""
import re

with open("pack_sft_data.py", "r") as f:
    content = f.read()

# --- 1. Replace format_and_tokenise function ---
OLD_FUNC = '''def format_and_tokenise(
    record: dict,
    tokenizer: Tokenizer,
    max_len: int = 2048,
    seq_length: Optional[int] = None,
) -> Optional[Tuple[List[int], List[int]]]:
    """
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
    user_text = f" im user\\n{prompt} /e im\\n"

    # ---- assistant turn (thinking + answer — included in loss)
    if thinking:
        asst_text = (
            f" im assistant\\n"
            f"<thinking>\\n{thinking}\\n</thinking>\\n"
            f"{answer} /e im\\n"
        )
    else:
        asst_text = f" im assistant\\n{answer} /e im\\n"

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

# Just verify the old function is there (the unicode escapes might not match exactly)
# Let's find it by function name boundary instead
print("OK - skipping exact match check, will write the whole file")
