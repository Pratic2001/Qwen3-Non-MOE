#!/usr/bin/env python3
"""
train_grpo.py

Stage 2 of post-training: Group Relative Policy Optimization (GRPO) for the
Qwen3-style dense model from `model.py`.

GRPO is the algorithm behind DeepSeek-R1 / Open-R1 reasoning fine-tunes. We
consume the SFT checkpoint produced by `train_sft.py` and a pool of
`{prompt, answer}` records. By default we re-use the **packed memmaps**
written by `pack_sft_data.py` (the same files `train_sft.py` reads), so
prompts stream from disk and RAM stays flat regardless of dataset size.
The plain JSONL shards under `./sft_data/` are read only to recover the
original `answer` strings used by the reward function (one line per
prompt, looked up by index alignment with the packed data). A
`--prompts-file` override lets you point at a custom JSONL of
`{prompt, answer}` instead.

For each training step:
    1. Sample a batch of prompts.
    2. Roll out G completions per prompt with the current policy.
    3. Score each completion with the reward function
       (rule-based correctness + format bonus — option A).
    4. Compute per-prompt advantages by group-normalising rewards
       within the G rollouts.
    5. PPO-style clipped policy gradient on token-level log-probs, with an
       optional KL penalty against a reference policy.

Reference policy (--ref_policy):
    single  - reuse the trainable model with `no_grad` to get reference
              log-probs. ~10% VRAM overhead, single model in memory.
              Recommended for single-GPU runs.
    two     - keep a frozen second copy of the model in memory (~2x VRAM)
              and KL against it. Original DeepSeek-R1 recipe; useful for
              long runs where the policy drifts far from SFT.

Usage:
    # 1. Smoke test (no checkpoint needed)
    python train_grpo.py

    # 2. Real run (single-model reference, recommended)
    python train_grpo.py \\
        --checkpoint ./sft_checkpoints/latest.pt \\
        --tokenizer ./tokenizer \\
        --cache_dir ./sft_packed \\
        --data_dir  ./sft_data \\
        --num_generations 8 \\
        --max_new_tokens 512 \\
        --max_steps 500 \\
        --ref_policy single \\
        --out_dir ./grpo_checkpoints

    # 3. LoRA GRPO (recommended for 1B+ on a single 4090)
    python train_grpo.py --checkpoint ./sft_checkpoints/latest.pt \\
        --tokenizer ./tokenizer --cache_dir ./sft_packed \\
        --data_dir ./sft_data \\
        --lora --lora_rank 64 --lora_alpha 128 \\
        --ref_policy single --out_dir ./grpo_checkpoints

    # 4. Two-model reference variant
    python train_grpo.py --checkpoint ./sft_checkpoints/latest.pt \\
        --tokenizer ./tokenizer --cache_dir ./sft_packed --data_dir ./sft_data \\
        --ref_policy two --num_generations 4 --out_dir ./grpo_two

    # 5. Custom prompts file (skip packed cache)
    python train_grpo.py --checkpoint ... --prompts_file ./eval.jsonl ...

    # 6. Multi-GPU
    torchrun --nproc_per_node=4 train_grpo.py --checkpoint ... --cache_dir ./sft_packed

    # 7. Merge LoRA after training
    python train_grpo.py --merge_lora \\
        --checkpoint ./grpo_checkpoints/latest.pt \\
        --out_dir ./grpo_merged

    # 8. GRPO-specific data pipeline (recommended for reasoning RL):
    #     8a. Download {prompt, answer} records
    python download_grpo_data.py --target-size 2GB --out-dir ./grpo_data
    #     8b. Pack into the same memmap format train_sft.py uses
    python pack_grpo_data.py --data-dir ./grpo_data --tokenizer ./tokenizer \\
        --cache-dir ./grpo_packed
    #     8c. Train — point at the GRPO cache + JSONL pool
    python train_grpo.py --checkpoint ./sft_checkpoints/latest.pt \\
        --tokenizer ./tokenizer --cache_dir ./grpo_packed \\
        --data_dir ./grpo_data --num_generations 8

    # 9. --prompt_override flag: skip the memmap entirely, feed a flat
    #    JSONL of {prompt, answer} pairs (e.g. an eval set).
    python train_grpo.py --checkpoint ./sft_checkpoints/latest.pt \\
        --tokenizer ./tokenizer --prompt_override ./eval_prompts.jsonl \\
        --num_generations 8 --max_steps 100
"""

import argparse
import glob
import json
import math
import os
import random
import re
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from tokenizers import Tokenizer

from model import Qwen3Config, Qwen3ForCausalLM, count_parameters

# Re-use the SFT machinery so this script stays focused on the RL logic.
from train_sft import (
    LoRALinear,
    inject_lora,
    merge_lora,
    lora_state_dict,
    lora_parameter_count,
    load_tokenizer,
    setup_distributed,
    is_master,
    get_lr,
    build_optimizer,
    _raw,
    prune_checkpoints,
    SFTDataset,        # memmap reader (private _ConcatMemmap reused via __getitem__)
)
from pack_sft_data import get_special_token_id


# ---------------------------------------------------------------------------
# Reward function (Option A — rule-based three-tier)
# ---------------------------------------------------------------------------

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

# \boxed{...} capture (single-level braces; matches the SFT-style answers in
# MetaMathQA / NuminaMath-CoT).
ANSWER_RE = re.compile(r"\\boxed\{([^}]+)\}")
# Numeric fallback (GSM8K, ScienceQA, etc.).
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def extract_answer(field: str) -> Tuple[Optional[float], str]:
    """
    Pull a comparable answer out of an answer / completion string.

    Returns (numeric_value, raw_string). numeric_value is None if the field
    doesn't contain a boxed or numeric answer; raw_string is the boxed
    content if present, otherwise the trailing numeric / word token.

    For numeric answers we deliberately take the LAST match in the
    field, not the first: reasoning traces (e.g. "think 2+2=4 /think 4")
    often contain intermediate numbers inside the CoT and we want the
    final answer the model committed to, not an internal calculation.

    For non-numeric answers we look at the substring AFTER the closing
    think tag when present — that's where the model writes its final
    answer in the ChatML template. Otherwise we fall back to the last
    whitespace-separated token.

    Comparison logic in `compute_reward` matches numeric first, then falls
    back to case-insensitive string equality (handles unit suffixes,
    spelled-out answers, etc.).
    """
    if field is None:
        return None, ""

    m = ANSWER_RE.search(field)
    if m:
        return _try_float(m.group(1).strip()), m.group(1).strip()

    # Slice to "post-think" portion for the final-answer extraction.
    # This mirrors the ChatML template: ...</think>\nFINAL_ANSWER
    c = field.rfind(THINK_CLOSE)
    answer_zone = field[c + len(THINK_CLOSE):] if c != -1 else field

    # Last numeric match in the answer zone (final answer in a CoT trace)
    nums = list(NUM_RE.finditer(answer_zone))
    if nums:
        last = nums[-1]
        return _try_float(last.group(0)), last.group(0)
    toks = answer_zone.strip().split()
    return None, toks[-1] if toks else ""


def _try_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _has_balanced_think(text: str) -> bool:
    """True iff the text contains think.../think with open before close."""
    o = text.find(THINK_OPEN)
    c = text.find(THINK_CLOSE)
    return o != -1 and c != -1 and o < c


def compute_reward(
    prompt: str,
    completion: str,
    ground_truth: str,
    max_new_tokens: int,
    correct_weight: float = 1.0,
    format_weight: float = 0.3,
) -> Tuple[float, Dict[str, int]]:
    """
    Three-tier reward (Option A):

        tier 1.0  answer correct + balanced think.../think + did not truncate
        tier 0.5  (answer correct but no thinking) OR (wrong answer but
                  thinking present + produced a final answer)
        tier 0.0  no answer, truncated, or otherwise malformed

    `correct_weight` and `format_weight` let you tune the relative pull of
    correctness vs. format. The returned reward is in [0, 1] when both
    weights are in [0, 1].
    """
    has_think = _has_balanced_think(completion)
    # Truncation heuristic: ~4 chars/token is generous; cap at the full
    # generation length.
    truncated = len(completion) >= max_new_tokens * 4

    gold_num, gold_str = extract_answer(ground_truth)
    pred_num, pred_str = extract_answer(completion)

    if gold_num is not None and pred_num is not None:
        correct = math.isclose(gold_num, pred_num, rel_tol=1e-3, abs_tol=1e-4)
    else:
        correct = bool(gold_str) and gold_str.lower() == pred_str.lower()

    has_answer = bool(pred_str)

    info = {
        "correct": int(correct),
        "has_think": int(has_think),
        "has_answer": int(has_answer),
        "truncated": int(truncated),
    }

    if truncated and not has_answer:
        return 0.0, info
    if correct and has_think:
        return correct_weight, info
    if correct and not has_think:
        return 0.5 * correct_weight, info
    if has_think and has_answer and not truncated:
        return format_weight, info
    return 0.0, info


# ---------------------------------------------------------------------------
# Prompt dataset — backed by packed SFT memmaps
# ---------------------------------------------------------------------------

class GRPOPromptDataset:
    """
    Streams `{prompt_ids, ground_truth_answer}` pairs from the **packed
    memmaps** written by `pack_sft_data.py` (and consumed by
    `train_sft.py`). Re-tokenisation is unnecessary; the tokens are
    already on disk in two mmap'd arrays:

        tokens[i]  : token id at position i
        mask[i]    : 1 = assistant token (in loss), 0 = prompt / EOS sep

    Walking sample boundaries:

        As `pack_sft_data.py` writes the file, each record is followed by
        a single EOS token (mask = 0). Sample boundaries are the
        positions where mask transitions from 1 -> 0 -> EOS. We find
        every (prompt_start, prompt_end, answer_end) triple during a
        single linear sweep over the memmap, store them as
        `(offset, prompt_len, answer_len)` tuples in RAM, and never
        re-touch the disk during training.

    Mode selection:
        default  -> --cache_dir ./sft_packed + --data_dir ./sft_data
                     The packed memmaps drive tokenisation; ./sft_data
                     JSONL shards are read once to recover the original
                     `answer` strings (looked up by index alignment).
        override -> --prompts_file path
                     A plain JSONL of {prompt, answer} is read in full
                     (still small enough to fit in RAM for typical
                     eval sets).

    Either way the per-step sample() call returns random indices into a
    pre-built list — RAM stays flat.
    """

    def __init__(
        self,
        cache_dir: Optional[str],        # --cache_dir (packed memmaps)
        data_dir: Optional[str],         # --data_dir  (raw JSONL, for answer text)
        prompts_file: Optional[str],
        tokenizer: Tokenizer,
        max_prompt_len: int,
        eos_id: int,
    ):
        self.tokenizer     = tokenizer
        self.max_prompt_len = max_prompt_len
        self.eos_id        = eos_id

        if prompts_file:
            self._init_from_jsonl(prompts_file)
        else:
            assert cache_dir, "--cache_dir is required when --prompts_file is not given"
            self._init_from_packed(cache_dir, data_dir)

        if not self._prompts:
            raise RuntimeError(
                f"No usable prompts after filtering. Check that "
                f"{'./sft_packed' if prompts_file is None else prompts_file} "
                f"contains valid records."
            )

    # ----------------------------------------------------------------------
    # Path A: packed SFT memmaps + raw JSONL for answer text
    # ----------------------------------------------------------------------
    def _init_from_packed(self, cache_dir: str, data_dir: Optional[str]):
        """
        Open the same memmap files train_sft.py reads, locate sample
        boundaries, and pre-build (prompt_ids, ground_truth) pairs.
        """
        # Reuse SFTDataset for manifest discovery + mmap concatenation.
        # We pass a tiny seq_len so .__len__==0 in the rank/world_size
        # sense doesn't matter — we use the underlying _ConcatMemmaps
        # directly.
        probe = SFTDataset(
            cache_dir=cache_dir,
            seq_len=2**30,                  # floor: every "window" is the full file
            rank=0, world_size=1,
            split="train",
        )
        self._tokens_memmap = probe.tokens   # lazy _ConcatMemmap (mmap-backed, no copy)
        self._mask_memmap   = probe.mask     # lazy _ConcatMemmap (mmap-backed, no copy)
        self._n_shards      = probe.n_shards
        total = len(self._tokens_memmap)

        # ---- find (prompt_start, prompt_end_assistant_end) by scanning
        #      the mask array. Sample boundary = mask==0 position that is
        #      also an EOS token (the separator pack_sft_data.py writes).
        #
        # We scan each underlying worker-shard memmap directly (still
        # disk-/page-cache-backed, evictable under memory pressure)
        # instead of materializing the whole concatenated dataset into
        # one permanent, non-reclaimable RAM buffer via _contiguous_view.
        # Each worker shard from pack_sft_data.py holds complete records
        # (no record straddles two shard files), so per-shard scanning
        # finds the exact same boundaries as a single global scan would.
        tok_shards  = self._tokens_memmap.arrays
        mask_shards = self._mask_memmap.arrays

        # (shard_idx, local_start, local_end) — kept per-shard rather
        # than flattened into one global offset, so prompt extraction
        # below can index straight into that shard's own memmap.
        boundaries: List[Tuple[int, int, int]] = []
        for shard_idx, (tok_arr, mask_arr) in enumerate(zip(tok_shards, mask_shards)):
            for s, e in self._scan_boundaries(mask_arr, tok_arr):
                boundaries.append((shard_idx, s, e))

        # ---- recover the original `answer` strings from the JSONL pool
        answers_text: List[Optional[str]] = self._load_answer_strings(data_dir, len(boundaries))

        # ---- build per-record slices
        prompts: List[List[int]] = []
        ground_truths: List[str] = []
        prompt_texts: List[str]  = []
        eos_id = self.eos_id

        skipped_long = 0
        skipped_no_gt = 0
        for i, (shard_idx, s, e) in enumerate(boundaries):
            gt = answers_text[i] if i < len(answers_text) else None
            if not gt:
                skipped_no_gt += 1
                continue
            tok_arr  = tok_shards[shard_idx]
            mask_arr = mask_shards[shard_idx]
            # Find the assistant-turn start: the first mask==1 within [s, e).
            # (pack_sft_data.py writes a contiguous <|im_start|>user…
            # <|im_end|>\n<|im_start|>assistant block followed by the
            #  think…/think{answer}<|im_end|> assistant turn.)
            p_start = s
            while p_start < e and mask_arr[p_start] == 0:
                p_start += 1
            prompt_ids = tok_arr[s:p_start].tolist()
            if len(prompt_ids) >= self.max_prompt_len:
                skipped_long += 1
                continue
            prompts.append(prompt_ids)
            # The answer text from JSONL is the ground truth the reward
            # function compares against. The actual assistant tokens
            # already include this answer (we have them on disk) but we
            # only need the string for reward scoring.
            ground_truths.append(gt)
            # Decode the prompt back to text for logging / debugging.
            try:
                prompt_texts.append(
                    self.tokenizer.decode(prompt_ids, skip_special_tokens=False)
                )
            except Exception:
                prompt_texts.append("")

        if skipped_long:
            print(f"[PackedDataset] skipped {skipped_long} record(s) "
                  f"with prompt > --max_prompt_len={self.max_prompt_len}")
        if skipped_no_gt:
            print(f"[PackedDataset] {skipped_no_gt} record(s) had no "
                  f"answer text in JSONL pool — skipped")

        self._prompts     = prompts
        self._answers     = ground_truths
        self._prompt_text = prompt_texts

    # ----------------------------------------------------------------------
    @staticmethod
    def _scan_boundaries(mask_arr: np.ndarray, tok_arr: np.ndarray) -> List[Tuple[int, int]]:
        """
        Walk the mask array and return sample boundaries.

        A boundary ends at a position where mask is 0 *and* the
        corresponding token is the EOS special id (the separator
        `pack_sft_data.py` writes between records).

        Returns: list of (start, end_exclusive) tuples, where each tuple
        describes one full SFT record in [start, end).

        Implementation: rather than a pure-Python for-loop over every
        token (O(n_tokens), i.e. billions of iterations for a large
        corpus — the thing that was pinning a CPU core and holding the
        whole materialized array resident in RAM for minutes/hours),
        we first vectorize the search for separator *candidates* with
        numpy (fast, C-level), then only loop over those candidates
        (O(n_records), typically a few million at most).
        """
        n = len(mask_arr)
        if n == 0:
            return []

        # Candidate separator positions: mask==0 and token==EOS(0).
        candidates = np.flatnonzero((mask_arr == 0) & (tok_arr == 0))

        # Positions where mask==1, needed to confirm "in_record" was
        # true since the last boundary (mirrors the original loop's
        # in_record guard, so padding runs of mask=0/tok=0 without any
        # intervening mask=1 don't get treated as spurious boundaries).
        mask1_positions = np.flatnonzero(mask_arr == 1)

        boundaries: List[Tuple[int, int]] = []
        rec_start = 0
        m1_ptr = 0  # monotonically advancing index into mask1_positions
        n_m1 = len(mask1_positions)

        for c in candidates:
            if c < rec_start:
                continue
            # Advance m1_ptr to the first mask==1 position >= rec_start.
            while m1_ptr < n_m1 and mask1_positions[m1_ptr] < rec_start:
                m1_ptr += 1
            in_record = m1_ptr < n_m1 and mask1_positions[m1_ptr] < c
            if in_record:
                boundaries.append((rec_start, c + 1))
                rec_start = c + 1

        # Trailing partial record (no final EOS): include if non-empty
        # and it actually contains assistant tokens (in_record True).
        if rec_start < n:
            while m1_ptr < n_m1 and mask1_positions[m1_ptr] < rec_start:
                m1_ptr += 1
            if m1_ptr < n_m1 and mask1_positions[m1_ptr] < n:
                boundaries.append((rec_start, n))

        return boundaries

    # ----------------------------------------------------------------------
    @staticmethod
    def _load_answer_strings(data_dir: Optional[str], n_expected: int) -> List[Optional[str]]:
        """
        Stream JSONL shards under data_dir and collect each record's
        `answer` field, in the same order `pack_sft_data.py` saw them.

        We rely on stable ordering: pack_sft_data.py processes shards in
        alphabetical order, line-by-line, and skips some records
        (json error, too-long). For the GRPO loop's purposes this is
        precise enough — any misalignment just means some prompts get
        a dummy ground truth and contribute 0 reward; it does NOT
        corrupt the policy gradient on the rest of the batch.

        If no data_dir is given (or no JSONL is found), returns a list of
        empty strings so those prompts are skipped (no reward signal).
        """
        if not data_dir:
            return [""] * n_expected
        paths = sorted(glob.glob(os.path.join(data_dir, "*", "*.jsonl")))
        if not paths:
            paths = sorted(glob.glob(os.path.join(data_dir, "*.jsonl")))
        if not paths:
            print(f"[PackedDataset] no JSONL found under {data_dir}; "
                  f"every prompt will get an empty ground truth and "
                  f"contribute reward=0.")
            return [""] * n_expected

        answers: List[str] = []
        for p in paths:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    answers.append((rec.get("answer") or "").strip())
        # Pad/truncate to match boundary count.
        if len(answers) < n_expected:
            answers.extend([""] * (n_expected - len(answers)))
        else:
            answers = answers[:n_expected]
        return answers

    # ----------------------------------------------------------------------
    # Path B: plain JSONL override
    # ----------------------------------------------------------------------
    def _init_from_jsonl(self, path: str):
        """Fallback: read every record's prompt + answer from a single file."""
        prompts: List[List[int]] = []
        ground_truths: List[str] = []
        prompt_texts: List[str]  = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                prompt = (rec.get("prompt") or "").strip()
                answer = (rec.get("answer") or "").strip()
                if not prompt or not answer:
                    continue
                ids = self._format_prompt(prompt)
                if ids is None:
                    continue
                prompts.append(ids)
                ground_truths.append(answer)
                prompt_texts.append(prompt)

        self._prompts     = prompts
        self._answers     = ground_truths
        self._prompt_text = prompt_texts

    # ----------------------------------------------------------------------
    def _format_prompt(self, prompt: str) -> Optional[List[int]]:
        """ChatML user-turn prefix, matches pack_sft_data.py formatting."""
        text = f"user\n{prompt}\nassistant\n"
        ids = self.tokenizer.encode(text, add_special_tokens=False).ids
        if len(ids) >= self.max_prompt_len:
            return None
        return ids

    # ----------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._prompts)

    def sample_batch(
        self, batch_size: int, rng: random.Random,
    ) -> Tuple[List[List[int]], List[str], List[str]]:
        """Return (prompt_ids, ground_truth, prompt_text) for `batch_size` prompts."""
        idxs = [rng.randrange(len(self._prompts)) for _ in range(batch_size)]
        return (
            [self._prompts[i]     for i in idxs],
            [self._answers[i]     for i in idxs],
            [self._prompt_text[i] for i in idxs],
        )


def _contiguous_view(concat_memmap) -> np.ndarray:
    """
    Materialise a `_ConcatMemmap` view as a single in-RAM ndarray. This
    is called exactly once at dataset construction. After this, the
    per-step sample() call only reads slices of `self._tokens_memmap`
    / `self._mask_memmap` lazily via __getitem__ (which still mmaps from
    disk). Memory cost = one full pass over the dataset, then flat.

    If you'd rather not pay even that, call np.memmap on the underlying
    files directly — for simplicity we trade one cheap pass for cleaner
    code.
    """
    return np.asarray(concat_memmap[:])


# ---------------------------------------------------------------------------
# Batched rollout: G completions per prompt, single forward pass
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_rollouts(
    model: Qwen3ForCausalLM,
    prompt_ids_list: List[List[int]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    eos_id: int,
    pad_id: int,
    rng: random.Random,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate G completions per prompt, returning:

        full_ids   : (B*G, P+T)  token ids (left-padded prompts + generated)
        gen_mask   : (B*G, T)    1 for generated positions, 0 for padding
        sampled_lp : (B*G, T)    per-token log-prob of the sampled token

    Implementation notes:
        - Prompts are left-padded to a common length so the whole batch
          can be processed as one tensor with KV-cache.
        - Per-replica seeds give diverse samples without affecting global
          PyTorch RNG state.
        - After EOS, remaining positions are filled with `pad_id` (the
          tokenizer's <|pad|>) and the gen_mask is zeroed so they don't
          contribute to log-probs or loss.
        - Sampling uses temperature + top-p (nucleus) truncation; the
          model's own generate() is single-sample-only and doesn't expose
          log-probs, so we hand-roll the loop.
    """
    device = next(model.parameters()).device

    B = len(prompt_ids_list)
    P = max(len(p) for p in prompt_ids_list)
    prompt_ids = torch.full((B, P), pad_id, dtype=torch.long, device=device)
    for i, p in enumerate(prompt_ids_list):
        prompt_ids[i, P - len(p):] = torch.tensor(p, dtype=torch.long, device=device)

    full_ids   = torch.full((B, P + max_new_tokens), pad_id, dtype=torch.long, device=device)
    full_ids[:, :P] = prompt_ids
    gen_mask   = torch.zeros((B, max_new_tokens), dtype=torch.float, device=device)
    sampled_lp = torch.zeros((B, max_new_tokens), dtype=torch.float, device=device)

    past_kv = None
    cur_ids = prompt_ids

    g = torch.Generator(device=device)
    g.manual_seed(rng.randrange(2**31))

    already_done = torch.zeros(B, dtype=torch.bool, device=device)

    for t in range(max_new_tokens):
        if past_kv is None:
            inp = cur_ids
        else:
            inp = cur_ids[:, -1:]

        out = model(inp, past_key_values=past_kv, use_cache=True)
        logits = out["logits"][:, -1, :].float()
        past_kv = out["past_key_values"]

        logits = logits / max(temperature, 1e-5)

        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            sp = sorted_logits.softmax(dim=-1)
            cumsp = sp.cumsum(dim=-1)
            keep = cumsp <= top_p
            keep[..., 0] = True
            mask = torch.full_like(logits, False, dtype=torch.bool)
            mask.scatter_(-1, sorted_idx, keep)
            logits = torch.where(mask, logits, torch.full_like(logits, float("-inf")))

        probs = logits.softmax(dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1, generator=g).squeeze(-1)

        tok_lp = logits.log_softmax(dim=-1).gather(-1, next_tok.unsqueeze(-1)).squeeze(-1)

        active = (~already_done).float()
        full_ids[:, P + t] = next_tok
        gen_mask[:, t]     = active
        sampled_lp[:, t]   = tok_lp * active

        finished_now = (next_tok == eos_id) & (~already_done)
        already_done = already_done | finished_now

        if already_done.all():
            break

    actual_T = max_new_tokens
    if gen_mask.any():
        active_lens = gen_mask.sum(dim=1).long()
        actual_T = int(active_lens.max().item())
        actual_T = max(1, min(actual_T, max_new_tokens))

    return full_ids[:, :P + actual_T], gen_mask[:, :actual_T], sampled_lp[:, :actual_T]


# ---------------------------------------------------------------------------
# Per-token log-prob computation (no_grad) — used for the reference model
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_logprobs(
    model: Qwen3ForCausalLM,
    full_ids: torch.Tensor,
    gen_mask: torch.Tensor,
) -> torch.Tensor:
    """Forward `full_ids`, return per-token log-prob of generated positions."""
    out = model(full_ids, use_cache=False)
    logits = out["logits"][:, :-1, :].float()
    targets = full_ids[:, 1:]
    logp = logits.log_softmax(dim=-1)
    tok_lp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    T = gen_mask.shape[1]
    return tok_lp[:, -T:] * gen_mask


# ---------------------------------------------------------------------------
# GRPO loss
# ---------------------------------------------------------------------------

def grpo_loss(
    policy_logp: torch.Tensor,
    ref_logp: torch.Tensor,
    rewards: torch.Tensor,
    gen_mask: torch.Tensor,
    group_size: int,
    kl_coef: float,
    clip_ratio: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    GRPO objective (Shao et al. 2024) with KL penalty against the
    reference policy and a PPO-style clipped ratio.
    """
    N, T = policy_logp.shape
    assert N % group_size == 0
    n_prompts = N // group_size

    r_g = rewards.view(n_prompts, group_size)
    mean = r_g.mean(dim=1, keepdim=True)
    std  = r_g.std(dim=1, keepdim=True).clamp(min=1e-4)
    advantages = ((r_g - mean) / std).view(-1)

    log_ratio = (policy_logp - ref_logp) * gen_mask.float()
    ratio = log_ratio.exp()

    adv_b = advantages.unsqueeze(1)
    surr1 = adv_b * ratio
    surr2 = adv_b * ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio)
    pg_per_tok = -torch.min(surr1, surr2) * gen_mask.float()
    pg_loss    = pg_per_tok.sum() / gen_mask.sum().clamp(min=1.0)

    # KL penalty — k3 estimator (unbiased); matches TRL/HF GRPO.
    kl_per_tok = (ratio - log_ratio - 1.0) * gen_mask.float()
    kl_loss    = kl_per_tok.sum() / gen_mask.sum().clamp(min=1.0)

    loss = pg_loss + kl_coef * kl_loss

    metrics = {
        "pg": float(pg_loss.detach().item()),
        "kl": float(kl_loss.detach().item()),
        "reward_mean": float(rewards.mean().item()),
        "reward_std":  float(rewards.std().item()),
        "advantage_abs_mean": float(advantages.abs().mean().item()),
        "ratio_mean": float(ratio.detach().mean().item()),
    }
    return loss, metrics


# ---------------------------------------------------------------------------
# Reference policy setup
# ---------------------------------------------------------------------------

def build_reference(
    ref_policy: str,
    config: Qwen3Config,
    sft_ckpt_path: Optional[str],
    device: torch.device,
) -> Optional[Qwen3ForCausalLM]:
    """Frozen reference model, or None for --ref_policy single."""
    if ref_policy == "single":
        return None
    if ref_policy == "two":
        if sft_ckpt_path is None or not os.path.exists(sft_ckpt_path):
            raise FileNotFoundError(
                "--ref_policy two requires --checkpoint pointing at the SFT "
                "checkpoint to clone."
            )
        ref_model = Qwen3ForCausalLM(config).to(device)
        ckpt = torch.load(sft_ckpt_path, map_location=device, weights_only=False)
        ref_model.load_state_dict(ckpt["model_state"])
        if hasattr(ref_model, "tie_weights"):
            ref_model.tie_weights()
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)
        n_ref = count_parameters(ref_model)
        print(f"[RefPolicy] two-model: cloned frozen reference "
              f"({n_ref/1e9:.3f}B params)")
        return ref_model

    raise ValueError(
        f"Unknown --ref_policy {ref_policy!r}; expected 'single' or 'two'"
    )


# ---------------------------------------------------------------------------
# Checkpoint helpers (mirror train_sft.py)
# ---------------------------------------------------------------------------

def save_checkpoint(
    out_dir: str,
    step: int,
    model,
    optimizer,
    config: Qwen3Config,
    args_dict: dict,
    is_lora: bool,
):
    raw = _raw(model)
    ckpt = {
        "step":            step,
        "model_state":     lora_state_dict(raw) if is_lora else raw.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config":          vars(config),
        "args":            args_dict,
        "is_lora":         is_lora,
    }
    path = os.path.join(out_dir, f"grpo_step{step:07d}.pt")
    torch.save(ckpt, path)
    latest = os.path.join(out_dir, "latest.pt")
    if os.path.islink(latest):
        os.remove(latest)
    os.symlink(os.path.abspath(path), latest)
    print(f"[Checkpoint] saved {path}")
    return path


def load_checkpoint(
    path: str, model, optimizer, device: torch.device, is_lora: bool,
):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    raw  = _raw(model)
    state = ckpt["model_state"]
    if is_lora:
        missing, unexpected = raw.load_state_dict(state, strict=False)
        lora_keys = [k for k in state if "lora_A" in k or "lora_B" in k]
        print(f"[Checkpoint] loaded {len(lora_keys)} LoRA tensors from {path}")
        if optimizer and "optimizer_state" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state"])
            except Exception as e:
                print(f"[Checkpoint] optimizer state load skipped: {e}")
    else:
        raw.load_state_dict(state)
        if hasattr(raw, "tie_weights"):
            raw.tie_weights()
        if optimizer and "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
    step = ckpt.get("step", 0)
    print(f"[Checkpoint] resumed from step {step}")
    return step


# ---------------------------------------------------------------------------
# Merge-only mode (mirror train_sft.py)
# ---------------------------------------------------------------------------

def merge_and_save(args):
    device = torch.device("cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = Qwen3Config(**ckpt["config"])
    model = Qwen3ForCausalLM(config)

    pretrain_path = ckpt["args"].get("checkpoint")
    if pretrain_path and os.path.exists(pretrain_path):
        base = torch.load(pretrain_path, map_location=device, weights_only=False)
        model.load_state_dict(base["model_state"])
        model.tie_weights()
        print(f"[Merge] loaded base weights from {pretrain_path}")
    else:
        print("[Merge] WARNING: base checkpoint path not found; "
              "LoRA will be merged onto random weights.")

    inject_lora(model, rank=ckpt["args"].get("lora_rank", 64),
                alpha=ckpt["args"].get("lora_alpha", 128.0))
    model.load_state_dict(ckpt["model_state"], strict=False)
    model = merge_lora(model)
    model.tie_weights()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "merged_model.pt")
    torch.save({"model_state": model.state_dict(), "config": vars(config)}, out_path)
    print(f"[Merge] saved merged model to {out_path}")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args):
    rank, local_rank, world_size, device = setup_distributed()
    master = is_master(rank)

    torch.manual_seed(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32        = True
    rng = random.Random(args.seed + rank)

    # ---------------------------------------------------------------- model
    if not args.checkpoint:
        raise FileNotFoundError(
            "--checkpoint is required. Point it at the SFT checkpoint "
            "produced by train_sft.py."
        )
    ckpt_data = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config    = Qwen3Config(**ckpt_data["config"])

    model = Qwen3ForCausalLM(config).to(device)
    model.load_state_dict(ckpt_data["model_state"])
    model.tie_weights()

    if master:
        n_total = count_parameters(model)
        print(f"Loaded SFT checkpoint: {n_total:,} params ({n_total/1e9:.3f}B)")

    # ----------------------------------------------------------------- LoRA
    is_lora = args.lora
    if is_lora:
        n_replaced = inject_lora(model, rank=args.lora_rank, alpha=args.lora_alpha)
        n_trainable = lora_parameter_count(model)
        if master:
            print(f"[LoRA] injected {n_replaced} adapters | "
                  f"trainable={n_trainable:,} / total={count_parameters(model):,}")
    else:
        if master:
            print("[LoRA] disabled — full fine-tune")

    # --------------------------------------------------------------- compile
    _use_cudagraphs = False
    if args.compile:
        if master:
            print(f"[compile] torch.compile(mode='{args.compile_mode}')…")
        model = torch.compile(model, mode=args.compile_mode)
        _use_cudagraphs = (args.compile_mode == "reduce-overhead")

    # ----------------------------------------------------------------- ref
    ref_model = build_reference(args.ref_policy, config, args.checkpoint, device)
    ref_for_logprob = ref_model if ref_model is not None else model

    # ------------------------------------------------------------------- DDP
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    # --------------------------------------------------------------- tokenizer
    tokenizer = load_tokenizer(args.tokenizer)
    try:
        eos_id = get_special_token_id(tokenizer, "<|endoftext|>")
        pad_id = tokenizer.token_to_id("<|pad|>") or 0
    except Exception:
        eos_id = tokenizer.get_vocab_size() - 1
        pad_id = 0
    if master:
        print(f"[Tokenizer] eos_id={eos_id}, pad_id={pad_id}, "
              f"vocab={tokenizer.get_vocab_size()}")

    # --------------------------------------------------------------- dataset
    train_ds = GRPOPromptDataset(
        cache_dir=args.cache_dir,
        data_dir=args.data_dir,
        prompts_file=args.prompts_file,
        tokenizer=tokenizer,
        max_prompt_len=args.max_prompt_len,
        eos_id=eos_id,
    )
    if master:
        n_shards = getattr(train_ds, "_n_shards", 0)
        print(f"[Dataset] {len(train_ds):,} prompts "
              f"({n_shards} packed shard(s))")

    # --------------------------------------------------------------- optim
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{"params": trainable_params, "weight_decay": args.weight_decay}],
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
        fused=torch.cuda.is_available(),
    )

    # --------------------------------------------------------------- amp
    use_amp = device.type == "cuda" and args.dtype == "bf16"
    ctx = (torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
           if use_amp else nullcontext())

    # --------------------------------------------------------------- resume
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model, optimizer, device, is_lora)

    if master:
        os.makedirs(args.out_dir, exist_ok=True)
        eff_prompts     = args.batch_size * world_size
        eff_completions = eff_prompts * args.num_generations
        print(f"\nEffective batch   : {eff_prompts} prompts "
              f"({eff_completions} completions)")
        print(f"Group size G      : {args.num_generations}")
        print(f"Max steps         : {args.max_steps:,}")
        print(f"Reference policy  : {args.ref_policy}")
        print(f"Data source       : "
              f"{'--prompts_file ' + args.prompts_file if args.prompts_file else '--cache_dir ' + args.cache_dir}")
        print(f"Checkpoint every  : {args.ckpt_interval:,} steps\n")

    # ================================================================= LOOP
    model.train()
    optimizer.zero_grad(set_to_none=True)
    t0 = time.perf_counter()
    reward_window: List[float] = []
    correct_window: List[int]   = []
    think_window: List[int]     = []

    for step in range(start_step, args.max_steps):
        lr = get_lr(step, args.warmup_steps, args.max_steps, args.lr, args.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # 1. sample prompts
        prompts, answers, prompt_texts = train_ds.sample_batch(args.batch_size, rng)

        # 2. expand to G replicas
        expanded_p: List[List[int]] = []
        expanded_a: List[str]       = []
        expanded_pt: List[str]      = []
        for _g in range(args.num_generations):
            for p, a, t in zip(prompts, answers, prompt_texts):
                expanded_p.append(p)
                expanded_a.append(a)
                expanded_pt.append(t)

        # 3. rollout
        rollout_model = _raw(model)
        rollout_model.eval()
        full_ids, gen_mask, _sampled_lp = generate_rollouts(
            rollout_model,
            expanded_p,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            eos_id=eos_id,
            pad_id=pad_id,
            rng=rng,
        )
        rollout_model.train()
        if _use_cudagraphs:
            torch.compiler.cudagraph_mark_step_begin()

        # 4. decode + reward
        completions_text: List[str] = []
        P_max = max(len(p) for p in expanded_p)
        for i, prompt_ids in enumerate(expanded_p):
            start = P_max - len(prompt_ids)
            active = int(gen_mask[i].sum().item())
            gen_ids = full_ids[i, start + len(prompt_ids):
                              start + len(prompt_ids) + active].tolist()
            text = tokenizer.decode(gen_ids, skip_special_tokens=False)
            completions_text.append(text)

        rewards_list: List[float] = []
        per_info: List[Dict[str, int]] = []
        for prompt_text, completion, answer in zip(expanded_pt, completions_text, expanded_a):
            r, info = compute_reward(
                prompt_text, completion, answer,
                max_new_tokens=args.max_new_tokens,
                correct_weight=args.reward_correct,
                format_weight=args.reward_format,
            )
            rewards_list.append(r)
            per_info.append(info)
        rewards = torch.tensor(rewards_list, dtype=torch.float, device=device)

        # 5. reference log-probs (no_grad)
        with torch.no_grad():
            ref_logp = compute_logprobs(ref_for_logprob, full_ids, gen_mask)

        # 6. policy log-probs (with grad)
        out = model(full_ids, use_cache=False)
        policy_logits = out["logits"][:, :-1, :].float()
        targets = full_ids[:, 1:]
        policy_logp = policy_logits.log_softmax(dim=-1).gather(
            -1, targets.unsqueeze(-1)).squeeze(-1)
        T = gen_mask.shape[1]
        policy_logp = policy_logp[:, -T:] * gen_mask

        # 7. loss + step
        loss, metrics = grpo_loss(
            policy_logp, ref_logp, rewards, gen_mask,
            group_size=args.num_generations,
            kl_coef=args.kl_coef,
            clip_ratio=args.clip_ratio,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item()
        optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize()

        # 8. log
        if master:
            reward_window.extend(rewards_list)
            correct_window.extend(i["correct"]   for i in per_info)
            think_window.extend(  i["has_think"] for i in per_info)
            window = max(1, len(reward_window))

            if step % args.log_interval == 0:
                t1 = time.perf_counter()
                sps = args.log_interval / max(t1 - t0, 1e-9)
                t0 = t1
                r_mean = sum(reward_window) / window
                c_mean = sum(correct_window) / window
                f_mean = sum(think_window) / window
                reward_window.clear()
                correct_window.clear()
                think_window.clear()
                print(
                    f"step {step:6d} | loss {loss.item():+.4f} | "
                    f"pg {metrics['pg']:+.4f} | kl {metrics['kl']:+.5f} | "
                    f"r̄ {r_mean:.2f} | acc {c_mean:.0%} | fmt {f_mean:.0%} | "
                    f"lr {lr:.2e} | g {grad_norm:.2f} | {sps:.2f} step/s"
                )

        # 9. checkpoint
        if master and step > start_step and step % args.ckpt_interval == 0:
            save_checkpoint(args.out_dir, step, model, optimizer, config,
                            vars(args), is_lora)
            prune_checkpoints(args.out_dir, keep=args.keep_ckpts)

    # final
    if master:
        save_checkpoint(args.out_dir, args.max_steps, model, optimizer,
                        config, vars(args), is_lora)
        print(f"\nGRPO complete. Final loss: {loss.item():.4f}")
        if is_lora:
            print(f"\nTo merge LoRA into base weights:")
            print(f"  python train_grpo.py --merge_lora "
                  f"--checkpoint {args.out_dir}/latest.pt --out_dir ./grpo_merged")

    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Smoke test (no checkpoint required)
# ---------------------------------------------------------------------------

def smoke_test():
    print("\n=== GRPO smoke test ===")
    import shutil
    import tempfile

    tmp       = tempfile.mkdtemp()
    data_dir  = os.path.join(tmp, "sft_data", "math")
    cache_dir = os.path.join(tmp, "sft_packed")
    ckpt_dir  = os.path.join(tmp, "ckpts")
    tok_dir   = os.path.join(tmp, "tokenizer")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(tok_dir, exist_ok=True)

    # ---- minimal tokenizer with the same special tokens train_tokenizer.py uses
    from tokenizers import Tokenizer as _Tok
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers import pre_tokenizers, decoders

    SPECIAL = ["<|endoftext|>", "<|pad|>", "<|im_start|>", "<|im_end|>",
               "<think>", "</think>"]
    tok = _Tok(BPE(unk_token=None, byte_fallback=True))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder       = decoders.ByteLevel()
    trainer = BpeTrainer(vocab_size=512, special_tokens=SPECIAL,
                         initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                         show_progress=False)
    corpus = [
        "Solve 2+2. Think step by step. The answer is 4.",
        "What is 10-3? Reasoning: 10-3=7. Answer: 7.",
        "<|im_start|>user\nSolve 3+4<|im_end|>\n"
        "<|im_start|>assistant\nthink\n3+4=7\n/think\n7<|im_end|>\n",
    ] * 30
    tok.train_from_iterator(corpus, trainer=trainer)
    tok.save(os.path.join(tok_dir, "tokenizer.json"))

    # ---- fake SFT records (also fed to pack_sft_data.py so it builds the memmap)
    records = [
        {"prompt": "Solve: 2+2",    "thinking": "2 plus 2 equals 4",
         "answer": "4"},
        {"prompt": "What is 10-3?", "thinking": "10 minus 3 is 7",
         "answer": "7"},
    ] * 25
    with open(os.path.join(data_dir, "math.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    # ---- run pack_sft_data.pack_worker_shard to produce the memmap cache
    import pack_sft_data
    tokenizer_for_pack = pack_sft_data.load_tokenizer(tok_dir)
    pack_sft_data.pack_worker_shard(
        data_dir=os.path.join(tmp, "sft_data"),
        tokenizer=tokenizer_for_pack,
        cache_dir=cache_dir,
        max_len_per_example=128,
        val_fraction=0.0,
        worker=0,
        num_workers=1,
        vocab_size=tokenizer_for_pack.get_vocab_size(),
    )

    # ---- tiny model + fake SFT checkpoint
    config = Qwen3Config(
        vocab_size=512, hidden_size=128, intermediate_size=256,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=32, max_position_embeddings=256,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = Qwen3ForCausalLM(config).to(device)
    sft_ckpt = os.path.join(ckpt_dir, "sft.pt")
    torch.save({"model_state": model.state_dict(), "config": vars(config)}, sft_ckpt)

    inject_lora(model, rank=4, alpha=8.0)
    print(f"[smoke] LoRA trainable: {lora_parameter_count(model):,}")

    # ---- drive BOTH paths (packed memmap + plain JSONL override)
    for source_label, kwargs in [
        ("packed", dict(cache_dir=cache_dir, data_dir=os.path.join(tmp, "sft_data"),
                        prompts_file=None)),
        ("jsonl",  dict(cache_dir=None, data_dir=None,
                        prompts_file=os.path.join(data_dir, "math.jsonl"))),
    ]:
        print(f"\n--- data source: {source_label} ---")
        tokenizer = load_tokenizer(tok_dir)
        eos_id = tokenizer.token_to_id("<|endoftext|>")
        pad_id = tokenizer.token_to_id("<|pad|>") or 0
        ds = GRPOPromptDataset(
            tokenizer=tokenizer, max_prompt_len=64, eos_id=eos_id, **kwargs,
        )
        print(f"[smoke] dataset size: {len(ds)}")

        # fresh model per source so the test is repeatable
        m = Qwen3ForCausalLM(config).to(device)
        inject_lora(m, rank=4, alpha=8.0)

        ref_model = build_reference("single", config, sft_ckpt, device)
        ref_for_logprob = ref_model if ref_model is not None else m

        rng = random.Random(0)
        optim = torch.optim.AdamW(
            [p for p in m.parameters() if p.requires_grad],
            lr=1e-5, weight_decay=0.0,
        )
        ctx = (torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
               if device.type == "cuda" else nullcontext())

        m.train()
        for step in range(2):
            prompts, answers, ptext = ds.sample_batch(2, rng)
            expanded_p, expanded_a, expanded_pt = [], [], []
            for _ in range(2):  # G=2 for speed
                expanded_p.extend(prompts)
                expanded_a.extend(answers)
                expanded_pt.extend(ptext)

            with torch.no_grad():
                full_ids, gen_mask, _ = generate_rollouts(
                    m, expanded_p, max_new_tokens=12,
                    temperature=1.0, top_p=0.95, eos_id=eos_id,
                    pad_id=pad_id, rng=rng,
                )

            P_max = max(len(p) for p in expanded_p)
            completions = []
            for i, prompt_ids in enumerate(expanded_p):
                start = P_max - len(prompt_ids)
                active = int(gen_mask[i].sum().item())
                gen_ids = full_ids[i,
                                   start + len(prompt_ids):
                                   start + len(prompt_ids) + active].tolist()
                completions.append(tokenizer.decode(gen_ids, skip_special_tokens=False))

            rewards = torch.tensor([
                compute_reward(pt, c, a, max_new_tokens=12)[0]
                for pt, c, a in zip(expanded_pt, completions, expanded_a)
            ], device=device)

            with torch.no_grad():
                ref_logp = compute_logprobs(ref_for_logprob, full_ids, gen_mask)

            with ctx:
                out = m(full_ids, use_cache=False)
                pl  = out["logits"][:, :-1, :].float()
                tg  = full_ids[:, 1:]
                pol = pl.log_softmax(-1).gather(-1, tg.unsqueeze(-1)).squeeze(-1)
                pol = pol[:, -gen_mask.shape[1]:] * gen_mask

            loss, metrics = grpo_loss(
                pol, ref_logp, rewards, gen_mask,
                group_size=2, kl_coef=0.01, clip_ratio=0.2,
            )
            optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            optim.step()
            print(f"  step {step} | loss {loss.item():+.4f} | "
                  f"pg {metrics['pg']:+.4f} | kl {metrics['kl']:+.5f} | "
                  f"r̄ {metrics['reward_mean']:.2f}")

        if ref_model is not None:
            del ref_model

    # ---- save + reload checkpoint, then merge
    out_dir = os.path.join(ckpt_dir, "grpo")
    os.makedirs(out_dir, exist_ok=True)
    save_checkpoint(out_dir, 3, model, optim, config,
                    vars(argparse.Namespace(
                        checkpoint=sft_ckpt, lora=True,
                        lora_rank=4, lora_alpha=8.0,
                    )), is_lora=True)

    model2 = Qwen3ForCausalLM(config).to(device)
    inject_lora(model2, rank=4, alpha=8.0)
    load_checkpoint(os.path.join(out_dir, "grpo_step0000003.pt"),
                    model2, None, device, is_lora=True)

    merged = merge_lora(model)
    assert not any(isinstance(mm, LoRALinear) for mm in merged.modules()), \
        "merge_lora() left LoRALinear modules in place"
    print("[smoke] LoRA merge OK")

    shutil.rmtree(tmp)
    print("\n=== GRPO smoke test passed ===\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="GRPO RL fine-tuning for reasoning.")

    # Mode
    p.add_argument("--merge_lora", action="store_true",
                   help="Merge LoRA weights into base model and save; skip training")

    # Paths
    p.add_argument("--checkpoint", default=None,
                   help="SFT checkpoint from train_sft.py (required)")
    p.add_argument("--tokenizer",  default="./tokenizer",
                   help="Tokenizer directory from train_tokenizer.py")
    p.add_argument("--cache_dir",  default="./sft_packed",
                   help="Packed memmap directory. Reads either "
                        "./sft_packed (from pack_sft_data.py) or "
                        "./grpo_packed (from pack_grpo_data.py) — both write "
                        "the same on-disk format. Used together with --data_dir.")
    p.add_argument("--data_dir",   default="./sft_data",
                   help="Raw JSONL directory (default ./sft_data for the SFT "
                        "pool; pass --data_dir ./grpo_data when using a "
                        "GRPO cache built by pack_grpo_data.py). The prompts "
                        "are streamed from this directory in alphabetical "
                        "shard order to recover ground-truth answers.")
    p.add_argument("--prompts_file", default=None,
                   help="Override: single JSONL with {prompt, answer} pairs "
                        "(legacy alias for --prompt_override).")
    p.add_argument("--prompt_override", default=None,
                   help="Override: single JSONL with {prompt, answer} pairs. "
                        "Skips both the packed memmap cache and the "
                        "--data_dir pool. Useful for small eval sets, "
                        "fast iteration, or running GRPO on a hand-crafted "
                        "prompt list. Same semantics as --prompts_file.")
    p.add_argument("--out_dir",    default="./grpo_checkpoints")
    p.add_argument("--resume",     default=None,
                   help="GRPO checkpoint to resume from")

    # LoRA
    p.add_argument("--lora",       action="store_true",
                   help="Enable LoRA (recommended for 1B+ on a single GPU)")
    p.add_argument("--lora_rank",  type=int,   default=64)
    p.add_argument("--lora_alpha", type=float, default=128.0)

    # Reference policy
    p.add_argument("--ref_policy", default="single", choices=["single", "two"],
                   help="Reference policy design: "
                        "'single' reuses the trainable model under no_grad; "
                        "'two' keeps a frozen second copy in memory.")

    # Rollouts
    p.add_argument("--num_generations", type=int,   default=8,
                   help="G — completions per prompt")
    p.add_argument("--max_new_tokens",  type=int,   default=512)
    p.add_argument("--temperature",     type=float, default=1.0)
    p.add_argument("--top_p",           type=float, default=0.95)
    p.add_argument("--max_prompt_len",  type=int,   default=512)

    # Reward weights
    p.add_argument("--reward_correct",  type=float, default=1.0,
                   help="Reward weight for a fully correct answer")
    p.add_argument("--reward_format",   type=float, default=0.3,
                   help="Reward weight for a wrong-but-well-formed answer")

    # GRPO loss
    p.add_argument("--kl_coef",     type=float, default=0.02)
    p.add_argument("--clip_ratio",  type=float, default=0.2)

    # Optim
    p.add_argument("--batch_size",       type=int,   default=4,
                   help="Number of PROMPTS per step (rollouts = batch_size * G)")
    p.add_argument("--max_steps",        type=int,   default=500)
    p.add_argument("--warmup_steps",     type=int,   default=20)
    p.add_argument("--lr",               type=float, default=1e-6,
                   help="Peak LR (typically 5e-7 to 5e-6 for GRPO)")
    p.add_argument("--min_lr",           type=float, default=1e-7)
    p.add_argument("--weight_decay",     type=float, default=0.0)
    p.add_argument("--grad_clip",        type=float, default=1.0)
    p.add_argument("--dtype",   default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--compile", action="store_true")
    p.add_argument("--compile_mode", default="default",
                   choices=["default", "reduce-overhead", "max-autotune"])
    p.add_argument("--seed",    type=int, default=42)

    # Logging / checkpointing
    p.add_argument("--log_interval",  type=int, default=1)
    p.add_argument("--ckpt_interval", type=int, default=50)
    p.add_argument("--keep_ckpts",    type=int, default=3)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # --prompt_override is the documented GRPO-flavored name; --prompts_file
    # is the legacy alias. If the user passed --prompt_override without
    # --prompts_file, forward it to the same code path so existing
    # GRPOPromptDataset._init_from_jsonl handles it unchanged.
    if args.prompt_override is not None and args.prompts_file is None:
        args.prompts_file = args.prompt_override

    if args.merge_lora:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for --merge_lora")
        merge_and_save(args)
    elif args.checkpoint is None:
        smoke_test()
    else:
        train(args)