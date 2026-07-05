#!/usr/bin/env python3
"""
calculate_settings.py

Chinchilla-aware calculator for Qwen3-Non-MOE pretraining, SFT, and GRPO.

Given ONE of:
  --data-size <amount>     (e.g. 5GB, 500MB, 1.2TB)
  --target-size <amount>   (e.g. 0.6B, 1.7B, 8B)
  --tokens  <int>          (raw token count)

...and optional knobs (seq-len, micro-batch, world-size, gpu), it prints
a full set of recommended settings for pretraining, SFT, and GRPO, plus
the implied command line invocation for this repo's train scripts.

Examples:
  # Pick the optimal model for 50 GB of data
  python calculate_settings.py --data-size 50GB

  # Pick the optimal dataset size for a 1.7B model
  python calculate_settings.py --target-size 1.7B --seq-len 4096 \\
      --micro-batch 2 --world-size 4

  # Raw token count
  python calculate_settings.py --tokens 34000000000

Outputs go to stdout. Use --json to dump a machine-readable config.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Size parsing
# ---------------------------------------------------------------------------

# Conversion helpers: amount → base unit (bytes for data, params for model).

_BYTES_UNITS = {
    "B":   1,
    "KB":  10**3,   "KIB": 10**3,   "K":   10**3,
    "MB":  10**6,   "MIB": 10**6,   "M":   10**6,
    "GB":  10**9,   "GIB": 10**9,   "G":   10**9,
    "TB":  10**12,  "TIB": 10**12,  "T":   10**12,
    "PB":  10**15,  "PIB": 10**15,  "P":   10**15,
}

# Approximate tokens per byte for natural-language text packed as uint16 ids.
#   1 token ≈ 4 characters of English text (Chinchilla reference).
#   1 character ≈ 1 byte in UTF-8.
# So 1 token ≈ 4 bytes on disk in the *raw* corpus.
# After tokenization with a 32k BPE the chars/token ratio shifts but the
# packed memmap (uint16 = 2 bytes/token) gives a tighter bound. We default
# to 4 bytes/token as the conventional "FineWeb-like" rule of thumb.
DEFAULT_BYTES_PER_TOKEN = 4.0

# Bytes per token in the *raw download*, not the packed memmap. These vary
# by stage because the source format differs:
#   - pretrain: web-crawled text (FineWeb, Wikipedia, TheStack, …).
#     JSONL with {"text": "..."} keys plus pretty-printing ≈ 5 B/tok
#     (text itself is ~4 B/tok, JSON overhead + newlines add ~25 %).
#   - SFT: instruction-tuning JSONL with {prompt, thinking, answer} fields
#     and <think> markup. The thinking/answer blocks often 2-3× the raw
#     prompt, plus JSON keys/quotes, so ≈ 8 B/tok.
#   - GRPO: prompts only (the completions are generated, not downloaded).
#     Short Q&A prompts ≈ 5 B/tok, but most GRPO corpora include CoT
#     solutions as exemplars; we use 6 B/tok as a planning estimate.
DEFAULT_BPT_PRETRAIN = 5.0
DEFAULT_BPT_SFT     = 8.0
DEFAULT_BPT_GRPO    = 6.0


def parse_size_to_bytes(value: str) -> int:
    """Parse '5GB', '500MB', '1.2T' → bytes (SI)."""
    s = value.strip().upper().replace(" ", "")
    # Find longest matching unit suffix
    for unit in sorted(_BYTES_UNITS, key=len, reverse=True):
        if s.endswith(unit):
            num = s[: -len(unit)].strip()
            try:
                return int(float(num) * _BYTES_UNITS[unit])
            except ValueError as e:
                raise argparse.ArgumentTypeError(
                    f"Could not parse size '{value}'"
                ) from e
    # No unit → assume bytes
    try:
        return int(s)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Could not parse size '{value}'") from e


def parse_target_params(value: str) -> int:
    """Parse '0.6B', '1.7B', '600M' → integer parameter count."""
    s = value.strip().upper().replace(" ", "")
    if s.endswith("B"):
        return int(float(s[:-1]) * 1_000_000_000)
    if s.endswith("M"):
        return int(float(s[:-1]) * 1_000_000)
    if s.endswith("K"):
        return int(float(s[:-1]) * 1_000)
    try:
        return int(s)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Could not parse param count '{value}'") from e


def human_params(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def human_bytes(n: int, decimal: bool = False) -> str:
    """Format bytes. `decimal=True` forces two-decimal output regardless
    of size (used when the same number is shown twice in different
    units — e.g. SI GB and binary GiB — and we want the digits to align)."""
    for unit, div in (("TB", 10**12), ("GB", 10**9), ("MB", 10**6), ("KB", 10**3)):
        if n >= div:
            return f"{n / div:.2f}{unit}"
    return f"{n}B"


def human_tokens(n: int) -> str:
    for unit, div in (("T", 10**12), ("B", 10**9), ("M", 10**6), ("K", 10**3)):
        if n >= div:
            return f"{n / div:.2f}{unit}"
    return str(n)


# ---------------------------------------------------------------------------
# Chinchilla math
# ---------------------------------------------------------------------------

CHINCHILLA_RATIO = 20  # tokens per non-embedding parameter


def chinchilla_tokens_for_params(N: int, ratio: float = CHINCHILLA_RATIO) -> int:
    return int(round(N * ratio))


def chinchilla_params_for_tokens(D: int, ratio: float = CHINCHILLA_RATIO) -> int:
    return int(round(D / ratio))


# ---------------------------------------------------------------------------
# Reference non-embedding parameter counts for "stock" Qwen3 sizes
# ---------------------------------------------------------------------------
#
# These are the actual non-embedding parameters (N) for each Qwen3 dense
# config returned by Qwen3Config.from_target_size(...). We hardcode them so
# the calculator can pick a sensible architecture by name even when model.py
# isn't importable (e.g. on a planning host without torch).

# (target label, N non-embedding params, hidden, layers, heads, kv_heads, inter)
QWEN3_PRESETS: list[dict[str, Any]] = [
    {"name": "0.6B",  "N":   595_734_528, "H": 1024, "L": 28, "heads": 16, "kv": 4, "I": 3072},
    {"name": "1.7B",  "N": 1_721_127_424, "H": 2048, "L": 28, "heads": 16, "kv": 4, "I": 6144},
    {"name": "4B",    "N": 4_022_458_368, "H": 2560, "L": 36, "heads": 20, "kv": 4, "I": 7680},
    {"name": "8B",    "N": 8_189_737_472, "H": 4096, "L": 36, "heads": 32, "kv": 8, "I": 12288},
    # Architectures a custom Qwen3 search would commonly hit; we use the
    # preset that best matches the user's N* target.
    {"name": "0.3B",  "N":   296_550_400, "H":  768, "L": 24, "heads": 12, "kv": 3, "I": 2304},
    {"name": "1B",    "N": 1_005_056_000, "H": 1536, "L": 28, "heads": 12, "kv": 3, "I": 4608},
    {"name": "3B",    "N": 3_000_000_000, "H": 2304, "L": 32, "heads": 18, "kv": 4, "I": 6912},
    {"name": "32B",   "N": 32_000_000_000, "H": 5120, "L": 64, "heads": 40, "kv": 8, "I": 15360},
]


def pick_arch_for_params(N_target: int) -> dict[str, Any]:
    """Pick the Qwen3 preset whose N is closest to N_target (in log space)."""
    if N_target <= 0:
        raise ValueError("N_target must be positive")
    best = min(QWEN3_PRESETS, key=lambda p: abs(math.log(p["N"] / N_target)))
    return dict(best)


def _normalize_target_label(s: str) -> str:
    """Normalize a user-typed size label to canonical preset form.
    '1.7B' / '1.7 B' / '1.7b' all become '1.7B'."""
    return s.strip().upper().replace(" ", "")


# Set of preset labels we have a closed-form for. Used to suppress the
# "reference" disclaimer when the user gives a target that exactly
# matches a known preset name (e.g. --target-size 1.7B) even if the
# preset's internal N rounds slightly differently.
_PRESET_NAMES = {p["name"] for p in QWEN3_PRESETS}


def pick_arch_for_budget(tokens: int) -> dict[str, Any]:
    """Inverse of the above: pick preset closest to N = tokens / 20."""
    N_star = chinchilla_params_for_tokens(tokens)
    return pick_arch_for_params(N_star)


# ---------------------------------------------------------------------------
# Hyperparameter formulas
# ---------------------------------------------------------------------------

# Effective batch size target (tokens/step). Smaller models want less.
def eff_batch_for_params(N: int) -> int:
    if N < 5e8:
        return 1_000_000         # 1 M tokens/step for <0.5B
    if N < 2e9:
        return 2_000_000         # 2 M for 0.5B–2B
    if N < 8e9:
        return 4_000_000         # 4 M for 2B–8B
    return 8_000_000             # 8 M for 8B+


# Peak LR (AdamW, bf16, dense transformer) via Chinchilla / µP sqrt scaling.
# Anchored at (B_ref=2e6, lr_ref=5e-4).
_LR_REF = 5.0e-4
_B_REF  = 2_000_000


def lr_peak_for_batch(B_eff: int, stage: str = "pretrain") -> float:
    """Peak LR. Stage applies a multiplier for SFT/GRPO."""
    lr = _LR_REF * math.sqrt(_B_REF / B_eff)
    if stage == "sft":
        lr *= 1.0 / 30.0          # ~30× lower than pretrain
    elif stage == "grpo":
        lr *= 1.0 / 100.0         # ~100× lower than pretrain
    elif stage == "sft_lora":
        lr *= 1.0 / 30.0          # same as full SFT
    return lr


def micro_batch_for_params(N: int, seq_len: int, has_grad_ckpt: bool = True) -> int:
    """Largest power of 2 that fits one 24 GB 4090 at seq_len, bf16."""
    # Empirical: ~ memory per param is ~20 bytes (model + AdamW + grads)
    # but with gradient checkpointing this drops to ~6 bytes.
    # These numbers are hand-tuned for bf16 + AdamW + 4090 24 GB.
    base = {
        5e8:  8,
        1e9:  4,
        2e9:  2,
        5e9:  1,
        1e10: 1,
        3e10: 1,
    }
    if not has_grad_ckpt:
        base = {k: max(1, v // 2) for k, v in base.items()}

    # Adjust for sequence length: memory grows linearly in seq_len past a point.
    if seq_len >= 8192:
        base = {k: max(1, v // 2) for k, v in base.items()}
    elif seq_len >= 16384:
        base = {k: 1 for k in base}

    # Find first threshold ≥ N
    for threshold, mb in sorted(base.items()):
        if N <= threshold:
            return mb
    return 1


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class PretrainPlan:
    model_name: str
    N_non_embedding: int
    D_tokens: int
    seq_len: int
    micro_batch: int
    grad_accum: int
    world_size: int
    B_eff: int
    num_steps: int
    lr_peak: float
    warmup_steps: int
    weight_decay: float
    grad_clip: float
    betas: tuple[float, float] = (0.9, 0.95)
    flops: int = 0
    arch: dict[str, Any] = field(default_factory=dict)

    def command_line(self, model_size_label: str, packed_dir: str, out_dir: str) -> str:
        return (
            f"torchrun --nproc_per_node={self.world_size} train.py "
            f"--model-size {model_size_label} --data-dir {packed_dir} "
            f"--out-dir {out_dir} "
            f"--seq-len {self.seq_len} --batch-size {self.micro_batch} "
            f"--grad-accum-steps {self.grad_accum} --max-steps {self.num_steps} "
            f"--warmup-steps {self.warmup_steps} --lr {self.lr_peak:.2e} "
            f"--weight-decay {self.weight_decay} --grad-clip {self.grad_clip}"
        )


@dataclass
class SFTPlan:
    D_tokens: int            # total SFT token budget (sum across epochs)
    samples: int             # samples per epoch
    epochs: int
    seq_len: int
    micro_batch: int
    grad_accum: int
    world_size: int
    B_eff: int
    num_steps: int
    lr_peak: float
    warmup_ratio: float
    weight_decay: float
    grad_clip: float
    lora: bool
    lora_rank: int
    lora_alpha: int
    use_flash_attn: bool = True

    def command_line(
        self, checkpoint: str, tokenizer: str, cache_dir: str, out_dir: str
    ) -> str:
        parts = [
            "python", "train_sft.py",
            f"--checkpoint {checkpoint}",
            f"--tokenizer {tokenizer}",
            f"--cache-dir {cache_dir}",
            f"--out-dir {out_dir}",
            f"--seq-len {self.seq_len}",
            f"--batch-size {self.micro_batch}",
            f"--grad-accum-steps {self.grad_accum}",
            f"--epochs {self.epochs}",
            f"--lr {self.lr_peak:.2e}",
            f"--grad-clip {self.grad_clip}",
        ]
        if self.lora:
            parts.append("--lora")
            parts.append(f"--lora-rank {self.lora_rank}")
            parts.append(f"--lora-alpha {self.lora_alpha}")
        return " ".join(parts)


@dataclass
class GRPOPlan:
    prompts: int
    group_size: int
    num_steps: int
    seq_len: int
    micro_batch: int
    grad_accum: int
    world_size: int
    B_eff: int
    lr_peak: float
    grad_clip: float
    kl_beta: float
    clip_eps: float
    max_prompt_len: int
    max_response_len: int
    rollout_batch: int
    entropy_bonus: float
    lora: bool
    lora_rank: int
    lora_alpha: int
    # Estimated token count for the prompt dataset (what you actually download).
    # Completions are sampled at runtime, not stored on disk.
    prompt_tokens: int = 0


@dataclass
class FullPlan:
    pretrain: PretrainPlan
    sft: SFTPlan
    grpo: GRPOPlan
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Plan builders
# ---------------------------------------------------------------------------

def build_pretrain_plan(
    N: int, arch: dict[str, Any], D_tokens: int,
    seq_len: int, micro_batch: int, grad_accum: int, world_size: int,
) -> PretrainPlan:
    B_eff = micro_batch * grad_accum * world_size * seq_len
    # Snap to target if user didn't fully specify
    target_B = eff_batch_for_params(N)
    if grad_accum == 0:
        # Auto-compute grad_accum to reach the target batch size.
        grad_accum = max(1, round(target_B / (micro_batch * world_size * seq_len)))
        B_eff = micro_batch * grad_accum * world_size * seq_len
    num_steps = max(1, D_tokens // B_eff)
    warmup = max(100, int(num_steps * 0.01))
    lr = lr_peak_batch(B_eff)
    flops = 6 * N * D_tokens   # Chinchilla/PaLM formula
    return PretrainPlan(
        model_name=arch["name"],
        N_non_embedding=N,
        D_tokens=D_tokens,
        seq_len=seq_len,
        micro_batch=micro_batch,
        grad_accum=grad_accum,
        world_size=world_size,
        B_eff=B_eff,
        num_steps=num_steps,
        lr_peak=lr,
        warmup_steps=warmup,
        weight_decay=0.1,
        grad_clip=1.0,
        flops=flops,
        arch=arch,
    )


def lr_peak_batch(B_eff: int) -> float:
    return _LR_REF * math.sqrt(_B_REF / B_eff)


def build_sft_plan(
    N: int, seq_len: int, micro_batch: int, grad_accum: int, world_size: int,
    D_sft_total: int, epochs: int, use_lora: bool,
) -> SFTPlan:
    """Build an SFT plan given a TOTAL SFT token budget (all epochs)."""
    B_eff = micro_batch * grad_accum * world_size * seq_len
    if B_eff <= 0:
        target_B = 500_000
        grad_accum = max(1, round(target_B / (micro_batch * world_size * seq_len)))
        B_eff = micro_batch * grad_accum * world_size * seq_len
    # Per-epoch tokens and steps
    tokens_per_epoch = max(B_eff, D_sft_total // max(1, epochs))
    steps_per_epoch = max(1, tokens_per_epoch // B_eff)
    samples_per_epoch = max(1, D_sft_total // 600 // max(1, epochs))
    return SFTPlan(
        D_tokens=D_sft_total,
        samples=samples_per_epoch,
        epochs=epochs,
        seq_len=seq_len,
        micro_batch=micro_batch,
        grad_accum=grad_accum,
        world_size=world_size,
        B_eff=B_eff,
        num_steps=steps_per_epoch,
        lr_peak=lr_peak_batch(B_eff) / 30.0,
        warmup_ratio=0.03,
        weight_decay=0.0,
        grad_clip=1.0,
        lora=use_lora,
        lora_rank=64,
        lora_alpha=128,
    )


def build_grpo_plan(
    N: int, sft_tokens: int, group_size: int,
    seq_len: int, world_size: int, use_lora: bool,
) -> GRPOPlan:
    # Number of prompts: aim for ~1 step per prompt
    prompts = 8000 if N >= 1e9 else 4000
    if N >= 5e9:
        prompts = 32_000
    num_steps = prompts
    kl_beta = 0.01
    # GRPO policy LR is roughly 50-100× lower than SFT.
    # SFT LR is ~pretrain_LR / 30. PPO/GRPO LR is ~5e-7 to 1e-6 for
    # 1-8B models. Hardcode the empirical safe default; do NOT derive it
    # from B_eff via the sqrt-scaling law, because the sqrt-scaling law is
    # for SGD/AdamW on next-token loss, not for policy-gradient updates
    # whose gradient magnitude is dominated by reward variance.
    if N < 1e9:
        lr_peak = 1.0e-6
    elif N < 5e9:
        lr_peak = 7.5e-7
    else:
        lr_peak = 5.0e-7
    return GRPOPlan(
        prompts=prompts,
        group_size=group_size,
        num_steps=num_steps,
        seq_len=seq_len,
        micro_batch=1,
        grad_accum=group_size,
        world_size=world_size,
        B_eff=group_size * seq_len,
        lr_peak=lr_peak,
        grad_clip=0.5,
        kl_beta=kl_beta,
        clip_eps=0.1 if N < 2e9 else 0.2,
        max_prompt_len=1024,
        max_response_len=2048 if N < 2e9 else 4096,
        rollout_batch=1,
        entropy_bonus=0.0,
        lora=use_lora,
        lora_rank=64,
        lora_alpha=128,
        # Estimate total prompt tokens. Average prompt is typically
        # ~50 % of max_prompt_len (math/code problems are usually
        # 200-500 tokens; longer few-shot exemplars pull the mean up).
        prompt_tokens=prompts * 512,
    )


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def _bar(label: str, value: str, width: int = 36) -> str:
    pad = max(1, width - len(label))
    return f"  {label:<{pad}}  {value}"


def print_plan(
    plan: FullPlan, args: argparse.Namespace,
    pretrain_bytes: int, sft_bytes: int, grpo_bytes: int, total_bytes: int,
    pretrain_packed: int, sft_packed: int, grpo_packed: int, total_packed: int,
) -> None:
    p, s, g = plan.pretrain, plan.sft, plan.grpo
    arch = p.arch

    print()
    print("=" * 78)
    print(" Qwen3-Non-MOE training plan (Chinchilla-aware)")
    print("=" * 78)

    # ---- Top-level summary ----
    print()
    print("[ INPUT ]")
    if args.data_size:
        print(_bar("Data size",       f"{human_bytes(args.data_size_bytes)} → "
                                     f"{human_tokens(p.D_tokens)} tokens"))
    elif args.tokens:
        print(_bar("Token budget",    f"{human_tokens(args.tokens)}"))
    if args.target_size:
        print(_bar("Target model",    f"{args.target_size} → "
                                     f"{human_params(p.N_non_embedding)} params "
                                     f"(Chinchilla N*)"))
    # Architecture line: if the user gave a target label that matches
    # a known preset (or gave us a data/tokens budget from which we
    # derived a preset), show the preset cleanly. If the user gave a
    # non-preset --target-size, mark the preset as "reference only" so
    # it's clear the math was done with the user's exact N.
    if args.target_size and _normalize_target_label(args.target_size) not in _PRESET_NAMES:
        arch_label = (
            f"reference: Qwen3 {p.model_name} "
            f"(H={arch['H']}, L={arch['L']}, heads={arch['heads']}, "
            f"kv={arch['kv']}, I={arch['I']}) — math uses user's N exactly"
        )
    else:
        arch_label = (f"Qwen3 {p.model_name} (H={arch['H']}, "
                      f"L={arch['L']}, heads={arch['heads']}, "
                      f"kv={arch['kv']}, I={arch['I']})")
    print(_bar("Architecture",     arch_label))
    print(_bar("Tokens / param",   f"{p.D_tokens / p.N_non_embedding:.2f}× "
                                  f"(target {CHINCHILLA_RATIO}×)"))
    print(_bar("Total compute",    f"{p.flops:.2e} FLOPs"))

    # ---- Download sizes (what to actually pull from HuggingFace) ----
    #
    # Three different bytes/token ratios because the on-disk formats differ:
    #   - pretrain: web text, ~5 B/tok (raw text 4 B/tok + JSON wrapper)
    #   - SFT:      instruction JSONL with thinking/answer fields, ~8 B/tok
    #   - GRPO:     prompts only (completions are generated at runtime),
    #               ~6 B/tok for short Q&A-style prompts
    # These are PLANNING estimates. After tokenization the packed memmap is
    # 2 bytes/token (uint16) regardless of source format. The download
    # column tells you how much raw data to fetch; the on-disk packed
    # directory will be ~2.5× smaller.
    print()
    print("[ DOWNLOAD SIZES — raw data to fetch from HuggingFace ]")
    print(_bar("Pretrain corpus",  f"{human_bytes(pretrain_bytes)}  "
                                  f"({human_tokens(p.D_tokens)} tok × "
                                  f"{args.pretrain_bytes_per_token} B/tok)"))
    print(_bar("SFT corpus",       f"{human_bytes(sft_bytes)}  "
                                  f"({human_tokens(s.D_tokens)} tok × "
                                  f"{args.sft_bytes_per_token} B/tok)"))
    print(_bar("GRPO prompts",     f"{human_bytes(grpo_bytes)}  "
                                  f"({human_tokens(g.prompt_tokens)} tok × "
                                  f"{args.grpo_bytes_per_token} B/tok)"))
    print(_bar("Total raw data",   human_bytes(total_bytes)))
    print()
    print(_bar("Pretrain packed",  f"{human_bytes(pretrain_packed)}  "
                                  f"(uint16, 2 B/tok)"))
    print(_bar("SFT packed (t+m)", f"{human_bytes(sft_packed)}  "
                                  f"(tokens + mask, 4 B/tok)"))
    print(_bar("GRPO packed",      f"{human_bytes(grpo_packed)}  "
                                  f"(uint16, 2 B/tok)"))
    print(_bar("Total packed",     human_bytes(total_packed)))

    # ---- Pretrain ----
    print()
    print("[ PRETRAIN ]")
    print(_bar("Model params (N)",  human_params(p.N_non_embedding)))
    print(_bar("Tokens",            human_tokens(p.D_tokens)))
    print(_bar("Sequence length",   f"{p.seq_len}"))
    print(_bar("Micro batch / GPU", f"{p.micro_batch}"))
    print(_bar("Grad accum",        f"{p.grad_accum}"))
    print(_bar("World size",        f"{p.world_size}"))
    print(_bar("B_eff (tok/step)",  human_tokens(p.B_eff)))
    print(_bar("Optimizer steps",   f"{p.num_steps:,}"))
    print(_bar("Peak LR (AdamW)",   f"{p.lr_peak:.2e}"))
    print(_bar("Warmup steps",      f"{p.warmup_steps:,}"))
    print(_bar("Weight decay",      f"{p.weight_decay}"))
    print(_bar("Grad clip",         f"{p.grad_clip}"))
    print(_bar("Betas",             f"{p.betas[0]}, {p.betas[1]}"))

    # ---- SFT ----
    print()
    print("[ SFT ]")
    sft_mode = "LoRA" if s.lora else "Full fine-tune"
    print(_bar("Mode",              sft_mode))
    if s.lora:
        print(_bar("LoRA rank / α",  f"{s.lora_rank} / {s.lora_alpha}"))
    print(_bar("Tokens / param",     f"{args.sft_tokens_per_param}×"))
    print(_bar("Total SFT tokens",   human_tokens(s.D_tokens)))
    print(_bar("Samples / epoch",    f"~{s.samples:,}"))
    print(_bar("Epochs",             f"{s.epochs}"))
    print(_bar("Steps / epoch",      f"{s.num_steps:,}"))
    print(_bar("Total steps",        f"{s.num_steps * s.epochs:,}"))
    print(_bar("Sequence length",   f"{s.seq_len}"))
    print(_bar("Micro batch / GPU", f"{s.micro_batch}"))
    print(_bar("Grad accum",        f"{s.grad_accum}"))
    print(_bar("B_eff (tok/step)",  human_tokens(s.B_eff)))
    print(_bar("Optimizer steps",   f"{s.num_steps:,}"))
    print(_bar("Peak LR",           f"{s.lr_peak:.2e}"))
    print(_bar("Grad clip",         f"{s.grad_clip}"))
    print(_bar("Weight decay",      f"{s.weight_decay}"))

    # ---- GRPO ----
    print()
    print("[ GRPO ]")
    grpo_mode = "LoRA" if g.lora else "Full fine-tune"
    print(_bar("Mode",              grpo_mode))
    if g.lora:
        print(_bar("LoRA rank / α",  f"{g.lora_rank} / {g.lora_alpha}"))
    print(_bar("Prompts",           f"{g.prompts:,}"))
    print(_bar("Group size (G)",    f"{g.group_size}"))
    print(_bar("Policy updates",    f"{g.num_steps:,}"))
    print(_bar("Sequence length",   f"{g.seq_len}"))
    print(_bar("Max prompt len",    f"{g.max_prompt_len}"))
    print(_bar("Max response len",  f"{g.max_response_len}"))
    print(_bar("Rollout batch",     f"{g.rollout_batch}"))
    print(_bar("KL coefficient β",  f"{g.kl_beta}"))
    print(_bar("Clip ε",            f"{g.clip_eps}"))
    print(_bar("Peak LR (policy)",  f"{g.lr_peak:.2e}"))
    print(_bar("Grad clip",         f"{g.grad_clip}"))

    # ---- Verdict ----
    print()
    print("[ VERDICT ]")
    ratio = p.D_tokens / p.N_non_embedding
    if 15 <= ratio <= 25:
        verdict = f"OK — {ratio:.1f}× tokens/param is Chinchilla-optimal"
    elif ratio < 15:
        deficit_b = (CHINCHILLA_RATIO * p.N_non_embedding - p.D_tokens) / 1e9
        verdict = (
            f"UNDERTRAINED — {ratio:.1f}× is below Chinchilla. "
            f"Need {deficit_b:.2f}B more tokens, or shrink the model "
            f"to {human_params(chinchilla_params_for_tokens(p.D_tokens))}."
        )
    else:
        n_optimal = chinchilla_params_for_tokens(p.D_tokens)
        verdict = (
            f"OVERTRAINED — {ratio:.1f}× exceeds Chinchilla. "
            f"Optimal model for this data: {human_params(n_optimal)}."
        )
    print(f"  {verdict}")
    for note in plan.notes:
        print(f"  • {note}")

    # ---- Command lines ----
    print()
    print("[ COMMAND LINES ]")
    # If the user gave a --target-size, echo it back as the model-size
    # label so the command line matches what they asked for. The plan
    # was computed using their exact N, so this is the right number to
    # train. (The user can swap to a real preset by re-running with
    # --target-size <preset_name> if their target was hypothetical.)
    if args.target_size:
        model_label = args.target_size
    else:
        model_label = p.model_name
    print("# Pretrain")
    print("  " + p.command_line(
        model_size_label=model_label,
        packed_dir=args.packed_dir,
        out_dir=args.pretrain_out,
    ))
    print("# SFT")
    print("  " + s.command_line(
        checkpoint=os.path.join(args.pretrain_out, "latest.pt"),
        tokenizer=args.tokenizer_dir,
        cache_dir=args.sft_cache_dir,
        out_dir=args.sft_out,
    ))
    print()
    print("=" * 78)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Chinchilla-aware calculator for Qwen3-Non-MOE training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input source (exactly one of these)
    src = p.add_argument_group("input (provide one)")
    src.add_argument("--data-size", type=str, default=None,
                    help="Dataset size on disk, e.g. 5GB, 500MB, 1.2T")
    src.add_argument("--tokens", type=int, default=None,
                    help="Raw training token count")
    src.add_argument("--target-size", type=str, default=None,
                    help="Model size, e.g. 0.6B, 1.7B, 8B")
    src.add_argument("--bytes-per-token", type=float, default=DEFAULT_BYTES_PER_TOKEN,
                    help=f"Bytes per token for --data-size input (default {DEFAULT_BYTES_PER_TOKEN})")
    src.add_argument("--pretrain-bytes-per-token", type=float, default=DEFAULT_BPT_PRETRAIN,
                    help=f"Bytes/token when sizing the pretrain download "
                         f"(default {DEFAULT_BPT_PRETRAIN})")
    src.add_argument("--sft-bytes-per-token", type=float, default=DEFAULT_BPT_SFT,
                    help=f"Bytes/token when sizing the SFT download "
                         f"(default {DEFAULT_BPT_SFT})")
    src.add_argument("--grpo-bytes-per-token", type=float, default=DEFAULT_BPT_GRPO,
                    help=f"Bytes/token when sizing the GRPO download "
                         f"(default {DEFAULT_BPT_GRPO})")

    # Training knobs
    kn = p.add_argument_group("training knobs")
    kn.add_argument("--seq-len", type=int, default=4096)
    kn.add_argument("--micro-batch", type=int, default=0,
                    help="Micro batch per GPU (0 = auto)")
    kn.add_argument("--grad-accum", type=int, default=0,
                    help="Gradient accumulation steps (0 = auto)")
    kn.add_argument("--world-size", type=int, default=1)
    kn.add_argument("--group-size", type=int, default=8)
    kn.add_argument("--sft-epochs", type=int, default=3)
    kn.add_argument("--sft-tokens-per-param", type=int, default=100,
                    help="SFT tokens per non-embedding param (50-200; Chinchilla=100)")
    kn.add_argument("--lora", action="store_true", help="Use LoRA for SFT and GRPO")
    kn.add_argument("--lora-rank", type=int, default=64)
    kn.add_argument("--lora-alpha", type=int, default=128)

    # Output paths (for the printed command lines)
    out = p.add_argument_group("paths")
    out.add_argument("--packed-dir",      default="./packed")
    out.add_argument("--pretrain-out",    default="./checkpoints")
    out.add_argument("--tokenizer-dir",   default="./tokenizer")
    out.add_argument("--sft-cache-dir",   default="./sft_packed")
    out.add_argument("--sft-out",         default="./sft_checkpoints")

    # Output format
    fmt = p.add_argument_group("output")
    fmt.add_argument("--json", action="store_true", help="Emit JSON instead of pretty print")
    fmt.add_argument("--quiet", action="store_true", help="Suppress everything but the plan")

    args = p.parse_args(argv)

    # ---- Validate exactly one input ----
    n_inputs = sum(x is not None for x in (args.data_size, args.tokens, args.target_size))
    if n_inputs == 0:
        # Default: 5 GB of data
        args.data_size = "5GB"
    elif n_inputs > 1:
        p.error("Provide at most one of --data-size, --tokens, --target-size")

    notes: list[str] = []

    # ---- Resolve tokens (D) and non-embedding params (N) ----
    if args.data_size:
        args.data_size_bytes = parse_size_to_bytes(args.data_size)
        D_tokens = int(args.data_size_bytes / args.bytes_per_token)
        arch = pick_arch_for_budget(D_tokens)
        N = arch["N"]
        notes.append(
            f"Estimated tokens from {args.data_size} at "
            f"{args.bytes_per_token} bytes/token."
        )
        notes.append(
            f"Chinchilla-optimal model for this data: {arch['name']} "
            f"(N*={human_params(N)})."
        )
    elif args.tokens:
        D_tokens = args.tokens
        arch = pick_arch_for_budget(D_tokens)
        N = arch["N"]
        notes.append(
            f"Chinchilla-optimal model for {human_tokens(D_tokens)} tokens: "
            f"{arch['name']} (N={human_params(N)})."
        )
    elif args.target_size:
        # The user gave us an exact parameter target. We use it AS-IS for
        # all math (D = 20N, SFT = 100N, GRPO scheduling, etc.) and do
        # NOT snap to a preset. We only look up the nearest Qwen3 preset
        # for display purposes (H, L, heads, I) — and we surface in a
        # note when the user's target is meaningfully different from
        # every actual Qwen3 architecture we have a closed-form for.
        N = parse_target_params(args.target_size)
        D_tokens = chinchilla_tokens_for_params(N)
        arch = pick_arch_for_params(N)
        # If the user typed a label that doesn't match any preset name
        # (e.g. "1.5B" or "2.3B"), the preset we picked is a reference
        # only — the math uses the user's exact N, not the preset's N.
        if _normalize_target_label(args.target_size) not in _PRESET_NAMES:
            diff_pct = abs(arch["N"] - N) / N * 100
            notes.append(
                f"User target N={human_params(N)} ({args.target_size}) is not "
                f"a standard Qwen3 preset; nearest preset is {arch['name']} "
                f"(N={human_params(arch['N'])}, diff {diff_pct:.1f}%). "
                f"All math uses the user's exact N. The preset dimensions "
                f"are shown for reference only — pass "
                f"--target-size {arch['name']} to compute for a real "
                f"Qwen3 architecture."
            )

    # ---- Auto batch sizing if user didn't set them ----
    micro_batch = args.micro_batch or micro_batch_for_params(N, args.seq_len, has_grad_ckpt=True)
    grad_accum  = args.grad_accum
    target_B    = eff_batch_for_params(N)
    if grad_accum == 0:
        grad_accum = max(1, round(target_B / (micro_batch * args.world_size * args.seq_len)))
    B_eff = micro_batch * grad_accum * args.world_size * args.seq_len

    # ---- SFT sample sizing ----
    # D_sft_opt ≈ 100 × N tokens of SFT data (50-200 is the valid range).
    # This is the TOTAL token budget across all epochs, not per-epoch.
    # A "sample" is roughly 600 tokens. "epochs" controls how many passes
    # over the dataset you make, but the optimizer-step count is fixed by
    # D_sft_total / B_eff.
    D_sft_total = int(args.sft_tokens_per_param * N)
    sft_samples_per_epoch = max(1, D_sft_total // 600)
    sft_epochs = max(1, args.sft_epochs)
    # Per-epoch tokens = D_sft_total / epochs (split the budget across passes)
    sft_tokens_per_epoch = D_sft_total // sft_epochs

    # ---- Build plans ----
    pretrain = build_pretrain_plan(
        N=N, arch=arch, D_tokens=D_tokens,
        seq_len=args.seq_len,
        micro_batch=micro_batch, grad_accum=grad_accum, world_size=args.world_size,
    )
    sft = build_sft_plan(
        N=N, seq_len=args.seq_len,
        micro_batch=micro_batch, grad_accum=grad_accum, world_size=args.world_size,
        D_sft_total=D_sft_total, epochs=sft_epochs,
        use_lora=args.lora,
    )
    grpo = build_grpo_plan(
        N=N, sft_tokens=D_sft_total, group_size=args.group_size,
        seq_len=args.seq_len, world_size=args.world_size, use_lora=args.lora,
    )

    plan = FullPlan(pretrain=pretrain, sft=sft, grpo=grpo, notes=notes)

    # ---- Compute download + packed sizes (used by both JSON and pretty print) ----
    pretrain_bytes = int(pretrain.D_tokens * args.pretrain_bytes_per_token)
    sft_bytes      = int(sft.D_tokens     * args.sft_bytes_per_token)
    grpo_bytes     = int(grpo.prompt_tokens * args.grpo_bytes_per_token)
    total_bytes    = pretrain_bytes + sft_bytes + grpo_bytes
    pretrain_packed = pretrain.D_tokens * 2
    sft_packed      = sft.D_tokens * 2 * 2   # tokens.bin + mask.bin
    grpo_packed     = grpo.prompt_tokens * 2
    total_packed    = pretrain_packed + sft_packed + grpo_packed

    # ---- Emit ----
    if args.json:
        out = {
            "input": {
                "data_size": args.data_size,
                "data_size_bytes": getattr(args, "data_size_bytes", None),
                "tokens": D_tokens,
                "target_size": args.target_size,
                "bytes_per_token": args.bytes_per_token,
            },
            "pretrain": {k: v for k, v in asdict(pretrain).items() if k != "betas"},
            "sft": asdict(sft),
            "grpo": asdict(grpo),
            "downloads": {
                "pretrain_bytes": pretrain_bytes,
                "sft_bytes":      sft_bytes,
                "grpo_bytes":     grpo_bytes,
                "total_bytes":    total_bytes,
                "pretrain_packed_bytes": pretrain_packed,
                "sft_packed_bytes":      sft_packed,
                "grpo_packed_bytes":     grpo_packed,
                "total_packed_bytes":    total_packed,
            },
            "notes": notes,
        }
        print(json.dumps(out, indent=2))
    else:
        if not args.quiet:
            print_plan(
                plan, args,
                pretrain_bytes=pretrain_bytes, sft_bytes=sft_bytes,
                grpo_bytes=grpo_bytes, total_bytes=total_bytes,
                pretrain_packed=pretrain_packed, sft_packed=sft_packed,
                grpo_packed=grpo_packed, total_packed=total_packed,
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
