#!/usr/bin/env python3
"""
train_sft.py

Supervised Fine-Tuning (SFT) script — Stage 1 of reasoning post-training.

Loads the pretrained checkpoint produced by train.py and the packed memmap
files produced by pack_sft_data.py, and fine-tunes the model using a
ChatML + <think>...</think> template, before the RL stage (grpo.py).

This script does NOT read or tokenise raw JSONL — that is pack_sft_data.py's
job. Run pack_sft_data.py first (optionally with --worker/--num-workers to
parallelise packing across processes), then point --cache-dir here at the
same directory.

Key features:
  - Loss masking: prompt tokens are masked (-100), loss computed only on
    the assistant turn (thinking + answer), so the model learns to generate
    reasoning, not to predict the question
  - Sample packing: multiple variable-length examples are packed into one
    fixed-length window; a position-level mask prevents loss from bleeding
    across sample boundaries
  - LoRA (optional): inject low-rank adapters into Q/K/V/O projections so
    fine-tuning a large model fits on a single GPU; adapters can be merged
    back into full weights for deployment
  - DDP multi-GPU (torchrun) support, identical to train.py
  - Checkpoint save/resume with LoRA-aware state handling
  - Reads packed data written by one or more pack_sft_data.py workers and
    concatenates them transparently via mmap (no copy into RAM)

Usage:
    # 1. Pack the data first (see pack_sft_data.py --help)
    python pack_sft_data.py --data-dir ./sft_data --tokenizer ./tokenizer \\
        --cache-dir ./sft_packed

    # 2. Full fine-tune (small model / LoRA for large)
    python train_sft.py \\
        --checkpoint ./checkpoints/latest.pt \\
        --tokenizer  ./tokenizer \\
        --cache-dir  ./sft_packed \\
        --out-dir    ./sft_checkpoints

    # LoRA fine-tune (recommended for 1B+ on a single 4090)
    python train_sft.py \\
        --checkpoint ./checkpoints/latest.pt \\
        --tokenizer  ./tokenizer \\
        --cache-dir  ./sft_packed \\
        --lora --lora-rank 64 --lora-alpha 128 \\
        --out-dir    ./sft_checkpoints

    # Multi-GPU
    torchrun --nproc_per_node=4 train_sft.py --checkpoint ... --lora

    # Merge LoRA weights back into the base model after training
    python train_sft.py --merge-lora \\
        --checkpoint ./sft_checkpoints/latest.pt \\
        --out-dir    ./sft_merged
"""

import argparse
import glob
import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from tokenizers import Tokenizer

from model import Qwen3Config, Qwen3ForCausalLM, count_parameters


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear that adds a low-rank adapter:
        output = W x + (B A x) * scale
    where A ∈ R^{rank × in}, B ∈ R^{out × rank}, scale = alpha / rank.

    The original weight W is frozen; only A and B are trained.
    """

    def __init__(self, linear: nn.Linear, rank: int = 64, alpha: float = 128.0):
        super().__init__()
        self.linear = linear
        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)

        in_f  = linear.in_features
        out_f = linear.out_features
        self.rank  = rank
        self.scale = alpha / rank

        device = linear.weight.device
        dtype  = linear.weight.dtype

        # Kaiming initialisation for A, zero for B (so adapter starts as
        # identity — the base model is unchanged at the start of SFT)
        self.lora_A = nn.Parameter(torch.empty(rank, in_f, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)
        lora = (x @ self.lora_A.T @ self.lora_B.T) * self.scale
        return base + lora

    def merge(self) -> nn.Linear:
        """Return a plain nn.Linear with the LoRA delta fused into W."""
        merged = nn.Linear(
            self.linear.in_features,
            self.linear.out_features,
            bias=self.linear.bias is not None,
            device=self.linear.weight.device,
            dtype=self.linear.weight.dtype,
        )
        delta = (self.lora_B @ self.lora_A) * self.scale
        merged.weight = nn.Parameter(self.linear.weight + delta.to(self.linear.weight.dtype))
        if self.linear.bias is not None:
            merged.bias = nn.Parameter(self.linear.bias.clone())
        return merged


def inject_lora(
    model: Qwen3ForCausalLM,
    rank: int = 64,
    alpha: float = 128.0,
    target_modules: Tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ),
) -> int:
    """
    Walk the model and replace every nn.Linear whose name ends with a
    target module name with a LoRALinear.  Returns the number of adapters
    injected.
    """
    replaced = 0
    for module_path, module in list(model.named_modules()):
        for target in target_modules:
            if module_path.endswith(target) and isinstance(module, nn.Linear):
                parent_path, attr = module_path.rsplit(".", 1)
                parent = model
                for part in parent_path.split("."):
                    parent = getattr(parent, part)
                setattr(parent, attr, LoRALinear(module, rank=rank, alpha=alpha))
                replaced += 1
                break
    return replaced


def merge_lora(model: Qwen3ForCausalLM) -> Qwen3ForCausalLM:
    """Replace every LoRALinear in the model with its merged nn.Linear."""
    for module_path, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            parent_path, attr = module_path.rsplit(".", 1)
            parent = model
            for part in parent_path.split("."):
                parent = getattr(parent, part)
            setattr(parent, attr, module.merge())
    return model


def lora_state_dict(model: nn.Module) -> dict:
    """Return only the LoRA parameters (A, B matrices) for compact checkpoints."""
    return {
        k: v for k, v in model.state_dict().items()
        if "lora_A" in k or "lora_B" in k
    }


def lora_parameter_count(model: nn.Module) -> int:
    return sum(
        p.numel() for n, p in model.named_parameters()
        if ("lora_A" in n or "lora_B" in n)
    )


# ---------------------------------------------------------------------------
# Tokenizer helpers
# ---------------------------------------------------------------------------

def load_tokenizer(tokenizer_dir: str) -> Tokenizer:
    path = os.path.join(tokenizer_dir, "tokenizer.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"tokenizer.json not found in {tokenizer_dir}")
    return Tokenizer.from_file(path)


# ---------------------------------------------------------------------------
# Dataset: reads packed memmap .bin files written by pack_sft_data.py
# ---------------------------------------------------------------------------
# pack_sft_data.py may have been run with multiple --worker shards, each
# writing its own pair of <split>_tokens/<split>_mask .bin files plus a
# manifest. This class discovers every manifest for the requested split in
# cache_dir and concatenates them into one logical array, still backed by
# mmap (OS page cache, not the Python heap), so RAM use stays flat
# regardless of dataset size or worker count.

class _ConcatMemmap:
    """
    Read-only view that makes several np.memmap arrays look like one
    contiguous array, without copying any of them into RAM.
    """

    def __init__(self, arrays):
        self.arrays = [a for a in arrays if len(a) > 0]
        self.lengths = [len(a) for a in self.arrays]
        self.offsets = np.cumsum([0] + self.lengths)
        self.total = int(self.offsets[-1])

    def __len__(self):
        return self.total

    def __array__(self, dtype=None):
        # Supports np.asarray(concat_memmap) / np.array(concat_memmap).
        # Materializes exactly the pieces this instance holds — for a
        # per-step training window (this class's main use case via
        # __getitem__ below) that's tiny (seq_len-ish elements, usually
        # just 1-2 pieces). Callers holding a _ConcatMemmap spanning a
        # much larger range (e.g. GRPO's whole-dataset probe) should
        # avoid calling this unless they specifically want the one-time
        # full-materialization cost.
        if not self.arrays:
            return np.array([], dtype=dtype or np.uint16)
        pieces = [np.asarray(a) for a in self.arrays]
        out = pieces[0] if len(pieces) == 1 else np.concatenate(pieces)
        return out.astype(dtype) if dtype is not None else out

    def _locate(self, idx):
        # Find which underlying array `idx` falls into.
        arr_i = int(np.searchsorted(self.offsets, idx, side="right") - 1)
        return arr_i, idx - self.offsets[arr_i]

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(self.total)
            assert step == 1, "strided slicing not supported"
            if start >= stop:
                return _ConcatMemmap([])
            # Fast path: slice lies entirely within one underlying array
            arr_i_start, local_start = self._locate(start)
            arr_i_end,   local_end   = self._locate(stop - 1)
            if arr_i_start == arr_i_end:
                return _ConcatMemmap([self.arrays[arr_i_start][local_start: local_end + 1]])
            # Slow path: spans multiple arrays. Historically this eagerly
            # np.concatenate'd the pieces into one RAM-resident array —
            # fine for a small per-step training window that happens to
            # straddle a shard boundary, but this same __getitem__ is also
            # what SFTDataset.__init__ uses for its rank-sharding slice,
            # which can span the *entire* dataset (e.g. world_size=1).
            # That silently forced a multi-GB anonymous-memory copy that
            # the OS can't reclaim the way it can mmap'd pages — breaking
            # the "RAM stays flat" promise this class is supposed to keep.
            # Instead, stitch the (still memmap-backed, no-copy) pieces
            # into a new lazy _ConcatMemmap. Callers that truly need a
            # single contiguous in-RAM ndarray (e.g. _contiguous_view)
            # materialize explicitly and deliberately, once.
            pieces = []
            cur = start
            while cur < stop:
                arr_i, local = self._locate(cur)
                arr = self.arrays[arr_i]
                take = min(len(arr) - local, stop - cur)
                pieces.append(arr[local: local + take])
                cur += take
            return _ConcatMemmap(pieces)
        else:
            arr_i, local = self._locate(key)
            return self.arrays[arr_i][local]


class SFTDataset:
    """
    Reads packed memmap files produced by pack_sft_data.py:
        <cache_dir>/sft_<split>_tokens.w{i}-of-{n}.bin   — uint16/uint32 token ids
        <cache_dir>/sft_<split>_mask.w{i}-of-{n}.bin     — uint8 loss mask (1=compute loss)
        <cache_dir>/sft_manifest.w{i}-of-{n}.json        — per-worker metadata

    All worker shards found for the requested split are discovered,
    sorted by worker index, and concatenated (still mmap-backed, no RAM
    copy). The result is then sharded again across DDP ranks by token
    count, exactly as before.
    """

    def __init__(
        self,
        cache_dir: str,
        seq_len: int,
        rank: int = 0,
        world_size: int = 1,
        split: str = "train",
    ):
        self.seq_len    = seq_len
        self.rank       = rank
        self.world_size = world_size
        self.split      = split

        manifests = self._discover_manifests(cache_dir, split)
        if not manifests:
            raise FileNotFoundError(
                f"No packed manifests found for split={split!r} in {cache_dir}. "
                f"Run pack_sft_data.py first (and make sure every worker "
                f"finished writing its sft_manifest.w*.json)."
            )

        dtype_t = np.dtype(manifests[0]["dtype_t"])
        dtype_m = np.dtype(manifests[0]["dtype_m"])

        token_arrays = []
        mask_arrays  = []
        total_records = 0
        for m in manifests:
            tok_path  = os.path.join(cache_dir, m[f"{split}_tokens_file"])
            mask_path = os.path.join(cache_dir, m[f"{split}_mask_file"])
            token_arrays.append(np.memmap(tok_path,  dtype=dtype_t, mode="r"))
            mask_arrays.append(np.memmap(mask_path, dtype=dtype_m, mode="r"))
            total_records += m["n_records"]

        self.tokens = _ConcatMemmap(token_arrays)
        self.mask   = _ConcatMemmap(mask_arrays)
        # NOTE: self.tokens/self.mask get reassigned below via __getitem__
        # for rank-sharding, which always returns a plain ndarray/memmap
        # (never another _ConcatMemmap) — so the shard count must be
        # captured here, before that happens, for any caller that wants it.
        self.n_shards = len(token_arrays)

        if rank == 0:
            print(f"[SFTDataset] {split}: discovered {len(manifests)} worker "
                  f"shard(s) in {cache_dir} ({total_records:,} records total)")

        # Shard across DDP ranks by token count
        total = len(self.tokens)
        shard_size = total // world_size
        start = rank * shard_size
        end   = start + shard_size if rank < world_size - 1 else total
        self.tokens = self.tokens[start:end]
        self.mask   = self.mask[start:end]

        n_windows = max(0, (len(self.tokens) - 1) // seq_len)
        print(f"[SFTDataset rank {rank}] {split}: {len(self.tokens):,} tokens "
              f"-> {n_windows:,} windows of {seq_len}")

    # ------------------------------------------------------------------
    @staticmethod
    def _discover_manifests(cache_dir: str, split: str):
        """
        Find every sft_manifest.w*.json in cache_dir that has packed data
        for `split`, sorted by worker index for deterministic concatenation.
        """
        manifest_paths = sorted(glob.glob(os.path.join(cache_dir, "sft_manifest.w*.json")))
        manifests = []
        for p in manifest_paths:
            with open(p, "r") as f:
                m = json.load(f)
            if m.get(f"{split}_tokens") is None and m.get(f"{split}_tokens_file") is None:
                continue
            manifests.append(m)
        manifests.sort(key=lambda m: m["worker"])
        return manifests

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return max(0, (len(self.tokens) - 1) // self.seq_len)

    def get_batch(self, batch_size: int, device: torch.device):
        """Sample `batch_size` random windows; return (x, y, loss_mask)."""
        n = len(self)
        if n == 0:
            raise RuntimeError(
                "SFT dataset has no complete windows. "
                "Try a smaller --seq-len, or re-run pack_sft_data.py with a "
                "larger --target-size in download_sft_data.py."
            )
        starts = torch.randint(0, n, (batch_size,)) * self.seq_len
        xs, ys, ms = [], [], []
        for s in starts.tolist():
            s = min(int(s), len(self.tokens) - self.seq_len - 1)
            xs.append(torch.from_numpy(
                np.asarray(self.tokens[s     : s + self.seq_len    ]).astype(np.int64)))
            ys.append(torch.from_numpy(
                np.asarray(self.tokens[s + 1 : s + self.seq_len + 1]).astype(np.int64)))
            ms.append(torch.from_numpy(
                np.asarray(self.mask  [s + 1 : s + self.seq_len + 1]).astype(np.float32)))
        x = torch.stack(xs)
        y = torch.stack(ys)
        m = torch.stack(ms)
        if device.type == "cuda":
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
            m = m.pin_memory().to(device, non_blocking=True)
        else:
            x, y, m = x.to(device), y.to(device), m.to(device)
        return x, y, m


# ---------------------------------------------------------------------------
# Masked loss (only assistant tokens contribute)
# ---------------------------------------------------------------------------

def masked_cross_entropy(
    logits: torch.Tensor,    # (B, T, V)
    targets: torch.Tensor,   # (B, T)
    mask: torch.Tensor,      # (B, T) float, 1=compute loss, 0=ignore
) -> torch.Tensor:
    B, T, V = logits.shape
    logits_flat  = logits.reshape(B * T, V)
    targets_flat = targets.reshape(B * T)
    mask_flat    = mask.reshape(B * T)

    # token-level NLL
    nll = F.cross_entropy(logits_flat, targets_flat, reduction="none")
    # mask and mean over active positions only
    denom = mask_flat.sum().clamp(min=1.0)
    return (nll * mask_flat).sum() / denom


# ---------------------------------------------------------------------------
# Distributed helpers  (same as train.py)
# ---------------------------------------------------------------------------

def setup_distributed():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank       = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank = local_rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return rank, local_rank, world_size, device


def is_master(rank):
    return rank == 0


# ---------------------------------------------------------------------------
# LR schedule (same cosine as train.py)
# ---------------------------------------------------------------------------

def get_lr(step, warmup, max_steps, max_lr, min_lr):
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step >= max_steps:
        return min_lr
    t = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + 0.5 * (1 + math.cos(math.pi * t)) * (max_lr - min_lr)


# ---------------------------------------------------------------------------
# Optimizer builder (excludes LoRA norms from weight decay automatically)
# ---------------------------------------------------------------------------

def build_optimizer(model, lr, weight_decay):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "norm" in name or "embed" in name or "lora_B" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"params": decay,    "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    n_decay    = sum(p.numel() for p in decay)
    n_no_decay = sum(p.numel() for p in no_decay)
    print(f"[Optimizer] trainable: decay={n_decay:,}  no_decay={n_no_decay:,}")
    return torch.optim.AdamW(
        groups, lr=lr, betas=(0.9, 0.95), eps=1e-8,
        fused=torch.cuda.is_available(),
    )


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _raw(model):
    m = model.module if isinstance(model, DDP) else model
    return m._orig_mod if hasattr(m, "_orig_mod") else m


def save_checkpoint(out_dir, step, model, optimizer, config, args_dict,
                    best_val_loss, is_lora):
    raw = _raw(model)
    ckpt = {
        "step":            step,
        "model_state":     lora_state_dict(raw) if is_lora else raw.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config":          vars(config),
        "args":            args_dict,
        "best_val_loss":   best_val_loss,
        "is_lora":         is_lora,
    }
    path = os.path.join(out_dir, f"sft_step{step:07d}.pt")
    torch.save(ckpt, path)
    latest = os.path.join(out_dir, "latest.pt")
    if os.path.islink(latest): os.remove(latest)
    os.symlink(os.path.abspath(path), latest)
    print(f"[Checkpoint] saved {path}")
    return path


def load_checkpoint(path, model, optimizer, device, is_lora):
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    raw   = _raw(model)
    state = ckpt["model_state"]
    if is_lora:
        # Only load the LoRA keys; base weights already loaded from pretrained ckpt
        missing, unexpected = raw.load_state_dict(state, strict=False)
        lora_keys = [k for k in state if "lora_A" in k or "lora_B" in k]
        print(f"[Checkpoint] loaded {len(lora_keys)} LoRA tensors from {path}")
    else:
        raw.load_state_dict(state)
        if hasattr(raw, "tie_weights"):
            raw.tie_weights()
    if optimizer and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    step          = ckpt.get("step", 0)
    best_val_loss = ckpt.get("best_val_loss", float("inf"))
    print(f"[Checkpoint] resumed from step {step}, best_val={best_val_loss:.4f}")
    return step, best_val_loss


def prune_checkpoints(out_dir, keep=3):
    ckpts = sorted(
        Path(out_dir).glob("sft_step*.pt"),
        key=lambda p: int(p.stem.replace("sft_step", "")),
    )
    for old in ckpts[:-keep]:
        old.unlink()
        print(f"[Checkpoint] pruned {old.name}")


# ---------------------------------------------------------------------------
# Eval pass
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, dataset, eval_steps, batch_size, device, ctx,
             use_cudagraphs=False):
    model.eval()
    losses = []
    for _ in range(eval_steps):
        x, y, m = dataset.get_batch(batch_size, device)
        with ctx:
            if use_cudagraphs:
                torch.compiler.cudagraph_mark_step_begin()
            out    = model(x)
            logits = out["logits"]
        loss = masked_cross_entropy(logits, y, m)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses)) if losses else float("inf")


# ---------------------------------------------------------------------------
# Merge-only mode: fuse LoRA into base weights and save
# ---------------------------------------------------------------------------

def merge_and_save(args):
    device = torch.device("cpu")
    ckpt   = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = Qwen3Config(**ckpt["config"])
    model  = Qwen3ForCausalLM(config)

    # Load base weights (from the original pretrain checkpoint stored in args)
    pretrain_path = ckpt["args"].get("checkpoint")
    if pretrain_path and os.path.exists(pretrain_path):
        base_ckpt = torch.load(pretrain_path, map_location=device, weights_only=False)
        model.load_state_dict(base_ckpt["model_state"])
        model.tie_weights()
        print(f"[Merge] loaded base weights from {pretrain_path}")
    else:
        print("[Merge] WARNING: base checkpoint path not found; "
              "LoRA will be merged onto random weights.")

    inject_lora(model, rank=ckpt["args"].get("lora_rank", 64),
                alpha=ckpt["args"].get("lora_alpha", 128.0))
    missing, _ = model.load_state_dict(ckpt["model_state"], strict=False)
    model      = merge_lora(model)
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

    # ---------------------------------------------------------------- model
    # Load config from the pretrained checkpoint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}\n"
            f"Run train.py first to produce a pretrained checkpoint."
        )
    ckpt_data = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config    = Qwen3Config(**ckpt_data["config"])

    model = Qwen3ForCausalLM(config).to(device)
    model.load_state_dict(ckpt_data["model_state"])
    model.tie_weights()

    if master:
        n_total = count_parameters(model)
        print(f"Loaded pretrained model: {n_total:,} params ({n_total/1e9:.3f}B)")

    # ----------------------------------------------------------------- LoRA
    is_lora = args.lora
    if is_lora:
        n_replaced = inject_lora(
            model,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
        )
        n_trainable = lora_parameter_count(model)
        n_total     = count_parameters(model)
        if master:
            print(f"[LoRA] injected {n_replaced} adapters | "
                  f"trainable={n_trainable:,} / total={n_total:,} "
                  f"({100*n_trainable/n_total:.2f}%)")
    else:
        if master:
            print("[LoRA] disabled — full fine-tune")

    # ---------------------------------------------------------- torch.compile
    _use_cudagraphs = False
    if args.compile:
        mode = args.compile_mode
        if master:
            print(f"[compile] torch.compile(mode='{mode}')…")
        model = torch.compile(model, mode=mode)
        _use_cudagraphs = (mode == "reduce-overhead")

    # ------------------------------------------------------------------- DDP
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    # ------------------------------------------------------------------ data
    if master:
        print(f"\nReading packed SFT data from {args.cache_dir} …")
        print(f"(produced by pack_sft_data.py — run it first if this fails)")

    train_ds = SFTDataset(
        cache_dir=args.cache_dir,
        seq_len=args.seq_len,
        rank=rank, world_size=world_size,
        split="train",
    )
    val_ds = SFTDataset(
        cache_dir=args.cache_dir,
        seq_len=args.seq_len,
        rank=rank, world_size=world_size,
        split="val",
    )

    if len(train_ds) == 0:
        raise RuntimeError(
            "Training dataset has no complete windows. "
            "Try a smaller --seq-len, or re-pack with a larger "
            "--target-size in download_sft_data.py."
        )

    # ---------------------------------------------------------------- optim
    optimizer = build_optimizer(model, args.lr, args.weight_decay)

    # ----------------------------------------------------------------- amp
    use_amp = device.type == "cuda" and args.dtype == "bf16"
    ctx     = (torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
               if use_amp else nullcontext())

    # --------------------------------------------------------------- resume
    start_step    = 0
    best_val_loss = float("inf")
    if args.resume:
        start_step, best_val_loss = load_checkpoint(
            args.resume, model, optimizer, device, is_lora
        )

    if master:
        os.makedirs(args.out_dir, exist_ok=True)

    tokens_per_step = (
        args.batch_size * args.seq_len * args.grad_accum_steps * world_size
    )
    if master:
        print(f"\nTokens / step    : {tokens_per_step:,}")
        print(f"Effective batch  : {args.batch_size * args.grad_accum_steps * world_size}")
        print(f"Max steps        : {args.max_steps:,}")
        print(f"Checkpoint every : {args.ckpt_interval:,} steps\n")

    # ================================================================= LOOP
    model.train()
    optimizer.zero_grad(set_to_none=True)
    t0          = time.perf_counter()
    loss_accum  = 0.0

    for step in range(start_step, args.max_steps):
        lr = get_lr(step, args.warmup_steps, args.max_steps, args.lr, args.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # ---- gradient accumulation
        for micro in range(args.grad_accum_steps):
            x, y, m = train_ds.get_batch(args.batch_size, device)
            sync     = (micro == args.grad_accum_steps - 1)
            ctx_ddp  = nullcontext() if (world_size == 1 or sync) else model.no_sync()

            with ctx_ddp:
                with ctx:
                    if _use_cudagraphs:
                        torch.compiler.cudagraph_mark_step_begin()
                    out    = model(x)
                    logits = out["logits"]
                # masked loss: only assistant tokens contribute
                loss = masked_cross_entropy(logits, y, m) / args.grad_accum_steps
                loss.backward()

            loss_accum += loss.item()

        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if device.type == "cuda":
            torch.cuda.synchronize()

        # ---- logging
        if master and step % args.log_interval == 0:
            t1  = time.perf_counter()
            tok_per_sec = tokens_per_step * args.log_interval / max(t1 - t0, 1e-9)
            loss_display = loss_accum / args.log_interval
            loss_accum = 0.0
            print(
                f"step {step:7d} | loss {loss_display:.4f} | lr {lr:.2e} | "
                f"grad {grad_norm:.3f} | {tok_per_sec/1e3:.1f}k tok/s"
            )
            t0 = t1

        # ---- validation
        if step % args.eval_interval == 0 and step > start_step:
            val_loss = evaluate(
                model, val_ds, args.eval_steps, args.batch_size,
                device, ctx, _use_cudagraphs,
            )
            if world_size > 1:
                vl = torch.tensor(val_loss, device=device)
                dist.all_reduce(vl, op=dist.ReduceOp.AVG)
                val_loss = vl.item()
            if master:
                improved = " ✓ best" if val_loss < best_val_loss else ""
                print(f"  [eval] step {step:7d} | val_loss {val_loss:.4f}{improved}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss

        # ---- checkpoint
        if master and step % args.ckpt_interval == 0 and step > start_step:
            save_checkpoint(
                args.out_dir, step, model, optimizer, config,
                vars(args), best_val_loss, is_lora,
            )
            prune_checkpoints(args.out_dir, keep=args.keep_ckpts)

    # ---- final checkpoint
    if master:
        save_checkpoint(
            args.out_dir, args.max_steps, model, optimizer, config,
            vars(args), best_val_loss, is_lora,
        )
        print(f"\nSFT complete. Best val loss: {best_val_loss:.4f}")
        if is_lora:
            print(f"\nTo merge LoRA into base weights for deployment:")
            print(f"  python train_sft.py --merge-lora "
                  f"--checkpoint {args.out_dir}/latest.pt "
                  f"--out-dir ./sft_merged")

    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def smoke_test():
    print("\n=== SFT smoke test ===")
    import tempfile, shutil
    import sys

    tmp       = tempfile.mkdtemp()
    data_dir  = os.path.join(tmp, "sft_data", "math")
    ckpt_dir  = os.path.join(tmp, "ckpts")
    tok_dir   = os.path.join(tmp, "tokenizer")
    cache_dir = os.path.join(tmp, "sft_packed")
    os.makedirs(data_dir,  exist_ok=True)
    os.makedirs(ckpt_dir,  exist_ok=True)
    os.makedirs(tok_dir,   exist_ok=True)

    # ---- minimal tokenizer
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers import pre_tokenizers, decoders

    SPECIAL = ["<|endoftext|>","<|pad|>","<|im_start|>","<|im_end|>",
               "<think>","</think>"]
    tok = Tokenizer(BPE(unk_token=None, byte_fallback=True))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder       = decoders.ByteLevel()
    trainer = BpeTrainer(vocab_size=512, special_tokens=SPECIAL,
                         initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                         show_progress=False)
    corpus  = [
        "Solve 2+2. Think step by step. The answer is 4.",
        "What is 10-3? Reasoning: 10-3=7. Answer: 7.",
        "<|im_start|>user\nSolve 3+4<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n3+4=7\n</think>\n7<|im_end|>\n",
    ] * 30
    tok.train_from_iterator(corpus, trainer=trainer)
    tok.save(os.path.join(tok_dir, "tokenizer.json"))

    # ---- tiny SFT data
    records = [
        {"prompt": "Solve: 2+2", "thinking": "2 plus 2 equals 4",
         "answer": "4", "source": "test", "category": "math"},
        {"prompt": "What is 10-3?", "thinking": "10 minus 3 is 7",
         "answer": "7", "source": "test", "category": "math"},
    ] * 50
    shard_path = os.path.join(data_dir, "math_00000.jsonl")
    with open(shard_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    # ---- pack the data using pack_sft_data.py (imported as a module)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import pack_sft_data

    tokenizer = pack_sft_data.load_tokenizer(tok_dir)
    print(f"Tokenizer vocab: {tokenizer.get_vocab_size()}")
    pack_sft_data.pack_worker_shard(
        data_dir=os.path.join(tmp, "sft_data"),
        tokenizer=tokenizer,
        cache_dir=cache_dir,
        max_len_per_example=128,
        val_fraction=0.1,
        worker=0,
        num_workers=1,
        vocab_size=tokenizer.get_vocab_size(),
    )

    # ---- tiny model
    config = Qwen3Config(
        vocab_size=512, hidden_size=128, intermediate_size=256,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=32, max_position_embeddings=128,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = Qwen3ForCausalLM(config).to(device)

    # Save a fake pretrain checkpoint
    pretrain_ckpt = os.path.join(ckpt_dir, "pretrain.pt")
    torch.save({"model_state": model.state_dict(), "config": vars(config)}, pretrain_ckpt)

    # Test LoRA injection
    n_lora = inject_lora(model, rank=4, alpha=8.0)
    print(f"LoRA adapters injected: {n_lora}")
    n_train = lora_parameter_count(model)
    print(f"LoRA trainable params: {n_train:,}")

    # Build dataset (reads packed files written above)
    ds = SFTDataset(
        cache_dir=cache_dir,
        seq_len=64,
        split="train",
    )
    print(f"Dataset windows: {len(ds)}")

    optimizer = build_optimizer(model, lr=1e-4, weight_decay=0.01)
    ctx = (torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
           if device.type == "cuda" else nullcontext())

    model.train()
    for step in range(3):
        x, y, m = ds.get_batch(2, device)
        optimizer.zero_grad(set_to_none=True)
        with ctx:
            out    = model(x)
            logits = out["logits"]
        loss = masked_cross_entropy(logits, y, m)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        print(f"  step {step} | loss {loss.item():.4f}")

    # Test checkpoint save/load
    save_checkpoint(ckpt_dir, 3, model, optimizer, config, {}, 99.0, is_lora=True)
    model2 = Qwen3ForCausalLM(config).to(device)
    inject_lora(model2, rank=4, alpha=8.0)
    load_checkpoint(os.path.join(ckpt_dir, "sft_step0000003.pt"),
                    model2, None, device, is_lora=True)

    # Test merge
    merged = merge_lora(model)
    print(f"LoRA merged. Param types: "
          f"{set(type(m).__name__ for m in merged.modules() if isinstance(m, (nn.Linear, LoRALinear)))}")
    assert not any(isinstance(m, LoRALinear) for m in merged.modules()), \
        "merge_lora() left LoRALinear modules in place"

    shutil.rmtree(tmp)
    print("\n=== SFT smoke test passed ===\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="SFT fine-tuning for reasoning.")

    # Mode
    p.add_argument("--merge-lora", action="store_true",
                   help="Merge LoRA weights into base model and save; skip training")

    # Paths
    p.add_argument("--checkpoint", default=None,
                   help="Pretrained checkpoint from train.py (required)")
    p.add_argument("--tokenizer",  default="./tokenizer",
                   help="Tokenizer directory from train_tokenizer.py")
    p.add_argument("--cache-dir",  default="./sft_packed",
                   help="Packed memmap files produced by pack_sft_data.py")
    p.add_argument("--out-dir",    default="./sft_checkpoints")
    p.add_argument("--resume",     default=None,
                   help="SFT checkpoint to resume from")

    # LoRA
    p.add_argument("--lora",       action="store_true",
                   help="Enable LoRA (recommended for 1B+ on a single GPU)")
    p.add_argument("--lora-rank",  type=int,   default=64)
    p.add_argument("--lora-alpha", type=float, default=128.0)

    # Training
    p.add_argument("--seq-len",             type=int,   default=2048)
    p.add_argument("--batch-size",          type=int,   default=4)
    p.add_argument("--grad-accum-steps",    type=int,   default=4)
    p.add_argument("--max-steps",           type=int,   default=10_000)
    p.add_argument("--warmup-steps",        type=int,   default=200)
    p.add_argument("--lr",                  type=float, default=2e-5,
                   help="Peak LR (typically 1e-5 to 5e-5 for SFT)")
    p.add_argument("--min-lr",              type=float, default=2e-6)
    p.add_argument("--weight-decay",        type=float, default=0.01)
    p.add_argument("--grad-clip",           type=float, default=1.0)
    p.add_argument("--dtype",    default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--compile",  action="store_true")
    p.add_argument("--compile-mode", default="default",
                   choices=["default", "reduce-overhead", "max-autotune"])
    p.add_argument("--seed",     type=int, default=42)

    # Checkpointing / logging
    p.add_argument("--ckpt-interval",  type=int, default=1_000)
    p.add_argument("--keep-ckpts",     type=int, default=3)
    p.add_argument("--log-interval",   type=int, default=10)
    p.add_argument("--eval-interval",  type=int, default=200)
    p.add_argument("--eval-steps",     type=int, default=20)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.merge_lora:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for --merge-lora")
        merge_and_save(args)
    elif args.checkpoint is None:
        smoke_test()
    else:
        train(args)
