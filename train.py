#!/usr/bin/env python3
"""
train.py

Production pretraining loop for the Qwen3-style dense LLM built in model.py.
Reads packed memmap token files from pack_dataset.py (./packed/train.bin,
./packed/val.bin) and trains with:

  - bf16 mixed precision
  - torch.compile for kernel fusion (+25-40% throughput)
  - Gradient accumulation  (simulate large batch on few GPUs)
  - Cosine LR schedule with linear warmup
  - AdamW with weight-decay excluded from norms / embeddings
  - Gradient clipping
  - Gradient checkpointing (optional, saves ~30-35% memory)
  - DDP multi-GPU (activated automatically via torchrun)
  - Periodic validation loss
  - Checkpoint save/resume
  - Optional Weights & Biases logging
  - Accurate per-GPU MFU reporting

Single GPU:
    python train.py --model-size 0.6B --data-dir ./packed --out-dir ./checkpoints

Multi-GPU (e.g. 4 GPUs on one node):
    torchrun --nproc_per_node=4 train.py --model-size 0.6B --data-dir ./packed

Resume from checkpoint:
    python train.py --resume ./checkpoints/ckpt_step5000.pt

Recommended command for RTX 4090 (0.3B model):
    python train.py --model-size 0.3B --data-dir ./packed \\
        --seq-len 2048 --batch-size 32 --grad-accum-steps 4 \\
        --compile --out-dir ./checkpoints
"""

import argparse
import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from model import Qwen3Config, Qwen3ForCausalLM, count_parameters


# ---------------------------------------------------------------------------
# GPU peak FLOP/s table  (bf16 Tensor Core, per card)
# ---------------------------------------------------------------------------
# Used for MFU estimation. Add your GPU here if missing.
GPU_PEAK_TFLOPS = {
    # NVIDIA consumer
    "NVIDIA GeForce RTX 4090":    165.2,
    "NVIDIA GeForce RTX 4080":    97.5,
    "NVIDIA GeForce RTX 4070 Ti": 80.8,
    "NVIDIA GeForce RTX 4070":    59.8,
    "NVIDIA GeForce RTX 3090":    71.0,
    "NVIDIA GeForce RTX 3080":    59.4,
    # NVIDIA data-center
    "NVIDIA A100-SXM4-80GB":      312.0,
    "NVIDIA A100-SXM4-40GB":      312.0,
    "NVIDIA A100-PCIE-40GB":      312.0,
    "NVIDIA H100 SXM5":           989.5,
    "NVIDIA H100 PCIe":           756.0,
    "NVIDIA L40S":                362.1,
    "NVIDIA L4":                  121.0,
    "NVIDIA A10G":                125.0,
    "NVIDIA V100-SXM2-16GB":      28.0,   # fp16 only; no bf16 HW
    # AMD
    "AMD Instinct MI300X":        1307.4,
    "AMD Instinct MI250X":        383.0,
}

_FALLBACK_TFLOPS = 100.0  # conservative fallback if GPU not in table


def get_gpu_peak_tflops(device: torch.device) -> float:
    if device.type != "cuda":
        return _FALLBACK_TFLOPS
    name = torch.cuda.get_device_name(device)
    for key, val in GPU_PEAK_TFLOPS.items():
        if key.lower() in name.lower() or name.lower() in key.lower():
            return val
    print(f"[MFU] Unknown GPU '{name}', using {_FALLBACK_TFLOPS} TFLOP/s fallback. "
          f"Add it to GPU_PEAK_TFLOPS for accurate MFU.")
    return _FALLBACK_TFLOPS


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def setup_distributed():
    """Init DDP if launched with torchrun, else single-process."""
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank       = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank       = 0
        local_rank = 0
        world_size = 1
        device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return rank, local_rank, world_size, device


def is_master(rank: int) -> bool:
    return rank == 0


def destroy_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------

class PackedDataLoader:
    """
    Streams random fixed-length windows from a flat memmap token file.
    Each rank gets a non-overlapping shard of the data so no two GPUs
    see the same tokens in the same step.

    Prefetch: the next batch is built on CPU while the GPU runs the
    current forward/backward, hiding the CPU cost completely.
    """

    def __init__(self, bin_path: str, seq_len: int, batch_size: int,
                 rank: int = 0, world_size: int = 1, dtype=np.uint16,
                 prefetch: bool = True):
        self.seq_len    = seq_len
        self.batch_size = batch_size
        self.prefetch   = prefetch

        data = np.memmap(bin_path, dtype=dtype, mode="r")
        shard_size = len(data) // world_size
        start = rank * shard_size
        end   = start + shard_size if rank < world_size - 1 else len(data)
        self.data = data[start:end]

        self.n_positions = max(1, len(self.data) - seq_len)
        self._prefetched: Optional[tuple] = None

        print(f"[DataLoader rank {rank}] {bin_path}: "
              f"{len(self.data):,} tokens, shard [{start}:{end}]")

    def _build_batch_cpu(self):
        """Build a (x, y) batch entirely on CPU as int64 tensors."""
        ix = torch.randint(self.n_positions, (self.batch_size,))
        x = torch.stack([
            torch.from_numpy(self.data[i : i + self.seq_len].astype(np.int64))
            for i in ix
        ])
        y = torch.stack([
            torch.from_numpy(self.data[i + 1 : i + 1 + self.seq_len].astype(np.int64))
            for i in ix
        ])
        return x, y

    def prime(self, device: torch.device):
        """Prefetch the very first batch before the loop starts."""
        if self.prefetch:
            self._prefetched = self._build_batch_cpu()

    def next_batch(self, device: torch.device):
        """
        Return (x, y) on `device`.
        If prefetch is on, the CPU batch for the *next* call is prepared
        while the caller is doing GPU work with the current one.
        """
        if self.prefetch and self._prefetched is not None:
            x_cpu, y_cpu = self._prefetched
            # Kick off the prefetch for the next call *before* H2D transfer
            self._prefetched = self._build_batch_cpu()
        else:
            x_cpu, y_cpu = self._build_batch_cpu()
            if self.prefetch:
                self._prefetched = self._build_batch_cpu()

        if device.type == "cuda":
            x = x_cpu.pin_memory().to(device, non_blocking=True)
            y = y_cpu.pin_memory().to(device, non_blocking=True)
        else:
            x = x_cpu.to(device)
            y = y_cpu.to(device)
        return x, y


# ---------------------------------------------------------------------------
# LR schedule: linear warmup → cosine decay → min_lr floor
# ---------------------------------------------------------------------------

def get_lr(step: int, warmup_steps: int, max_steps: int,
           max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff    = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)


# ---------------------------------------------------------------------------
# Optimizer: AdamW with separate param groups for weight decay
# ---------------------------------------------------------------------------

def build_optimizer(model: nn.Module, lr: float, weight_decay: float,
                    betas=(0.9, 0.95), eps=1e-8) -> torch.optim.AdamW:
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "norm" in name or "embed" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    groups = [
        {"params": decay_params,    "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    print(f"[Optimizer] decay={sum(p.numel() for p in decay_params):,}  "
          f"no_decay={sum(p.numel() for p in no_decay_params):,}")
    return torch.optim.AdamW(
        groups, lr=lr, betas=betas, eps=eps,
        fused=torch.cuda.is_available(),   # fused kernel: ~15% faster on CUDA
    )


# ---------------------------------------------------------------------------
# MFU (model FLOPs utilization) — calibrated per GPU
# ---------------------------------------------------------------------------

def estimate_mfu(model: nn.Module, tokens_per_sec: float,
                 gpu_peak_tflops: float) -> float:
    """
    Estimate what fraction of the GPU's theoretical bf16 peak we are using.

    Formula: each token costs ~6N FLOPs for forward + backward
    (Chinchilla / PaLM approximation, N = non-embedding params).
    Attention FLOPs (quadratic in seq_len) are omitted here because they
    are a small fraction for typical seq_len << hidden_size*n_layers.
    """
    raw = model.module if isinstance(model, DDP) else model
    # exclude embedding table — not a matmul, much lower arithmetic intensity
    n_params = sum(p.numel() for name, p in raw.named_parameters()
                   if "embed_tokens" not in name)
    flops_per_sec = 6 * n_params * tokens_per_sec
    return flops_per_sec / (gpu_peak_tflops * 1e12)


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------

def save_checkpoint(out_dir: str, step: int, model: nn.Module,
                    optimizer: torch.optim.Optimizer, config: Qwen3Config,
                    train_args: dict, best_val_loss: float):
    raw_model = model.module if isinstance(model, DDP) else model
    # unwrap compiled model if present
    state_model = (raw_model._orig_mod if hasattr(raw_model, "_orig_mod")
                   else raw_model)
    ckpt = {
        "step":            step,
        "model_state":     state_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config":          vars(config),
        "train_args":      train_args,
        "best_val_loss":   best_val_loss,
    }
    path   = os.path.join(out_dir, f"ckpt_step{step:07d}.pt")
    torch.save(ckpt, path)
    latest = os.path.join(out_dir, "latest.pt")
    if os.path.islink(latest):
        os.remove(latest)
    os.symlink(os.path.abspath(path), latest)
    print(f"[Checkpoint] saved {path}")
    return path


def load_checkpoint(path: str, model: nn.Module,
                    optimizer: Optional[torch.optim.Optimizer],
                    device: torch.device):
    ckpt      = torch.load(path, map_location=device)
    raw_model = model.module if isinstance(model, DDP) else model
    state_model = (raw_model._orig_mod if hasattr(raw_model, "_orig_mod")
                   else raw_model)
    state_model.load_state_dict(ckpt["model_state"])
    # Re-tie lm_head -> embed_tokens after state_dict restore, because
    # load_state_dict replaces the weight tensor object so the tie is broken.
    if hasattr(state_model, "tie_weights"):
        state_model.tie_weights()
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    step           = ckpt.get("step", 0)
    best_val_loss  = ckpt.get("best_val_loss", float("inf"))
    print(f"[Checkpoint] resumed from {path} at step {step}")
    return step, best_val_loss


# ---------------------------------------------------------------------------
# Validation pass
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: nn.Module, val_loader: PackedDataLoader,
             eval_steps: int, device: torch.device, ctx,
             use_cudagraphs: bool = False) -> float:
    model.eval()
    losses = []
    for _ in range(eval_steps):
        x, y = val_loader.next_batch(device)
        with ctx:
            if use_cudagraphs:
                torch.compiler.cudagraph_mark_step_begin()
            out = model(x, labels=y)
        losses.append(out["loss"].item())
    model.train()
    return float(np.mean(losses))


# ---------------------------------------------------------------------------
# Optional W&B
# ---------------------------------------------------------------------------

def try_init_wandb(args, config: Qwen3Config, n_params: int):
    try:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"qwen3-{args.model_size}",
            config={**vars(config), "n_params": n_params, **vars(args)},
        )
        return True
    except Exception as e:
        print(f"[W&B] disabled: {e}")
        return False


def log_wandb(metrics: dict, step: int):
    try:
        import wandb
        wandb.log(metrics, step=step)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Checkpoint pruning
# ---------------------------------------------------------------------------

def _prune_checkpoints(out_dir: str, keep: int = 3):
    ckpts = sorted(
        Path(out_dir).glob("ckpt_step*.pt"),
        key=lambda p: int(p.stem.replace("ckpt_step", "")),
    )
    for old in ckpts[:-keep]:
        old.unlink()
        print(f"[Checkpoint] pruned {old}")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args):
    rank, local_rank, world_size, device = setup_distributed()
    master = is_master(rank)

    torch.manual_seed(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32        = True

    # ------------------------------------------------------------------ meta
    meta_path = os.path.join(args.data_dir, "meta.json")
    with open(meta_path) as f:
        meta = json.load(f)

    vocab_size = meta["vocab_size"]
    dtype_np   = np.uint16 if meta["dtype"] == "uint16" else np.uint32

    # ------------------------------------------------------------------ model
    if args.resume:
        ckpt_raw = torch.load(args.resume, map_location="cpu")
        config   = Qwen3Config(**ckpt_raw["config"])
        if master:
            print(f"[Resume] loaded config from checkpoint")
    else:
        config = Qwen3Config.from_target_size(
            args.model_size,
            vocab_size=vocab_size,
            quality_mode=args.quality_mode,
            param_slack=args.param_slack,
            verbose=master,
        )

    model    = Qwen3ForCausalLM(config).to(device)
    n_params = count_parameters(model)

    if master:
        print(f"Model params : {n_params:,}  ({n_params / 1e9:.3f}B)")
        if device.type == "cuda":
            vram_total  = torch.cuda.get_device_properties(device).total_memory
            vram_gb     = vram_total / 1024**3
            # Static VRAM: weights(bf16) + grads(bf16) + Adam states(fp32 m+v)
            static_gb   = n_params * (2 + 2 + 8) / 1024**3
            # Activation VRAM per token (rough): ~60 bytes for hidden+FFN+attn
            # per layer in bf16, with flash attention (O(seq) not O(seq^2))
            cfg = config
            bytes_per_token_per_layer = (
                cfg.hidden_size * 2               # hidden states (bf16)
                + cfg.intermediate_size * 2 * 2   # gate + up proj (bf16)
                + cfg.num_attention_heads * cfg.head_dim * 2  # attn output (bf16)
            )
            act_gb_per_step = (
                args.batch_size * args.seq_len
                * cfg.num_hidden_layers
                * bytes_per_token_per_layer
                / 1024**3
            )
            total_est   = static_gb + act_gb_per_step
            headroom_gb = vram_gb - total_est

            print(f"GPU          : {torch.cuda.get_device_name(device)}")
            print(f"VRAM         : {vram_gb:.1f} GB total")
            print(f"  static     : ~{static_gb:.1f} GB  (weights + grads + Adam)")
            print(f"  activations: ~{act_gb_per_step:.1f} GB  "
                  f"(batch={args.batch_size}, seq={args.seq_len})")
            print(f"  headroom   : ~{headroom_gb:.1f} GB")

            if headroom_gb < 1.5:
                # Suggest a safe batch size that leaves 2 GB headroom
                safe_batch = max(1, int(
                    (vram_gb - static_gb - 2.0) * 1024**3
                    / (args.seq_len * cfg.num_hidden_layers * bytes_per_token_per_layer)
                ))
                print(f"\n  ⚠  WARNING: estimated VRAM is tight (<1.5 GB headroom).")
                print(f"     Likely OOM at batch={args.batch_size}, seq={args.seq_len}.")
                print(f"     Suggestions:")
                print(f"       --batch-size {safe_batch}  (estimated safe)")
                print(f"       --gradient-checkpointing   (cuts activation VRAM ~35%)")
                print(f"       --seq-len {args.seq_len // 2}  (halves activation VRAM)\n")

    # ---- gradient checkpointing (trades ~30% compute for ~35% VRAM reduction)
    if args.gradient_checkpointing:
        # Call the real method on our custom model, not the HF stub
        model.model.enable_gradient_checkpointing()

    # ---- torch.compile  (biggest single performance lever on modern PyTorch)
    if args.compile:
        if master:
            print(f"[compile] torch.compile(mode='{args.compile_mode}')…")
            if args.compile_mode == "reduce-overhead":
                print("          Using CUDAGraphs (reduce-overhead). If you see tensor")
                print("          overwrite errors, switch to --compile-mode default.")
            print("          First step will be slow (~60-120s). Subsequent steps are fast.")
        model = torch.compile(model, mode=args.compile_mode)

    # Whether to call cudagraph_mark_step_begin() before every forward.
    # Required when using reduce-overhead (CUDAGraphs) to prevent the
    # "tensor overwritten by subsequent run" error on tied-weight models.
    _use_cudagraphs = args.compile and args.compile_mode == "reduce-overhead"

    # ------------------------------------------------------------------ DDP
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    # ------------------------------------------------------------------ data
    train_loader = PackedDataLoader(
        os.path.join(args.data_dir, "train.bin"),
        seq_len=args.seq_len, batch_size=args.batch_size,
        rank=rank, world_size=world_size, dtype=dtype_np,
    )
    val_loader = PackedDataLoader(
        os.path.join(args.data_dir, "val.bin"),
        seq_len=args.seq_len, batch_size=args.batch_size,
        rank=rank, world_size=world_size, dtype=dtype_np,
    )
    train_loader.prime(device)   # fill prefetch buffer
    val_loader.prime(device)

    # ------------------------------------------------------------------ optim
    optimizer = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)

    # ------------------------------------------------------------------ amp
    use_amp   = device.type == "cuda" and args.dtype == "bf16"
    amp_dtype = torch.bfloat16 if use_amp else torch.float32
    ctx       = (torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
                 if use_amp else nullcontext())

    # ------------------------------------------------------------------ resume
    start_step    = 0
    best_val_loss = float("inf")
    if args.resume:
        start_step, best_val_loss = load_checkpoint(
            args.resume, model, optimizer, device
        )
        # Re-tie weights after loading state_dict — loading replaces the
        # lm_head weight tensor so we need to re-point it at embed_tokens.
        raw = model.module if isinstance(model, DDP) else model
        raw_model = getattr(raw, "_orig_mod", raw)
        if hasattr(raw_model, "tie_weights"):
            raw_model.tie_weights()

    # ------------------------------------------------------------------ MFU
    gpu_peak_tflops = get_gpu_peak_tflops(device)
    if master:
        print(f"GPU peak bf16: {gpu_peak_tflops:.1f} TFLOP/s")

    # ------------------------------------------------------------------ W&B
    if master:
        os.makedirs(args.out_dir, exist_ok=True)
    use_wandb = (master and args.wandb_project
                 and try_init_wandb(args, config, n_params))

    # ------------------------------------------------------------------ tokens accounting
    tokens_per_step = (
        args.batch_size * args.seq_len * args.grad_accum_steps * world_size
    )
    if master:
        print(f"\nTokens / optimizer step : {tokens_per_step:,}")
        print(f"Effective batch size    : {args.batch_size * args.grad_accum_steps * world_size}")
        print(f"Max steps               : {args.max_steps:,}")
        print(f"Checkpoint every        : {args.ckpt_interval:,} steps")
        print(f"Total tokens (planned)  : {args.max_steps * tokens_per_step:,}\n")

    # ================================================================== LOOP
    model.train()
    optimizer.zero_grad(set_to_none=True)

    t0              = time.perf_counter()
    loss_accum      = 0.0
    grad_norm_accum = 0.0

    for step in range(start_step, args.max_steps):

        # ---- LR
        lr = get_lr(step, args.warmup_steps, args.max_steps, args.lr, args.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # ---- gradient accumulation micro-steps
        for micro_step in range(args.grad_accum_steps):
            x, y = train_loader.next_batch(device)

            sync = (micro_step == args.grad_accum_steps - 1)
            ctx_ddp = nullcontext() if (world_size == 1 or sync) else model.no_sync()

            with ctx_ddp:
                with ctx:
                    # CUDAGraphs (reduce-overhead mode) requires this marker
                    # before every forward pass so it knows a new "step" has
                    # begun and won't confuse output buffers across calls.
                    if _use_cudagraphs:
                        torch.compiler.cudagraph_mark_step_begin()
                    out  = model(x, labels=y)
                loss = out["loss"] / args.grad_accum_steps
                loss.backward()

            loss_accum += loss.item()

        # ---- gradient clipping
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item()
        grad_norm_accum += grad_norm

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # flush CUDA so timing is accurate
        if device.type == "cuda":
            torch.cuda.synchronize()

        # ---- logging
        if master and step % args.log_interval == 0:
            t1  = time.perf_counter()
            dt  = max(t1 - t0, 1e-9)
            tok_per_sec = tokens_per_step * args.log_interval / dt
            mfu         = estimate_mfu(model, tok_per_sec, gpu_peak_tflops)

            # loss_accum already holds the sum, over log_interval optimizer
            # steps, of each step's true average loss (the per-micro-step
            # division by grad_accum_steps is canceled out by summing
            # grad_accum_steps of them inside the inner loop). Only the
            # log_interval averaging is needed here -- multiplying by
            # grad_accum_steps again was double-counting that scaling and
            # inflated the displayed loss by a factor of grad_accum_steps.
            loss_display      = loss_accum / args.log_interval
            grad_norm_display = grad_norm_accum / args.log_interval
            loss_accum        = 0.0
            grad_norm_accum   = 0.0

            print(
                f"step {step:7d} | loss {loss_display:.4f} | lr {lr:.2e} | "
                f"grad {grad_norm_display:.3f} | "
                f"{tok_per_sec / 1e3:.1f}k tok/s | "
                f"mfu {mfu * 100:.2f}%"
            )

            if use_wandb:
                log_wandb({
                    "train/loss":         loss_display,
                    "train/lr":           lr,
                    "train/grad_norm":    grad_norm_display,
                    "perf/tokens_per_sec": tok_per_sec,
                    "perf/mfu_pct":       mfu * 100,
                }, step=step)

            t0 = t1

        # ---- validation
        if step % args.eval_interval == 0 and step > start_step:
            val_loss = evaluate(model, val_loader, args.eval_steps, device, ctx,
                                use_cudagraphs=_use_cudagraphs)
            if world_size > 1:
                vl = torch.tensor(val_loss, device=device)
                dist.all_reduce(vl, op=dist.ReduceOp.AVG)
                val_loss = vl.item()

            if master:
                print(f"  [eval] step {step:7d} | val_loss {val_loss:.4f}")
                if use_wandb:
                    log_wandb({"val/loss": val_loss}, step=step)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss

        # ---- checkpoint
        if master and step % args.ckpt_interval == 0 and step > start_step:
            save_checkpoint(
                args.out_dir, step, model, optimizer,
                config, vars(args), best_val_loss,
            )
            _prune_checkpoints(args.out_dir, keep=args.keep_ckpts)

    # ---- final checkpoint
    if master:
        save_checkpoint(
            args.out_dir, args.max_steps, model, optimizer,
            config, vars(args), best_val_loss,
        )
        print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")

    destroy_distributed()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Pretrain a Qwen3-style dense LLM.")

    # model
    p.add_argument("--model-size", default="0.6B")
    p.add_argument("--vocab-size", type=int, default=None)
    p.add_argument("--quality-mode", default="shape", choices=["shape", "exact"],
                   help=(
                       "How from_target_size picks the architecture. "
                       "'shape' (default) — pick the best-shape config within "
                       "±param-slack of the target (recommended for training quality). "
                       "'exact' — pick the config whose param count is closest to the "
                       "target (legacy behavior; can give bad shape on small models)."
                   ))
    p.add_argument("--param-slack", type=float, default=0.10,
                   help=(
                       "In quality-mode 'shape', allowed param-count deviation from "
                       "target as a fraction (default 0.10 = ±10%%). Higher values "
                       "expand the search space."
                   ))

    # data
    p.add_argument("--data-dir", default="./packed")
    p.add_argument("--seq-len",  type=int, default=2048)

    # training
    p.add_argument("--batch-size",       type=int,   default=8)
    p.add_argument("--grad-accum-steps", type=int,   default=4)
    p.add_argument("--max-steps",        type=int,   default=100_000)
    p.add_argument("--warmup-steps",     type=int,   default=2_000)
    p.add_argument("--lr",               type=float, default=3e-4)
    p.add_argument("--min-lr",           type=float, default=3e-5)
    p.add_argument("--weight-decay",     type=float, default=0.1)
    p.add_argument("--grad-clip",        type=float, default=1.0)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--compile", action="store_true",
                   help="Run torch.compile for kernel fusion (+25-40%% throughput)")
    p.add_argument("--compile-mode", default="default",
                   choices=["default", "reduce-overhead", "max-autotune"],
                   help=(
                       "torch.compile mode. "
                       "'default' — safe, good speedup, no CUDAGraphs. "
                       "'reduce-overhead' — uses CUDAGraphs for lower kernel-launch overhead "
                       "(marginally faster for small batches, needs cudagraph_mark_step_begin). "
                       "'max-autotune' — exhaustive kernel search, very slow to compile."
                   ))
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    # checkpointing
    p.add_argument("--out-dir",       default="./checkpoints")
    p.add_argument("--resume",        default=None)
    p.add_argument("--ckpt-interval", type=int, default=5_000,
                   help="Save checkpoint every N steps (default 5000)")
    p.add_argument("--keep-ckpts",    type=int, default=3)

    # logging / eval
    p.add_argument("--log-interval",  type=int, default=10)
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--eval-steps",    type=int, default=50)
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--wandb-run-name",default=None)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def smoke_test():
    print("\n=== smoke test (no real data found) ===")
    import tempfile, shutil

    tmp    = tempfile.mkdtemp()
    packed = os.path.join(tmp, "packed")
    os.makedirs(packed)
    ckpt_dir = os.path.join(tmp, "ckpts")

    vocab_size = 1024
    n_tokens   = 50_000
    arr = np.random.randint(0, vocab_size, n_tokens, dtype=np.uint16)
    arr.tofile(os.path.join(packed, "train.bin"))
    arr[:5000].tofile(os.path.join(packed, "val.bin"))
    with open(os.path.join(packed, "meta.json"), "w") as f:
        json.dump({"vocab_size": vocab_size, "dtype": "uint16",
                   "train_tokens": n_tokens, "val_tokens": 5000,
                   "total_tokens": n_tokens, "category_token_counts": {}}, f)

    config = Qwen3Config(
        vocab_size=vocab_size, hidden_size=256, intermediate_size=512,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=64, max_position_embeddings=256,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = Qwen3ForCausalLM(config).to(device)
    print(f"Smoke-test model: {count_parameters(model):,} params on {device}")

    optimizer = build_optimizer(model, lr=3e-4, weight_decay=0.1)
    ctx = (torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
           if device.type == "cuda" else nullcontext())

    loader     = PackedDataLoader(os.path.join(packed, "train.bin"), 64, 2, dtype=np.uint16)
    val_loader = PackedDataLoader(os.path.join(packed, "val.bin"),   64, 2, dtype=np.uint16)
    loader.prime(device)
    val_loader.prime(device)

    gpu_peak = get_gpu_peak_tflops(device)
    model.train()
    os.makedirs(ckpt_dir, exist_ok=True)
    t0 = time.perf_counter()

    for step in range(5):
        lr = get_lr(step, 2, 5, 3e-4, 3e-5)
        for pg in optimizer.param_groups: pg["lr"] = lr
        x, y = loader.next_batch(device)
        optimizer.zero_grad(set_to_none=True)
        with ctx:
            out = model(x, labels=y)
        out["loss"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        mfu = estimate_mfu(model, 2 * 64 / max(time.perf_counter() - t0, 1e-6), gpu_peak)
        print(f"  step {step} | loss {out['loss'].item():.4f} | "
              f"lr {lr:.2e} | mfu {mfu*100:.2f}%")

    val_loss = evaluate(model, val_loader, 3, device, ctx)
    print(f"  val_loss: {val_loss:.4f}")

    save_checkpoint(ckpt_dir, 5, model, optimizer, config, {}, val_loss)

    model2     = Qwen3ForCausalLM(config).to(device)
    optimizer2 = build_optimizer(model2, lr=3e-4, weight_decay=0.1)
    step_r, _  = load_checkpoint(
        os.path.join(ckpt_dir, "ckpt_step0000005.pt"), model2, optimizer2, device
    )
    assert step_r == 5

    shutil.rmtree(tmp)
    print(f"\n=== smoke test passed in {time.perf_counter()-t0:.2f}s ===\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(os.path.join(args.data_dir, "train.bin")):
        smoke_test()
    else:
        train(args)
