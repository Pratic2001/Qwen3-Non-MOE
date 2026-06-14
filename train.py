#!/usr/bin/env python3
"""
train.py

Production pretraining loop for the Qwen3-style dense LLM built in model.py.
Reads packed memmap token files from pack_dataset.py (./packed/train.bin,
./packed/val.bin) and trains with:

  - bf16 mixed precision
  - Gradient accumulation  (simulate large batch on few GPUs)
  - Cosine LR schedule with linear warmup
  - AdamW with weight-decay excluded from norms / embeddings
  - Gradient clipping
  - Gradient checkpointing (optional, saves ~30-35% memory)
  - DDP multi-GPU (activated automatically via torchrun)
  - Periodic validation loss
  - Checkpoint save/resume
  - Optional Weights & Biases logging

Single GPU:
    python train.py --model-size 0.6B --data-dir ./packed --out-dir ./checkpoints

Multi-GPU (e.g. 4 GPUs on one node):
    torchrun --nproc_per_node=4 train.py --model-size 0.6B --data-dir ./packed

Resume from checkpoint:
    python train.py --resume ./checkpoints/ckpt_step5000.pt

Full production run (8B model, large effective batch)
    torchrun --nproc_per_node=8 train.py \
        --model-size 8B \
        --data-dir ./packed \
        --seq-len 4096 \
        --batch-size 4 \
        --grad-accum-steps 16 \
        --max-steps 500000 \
        --warmup-steps 5000 \
        --lr 3e-4 \
        --wandb-project my-llm
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
# Distributed helpers
# ---------------------------------------------------------------------------

def setup_distributed():
    """Init DDP if launched with torchrun, else single-process."""
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    """

    def __init__(self, bin_path: str, seq_len: int, batch_size: int,
                 rank: int = 0, world_size: int = 1, dtype=np.uint16):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size

        data = np.memmap(bin_path, dtype=dtype, mode="r")
        # shard across ranks
        shard_size = len(data) // world_size
        start = rank * shard_size
        end = start + shard_size if rank < world_size - 1 else len(data)
        self.data = data[start:end]

        self.n_positions = max(1, len(self.data) - seq_len)
        self._ptr = 0          # sequential pointer for reproducibility
        self._step = 0
        print(f"[DataLoader rank {rank}] {bin_path}: "
              f"{len(self.data):,} tokens, shard [{start}:{end}]")

    def next_batch(self, device: torch.device):
        """Return (x, y) each of shape (batch_size, seq_len)."""
        ix = torch.randint(self.n_positions, (self.batch_size,))
        x = torch.stack([
            torch.from_numpy(self.data[i: i + self.seq_len].astype(np.int64))
            for i in ix
        ])
        y = torch.stack([
            torch.from_numpy(self.data[i + 1: i + 1 + self.seq_len].astype(np.int64))
            for i in ix
        ])
        # pin_memory -> non_blocking for faster H2D transfer
        x = x.pin_memory().to(device, non_blocking=True) if device.type == "cuda" else x.to(device)
        y = y.pin_memory().to(device, non_blocking=True) if device.type == "cuda" else y.to(device)
        self._step += 1
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
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
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
        # exclude 1-D params (norms, bias) and embeddings from weight decay
        if param.ndim < 2 or "norm" in name or "embed" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    print(f"[Optimizer] decay={sum(p.numel() for p in decay_params):,} "
          f"no_decay={sum(p.numel() for p in no_decay_params):,}")
    return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps, fused=True
                             if torch.cuda.is_available() else False)


# ---------------------------------------------------------------------------
# MFU (model FLOPs utilization)
# ---------------------------------------------------------------------------

def estimate_mfu(model: Qwen3ForCausalLM, tokens_per_sec: float) -> float:
    """Estimate MFU vs. the theoretical peak of the current GPU (A100 bf16)."""
    cfg = model.config if not isinstance(model, DDP) else model.module.config
    n_params = count_parameters(model)
    # ~6 * N * tokens forward+backward FLOPs (Chinchilla approximation)
    flops_per_token = 6 * n_params
    flops_per_sec = flops_per_token * tokens_per_sec
    # A100 80GB bf16 peak = 312e12 FLOP/s (adjust for your GPU)
    gpu_peak_flops = 312e12
    return flops_per_sec / gpu_peak_flops


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------

def save_checkpoint(out_dir: str, step: int, model: nn.Module,
                    optimizer: torch.optim.Optimizer, config: Qwen3Config,
                    train_args: dict, best_val_loss: float):
    raw_model = model.module if isinstance(model, DDP) else model
    ckpt = {
        "step": step,
        "model_state": raw_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": vars(config),
        "train_args": train_args,
        "best_val_loss": best_val_loss,
    }
    path = os.path.join(out_dir, f"ckpt_step{step:07d}.pt")
    torch.save(ckpt, path)
    # keep a symlink 'latest.pt' pointing to the newest checkpoint
    latest = os.path.join(out_dir, "latest.pt")
    if os.path.islink(latest):
        os.remove(latest)
    os.symlink(os.path.abspath(path), latest)
    print(f"[Checkpoint] saved {path}")
    return path


def load_checkpoint(path: str, model: nn.Module,
                    optimizer: Optional[torch.optim.Optimizer],
                    device: torch.device):
    ckpt = torch.load(path, map_location=device)
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    step = ckpt.get("step", 0)
    best_val_loss = ckpt.get("best_val_loss", float("inf"))
    print(f"[Checkpoint] resumed from {path} at step {step}")
    return step, best_val_loss


# ---------------------------------------------------------------------------
# Validation pass
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: nn.Module, val_loader: PackedDataLoader, eval_steps: int,
             device: torch.device, ctx) -> float:
    model.eval()
    losses = []
    for _ in range(eval_steps):
        x, y = val_loader.next_batch(device)
        with ctx:
            out = model(x, labels=y)
        losses.append(out["loss"].item())
    model.train()
    val_loss = float(np.mean(losses))
    return val_loss


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
# Main training loop
# ---------------------------------------------------------------------------

def train(args):
    rank, local_rank, world_size, device = setup_distributed()
    master = is_master(rank)

    torch.manual_seed(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ------------------------------------------------------------------ meta
    meta_path = os.path.join(args.data_dir, "meta.json")
    with open(meta_path) as f:
        meta = json.load(f)

    vocab_size = meta["vocab_size"]
    dtype_np = np.uint16 if meta["dtype"] == "uint16" else np.uint32

    # ------------------------------------------------------------------ model
    if args.resume:
        ckpt_raw = torch.load(args.resume, map_location="cpu")
        config = Qwen3Config(**ckpt_raw["config"])
        if master:
            print(f"[Resume] loaded config from checkpoint")
    else:
        config = Qwen3Config.from_target_size(
            args.model_size,
            vocab_size=vocab_size,
            verbose=master,
        )

    model = Qwen3ForCausalLM(config).to(device)
    n_params = count_parameters(model)

    if args.gradient_checkpointing:
        # Enable activation checkpointing on each decoder layer
        for layer in model.model.layers:
            layer.__class__.__call__ = torch.utils.checkpoint.checkpoint_wrapper(
                layer.__class__.__call__
            )

    if master:
        print(f"Model params: {n_params:,} ({n_params / 1e9:.3f}B)")

    # ------------------------------------------------------------------ DDP
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    # ------------------------------------------------------------------ data
    train_loader = PackedDataLoader(
        os.path.join(args.data_dir, "train.bin"),
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        rank=rank,
        world_size=world_size,
        dtype=dtype_np,
    )
    val_loader = PackedDataLoader(
        os.path.join(args.data_dir, "val.bin"),
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        rank=rank,
        world_size=world_size,
        dtype=dtype_np,
    )

    # ------------------------------------------------------------------ optim
    optimizer = build_optimizer(
        model,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # ------------------------------------------------------------------ amp
    use_amp = device.type == "cuda" and args.dtype == "bf16"
    amp_dtype = torch.bfloat16 if use_amp else torch.float32
    scaler = None  # bf16 doesn't need GradScaler (only fp16 does)
    ctx = (
        torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
        if use_amp else nullcontext()
    )

    # ------------------------------------------------------------------ resume
    start_step = 0
    best_val_loss = float("inf")
    if args.resume:
        start_step, best_val_loss = load_checkpoint(args.resume, model, optimizer, device)

    if master:
        os.makedirs(args.out_dir, exist_ok=True)

    # ------------------------------------------------------------------ W&B
    use_wandb = master and args.wandb_project and try_init_wandb(args, config, n_params)

    # ------------------------------------------------------------------ tokens accounting
    tokens_per_step = (
        args.batch_size * args.seq_len * args.grad_accum_steps * world_size
    )
    if master:
        print(f"\nTokens per optimizer step: {tokens_per_step:,}")
        print(f"Effective batch size:       {args.batch_size * args.grad_accum_steps * world_size}")
        print(f"Max steps:                  {args.max_steps}")
        print(f"Total tokens (planned):     {args.max_steps * tokens_per_step:,}\n")

    # ================================================================== LOOP
    model.train()
    optimizer.zero_grad(set_to_none=True)

    t0 = time.perf_counter()
    local_loss_accum = 0.0
    local_grad_norm = 0.0

    for step in range(start_step, args.max_steps):
        # ---- learning rate
        lr = get_lr(step, args.warmup_steps, args.max_steps, args.lr, args.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # ---- gradient accumulation micro-steps
        for micro_step in range(args.grad_accum_steps):
            x, y = train_loader.next_batch(device)

            # in DDP, only sync gradients on the last micro-step
            if world_size > 1:
                sync = micro_step == args.grad_accum_steps - 1
                ctx_ddp = nullcontext() if sync else model.no_sync()
            else:
                ctx_ddp = nullcontext()

            with ctx_ddp:
                with ctx:
                    out = model(x, labels=y)
                loss = out["loss"] / args.grad_accum_steps
                loss.backward()

            local_loss_accum += loss.item()

        # ---- gradient clipping
        raw_model = model.module if isinstance(model, DDP) else model
        local_grad_norm = nn.utils.clip_grad_norm_(
            model.parameters(), args.grad_clip
        ).item()

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # ---- logging
        if master and (step % args.log_interval == 0 or step == start_step):
            t1 = time.perf_counter()
            dt = t1 - t0
            tokens_per_sec = tokens_per_step * args.log_interval / max(dt, 1e-9)
            mfu = estimate_mfu(raw_model, tokens_per_sec)

            loss_display = local_loss_accum * args.grad_accum_steps  # un-scale
            print(
                f"step {step:7d} | loss {loss_display:.4f} | lr {lr:.2e} | "
                f"grad_norm {local_grad_norm:.3f} | "
                f"{tokens_per_sec / 1e3:.1f}k tok/s | mfu {mfu * 100:.2f}%"
            )

            if use_wandb:
                log_wandb({
                    "train/loss": loss_display,
                    "train/lr": lr,
                    "train/grad_norm": local_grad_norm,
                    "perf/tokens_per_sec": tokens_per_sec,
                    "perf/mfu_pct": mfu * 100,
                }, step=step)

            local_loss_accum = 0.0
            t0 = t1

        # ---- validation
        if step % args.eval_interval == 0 and step > start_step:
            val_loss = evaluate(
                model, val_loader, args.eval_steps, device, ctx
            )
            if world_size > 1:
                vl = torch.tensor(val_loss, device=device)
                dist.all_reduce(vl, op=dist.ReduceOp.AVG)
                val_loss = vl.item()

            if master:
                print(f"  [eval] step {step} | val_loss {val_loss:.4f}")
                if use_wandb:
                    log_wandb({"val/loss": val_loss}, step=step)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss

        # ---- checkpoint
        if master and step % args.ckpt_interval == 0 and step > start_step:
            save_checkpoint(
                args.out_dir, step, model, optimizer, config,
                vars(args), best_val_loss,
            )
            # keep only the last N checkpoints to save disk
            _prune_checkpoints(args.out_dir, keep=args.keep_ckpts)

    # ---- final checkpoint
    if master:
        save_checkpoint(
            args.out_dir, args.max_steps, model, optimizer, config,
            vars(args), best_val_loss,
        )
        print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")

    destroy_distributed()


# ---------------------------------------------------------------------------
# Checkpoint pruning
# ---------------------------------------------------------------------------

def _prune_checkpoints(out_dir: str, keep: int = 3):
    ckpts = sorted(Path(out_dir).glob("ckpt_step*.pt"),
                   key=lambda p: int(p.stem.replace("ckpt_step", "")))
    for old in ckpts[:-keep]:
        old.unlink()
        print(f"[Checkpoint] pruned {old}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Pretrain a Qwen3-style dense LLM.")

    # model
    p.add_argument("--model-size", default="0.6B",
                   help="Target model size passed to Qwen3Config.from_target_size (e.g. 0.6B, 1.7B, 8B)")
    p.add_argument("--vocab-size", type=int, default=None,
                   help="Override vocab size (default: read from packed/meta.json)")

    # data
    p.add_argument("--data-dir", default="./packed",
                   help="Directory with train.bin, val.bin, meta.json from pack_dataset.py")
    p.add_argument("--seq-len", type=int, default=2048,
                   help="Sequence length (tokens per example)")

    # training
    p.add_argument("--batch-size", type=int, default=8,
                   help="Micro-batch size per GPU per grad-accum step")
    p.add_argument("--grad-accum-steps", type=int, default=4,
                   help="Gradient accumulation steps (effective_batch = batch * accum * world_size)")
    p.add_argument("--max-steps", type=int, default=100_000)
    p.add_argument("--warmup-steps", type=int, default=2_000)
    p.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate")
    p.add_argument("--min-lr", type=float, default=3e-5,
                   help="Minimum LR at end of cosine decay (default: lr/10)")
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--gradient-checkpointing", action="store_true",
                   help="Trade compute for memory by recomputing activations in backward")
    p.add_argument("--seed", type=int, default=42)

    # checkpointing
    p.add_argument("--out-dir", default="./checkpoints")
    p.add_argument("--resume", default=None,
                   help="Path to a checkpoint .pt file to resume from")
    p.add_argument("--ckpt-interval", type=int, default=5_000,
                   help="Save a checkpoint every N steps")
    p.add_argument("--keep-ckpts", type=int, default=3,
                   help="Number of recent checkpoints to keep on disk")

    # logging / eval
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--eval-steps", type=int, default=50,
                   help="Number of val batches per eval pass")
    p.add_argument("--wandb-project", default=None,
                   help="W&B project name. Omit to disable W&B logging.")
    p.add_argument("--wandb-run-name", default=None)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Smoke-test (runs automatically when no packed data is found)
# ---------------------------------------------------------------------------

def smoke_test():
    """
    Runs a 5-step training loop on synthetic data so the whole stack
    (model, dataloader, optimizer, scaler, checkpointing) can be validated
    without a real dataset.
    """
    print("\n=== smoke test (no real data found) ===")
    import tempfile, shutil

    tmp = tempfile.mkdtemp()
    packed = os.path.join(tmp, "packed")
    os.makedirs(packed)
    ckpt_dir = os.path.join(tmp, "ckpts")

    # synthetic packed data
    vocab_size = 1024
    n_tokens = 50_000
    arr = np.random.randint(0, vocab_size, n_tokens, dtype=np.uint16)
    arr.tofile(os.path.join(packed, "train.bin"))
    arr[:5000].tofile(os.path.join(packed, "val.bin"))
    meta = {
        "vocab_size": vocab_size,
        "dtype": "uint16",
        "train_tokens": n_tokens,
        "val_tokens": 5000,
        "total_tokens": n_tokens,
        "category_token_counts": {},
    }
    with open(os.path.join(packed, "meta.json"), "w") as f:
        json.dump(meta, f)

    # tiny model config
    config = Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=256,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Qwen3ForCausalLM(config).to(device)
    print(f"Smoke-test model: {count_parameters(model):,} params on {device}")

    optimizer = build_optimizer(model, lr=3e-4, weight_decay=0.1)
    use_amp = device.type == "cuda"
    ctx = (torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
           if use_amp else nullcontext())

    loader = PackedDataLoader(
        os.path.join(packed, "train.bin"),
        seq_len=64,
        batch_size=2,
        dtype=np.uint16,
    )
    val_loader = PackedDataLoader(
        os.path.join(packed, "val.bin"),
        seq_len=64,
        batch_size=2,
        dtype=np.uint16,
    )

    model.train()
    os.makedirs(ckpt_dir, exist_ok=True)
    t0 = time.perf_counter()

    for step in range(5):
        lr = get_lr(step, warmup_steps=2, max_steps=5, max_lr=3e-4, min_lr=3e-5)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        x, y = loader.next_batch(device)
        optimizer.zero_grad(set_to_none=True)
        with ctx:
            out = model(x, labels=y)
        out["loss"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        print(f"  step {step} | loss {out['loss'].item():.4f} | lr {lr:.2e}")

    # validation
    val_loss = evaluate(model, val_loader, eval_steps=3, device=device, ctx=ctx)
    print(f"  val_loss: {val_loss:.4f}")

    # checkpoint
    save_checkpoint(ckpt_dir, 5, model, optimizer, config,
                    train_args={}, best_val_loss=val_loss)

    # reload checkpoint and verify
    model2 = Qwen3ForCausalLM(config).to(device)
    optimizer2 = build_optimizer(model2, lr=3e-4, weight_decay=0.1)
    step_resumed, _ = load_checkpoint(
        os.path.join(ckpt_dir, "ckpt_step0000005.pt"),
        model2, optimizer2, device,
    )
    assert step_resumed == 5, "checkpoint step mismatch"

    dt = time.perf_counter() - t0
    shutil.rmtree(tmp)
    print(f"\n=== smoke test passed in {dt:.2f}s ===\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    # if no real data present, run smoke test and exit
    train_bin = os.path.join(args.data_dir, "train.bin")
    if not os.path.exists(train_bin):
        smoke_test()
    else:
        train(args)
