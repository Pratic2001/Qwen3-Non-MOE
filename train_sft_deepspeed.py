#!/usr/bin/env python3
"""
train_sft_deepspeed.py

DeepSpeed-powered supervised fine-tuning for the Qwen3-style dense LLM
from model.py — Stage 1 of the reasoning post-training pipeline.

This is the DeepSpeed twin of train_sft.py: every architectural choice
that train_deepspeed.py makes for pretraining is mirrored here for SFT:

  - full hardware audit (VRAM, NVLink, InfiniBand, CPU RAM) before any
    training logic runs
  - automatic selection of ZeRO stage and CPU offload configuration
  - generated ds_config.json is written to --out-dir and printed
  - MFU estimation via the same 4-tier GPU peak-FLOP/s resolver
  - DeepSpeed engine is the optimizer / scheduler / gradient owner
  - checkpoints are written in DeepSpeed's native directory format so
    ZeRO-3 sharded weights round-trip correctly

SFT-specific behaviour kept identical to train_sft.py:

  - loads a pretrained checkpoint (raw .pt from train.py OR a
    consolidated .pt produced by deepspeed_shard_consolidator.py)
  - reads packed memmap .bin files written by pack_sft_data.py and
    concatenates the worker shards via mmap
  - applies a position-level loss mask: only assistant tokens
    (thinking + answer) contribute to the loss
  - supports LoRA on q/k/v/o/gate/up/down projections for single-GPU
    fine-tuning of large models; LoRA state is checkpointed separately
  - merge-lora mode is available without launching the engine

Launch:
    # Single node, 1 GPU
    deepspeed train_sft_deepspeed.py --checkpoint ./checkpoints/latest.pt \\
        --cache-dir ./sft_packed --out-dir ./sft_checkpoints_ds

    # Single node, 4 GPUs
    deepspeed --num_gpus 4 train_sft_deepspeed.py --checkpoint ... \\
        --cache-dir ./sft_packed --out-dir ./sft_checkpoints_ds

    # Multi-node (2 nodes × 8 GPUs)
    deepspeed --hostfile hostfile.txt train_sft_deepspeed.py \\
        --checkpoint ./checkpoints/latest.pt \\
        --cache-dir ./sft_packed --out-dir ./sft_checkpoints_ds

    # LoRA (recommended for >=1B on a single 4090)
    deepspeed train_sft_deepspeed.py --checkpoint ... --lora \\
        --lora-rank 64 --lora-alpha 128 --out-dir ./sft_checkpoints_ds

    # Force a specific ZeRO stage (skip auto-selection)
    deepspeed train_sft_deepspeed.py --checkpoint ... --zero-stage 3 \\
        --cpu-offload-optimizer --cache-dir ./sft_packed

    # Resume from a DeepSpeed SFT checkpoint
    deepspeed train_sft_deepspeed.py --checkpoint ./checkpoints/latest.pt \\
        --cache-dir ./sft_packed \\
        --resume ./sft_checkpoints_ds/latest_ds --out-dir ./sft_checkpoints_ds

    # Merge LoRA back into base weights for deployment (no DeepSpeed needed)
    python train_sft_deepspeed.py --merge-lora \\
        --checkpoint ./sft_checkpoints_ds/latest.pt \\
        --out-dir ./sft_merged
"""

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist

import deepspeed

from model import Qwen3Config, Qwen3ForCausalLM, count_parameters

# Re-use the SFT data path, LoRA implementation, masked loss, and tokenizer
# loader from train_sft.py — that file is the authoritative implementation;
# this script only swaps the DDP + checkpoint + optimizer layer for DeepSpeed.
# Importing it does NOT modify train_sft.py.
from train_sft import (
    LoRALinear,
    SFTDataset,
    inject_lora,
    lora_parameter_count,
    lora_state_dict,
    masked_cross_entropy,
    merge_lora,
)


# ---------------------------------------------------------------------------
# GPU FLOP/s table  (bf16 Tensor Core, per card)
# ---------------------------------------------------------------------------
# Mirrors train_deepspeed.py exactly so MFU numbers are comparable across
# pretraining and SFT runs on the same hardware.

GPU_PEAK_TFLOPS = {
    # NVIDIA consumer
    "NVIDIA GeForce RTX 4090":      165.2,
    "NVIDIA GeForce RTX 4080 SUPER": 105.0,
    "NVIDIA GeForce RTX 4080":       97.5,
    "NVIDIA GeForce RTX 4070 Ti SUPER": 88.0,
    "NVIDIA GeForce RTX 4070 Ti":    80.8,
    "NVIDIA GeForce RTX 4070 SUPER": 70.9,
    "NVIDIA GeForce RTX 4070":       59.8,
    "NVIDIA GeForce RTX 4060 Ti":    44.0,
    "NVIDIA GeForce RTX 4060":       30.0,
    "NVIDIA GeForce RTX 3090 Ti":    80.0,
    "NVIDIA GeForce RTX 3090":       71.0,
    "NVIDIA GeForce RTX 3080 Ti":    81.1,
    "NVIDIA GeForce RTX 3080":       59.4,
    "NVIDIA GeForce RTX 3070 Ti":    43.1,
    "NVIDIA GeForce RTX 3070":       40.4,
    "NVIDIA GeForce RTX 3060 Ti":    32.0,
    "NVIDIA GeForce RTX 3060":       25.0,
    # NVIDIA data-centre
    "NVIDIA A100-SXM4-80GB":        312.0,
    "NVIDIA A100-SXM4-40GB":        312.0,
    "NVIDIA A100-PCIE-80GB":        312.0,
    "NVIDIA A100-PCIE-40GB":        312.0,
    "NVIDIA H100 SXM5":             989.5,
    "NVIDIA H100 PCIe":             756.0,
    "NVIDIA H200":                  989.5,
    "NVIDIA L40S":                  362.1,
    "NVIDIA L40":                   181.0,
    "NVIDIA L4":                    121.0,
    "NVIDIA A10G":                  125.0,
    "NVIDIA A10":                   125.0,
    "NVIDIA A30":                   165.0,
    "NVIDIA A40":                   149.7,
    "NVIDIA V100-SXM2-32GB":         28.0,
    "NVIDIA V100-SXM2-16GB":         28.0,
    "NVIDIA V100-PCIE-16GB":         14.0,
    # AMD Instinct
    "AMD Instinct MI300X":          1307.4,
    "AMD Instinct MI300A":           383.0,
    "AMD Instinct MI250X":           383.0,
    "AMD Instinct MI210":            181.0,
    "AMD Instinct MI100":             46.1,
    # NVIDIA Jetson / embedded
    "NVIDIA Orin":                     1.3,
    "NVIDIA Xavier":                   1.0,
}


# ---------------------------------------------------------------------------
# Multi-tier TFLOPS resolution  (identical to train_deepspeed.py)
# ---------------------------------------------------------------------------

def _run(cmd: str) -> str:
    """Run a shell command, return stdout or '' on error."""
    try:
        return subprocess.check_output(
            cmd, shell=True, stderr=subprocess.DEVNULL, timeout=10
        ).decode().strip()
    except Exception:
        return ""


def _tflops_from_smi(gpu_index: int) -> Optional[float]:
    """Hook for future nvidia-smi TFLOPS support; currently always None."""
    return None


def _tflops_from_cuda_props(props) -> Tuple[float, str]:
    """Estimate bf16 TFLOPS from CUDA device properties (±15% accuracy)."""
    cc_major  = props.major
    cc_minor  = props.minor
    n_sm      = props.multi_processor_count
    clock_hz  = props.clock_rate * 1000   # kHz → Hz

    cores_per_sm = {
        (9, 0): 128, (8, 9): 128, (8, 6): 128, (8, 0): 64,
        (7, 5): 64,  (7, 0): 64,  (6, 1): 128, (6, 0): 64,
    }.get((cc_major, cc_minor), 128 if cc_major >= 8 else 64)

    fp32_tflops = (2 * n_sm * cores_per_sm * clock_hz) / 1e12
    if cc_major >= 9:
        bf16_mult = 2.0
    elif cc_major == 8:
        bf16_mult = 2.0
    elif cc_major == 7:
        bf16_mult = 8.0
    else:
        bf16_mult = 1.0

    est = fp32_tflops * bf16_mult
    method = (f"estimated from {n_sm} SMs × {cores_per_sm} cores/SM "
              f"@ {clock_hz/1e9:.2f} GHz × {bf16_mult}× TC")
    return round(est, 1), method


def resolve_gpu_peak_tflops(name: str, gpu_index: int, props) -> Tuple[float, str]:
    """
    Four-tier TFLOPS resolution (mirrors train_deepspeed.py):
      Tier 1 — exact table match
      Tier 2 — partial token match
      Tier 3 — nvidia-smi
      Tier 4 — derived from CUDA properties
    """
    name_lo = name.lower()
    for key, val in GPU_PEAK_TFLOPS.items():
        if key.lower() == name_lo:
            return val, "spec-sheet (exact match)"

    import re
    def _tokens(s):
        return set(re.split(r'[\s\-_]+', s.lower()))

    name_tokens = _tokens(name)
    best_score, best_key, best_val = 0, None, None
    for key, val in GPU_PEAK_TFLOPS.items():
        key_tokens = _tokens(key)
        score = len(key_tokens & name_tokens)
        if score >= 3 and score > best_score:
            best_score, best_key, best_val = score, key, val
    if best_key is not None:
        return best_val, f"spec-sheet (token match on '{best_key}', {best_score} tokens)"

    smi_val = _tflops_from_smi(gpu_index)
    if smi_val is not None:
        return smi_val, "nvidia-smi query"

    est, method = _tflops_from_cuda_props(props)
    return est, f"computed ({method})"


# ---------------------------------------------------------------------------
# ── HARDWARE AUDIT ──────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def audit_hardware() -> dict:
    """Per-GPU and CPU information for the current node (drives ZeRO choice)."""
    info: dict = {
        "node": platform.node(),
        "gpus": [],
        "cpu": {},
        "interconnect": {},
    }

    n_gpus = torch.cuda.device_count()
    for i in range(n_gpus):
        props    = torch.cuda.get_device_properties(i)
        name     = props.name
        vram_gb  = props.total_memory / 1024**3
        cc_major = props.major
        cc_minor = props.minor

        peak_tflops, tflops_source = resolve_gpu_peak_tflops(name, i, props)

        bw_str = _run(
            f"nvidia-smi --query-gpu=memory.bandwidth --format=csv,noheader,nounits "
            f"-i {i} 2>/dev/null"
        )
        try:
            bw_gb_s = float(bw_str) / 1000
        except ValueError:
            bw_gb_s = {
                "4090": 1008, "3090": 936, "A100": 2000,
                "H100": 3350, "V100": 900,
            }.get(next((k for k in ["4090", "3090", "A100", "H100", "V100"]
                        if k in name), ""), 800)

        nvlink_str = _run(f"nvidia-smi nvlink -s -i {i} 2>/dev/null | grep 'Speed' | head -1")
        has_nvlink = bool(nvlink_str)

        info["gpus"].append({
            "index":         i,
            "name":          name,
            "vram_gb":       round(vram_gb, 2),
            "cc":            f"{cc_major}.{cc_minor}",
            "bf16":          cc_major >= 8,
            "peak_tflops":   peak_tflops,
            "tflops_source": tflops_source,
            "bw_gb_s":       bw_gb_s,
            "has_nvlink":    has_nvlink,
        })

    try:
        import psutil
        cpu_ram_gb = psutil.virtual_memory().total / 1024**3
        cpu_cores  = psutil.cpu_count(logical=False) or 1
    except ImportError:
        cpu_ram_gb = 0.0
        cpu_cores  = os.cpu_count() or 1

    info["cpu"] = {
        "ram_gb": round(cpu_ram_gb, 1),
        "cores":  cpu_cores,
        "model":  platform.processor(),
    }

    ib_str = _run("ibstat 2>/dev/null | grep 'State: Active' | wc -l")
    try:
        info["interconnect"]["infiniband_ports"] = int(ib_str)
    except ValueError:
        info["interconnect"]["infiniband_ports"] = 0
    info["interconnect"]["nvlink"] = any(g["has_nvlink"] for g in info["gpus"])

    return info


def print_audit(info: dict, n_trainable: int, n_total: int, lora: bool):
    sep = "─" * 64
    print(f"\n{sep}")
    print(f"  HARDWARE AUDIT  —  node: {info['node']}")
    print(sep)
    print(f"  GPUs: {len(info['gpus'])}")
    for g in info["gpus"]:
        bf16_tag = "bf16✓" if g["bf16"] else "fp16-only"
        nvlink   = " NVLink✓" if g["has_nvlink"] else ""
        src      = g.get("tflops_source", "unknown")
        acc_tag  = "" if "spec-sheet" in src else f"  ⚠ TFLOPS estimated ({src})"
        print(f"    [{g['index']}] {g['name']}  "
              f"{g['vram_gb']:.1f} GB VRAM  "
              f"{g['peak_tflops']:.0f} TFLOP/s [{src}]  "
              f"CC{g['cc']}  {bf16_tag}{nvlink}{acc_tag}")
    cpu = info["cpu"]
    print(f"  CPU: {cpu.get('model','?')[:50]}  "
          f"{cpu['cores']} cores  {cpu['ram_gb']:.0f} GB RAM")
    ib = info["interconnect"]["infiniband_ports"]
    nv = "NVLink✓" if info["interconnect"]["nvlink"] else "PCIe"
    print(f"  Interconnect: {nv}  "
          f"{'InfiniBand (' + str(ib) + ' ports)' if ib else 'Ethernet'}")

    # Model / trainable breakdown
    mode = "LoRA" if lora else "full fine-tune"
    trainable_pct = 100.0 * n_trainable / max(n_total, 1)
    print(f"\n  Model:       {n_total/1e9:.3f}B params total  "
          f"(~{n_total*2/1024**3:.1f} GB bf16 weights)")
    print(f"  Trainable:   {n_trainable/1e6:.2f}M params  "
          f"({trainable_pct:.2f}%)  [{mode}]")
    print(sep + "\n")


# ---------------------------------------------------------------------------
# ── ZERO STAGE AUTO-SELECTION ────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def select_zero_stage_and_offload(
    info: dict,
    n_trainable: int,
    n_total: int,
    world_size: int,
    force_stage: Optional[int],
    force_cpu_offload_optimizer: bool,
    force_cpu_offload_param: bool,
) -> Tuple[int, bool, bool]:
    """
    SFT has two different memory profiles depending on the mode:
      - LoRA  : only the adapter parameters + their optimizer states are
                trainable; base weights are frozen and live in bf16
      - full  : every parameter is trainable, identical to pretraining

    We size the budget off `n_total` (the bigger of the two), because
    base weights still need to be resident in VRAM regardless of mode.
    Adam state is sized off `n_trainable` since only those parameters
    carry optimizer state.
    """
    if not info["gpus"]:
        return 1, False, False

    min_vram   = min(g["vram_gb"] for g in info["gpus"]) * 0.85
    n_gpus     = max(len(info["gpus"]), 1)

    # Static components per GPU (GB)
    # weights(bf16) + grads(bf16) + Adam(m+v fp32)
    full_gb   = n_total  * (2 + 2 + 8) / 1024**3 / n_gpus
    zero2_gb  = n_total  * (2 + 2)     / 1024**3 / n_gpus
    zero3_gb  = n_total  * 2           / 1024**3 / n_gpus

    if force_stage is not None:
        stage = force_stage
    elif min_vram >= full_gb:
        stage = 1
    elif min_vram >= zero2_gb:
        stage = 2
    else:
        stage = 3

    cpu_offload_opt   = force_cpu_offload_optimizer
    cpu_offload_param = force_cpu_offload_param

    if not cpu_offload_opt and not cpu_offload_param:
        if stage == 3 and min_vram < zero3_gb:
            cpu_offload_opt = True
            print(f"[AutoConfig] ZeRO-3 params still exceed VRAM "
                  f"({zero3_gb:.1f} GB needed, {min_vram:.1f} GB available). "
                  f"Enabling CPU optimizer offload.")
        if stage == 3 and min_vram < (zero3_gb * 0.6):
            cpu_offload_param = True
            print(f"[AutoConfig] VRAM very tight — also enabling CPU parameter offload.")

    cpu_ram = info["cpu"].get("ram_gb", 0)
    if (cpu_offload_opt or cpu_offload_param) and cpu_ram > 0:
        # Adam m+v for the trainable parameters (in fp32) is the dominant
        # offload cost in SFT (LoRA mode in particular).
        needed_gb = n_trainable * 8 / 1024**3
        if needed_gb > cpu_ram * 0.6:
            print(f"[AutoConfig] WARNING: optimizer offload needs ~{needed_gb:.1f} GB "
                  f"CPU RAM but only {cpu_ram:.0f} GB available.")

    return stage, cpu_offload_opt, cpu_offload_param


# ---------------------------------------------------------------------------
# ── DS CONFIG BUILDER ────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def build_ds_config(
    args,
    zero_stage: int,
    cpu_offload_optimizer: bool,
    cpu_offload_param: bool,
    gpu_info: List[dict],
) -> dict:
    """
    Construct a deepspeed config dict. The shape is identical to
    train_deepspeed.build_ds_config; the LR defaults and weight-decay
    defaults differ because SFT uses a different regime.
    """
    optimizer_cfg = {
        "type": "AdamW",
        "params": {
            "lr":           args.lr,
            "betas":        [0.9, 0.95],
            "eps":          1e-8,
            "weight_decay": args.weight_decay,
        },
    }

    # We do not use DS's built-in scheduler — train_sft.py drives a
    # manual warmup+cosine schedule via engine.optimizer.param_groups
    # overrides, same as train_deepspeed.py does for pretraining.
    scheduler_cfg = {
        "type": "WarmupCosineLR",
        "params": {
            "warmup_num_steps":      args.warmup_steps,
            "total_num_steps":       args.max_steps,
            "warmup_type":           "linear",
            "last_batch_iteration": -1,
        },
    }

    bf16_cfg = {"enabled": args.dtype == "bf16"}
    fp16_cfg = {"enabled": False}

    zero_cfg: dict = {
        "stage": zero_stage,
        "reduce_bucket_size":    5e8,
        "allgather_bucket_size": 5e8,
        "overlap_comm":          True,
        "contiguous_gradients":  True,
        "sub_group_size":        1e9,
        "stage3_max_live_parameters":   1e9,
        "stage3_max_reuse_distance":    1e9,
        "stage3_gather_16bit_weights_on_model_save": True,
    }

    if cpu_offload_optimizer:
        zero_cfg["offload_optimizer"] = {
            "device":     "cpu",
            "pin_memory": True,
            "ratio":      1.0,
        }
    if cpu_offload_param and zero_stage == 3:
        zero_cfg["offload_param"] = {
            "device":       "cpu",
            "pin_memory":   True,
            "buffer_count": 5,
            "buffer_size":  1e8,
        }

    has_nvlink = any(g.get("has_nvlink") for g in gpu_info)
    zero_cfg["reduce_scatter"]       = True
    zero_cfg["allgather_partitions"] = True
    if not has_nvlink:
        zero_cfg["reduce_bucket_size"]    = 2e8
        zero_cfg["allgather_bucket_size"] = 2e8

    act_ckpt: dict = {}
    if args.gradient_checkpointing:
        act_ckpt = {
            "partition_activations":          False,
            "cpu_checkpointing":              False,
            "contiguous_memory_optimization": False,
            "synchronize_checkpoint_boundary": False,
        }

    cfg = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps":    args.grad_accum_steps,
        "gradient_clipping":              args.grad_clip,
        "steps_per_print":                args.log_interval,
        "wall_clock_breakdown":           False,
        "optimizer":                      optimizer_cfg,
        "scheduler":                      scheduler_cfg,
        "bf16":                           bf16_cfg,
        "fp16":                           fp16_cfg,
        "zero_optimization":              zero_cfg,
    }
    if act_ckpt:
        cfg["activation_checkpointing"] = act_ckpt

    return cfg


def print_ds_config_summary(cfg: dict, zero_stage: int,
                             cpu_opt: bool, cpu_param: bool):
    sep = "─" * 64
    print(f"\n{sep}")
    print(f"  DEEPSPEED CONFIG SUMMARY")
    print(sep)
    print(f"  ZeRO Stage            : {zero_stage}")
    print(f"  CPU offload optimizer : {cpu_opt}")
    print(f"  CPU offload params    : {cpu_param}")
    print(f"  BF16                  : {cfg['bf16']['enabled']}")
    print(f"  Grad accum steps      : {cfg['gradient_accumulation_steps']}")
    print(f"  Micro batch / GPU     : {cfg['train_micro_batch_size_per_gpu']}")
    print(f"  Grad clip             : {cfg['gradient_clipping']}")
    z = cfg["zero_optimization"]
    print(f"  Reduce bucket         : {z['reduce_bucket_size']/1e6:.0f} MB")
    print(f"  Overlap comm          : {z['overlap_comm']}")
    print(sep + "\n")


# ---------------------------------------------------------------------------
# ── PRETRAINED CHECKPOINT LOADER ─────────────────────────────────────────────
# ---------------------------------------------------------------------------

def load_pretrained_checkpoint(path: str) -> Tuple[Qwen3Config, dict]:
    """
    Accept either:
      - a raw .pt from train.py / deepspeed_shard_consolidator.py with
        keys {'model_state', 'config'}
      - a DeepSpeed checkpoint directory (from train_deepspeed.py) —
        in that case we point the user at deepspeed_shard_consolidator.py
        rather than silently re-implementing ZeRO gathering here.
    """
    if os.path.isdir(path):
        # Could be a DeepSpeed checkpoint directory. The directory layout
        # always contains a meta.json written by train_deepspeed.py.
        meta_path = os.path.join(path, "meta.json")
        if os.path.exists(meta_path):
            raise RuntimeError(
                f"{path} looks like a DeepSpeed checkpoint directory.\n"
                f"Run deepspeed_shard_consolidator.py first to produce a "
                f"single .pt, then point --checkpoint at the consolidated file."
            )
        # Otherwise treat as a regular directory of state files — error out
        # with a clear message rather than guessing.
        raise RuntimeError(
            f"{path} is a directory but does not contain meta.json; not a "
            f"recognised SFT input. Pass the path to a consolidated .pt."
        )

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Pretrained checkpoint not found: {path}")

    blob = torch.load(path, map_location="cpu", weights_only=False)
    if "config" not in blob or "model_state" not in blob:
        raise RuntimeError(
            f"{path} does not contain the expected keys "
            f"'config' and 'model_state'. Was it produced by train.py or "
            f"deepspeed_shard_consolidator.py?"
        )
    return Qwen3Config(**blob["config"]), blob["model_state"]


# ---------------------------------------------------------------------------
# ── MFU ─────────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def estimate_mfu(model, tokens_per_sec: float, gpu_info: List[dict]) -> float:
    """
    MFU is reported against the *trainable* parameter count when running
    in LoRA mode (that's the work that's actually happening) and against
    the *non-embedding total* otherwise. This keeps the metric meaningful
    for sparse-adapter runs where 99.9% of weights are frozen.
    """
    raw = model.module if hasattr(model, "module") else model
    inner = raw._orig_mod if hasattr(raw, "_orig_mod") else raw
    # Treat LoRA matrices as the trainable work; include norms/embeddings
    # in the count only when we're doing a full fine-tune.
    is_lora = any(isinstance(m, LoRALinear) for m in inner.modules())
    if is_lora:
        n = sum(
            p.numel() for pname, p in inner.named_parameters()
            if ("lora_A" in pname or "lora_B" in pname) and p.requires_grad
        )
    else:
        n = sum(p.numel() for pname, p in inner.named_parameters()
                if "embed_tokens" not in pname)

    flops = 6 * n * tokens_per_sec
    if not gpu_info:
        return 0.0
    peak = gpu_info[0]["peak_tflops"] * 1e12 * len(gpu_info)
    return flops / peak if peak > 0 else 0.0


# ---------------------------------------------------------------------------
# ── CHECKPOINT ──────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def save_checkpoint(engine, step: int, out_dir: str,
                    config: Qwen3Config, args_dict: dict,
                    best_val_loss: float, is_lora: bool):
    """
    DeepSpeed-native save. Writes a directory `step_<n>/` containing
    ZeRO-sharded model/optimizer states, plus a sidecar meta.json that
    holds our config, original CLI args, best val loss, and the LoRA
    adapter (if applicable) so the model can be reconstructed outside
    DeepSpeed without parsing the engine internals.
    """
    tag  = f"step_{step:07d}"
    path = os.path.join(out_dir, tag)
    engine.save_checkpoint(out_dir, tag=tag)

    sidecar: dict = {
        "step":          step,
        "config":        vars(config),
        "args":          args_dict,
        "best_val_loss": best_val_loss,
        "ds_tag":        tag,
        "is_lora":       is_lora,
    }

    # Pull a copy of the LoRA tensors off the (sharded) engine. Saving the
    # full model state for a ZeRO-3 LoRA run still works because
    # stage3_gather_16bit_weights_on_model_save=True, but we only need
    # the adapter params (a few MB) for LoRA resume.
    if is_lora:
        raw = engine.module
        inner = raw._orig_mod if hasattr(raw, "_orig_mod") else raw
        sidecar["lora_state"] = lora_state_dict(inner)

    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump(sidecar, f, indent=2)

    latest = os.path.join(out_dir, "latest_ds")
    if os.path.islink(latest):
        os.remove(latest)
    os.symlink(os.path.abspath(path), latest)
    print(f"[Checkpoint] saved {path}  (lora={is_lora})")


def load_checkpoint(engine, resume_path: str, is_lora: bool
                    ) -> Tuple[int, float]:
    """
    Reload a DeepSpeed SFT checkpoint.  LoRA mode is special: the sidecar
    `meta.json` contains `lora_state`; we restore that into the in-memory
    model after engine.load_checkpoint has restored the ZeRO-sharded base
    weights.
    """
    meta_path = os.path.join(resume_path, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"No meta.json in {resume_path} — not a DeepSpeed SFT checkpoint?"
        )
    with open(meta_path) as f:
        meta = json.load(f)
    tag = meta["ds_tag"]
    engine.load_checkpoint(os.path.dirname(resume_path), tag=tag)

    if is_lora and "lora_state" in meta:
        raw = engine.module
        inner = raw._orig_mod if hasattr(raw, "_orig_mod") else raw
        # strict=False: base weights were just restored by load_checkpoint,
        # only the adapter keys should appear in sidecar.
        missing, unexpected = inner.load_state_dict(meta["lora_state"], strict=False)
        if unexpected:
            print(f"[Checkpoint] WARNING: {len(unexpected)} unexpected LoRA "
                  f"keys when resuming; ignoring")
        print(f"[Checkpoint] restored {len(meta['lora_state'])} LoRA tensors "
              f"from sidecar")

    step          = meta.get("step", 0)
    best_val_loss = meta.get("best_val_loss", float("inf"))
    print(f"[Checkpoint] resumed from {resume_path} at step {step}")
    return step, best_val_loss


def prune_checkpoints(out_dir: str, keep: int = 3):
    dirs = sorted(
        [d for d in Path(out_dir).iterdir()
         if d.is_dir() and d.name.startswith("step_")],
        key=lambda d: int(d.name.replace("step_", "")),
    )
    for old in dirs[:-keep]:
        import shutil
        shutil.rmtree(old)
        print(f"[Checkpoint] pruned {old.name}")


# ---------------------------------------------------------------------------
# ── EVAL ────────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(engine, val_ds: SFTDataset, eval_steps: int,
             batch_size: int, device: torch.device, world_size: int):
    engine.eval()
    losses: List[float] = []
    for _ in range(eval_steps):
        x, y, m = val_ds.get_batch(batch_size, device=device)
        out     = engine(x)
        loss    = masked_cross_entropy(out["logits"], y, m)
        losses.append(loss.item())
    engine.train()
    mean_loss = float(np.mean(losses)) if losses else float("inf")
    if world_size > 1:
        t = torch.tensor(mean_loss, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.AVG)
        mean_loss = t.item()
    return mean_loss


# ---------------------------------------------------------------------------
# ── LR SCHEDULE (manual override; mirrors train_deepspeed.py) ────────────────
# ---------------------------------------------------------------------------

def _cosine_lr(step: int, warmup: int, max_steps: int,
               max_lr: float, min_lr: float) -> float:
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step >= max_steps:
        return min_lr
    t = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + 0.5 * (1 + math.cos(math.pi * t)) * (max_lr - min_lr)


# ---------------------------------------------------------------------------
# ── MAIN ────────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def train(args):
    # DeepSpeed initialises its own process group — don't call dist.init
    local_rank  = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK",       0))
    world_size  = int(os.environ.get("WORLD_SIZE",  1))
    master      = global_rank == 0

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    torch.manual_seed(args.seed + global_rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32        = True

    # ---------------------------------------------------------------- model
    config, base_state = load_pretrained_checkpoint(args.checkpoint)
    model = Qwen3ForCausalLM(config)
    # Tied weights must be re-established after load_state_dict, exactly
    # as train.py and train_sft.py do.
    model.load_state_dict(base_state)
    model.tie_weights()

    is_lora = args.lora
    if is_lora:
        n_replaced = inject_lora(
            model, rank=args.lora_rank, alpha=args.lora_alpha,
        )
        n_trainable = lora_parameter_count(model)
        if master:
            print(f"[LoRA] injected {n_replaced} adapters  "
                  f"target=q,k,v,o,gate,up,down  rank={args.lora_rank}  "
                  f"alpha={args.lora_alpha}")
    else:
        n_trainable = sum(p.numel() for p in model.parameters()
                          if p.requires_grad)

    n_total = count_parameters(model)
    if master:
        print(f"Pretrained model: {n_total/1e9:.3f}B params  "
              f"({n_total:,} total)")

    # -------------------------------------------------------- gradient ckpt
    if args.gradient_checkpointing:
        model.model.enable_gradient_checkpointing()

    # ---------------------------------------------------------------- audit
    hw = audit_hardware()
    if master:
        print_audit(hw, n_trainable, n_total, lora=is_lora)

    # -------------------------------------------------------- ZeRO selection
    zero_stage, cpu_offload_opt, cpu_offload_param = select_zero_stage_and_offload(
        hw, n_trainable, n_total, world_size,
        force_stage=args.zero_stage,
        force_cpu_offload_optimizer=args.cpu_offload_optimizer,
        force_cpu_offload_param=args.cpu_offload_param,
    )
    if master:
        print(f"[AutoConfig] Selected ZeRO-{zero_stage}  "
              f"cpu_offload_opt={cpu_offload_opt}  "
              f"cpu_offload_param={cpu_offload_param}")

    # ---------------------------------------------------------------- DS cfg
    ds_cfg = build_ds_config(
        args, zero_stage, cpu_offload_opt, cpu_offload_param,
        gpu_info=hw["gpus"],
    )
    if master:
        os.makedirs(args.out_dir, exist_ok=True)
        cfg_path = os.path.join(args.out_dir, "ds_config.json")
        with open(cfg_path, "w") as f:
            json.dump(ds_cfg, f, indent=2)
        print_ds_config_summary(ds_cfg, zero_stage, cpu_offload_opt, cpu_offload_param)
        print(f"[DeepSpeed] config written to {cfg_path}")

    # --------------------------------------------------------- param groups
    # Exclude norms/embeddings/LoRA-B from weight decay, exactly as
    # train_sft.build_optimizer does.
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "norm" in name or "embed" in name or "lora_B" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    param_groups = [
        {"params": decay,    "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    if master:
        print(f"[Optimizer] decay={sum(p.numel() for p in decay):,}  "
              f"no_decay={sum(p.numel() for p in no_decay):,}")

    # --------------------------------------------------------- DeepSpeed init
    engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=param_groups,
        config=ds_cfg,
    )

    # ---------------------------------------------------------------- data
    if master:
        print(f"\nReading packed SFT data from {args.cache_dir} …")
    train_ds = SFTDataset(
        cache_dir=args.cache_dir, seq_len=args.seq_len,
        rank=global_rank, world_size=world_size, split="train",
    )
    val_ds = SFTDataset(
        cache_dir=args.cache_dir, seq_len=args.seq_len,
        rank=global_rank, world_size=world_size, split="val",
    )
    if len(train_ds) == 0:
        raise RuntimeError(
            "Training dataset has no complete windows. Try a smaller --seq-len, "
            "or re-pack with a larger --target-size in download_sft_data.py."
        )

    # ---------------------------------------------------------------- resume
    start_step    = 0
    best_val_loss = float("inf")
    if args.resume:
        start_step, best_val_loss = load_checkpoint(engine, args.resume, is_lora)

    # ---------------------------------------------------------------- W&B
    use_wandb = False
    if master and args.wandb_project:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name or f"qwen3-sft-{args.model_size_label}-z{zero_stage}",
                config={
                    **vars(config), "n_params": n_total, "n_trainable": n_trainable,
                    "lora": is_lora, "lora_rank": args.lora_rank if is_lora else None,
                    "zero_stage": zero_stage, **vars(args),
                },
            )
            use_wandb = True
        except Exception as e:
            print(f"[W&B] disabled: {e}")

    # --------------------------------------------------------- accounting
    tokens_per_step = (
        args.batch_size * args.seq_len * args.grad_accum_steps * world_size
    )
    tokens_per_microbatch = args.batch_size * args.seq_len * world_size
    if master:
        print(f"\nTokens / optimizer step : {tokens_per_step:,}")
        print(f"Effective batch size    : {tokens_per_step // args.seq_len:,} samples")
        print(f"Max steps               : {args.max_steps:,}")
        print(f"Checkpoint every        : {args.ckpt_interval:,} steps\n")

    # ================================================================ LOOP
    engine.train()
    t0         = time.perf_counter()
    loss_accum = 0.0

    for step in range(start_step, args.max_steps):
        # Manual LR override; same rationale as train_deepspeed.py
        lr = _cosine_lr(step, args.warmup_steps, args.max_steps,
                        args.lr, args.min_lr)
        for pg in engine.optimizer.param_groups:
            pg["lr"] = lr

        # engine.backward + engine.step replace the manual
        # micro-step + accumulation loop from train_sft.py; DeepSpeed
        # counts micro-steps itself and only steps the optimizer at the
        # end of the configured accumulation window.
        x, y, m = train_ds.get_batch(args.batch_size, device)
        out     = engine(x)
        # masked cross entropy: only assistant tokens contribute
        loss    = masked_cross_entropy(out["logits"], y, m)

        engine.backward(loss)
        engine.step()

        loss_accum += loss.item()

        # ---- logging
        if master and step % args.log_interval == 0:
            t1          = time.perf_counter()
            tok_per_sec = tokens_per_microbatch * args.log_interval / max(t1 - t0, 1e-9)
            mfu         = estimate_mfu(engine, tok_per_sec, hw["gpus"])
            loss_display = loss_accum / args.log_interval
            loss_accum   = 0.0
            grad_norm    = engine.get_global_grad_norm() or 0.0

            print(
                f"step {step:7d} | loss {loss_display:.4f} | lr {lr:.2e} | "
                f"grad {grad_norm:.3f} | {tok_per_sec/1e3:.1f}k tok/s | "
                f"mfu {mfu*100:.2f}%"
            )
            if use_wandb:
                import wandb
                wandb.log({
                    "train/loss":          loss_display,
                    "train/lr":            lr,
                    "train/grad_norm":     grad_norm,
                    "perf/tokens_per_sec": tok_per_sec,
                    "perf/mfu_pct":        mfu * 100,
                }, step=step)
            t0 = t1

        # ---- validation
        if step % args.eval_interval == 0 and step > start_step:
            val_loss = evaluate(
                engine, val_ds, args.eval_steps, args.batch_size, device, world_size,
            )
            if master:
                improved = " ✓ best" if val_loss < best_val_loss else ""
                print(f"  [eval] step {step:7d} | val_loss {val_loss:.4f}{improved}")
                if use_wandb:
                    import wandb
                    wandb.log({"val/loss": val_loss}, step=step)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss

        # ---- checkpoint (all ranks must call save_checkpoint together)
        if step % args.ckpt_interval == 0 and step > start_step:
            if master:
                save_checkpoint(
                    engine, step, args.out_dir,
                    config, vars(args), best_val_loss, is_lora,
                )
                prune_checkpoints(args.out_dir, keep=args.keep_ckpts)

    # ---- final checkpoint
    if master:
        save_checkpoint(
            engine, args.max_steps, args.out_dir,
            config, vars(args), best_val_loss, is_lora,
        )
        print(f"\nSFT complete. Best val loss: {best_val_loss:.4f}")
        if is_lora:
            print(f"\nTo merge LoRA into base weights for deployment:")
            print(f"  python train_sft_deepspeed.py --merge-lora \\")
            print(f"      --checkpoint <base .pt> \\")
            print(f"      --out-dir ./sft_merged")
        if use_wandb:
            import wandb
            wandb.finish()


# ---------------------------------------------------------------------------
# ── MERGE-ONLY MODE ─────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------
# CPU-only path. Reads a (raw or DeepSpeed SFT) LoRA checkpoint and writes
# a consolidated .pt with the adapter folded into the base weights.

def merge_and_save(args):
    device = torch.device("cpu")
    print(f"[Merge] loading base checkpoint {args.checkpoint} …")
    config, base_state = load_pretrained_checkpoint(args.checkpoint)
    model = Qwen3ForCausalLM(config)
    model.load_state_dict(base_state)
    model.tie_weights()

    # The SFT sidecar holds the LoRA tensors in raw form
    ckpt_blob = torch.load(args.checkpoint, map_location=device, weights_only=False)
    lora_sd   = ckpt_blob.get("lora_state")
    if lora_sd is None:
        raise RuntimeError(
            f"{args.checkpoint} has no 'lora_state' field — was it saved by "
            f"train_sft_deepspeed.py in --lora mode?"
        )

    rank  = ckpt_blob.get("args", {}).get("lora_rank",  64)
    alpha = ckpt_blob.get("args", {}).get("lora_alpha", 128.0)
    n_lora = inject_lora(model, rank=rank, alpha=alpha)
    print(f"[Merge] injected {n_lora} LoRA adapters (rank={rank}, alpha={alpha})")

    missing, unexpected = model.load_state_dict(lora_sd, strict=False)
    if unexpected:
        print(f"[Merge] WARNING: {len(unexpected)} unexpected keys when "
              f"loading LoRA state; ignoring")
    if missing:
        print(f"[Merge] WARNING: {len(missing)} missing keys when loading "
              f"LoRA state (should be empty)")

    model = merge_lora(model)
    model.tie_weights()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "merged_model.pt")
    torch.save({"model_state": model.state_dict(),
                "config":      vars(config)}, out_path)
    print(f"[Merge] saved merged model to {out_path}")


# ---------------------------------------------------------------------------
# ── CLI ─────────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="DeepSpeed SFT for Qwen3-style dense LLM (Stage 1 of "
                    "reasoning post-training)."
    )

    # mode
    p.add_argument("--merge-lora", action="store_true",
                   help="Merge LoRA into base weights and save; skip training. "
                        "Runs on CPU, DeepSpeed not required.")

    # paths
    p.add_argument("--checkpoint", default=None,
                   help="Pretrained checkpoint (raw .pt or consolidated DS .pt). "
                        "Required unless --merge-lora is set without one.")
    p.add_argument("--cache-dir",  default="./sft_packed",
                   help="Packed memmap files from pack_sft_data.py")
    p.add_argument("--out-dir",    default="./sft_checkpoints_ds")
    p.add_argument("--resume",     default=None,
                   help="Path to a DeepSpeed SFT checkpoint directory to resume from")

    # LoRA
    p.add_argument("--lora",       action="store_true",
                   help="Enable LoRA (recommended for >=1B on a single GPU)")
    p.add_argument("--lora-rank",  type=int,   default=64)
    p.add_argument("--lora-alpha", type=float, default=128.0)

    # training
    p.add_argument("--model-size-label", default="sft",
                   help="Label used in W&B run name; the actual architecture "
                        "is read from --checkpoint's config")
    p.add_argument("--seq-len",          type=int,   default=2048)
    p.add_argument("--batch-size",       type=int,   default=4,
                   help="Micro-batch size PER GPU (before grad accum)")
    p.add_argument("--grad-accum-steps", type=int,   default=8)
    p.add_argument("--max-steps",        type=int,   default=10_000)
    p.add_argument("--warmup-steps",     type=int,   default=200)
    p.add_argument("--lr",               type=float, default=2e-5,
                   help="Peak LR (typically 1e-5 to 5e-5 for SFT)")
    p.add_argument("--min-lr",           type=float, default=2e-6)
    p.add_argument("--weight-decay",     type=float, default=0.01)
    p.add_argument("--grad-clip",        type=float, default=1.0)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--gradient-checkpointing", action="store_true",
                   help="Recompute activations on backward (~35%% less VRAM)")
    p.add_argument("--seed", type=int, default=42)

    # ZeRO / offload
    p.add_argument("--zero-stage", type=int, default=None, choices=[1, 2, 3],
                   help="Force ZeRO stage. Default: auto-selected from hardware audit.")
    p.add_argument("--cpu-offload-optimizer", action="store_true",
                   help="Force CPU offload of optimizer states")
    p.add_argument("--cpu-offload-param",     action="store_true",
                   help="Force CPU offload of model parameters (ZeRO-3 only)")

    # checkpointing / logging
    p.add_argument("--ckpt-interval",  type=int, default=1_000)
    p.add_argument("--keep-ckpts",     type=int, default=3)
    p.add_argument("--log-interval",   type=int, default=10)
    p.add_argument("--eval-interval",  type=int, default=200)
    p.add_argument("--eval-steps",     type=int, default=20)
    p.add_argument("--wandb-project",  default=None)
    p.add_argument("--wandb-run-name", default=None)

    # DeepSpeed passes its own args
    p.add_argument("--local_rank", type=int, default=-1,
                   help="Set by DeepSpeed launcher; do not set manually.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# ── ENTRY ───────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    # merge-lora runs on CPU without DeepSpeed
    if args.merge_lora:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for --merge-lora")
        merge_and_save(args)
        sys.exit(0)

    if not args.checkpoint:
        raise ValueError(
            "--checkpoint is required (path to a pretrained .pt produced by "
            "train.py or deepspeed_shard_consolidator.py)."
        )

    if not torch.cuda.is_available():
        print("ERROR: train_sft_deepspeed.py requires at least one CUDA GPU.")
        sys.exit(1)

    try:
        import deepspeed  # noqa: F401
    except ImportError:
        print("ERROR: DeepSpeed not installed.  Run:")
        print("  pip install deepspeed")
        sys.exit(1)

    try:
        import psutil  # noqa: F401
    except ImportError:
        print("[warn] psutil not installed — CPU RAM reporting will be incomplete.")
        print("       pip install psutil")

    if not _has_sft_manifests(args.cache_dir):
        print(f"ERROR: no sft_manifest.w*.json found in {args.cache_dir}.")
        print("Run pack_sft_data.py first to produce packed SFT data.")
        sys.exit(1)

    train(args)


# ---------------------------------------------------------------------------
# ── HELPERS ─────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def _has_sft_manifests(cache_dir: str) -> bool:
    """True if at least one pack_sft_data.py worker manifest is present."""
    if not os.path.isdir(cache_dir):
        return False
    return any(
        p.name.startswith("sft_manifest.w") and p.name.endswith(".json")
        for p in Path(cache_dir).iterdir()
    )
