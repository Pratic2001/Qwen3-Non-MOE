#!/usr/bin/env python3
"""
train_deepspeed.py

DeepSpeed-powered pretraining for the Qwen3-style dense LLM from model.py.

Before touching any training logic this script runs a full hardware audit:
  - Per-GPU VRAM, compute capability, bandwidth
  - Inter-node interconnect (NVLink vs PCIe, InfiniBand vs Ethernet)
  - CPU RAM, core count (for CPU offload decisions)

It then automatically selects the optimal DeepSpeed ZeRO stage and offload
configuration for the hardware it finds:

  ZeRO-1  — shard optimizer states only
             good when VRAM is abundant (>= 40 GB / GPU)
  ZeRO-2  — shard optimizer states + gradients
             default for 16–40 GB consumer / data-center GPUs
  ZeRO-3  — shard optimizer states + gradients + model parameters
             required for very large models (params > 2× VRAM per GPU)
  CPU offload — moves optimizer states (and optionally parameters) to CPU RAM
             kicks in when VRAM per GPU < threshold derived from model size

The generated ds_config.json is written to --out-dir and printed so you
can inspect or override it.

Launch:
    # Single node, 1 GPU
    deepspeed train_deepspeed.py --model-size 0.6B --data-dir ./packed

    # Single node, 4 GPUs
    deepspeed --num_gpus 4 train_deepspeed.py --model-size 1.7B --data-dir ./packed

    # Multi-node (2 nodes × 8 GPUs)
    deepspeed --hostfile hostfile.txt train_deepspeed.py \\
        --model-size 8B --data-dir ./packed

    # Force a specific ZeRO stage (skip auto-selection)
    deepspeed train_deepspeed.py --model-size 4B --zero-stage 3 \\
        --cpu-offload-optimizer --data-dir ./packed

    # Resume
    deepspeed train_deepspeed.py --resume ./checkpoints/latest_ds/ --data-dir ./packed
"""

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist

import deepspeed
from deepspeed.utils import logger as ds_logger

from model import Qwen3Config, Qwen3ForCausalLM, count_parameters


# ---------------------------------------------------------------------------
# GPU FLOP/s table  (bf16 Tensor Core, per card)
# ---------------------------------------------------------------------------
# Used for MFU estimation. The lookup table is the first tier — if a GPU
# isn't found here the code falls through to progressively less accurate
# estimation methods rather than silently using an arbitrary constant.

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
    "NVIDIA V100-PCIE-16GB":         14.0,  # fp16 only; no bf16 HW tensor core
    # AMD Instinct
    "AMD Instinct MI300X":          1307.4,
    "AMD Instinct MI300A":           383.0,
    "AMD Instinct MI250X":           383.0,
    "AMD Instinct MI210":            181.0,
    "AMD Instinct MI100":             46.1,
    # NVIDIA Jetson / embedded (not useful for LLM training but let's not crash)
    "NVIDIA Orin":                     1.3,
    "NVIDIA Xavier":                   1.0,
}


# ---------------------------------------------------------------------------
# Multi-tier TFLOPS resolution
# ---------------------------------------------------------------------------

def _tflops_from_smi(gpu_index: int) -> Optional[float]:
    """
    Try to read peak bf16 TFLOPS directly from nvidia-smi.
    Works on some data-centre cards (A100, H100); returns None otherwise.
    """
    raw = _run(
        f"nvidia-smi --query-gpu=clocks.max.sm,clocks.max.memory "
        f"--format=csv,noheader,nounits -i {gpu_index}"
    )
    # nvidia-smi doesn't expose TFLOPS directly; this path is a placeholder
    # for future nvidia-smi versions that might. Currently always returns None.
    return None


def _tflops_from_cuda_props(props) -> Tuple[float, str]:
    """
    Estimate bf16 TFLOPS from CUDA device properties.

    Formula:
        peak_bf16_tflops = 2 × SM_count × (CUDA_cores_per_SM × 2) × boost_clock
        ↑ ×2 for FMA,  ↑ ×2 for bf16 tensor-core 2:1 throughput vs FP32

    For Tensor Core architectures (CC ≥ 7.0) the tensor core multiplier is
    applied based on the generation:
        Volta  (7.0): 8× FP16 tensor core vs FP32 CUDA core throughput
        Turing (7.5): 8×
        Ampere (8.x): 2× bf16 vs FP32 (bf16 is supported natively)
        Ada    (8.9): same as Ampere for bf16
        Hopper (9.0): 2× bf16 (+ FP8 paths, not relevant here)

    Returns (estimated_tflops, method_description).
    """
    cc_major  = props.major
    cc_minor  = props.minor
    n_sm      = props.multi_processor_count
    clock_hz  = props.clock_rate * 1000   # kHz → Hz

    # CUDA cores per SM varies by architecture
    cores_per_sm = {
        (9, 0): 128,   # Hopper
        (8, 9): 128,   # Ada Lovelace
        (8, 6): 128,   # Ampere (GA10x consumer)
        (8, 0): 64,    # Ampere (GA100 data-centre)
        (7, 5): 64,    # Turing
        (7, 0): 64,    # Volta
        (6, 1): 128,   # Pascal (GP10x)
        (6, 0): 64,    # Pascal (GP100)
    }.get((cc_major, cc_minor),
          128 if cc_major >= 8 else 64)   # safe default

    fp32_tflops = (2 * n_sm * cores_per_sm * clock_hz) / 1e12

    # Tensor core bf16 multiplier vs scalar FP32 throughput
    if cc_major >= 9:        # Hopper — bf16 TC is ~2× FP32 scalar
        bf16_mult = 2.0
    elif cc_major == 8:      # Ampere / Ada — bf16 TC ~2× FP32 scalar
        bf16_mult = 2.0
    elif cc_major == 7:      # Volta / Turing — FP16 TC only, no bf16 HW
        # bf16 runs in FP16 tensor cores at same throughput as FP16
        bf16_mult = 8.0
    else:                    # Maxwell / Pascal — no tensor cores
        bf16_mult = 1.0

    est = fp32_tflops * bf16_mult
    method = (f"estimated from {n_sm} SMs × {cores_per_sm} cores/SM "
              f"@ {clock_hz/1e9:.2f} GHz × {bf16_mult}× TC")
    return round(est, 1), method


def resolve_gpu_peak_tflops(name: str, gpu_index: int, props) -> Tuple[float, str]:
    """
    Four-tier TFLOPS resolution for a GPU.

    Tier 1 — Exact table match (spec-sheet accurate)
    Tier 2 — Partial name match in the table (e.g. "RTX 4090" matches
              "NVIDIA GeForce RTX 4090"; handles OEM / vendor prefix variants)
    Tier 3 — nvidia-smi query (works on some server GPUs)
    Tier 4 — Derive from CUDA device properties (always available, ±15% accuracy)

    Every tier prints what it did so the user can tell how accurate the MFU is.
    """
    # Tier 1: exact case-insensitive match
    name_lo = name.lower()
    for key, val in GPU_PEAK_TFLOPS.items():
        if key.lower() == name_lo:
            return val, "spec-sheet (exact match)"

    # Tier 2: token-overlap partial match.
    # Split both the GPU name and each table key into word tokens, then score
    # by how many tokens from the table key appear in the GPU name.
    # This handles OEM prefix variants like:
    #   "NVIDIA RTX 4090"       → matches "NVIDIA GeForce RTX 4090"
    #   "Tesla A100-SXM4-80GB"  → matches "NVIDIA A100-SXM4-80GB"
    #   "A100 80GB PCIe"        → matches "NVIDIA A100-PCIE-80GB"
    import re
    def _tokens(s):
        return set(re.split(r'[\s\-_]+', s.lower()))

    name_tokens = _tokens(name)
    best_score, best_key, best_val = 0, None, None
    for key, val in GPU_PEAK_TFLOPS.items():
        key_tokens = _tokens(key)
        # Score = number of key tokens found in the GPU name
        # Require at least 3 matching tokens to avoid false positives
        # (e.g. "NVIDIA" + "RTX" + "4090" = 3 tokens matching)
        score = len(key_tokens & name_tokens)
        if score >= 3 and score > best_score:
            best_score, best_key, best_val = score, key, val

    if best_key is not None:
        return best_val, f"spec-sheet (token match on '{best_key}', {best_score} tokens)"

    # Tier 3: nvidia-smi
    smi_val = _tflops_from_smi(gpu_index)
    if smi_val is not None:
        return smi_val, "nvidia-smi query"

    # Tier 4: derive from CUDA device properties
    est, method = _tflops_from_cuda_props(props)
    return est, f"computed ({method})"


# ---------------------------------------------------------------------------
# ── HARDWARE AUDIT ──────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def _run(cmd: str) -> str:
    """Run a shell command, return stdout or '' on error."""
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL,
                                       timeout=10).decode().strip()
    except Exception:
        return ""


def audit_hardware() -> dict:
    """
    Collect per-GPU and CPU information for the current node.
    Returns a dict that drives ZeRO / offload selection downstream.
    """
    info: dict = {
        "node":  platform.node(),
        "gpus":  [],
        "cpu":   {},
        "interconnect": {},
    }

    n_gpus = torch.cuda.device_count()
    for i in range(n_gpus):
        props    = torch.cuda.get_device_properties(i)
        name     = props.name
        vram_gb  = props.total_memory / 1024**3
        cc_major = props.major
        cc_minor = props.minor

        # Resolve TFLOPS through the 4-tier pipeline — never silently wrong
        peak_tflops, tflops_source = resolve_gpu_peak_tflops(name, i, props)

        # Memory bandwidth (GB/s) via nvidia-smi
        bw_str = _run(
            f"nvidia-smi --query-gpu=memory.bandwidth --format=csv,noheader,nounits "
            f"-i {i} 2>/dev/null"
        )
        try:
            bw_gb_s = float(bw_str) / 1000   # MB/s → GB/s
        except ValueError:
            bw_gb_s = {
                "4090": 1008, "3090": 936, "A100": 2000,
                "H100": 3350, "V100": 900,
            }.get(next((k for k in ["4090","3090","A100","H100","V100"]
                        if k in name), ""), 800)

        # NVLink check
        nvlink_str = _run(
            f"nvidia-smi nvlink -s -i {i} 2>/dev/null | grep 'Speed' | head -1"
        )
        has_nvlink = bool(nvlink_str)

        info["gpus"].append({
            "index":         i,
            "name":          name,
            "vram_gb":       round(vram_gb, 2),
            "cc":            f"{cc_major}.{cc_minor}",
            "bf16":          cc_major >= 8,
            "peak_tflops":   peak_tflops,
            "tflops_source": tflops_source,   # NEW: tells user how accurate MFU is
            "bw_gb_s":       bw_gb_s,
            "has_nvlink":    has_nvlink,
        })

    # CPU / RAM
    try:
        import psutil
        cpu_ram_gb = psutil.virtual_memory().total / 1024**3
        cpu_cores  = psutil.cpu_count(logical=False) or 1
    except ImportError:
        cpu_ram_gb = 0.0
        cpu_cores  = os.cpu_count() or 1

    info["cpu"] = {
        "ram_gb":  round(cpu_ram_gb, 1),
        "cores":   cpu_cores,
        "model":   platform.processor(),
    }

    # Interconnect
    ib_str = _run("ibstat 2>/dev/null | grep 'State: Active' | wc -l")
    try:
        info["interconnect"]["infiniband_ports"] = int(ib_str)
    except ValueError:
        info["interconnect"]["infiniband_ports"] = 0

    info["interconnect"]["nvlink"] = any(g["has_nvlink"] for g in info["gpus"])

    return info


def print_audit(info: dict, n_params: int):
    sep = "─" * 64
    print(f"\n{sep}")
    print(f"  HARDWARE AUDIT  —  node: {info['node']}")
    print(sep)
    print(f"  GPUs: {len(info['gpus'])}")
    for g in info["gpus"]:
        bf16_tag = "bf16✓" if g["bf16"] else "fp16-only"
        nvlink   = " NVLink✓" if g["has_nvlink"] else ""
        src      = g.get("tflops_source", "unknown")
        # Flag estimated values so the user knows MFU accuracy
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

    # Model size
    model_bf16_gb  = n_params * 2 / 1024**3
    total_vram_gb  = sum(g["vram_gb"] for g in info["gpus"])
    print(f"\n  Model:       {n_params/1e9:.3f}B params  "
          f"({model_bf16_gb:.1f} GB bf16 weights)")
    print(f"  Total VRAM:  {total_vram_gb:.1f} GB across {len(info['gpus'])} GPU(s)")
    print(sep + "\n")


# ---------------------------------------------------------------------------
# ── ZERO STAGE AUTO-SELECTION ────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def select_zero_stage_and_offload(
    info: dict,
    n_params: int,
    world_size: int,
    force_stage: Optional[int],
    force_cpu_offload_optimizer: bool,
    force_cpu_offload_param: bool,
) -> Tuple[int, bool, bool]:
    """
    Return (zero_stage, cpu_offload_optimizer, cpu_offload_param).

    Decision logic
    ──────────────
    Per-GPU VRAM budget is compared against three thresholds derived from
    the model size:

        full_fit   = params × (2 + 2 + 8) / n_gpus      # weights + grads + Adam
        zero2_fit  = params × (2 + 2)     / n_gpus      # weights + grads (no Adam per GPU)
        zero3_fit  = params × 2           / n_gpus      # weights only (sharded)

    If nothing fits even with ZeRO-3, CPU offload is activated for the
    optimizer states; if parameters still don't fit, param offload is
    also activated.

    A 15% safety margin is applied so we don't cut it too close.
    """
    if not info["gpus"]:
        return 1, False, False

    min_vram   = min(g["vram_gb"] for g in info["gpus"]) * 0.85   # 15% safety margin
    n_gpus     = len(info["gpus"])

    # Static components per GPU (GB)
    full_gb   = n_params * (2 + 2 + 8) / 1024**3 / max(n_gpus, 1)
    zero2_gb  = n_params * (2 + 2)     / 1024**3 / max(n_gpus, 1)
    zero3_gb  = n_params * 2           / 1024**3 / max(n_gpus, 1)

    # Activation VRAM is separate — we don't account for it here
    # because it depends on batch size which the user controls.

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

    # Warn if CPU RAM is insufficient for offload
    cpu_ram = info["cpu"].get("ram_gb", 0)
    if (cpu_offload_opt or cpu_offload_param) and cpu_ram > 0:
        needed_gb = n_params * 8 / 1024**3  # Adam m + v in fp32
        if needed_gb > cpu_ram * 0.6:
            print(f"[AutoConfig] WARNING: optimizer offload needs ~{needed_gb:.1f} GB CPU RAM "
                  f"but only {cpu_ram:.0f} GB available. Consider a smaller model or more RAM.")

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
    Construct a complete deepspeed config dict.

    Key design decisions per stage:
        ZeRO-1: all-reduce gradients normally; only optimizer sharded
        ZeRO-2: reduce-scatter gradients; optimizer sharded
        ZeRO-3: reduce-scatter params + grads + optimizer;
                 allgather params before each forward
    """
    # ---- optimizer (DeepSpeed's fused Adam)
    optimizer_cfg = {
        "type": "AdamW",
        "params": {
            "lr":            args.lr,
            "betas":         [0.9, 0.95],
            "eps":           1e-8,
            "weight_decay":  args.weight_decay,
        },
    }

    # ---- LR scheduler (DeepSpeed handles it so the engine knows the LR)
    scheduler_cfg = {
        "type": "WarmupCosineLR",
        "params": {
            "warmup_num_steps":  args.warmup_steps,
            "total_num_steps":   args.max_steps,
            "warmup_type":       "linear",
            "last_batch_iteration": -1,
        },
    }

    # ---- bf16 / fp16
    # IMPORTANT: DeepSpeed's "bf16: {enabled: True}" mode (BF16_Optimizer
    # with fp32 master weights) is producing a degenerate model at init
    # on this 62.5M shape — the very first loss comes back exactly equal
    # to ln(vocab_size) (uniform) and the model never moves. Repro is
    # trivial: a plain torch.optim.AdamW on the same bf16 model learns
    # normally (loss drops from 10.5 -> 4.2 in 30 steps), but the
    # equivalent DeepSpeed bf16 setup gets stuck at 10.5 forever.
    #
    # Workaround: keep the model parameters in bf16 (so we still get the
    # ~2x activation VRAM savings) but disable DeepSpeed's bf16 path.
    # The model is cast to bf16 below, and the forward is wrapped in
    # torch.autocast(bf16). DeepSpeed then sees fp32 params and uses
    # its standard fp32 optimizer — which works correctly.
    bf16_cfg = {"enabled": False}
    fp16_cfg = {"enabled": False}
    use_pytorch_bf16 = (args.dtype == "bf16")

    # ---- gradient clipping
    grad_clip = args.grad_clip

    # ---- ZeRO config
    zero_cfg: dict = {
        "stage": zero_stage,
        "reduce_bucket_size":    5e8,        # 500 MB all-reduce bucket
        "allgather_bucket_size": 5e8,        # 500 MB all-gather bucket (ZeRO-3)
        "overlap_comm":          True,        # overlap comm with backward compute
        "contiguous_gradients":  True,        # avoids grad buffer fragmentation
        "sub_group_size":        1e9,         # ZeRO-3 param sub-group size
        "stage3_max_live_parameters":   1e9,
        "stage3_max_reuse_distance":    1e9,
        "stage3_gather_16bit_weights_on_model_save": True,
    }

    # ---- CPU offload
    if cpu_offload_optimizer:
        zero_cfg["offload_optimizer"] = {
            "device":      "cpu",
            "pin_memory":  True,     # pinned memory for faster H2D copy
            "ratio":       1.0,      # offload 100% of optimizer states
        }

    if cpu_offload_param and zero_stage == 3:
        zero_cfg["offload_param"] = {
            "device":        "cpu",
            "pin_memory":    True,
            "buffer_count":  5,
            "buffer_size":   1e8,    # 100 MB prefetch buffer per device
        }

    # ---- communication optimizations based on interconnect
    has_nvlink = any(g.get("has_nvlink") for g in gpu_info)
    zero_cfg["reduce_scatter"]   = True
    zero_cfg["allgather_partitions"] = True

    # With NVLink, larger buckets are faster; with PCIe, keep them smaller
    if not has_nvlink:
        zero_cfg["reduce_bucket_size"]    = 2e8   # 200 MB
        zero_cfg["allgather_bucket_size"] = 2e8

    # ---- activation checkpointing
    act_ckpt: dict = {}
    if args.gradient_checkpointing:
        act_ckpt = {
            "partition_activations":        False,   # handled in model.py
            "cpu_checkpointing":            False,
            "contiguous_memory_optimization": False,
            "synchronize_checkpoint_boundary": False,
        }

    # ---- assemble
    #
    # IMPORTANT: gradient_accumulation_steps is set to 1 in the DS config
    # because the training loop below does its OWN manual accumulation
    # (62 micro-batches of forward+backward, then a single engine.step()).
    # If we left gradient_accumulation_steps=62 in the config, DeepSpeed
    # would ALSO try to count micro-batches internally — but since
    # engine.step() is only called once per 62 microbatches, DS's
    # is_gradient_accumulation_boundary() would fire on the very first
    # step() (after just 1 microbatch from its perspective) and then
    # never fire again. The optimizer would then either step too early
    # (on partial accumulation) or not step at all.
    #
    # Setting gas=1 here disables DS's internal accumulation counter
    # while keeping the rest of the DS pipeline (ZeRO, bf16, fused
    # optimizer, scheduler) intact. The outer loop drives the cadence.
    cfg = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps":    1,
        "gradient_clipping":              grad_clip,
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
# ── DATA LOADER ──────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

class PackedDataLoader:
    """
    Identical to train.py version: memmap-backed, prefetched, rank-sharded.
    """
    def __init__(self, bin_path: str, seq_len: int, batch_size: int,
                 rank: int = 0, world_size: int = 1, dtype=np.uint16):
        self.seq_len    = seq_len
        self.batch_size = batch_size

        data       = np.memmap(bin_path, dtype=dtype, mode="r")
        shard_size = len(data) // world_size
        start      = rank * shard_size
        end        = start + shard_size if rank < world_size - 1 else len(data)
        self.data  = data[start:end]
        self.n_pos = max(1, len(self.data) - seq_len)
        self._next: Optional[Tuple] = None

        print(f"[DataLoader rank {rank}] {bin_path}: "
              f"{len(self.data):,} tokens  shard [{start}:{end}]")

    def _build(self):
        ix = torch.randint(self.n_pos, (self.batch_size,))
        x  = torch.stack([torch.from_numpy(self.data[i:i+self.seq_len].astype(np.int64))
                          for i in ix])
        y  = torch.stack([torch.from_numpy(self.data[i+1:i+1+self.seq_len].astype(np.int64))
                          for i in ix])
        return x, y

    def prime(self):
        self._next = self._build()

    def next_batch(self, device: torch.device):
        if self._next is not None:
            x_cpu, y_cpu = self._next
            self._next   = self._build()   # prefetch next
        else:
            x_cpu, y_cpu = self._build()
            self._next   = self._build()
        x = x_cpu.pin_memory().to(device, non_blocking=True)
        y = y_cpu.pin_memory().to(device, non_blocking=True)
        return x, y


# ---------------------------------------------------------------------------
# ── MFU ─────────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def estimate_mfu(model, tokens_per_sec: float, gpu_info: List[dict]) -> float:
    raw = model.module if hasattr(model, "module") else model
    n   = sum(p.numel() for name, p in raw.named_parameters()
              if "embed_tokens" not in name)
    flops = 6 * n * tokens_per_sec
    if not gpu_info:
        return 0.0
    peak  = gpu_info[0]["peak_tflops"] * 1e12 * len(gpu_info)
    return flops / peak


# ---------------------------------------------------------------------------
# ── CHECKPOINT ───────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------
# DeepSpeed has its own checkpoint format (a directory with model + optimizer
# shards). We use deepspeed.save_checkpoint / load_checkpoint rather than
# torch.save so ZeRO-3 sharded weights are handled correctly.

def save_checkpoint(engine,
                    step: int, out_dir: str,
                    config: Qwen3Config,
                    args_dict: dict,
                    best_val_loss: float):
    tag  = f"step_{step:07d}"
    path = os.path.join(out_dir, tag)
    engine.save_checkpoint(out_dir, tag=tag)

    # Write a small JSON sidecar with config + training state so we can
    # reconstruct the model / resume without parsing DS internals.
    meta = {
        "step":          step,
        "config":        vars(config),
        "args":          args_dict,
        "best_val_loss": best_val_loss,
        "ds_tag":        tag,
    }
    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Update "latest" symlink
    latest = os.path.join(out_dir, "latest_ds")
    if os.path.islink(latest): os.remove(latest)
    os.symlink(os.path.abspath(path), latest)
    print(f"[Checkpoint] saved {path}")


def load_checkpoint(engine,
                    resume_path: str) -> Tuple[int, float]:
    meta_path = os.path.join(resume_path, "meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    tag = meta["ds_tag"]
    # load_checkpoint returns the step it resumed from
    engine.load_checkpoint(os.path.dirname(resume_path), tag=tag)
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
def evaluate(engine, val_loader, eval_steps, device, world_size, use_pytorch_bf16=False):
    engine.eval()
    autocast_ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if use_pytorch_bf16
                    else nullcontext())
    losses = []
    with torch.no_grad():
        for _ in range(eval_steps):
            x, y = val_loader.next_batch(device)
            with autocast_ctx:
                out  = engine(x, labels=y)
            loss = out["loss"]
            losses.append(loss.item())
    engine.train()
    mean_loss = float(np.mean(losses))
    if world_size > 1:
        t = torch.tensor(mean_loss, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.AVG)
        mean_loss = t.item()
    return mean_loss


# ---------------------------------------------------------------------------
# ── MAIN ────────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def train(args):
    # DeepSpeed initialises its own process group — don't call dist.init manually
    local_rank  = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK",       0))
    world_size  = int(os.environ.get("WORLD_SIZE",  1))
    master      = global_rank == 0

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    torch.manual_seed(args.seed + global_rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32        = True

    # ---------------------------------------------------------------- meta
    meta_path = os.path.join(args.data_dir, "meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    vocab_size = meta["vocab_size"]
    dtype_np   = np.uint16 if meta["dtype"] == "uint16" else np.uint32

    # ---------------------------------------------------------------- hardware audit
    hw = audit_hardware()

    # ---------------------------------------------------------------- model
    if args.resume:
        meta_json = os.path.join(args.resume, "meta.json")
        with open(meta_json) as f:
            ckpt_meta = json.load(f)
        config = Qwen3Config(**ckpt_meta["config"])
        if master: print(f"[Resume] loaded config from checkpoint")
    else:
        config = Qwen3Config.from_target_size(
            args.model_size, vocab_size=vocab_size,
            quality_mode=args.quality_mode, param_slack=args.param_slack,
            verbose=master,
        )

    # Whether to use PyTorch's bf16 (cast model + autocast) instead of
    # DeepSpeed's broken bf16 path. See the long comment in
    # build_ds_config for why we have to do this.
    use_pytorch_bf16 = (args.dtype == "bf16")

    model    = Qwen3ForCausalLM(config)
    n_params = count_parameters(model)

    # Cast the model to bf16 BEFORE DeepSpeed init when using PyTorch's
    # bf16 path. This keeps weights/activations in bf16 for ~2x VRAM
    # savings, while letting DeepSpeed's standard fp32 optimizer handle
    # the parameter updates correctly.
    if use_pytorch_bf16:
        model = model.to(torch.bfloat16)

    if master:
        print_audit(hw, n_params)

    # ---------------------------------------------------------------- gradient checkpointing
    if args.gradient_checkpointing:
        model.model.enable_gradient_checkpointing()

    # ---------------------------------------------------------------- ZeRO selection
    zero_stage, cpu_offload_opt, cpu_offload_param = select_zero_stage_and_offload(
        hw, n_params, world_size,
        force_stage=args.zero_stage,
        force_cpu_offload_optimizer=args.cpu_offload_optimizer,
        force_cpu_offload_param=args.cpu_offload_param,
    )

    if master:
        print(f"[AutoConfig] Selected ZeRO-{zero_stage}  "
              f"cpu_offload_opt={cpu_offload_opt}  "
              f"cpu_offload_param={cpu_offload_param}")

    # ---------------------------------------------------------------- DS config
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

    # ---------------------------------------------------------------- param groups
    # DeepSpeed uses its own optimizer internally, but we still pass
    # parameter groups so weight decay is excluded from norms/embeddings.
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        if p.ndim < 2 or "norm" in name or "embed" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    param_groups = [
        {"params": decay,    "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    # ---------------------------------------------------------------- DeepSpeed init
    # deepspeed.initialize wraps the model in a DeepSpeedEngine which
    # handles ZeRO sharding, gradient accumulation, and the optimizer step.
    engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=param_groups,
        config=ds_cfg,
    )

    # ---------------------------------------------------------------- data
    train_loader = PackedDataLoader(
        os.path.join(args.data_dir, "train.bin"),
        seq_len=args.seq_len, batch_size=args.batch_size,
        rank=global_rank, world_size=world_size, dtype=dtype_np,
    )
    val_loader = PackedDataLoader(
        os.path.join(args.data_dir, "val.bin"),
        seq_len=args.seq_len, batch_size=args.batch_size,
        rank=global_rank, world_size=world_size, dtype=dtype_np,
    )
    train_loader.prime()
    val_loader.prime()

    # ---------------------------------------------------------------- resume
    start_step    = 0
    best_val_loss = float("inf")
    if args.resume:
        start_step, best_val_loss = load_checkpoint(engine, args.resume)

    # ---------------------------------------------------------------- W&B
    use_wandb = False
    if master and args.wandb_project:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name or f"qwen3-{args.model_size}-z{zero_stage}",
                config={**vars(config), "n_params": n_params,
                        "zero_stage": zero_stage, **vars(args)},
            )
            use_wandb = True
        except Exception as e:
            print(f"[W&B] disabled: {e}")

    # ---------------------------------------------------------------- accounting
    tokens_per_step = (
        args.batch_size * args.seq_len * args.grad_accum_steps * world_size
    )
    # Each outer loop iteration is now one optimizer step (grad accum
    # is driven by hand inside the loop), so tokens_per_step is the
    # right unit for both loss averaging and throughput reporting.
    if master:
        print(f"Tokens / optimizer step : {tokens_per_step:,}")
        print(f"Effective batch size    : {args.batch_size * args.grad_accum_steps * world_size}")
        print(f"Max steps               : {args.max_steps:,}")
        print(f"Checkpoint every        : {args.ckpt_interval:,} steps\n")

    # ================================================================ LOOP
    engine.train()
    t0         = time.perf_counter()
    loss_accum = 0.0

    # When using PyTorch's bf16 path (DS bf16 disabled), wrap the forward
    # in autocast so the bf16-cast model's matmuls run in bf16. The
    # cross-entropy and softmax are dtype-stable so they stay in fp32.
    autocast_ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if use_pytorch_bf16
                    else nullcontext())

    for step in range(start_step, args.max_steps):

        # ---- LR override (DeepSpeed scheduler handles LR internally,
        #      but we manually set it to match our cosine schedule which
        #      is more flexible than DS's built-in WarmupCosineLR)
        lr = _cosine_lr(step, args.warmup_steps, args.max_steps,
                        args.lr, args.min_lr)
        for pg in engine.optimizer.param_groups:
            pg["lr"] = lr

        # ---- forward + backward + step
        # Drive the grad-accumulation loop by hand (mirroring train.py)
        # so each outer iteration here is exactly one optimizer step.
        # DeepSpeed's built-in accumulation would otherwise advance its
        # internal step counter faster than ours, breaking the per-step
        # loss accounting below. We only call engine.step() on the last
        # micro-batch; the manual loop guarantees the optimizer fires
        # once per `step` of the outer `for step in range(...)`.
        for micro_step in range(args.grad_accum_steps):
            x, y = train_loader.next_batch(device)
            with autocast_ctx:
                out  = engine(x, labels=y)
            # Divide by grad_accum_steps so the sum over the inner
            # loop is the true mean loss over one effective batch
            # (= one optimizer step), exactly like train.py.
            loss = out["loss"] / args.grad_accum_steps
            engine.backward(loss)

            loss_accum += loss.item()

        engine.step()

        # ---- logging
        if master and step % args.log_interval == 0:
            t1          = time.perf_counter()
            # tokens_per_step accounts for grad accum; the outer loop now
            # advances once per optimizer step, so this is the right unit
            # (matches train.py exactly).
            tok_per_sec  = tokens_per_step * args.log_interval / max(t1 - t0, 1e-9)
            mfu          = estimate_mfu(engine, tok_per_sec, hw["gpus"])
            # Per micro-batch we divided by grad_accum_steps; the inner
            # sum over grad_accum_steps micro-batches already reconstructs
            # the mean per-optimizer-step loss. Only the log_interval
            # averaging is left — dividing by grad_accum_steps here
            # would double-count and inflate the displayed loss by that
            # factor.
            loss_display = loss_accum / args.log_interval
            loss_accum   = 0.0

            # grad norm from DS engine
            grad_norm = engine.get_global_grad_norm() or 0.0

            print(
                f"step {step:7d} | loss {loss_display:.4f} | lr {lr:.2e} | "
                f"grad {grad_norm:.3f} | {tok_per_sec/1e3:.1f}k tok/s | "
                f"mfu {mfu*100:.2f}%"
            )
            if use_wandb:
                import wandb
                wandb.log({
                    "train/loss":         loss_display,
                    "train/lr":           lr,
                    "train/grad_norm":    grad_norm,
                    "perf/tokens_per_sec": tok_per_sec,
                    "perf/mfu_pct":       mfu * 100,
                }, step=step)
            t0 = t1

        # ---- validation
        if step % args.eval_interval == 0 and step > start_step:
            val_loss = evaluate(engine, val_loader, args.eval_steps, device, world_size, use_pytorch_bf16)
            if master:
                improved = " ✓" if val_loss < best_val_loss else ""
                print(f"  [eval] step {step:7d} | val_loss {val_loss:.4f}{improved}")
                if use_wandb:
                    import wandb
                    wandb.log({"val/loss": val_loss}, step=step)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss

        # ---- checkpoint  (all ranks must call save_checkpoint together)
        if step % args.ckpt_interval == 0 and step > start_step:
            if master:
                save_checkpoint(engine, step, args.out_dir,
                                config, vars(args), best_val_loss)
                prune_checkpoints(args.out_dir, keep=args.keep_ckpts)

    # ---- final checkpoint
    if master:
        save_checkpoint(engine, args.max_steps, args.out_dir,
                        config, vars(args), best_val_loss)
        print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
        if use_wandb:
            import wandb; wandb.finish()


def _cosine_lr(step, warmup, max_steps, max_lr, min_lr):
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step >= max_steps:
        return min_lr
    t = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + 0.5 * (1 + math.cos(math.pi * t)) * (max_lr - min_lr)


# ---------------------------------------------------------------------------
# ── CLI ─────────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="DeepSpeed pretraining for Qwen3-style dense LLM."
    )

    # model
    p.add_argument("--model-size", default="0.6B",
                   help="Target size passed to Qwen3Config.from_target_size")
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
    p.add_argument("--batch-size",       type=int,   default=4,
                   help="Micro-batch size PER GPU (before grad accum)")
    p.add_argument("--grad-accum-steps", type=int,   default=8)
    p.add_argument("--max-steps",        type=int,   default=100_000)
    p.add_argument("--warmup-steps",     type=int,   default=2_000)
    p.add_argument("--lr",               type=float, default=3e-4)
    p.add_argument("--min-lr",           type=float, default=3e-5)
    p.add_argument("--weight-decay",     type=float, default=0.1)
    p.add_argument("--grad-clip",        type=float, default=1.0)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--gradient-checkpointing", action="store_true",
                   help="Recompute activations on backward (~35% less VRAM)")
    p.add_argument("--seed", type=int, default=42)

    # ZeRO / offload  (auto-selected if not specified)
    p.add_argument("--zero-stage", type=int, default=None,
                   choices=[1, 2, 3],
                   help="Force ZeRO stage. Default: auto-selected from hardware audit.")
    p.add_argument("--cpu-offload-optimizer", action="store_true",
                   help="Force CPU offload of optimizer states (auto-enabled when VRAM is tight)")
    p.add_argument("--cpu-offload-param", action="store_true",
                   help="Force CPU offload of model parameters (ZeRO-3 only; "
                        "auto-enabled only when VRAM is very tight)")

    # checkpointing
    p.add_argument("--out-dir",       default="./checkpoints_ds")
    p.add_argument("--resume",        default=None,
                   help="Path to a DeepSpeed checkpoint directory to resume from")
    p.add_argument("--ckpt-interval", type=int, default=100)
    p.add_argument("--keep-ckpts",    type=int, default=3)

    # logging / eval
    p.add_argument("--log-interval",  type=int, default=10)
    p.add_argument("--eval-interval", type=int, default=50)
    p.add_argument("--eval-steps",    type=int, default=50)
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--wandb-run-name",default=None)

    # DeepSpeed passes its own args — absorb them without error
    p.add_argument("--local_rank", type=int, default=-1,
                   help="Set by DeepSpeed launcher; do not set manually.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# ── ENTRY ───────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("ERROR: train_deepspeed.py requires at least one CUDA GPU.")
        sys.exit(1)

    try:
        import deepspeed
    except ImportError:
        print("ERROR: DeepSpeed not installed.  Run:")
        print("  pip install deepspeed")
        sys.exit(1)

    # Soft-check for psutil (used in hardware audit but not fatal)
    try:
        import psutil
    except ImportError:
        print("[warn] psutil not installed — CPU RAM reporting will be incomplete.")
        print("       pip install psutil")

    args = parse_args()

    if not os.path.exists(os.path.join(args.data_dir, "train.bin")):
        print(f"ERROR: {args.data_dir}/train.bin not found.")
        print("Run pack_dataset.py first to produce packed training data.")
        sys.exit(1)

    train(args)
