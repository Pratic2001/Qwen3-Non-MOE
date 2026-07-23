#!/usr/bin/env python3
"""
train_grpo_deepspeed.py

DeepSpeed-powered GRPO (Group Relative Policy Optimization) for the
Qwen3-style dense model from `model.py`. Mirror of `train_grpo.py` that
replaces DDP + AdamW with the DeepSpeed engine so the same training loop
scales across multi-GPU / multi-node hardware with auto-selected ZeRO
stages and optional CPU offload.

The training algorithm is identical to `train_grpo.py`:

    1. Sample a batch of prompts.
    2. Roll out G completions per prompt with the current policy.
    3. Score each completion with the rule-based reward function
       (correctness + format bonus — option A).
    4. Compute per-prompt advantages by group-normalising rewards
       within the G rollouts.
    5. PPO-style clipped policy gradient on token-level log-probs, with
       an optional KL penalty against a reference policy.

This script reuses the GRPO primitives (dataset, rollout generator,
reward function, GRPO loss) verbatim from `train_grpo.py` and only
swaps the distribution/optimizer layer for DeepSpeed.

Launch:
    # Single node, 1 GPU
    deepspeed --num_gpus 1 train_grpo_deepspeed.py --checkpoint ./sft_checkpoints/latest.pt

    # Single node, 4 GPUs
    deepspeed --num_gpus 4 train_grpo_deepspeed.py --checkpoint ./sft_checkpoints/latest.pt

    # Multi-node
    deepspeed --hostfile hostfile train_grpo_deepspeed.py --checkpoint ...

    # Force a specific ZeRO stage
    deepspeed train_grpo_deepspeed.py --checkpoint ... --zero-stage 3 \\
        --cpu-offload-optimizer

    # LoRA GRPO
    deepspeed train_grpo_deepspeed.py --checkpoint ... --lora \\
        --lora-rank 64 --lora-alpha 128
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.distributed as dist
from tokenizers import Tokenizer

import deepspeed

from model import Qwen3Config, Qwen3ForCausalLM, count_parameters

# Re-use the GRPO machinery so this script stays focused on the engine swap.
from train_grpo import (
    compute_reward,
    GRPOPromptDataset,
    generate_rollouts,
    compute_logprobs,
    grpo_loss,
    build_reference,
    save_checkpoint,
    load_checkpoint,
    merge_and_save,
    smoke_test,
)

# Re-use the SFT machinery for LoRA + utilities.
from train_sft import (
    inject_lora,
    lora_parameter_count,
    load_tokenizer,
)

# Re-use train_deepspeed.py's hardware audit + ZeRO selection plumbing so
# the choice of stage / offload is consistent with the pretraining script.
from train_deepspeed import (
    audit_hardware,
    print_audit,
    select_zero_stage_and_offload,
)


# ---------------------------------------------------------------------------
# DeepSpeed config builder (GRPO-specific)
# ---------------------------------------------------------------------------
#
# Mirrors train_deepspeed.build_ds_config but uses GRPO-relevant defaults
# (no grad-accum, no eval, simple cosine schedule with our own scheduler).
# GRPO's per-step loss is a single PPO-style scalar; the DeepSpeed engine
# still drives ZeRO sharding + optimizer step + gradient clipping.

def build_ds_config(
    args,
    zero_stage: int,
    cpu_offload_optimizer: bool,
    cpu_offload_param: bool,
    gpu_info: List[dict],
) -> dict:
    """Construct a complete deepspeed config dict for GRPO."""

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

    # ---- LR: no DS "scheduler" block. We override LR each step manually
    # to match train_grpo.py's cosine schedule. A DS scheduler block is
    # not just "informational" — it's live, checkpointed state that DS
    # calls internally on every engine.step(); leaving it out avoids that
    # entirely rather than relying on it staying harmless.

    # ---- bf16 / fp16
    bf16_cfg = {"enabled": args.dtype == "bf16"}
    fp16_cfg = {"enabled": False}

    # ---- gradient clipping
    grad_clip = args.grad_clip

    # ---- ZeRO config
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
            "device":      "cpu",
            "pin_memory":  True,
            "ratio":       1.0,
        }

    if cpu_offload_param and zero_stage == 3:
        zero_cfg["offload_param"] = {
            "device":        "cpu",
            "pin_memory":    True,
            "buffer_count":  5,
            "buffer_size":   1e8,
        }

    has_nvlink = any(g.get("has_nvlink") for g in gpu_info)
    zero_cfg["reduce_scatter"]      = True
    zero_cfg["allgather_partitions"] = True
    if not has_nvlink:
        zero_cfg["reduce_bucket_size"]    = 2e8
        zero_cfg["allgather_bucket_size"] = 2e8

    # ---- assemble
    cfg = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps":    1,    # GRPO: 1 micro-step per "step"
        "gradient_clipping":              grad_clip,
        "steps_per_print":                args.log_interval,
        "wall_clock_breakdown":           False,
        "optimizer":                      optimizer_cfg,
        "bf16":                           bf16_cfg,
        "fp16":                           fp16_cfg,
        "zero_optimization":              zero_cfg,
    }
    return cfg


def print_ds_config_summary(cfg: dict, zero_stage: int,
                             cpu_opt: bool, cpu_param: bool):
    sep = "─" * 64
    print(f"\n{sep}")
    print(f"  DEEPSPEED CONFIG SUMMARY (GRPO)")
    print(sep)
    print(f"  ZeRO Stage            : {zero_stage}")
    print(f"  CPU offload optimizer : {cpu_opt}")
    print(f"  CPU offload params    : {cpu_param}")
    print(f"  BF16                  : {cfg['bf16']['enabled']}")
    print(f"  Micro batch / GPU     : {cfg['train_micro_batch_size_per_gpu']}")
    print(f"  Grad clip             : {cfg['gradient_clipping']}")
    z = cfg["zero_optimization"]
    print(f"  Reduce bucket         : {z['reduce_bucket_size']/1e6:.0f} MB")
    print(f"  Overlap comm          : {z['overlap_comm']}")
    print(sep + "\n")


# ---------------------------------------------------------------------------
# GRPO-specific DeepSpeed checkpoint helpers
# ---------------------------------------------------------------------------
#
# DeepSpeed's engine.save_checkpoint() handles ZeRO sharded weights. We
# piggy-back on engine.save_checkpoint but also need to save the
# train_grpo.py-style metadata (config, args, lora state) so the
# downstream merge_lora path keeps working unchanged.
#
# For LoRA checkpoints the model is small enough that saving the LoRA
# adapters through the train_grpo.save_checkpoint() helper is much
# simpler — and keeps the .pt format that the existing merge tooling
# consumes. We detect this case in main() and branch on it.

def save_ds_checkpoint(
    engine,
    out_dir: str,
    step: int,
    config: Qwen3Config,
    args_dict: dict,
):
    """Save a DeepSpeed checkpoint (full model) at `step`."""
    tag  = f"step_{step:07d}"
    path = os.path.join(out_dir, tag)
    engine.save_checkpoint(out_dir, tag=tag)

    # Sidecar metadata so we can resume / inspect.
    meta = {
        "step":   step,
        "config": vars(config),
        "args":   args_dict,
        "ds_tag": tag,
    }
    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    latest = os.path.join(out_dir, "latest_ds")
    if os.path.islink(latest): os.remove(latest)
    os.symlink(os.path.abspath(path), latest)
    if dist.get_rank() == 0:
        print(f"[Checkpoint] saved {path}")
    return path


def load_ds_checkpoint(engine, resume_path: str) -> int:
    """Load a DeepSpeed checkpoint and return the step we resumed from."""
    meta_path = os.path.join(resume_path, "meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    tag = meta["ds_tag"]
    engine.load_checkpoint(os.path.dirname(resume_path), tag=tag)
    step = meta.get("step", 0)
    if dist.get_rank() == 0:
        print(f"[Checkpoint] resumed from {resume_path} at step {step}")
    return step


# ---------------------------------------------------------------------------
# Cosine LR (matches train_grpo.py)
# ---------------------------------------------------------------------------

def _cosine_lr(step, warmup, max_steps, max_lr, min_lr):
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step >= max_steps:
        return min_lr
    t = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + 0.5 * (1 + math.cos(math.pi * t)) * (max_lr - min_lr)


# ---------------------------------------------------------------------------
# Main training loop
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
    rng = random.Random(args.seed + global_rank)

    # ----------------------------------------------------------------- ckpt
    if not args.checkpoint:
        raise FileNotFoundError(
            "--checkpoint is required. Point it at the SFT checkpoint "
            "produced by train_sft.py."
        )
    ckpt_data = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config    = Qwen3Config(**ckpt_data["config"])

    # ---------------------------------------------------------------- hardware audit
    hw = audit_hardware()

    # ---------------------------------------------------------------- model
    # Build on CPU first; the DeepSpeed engine moves shards to GPU.
    model    = Qwen3ForCausalLM(config)
    n_params = count_parameters(model)
    if master:
        print_audit(hw, n_params)
        print(f"Loaded SFT checkpoint: {n_params:,} params ({n_params/1e9:.3f}B)")

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

    # ----------------------------------------------------------------- ZeRO
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

    # ----------------------------------------------------------------- DS config
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

    # ----------------------------------------------------------------- ref
    # Build the reference model on CPU (two-model variant only). For
    # --ref_policy single, we reuse the trainable model under no_grad
    # inside the loop, so no second model is allocated here.
    if args.ref_policy == "two":
        ref_model = build_reference("two", config, args.checkpoint, device)
    else:
        ref_model = None
    ref_for_logprob = ref_model if ref_model is not None else model  # bound below after engine init

    # ----------------------------------------------------------------- param groups
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

    # ----------------------------------------------------------------- DeepSpeed init
    # deepspeed.initialize wraps the model in a DeepSpeedEngine which
    # handles ZeRO sharding, gradient accumulation, and the optimizer step.
    engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=param_groups,
        config=ds_cfg,
    )

    # Now bind the reference model: when --ref_policy single, we reuse the
    # trainable model (still the un-wrapped underlying Qwen3ForCausalLM).
    # engine.module is the underlying model; engine.module() is the same.
    if ref_model is None:
        ref_for_logprob = engine.module

    # ----------------------------------------------------------------- tokenizer
    tokenizer = load_tokenizer(args.tokenizer)
    try:
        eos_id = get_special_token_id_safe(tokenizer, "<|endoftext|>")
        pad_id = tokenizer.token_to_id("<|pad|>") or 0
    except Exception:
        eos_id = tokenizer.get_vocab_size() - 1
        pad_id = 0
    if master:
        print(f"[Tokenizer] eos_id={eos_id}, pad_id={pad_id}, "
              f"vocab={tokenizer.get_vocab_size()}")

    # ----------------------------------------------------------------- dataset
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

    # ----------------------------------------------------------------- resume
    start_step = 0
    if args.resume:
        if args.resume.endswith(".pt"):
            # train_grpo.py-style checkpoint (LoRA or full)
            start_step = load_checkpoint(args.resume, engine, optimizer, device, is_lora)
        else:
            # DeepSpeed checkpoint directory
            start_step = load_ds_checkpoint(engine, args.resume)

    if master:
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
    # Note: engine.train() / engine.eval() toggle autograd. Rollouts are
    # generated under no_grad in generate_rollouts() so we don't have to
    # call .eval() before each rollout. We DO call .eval() to silence
    # autograd noise on the rollout forward (no different to train_grpo).
    engine.train()
    t0 = time.perf_counter()
    reward_window: List[float] = []
    correct_window: List[int]   = []
    think_window: List[int]     = []

    # Unwrap helper for the underlying model (used for the rollout
    # forward, which is a plain Qwen3ForCausalLM call).
    def _underlying():
        return engine.module

    last_loss = torch.tensor(0.0)

    for step in range(start_step, args.max_steps):
        # ---- LR (DeepSpeed scheduler is just a placeholder; we move LR
        #      on every step to match train_grpo.py's cosine schedule)
        lr = _cosine_lr(step, args.warmup_steps, args.max_steps,
                        args.lr, args.min_lr)
        for pg in engine.optimizer.param_groups:
            pg["lr"] = lr

        # 1. sample prompts (only on master to keep RNG state consistent;
        #    every rank needs the same prompts to keep group-relative
        #    advantages valid — but in single-rank training this is
        #    trivially satisfied; in multi-rank we broadcast below)
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

        # 3. rollout  (uses the underlying unwrapped model)
        rollout_model = _underlying()
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

        # 6. policy log-probs (with grad) — engine drives the forward +
        #    backward + (optional) all-reduce + optimizer step in one go.
        out = engine(full_ids)
        # engine() returns a dict with 'logits' for Qwen3ForCausalLM;
        # under ZeRO-3 the output is materialized after param gather.
        if isinstance(out, dict):
            policy_logits = out["logits"]
        else:
            # Some wrappers return ModelOutput; fall back to attribute access.
            policy_logits = out.logits if hasattr(out, "logits") else out["logits"]
        policy_logits = policy_logits[:, :-1, :].float()
        targets = full_ids[:, 1:]
        policy_logp = policy_logits.log_softmax(dim=-1).gather(
            -1, targets.unsqueeze(-1)).squeeze(-1)
        T = gen_mask.shape[1]
        policy_logp = policy_logp[:, -T:] * gen_mask

        # 7. GRPO loss
        loss, metrics = grpo_loss(
            policy_logp, ref_logp, rewards, gen_mask,
            group_size=args.num_generations,
            kl_coef=args.kl_coef,
            clip_ratio=args.clip_ratio,
        )

        # 8. DeepSpeed step (handles backward + ZeRO all-reduce + optimizer)
        engine.backward(loss)
        engine.step()

        if device.type == "cuda":
            torch.cuda.synchronize()
        last_loss = loss.detach()

        # 9. log
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
                grad_norm = engine.get_global_grad_norm() or 0.0
                print(
                    f"step {step:6d} | loss {last_loss.item():+.4f} | "
                    f"pg {metrics['pg']:+.4f} | kl {metrics['kl']:+.5f} | "
                    f"r̄ {r_mean:.2f} | acc {c_mean:.0%} | fmt {f_mean:.0%} | "
                    f"lr {lr:.2e} | g {grad_norm:.2f} | {sps:.2f} step/s"
                )

        # 10. checkpoint
        # LoRA checkpoints: write a .pt via train_grpo.save_checkpoint so
        # the existing merge_lora path works unchanged.
        # Full-FT checkpoints: write a DeepSpeed directory.
        if step > start_step and step % args.ckpt_interval == 0:
            if master:
                if is_lora:
                    save_checkpoint(args.out_dir, step, engine, engine.optimizer,
                                    config, vars(args), is_lora=True)
                else:
                    save_ds_checkpoint(engine, args.out_dir, step, config, vars(args))
                prune_checkpoints_ds(args.out_dir, keep=args.keep_ckpts, is_lora=is_lora)

    # ---- final checkpoint
    if master:
        if is_lora:
            save_checkpoint(args.out_dir, args.max_steps, engine, engine.optimizer,
                            config, vars(args), is_lora=True)
        else:
            save_ds_checkpoint(engine, args.out_dir, args.max_steps, config, vars(args))
        print(f"\nGRPO complete. Final loss: {last_loss.item():.4f}")
        if is_lora:
            print(f"\nTo merge LoRA into base weights:")
            print(f"  python train_grpo.py --merge_lora "
                  f"--checkpoint {args.out_dir}/latest.pt --out_dir ./grpo_merged")


def prune_checkpoints_ds(out_dir: str, keep: int = 3, is_lora: bool = False):
    """
    Mirror train_deepspeed.prune_checkpoints but knows about both .pt
    (LoRA) and step_* (DeepSpeed) directories.
    """
    if is_lora:
        # Mirror train_grpo.prune_checkpoints (keeps newest N .pt files)
        files = sorted(
            [f for f in Path(out_dir).iterdir()
             if f.is_file() and f.name.startswith("grpo_step") and f.suffix == ".pt"],
            key=lambda f: int(f.name.replace("grpo_step", "").replace(".pt", "")),
        )
        for old in files[:-keep]:
            old.unlink()
            print(f"[Checkpoint] pruned {old.name}")
    else:
        # Mirror train_deepspeed.prune_checkpoints (keeps newest N DS dirs)
        dirs = sorted(
            [d for d in Path(out_dir).iterdir()
             if d.is_dir() and d.name.startswith("step_")],
            key=lambda d: int(d.name.replace("step_", "")),
        )
        for old in dirs[:-keep]:
            import shutil
            shutil.rmtree(old)
            print(f"[Checkpoint] pruned {old.name}")


def get_special_token_id_safe(tokenizer: Tokenizer, name: str) -> int:
    """
    Same behaviour as pack_sft_data.get_special_token_id but inlined
    here so we don't add a hard import dependency on the
    (optional, GRPO-flavored) pack_grpo_data module.
    """
    tid = tokenizer.token_to_id(name)
    if tid is None or tid < 0:
        # Fall back to the last vocab id (matches the existing error path
        # in train_grpo.py when pack_sft_data isn't importable).
        return tokenizer.get_vocab_size() - 1
    return tid


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    """
    Compose train_grpo.py's CLI (so every existing flag keeps working) with
    train_deepspeed.py's ZeRO / offload / DeepSpeed-launcher flags.
    """
    p = argparse.ArgumentParser(
        description="DeepSpeed GRPO RL fine-tuning for Qwen3-style dense LLM.",
    )

    # Mode
    p.add_argument("--merge_lora", action="store_true",
                   help="Merge LoRA weights into base model and save; skip training")

    # Paths
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--tokenizer",  default="./tokenizer")
    p.add_argument("--cache_dir",  default="./sft_packed")
    p.add_argument("--data_dir",   default="./sft_data")
    p.add_argument("--prompts_file", default=None)
    p.add_argument("--prompt_override", default=None)
    p.add_argument("--out_dir",    default="./grpo_checkpoints")
    p.add_argument("--resume",     default=None)

    # LoRA
    p.add_argument("--lora",       action="store_true")
    p.add_argument("--lora_rank",  type=int,   default=64)
    p.add_argument("--lora_alpha", type=float, default=128.0)

    # Reference policy
    p.add_argument("--ref_policy", default="single", choices=["single", "two"])

    # Rollouts
    p.add_argument("--num_generations", type=int,   default=8)
    p.add_argument("--max_new_tokens",  type=int,   default=2048)
    p.add_argument("--temperature",     type=float, default=1.0)
    p.add_argument("--top_p",           type=float, default=0.95)
    p.add_argument("--max_prompt_len",  type=int,   default=4096)

    # Reward weights
    p.add_argument("--reward_correct",  type=float, default=1.0)
    p.add_argument("--reward_format",   type=float, default=0.3)

    # GRPO loss
    p.add_argument("--kl_coef",     type=float, default=0.02)
    p.add_argument("--clip_ratio",  type=float, default=0.2)

    # Optim
    p.add_argument("--batch_size",       type=int,   default=4,
                   help="Number of PROMPTS per step (rollouts = batch_size * G)")
    p.add_argument("--max_steps",        type=int,   default=500)
    p.add_argument("--warmup_steps",     type=int,   default=20)
    p.add_argument("--lr",               type=float, default=1e-6)
    p.add_argument("--min_lr",           type=float, default=1e-7)
    p.add_argument("--weight_decay",     type=float, default=0.0)
    p.add_argument("--grad_clip",        type=float, default=1.0)
    p.add_argument("--dtype",   default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--seed",    type=int, default=42)

    # Logging / checkpointing
    p.add_argument("--log_interval",  type=int, default=1)
    p.add_argument("--ckpt_interval", type=int, default=50)
    p.add_argument("--keep_ckpts",    type=int, default=3)

    # ZeRO / offload  (auto-selected if not specified)
    p.add_argument("--zero-stage", type=int, default=None, choices=[1, 2, 3],
                   help="Force ZeRO stage. Default: auto-selected from hardware audit.")
    p.add_argument("--cpu-offload-optimizer", action="store_true",
                   help="Force CPU offload of optimizer states (auto-enabled when VRAM is tight)")
    p.add_argument("--cpu-offload-param", action="store_true",
                   help="Force CPU offload of model parameters (ZeRO-3 only)")

    # DeepSpeed launcher
    p.add_argument("--local_rank", type=int, default=-1,
                   help="Set by DeepSpeed launcher; do not set manually.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("ERROR: train_grpo_deepspeed.py requires at least one CUDA GPU.")
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
