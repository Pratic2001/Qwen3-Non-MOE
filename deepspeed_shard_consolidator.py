#!/usr/bin/env python3
"""
deepspeed_shard_consolidator.py

Converts a DeepSpeed checkpoint directory produced by train_deepspeed.py into
a single consolidated .pt file that sft.py can load directly via --checkpoint.

Background
----------
train_deepspeed.py saves checkpoints using engine.save_checkpoint(), which
produces a *directory* layout like:

    <out_dir>/
        step_0100000/
            meta.json                           ← training state + Qwen3Config
            mp_rank_00_model_states.pt          ← full model weights (ZeRO-1/2)
            zero_pp_rank_0_mp_rank_00_optim_states.pt
            zero_pp_rank_1_mp_rank_00_optim_states.pt
            ...
        latest_ds -> step_0100000/              ← symlink

For ZeRO-3 the model parameters are *also* sharded across ranks and must be
gathered before saving.  train_deepspeed.py sets:

    "stage3_gather_16bit_weights_on_model_save": True

which makes DeepSpeed write the full (gathered) fp16/bf16 weights into
mp_rank_00_model_states.pt even for ZeRO-3, so the same consolidation logic
works for all three stages.

sft.py expects a checkpoint with the keys:
    {
        "model_state":   <OrderedDict>   # model.state_dict() — full weights
        "config":        <dict>          # vars(Qwen3Config(...))
        "step":          <int>           # optional, for bookkeeping
        "best_val_loss": <float>         # optional, for bookkeeping
        "args":          <dict>          # optional, original training args
    }

Usage
-----
    # Consolidate the latest checkpoint
    python deepspeed_shard_consolidator.py \\
        --ds-dir ./checkpoints_ds/latest_ds \\
        --out    ./checkpoints/pretrained.pt

    # Consolidate a specific step
    python deepspeed_shard_consolidator.py \\
        --ds-dir ./checkpoints_ds/step_0100000 \\
        --out    ./checkpoints/pretrained_step100k.pt

    # Scan an output directory and consolidate the latest step automatically
    python deepspeed_shard_consolidator.py \\
        --ds-out-dir ./checkpoints_ds \\
        --out        ./checkpoints/pretrained.pt

    # Convert to fp32 (default keeps original dtype, usually bf16)
    python deepspeed_shard_consolidator.py \\
        --ds-dir ./checkpoints_ds/latest_ds \\
        --out    ./checkpoints/pretrained.pt \\
        --dtype  fp32

    # Dry-run: inspect checkpoint contents without writing
    python deepspeed_shard_consolidator.py \\
        --ds-dir ./checkpoints_ds/latest_ds \\
        --dry-run
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_ds_dir(args) -> Path:
    """
    Return the specific DeepSpeed step directory to consolidate.

    Priority:
      1. --ds-dir  (explicit path to a step_XXXXXXX dir or symlink)
      2. --ds-out-dir + auto-detection of latest step_* subdirectory
    """
    if args.ds_dir:
        p = Path(args.ds_dir).resolve()
        if not p.exists():
            sys.exit(f"[ERROR] --ds-dir does not exist: {p}")
        return p

    if args.ds_out_dir:
        root = Path(args.ds_out_dir).resolve()
        if not root.exists():
            sys.exit(f"[ERROR] --ds-out-dir does not exist: {root}")

        # Check for latest_ds symlink first
        latest_link = root / "latest_ds"
        if latest_link.is_symlink() or latest_link.exists():
            return latest_link.resolve()

        # Otherwise pick the highest-numbered step directory
        step_dirs = sorted(
            [d for d in root.iterdir() if d.is_dir() and d.name.startswith("step_")],
            key=lambda d: int(d.name.replace("step_", "")),
        )
        if not step_dirs:
            sys.exit(f"[ERROR] No step_* directories found under {root}")
        chosen = step_dirs[-1]
        print(f"[Info] Auto-selected checkpoint: {chosen.name}")
        return chosen

    sys.exit("[ERROR] Provide either --ds-dir or --ds-out-dir.")


def _load_meta(ds_dir: Path) -> dict:
    meta_path = ds_dir / "meta.json"
    if not meta_path.exists():
        sys.exit(
            f"[ERROR] meta.json not found in {ds_dir}\n"
            f"        Was this checkpoint saved by train_deepspeed.py?"
        )
    with open(meta_path) as f:
        return json.load(f)


def _find_model_states_file(ds_dir: Path) -> Path:
    """
    Locate the model-states shard file.

    DeepSpeed names it:  mp_rank_<XX>_model_states.pt
    For single-pipeline-rank jobs this is always mp_rank_00_model_states.pt.
    We look for all matching files and prefer rank 00.
    """
    candidates = sorted(ds_dir.glob("mp_rank_*_model_states.pt"))
    if not candidates:
        sys.exit(
            f"[ERROR] No mp_rank_*_model_states.pt found in {ds_dir}\n"
            f"        Expected files:\n"
            f"            {ds_dir}/mp_rank_00_model_states.pt\n"
            f"        Make sure DeepSpeed saved successfully and all ranks finished."
        )
    if len(candidates) > 1:
        print(f"[Warn]  Found {len(candidates)} model-state files; using {candidates[0].name}")
    return candidates[0]


def _extract_state_dict(model_states_path: Path, target_dtype: Optional[torch.dtype]) -> dict:
    """
    Load the DeepSpeed model-states file and extract a clean state_dict.

    DeepSpeed stores the consolidated weights under the key 'module' inside
    the model-states file.  For ZeRO-3 with
        "stage3_gather_16bit_weights_on_model_save": True
    DeepSpeed gathers and stores the full weights in this same file so no
    extra all-gather is needed offline.

    The keys in 'module' have a 'module.' prefix added by the DeepSpeedEngine
    wrapper, which we strip to match the bare Qwen3ForCausalLM state_dict.
    """
    print(f"[Load]  {model_states_path}  ({model_states_path.stat().st_size / 1e9:.2f} GB)")
    raw = torch.load(model_states_path, map_location="cpu", weights_only=False)

    # Identify which key holds the weights
    # DeepSpeed engine wraps the model, so parameters live under "module"
    if "module" in raw:
        sd = raw["module"]
    elif "model_state_dict" in raw:
        # Some DS versions use this key
        sd = raw["model_state_dict"]
    else:
        # Fall back: assume the whole dict is the state_dict
        # (happens when DS writes a pre-consolidated fp16 state)
        sd = raw
        print("[Warn]  'module' key not found — treating entire file as state_dict.")

    # Strip the 'module.' prefix that DeepSpeedEngine adds
    clean = {}
    for k, v in sd.items():
        new_key = k[len("module."):] if k.startswith("module.") else k
        # Also strip torch.compile _orig_mod prefix if present
        if new_key.startswith("_orig_mod."):
            new_key = new_key[len("_orig_mod."):]
        if target_dtype is not None and v.is_floating_point():
            v = v.to(target_dtype)
        clean[new_key] = v

    return clean


def _verify_state_dict(sd: dict, config: dict):
    """
    Sanity-check the extracted state dict against the model config.
    Prints a summary; does not abort (the model class itself will catch errors).
    """
    n_params = sum(v.numel() for v in sd.values())
    n_keys   = len(sd)
    dtypes   = set(str(v.dtype) for v in sd.values() if isinstance(v, torch.Tensor))

    print(f"\n[Verify] State dict summary:")
    print(f"         Keys     : {n_keys:,}")
    print(f"         Params   : {n_params:,}  ({n_params / 1e9:.3f}B)")
    print(f"         Dtypes   : {', '.join(sorted(dtypes))}")

    # Check a few expected top-level keys for a Qwen3 model
    expected_prefixes = ["model.embed_tokens", "model.layers.0", "lm_head"]
    missing = [p for p in expected_prefixes if not any(k.startswith(p) for k in sd)]
    if missing:
        print(f"[Warn]   Missing expected key prefixes: {missing}")
        print(f"         The state_dict may have an unexpected structure.")
    else:
        print(f"         Key structure looks correct (embed_tokens, layers, lm_head ✓)")

    # Cross-check param count vs config if possible
    n_layers = config.get("num_hidden_layers", 0)
    if n_layers:
        layer_keys = [k for k in sd if k.startswith("model.layers.")]
        n_layer_keys = len(layer_keys)
        print(f"         Layer keys found: {n_layer_keys} across {n_layers} layers")


def _dry_run(ds_dir: Path):
    """
    Inspect the checkpoint without writing anything.
    """
    meta = _load_meta(ds_dir)
    model_states_path = _find_model_states_file(ds_dir)

    print(f"\n{'='*60}")
    print(f"  DRY-RUN — DeepSpeed checkpoint inspection")
    print(f"{'='*60}")
    print(f"  Directory  : {ds_dir}")
    print(f"  Step       : {meta.get('step', 'unknown')}")
    print(f"  Val loss   : {meta.get('best_val_loss', 'unknown')}")
    print(f"  DS tag     : {meta.get('ds_tag', 'unknown')}")
    print(f"\n  Model config:")
    for k, v in meta.get("config", {}).items():
        print(f"    {k:30s}: {v}")

    print(f"\n  Files in checkpoint directory:")
    for f in sorted(ds_dir.iterdir()):
        size_mb = f.stat().st_size / 1e6 if f.is_file() else 0
        print(f"    {f.name:<55s}  {size_mb:8.1f} MB" if f.is_file() else f"    {f.name}/")

    sd = _extract_state_dict(model_states_path, target_dtype=None)
    _verify_state_dict(sd, meta.get("config", {}))
    print(f"\n[DryRun] No output file written.")


# ---------------------------------------------------------------------------
# Main consolidation logic
# ---------------------------------------------------------------------------

def consolidate(args):
    ds_dir = _resolve_ds_dir(args)
    print(f"[Info]  DeepSpeed checkpoint dir : {ds_dir}")

    # ---- dry-run mode
    if args.dry_run:
        _dry_run(ds_dir)
        return

    # ---- resolve output path
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not args.force:
        sys.exit(
            f"[ERROR] Output already exists: {out_path}\n"
            f"        Use --force to overwrite."
        )

    # ---- load meta sidecar
    meta   = _load_meta(ds_dir)
    config = meta.get("config", {})
    if not config:
        sys.exit(
            "[ERROR] meta.json has no 'config' key.\n"
            "        This checkpoint may not have been produced by train_deepspeed.py."
        )

    print(f"[Info]  Step          : {meta.get('step', 'unknown')}")
    print(f"[Info]  Best val loss : {meta.get('best_val_loss', 'unknown')}")
    print(f"[Info]  Model config  : {config.get('num_hidden_layers')} layers, "
          f"hidden={config.get('hidden_size')}, "
          f"vocab={config.get('vocab_size')}")

    # ---- target dtype
    target_dtype: Optional[torch.dtype] = None
    if args.dtype == "fp32":
        target_dtype = torch.float32
    elif args.dtype == "bf16":
        target_dtype = torch.bfloat16
    elif args.dtype == "fp16":
        target_dtype = torch.float16
    # "keep" → None (no conversion)

    # ---- extract weights
    model_states_path = _find_model_states_file(ds_dir)
    state_dict        = _extract_state_dict(model_states_path, target_dtype)

    # ---- verify
    _verify_state_dict(state_dict, config)

    # ---- build the sft.py-compatible checkpoint dict
    # sft.py's load_checkpoint (and its pretrained-ckpt loader) reads:
    #   ckpt["config"]      → Qwen3Config(**ckpt["config"])
    #   ckpt["model_state"] → model.load_state_dict(...)
    # The remaining keys are optional but useful for provenance.
    ckpt = {
        "model_state":   state_dict,
        "config":        config,
        "step":          meta.get("step", 0),
        "best_val_loss": meta.get("best_val_loss", float("inf")),
        "args":          meta.get("args", {}),
        # Provenance: record where this came from
        "_source":       str(ds_dir),
        "_ds_tag":       meta.get("ds_tag", ""),
    }

    # ---- save
    print(f"\n[Save]  Writing consolidated checkpoint → {out_path}")
    torch.save(ckpt, out_path)
    size_gb = out_path.stat().st_size / 1e9
    print(f"[Save]  Done.  File size: {size_gb:.2f} GB")

    print(f"""
{'='*60}
  Consolidation complete!
  Output: {out_path}

  To start SFT:
      python sft.py \\
          --checkpoint {out_path} \\
          --tokenizer  ./tokenizer \\
          --data-dir   ./sft_data \\
          --lora \\
          --out-dir    ./sft_checkpoints
{'='*60}
""")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Consolidate a DeepSpeed checkpoint directory (from train_deepspeed.py) "
            "into a single .pt file compatible with sft.py."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ---- source (mutually exclusive but we handle it in code for better errors)
    src = p.add_argument_group("Source (pick one)")
    src.add_argument(
        "--ds-dir",
        default=None,
        metavar="PATH",
        help=(
            "Path to a specific DeepSpeed step directory, e.g. "
            "./checkpoints_ds/step_0100000  or  ./checkpoints_ds/latest_ds"
        ),
    )
    src.add_argument(
        "--ds-out-dir",
        default=None,
        metavar="PATH",
        help=(
            "Root output directory used by train_deepspeed.py (--out-dir). "
            "The script will auto-select the latest step_ subdirectory."
        ),
    )

    # ---- destination
    p.add_argument(
        "--out",
        required=False,
        default="./checkpoints/pretrained.pt",
        metavar="FILE",
        help="Output .pt file path (default: ./checkpoints/pretrained.pt)",
    )

    # ---- options
    p.add_argument(
        "--dtype",
        default="keep",
        choices=["keep", "bf16", "fp16", "fp32"],
        help=(
            "Dtype for saved weights.  "
            "'keep' preserves the dtype in the checkpoint (usually bf16). "
            "Use 'fp32' if downstream code requires full precision. "
            "(default: keep)"
        ),
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite --out if it already exists.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Inspect the checkpoint structure without writing any output file. "
            "Useful for verifying that the DS checkpoint is readable."
        ),
    )

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    consolidate(args)
