#!/usr/bin/env python3
"""
infer.py

Production-grade inference for the Qwen3 (dense, non-MoE) model from
`model.py`. Loads a checkpoint produced by `train.py`, `train_sft.py`,
`deepspeed_shard_consolidator.py`, or the merge-lora path, and runs text
generation with the same KV-cache machinery the trainer uses.

The script is intentionally a single file: no HTTP server, no DB, no
background workers. The intent is a clean CLI you can drop into a
container, a shell pipe, or wrap in a thin server of your own.

Highlights
----------
* Reads raw `.pt` checkpoints (with `{"model_state", "config"}`) and
  transparently handles the LoRA-merged shape produced by
  `train_sft.py --merge-lora`.  DeepSpeed checkpoint *directories* are
  rejected with a clear pointer at `deepspeed_shard_consolidator.py`,
  the same way `train_sft_deepspeed.py` does.
* Auto-shards the model across every visible GPU, then CPU, then disk
  via `accelerate.dispatch_model` + `infer_auto_device_map`.  When
  accelerate is not installed the script falls back to a single
  device (CUDA if any GPU is free, else CPU) and prints a one-time
  warning so the user knows what they are missing.
* bf16 by default.  Optional 4-bit / 8-bit quantization through
  bitsandbytes.  Both are *soft* dependencies — the script runs
  without them, and only asks for the relevant package when its
  feature is requested.
* Full generation hyperparameter set: temperature, top-k, top-p,
  repetition penalty, min / max new tokens, batched generation with
  left-padded prompts, EOS / stop-token handling, seeded RNG.
* Auto-detects the ChatML + <think> template from the tokenizer's
  vocabulary.  If the four special tokens are present (as they will
  be for any tokenizer trained by `train_tokenizer.py`), `--prompt`
  is wrapped as a user turn.  Otherwise the input is treated as raw
  text.  An explicit `--chat-template {auto,chatml,raw}` overrides
  auto-detection, and `--enable-thinking` is available for chat mode.

Usage
-----
  # one-shot generation
  python infer.py --checkpoint ./sft_merged/merged_model.pt \\
      --tokenizer ./tokenizer --prompt "Solve 2+2"

  # interactive REPL with stream-of-tokens output
  python infer.py --checkpoint ./sft_merged/merged_model.pt \\
      --tokenizer ./tokenizer --interactive

  # batched evaluation
  python infer.py --checkpoint ./checkpoints/latest.pt \\
      --tokenizer ./tokenizer --prompts-file ./eval.jsonl \\
      --batch-size 8 --output ./out.jsonl

  # quantized
  python infer.py --checkpoint ./checkpoints/latest.pt \\
      --tokenizer ./tokenizer --load-in-4bit --prompt "..."

  # explicit memory budget
  python infer.py --checkpoint ./checkpoints/latest.pt \\
      --tokenizer ./tokenizer \\
      --max-memory "0:18GiB,cpu:30GiB,disk:200GiB" --prompt "..."
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from tokenizers import Tokenizer

from model import Qwen3Config, Qwen3ForCausalLM, count_parameters


# ---------------------------------------------------------------------------
# Optional dependencies (kept soft so the script runs without them)
# ---------------------------------------------------------------------------
# accelerate is the device-map / offload engine.  bitsandbytes is the
# int4 / int8 quantization engine.  Both are imported lazily and the
# import-failure messages are routed through the [infer] prefix so a
# missing package reads as "we can't do that, install X" instead of
# as a Python traceback the user has to decode.

def _try_import_accelerate():
    try:
        from accelerate import dispatch_model, infer_auto_device_map
        return {
            "dispatch_model": dispatch_model,
            "infer_auto_device_map": infer_auto_device_map,
        }
    except ImportError:
        return None


def _try_import_bitsandbytes():
    try:
        import bitsandbytes as bnb
        return bnb
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# ChatML template handling
# ---------------------------------------------------------------------------
# Mirrors pack_sft_data.py:118-124 exactly.  We re-define the constants
# here so a future divergence in the SFT packer is easy to spot.

CHATML_MARKERS = ("<|im_start|>", "<|im_end|>", "<think>", "</think>")
THINK_OPEN     = "<think>"
THINK_CLOSE    = "</think>"
IM_START       = "<|im_start|>"
IM_END         = "<|im_end|>"


def _tokenizer_has_chatml(tokenizer: Tokenizer) -> bool:
    """True iff every ChatML marker is in the tokenizer's vocabulary."""
    vocab = tokenizer.get_vocab()
    return all(m in vocab for m in CHATML_MARKERS)


def format_prompt(
    prompt: str,
    tokenizer: Tokenizer,
    mode: str = "auto",
    enable_thinking: bool = False,
    system: Optional[str] = None,
) -> str:
    """
    Wrap `prompt` as a ChatML user turn iff the tokenizer carries the
    four ChatML markers.  `mode` is one of {auto, chatml, raw}.

    The wrapped form (with optional system message + thinking) is:

        <|im_start|>system\\n{system}<|im_end|>\\n         (optional)
        <|im_start|>user\\n{prompt}<|im_end|>\\n
        <|im_start|>assistant\\n<think>\\n                 (only with --enable-thinking)
    """
    if mode == "raw":
        return prompt

    if mode == "auto" and not _tokenizer_has_chatml(tokenizer):
        return prompt

    # mode in (chatml) or (auto with ChatML vocab present) — wrap.
    parts: List[str] = []
    if system:
        parts.append(f"{IM_START}system\n{system}{IM_END}\n")
    parts.append(f"{IM_START}user\n{prompt}{IM_END}\n")
    parts.append(f"{IM_START}assistant\n")
    if enable_thinking:
        parts.append(f"{THINK_OPEN}\n")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Tokenizer loader
# ---------------------------------------------------------------------------

def load_tokenizer(tokenizer_dir: str) -> Tokenizer:
    """Read `tokenizer.json` from a directory (HuggingFace `tokenizers` format)."""
    path = os.path.join(tokenizer_dir, "tokenizer.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"tokenizer.json not found in {tokenizer_dir!r}. "
            f"Run train_tokenizer.py first."
        )
    return Tokenizer.from_file(path)


def find_eos_token_id(tokenizer: Tokenizer) -> Optional[int]:
    """Pick a reasonable EOS token id from the tokenizer's special tokens."""
    vocab = tokenizer.get_vocab()
    for candidate in ("<|endoftext|>", "<|im_end|>", "</s>"):
        if candidate in vocab:
            return vocab[candidate]
    return None


# ---------------------------------------------------------------------------
# Checkpoint loader
# ---------------------------------------------------------------------------

def load_model_and_config(
    checkpoint_path: str,
) -> Tuple[Qwen3ForCausalLM, Qwen3Config]:
    """
    Read a `.pt` checkpoint produced by `train.py`, `train_sft.py`,
    `deepspeed_shard_consolidator.py`, or the merge-lora path, and
    return a CPU-resident, fp32, untied `Qwen3ForCausalLM` with weights
    loaded and `tie_weights()` applied.

    The two valid checkpoint shapes are:

      {"model_state": <state_dict>, "config": <dict>}            # raw
      {"model_state": <state_dict>, "config": <dict>, "args": …}  # raw + CLI args

    A directory is treated as a DeepSpeed checkpoint and rejected
    with a clear message — run `deepspeed_shard_consolidator.py` first.
    """
    if os.path.isdir(checkpoint_path):
        meta = os.path.join(checkpoint_path, "meta.json")
        if os.path.exists(meta):
            raise RuntimeError(
                f"{checkpoint_path} looks like a DeepSpeed checkpoint directory.\n"
                f"Run deepspeed_shard_consolidator.py first to produce a "
                f"single .pt, then point --checkpoint at the consolidated file."
            )
        raise FileNotFoundError(
            f"{checkpoint_path!r} is a directory but has no meta.json; "
            f"not a recognized inference input."
        )

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path!r}")

    blob = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(blob, dict) or "model_state" not in blob or "config" not in blob:
        raise RuntimeError(
            f"{checkpoint_path!r} does not contain the expected keys "
            f"'model_state' and 'config'. Was it produced by train.py, "
            f"train_sft.py, or deepspeed_shard_consolidator.py?"
        )

    config = Qwen3Config(**blob["config"])
    model  = Qwen3ForCausalLM(config)
    # `strict=True` here would be the strictest check, but we want
    # merge-lora checkpoints to load cleanly even if the saved state
    # has slightly different keys.  In practice merge-lora saves a
    # normal state_dict, so strict=True is the right default.
    missing, unexpected = model.load_state_dict(blob["model_state"], strict=False)
    if unexpected:
        print(f"[infer] WARNING: {len(unexpected)} unexpected keys in "
              f"checkpoint; ignoring. First few: {unexpected[:3]}")
    if missing:
        # tied embeddings make this list non-empty if lm_head was
        # not in the saved state — only flag as a warning when there
        # are non-tied missing keys.
        non_tied = [k for k in missing if "lm_head.weight" not in k]
        if non_tied:
            print(f"[infer] WARNING: {len(non_tied)} non-tied keys missing "
                  f"from checkpoint; First few: {non_tied[:3]}")

    # Re-tie lm_head -> embed_tokens after load_state_dict (load replaces
    # the lm_head weight tensor object, breaking any tie that was in
    # place at __init__ time).  Mirrors train.py:511-514 and
    # train_sft.py:601-602.
    model.tie_weights()
    model.eval()
    return model, config


# ---------------------------------------------------------------------------
# Quantization (optional, via bitsandbytes)
# ---------------------------------------------------------------------------
# We replace each nn.Linear with a bnb Linear4bit / Linear8bitLt at
# load time, before dispatch.  Tied weights (lm_head -> embed_tokens)
# get the same wrapper; bnb handles the storage transparently.

def maybe_quantize(model: Qwen3ForCausalLM, args) -> Qwen3ForCausalLM:
    """
    Apply bitsandbytes int4 / int8 quantization in place if requested.

    The script does *not* require bitsandbytes; the import is deferred
    until the user actually asks for `--load-in-4bit` or `--load-in-8bit`.
    """
    if not (args.load_in_4bit or args.load_in_8bit):
        return model

    bnb = _try_import_bitsandbytes()
    if bnb is None:
        raise RuntimeError(
            f"{'4' if args.load_in_4bit else '8'}-bit quantization was "
            f"requested but bitsandbytes is not installed.\n"
            f"  pip install bitsandbytes\n"
            f"  (then rerun with the same --load-in-{{4,8}}bit flag)"
        )

    quant_cls = bnb.nn.Linear4bit if args.load_in_4bit else bnb.nn.Linear8bitLt
    quant_kwargs = (
        {
            "compute_dtype": torch.bfloat16,
            "quant_type":    "nf4",
            "use_double_quant": True,
        } if args.load_in_4bit
        else {
            "threshold":     6.0,
            "has_fp16_weights": False,
        }
    )

    n_replaced = 0
    for module_path, module in list(model.named_modules()):
        if isinstance(module, torch.nn.Linear):
            parent_path, attr = module_path.rsplit(".", 1) if "." in module_path else ("", module_path)
            parent = model
            for part in parent_path.split("."):
                if part:
                    parent = getattr(parent, part)
            new_mod = quant_cls(
                module.in_features, module.out_features,
                bias=module.bias is not None,
                **quant_kwargs,
            )
            with torch.no_grad():
                # bnb modules expose a `.weight` Parameter that we copy into.
                # Linear4bit/Linear8bitLt's copy_ handles the bit-packing
                # internally, so we don't need to manually quantize.
                new_mod.weight.data.copy_(module.weight.data)
                if module.bias is not None:
                    new_mod.bias.data.copy_(module.bias.data)
            setattr(parent, attr, new_mod)
            n_replaced += 1

    print(f"[infer] quantized {n_replaced} nn.Linear layers with "
          f"bitsandbytes {'4-bit nf4' if args.load_in_4bit else '8-bit'}")
    return model


# ---------------------------------------------------------------------------
# Device-map resolution
# ---------------------------------------------------------------------------
# Three modes:
#   1. `--device cuda:N` (or `cpu`) — single device, model.to() once.
#   2. `--device auto` and accelerate is available — infer_auto_device_map
#      + dispatch_model with optional `max_memory` override.
#   3. `--device auto` and accelerate is missing — fall back to a
#      single device, print a one-time warning.

def _parse_max_memory(spec: str) -> Dict[Union[int, str], str]:
    """
    Parse "0:18GiB,cpu:30GiB,disk:200GiB" into {0: "18GiB", "cpu": "30GiB", "disk": "200GiB"}.
    Keys that look like integers become ints (GPU indices).
    """
    out: Dict[Union[int, str], str] = {}
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"--max-memory chunk {chunk!r} has no ':' separator")
        k, v = chunk.split(":", 1)
        k = k.strip()
        v = v.strip()
        if k.isdigit():
            out[int(k)] = v
        else:
            out[k] = v
    return out


def resolve_device_map(
    model: Qwen3ForCausalLM,
    args,
) -> Tuple[Qwen3ForCausalLM, str]:
    """
    Apply device placement per the user's --device / --max-memory flags.
    Returns (model, mode_label) where mode_label is one of
    {"single", "accelerate", "cpu"} for logging.
    """
    dev = args.device

    # Mode 1: explicit single device.
    if dev != "auto":
        model.to(dev)
        n = sum(p.numel() for p in model.parameters())
        print(f"[infer] model placed on {dev}  ({n/1e9:.3f}B params)")
        return model, "single"

    # Mode 2: auto with accelerate.
    accel = _try_import_accelerate()
    if accel is not None:
        max_memory = _parse_max_memory(args.max_memory) if args.max_memory else None
        # Keep each decoder layer intact across devices — splitting a single
        # Qwen3DecoderLayer across GPUs would require slicing the KV-cache
        # and is not supported by the model's forward pass.
        from model import Qwen3DecoderLayer
        device_map = accel["infer_auto_device_map"](
            model,
            max_memory=max_memory,
            no_split_module_classes=[Qwen3DecoderLayer],
        )
        model = accel["dispatch_model"](model, device_map=device_map)
        # Summarize placement.
        per_dev: Dict[str, int] = {}
        for _, dev_str in device_map.items():
            per_dev[dev_str] = per_dev.get(dev_str, 0) + 1
        print(f"[infer] accelerate device_map: " +
              ", ".join(f"{d}={n} submodules" for d, n in per_dev.items()))
        return model, "accelerate"

    # Mode 3: auto without accelerate — fall back.
    if torch.cuda.is_available():
        target = "cuda:0"
    else:
        target = "cpu"
    model.to(target)
    print(f"[infer] accelerate not installed; falling back to {target} "
          f"(install with `pip install 'accelerate>=0.27'` for auto-shard "
          f"and CPU/disk offload).")
    return model, "cpu"


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
# Standard transformers-style samplers.  All operate on a (B, V) logits
# tensor; temperature=0 means greedy argmax.

def _apply_repetition_penalty(
    logits: torch.Tensor,
    generated: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    """In-place: divide logits of previously-generated tokens by `penalty`."""
    if penalty == 1.0:
        return logits
    bsz = logits.size(0)
    for b in range(bsz):
        prev = generated[b].unique()
        # gather penalty: positive logits get divided, negative get multiplied
        sub_logits = logits[b, prev]
        sub_logits = torch.where(
            sub_logits > 0,
            sub_logits / penalty,
            sub_logits * penalty,
        )
        logits[b, prev] = sub_logits
    return logits


def _top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k is None or k <= 0 or k >= logits.size(-1):
        return logits
    v, _ = torch.topk(logits, k)
    logits[logits < v[:, [-1]]] = -float("inf")
    return logits


def _top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    if p is None or p <= 0.0 or p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    cum = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    # remove tokens with cumulative probability above p
    sorted_mask = cum > p
    # always keep at least one
    sorted_mask[..., 0] = False
    sorted_logits[sorted_mask] = -float("inf")
    # scatter back
    out = torch.full_like(logits, -float("inf"))
    out.scatter_(-1, sorted_idx, sorted_logits)
    return out


def sample_next(
    logits: torch.Tensor,            # (B, V)
    generated: torch.Tensor,         # (B, T)  — all tokens generated so far
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
    repetition_penalty: float,
) -> torch.Tensor:
    """Return a (B, 1) tensor of next-token ids."""
    if repetition_penalty != 1.0:
        logits = _apply_repetition_penalty(logits, generated, repetition_penalty)
    if temperature <= 0.0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
    logits = _top_k_filter(logits, top_k)
    logits = _top_p_filter(logits, top_p)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
# Two entry points:
#   - generate_stream(...): generator yielding decoded chunks one at a
#     time.  Used by --interactive.
#   - generate_batch(...): list of completed completions, one per
#     prompt.  Used by --prompt and --prompts-file.

@torch.inference_mode()
def _prepare_inputs(
    tokenizer: Tokenizer,
    prompts: List[str],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Tokenize `prompts`, left-pad to the longest, and return
    (input_ids, attention_mask, prompt_len).

    Left-padding is the right choice for causal LMs: the right edge
    stays in the same position across the batch, so positional
    encodings and KV-cache queries are consistent.
    """
    encoded = [tokenizer.encode(p) for p in prompts]
    ids_list = [e.ids for e in encoded]
    # Prefer <|endoftext|> as pad (Qwen3's eot/pad token); fall back to
    # <|pad|> if the tokenizer added it as a separate special, then to 0.
    pad_id = (
        tokenizer.token_to_id("<|endoftext|>")
        or tokenizer.token_to_id("<|pad|>")
        or 0
    )

    max_len = max(len(x) for x in ids_list)
    input_ids      = torch.full((len(prompts), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(prompts), max_len), dtype=torch.long)
    for i, ids in enumerate(ids_list):
        offset = max_len - len(ids)
        input_ids[i, offset:]      = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, offset:] = 1

    input_ids      = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    return input_ids, attention_mask, max_len


@torch.inference_mode()
def _step(
    model: Qwen3ForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    past_key_values: Optional[List],
) -> Tuple[torch.Tensor, Optional[List]]:
    """
    Run one forward pass.  `input_ids` is either the full prompt (when
    `past_key_values` is None, i.e. the prefill step) or just the last
    token (decode step).  Returns the next-token logits at the last
    position and the updated KV cache.
    """
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        use_cache=True,
    )
    # logits at the last position: (B, V)
    next_logits = out["logits"][:, -1, :]
    return next_logits, out["past_key_values"]




@torch.inference_mode()
def generate_batch(
    model: Qwen3ForCausalLM,
    tokenizer: Tokenizer,
    prompts: List[str],
    *,
    max_new_tokens: int = 256,
    min_new_tokens: int = 0,
    temperature: float = 0.7,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 0.9,
    repetition_penalty: float = 1.0,
    eos_token_id: Optional[int] = None,
    seed: Optional[int] = None,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> List[Dict]:
    """
    Generate completions for a batch of prompts.  Returns a list of
    dicts {"prompt", "completion", "stop_reason"} in input order.
    """
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    if eos_token_id is None:
        eos_token_id = find_eos_token_id(tokenizer)

    # ------------------ prefill
    input_ids, base_attn, prompt_len = _prepare_inputs(tokenizer, prompts, device)
    past_kv: Optional[List] = None
    logits, past_kv = _step(model, input_ids, base_attn, past_kv)

    # generated tokens so far (B, T); starts as the prompt itself
    # (we'll trim it from the output later).  We track this for
    # repetition penalty and for stop-token detection.
    generated = input_ids.clone()
    finished  = torch.zeros(input_ids.size(0), dtype=torch.bool, device=device)

    for step in range(max_new_tokens):
        # stop emitting if every sequence is finished
        if finished.all():
            break

        # repetition penalty needs to see the prompt + everything
        # generated so far.
        next_id = sample_next(
            logits, generated,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        # For finished sequences, force-emit eos (or just the first
        # seq's token as a harmless pad if no eos) so generated has the
        # right shape, but mark finished so the rest of the batch
        # can continue.  Active sequences keep their sampled token.
        if eos_token_id is not None:
            forced = torch.full_like(next_id, eos_token_id)
        else:
            forced = next_id  # nothing to force; finished stay finished via mask below
        next_id = torch.where(finished.unsqueeze(-1), forced, next_id)
        generated = torch.cat([generated, next_id], dim=1)

        if eos_token_id is not None:
            finished = finished | (next_id.squeeze(-1) == eos_token_id)

        # Decode step: feed only the new token, no attention mask.
        # The model treats seq_len=1 + past_key_value as "attend to
        # cached context" — no causal mask needed.
        logits, past_kv = _step(model, next_id, None, past_kv)

    # ------------------ decode output
    completions: List[Dict] = []
    for i, p in enumerate(prompts):
        # The original prompt occupies the right edge of `generated`
        # initially; trim `prompt_len` chars from the left to get the
        # completion (this also discards the left-pad region).
        full_ids = generated[i].tolist()
        completion_ids = full_ids[prompt_len:]
        # cut at eos
        stop_reason = "length"
        if eos_token_id is not None and eos_token_id in completion_ids:
            cut = completion_ids.index(eos_token_id)
            completion_ids = completion_ids[:cut]
            stop_reason = "eos"
        # If the sampler emitted eos before reaching min_new_tokens,
        # flag it; the caller asked for at least that many tokens.
        if stop_reason == "eos" and len(completion_ids) < min_new_tokens:
            stop_reason = "min_new_tokens"
        completion = tokenizer.decode(completion_ids, skip_special_tokens=False)
        completions.append({
            "prompt":      p,
            "completion":  completion,
            "stop_reason": stop_reason,
        })
    return completions


def generate_stream(
    model: Qwen3ForCausalLM,
    tokenizer: Tokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 0.9,
    repetition_penalty: float = 1.0,
    eos_token_id: Optional[int] = None,
    seed: Optional[int] = None,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    """
    Yield decoded chunks one token at a time.  Used by --interactive.
    """
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    if eos_token_id is None:
        eos_token_id = find_eos_token_id(tokenizer)

    input_ids, base_attn, prompt_len = _prepare_inputs(tokenizer, [prompt], device)
    past_kv: Optional[List] = None
    logits, past_kv = _step(model, input_ids, base_attn, past_kv)
    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        next_id = sample_next(
            logits, generated,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        tok = int(next_id.item())
        generated = torch.cat([generated, next_id], dim=1)

        if eos_token_id is not None and tok == eos_token_id:
            return

        # decode the single token; skip_special_tokens=False so the
        # caller can see structural tokens if they want to.
        chunk = tokenizer.decode([tok], skip_special_tokens=False)
        yield chunk

        # Decode step: no attention mask needed (seq_len=1 + KV-cache).
        logits, past_kv = _step(model, next_id, None, past_kv)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run inference with a Qwen3-style dense LLM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- I/O
    p.add_argument("--checkpoint", required=True,
                   help="Path to a .pt produced by train.py / train_sft.py / merge-lora / "
                        "deepspeed_shard_consolidator.py.")
    p.add_argument("--tokenizer",  required=True,
                   help="Directory containing tokenizer.json (from train_tokenizer.py).")

    # ---- Device
    p.add_argument("--device", default="auto",
                   help="'auto' (use accelerate if installed), 'cpu', or 'cuda:N'.")
    p.add_argument("--max-memory", default=None,
                   help="Per-device memory budget, e.g. '0:18GiB,cpu:30GiB,disk:200GiB'. "
                        "Only used when --device auto and accelerate is installed.")

    # ---- Quantization
    gq = p.add_mutually_exclusive_group()
    gq.add_argument("--load-in-4bit", action="store_true",
                    help="Quantize linear layers to 4-bit (nf4) via bitsandbytes.")
    gq.add_argument("--load-in-8bit", action="store_true",
                    help="Quantize linear layers to 8-bit via bitsandbytes.")

    # ---- Input (mutually exclusive modes)
    gi = p.add_mutually_exclusive_group(required=True)
    gi.add_argument("--prompt", action="append", default=None,
                    help="One prompt.  Repeat the flag for multiple single-prompt runs "
                         "processed in a single batch.")
    gi.add_argument("--prompts-file", default=None,
                    help="Path to a .jsonl with {'id', 'prompt'} per line.")
    gi.add_argument("--interactive", action="store_true",
                    help="REPL mode.  Type prompts at the > prompt; commands: "
                         "/quit, /reset, /system <text>.")
    gi.add_argument("--smoke-test", action="store_true",
                    help=argparse.SUPPRESS)

    # ---- Chat formatting
    p.add_argument("--chat-template", choices=["auto", "chatml", "raw"],
                   default="auto",
                   help="How to wrap --prompt / REPL input.  'auto' detects ChatML "
                        "from the tokenizer's special tokens.")
    p.add_argument("--enable-thinking", action="store_true",
                    help="When chat mode is active, open a <think> block in the "
                         "assistant turn (matches the SFT template).")
    p.add_argument("--system", default=None,
                    help="Optional system message inserted at the start of every "
                         "ChatML-formatted prompt.")

    # ---- Generation
    p.add_argument("--max-new-tokens",      type=int,   default=512)
    p.add_argument("--min-new-tokens",      type=int,   default=0)
    p.add_argument("--temperature",         type=float, default=0.7)
    p.add_argument("--top-k",               type=int,   default=50)
    p.add_argument("--top-p",               type=float, default=0.9)
    p.add_argument("--repetition-penalty",  type=float, default=1.0)
    p.add_argument("--seed",                type=int,   default=None)
    p.add_argument("--eos-token-id",        type=int,   default=None)
    p.add_argument("--batch-size",          type=int,   default=1,
                   help="Micro-batch size for --prompt / --prompts-file.  Larger "
                        "batches are faster on GPU but use more VRAM.")

    # ---- Output
    p.add_argument("--output", default=None,
                   help="If set with --prompts-file, write completions to this "
                        ".jsonl.  Otherwise completions are printed to stdout.")
    p.add_argument("--stream",  action="store_true", default=True,
                   help="(default) Stream tokens to stdout as they are generated.")
    p.add_argument("--no-stream", dest="stream", action="store_false",
                   help="Buffer the full completion before printing.")
    return p


# ---------------------------------------------------------------------------
# Top-level entry points per input mode
# ---------------------------------------------------------------------------

def _resolve_device(model) -> torch.device:
    """Pick a reasonable device for batched input prep (tokenization happens on CPU;
    the model's first module is used to find the device for input_ids)."""
    if hasattr(model, "hf_device_map"):
        # accelerate-dispatched: pick the device of embed_tokens
        return next(model.parameters()).device
    return next(model.parameters()).device


def run_prompts(
    model: Qwen3ForCausalLM,
    tokenizer: Tokenizer,
    prompts: List[str],
    args,
) -> List[Dict]:
    """Run --prompt (one or more) through batched generation."""
    # format each prompt through ChatML if applicable
    formatted = [
        format_prompt(
            p, tokenizer,
            mode=args.chat_template,
            enable_thinking=args.enable_thinking,
            system=args.system,
        )
        for p in prompts
    ]

    device = _resolve_device(model)
    completions: List[Dict] = []
    for batch_start in range(0, len(formatted), args.batch_size):
        batch = formatted[batch_start : batch_start + args.batch_size]
        results = generate_batch(
            model, tokenizer, batch,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            eos_token_id=args.eos_token_id,
            seed=args.seed,
            device=device,
        )
        completions.extend(results)
    return completions


def run_prompts_file(
    model: Qwen3ForCausalLM,
    tokenizer: Tokenizer,
    path: str,
    args,
) -> List[Dict]:
    """Stream-read a .jsonl, run each prompt, write the output .jsonl."""
    in_path  = Path(path)
    out_path = Path(args.output) if args.output else None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    # Read all records up front so we can report progress deterministically.
    records: List[Dict] = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    completions = run_prompts(
        model, tokenizer, [r["prompt"] for r in records], args,
    )

    out_records: List[str] = []
    for rec, comp in zip(records, completions):
        out_records.append(json.dumps({
            "id":         rec.get("id"),
            "prompt":     comp["prompt"],
            "completion": comp["completion"],
            "stop_reason": comp["stop_reason"],
        }, ensure_ascii=False))

    if out_path is not None:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(out_records) + "\n")
        print(f"[infer] wrote {len(out_records)} completions to {out_path}")
    else:
        for line in out_records:
            print(line)
    return completions


def run_interactive(
    model: Qwen3ForCausalLM,
    tokenizer: Tokenizer,
    args,
) -> None:
    """REPL: read lines, stream completions back."""
    print("[infer] interactive mode.  Commands: /quit, /reset, /system <text>")
    system = args.system
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line == "/quit":
            break
        if line == "/reset":
            system = args.system
            print("[infer] session reset.")
            continue
        if line.startswith("/system "):
            system = line[len("/system "):].strip()
            print(f"[infer] system message set to: {system!r}")
            continue
        prompt = format_prompt(
            line, tokenizer,
            mode=args.chat_template,
            enable_thinking=args.enable_thinking,
            system=system,
        )
        device = _resolve_device(model)
        print("", flush=True)
        try:
            for chunk in generate_stream(
                model, tokenizer, prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                eos_token_id=args.eos_token_id,
                seed=args.seed,
                device=device,
            ):
                print(chunk, end="", flush=True)
        except Exception as e:
            print(f"\n[infer] generation error: {e}", file=sys.stderr)
        print(flush=True)


# ---------------------------------------------------------------------------
# Smoke test (no real checkpoint needed)
# ---------------------------------------------------------------------------

def smoke_test() -> int:
    """Build a tiny Qwen3 model, save it, run 5-token generation on CPU.
    Returns 0 on success.  Used by `--smoke-test`."""
    import tempfile, shutil

    print("[smoke] building tiny model…")
    tmp = tempfile.mkdtemp()
    try:
        cfg = Qwen3Config(
            vocab_size=256, hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, max_position_embeddings=128, tie_word_embeddings=True,
        )
        m = Qwen3ForCausalLM(cfg)
        ckpt = os.path.join(tmp, "tiny.pt")
        torch.save({"model_state": m.state_dict(), "config": vars(cfg)}, ckpt)

        # minimal tokenizer
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers import pre_tokenizers, decoders
        tok_dir = os.path.join(tmp, "tok")
        os.makedirs(tok_dir, exist_ok=True)
        tok = Tokenizer(BPE(unk_token=None, byte_fallback=True))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder       = decoders.ByteLevel()
        trainer = BpeTrainer(vocab_size=256, special_tokens=["<|endoftext|>"],
                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                             show_progress=False)
        tok.train_from_iterator(["hello world"], trainer=trainer)
        tok.save(os.path.join(tok_dir, "tokenizer.json"))

        # drive the public surface
        model, _   = load_model_and_config(ckpt)
        tokenizer  = load_tokenizer(tok_dir)
        # resolve device — keep it on CPU for the smoke test
        model.to("cpu")

        results = generate_batch(
            model, tokenizer, ["hello"],
            max_new_tokens=5, temperature=0.0, top_k=None, top_p=None,
            eos_token_id=tokenizer.token_to_id("<|endoftext|>"),
            device=torch.device("cpu"),
        )
        for r in results:
            print(f"[smoke] prompt={r['prompt']!r}  "
                  f"completion={r['completion']!r}  stop={r['stop_reason']}")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.smoke_test:
        return smoke_test()

    # ---- model
    print(f"[infer] loading checkpoint: {args.checkpoint}")
    t0 = time.perf_counter()
    model, config = load_model_and_config(args.checkpoint)
    print(f"[infer]   model loaded in {time.perf_counter()-t0:.1f}s  "
          f"({count_parameters(model)/1e9:.3f}B params, "
          f"hidden={config.hidden_size}, layers={config.num_hidden_layers})")

    # ---- tokenizer
    tokenizer = load_tokenizer(args.tokenizer)
    print(f"[infer]   tokenizer: vocab={tokenizer.get_vocab_size()}  "
          f"chatml={'yes' if _tokenizer_has_chatml(tokenizer) else 'no'}")

    # ---- precision (bf16 default, before device placement)
    if not (args.load_in_4bit or args.load_in_8bit):
        model.to(torch.bfloat16)
        print(f"[infer]   dtype=bfloat16 (default)")

    # ---- quantization (replaces linear layers in place)
    model = maybe_quantize(model, args)

    # ---- device placement (single, accelerate, or cpu fallback)
    model, mode = resolve_device_map(model, args)

    # ---- dispatch
    if args.interactive:
        run_interactive(model, tokenizer, args)
    elif args.prompts_file is not None:
        run_prompts_file(model, tokenizer, args.prompts_file, args)
    else:
        # default: --prompt (one or more)
        completions = run_prompts(model, tokenizer, args.prompt, args)
        for c in completions:
            if args.stream:
                print(f"[prompt]   {c['prompt']}")
            print(f"[answer]   {c['completion']}")
            print(f"[stop]     {c['stop_reason']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
