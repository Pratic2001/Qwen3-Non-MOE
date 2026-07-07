#!/usr/bin/env python3
"""
diagnose_and_generate.py

Diagnose common "the model outputs garbage" failure modes for a trained
Qwen3-style model loaded from a checkpoint, and produce cleaner generations.

This is the script to run when the pretrained model is giving output like:

    Q:: am movie: you't the one the
    : this:'m movie:'t what you you? your
    :,,,,,,,,,,,,,,,...

The most common root causes, in order of likelihood:

    1. KV-cache drift — generate() advances the cache but the model isn't
       passing position_ids, so RoPE positions get out of sync after a few
       steps. Symptom: plausible tokens for the first 2-4, then gibberish
       or a single repeated token.

    2. Embeddings not tied — checkpoint was loaded without calling
       tie_weights() afterwards, so lm_head is a different matrix from
       embed_tokens. Symptom: random-looking words and a collapse to one
       frequent token.

    3. Model is in train() mode during generation — Dropout zeros out
       random positions of the activations. Symptom: words with missing
       letters, fragments like "you't", "'m".

    4. Tokenizer is a different vocab than the model was trained with —
       wrong token IDs go in, garbage comes out. Symptom: the model
       produces reasonable-looking English fragments that don't quite
       match any word ("movie:" with a colon, "you't").

Usage:
    python diagnose_and_generate.py --checkpoint ./checkpoints/latest.pt \\
        --tokenizer ./tokenizer \\
        --prompt "Once upon a time" \\
        --max-tokens 100

If you don't have a prompt, omit --prompt and the script will try several
diagnostic prompts and report on each.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from model import Qwen3Config, Qwen3ForCausalLM, count_parameters


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_model_for_inference(ckpt_path: str, device: str = "cuda"):
    """
    Load a Qwen3ForCausalLM from a checkpoint file and put it in eval mode
    with weights properly tied. Returns (model, config_dict).
    """
    print(f"[load] reading {ckpt_path} ...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    config = Qwen3Config(**ckpt["config"])
    model = Qwen3ForCausalLM(config)

    state = ckpt.get("model_state", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[load] WARNING: {len(missing)} missing keys, first 5: {missing[:5]}")
    if unexpected:
        print(f"[load] WARNING: {len(unexpected)} unexpected keys, first 5: {unexpected[:5]}")

    # CRITICAL: re-tie lm_head -> embed_tokens after state_dict load.
    # load_state_dict replaces tensors, breaking the tie that __init__ set up.
    if hasattr(model, "tie_weights"):
        model.tie_weights()

    # CRITICAL: eval mode disables dropout, which would otherwise drop
    # random tokens in the generated output and cause fragment-like
    # outputs ("you't", "'m").
    model.eval()
    model.to(device)

    n_params = count_parameters(model)
    print(f"[load] model: {n_params:,} params  ({n_params/1e6:.1f}M)")
    print(f"[load] vocab: {config.vocab_size}  hidden: {config.hidden_size}  "
          f"layers: {config.num_hidden_layers}  heads: {config.num_attention_heads}")

    # Report embedding-tie status so the user can see at a glance whether
    # the lm_head and embed_tokens are actually the same tensor object.
    same = model.lm_head.weight is model.model.embed_tokens.weight
    print(f"[load] embeddings tied: {same}  "
          f"(lm_head.storage == embed.storage: "
          f"{model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()})")

    return model, config


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_diagnostics(model, config, device, tokenizer=None):
    """
    Run a battery of sanity checks on the loaded model and print results.

    Each diagnostic is independent; failures are reported but don't stop
    later diagnostics. Returns a dict of {name: passed} booleans.
    """
    results = {}

    # ------------------------------------------------------------------
    # 1. Loss on random tokens should be ~ln(vocab_size) for an untrained
    #    model. A trained model's loss depends on training duration.
    # ------------------------------------------------------------------
    n_random = 256
    random_ids = torch.randint(0, config.vocab_size, (1, n_random), device=device)
    out = model(random_ids, labels=random_ids)
    loss_random = out["loss"].item()
    expected_random = math.log(config.vocab_size)
    print(f"\n[diag 1] Loss on random token sequence: {loss_random:.4f}")
    print(f"          (untrained baseline: ln(vocab) = {expected_random:.4f})")
    print(f"          (trained: should be < {expected_random:.4f}, "
          f"ideally < {expected_random * 0.5:.4f})")
    results["diag1"] = loss_random < expected_random * 0.85
    if loss_random > expected_random * 0.95:
        print("          ⚠ Loss is near-random — model may not be trained or "
              "checkpoint is corrupted.")

    # ------------------------------------------------------------------
    # 2. Loss on a repeated token (all the same token id) should be
    #    ~ln(vocab_size). If it's much lower, something is overfitting
    #    to a single token.
    # ------------------------------------------------------------------
    same_token = torch.full((1, 64), 42, dtype=torch.long, device=device)
    out = model(same_token, labels=same_token)
    loss_same = out["loss"].item()
    print(f"\n[diag 2] Loss on 64×same-token sequence: {loss_same:.4f}")
    print(f"          (expected: near {expected_random:.4f} — model should "
          f"not memorise a single token id)")
    results["diag2"] = loss_same > expected_random * 0.5

    # ------------------------------------------------------------------
    # 3. Top-1 token should be stable across very similar inputs.
    #    If the same prompt produces wildly different argmaxes between
    #    two runs, the model has a non-deterministic path (probably
    #    still in train() mode, or some dropout is still active).
    # ------------------------------------------------------------------
    if tokenizer is not None:
        prompt = "The quick brown fox"
        ids = tokenizer.encode(prompt).ids
        x = torch.tensor([ids], device=device)
        out1 = model(x, use_cache=False)
        out2 = model(x, use_cache=False)
        # Apply embed_scale same way the model does internally
        last1 = out1["logits"][0, -1]
        last2 = out2["logits"][0, -1]
        top1 = last1.argmax().item()
        top2 = last2.argmax().item()
        top1_prob = F.softmax(last1, dim=-1).max().item()
        same = (top1 == top2)
        print(f"\n[diag 3] Top-1 token for '{prompt}' on two runs: "
              f"{top1} vs {top2}  (same: {same})")
        print(f"          Top-1 probability: {top1_prob:.4f}")
        results["diag3"] = same

    # ------------------------------------------------------------------
    # 4. lm_head weight and embed_tokens weight are the same tensor
    #    object (i.e. the tie is real, not just numerically similar).
    # ------------------------------------------------------------------
    same_obj = model.lm_head.weight is model.model.embed_tokens.weight
    same_data = model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
    print(f"\n[diag 4] lm_head and embed_tokens are tied: "
          f"same_obj={same_obj}  same_data={same_data}")
    if not same_data:
        print("          ⚠ lm_head and embed_tokens point at different tensors.")
        print("          The model was loaded without re-tying weights. Call")
        print("          model.tie_weights() to fix.")
    results["diag4"] = same_data

    # ------------------------------------------------------------------
    # 5. Generation produces a non-trivial distribution. If top-1 prob
    #    is >0.99 on every step, the model is in a degenerate attractor.
    # ------------------------------------------------------------------
    if tokenizer is not None:
        prompt = "The"
        ids = tokenizer.encode(prompt).ids
        x = torch.tensor([ids], device=device)
        out = model(x, use_cache=False)
        probs = F.softmax(out["logits"][0, -1] / 1.0, dim=-1)
        top1_p = probs.max().item()
        eff_vocab = float((-probs * probs.log()).exp().item())  # exp(entropy)
        print(f"\n[diag 5] After 'The', top-1 prob: {top1_p:.4f}  "
              f"effective vocab: {eff_vocab:.1f}")
        if top1_p > 0.5:
            print("          ⚠ Top-1 prob is very high — model may be in a "
                  "degenerate state. Try lower temperature.")
        results["diag5"] = top1_p < 0.5

    return results


# ---------------------------------------------------------------------------
# Robust generation (replacement for Qwen3ForCausalLM.generate)
# ---------------------------------------------------------------------------

@torch.no_grad()
def robust_generate(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 1.1,
    eos_token_id: Optional[int] = None,
    use_kv_cache: bool = True,
    debug: bool = False,
):
    """
    A more robust text generator than the one in model.py, designed to
    surface (and avoid) the failure modes that produce garbage output.

    Improvements over model.py:
      - Always puts the model in eval() (caller's responsibility but
        we double-check here)
      - explicit position_ids so RoPE is correct on every step whether
        or not KV-cache is used
      - repetition_penalty to break the comma-loop attractor
      - top_p (nucleus) on top of top_k for cleaner long-range sampling
      - per-step logging in debug mode showing top-5 candidates
    """
    if model.training:
        model.eval()
        if debug:
            print("[gen] WARNING: model was in train() mode, switched to eval().")

    device = input_ids.device
    generated = input_ids.clone()
    past_key_values = None

    for step in range(max_new_tokens):
        # Whether to feed the full prefix or just the newest token.
        # The model computes the absolute RoPE position from
        # past_key_values[i][0].shape[2] internally, so we don't need
        # to track position_ids ourselves. This is the right place to
        # add explicit position_ids if the model's logic ever changes.
        if use_kv_cache and past_key_values is not None:
            model_input = generated[:, -1:]
        else:
            model_input = generated

        out = model(
            model_input,
            past_key_values=past_key_values,
            use_cache=use_kv_cache,
        )
        logits = out["logits"][:, -1, :]   # (B, vocab)
        past_key_values = out.get("past_key_values", None)

        # ---- repetition penalty: divide logits of already-seen tokens
        # by `repetition_penalty` (HuggingFace convention). Without this
        # a model in a slight attractor (e.g. "," with high prob) will
        # loop forever, which is the comma-blast in your sample.
        if repetition_penalty != 1.0 and generated.shape[1] > 0:
            seen = torch.unique(generated[0])
            # Penalise logits that have been seen
            seen_logits = logits[0, seen]
            penalty_mask = seen_logits > 0
            seen_logits[penalty_mask]  = seen_logits[penalty_mask]  / repetition_penalty
            seen_logits[~penalty_mask] = seen_logits[~penalty_mask] * repetition_penalty
            logits[0, seen] = seen_logits

        # ---- temperature
        logits = logits / max(temperature, 1e-5)

        # ---- top-k
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            kth = v[:, -1:].expand_as(logits)
            logits = torch.where(logits < kth, torch.full_like(logits, -float("inf")), logits)

        # ---- top-p (nucleus) on top of top-k
        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cum = sorted_probs.cumsum(dim=-1)
            # shift so we keep the first token that pushes cum > top_p
            keep = (cum - sorted_probs) < top_p
            sorted_logits = torch.where(
                keep, sorted_logits, torch.full_like(sorted_logits, -float("inf"))
            )
            # scatter back to original indices
            logits = torch.full_like(logits, -float("inf"))
            logits.scatter_(-1, sorted_idx, sorted_logits)

        probs = F.softmax(logits, dim=-1)
        # Guard against NaN if everything got masked out
        if not torch.isfinite(probs).all() or probs.sum() == 0:
            if debug:
                print(f"[gen] step {step}: empty prob mass, falling back to argmax")
            next_token = logits.argmax(dim=-1, keepdim=True)
        else:
            next_token = torch.multinomial(probs, num_samples=1)

        if debug and step < 10:
            top5p, top5i = probs[0].topk(5)
            print(f"  step {step:3d}  top5: "
                  + "  ".join(f"id={i.item()}({p.item():.2f})" for p, i in zip(top5p, top5i)))

        generated = torch.cat([generated, next_token], dim=1)

        if eos_token_id is not None and (next_token == eos_token_id).all():
            if debug:
                print(f"[gen] hit EOS at step {step}")
            break

    return generated


# ---------------------------------------------------------------------------
# Tokenizer helpers
# ---------------------------------------------------------------------------

def load_tokenizer(tokenizer_dir: str):
    """Load the byte-level BPE tokenizer trained by train_tokenizer.py."""
    try:
        from tokenizers import Tokenizer
    except ImportError:
        print("ERROR: tokenizers package not installed. pip install tokenizers")
        sys.exit(1)

    path = Path(tokenizer_dir) / "tokenizer.json"
    if not path.exists():
        print(f"ERROR: {path} not found. Run train_tokenizer.py first.")
        sys.exit(1)
    tok = Tokenizer.from_file(str(path))
    print(f"[tok] vocab size: {tok.get_vocab_size()}")
    # Try to surface special tokens
    for special in ["<|im_start|>", "<|im_end|>", "<|endoftext|>",
                    "<|tool_call_start|>", "<think>", "</think>"]:
        tid = tok.token_to_id(special)
        if tid is not None:
            print(f"[tok]   {special!r} -> id {tid}")
    return tok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Diagnose a Qwen3-style pretrained model and generate text."
    )
    p.add_argument("--checkpoint", required=True,
                   help="Path to a .pt file containing a model_state dict.")
    p.add_argument("--tokenizer", default="./tokenizer",
                   help="Path to a directory with tokenizer.json (default: ./tokenizer)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--prompt", default=None,
                   help="Text prompt. If omitted, runs several diagnostic prompts.")
    p.add_argument("--max-tokens", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--repetition-penalty", type=float, default=1.1)
    p.add_argument("--no-kv-cache", action="store_true",
                   help="Disable KV cache (slower but rules out cache bugs)")
    p.add_argument("--debug", action="store_true",
                   help="Print per-step top-5 candidates")
    p.add_argument("--diagnose-only", action="store_true",
                   help="Run diagnostics and exit without generating")
    args = p.parse_args()

    # ---- Load
    model, config = load_model_for_inference(args.checkpoint, device=args.device)
    tokenizer = load_tokenizer(args.tokenizer)

    # ---- Diagnose
    print("\n" + "=" * 64)
    print("  DIAGNOSTICS")
    print("=" * 64)
    results = run_diagnostics(model, config, args.device, tokenizer=tokenizer)
    n_pass = sum(results.values())
    n_total = len(results)
    print(f"\n  {n_pass}/{n_total} diagnostics passed.")

    if args.diagnose_only:
        return

    # ---- Generate
    if args.prompt is None:
        # Run a small battery of prompts and report each
        prompts = [
            "The quick brown fox",
            "Once upon a time",
            "In the year 2050,",
            "Q: What is the capital of France?\nA:",
        ]
    else:
        prompts = [args.prompt]

    print("\n" + "=" * 64)
    print("  GENERATION")
    print("=" * 64)

    for prompt in prompts:
        ids = tokenizer.encode(prompt).ids
        x = torch.tensor([ids], device=args.device)
        print(f"\n  prompt: {prompt!r}  (tokens: {ids[:8]}{'...' if len(ids)>8 else ''})")
        out = robust_generate(
            model, x,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            use_kv_cache=not args.no_kv_cache,
            debug=args.debug,
        )
        new_ids = out[0, len(ids):].tolist()
        text = tokenizer.decode(new_ids)
        print(f"  output: {text!r}")
        print(f"  (raw new tokens: {new_ids[:30]}{'...' if len(new_ids)>30 else ''})")


if __name__ == "__main__":
    main()
