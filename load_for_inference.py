#!/usr/bin/env python3
"""
load_for_inference.py

Reference implementation for loading a Qwen3-style model from a consolidated
.pt file and generating text. Copy this pattern into your own scripts.

The single most common bug in custom load scripts is forgetting to call
model.tie_weights() after model.load_state_dict(). load_state_dict
substitutes new tensor objects for the ones in __init__, breaking the
lm_head <-> embed_tokens pointer relationship. The result is generation
output that looks like:

    Q:: am movie: you't the one the
    : this:'m movie:'t what you you? your
    :,,,,,,,,,,,,,,,,,,,,,,,,,,,,...

This script shows the four things you must always do:

    1. Build the model
    2. load_state_dict(...)
    3. tie_weights()         <-- the one everyone forgets
    4. eval()                <-- also commonly forgotten; otherwise
                                dropout ruins the output

Usage:
    python load_for_inference.py \\
        --checkpoint ./checkpoints/pretrained.pt \\
        --tokenizer  ./tokenizer \\
        --prompt     "Once upon a time"
"""

import argparse
from pathlib import Path
from typing import Optional

import torch

from model import Qwen3Config, Qwen3ForCausalLM, count_parameters


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_pretrained(checkpoint_path: str, device: str = "cuda") -> tuple[Qwen3ForCausalLM, dict]:
    """
    Load a Qwen3ForCausalLM from a .pt file with all the safety checks
    needed for inference. Returns (model, ckpt_dict) so the caller can
    inspect args / step / best_val_loss if desired.
    """
    # 1. Read the file
    print(f"[load] reading {checkpoint_path} ...")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # 2. Rebuild config (the .pt file stores the full Qwen3Config dict)
    config = Qwen3Config(**ckpt["config"])
    print(f"[load] config: hidden={config.hidden_size} layers={config.num_hidden_layers} "
          f"heads={config.num_attention_heads} vocab={config.vocab_size} "
          f"tie_embeddings={config.tie_word_embeddings}")

    # 3. Build the model. Qwen3ForCausalLM.__init__ ties lm_head to
    #    embed_tokens if config.tie_word_embeddings is True.
    model = Qwen3ForCausalLM(config)
    print(f"[load] built model: {count_parameters(model):,} params")

    # 4. Load weights. load_state_dict SUBSTITUTES new tensor objects
    #    into model.lm_head.weight and model.model.embed_tokens.weight,
    #    which BREAKS the tie set up in step 3 — the two are now
    #    different Python objects (numerically equal because the
    #    training-time tie meant the same data was saved twice, but
    #    not the same .data_ptr()).
    state = ckpt["model_state"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[load] WARNING: {len(missing)} missing keys, first 5: {missing[:5]}")
    if unexpected:
        print(f"[load] WARNING: {len(unexpected)} unexpected keys, first 5: {unexpected[:5]}")

    # 5. Re-establish the tie. After this call, model.lm_head.weight
    #    and model.model.embed_tokens.weight are the SAME Python
    #    object (same .data_ptr()), which is what the model was
    #    trained with and what torch.compile / CUDAGraphs expect.
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    # Verify the tie actually took effect
    embed_ptr = model.model.embed_tokens.weight.data_ptr()
    head_ptr  = model.lm_head.weight.data_ptr()
    if embed_ptr == head_ptr:
        print(f"[load] tie_weights ✓ (lm_head and embed_tokens share storage)")
    else:
        print(f"[load] WARNING: tie_weights() did not converge to a shared tensor")
        print(f"        embed.data_ptr() = {embed_ptr}")
        print(f"        head.data_ptr()  = {head_ptr}")

    # 6. Switch to eval mode. The model has no dropout in this repo
    #    (Qwen3 style), but if any future layer adds one, this prevents
    #    it from zeroing random positions during generation.
    model.eval()

    # 7. Move to device
    model.to(device)
    print(f"[load] model on {device}  step={ckpt.get('step', '?')}  "
          f"best_val_loss={ckpt.get('best_val_loss', '?')}")

    return model, ckpt


# ---------------------------------------------------------------------------
# Tokenizer helper
# ---------------------------------------------------------------------------

def load_tokenizer(tokenizer_dir: str):
    """Load a byte-level BPE tokenizer. Returns a tokenizers.Tokenizer."""
    from tokenizers import Tokenizer
    path = Path(tokenizer_dir) / "tokenizer.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run train_tokenizer.py first."
        )
    return Tokenizer.from_file(str(path))


# ---------------------------------------------------------------------------
# Generation — uses model.generate() (the simple one in model.py)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(
    model: Qwen3ForCausalLM,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_k: int = 50,
    eos_token_id: Optional[int] = None,
):
    """
    Generate text using the built-in Qwen3ForCausalLM.generate method.

    For better quality (repetition penalty, top-p, debug output), use
    diagnose_and_generate.py instead. This is the minimal example.
    """
    ids = tokenizer.encode(prompt).ids
    input_ids = torch.tensor([ids], device=next(model.parameters()).device)
    out = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_token_id=eos_token_id,
    )
    new_ids = out[0, len(ids):].tolist()
    return tokenizer.decode(new_ids), new_ids


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Load a Qwen3 pretrained model and generate text."
    )
    p.add_argument("--checkpoint", required=True,
                   help="Path to a .pt file with model_state, config, step, ...")
    p.add_argument("--tokenizer", default="./tokenizer")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--prompt", default="Once upon a time,")
    p.add_argument("--max-tokens", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    args = p.parse_args()

    model, ckpt = load_pretrained(args.checkpoint, device=args.device)
    tokenizer  = load_tokenizer(args.tokenizer)

    text, ids = generate(
        model, tokenizer, args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(f"\nprompt: {args.prompt!r}")
    print(f"output: {text!r}")
    print(f"raw ids: {ids[:40]}{'...' if len(ids) > 40 else ''}")


if __name__ == "__main__":
    main()
