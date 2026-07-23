#!/usr/bin/env python3
"""
model.py

A from-scratch PyTorch implementation of the Qwen3 (dense, non-MoE) transformer
architecture:

    - RMSNorm (pre-norm)
    - Rotary Position Embeddings (RoPE)
    - Grouped Query Attention (GQA) with QK-Norm (RMSNorm on Q/K per-head,
      applied before RoPE -- this is Qwen3's distinguishing stabilization trick)
    - SwiGLU MLP
    - Causal attention with KV-cache support for generation

The headline feature: instead of hand-picking hidden_size / num_layers / etc,
you specify a `target_params` (e.g. "0.6B", "1.7B", "4B", "8B", "1B", "600M")
and `Qwen3Config.from_target_size(...)` searches for an architecture whose
parameter count matches the target, following Qwen3's width/depth conventions
(head_dim=128, GQA ratio, SwiGLU ratio ~3x, embedding tying for small models).

Usage:
    python model.py --target-size 0.6B
    python model.py --target-size 1B --vocab-size 32000
"""

import argparse
import math
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Qwen3Config:
    vocab_size: int = 151936
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    max_position_embeddings: int = 32768
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    attention_bias: bool = False

    # ------------------------------------------------------------------
    # Auto-sizing
    # ------------------------------------------------------------------
    @staticmethod
    def _param_count(hidden_size: int, num_layers: int, intermediate_size: int,
                      num_heads: int, num_kv_heads: int, head_dim: int,
                      vocab_size: int, tie_embeddings: bool) -> int:
        """Closed-form parameter count for this architecture (no biases)."""
        embed = vocab_size * hidden_size

        q = hidden_size * (num_heads * head_dim)
        k = hidden_size * (num_kv_heads * head_dim)
        v = hidden_size * (num_kv_heads * head_dim)
        o = (num_heads * head_dim) * hidden_size
        qk_norm = 2 * head_dim  # q_norm + k_norm, RMSNorm over head_dim

        attn = q + k + v + o + qk_norm

        mlp = 3 * hidden_size * intermediate_size  # gate, up, down

        norms = 2 * hidden_size  # input_layernorm + post_attention_layernorm

        per_layer = attn + mlp + norms
        total = embed + num_layers * per_layer + hidden_size  # + final norm

        if not tie_embeddings:
            total += vocab_size * hidden_size  # separate lm_head

        return total

    @classmethod
    def from_target_size(
        cls,
        target_params: "int | str",
        vocab_size: int = 151936,
        head_dim: int = 128,
        gqa_ratio: int = 4,
        mlp_ratio: float = 3.0,
        max_position_embeddings: int = 32768,
        rope_theta: float = 1_000_000.0,
        tie_embeddings: Optional[bool] = None,
        verbose: bool = True,
        quality_mode: str = "shape",
        param_slack: float = 0.10,
    ) -> "Qwen3Config":
        """
        Search for (hidden_size, num_layers, num_heads, num_kv_heads,
        intermediate_size) that produces a model with ~target_params
        parameters, following Qwen3's architectural conventions:

          - head_dim fixed at 128 (Qwen3's value across all dense sizes)
          - num_kv_heads = num_heads / gqa_ratio (GQA grouping, default 4:1,
            matching Qwen3-8B's 32:8 ratio)
          - intermediate_size ~= mlp_ratio * hidden_size, rounded to a
            multiple of 256
          - embeddings tied for small models (<~2B params), untied for
            larger ones (matches Qwen3's own convention)

        quality_mode:
            "exact"  — pick the config whose parameter count is closest to
                       target. Ties broken by quality score.
            "shape"  — pick the config with the best quality score among all
                       configs within ±param_slack of the target param count.
                       Closer-to-target is used as a tie-breaker.
                       (default — recommended for training quality)

        param_slack:
            In quality_mode="shape", the allowed param-count deviation from
            the target as a fraction (default 0.10 = ±10%). Set higher if
            no good-shape config exists near your target.

        Quality score (higher = better):
            - depth/width: peak at hidden_size / num_layers ≈ 35, falls off
              linearly outside [25, 50].
            - head count:  +0.4 per head above 4 (capped at +2.4 = 10+ heads)
            - MLP ratio:   +0.5 if intermediate/hidden is in [2.5, 3.5],
                           -0.5 if outside [2.0, 4.0].
            - gqa:         +0.2 if num_kv_heads >= 2 (penalize degenerate MQA).
            - param slack: small bonus for staying close to target so
                           100M target doesn't pick a 60M config.
        """
        target = parse_param_count(target_params)

        if tie_embeddings is None:
            tie_embeddings = target < 2_000_000_000

        if quality_mode not in ("exact", "shape"):
            raise ValueError(
                f"quality_mode must be 'exact' or 'shape', got {quality_mode!r}"
            )

        # Candidate head counts -- Qwen3 chooses num_heads independently of
        # hidden_size (q/k/v project to num_heads*head_dim, which need not
        # equal hidden_size; o_proj maps back to hidden_size). We require
        # num_kv_heads >= 2 to avoid degenerate near-MQA shapes.
        head_count_candidates = [4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64]

        # Bound the search to depth/width ratios resembling real Qwen3 dense
        # models so we don't get pathological single-layer-but-huge or
        # hundred-layer-but-tiny architectures that happen to hit the exact
        # param count. We widen the upper end in quality_mode="shape" since
        # deeper-but-narrower often wins for small models.
        if quality_mode == "shape":
            ratio_lo, ratio_hi = 18, 180
        else:
            ratio_lo, ratio_hi = 20, 150

        def _quality_score(hidden_size: int, num_layers: int,
                           num_heads: int, num_kv_heads: int,
                           intermediate_size: int,
                           actual_params: int) -> float:
            """Higher is better. See docstring for the components."""
            # 1) depth/width ratio: peak at 35, smooth falloff
            ratio = hidden_size / num_layers
            if 30.0 <= ratio <= 45.0:
                ratio_score = 2.0
            elif 25.0 <= ratio <= 50.0:
                # linear ramp from 1.0 (at 25 or 50) to 2.0 (at 30 or 45)
                d = min(ratio - 25.0, 50.0 - ratio)
                ratio_score = 1.0 + d / 5.0
            elif 18.0 <= ratio <= 65.0:
                # outer band: 0.0 to 1.0
                d = min(ratio - 18.0, 65.0 - ratio)
                ratio_score = max(0.0, d / 7.0)
            else:
                ratio_score = -1.0  # strongly disfavor

            # 2) head count: 4 is the floor, more is better (capped)
            head_score = min(2.4, 0.4 * max(0, num_heads - 4))

            # 3) MLP ratio sweet spot
            mlp_r = intermediate_size / hidden_size
            if 2.5 <= mlp_r <= 3.5:
                mlp_score = 0.5
            elif 2.0 <= mlp_r <= 4.0:
                mlp_score = 0.0
            else:
                mlp_score = -0.5

            # 4) gqa sanity (degenerate MQA already filtered by caller)
            gqa_score = 0.2 if num_kv_heads >= 2 else 0.0

            # 5) closeness to target: small bonus for staying on-budget
            slack = abs(actual_params - target) / target
            target_score = 1.0 - min(slack, 0.30) / 0.30  # 1.0 at target, 0 at 30% off

            return ratio_score + head_score + mlp_score + gqa_score + target_score

        # Enumerate candidates
        candidates = []  # list of (sort_key, kwargs, actual_params, score)
        for num_layers in range(12, 65, 2):
            for hidden_size in range(256, 8192 + 1, 64):
                ratio = hidden_size / num_layers
                if not (ratio_lo <= ratio <= ratio_hi):
                    continue

                for num_heads in head_count_candidates:
                    num_kv_heads = max(2, num_heads // gqa_ratio)
                    if num_heads % num_kv_heads != 0:
                        continue

                    intermediate_size = _round_to_multiple(int(hidden_size * mlp_ratio), 256)

                    actual = cls._param_count(
                        hidden_size, num_layers, intermediate_size, num_heads,
                        num_kv_heads, head_dim, vocab_size, tie_embeddings,
                    )

                    # Apply the quality-mode filter
                    if quality_mode == "exact":
                        rel_diff = abs(actual - target) / target
                        # sort_key is (rel_diff,) — lower is better
                        sort_key = (round(rel_diff, 4),)
                        candidates.append((sort_key, dict(
                            hidden_size=hidden_size,
                            num_hidden_layers=num_layers,
                            num_attention_heads=num_heads,
                            num_key_value_heads=num_kv_heads,
                            intermediate_size=intermediate_size,
                        ), actual, 0.0))
                    else:  # "shape"
                        rel_diff = abs(actual - target) / target
                        if rel_diff > param_slack:
                            continue
                        score = _quality_score(
                            hidden_size, num_layers, num_heads, num_kv_heads,
                            intermediate_size, actual,
                        )
                        # sort_key: maximize score, break ties by closeness to target
                        sort_key = (-score, round(rel_diff, 4))
                        candidates.append((sort_key, dict(
                            hidden_size=hidden_size,
                            num_hidden_layers=num_layers,
                            num_attention_heads=num_heads,
                            num_key_value_heads=num_kv_heads,
                            intermediate_size=intermediate_size,
                        ), actual, score))

        if not candidates:
            raise ValueError(
                f"Could not find a config for target_params={target} "
                f"(quality_mode={quality_mode!r}, param_slack={param_slack}). "
                f"Try a larger param_slack (e.g. 0.25) or quality_mode='exact'."
            )

        candidates.sort(key=lambda t: t[0])
        _, kwargs, actual, score = candidates[0]
        config = cls(
            vocab_size=vocab_size,
            head_dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            tie_word_embeddings=tie_embeddings,
            **kwargs,
        )

        if verbose:
            pct = 100 * abs(actual - target) / target
            print(f"[Qwen3Config.from_target_size] target={target:,} -> actual={actual:,} "
                  f"(diff {pct:+.2f}%, quality_mode={quality_mode!r}, "
                  f"param_slack={param_slack}, score={score:.2f})")
            print(f"  hidden_size={config.hidden_size}, num_layers={config.num_hidden_layers}, "
                  f"num_heads={config.num_attention_heads}, num_kv_heads={config.num_key_value_heads}, "
                  f"head_dim={config.head_dim}, intermediate_size={config.intermediate_size}, "
                  f"tie_embeddings={config.tie_word_embeddings}, "
                  f"H/L={config.hidden_size/config.num_hidden_layers:.1f}, "
                  f"intermediate/H={config.intermediate_size/config.hidden_size:.2f}")

        return config


def parse_param_count(value: "int | str") -> int:
    """Parse '0.6B', '1.7B', '600M', '8B', or a raw int into a parameter count."""
    if isinstance(value, (int, float)):
        return int(value)
    s = value.strip().upper()
    if s.endswith("B"):
        return int(float(s[:-1]) * 1_000_000_000)
    if s.endswith("M"):
        return int(float(s[:-1]) * 1_000_000)
    return int(s)


def _round_to_multiple(x: int, multiple: int) -> int:
    return max(multiple, int(round(x / multiple)) * multiple)


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            x32 = x.float()
            variance = x32.pow(2).mean(-1, keepdim=True)
            x32 = x32 * torch.rsqrt(variance + self.eps)
            out = self.weight.float() * x32
        return out.to(dtype)


# ---------------------------------------------------------------------------
# Rotary Position Embeddings
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 32768, base: float = 1_000_000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_position_embeddings = max_position_embeddings

    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # position_ids: (batch, seq_len)
        inv_freq = self.inv_freq.to(position_ids.device)
        freqs = torch.einsum("bi,j->bij", position_ids.float(), inv_freq)  # (b, seq, dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (b, seq, dim)
        return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor,
                          cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # q, k: (batch, num_heads, seq, head_dim)
    # cos, sin: (batch, seq, head_dim) -> unsqueeze for heads dim
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """(batch, num_kv_heads, seq, head_dim) -> (batch, num_kv_heads * n_rep, seq, head_dim)"""
    if n_rep == 1:
        return x
    b, num_kv_heads, seq, head_dim = x.shape
    x = x[:, :, None, :, :].expand(b, num_kv_heads, n_rep, seq, head_dim)
    return x.reshape(b, num_kv_heads * n_rep, seq, head_dim)


# ---------------------------------------------------------------------------
# Attention (GQA + QK-Norm + RoPE)
# ---------------------------------------------------------------------------

class Qwen3Attention(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.n_rep = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        # QK-Norm: RMSNorm applied per-head to Q and K, before RoPE.
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        bsz, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim)

        # QK-Norm (per-head RMSNorm), applied before RoPE.
        q = self.q_norm(q)
        k = self.k_norm(k)

        # -> (batch, heads, seq, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Defensive: q/k must match v's dtype going into SDPA. RMSNorm (and any
        # autocast interaction around it) can otherwise silently promote q/k to
        # fp32 while v stays in the engine's compute dtype.
        if q.dtype != v.dtype:
            q = q.to(v.dtype)
        if k.dtype != v.dtype:
            k = k.to(v.dtype)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        new_past_kv = (k, v) if use_cache else None

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        is_causal = attention_mask is None and (past_key_value is None) and seq_len > 1

        # Explicitly pin the SDPA backend instead of letting PyTorch choose.
        # Once we pass a custom float attn_mask (needed for left-padding),
        # the flash-attention kernel is unavailable — fine, we want the
        # memory-efficient (xFormers-style) kernel in that case, which is
        # still ~linear in seq_len. What we must NOT allow is a silent
        # fallback to the naive "math" kernel: that materializes the full
        # (batch, heads, seq_len, seq_len) attention-probability matrix and
        # keeps it around for backward on every layer, turning what should
        # be linear memory into quadratic. This was happening silently on
        # every GRPO forward pass (rollout, ref logprobs, policy logprobs)
        # because all of them pass an explicit padding mask.
        backend_ctx = nullcontext()
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel
            backends = ([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
                        if is_causal else [SDPBackend.EFFICIENT_ATTENTION])
            backend_ctx = sdpa_kernel(backends)
        except ImportError:
            # Older torch: same intent via the legacy context manager.
            backend_ctx = torch.backends.cuda.sdp_kernel(
                enable_flash=is_causal,
                enable_math=False,
                enable_mem_efficient=True,
            )

        with backend_ctx:
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                is_causal=is_causal,
                scale=1.0 / math.sqrt(self.head_dim),
            )

        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, self.num_heads * self.head_dim)
        attn_out = self.o_proj(attn_out)
        return attn_out, new_past_kv


# ---------------------------------------------------------------------------
# SwiGLU MLP
# ---------------------------------------------------------------------------

class Qwen3MLP(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.self_attn = Qwen3Attention(config)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, new_past_kv = self.self_attn(
            hidden_states, cos, sin, attention_mask, past_key_value, use_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, new_past_kv


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class Qwen3Model(nn.Module):
    """Transformer body: embeddings -> decoder layers -> final norm."""

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(
            config.head_dim, config.max_position_embeddings, config.rope_theta,
        )
        self.gradient_checkpointing = False
        # Embedding output scaling — multiplies the embedding lookup by
        # 1/sqrt(hidden_size). This is the single biggest stability fix for
        # small/narrow models (hidden_size < 1024) where the std=0.02 init
        # produces activations whose variance drifts up the layers and
        # combines with the lm_head gradient (which is 1/sqrt(vocab_size)
        # on a tied embedding) to cause gradient explosion in the first
        # ~50 steps. Used by Qwen3, Gemma, and most modern small-model
        # recipes.
        self.embed_scale = 1.0 / math.sqrt(config.hidden_size)

    def enable_gradient_checkpointing(self):
        """
        Trade ~30-35% compute for ~30-35% VRAM reduction by recomputing each
        layer's activations during the backward pass instead of storing them.
        Call this before torch.compile() if using both together.
        """
        self.gradient_checkpointing = True
        print("[GradCkpt] gradient checkpointing enabled — activations will be "
              "recomputed on backward (saves VRAM, ~30% slower per step)")

    def _ckpt_layer(self, layer, hidden_states, cos, sin):
        """Wrapper so torch.utils.checkpoint can call a layer with no kwargs."""
        # use_reentrant=False is required for compatibility with torch.compile
        # and avoids issues with in-place ops during recompute.
        return torch.utils.checkpoint.checkpoint(
            layer, hidden_states, cos, sin, None, None, False,
            use_reentrant=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[list] = None,
        use_cache: bool = False,
    ):
        bsz, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids) * self.embed_scale

        if position_ids is None:
            past_len = past_key_values[0][0].shape[2] if past_key_values is not None else 0
            position_ids = torch.arange(past_len, past_len + seq_len, device=input_ids.device).unsqueeze(0)
            position_ids = position_ids.expand(bsz, -1)

        cos, sin = self.rotary_emb(position_ids)

        # gradient checkpointing is incompatible with KV-cache (the cached
        # tensors are not recomputed, so we disable use_cache when enabled).
        if self.gradient_checkpointing and self.training:
            use_cache = False

        new_past_key_values = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                # Recompute this layer's activations during backward instead
                # of storing them — saves (num_layers - 1) * activation_size
                # of VRAM at the cost of one extra forward pass per layer.
                hidden_states, new_kv = self._ckpt_layer(layer, hidden_states, cos, sin)
            else:
                hidden_states, new_kv = layer(
                    hidden_states, cos, sin, attention_mask, past_kv, use_cache
                )

            if use_cache:
                new_past_key_values.append(new_kv)

        hidden_states = self.norm(hidden_states)
        return hidden_states, new_past_key_values


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.model  = Qwen3Model(config)

        # Always create lm_head as a proper nn.Linear so torch.compile /
        # CUDAGraphs sees a single unambiguous tensor owner for the output
        # projection.  When tie_word_embeddings=True we later point
        # lm_head.weight at the embedding table via tie_weights() — keeping
        # them parameter-identical but graph-distinct, which is what
        # CUDAGraphs needs to avoid the "tensor overwritten by a subsequent
        # run" error that fires when the same Parameter is consumed by two
        # different graph nodes (embed lookup + output projection).
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.apply(self._init_weights)

        # Tie weights immediately after init so the model is correct by
        # default even before tie_weights() is called explicitly.
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def tie_weights(self):
        """Re-tie lm_head -> embed_tokens after loading a checkpoint."""
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def _init_weights(self, module):
        # Init scheme: standard Qwen3 (std=0.02 for all params) with the
        # embedding output scaled by 1/sqrt(hidden_size) so the residual
        # stream starts at variance ~1.0, matching what RMSNorm produces
        # after the first layer. This is the canonical small-model fix
        # and was missing from the original release of this file, which
        # is why the 62.5M model (hidden=448) was unstable at LR >= 5e-4.
        std = 0.02
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[list] = None,
        use_cache: bool = False,
        num_logits_to_keep: int = 0,
    ):
        hidden_states, new_past_key_values = self.model(
            input_ids, attention_mask, position_ids, past_key_values, use_cache,
        )

        # num_logits_to_keep > 0: caller only needs logits to score the
        # last N *target* tokens (e.g. GRPO log-prob scoring, where the
        # sequence is prompt+completion but only completion tokens matter).
        # Projecting every position through lm_head materializes a
        # (B, L, vocab_size) tensor — with vocab_size in the 100k+ range
        # this dwarfs everything else in the forward pass. Slicing the
        # hidden states BEFORE lm_head (rather than slicing logits after,
        # as the old code effectively did by discarding rows post-hoc)
        # avoids ever allocating the unused rows.
        # Note this is mutually exclusive with `labels`, which needs
        # every position for its own internal shift; GRPO-style callers
        # never pass both.
        if num_logits_to_keep > 0:
            assert labels is None, (
                "num_logits_to_keep and labels are mutually exclusive: "
                "labels needs the full-sequence shift, num_logits_to_keep "
                "is for the pre-sliced last-N-targets case."
            )
            # +1 then drop the last row == the standard "shift left by one"
            # used for next-token prediction, restricted to the last N
            # target positions instead of the whole sequence.
            hidden_for_logits = hidden_states[:, -(num_logits_to_keep + 1):-1, :]
        else:
            hidden_for_logits = hidden_states

        # lm_head is always an nn.Linear; when tie_word_embeddings=True its
        # weight was pointed at embed_tokens.weight in __init__ / tie_weights,
        # so no extra memory is used and gradients flow correctly.
        logits = self.lm_head(hidden_for_logits)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {"logits": logits, "loss": loss, "past_key_values": new_past_key_values}

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 32,
                  temperature: float = 1.0, top_k: Optional[int] = None,
                  eos_token_id: Optional[int] = None):
        self.eval()
        past_key_values = None
        generated = input_ids

        for _ in range(max_new_tokens):
            if past_key_values is None:
                model_input = generated
            else:
                model_input = generated[:, -1:]

            out = self.forward(model_input, past_key_values=past_key_values, use_cache=True)
            logits = out["logits"][:, -1, :] / max(temperature, 1e-5)
            past_key_values = out["past_key_values"]

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# CLI: build & sanity-check a model for a given target size
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build a Qwen3-style dense model auto-sized to a target param count.")
    parser.add_argument("--target-size", default="0.6B", help="Target parameter count, e.g. 0.6B, 1B, 1.7B, 4B, 8B, 600M")
    parser.add_argument("--vocab-size", type=int, default=151936)
    parser.add_argument("--gqa-ratio", type=int, default=4, help="num_heads / num_kv_heads")
    parser.add_argument("--mlp-ratio", type=float, default=3.0, help="intermediate_size / hidden_size")
    args = parser.parse_args()

    config = Qwen3Config.from_target_size(
        args.target_size,
        vocab_size=args.vocab_size,
        gqa_ratio=args.gqa_ratio,
        mlp_ratio=args.mlp_ratio,
    )

    model = Qwen3ForCausalLM(config)
    n_params = count_parameters(model)
    print(f"\nActual parameter count: {n_params:,} ({n_params / 1e9:.3f}B)")

    # Forward pass sanity check
    bsz, seq_len = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (bsz, seq_len))
    labels = torch.randint(0, config.vocab_size, (bsz, seq_len))

    out = model(input_ids, labels=labels)
    print(f"\nForward pass OK.")
    print(f"  logits shape: {tuple(out['logits'].shape)}")
    print(f"  loss: {out['loss'].item():.4f} (expect ~ln(vocab_size)={math.log(config.vocab_size):.4f} for random init)")

    # KV-cache generation sanity check
    gen = model.generate(input_ids[:, :4], max_new_tokens=8, top_k=10)
    print(f"\nGeneration OK. Output shape: {tuple(gen.shape)} (input was 4 tokens, +8 generated)")


if __name__ == "__main__":
    main()
