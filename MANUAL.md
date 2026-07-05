# Qwen3-Non-MOE Training Manual

A practical reference for computing dataset sizes, hyperparameters, and
optimal model sizes for pretraining, SFT, and GRPO stages, all grounded in
the Chinchilla scaling laws.

---

## 1. The Chinchilla rule (foundation for everything)

Hoffmann et al. (2022) found that, for a given compute budget, the
optimal model is trained on ~20 tokens per parameter:

```
D* (optimal tokens) ≈ 20 × N (non-embedding parameters)
```

For a fixed dataset, the optimal model size is the inverse:

```
N* (optimal params)  ≈  D (training tokens) / 20
```

All numbers below are derived from this single ratio. Use non-embedding
parameters (`N`) everywhere — that's what the Chinchilla paper counts. The
formula works best in the 50M–10B regime; outside that, lean on more
recent guidance (see §7).

---

## 2. How to get N (non-embedding parameters)

For a Qwen3-style dense transformer:

```
N = V × H                         # embedding table (tied to lm_head when tie_word_embeddings=True)
   + L × (                       # per-layer
        4 × H × (H + 2·H·G)      # Q (H) + K,V (each H·G, G = num_kv_groups)
      + 2 × H                    # Q/K RMSNorms (Qwen3 has 2 norms × H each)
      + 3 × H × I                # SwiGLU: gate, up, down (I = intermediate_size)
      + 2 × H                    # pre+post RMSNorm
     )
   − 2 × V × H                   # subtract tied output projection when tie_word_embeddings=True
```

With Qwen3's standard ratios (`head_dim=128`, `gqa_ratio=4`,
`mlp_ratio=3`, `tie_embeddings=true` for <2B), this reduces to roughly:

```
N ≈ L × (12 · H²)  +  V · H       (tied)  for hidden H, layers L
```

Quick lookup (Qwen3 conventions, tied embeddings):

| Model  | Hidden H | Layers L | Heads | KV groups | Inter I | Params (B) |
|--------|---------:|---------:|------:|----------:|--------:|-----------:|
| 0.6B   | 1024     | 28       | 16    | 4         | 3072    | ~0.6       |
| 1.7B   | 2048     | 28       | 16    | 4         | 6144    | ~1.7       |
| 4B     | 2560     | 36       | 20    | 4         | 7680    | ~4.0       |
| 8B     | 4096     | 36       | 32    | 8         | 12288   | ~8.0       |

For exact numbers run:

```bash
python model.py --target-size 1.7B
```

Use the printed `total_params` for the embedding-tied head and
`total_params - embed_params` for the Chinchilla N.

---

## 3. Pretraining: dataset size, schedule, hyperparameters

### 3.1 Optimal pretraining tokens

```
D_pretrain = 20 × N
```

| Model | N (B) | Optimal tokens | In 4B-token units |
|-------|------:|---------------:|------------------:|
| 0.6B  | 0.6   | 12 B           | 3.0 ×             |
| 1.7B  | 1.7   | 34 B           | 8.5 ×             |
| 4B    | 4.0   | 80 B           | 20 ×              |
| 8B    | 8.0   | 160 B          | 40 ×              |

### 3.2 Context length and sequence packing

- Choose `--seq-len` from your longest coherent span. 2048 is the
  pragmatic floor; 4096 is the Qwen3 default.
- Always use the packed memmap format (`pack_dataset.py`). One packed
  sequence contains many training examples concatenated; the position IDs
  reset at sample boundaries to avoid cross-document attention.
- Effective tokens per step:

```
tokens_per_step = global_batch_size × seq_len
                = (micro_batch × grad_accum × world_size) × seq_len
```

### 3.3 Number of optimizer steps

```
num_steps = D_pretrain / tokens_per_step
```

Round up; cosine schedule reaches zero LR at exactly `num_steps`.

### 3.4 Peak learning rate (AdamW)

Use the scaling rule from µP / scaled-init literature:

```
lr_peak = lr_base × (B_ref / B_eff)^0.5       # B_eff = tokens_per_step
```

Anchors (work for AdamW, bf16, dense transformers):

| B_eff (tokens/step) | lr_peak |
|--------------------:|--------:|
| 0.5 M               | 1.0e-3  |
| 1 M                 | 7.0e-4  |
| 2 M                 | 5.0e-4  |
| 4 M                 | 3.5e-4  |
| 8 M                 | 2.5e-4  |
| 16 M                | 1.7e-4  |

Interpolate logarithmically. For Qwen3-style models, the "safe window"
is roughly `1e-4 ≤ lr_peak ≤ 1e-3`. Below 5e-5, training stalls; above
2e-3, you will see loss spikes within the first 1k steps.

### 3.5 LR schedule

- **Warmup**: 1–2 % of total steps, linear from `lr_peak / warmup_ratio` to
  `lr_peak`. Default `warmup_ratio=0.01`.
- **Decay**: cosine to `lr_min = 0.1 × lr_peak` over the remaining steps.
- **Weight decay**: `0.1` on weights, `0.0` on biases / norms / embeddings.
- **Beta1, Beta2**: 0.9, 0.95. Eps 1e-8.

### 3.6 Effective batch size

For stability and MFU, target `B_eff ≈ 2–4 M tokens/step` for sub-2B
models and 4–8 M for 2–8B models.

```
micro_batch        = largest power of 2 that fits one H100/4090 at seq_len
grad_accum         = ceil(B_eff / (micro_batch × seq_len × world_size))
```

Concrete examples (bf16, no checkpointing, 4090 24 GB):

| Model | seq_len | micro_batch (per GPU) | GPUs | B_eff (M) | grad_accum |
|-------|--------:|----------------------:|-----:|----------:|-----------:|
| 0.6B  | 4096    | 4                     | 1    | 0.5       | 32         |
| 0.6B  | 4096    | 8                     | 4    | 2.0       | 16         |
| 1.7B  | 4096    | 2                     | 4    | 2.0       | 64         |
| 8B    | 4096    | 1 (ckpt)              | 8    | 4.0       | 128        |

### 3.7 Gradient clipping

- **Global L2 norm** at `grad_clip = 1.0`. Universal default across
  GPT-3, LLaMA, Qwen, DeepSeek.
- For sub-200M models or unstable early runs, drop to `0.5`.
- Skip clipping if using ZeRO-3 with `--cpu-offload` (offload already
  caps magnitudes).

### 3.8 bfloat16, mixed precision, compile

- Compute dtype: `bf16`. Accumulation in `fp32`. Loss in `fp32`.
- `torch.set_float32_matmul_precision("high")` to enable TF32 on Ampere+.
- `torch.compile(model, mode="default")` gives ~15–25 % speedup. Use
  `mode="reduce-overhead"` only with the CUDAGraphs plumbing already
  implemented in `train.py` (see Architecture note in CLAUDE.md).

### 3.9 Total pretraining compute (sanity check)

```
C_total = 6 × N × D_pretrain            FLOPs  (Chinchilla / PaLM formula)
       = 120 × N²                      FLOPs  (when D = 20N)
```

Example: 1.7B params → 34B tokens → ~3.5 × 10²⁰ FLOPs ≈ 4 days on a
single 4090 (≈100 TFLOP/s sustained, MFU ~0.4).

---

## 4. SFT (supervised fine-tuning): dataset and schedule

SFT sits on top of a Chinchilla-optimal pretrained model. The SFT
corpus is two orders of magnitude smaller than pretraining.

### 4.1 Optimal SFT tokens

Empirically the best loss-vs-tokens curves for SFT flatten at
~50–200 tokens per parameter of the base model:

```
D_sft_opt ≈ 100 × N
```

| Model | N (B) | D_sft_opt (B) | Equivalent in JSONL (avg 600 tok/sample) |
|-------|------:|--------------:|----------------------------------------:|
| 0.6B  | 0.6   | 0.06          | ~100 k samples                          |
| 1.7B  | 1.7   | 0.17          | ~280 k samples                          |
| 4B    | 4.0   | 0.40          | ~670 k samples                          |
| 8B    | 8.0   | 0.80          | ~1.3 M samples                          |

Going past 200× N is over-fitting on instruction format; below 20× N
the model under-fits and degrades base capabilities.

### 4.2 SFT epochs (not tokens-per-param)

SFT uses multiple passes because the data is small. Two to four epochs
is the sweet spot:

```
epochs_sft = clamp(50 × N / samples_available, 2, 4)
```

If you have 50 k samples for a 0.6B model: `50×0.6e9 / 50e3 ≈ 600 k
samples-per-pass` → train for 2–3 full passes.

### 4.3 SFT hyperparameters (full fine-tune)

| Knob            | Value         | Notes                                            |
|-----------------|--------------:|--------------------------------------------------|
| peak LR         | 1e-5 – 5e-5   | 10–30× lower than pretraining                    |
| LR schedule     | cosine        | warmup 3 %, decay to 0                           |
| weight decay    | 0.0           | Don't regularize the format-distillation phase   |
| dropout         | 0.0           | Already baked in during pretraining              |
| grad clip       | 1.0           | Same as pretraining                              |
| micro batch     | fits memory   | Bump grad accum to reach B_eff ≥ 0.5 M           |
| B_eff           | 0.5 – 1 M     | Smaller window than pretraining                  |
| sequence len    | 4096 – 8192   | Long enough for `<think>...</think>` spans       |
| epochs          | 2 – 4         | More → overfit on style, less → underfit on task |

### 4.4 LoRA (preferred for 1B+ on a single GPU)

- Rank `r = α/2` with `α` between 1× and 2× of `r`; common `r=64, α=128`.
- Target `q_proj, k_proj, v_proj, o_proj` (sometimes also gate/up/down).
- LR is the same as full fine-tune; the smaller parameter count
  compensates for the larger gradient magnitudes on the adapter weights.
- Total steps: same formula as §3.3, with `N` replaced by the
  LoRA-trainable parameter count (typically 1–3 % of N). The
  `D_sft_opt` budget still references the *base* model N.

### 4.5 Loss masking

Always mask the prompt tokens (set labels to -100) so the loss only
flows over the assistant / `<think>` / answer region. The `pack_sft_data.py`
script writes the `mask.bin` to do this automatically. Position IDs must
also reset at sample boundaries within a packed window.

---

## 5. GRPO (RLHF-style fine-tuning): dataset and schedule

GRPO is Group Relative Policy Optimization (DeepSeek, 2024). It does not
fit the Chinchilla curve — RLHF is reward-driven, not likelihood-driven,
so the right metric is *number of policy updates*, not total tokens.

### 5.1 Prompt dataset size

```
prompts_total ≈ 10 × num_steps_grpo × group_size
```

- `group_size` (G) is the number of completions sampled per prompt
  (typical G = 4–16). A larger G reduces gradient variance but costs
  proportionally more rollouts.
- For a 1.7B model, 1000–5000 prompts is usually enough; for 8B+ go
  up to 10–50 k. Beyond that, the policy saturates and KL dominates.

### 5.2 Number of GRPO updates

Start with one prompt per step. Then:

```
num_steps_grpo = min(prompts_available, 0.5 × D_sft_opt / mean_prompt_len)
```

The cap is the Chinchilla-consistent sanity bound: don't burn more
tokens on RL than you would have spent on continued SFT.

| Model | group_size | prompts | steps | rule of thumb       |
|-------|-----------:|--------:|------:|---------------------|
| 0.6B  | 8          | 4 k     | 4 k   | 1 pass              |
| 1.7B  | 8          | 8 k     | 8 k   | 1 pass              |
| 8B    | 16         | 32 k    | 32 k  | 1 pass, 2 GPUs roll |

### 5.3 GRPO hyperparameters

| Knob                | Value          | Notes                                     |
|---------------------|---------------:|-------------------------------------------|
| rollout batch       | 1 – 4 prompts  | Bounded by KV-cache memory                |
| group size G        | 4 – 16         | Higher = lower variance, more memory      |
| KL coefficient β    | 0.001 – 0.04   | Start 0.01; raise if drift > 5 nats      |
| clip ratio ε        | 0.1 – 0.2      | PPO-style dual clip                       |
| entropy bonus       | 0.0 – 0.01     | 0.0 for math, +0.005 for open chat        |
| policy LR           | 5e-7 – 1e-6    | 50–100× lower than SFT                    |
| value LR (if used)  | 1e-5           | Only with separate critic                |
| grad clip           | 0.5            | Tighter than SFT                          |
| micro batch (PPO)   | 1              | LoRA usually required                     |
| epochs per rollout  | 1              | More → off-policy bias                    |
| max prompt len      | 1024           | Trim long prompts                         |
| max response len    | 1024 – 4096    | Math: 2048+, Chat: 1024                   |
| reward shaping      | length-norm    | Divide by completion length               |

### 5.4 Reward model

GRPO needs a verifier, not a learned reward. For math it is exact
match; for code it is unit tests; for chat it is preference model
or rule-based rubric. Rollouts that don't change the reward signal
(after length normalization) add no gradient and waste compute — log
their fraction and drop them if > 30 %.

### 5.5 Compute budget split

For a 1.7B model spending 200 GPU-hours on alignment:

| Phase         | Allocation | Hours  |
|---------------|-----------:|-------:|
| SFT           | 30 %       | 60     |
| GRPO rollouts | 50 %       | 100    |
| GRPO updates  | 15 %       | 30     |
| Eval/merge    | 5 %        | 10     |

---

## 6. Reverse: choosing the model size for a fixed dataset

If you already have D tokens of pretraining data, the Chinchilla-optimal
model size is:

```
N* = D / 20
```

| D (tokens)  | N* (B) | Suggested Qwen3 size | Heads × Layers × H |
|-------------|-------:|---------------------:|-------------------:|
| 500 M       | 0.025  | 25 M scratch         | 4×8×256            |
| 5 B         | 0.25   | 0.3 B (Chinchilla)   | 8×16×512           |
| 50 B        | 2.5    | 1.7 B                | 16×28×2048         |
| 200 B       | 10     | 8 B                  | 32×36×4096         |
| 1 T         | 50     | 32 B (out of Chinchilla sweet spot — scale tokens instead) |

Two important caveats:

1. **Below 50 M params**, Chinchilla over-counts embedding parameters.
   Halve the formula: `N* = D / 40`.
2. **Above 10 B params**, Chinchilla under-trains. Use 15–20 tokens per
   param for 10–70 B, and 10–15 tokens per param for 70 B+ (LLaMA-2/3
   scaling). The Qwen3 dense line stops at 32 B; if you need larger,
   consider MoE.

### 6.1 Verdict logic

```
D < 10 × N           → undertrained, get more data or shrink model
10 N ≤ D ≤ 30 N      → Chinchilla-acceptable
D > 30 N             → overtrained for the model, consider scaling up
```

### 6.2 Decision flowchart

```
start
  │
  ▼
How much data D do you have? (tokens)
  │
  ├── fixed D, free model size → N* = D / 20
  │
  ├── fixed N, free data       → D* = 20 × N
  │
  └── fixed both, want optimal compute
        ├── N < 50M           → use N* = D / 40
        ├── 50M ≤ N ≤ 10B     → use Chinchilla (20×)
        └── N > 10B           → use 15× (LLaMA-2 rule)
```

---

## 7. Beyond Chinchilla: when to deviate

| Regime                       | Recommended tokens-per-param | Source                          |
|------------------------------|-----------------------------:|---------------------------------|
| < 50 M params                | 40 – 80                      | Chinchilla Appendix, Kaplan      |
| 50 M – 10 B (dense)          | 20                           | Chinchilla                      |
| 10 B – 70 B (dense)          | 15 – 20                      | LLaMA-2                         |
| 70 B+ (dense)                | 10 – 15                      | LLaMA-3, DeepSeek               |
| Reasoning models             | 30 – 50                      | DeepSeek-R1, Qwen3-thinking     |
| Multilingual / code-heavy    | 25 – 30                      | Qwen2.5, Yi                     |

The reasoning bump comes from the fact that long chain-of-thought
trajectories inflate the effective token count the model needs.

---

## 8. Worked examples (end-to-end)

### Example A: 1.7B on a single 4090

```
N = 1.7e9
D_pretrain = 20 × 1.7e9 = 34 B tokens
D_sft_opt  = 100 × 1.7e9 = 0.17 B tokens ≈ 280 k samples
prompts_grpo = 8000 (8 k prompts × 1 step × G=8)
```

| Stage    | Steps | B_eff (M) | LR_peak | grad clip |
|----------|------:|----------:|--------:|----------:|
| Pretrain | 132 k | 1.0       | 7e-4    | 1.0       |
| SFT      | 500   | 0.5       | 2e-5    | 1.0       |
| GRPO     | 8 k   | 0.13      | 1e-6    | 0.5       |

(Micro batch × grad accum chosen to hit B_eff on 4× 4090 DDP for
pretraining, single GPU + LoRA for SFT, single GPU + LoRA for GRPO.)

### Example B: 0.6B on a single 4090, hobby budget

```
N = 0.6e9
D_pretrain = 12 B tokens
D_sft_opt  = 0.06 B tokens ≈ 100 k samples
prompts_grpo = 4000
```

| Stage    | Steps | B_eff (M) | LR_peak |
|----------|------:|----------:|--------:|
| Pretrain | 24 k  | 0.5       | 1.0e-3  |
| SFT      | 200   | 0.5       | 3.0e-5  |
| GRPO     | 4 k   | 0.13      | 1.0e-6  |

### Example C: 8B, fixed data = 200 B tokens

```
D = 200 B
N* = 200e9 / 20 = 10 B   → pick 8B (close, under-trained by 17 %)
                          → use 8B and 200 B for 25 tok/param
D_sft_opt  = 100 × 8e9 = 0.8 B tokens
prompts_grpo = 32 k
```

If you had only 80 B tokens, `N* = 4 B` — use the 4B config instead
of wasting tokens on an 8B undertrained.

---

## 9. Quick reference card

```
Chinchilla:
  D* = 20 N           (optimal tokens for given params)
  N* = D / 20         (optimal params for given tokens)

Pretrain schedule:
  num_steps    = D / (B_eff)
  B_eff        = 2 – 8 M tokens / step
  lr_peak      ≈ 5e-4 × sqrt(2e6 / B_eff)
  warmup_ratio = 0.01
  grad_clip    = 1.0
  weight_decay = 0.1
  betas        = (0.9, 0.95)
  dtype        = bf16

SFT:
  D_sft        ≈ 100 N
  epochs       = 2 – 4
  lr_peak      = 1e-5 – 5e-5
  grad_clip    = 1.0
  B_eff        = 0.5 – 1 M
  LoRA         = r 64, α 128 on qkvo

GRPO:
  prompts      ≈ 10 × steps × G
  G            = 4 – 16
  lr_peak      = 5e-7 – 1e-6
  grad_clip    = 0.5
  KL β         = 0.001 – 0.04
  clip ε       = 0.1 – 0.2
```

---

## 10. Sanity checks (run before launching)

1. **Tokens / param ratio**: 15 ≤ D/N ≤ 25. Outside this band, you're
   leaving quality on the table (recompute `N*` or `D*`).
2. **LR × sqrt(B_eff) ≈ constant**: if you change B_eff, scale LR by
   the square root in the opposite direction.
3. **Step budget ÷ warmup**: warmup steps = max(100, 0.01 × total).
4. **Disk budget**: `D × 4 bytes` (bf16) or `D × 2 bytes` (int8) of
   token-id storage before you start.
5. **MFU target**: 35–45 % on bf16, 50–60 % on fp8. If you're
   consistently below 30 %, fix the dataloader (pin memory, non-blocking)
   before lowering batch size.
6. **LoRA sanity**: if SFT loss diverges with LoRA but converges
   full-finetune, lower the LR by 2× and check that the base model
   weights are frozen (`requires_grad=False`).

---

## 11. Tokenizer: one per pretrain run, shared across stages

**Build exactly one tokenizer** at the start of a project, on a diverse
sample of the **pretrain corpus**. Reuse it for SFT and GRPO.

**Why one is enough**

- The tokenizer is a fixed `vocab → id` mapping. Switching it mid-pipeline
  would shift every embedding and break the learned representations.
- The pretrain corpus is the largest and most varied data you will see,
  so its vocabulary is the widest. Training on English-only would split
  code identifiers into many small tokens; training on math/code only
  would tokenize prose poorly.

**Practical sequence for this repo**

```bash
# 1. Train ONE tokenizer (once per project, before everything else)
python train_tokenizer.py --data-dir ./data --vocab-size 32000 --out-dir ./tokenizer

# 2. Pretrain — uses ./tokenizer for packing and the LM head
python build_dataset.py --target-size 50GB
python pack_dataset.py --data-dir ./data --tokenizer ./tokenizer
torchrun --nproc_per_node=1 train.py --model-size 0.6B --data-dir ./packed

# 3. SFT — same ./tokenizer, different packed cache
python download_sft_data.py --target-size 2GB
python pack_sft_data.py --data-dir ./sft_data --tokenizer ./tokenizer \
    --cache-dir ./sft_packed
python train_sft.py --checkpoint ./checkpoints/latest.pt \
    --tokenizer ./tokenizer --cache-dir ./sft_packed

# 4. GRPO — same ./tokenizer, prompt dataset only
# (tokenizer is loaded inside the rollout server / reward function)
```

**Vocab size trade-offs**

The Qwen3 architecture allocates a `vocab_size × hidden_size` embedding
table (tied to `lm_head` for sub-2B models):

| vocab_size | hidden | embed params | % of 0.6B model |
|-----------:|-------:|-------------:|----------------:|
| 16 000     | 1024   | 16.4 M       | 2.7 %           |
| 32 000     | 1024   | 32.8 M       | 5.4 %           |
| 64 000     | 1024   | 65.5 M       | 11 %            |
| 151 936    | 1024   | 155.6 M      | 26 %            |

- **32 k** is the hobby sweet spot: small enough to keep embeddings
  cheap, large enough to tokenize multilingual text without exploding
  sequence length.
- **64 k+** is what production models (Qwen3, LLaMA-3) use. Justifies
  the embedding cost with cleaner compression on long-tail tokens.
- **< 16 k** is rarely useful — you start seeing OOV fallback on any
  real-world text.

**When to rebuild the tokenizer**

Only when starting a **new pretrain run** on a substantially different
domain (e.g. all-Chinese, biomedical, code-only, legal). In that case:

```
new domain → train new tokenizer → pretrain from scratch
                                            ↓
                                same tokenizer for SFT, GRPO
```

**Never** rebuild between SFT and GRPO of the same model. That would
invalidate the embeddings the model already learned during pretraining
and SFT, and the model would have to re-learn tokenization implicitly
through gradient updates — guaranteed to destroy task performance.

---

*All formulas in this manual are dimensionless — they apply to any
dense transformer. Plug in your N from `python model.py` and the
data size from your packed `meta.json`.*
