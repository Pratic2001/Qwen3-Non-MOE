# Design rationale: GRPO stage (`train_grpo.py`)

> This file was the design spec for the GRPO stage before
> `train_grpo.py` was written. The script has since been implemented
> (and so has its DeepSpeed twin, `train_grpo_deepspeed.py`).
> Kept as a reference for **why** the trainer is shaped the way it is
> — the locked-in design decisions, the reward function's three tiers,
> and the rationale for the single-model vs two-model reference policy
> are still the source of truth. Read this if you're modifying the
> GRPO loss, reward, or rollout shape.

## Context

This repo already has the full pretrain → SFT pipeline (`train.py`, `train_sft.py`). The missing piece is the RL post-training stage. The user wants a single script `train_grpo.py` that consumes the SFT checkpoint and applies **Group Relative Policy Optimization (GRPO)** — the same algorithm used in DeepSeek-R1 / Open-R1 reasoning fine-tunes.

GRPO requires three things the codebase doesn't have yet:

1. A **rollout / generation loop** with KV-cache that produces `G` completions per prompt, along with their per-token log-probs.
2. A **reward function** that scores completions against a ground-truth answer.
3. A **policy-gradient loss** that uses group-normalized advantages + a clipped ratio + an optional KL penalty against a reference policy.

## Design decisions (locked in)

| Choice | Value |
|---|---|
| Reward | **Option A — rule-based three-tier** (correctness + format + thinking bonus). |
| Reference | **Switchable** via `--ref-policy {single,two}`. `single` = reuse the trainable model with `no_grad` for the reference pass (frozen SFT reference, ~10% VRAM overhead). `two` = keep a second copy of the model frozen in memory and KL against it (~2× VRAM, original DeepSeek-R1 recipe). |
| Dataset | Reuse `./sft_data/*/*.jsonl` by default; `--prompts-file` overrides. |
| Trainable | Full fine-tune or LoRA (re-use `inject_lora` from `train_sft.py`). |
| Multi-GPU | DDP via `torchrun`, same pattern as `train_sft.py`. |

## File to create

**`/home/pratic/Desktop/Qwen3-Non-MOE/train_grpo.py`** — single self-contained script, mirrors the structure of `train_sft.py` so the user has one file to read.

## Implementation outline

### 1. Imports & constants
Re-use everything already in the repo:
```python
from model import Qwen3Config, Qwen3ForCausalLM, count_parameters
from train_sft import (
    LoRALinear, inject_lora, merge_lora,
    lora_state_dict, lora_parameter_count,
    load_tokenizer,
    setup_distributed, is_master, get_lr, build_optimizer,
    _raw, prune_checkpoints,
)
from pack_sft_data import load_tokenizer, get_special_token_id
```

### 2. Reward function (option A — three tiers)

```python
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
ANSWER_RE = re.compile(r"\\boxed\{([^}]+)\}")
NUM_RE    = re.compile(r"-?\d+(?:\.\d+)?")

def extract_gold_answer(answer_field: str) -> tuple[float | None, str]:
    """Return (numeric_value, raw_string) from an answer field, or (None, raw)."""
    m = ANSWER_RE.search(answer_field)
    if m: return None, m.group(1).strip()
    m = NUM_RE.search(answer_field)
    if m: return float(m.group(0)), m.group(0)
    return None, answer_field.strip()

def extract_completion_answer(text: str) -> tuple[float | None, str]:
    """Same extraction applied to the model's completion."""
    # Cut off anything after a final </think> block before extracting
    end = text.find(THINK_CLOSE)
    after = text[end + len(THINK_CLOSE):] if end != -1 else text
    m = ANSWER_RE.search(after)
    if m: return None, m.group(1).strip()
    m = NUM_RE.search(after)
    if m: return float(m.group(0)), m.group(0)
    return None, after.strip().split()[-1] if after.strip() else ""

def compute_reward(prompt: str, completion: str, ground_truth: str,
                    max_new_tokens: int) -> tuple[float, dict]:
    """
    Three-tier reward:
      1.0  — answer correct + has <think>...</think> + did not truncate
      0.5  — answer correct but no thinking block, OR wrong answer but
             well-formed thinking + final answer
      0.0  — no answer, or truncated, or malformed
    """
    has_think  = THINK_OPEN in completion and completion.find(THINK_OPEN) < completion.find(THINK_CLOSE)
    truncated  = (len(completion) >= max_new_tokens * 4)  # ~chars/token heuristic
    gold_num, gold_str = extract_gold_answer(ground_truth)
    pred_num, pred_str = extract_completion_answer(completion)
    correct = (gold_num is not None and pred_num is not None
               and math.isclose(gold_num, pred_num, rel_tol=1e-3)) \
              or (gold_str.lower() == pred_str.lower())
    info = dict(correct=int(correct), has_think=int(has_think), truncated=int(truncated))
    if correct and has_think and not truncated: return 1.0, info
    if correct or (has_think and pred_str and not truncated): return 0.5, info
    return 0.0, info
```

### 3. Prompt dataset

```python
class GRPOPromptDataset:
    """Loads {prompt, answer} records from JSONL shards, streaming to keep RAM flat."""
    def __init__(self, sources: list[str], tokenizer, max_prompt_len: int):
        # sources = list of file paths; either ./sft_data/*/*.jsonl or
        # the file passed via --prompts-file
        ...

    def __len__(self): return self._count_records()

    def sample_batch(self, batch_size: int) -> list[tuple[list[int], str]]:
        """Return [(prompt_token_ids, ground_truth_answer_str), ...]."""
        ...
```

Mirrors the discovery pattern in `SFTDataset._discover_manifests` so the same `--data-dir ./sft_data` argument works. Each record yields `(prompt_ids, ground_truth)` where `ground_truth = record["answer"]`.

### 4. Rollout (`generate_rollouts`)

Hand-rolled generation loop (don't use `Qwen3ForCausalLM.generate()` because we need **per-token log-probs under `no_grad`**, and because we want G completions per prompt batched together for speed).

```python
@torch.no_grad()
def generate_rollouts(model, prompt_ids: torch.Tensor,  # (B, P)
                      max_new_tokens: int, temperature: float,
                      top_p: float, eos_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate G completions per prompt in a single batched pass:
      - Replicate prompt_ids to (B*G, P), set seed per replica for diversity.
      - Loop max_new_tokens steps with KV-cache.
      - Sample from softmax(logits / temp) with optional top-p.
      - Stop early on eos; right-pad survivors with eos_id for log-prob alignment.
    Returns:
      full_ids   : (B*G, P+T)   token ids (prompt + generated)
      gen_mask   : (B*G, T)     1 for generated positions, 0 for padding
      gen_logp   : (B*G, T)     per-token log-prob of the sampled token
    """
```

This is the same pattern as `infer.py`'s batched generation but returns log-probs explicitly. Uses the model's existing `use_cache=True` path (verified safe at training-time dtype in `model.py:454-455`).

### 5. Reference log-probs (`compute_logprobs`)

```python
@torch.no_grad()
def compute_logprobs(model, full_ids: torch.Tensor, gen_mask: torch.Tensor) -> torch.Tensor:
    """Run model on (B*G, P+T), return per-token log-prob of every position."""
    out = model(full_ids, use_cache=False)
    logits = out["logits"][:, :-1, :]                 # predict next-token at every position
    targets = full_ids[:, 1:]
    logp = F.log_softmax(logits.float(), dim=-1)
    tok_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return tok_logp[:, -gen_mask.shape[1]:] * gen_mask.float()  # (B*G, T)
```

Two-model variant calls this on the reference; single-model variant calls it on the trainable model with `no_grad`.

### 6. GRPO loss

```python
def grpo_loss(policy_logp: torch.Tensor,   # (B*G, T) requires grad
              ref_logp: torch.Tensor,       # (B*G, T) detached
              rewards: torch.Tensor,        # (B*G,)
              gen_mask: torch.Tensor,       # (B*G, T)
              kl_coef: float,
              clip_ratio: float) -> tuple[torch.Tensor, dict]:
    """
    Per-token objective (GRPO, Shao et al. 2024):
      advantage_i = (r_i - mean(r_group)) / (std(r_group) + eps)
      ratio_t     = exp(policy_logp_t - ref_logp_t)         (KL proxy)
      loss_t      = -min(advantage * ratio, advantage * clip(ratio, 1-eps, 1+eps))
      + kl_coef * (policy_logp_t - ref_logp_t)
    Mask all token-level terms by gen_mask, average over generated tokens.
    """
    B, T = policy_logp.shape
    G = B  # B*G flattened as B for math; rewards is (B,)
    # group-normalize: rewards shape (G,) replicated, so we need (B,G) reshape
    rewards_g = rewards.view(-1, G)                 # (num_prompts, G)
    adv = (rewards_g - rewards_g.mean(dim=1, keepdim=True)) / (rewards_g.std(dim=1, keepdim=True) + 1e-4)
    advantages = adv.view(-1)                        # (B,)

    log_ratio = (policy_logp - ref_logp) * gen_mask.float()     # (B, T)
    ratio = log_ratio.exp()
    surr1 = advantages.unsqueeze(1) * ratio
    surr2 = advantages.unsqueeze(1) * ratio.clamp(1 - clip_ratio, 1 + clip_ratio)
    pg_loss = -torch.min(surr1, surr2) * gen_mask.float()
    kl_term = log_ratio * kl_coef
    loss = (pg_loss + kl_term).sum() / gen_mask.float().sum().clamp(min=1.0)
    return loss, dict(pg=pg_loss.sum().item(), kl=kl_term.sum().item())
```

### 7. Reference model setup

```python
def build_reference(model, ref_policy: str, config, sft_ckpt_path, device):
    if ref_policy == "single":
        return None  # use the trainable model with no_grad
    elif ref_policy == "two":
        ref_model = Qwen3ForCausalLM(config).to(device)
        ref_model.load_state_dict(torch.load(sft_ckpt_path)["model_state"])
        ref_model.tie_weights()
        ref_model.eval()
        for p in ref_model.parameters(): p.requires_grad_(False)
        return ref_model
```

If LoRA is active in `single` mode, the reference is the trainable LoRA model with `requires_grad=False` on every parameter — close to SFT but includes the current LoRA delta. Acceptable: GRPO updates the LoRA delta by small amounts per step and the reference drift is bounded. (Documented in the script's docstring.)

### 8. Main training loop

```python
def train(args):
    rank, local_rank, world_size, device = setup_distributed()
    torch.manual_seed(args.seed + rank)
    # ... model + config + load SFT ckpt ...
    # ... optional LoRA injection ...
    # ... reference policy setup ...

    train_ds = GRPOPromptDataset(args.data_dir or args.prompts_file,
                                 tokenizer, args.max_prompt_len)

    optimizer = build_optimizer(model, args.lr, args.weight_decay)
    ctx = (torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
           if device.type == "cuda" and args.dtype == "bf16" else nullcontext())

    for step in range(start_step, args.max_steps):
        # 1. Sample a batch of prompts
        prompts, ground_truths = train_ds.sample_batch(args.batch_size)

        # 2. Roll out G completions per prompt (no_grad)
        full_ids, gen_mask, sampled_logp = generate_rollouts(
            model, prompts, args.max_new_tokens, args.temperature,
            args.top_p, eos_id)

        # 3. Compute rewards
        completions_text = tokenizer.decode_batch(full_ids[:, prompts.shape[1]:])
        rewards = torch.tensor([
            compute_reward(p, c, g, args.max_new_tokens)[0]
            for p, c, g in zip(prompts_text, completions_text, ground_truths)
        ], device=device)

        # 4. Compute reference log-probs
        with torch.no_grad():
            ref_logp = compute_logprobs(
                ref_model if ref_model is not None else model,
                full_ids, gen_mask)

        # 5. Recompute policy log-probs WITH grad
        out = model(full_ids, use_cache=False)
        logits = out["logits"][:, :-1, :]
        policy_logp = F.log_softmax(logits.float(), dim=-1).gather(
            -1, full_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        policy_logp = policy_logp[:, -gen_mask.shape[1]:] * gen_mask.float()

        # 6. Loss + backward
        loss, metrics = grpo_loss(policy_logp, ref_logp, rewards, gen_mask,
                                  args.kl_coef, args.clip_ratio)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        # 7. Logging + checkpointing (same pattern as train_sft.py)
```

### 9. CLI surface

Mirrors `train_sft.py` so muscle memory carries over:

```
python train_grpo.py \
    --checkpoint ./sft_checkpoints/latest.pt \
    --tokenizer ./tokenizer \
    --data-dir ./sft_data \
    --out-dir ./grpo_checkpoints \
    --ref-policy single \           # or "two"
    --num-generations 8 \
    --max-new-tokens 512 \
    --max-steps 500

# Optional LoRA + alternative prompts file:
python train_grpo.py --checkpoint ... --lora --lora-rank 64 \
    --prompts-file ./my_eval.jsonl --ref-policy two

# Merge LoRA after training:
python train_grpo.py --merge-lora \
    --checkpoint ./grpo_checkpoints/latest.pt --out-dir ./grpo_merged
```

Full arg list (matching the style of `train_sft.py`):
- Mode: `--merge-lora`
- Paths: `--checkpoint`, `--tokenizer`, `--data-dir`, `--prompts-file`, `--out-dir`, `--resume`
- LoRA: `--lora`, `--lora-rank`, `--lora-alpha`
- RL: `--ref-policy {single,two}`, `--num-generations`, `--max-new-tokens`, `--temperature`, `--top-p`, `--kl-coef`, `--clip-ratio`, `--reward-correct`, `--reward-format`
- Training: `--max-prompt-len`, `--max-steps`, `--lr`, `--min-lr`, `--warmup-steps`, `--weight-decay`, `--grad-clip`, `--dtype`, `--compile`, `--compile-mode`, `--seed`
- Logging: `--log-interval`, `--ckpt-interval`, `--keep-ckpts`, `--eval-prompts-file`, `--eval-every`

### 10. Smoke test
Bottom-of-file `smoke_test()` that:
1. Builds a tiny `Qwen3Config` + dummy tokenizer.
2. Writes a tiny JSONL of `{prompt, answer}` math records.
3. Runs 3 GRPO steps with `--num-generations 2 --max-new-tokens 16` and asserts the loss decreases.
4. Tests both `--ref-policy single` and `--ref-policy two` paths.

## Files touched
- **New:** `train_grpo.py` (~700 lines, single file).
- **Untouched:** `model.py`, `train.py`, `train_sft.py`, `pack_sft_data.py`. Reuses their public helpers and follows the same conventions (DDP, LoRA, cosine LR, checkpoint format, amp, compile).

## Verification

```bash
# 1. Smoke test (no GPU needed for the rollouts if --max-new-tokens is small)
python train_grpo.py

# 2. End-to-end on a tiny SFT checkpoint
python train_sft.py --model-size 0.3B --checkpoint ./checkpoints/latest.pt \
    --cache-dir ./sft_packed --max-steps 100 --out-dir ./sft_checkpoints
python train_grpo.py --checkpoint ./sft_checkpoints/latest.pt \
    --tokenizer ./tokenizer --data-dir ./sft_data \
    --num-generations 4 --max-new-tokens 256 --max-steps 50 \
    --ref-policy single --out-dir ./grpo_checkpoints

# 3. Two-model variant on a small model
python train_grpo.py --checkpoint ... --ref-policy two \
    --max-steps 50 --out-dir ./grpo_checkpoints_two

# 4. Merge LoRA
python train_grpo.py --merge-lora \
    --checkpoint ./grpo_checkpoints/latest.pt --out-dir ./grpo_merged
```

The script should produce monotonically decreasing loss curves, positive reward rates that increase over steps, and valid LoRA-merged checkpoints readable by `infer.py`.