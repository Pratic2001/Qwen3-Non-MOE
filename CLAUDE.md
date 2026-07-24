# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A from-scratch PyTorch reproduction of the **Qwen3 dense (non-MoE) transformer** plus the full post-training pipeline: **pretrain → SFT → GRPO (RL) → inference**. The model is defined once in `model.py`; everything else is data preparation, training loops, checkpoint tooling, or inference / debugging utilities.

The model implements: RMSNorm (pre-norm), RoPE, Grouped Query Attention with **per-head QK-Norm before RoPE** (Qwen3's distinguishing trick), SwiGLU MLP, causal attention with KV-cache. `Qwen3Config.from_target_size(...)` searches `(hidden_size, num_layers, num_heads, num_kv_heads, intermediate_size)` combinations that hit a target parameter count while respecting Qwen3 conventions (`head_dim=128`, `gqa_ratio=4`, `mlp_ratio=3`, embedding tying for <2B).

`MANUAL.md` is the canonical Chinchilla-aware reference for sizing data, hyperparameters, and model sizes across all three training stages. Read it before sizing a run.

## File map

### Core model
- `model.py` — `Qwen3Config`, `RMSNorm`, `RoPE`, GQA+QK-Norm attention, SwiGLU MLP, decoder layer, full model + `generate()`. Imported by every training/inference script. Run `python model.py --target-size 0.6B` for a forward+generate smoke test.

### Data preparation
- `train_tokenizer.py` — Train a byte-level BPE tokenizer (Qwen3-family) with ChatML + `<think>`/`</think>` special tokens. Writes `./tokenizer/`.
- `pack_dataset.py` — Tokenize `./data/**/*.jsonl`, write `./packed/{train,val}.bin` + `meta.json` (uint16 or uint32 token ids). Supports `--seq-length N` to truncate individual documents to N tokens (useful for short-context training). The JSONL shape is documented in its own docstring; produce shards with `webscrapped_dataset_curator_AI_MCP/` or any compatible producer.
- `pack_sft_data.py` — Tokenize SFT JSONL (`{prompt, thinking, answer}` records), apply ChatML + `<think>` template, write packed `{tokens, mask}` memmaps with per-worker manifests (multi-process safe). Supports `--seq-length N` as a shorthand for `--max-len-per-example N`.
- `pack_grpo_data.py` — Single-turn ChatML packer for GRPO records. Output is the same on-disk format as `pack_sft_data.py` so `train_grpo.py` can read it via the same `SFTDataset` manifest convention. The packed `answer` is **not** used for the GRPO loss — it's only read back to recover the ground-truth string for reward scoring at rollout time. Supports `--seq-length N` as a shorthand for `--max-len-per-example N`.

> **Note.** The earlier `build_dataset.py` / `download_sft_data.py` /
> `download_grpo_data.py` scripts have been removed. The live producer
> of pretrain and SFT/GRPO JSONL shards is the
> `webscrapped_dataset_curator_AI_MCP/` agent (HuggingFace/Kaggle
> top-up + live web scrape). Write your own producer if you have a
> different source — only the JSONL shape on disk matters to the
> packers.

### Training
Pretrain:
- `train.py` — bf16, `torch.compile`, grad accumulation, cosine LR, AdamW, optional gradient checkpointing, DDP via `torchrun`, checkpoint save/resume, MFU, optional W&B.
- `train_deepspeed.py` — Pretraining with DeepSpeed. Auto-runs a hardware audit (VRAM, NVLink, IB) and picks ZeRO stage 1/2/3 + CPU offload. Writes `ds_config.json` to `--out-dir`.

SFT:
- `train_sft.py` — SFT reading packed memmaps. Supports LoRA on Q/K/V/O, DDP, `merge-lora`.
- `train_sft_deepspeed.py` — DeepSpeed twin of `train_sft.py`; mirrors the hardware audit / ZeRO selection / native sharded-checkpoint format of `train_deepspeed.py`.

GRPO (RL):
- `train_grpo.py` — Group Relative Policy Optimization. Consumes the SFT checkpoint + packed prompts, rolls out `G` completions per prompt with the current policy, scores with a rule-based reward (correctness + format + thinking bonus), group-normalises advantages, applies a PPO-style clipped policy gradient with optional KL penalty against a reference policy. Re-uses SFT packed memmaps by default (`--prompts-file` to override). Reference policy is switchable: `--ref-policy single` reuses the trainable model with `no_grad` (~10% VRAM overhead, frozen SFT reference); `two` keeps a second copy frozen in memory (DeepSeek-R1 recipe, ~2× VRAM).
- `train_grpo_deepspeed.py` — DeepSpeed twin of `train_grpo.py`. Re-uses the GRPO primitives (dataset, rollout generator, reward function, GRPO loss) verbatim and only swaps the distribution/optimizer layer for the DeepSpeed engine.

Checkpoint tooling:
- `deepspeed_shard_consolidator.py` — Convert a DeepSpeed checkpoint directory into a single `.pt` that `train_sft.py` / `infer.py` can load.

### Inference & debugging
- `infer.py` — Production-grade inference CLI. Auto-shards the model across visible GPUs / CPU / disk via `accelerate.dispatch_model`. Reads raw `.pt` checkpoints and transparently handles the LoRA-merged shape. bf16 default; optional bitsandbytes 4/8-bit (soft deps). DeepSpeed directories are rejected with a pointer at `deepspeed_shard_consolidator.py`.
- `load_for_inference.py` — Minimal reference loader showing the four things every custom load script must do: build, `load_state_dict`, `tie_weights()` (the easy one to forget — see Architecture notes), `eval()`.
- `diagnose_and_generate.py` — Diagnose the common "model outputs garbage" failure modes (KV-cache drift / position_ids, untied embeddings, train-mode dropout, tokenizer mismatch) and produce cleaner generations.
- `check_val_sanity.py <data_dir>` — Verify `val.bin` is decodable, in-vocab, and not degenerate (catches dtype/tokenizer mismatches in `pack_dataset.py`).

### Tooling
- `calculate_settings.py` — Chinchilla-aware calculator. Given `--data-size`, `--target-size`, or `--tokens`, prints a full set of recommended settings for pretraining, SFT, and GRPO plus the implied command line. Use `--json` for machine output.
- `make_anneal_dataset.py` — Replays the deterministic `PackedDataLoader` RNG stream from `train.py` / `train_deepspeed.py` to extract the "never sampled" token ranges and write an annealing `train.bin`. The seed formula (`seed*1_000_003 + rank*31 + loader_id`) and the per-worker seed formula (`base_seed + worker_id*9973`) are shared between both trainers and this script — keep them in sync if you change either trainer.

### Patch utilities
- `_apply_seq_length.py` — One-shot script that applies `--seq-length` support to `pack_sft_data.py` and `pack_grpo_data.py`. Already applied; kept for reference only.
- `_patch_pack_sft.py`, `_patch_sft.py`, `_write_patch.py` — Earlier iterations of the seq-length patch. Superseded by `_apply_seq_length.py`; kept for reference only.

### Web data agent
- `webscrapped_dataset_curator_AI_MCP/` — Separate subproject. Live-scraping companion that writes JSONL shards in the **same format** the packers (`pack_dataset.py` / `pack_sft_data.py` / `pack_grpo_data.py`) consume directly, so they need zero changes. Uses a local Ollama model as planner + quality judge and an MCP server for search/fetch/extract. Has its own `README.md` and `requirements.txt`.

### Environment files
- `hostfile` — `192.168.0.105 slots=1` / `192.168.0.111 slots=1` (multi-node DeepSpeed launch).
- `.deepspeed_env` — `NCCL_SOCKET_IFNAME=enp6s18,enp7s0`, `NCCL_IB_DISABLE=1`, `NCCL_DEBUG=INFO`. Keep this file when launching multi-node DeepSpeed jobs on the same network.
- `requirements.txt` — pinned: `torch==2.8.0`, `transformers==4.56.1`, `deepspeed==0.18.4`, `tokenizers==0.22.0`, `datasets==4.1.1`, plus CUDA 12.8 nvidia-* packages.

## End-to-end workflow

```bash
# 1. Train tokenizer
python train_tokenizer.py --data-dir ./data --vocab-size 32000 --out-dir ./tokenizer

# 2. Produce pretrain JSONL shards under ./data/<category>/*.jsonl
#    in the shape documented by pack_dataset.py. Use the live producer:
#       python agent/dataset_agent.py --target-size 5GB \
#           --categories web,knowledge,reasoning,code,math \
#           --out-dir ./data --mode pretrain
#    See webscrapped_dataset_curator_AI_MCP/README.md for setup, options,
#    and the on-disk JSONL shape contract.

# 3. Pack tokens
python pack_dataset.py --data-dir ./data --tokenizer ./tokenizer
#    Optional: --seq-length 512 to truncate long documents for short-context training

# 4a. Pretrain (single GPU)
python train.py --model-size 0.6B --data-dir ./packed --out-dir ./checkpoints
# 4a'. Pretrain (multi-GPU DDP)
torchrun --nproc_per_node=4 train.py --model-size 0.6B --data-dir ./packed
# 4b. Pretrain (DeepSpeed)
deepspeed --num_gpus 4 train_deepspeed.py --model-size 1.7B --data-dir ./packed
# 4c. Pretrain (multi-node)
deepspeed --hostfile hostfile train_deepspeed.py --model-size 8B --data-dir ./packed

# 5. SFT data
#    Produce SFT JSONL shards under ./sft_data/<category>/*.jsonl
#    in the {prompt, thinking, answer} shape. Same producer as step 2
#    with --mode sft:
#       python agent/dataset_agent.py --target-size 2GB \
#           --categories math,code,reasoning --out-dir ./sft_data --mode sft

# 6. Pack SFT
python pack_sft_data.py --data-dir ./sft_data --tokenizer ./tokenizer \
    --cache-dir ./sft_packed
#    Optional: --seq-length 512 to truncate long examples for short-context training

# 7. SFT (vanilla)
python train_sft.py --checkpoint ./checkpoints/latest.pt \
    --tokenizer ./tokenizer --cache-dir ./sft_packed --out-dir ./sft_checkpoints
# 7b. SFT (LoRA, recommended for 1B+ on a single GPU)
python train_sft.py --checkpoint ./checkpoints/latest.pt \
    --tokenizer ./tokenizer --cache-dir ./sft_packed \
    --lora --lora-rank 64 --lora-alpha 128 --out-dir ./sft_checkpoints
# 7c. SFT (DeepSpeed)
deepspeed --num_gpus 4 train_sft_deepspeed.py \
    --checkpoint ./checkpoints/latest.pt --tokenizer ./tokenizer \
    --cache-dir ./sft_packed --out-dir ./sft_checkpoints

# 8. Convert DeepSpeed checkpoint before downstream use
python deepspeed_shard_consolidator.py \
    --ds-dir ./checkpoints_ds/latest_ds --out ./checkpoints/pretrained.pt

# 9. Merge LoRA back into base model
python train_sft.py --merge-lora \
    --checkpoint ./sft_checkpoints/latest.pt --out-dir ./sft_merged

# 10. GRPO data
#     Produce GRPO JSONL shards under ./grpo_data/<category>/*.jsonl
#     in the {prompt, answer} shape (rule-based reward needs clean
#     numeric / boxed ground truths):
#        python agent/dataset_agent.py --target-size 2GB \
#            --categories math,reasoning --out-dir ./grpo_data --mode sft

# 11. Pack GRPO
python pack_grpo_data.py --data-dir ./grpo_data --tokenizer ./tokenizer \
    --cache-dir ./grpo_packed
#    Optional: --seq-length 512 to truncate long examples for short-context training

# 12. GRPO (consume the SFT checkpoint)
python train_grpo.py --checkpoint ./sft_checkpoints/latest.pt \
    --tokenizer ./tokenizer --cache-dir ./grpo_packed --out-dir ./grpo_checkpoints
# 12b. GRPO (DeepSpeed, multi-GPU/multi-node)
deepspeed --num_gpus 4 train_grpo_deepspeed.py \
    --checkpoint ./sft_checkpoints/latest.pt --tokenizer ./tokenizer \
    --cache-dir ./grpo_packed --out-dir ./grpo_checkpoints

# 13. Inference
python infer.py --checkpoint ./sft_checkpoints/latest.pt --tokenizer ./tokenizer \
    --prompt "Once upon a time"
```

## Common development commands

```bash
# Sanity-check the model: build, forward, generate with KV-cache
python model.py --target-size 0.6B

# Single-GPU smoke test (runs if ./packed/train.bin is missing)
python train.py --model-size 0.3B

# Quick GPU check
python test.py

# Verify a packed dataset isn't dtype-broken
python check_val_sanity.py ./packed

# Pick dataset/model sizes from Chinchilla
python calculate_settings.py --data-size 50GB
python calculate_settings.py --target-size 1.7B --json

# Diagnose "model outputs garbage"
python diagnose_and_generate.py --checkpoint ./checkpoints/latest.pt --tokenizer ./tokenizer

# Minimal load-and-generate reference (use as a template for custom loaders)
python load_for_inference.py --checkpoint ./checkpoints/pretrained.pt --tokenizer ./tokenizer
```

## Architecture notes worth knowing

- **Tied embeddings.** `Qwen3ForCausalLM` always allocates `lm_head` as an `nn.Linear`, but when `tie_word_embeddings=True` it points `lm_head.weight` at `embed_tokens.weight` immediately in `__init__`. After any `load_state_dict(...)` the tie is broken (the loader replaces the tensor), so `train.py`, `train_sft.py`, and `train_grpo.py` all call `tie_weights()` after resume. This dance exists specifically so `torch.compile` / CUDAGraphs doesn't trip on tied parameters. `load_for_inference.py` documents this as the #1 custom-loader bug — the symptom is gibberish like `Q:: am movie: you't the one the`.

- **CUDAGraphs interaction.** When using `--compile-mode reduce-overhead`, `_use_cudagraphs` is set and every forward (training **and** validation) must be preceded by `torch.compiler.cudagraph_mark_step_begin()`, otherwise you get "tensor overwritten by a subsequent run" errors.

- **Gradient checkpointing is incompatible with KV-cache**; `Qwen3Model.forward` forces `use_cache=False` when `gradient_checkpointing` is on during training. Don't enable cache in train mode.

- **`Qwen3Attention` defends against Q/K dtype drift.** `RMSNorm` runs in fp32 inside autocast-disabled scope; the defensive `q = q.to(v.dtype)` after RoPE ensures SDPA doesn't see mixed dtypes.

- **`PackedDataLoader`** shards the memmap across ranks (`rank * shard_size : (rank+1) * shard_size`), prebuilds the next batch on CPU while the GPU runs (`prime()` + `_prefetched`), and uses `pin_memory()` + `non_blocking=True` for the H2D copy. Its seed formula `seed*1_000_003 + rank*31 + loader_id` (with per-worker `base_seed + worker_id*9973`) is what `make_anneal_dataset.py` replays to derive unsampled token ranges.

- **`SFT` loss masking** lives in the `mask.bin` produced by `pack_sft_data.py` (1 = compute loss, 0 = mask to -100). Position-level mask also prevents loss from crossing sample boundaries inside packed windows. The same on-disk format is used for GRPO prompts, but the loss in the GRPO objective is computed against on-policy completions, not against this packed region.

- **DeepSpeed auto-ZeRO.** Both `train_deepspeed.py` and `train_sft_deepspeed.py` auto-select ZeRO stage from a hardware probe (VRAM/GPU, NVLink/PCIe, IB/Ethernet, CPU RAM). Force a stage with `--zero-stage` / `--cpu-offload-optimizer` to override. The generated `ds_config.json` is written to `--out-dir`. `train_grpo_deepspeed.py` mirrors the same probe.

- **MFU estimation** uses the standard `6 * N * tokens/sec` Chinchilla/PaLM formula over non-embedding parameters. The `GPU_PEAK_TFLOPS` table is duplicated across `train.py`, `train_deepspeed.py`, `train_sft_deepspeed.py`, and `train_grpo_deepspeed.py`; add new GPUs there if they aren't recognized (unknown GPUs fall back to a conservative 100 TFLOP/s with a warning).

- **LoRA merge path.** `--merge-lora` on `train_sft.py` writes a self-contained checkpoint with full base weights, so `infer.py` and `train_grpo.py` can load the result with no LoRA machinery.

- **GRPO rollout reuses the SFT packer** (`pack_sft_data.py`'s memmap format). This keeps RAM flat regardless of prompt pool size; the original JSONL shards are read only on-demand to recover ground-truth answer strings for the reward function.

## Environment

Hardware: training has been running on RTX 4090s (24 GB). Two-node multi-GPU cluster per `hostfile`. `.deepspeed_env` forces NCCL over `enp6s18`/`enp7s0` with IB disabled — keep that file when launching multi-node DeepSpeed jobs on the same network.

Python deps are pinned in `requirements.txt`: `torch==2.8.0`, `transformers==4.56.1`, `deepspeed==0.18.4`, `tokenizers==0.22.0`, `datasets==4.1.1`, plus CUDA 12.8 nvidia-* packages.
