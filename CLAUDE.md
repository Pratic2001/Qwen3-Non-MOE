# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A from-scratch PyTorch reproduction of the **Qwen3 dense (non-MoE) transformer** plus a complete pretraining → SFT → RL pipeline. The model is defined in `model.py`; everything else is data preparation, training loops, or checkpoint tooling.

The model implements: RMSNorm (pre-norm), RoPE, Grouped Query Attention with **per-head QK-Norm before RoPE** (Qwen3's distinguishing trick), SwiGLU MLP, causal attention with KV-cache. `Qwen3Config.from_target_size(...)` searches (hidden_size, num_layers, num_heads, num_kv_heads, intermediate_size) combinations that hit a target parameter count while respecting Qwen3 conventions (head_dim=128, gqa_ratio=4, mlp_ratio=3, embedding tying for <2B).

## File map and dependencies

```
model.py                       # Qwen3Config, RMSNorm, RoPE, GQA+QK-Norm attention,
                               # SwiGLU MLP, decoder layer, full model + generate().
                               # Imported by every training script.

train_tokenizer.py             # Train a byte-level BPE tokenizer (Qwen3-family)
                               # with ChatML + <think>/</think> special tokens.
                               # Writes ./tokenizer/.

build_dataset.py               # Stream from HuggingFace (FineWeb, TheStack,
                               # FineMath, Wikipedia, OpenOrca) and write
                               # JSONL shards to ./data/<category>/.

pack_dataset.py                # Tokenize ./data/**/*.jsonl with the trained
                               # tokenizer, write ./packed/{train,val}.bin
                               # + meta.json (uint16 or uint32 token ids).

train.py                       # Pretraining loop. bf16, torch.compile,
                               # grad accumulation, cosine LR, AdamW, optional
                               # gradient checkpointing, DDP via torchrun,
                               # checkpoint save/resume, MFU, optional W&B.

train_deepspeed.py             # Pretraining loop with DeepSpeed. Auto-runs a
                               # hardware audit (VRAM, NVLink, IB) and picks
                               # ZeRO stage 1/2/3 + CPU offload. Writes
                               # ds_config.json. Launches with `deepspeed`.

deepspeed_shard_consolidator.py  # Convert a DeepSpeed checkpoint directory
                                 # into a single .pt that train_sft.py can load.

download_sft_data.py           # Download/format SFT records (NuminaMath,
                               # Evol-Instruct-Code, OpenThoughts, ARC, etc.)
                               # with {prompt, thinking, answer} schema to
                               # ./sft_data/<category>/.

pack_sft_data.py               # Tokenize SFT JSONL, apply ChatML + <think>
                               # template, write packed {tokens, mask} memmaps
                               # with per-worker manifests (multi-process safe).

sft_standalone.py              # SFT loop reading raw JSONL (older entry point).
train_sft.py                   # SFT loop reading packed memmaps from
                               # pack_sft_data.py (newer, preferred). Both
                               # support LoRA on Q/K/V/O, DDP, merge-lora.

test.py                        # Trivial CUDA availability check.
.deepspeed_env                 # NCCL env: SOCKET_IFNAME=enp6s18,enp7s0,
                               # IB_DISABLE=1, DEBUG=INFO.
hostfile                       # 192.168.0.105 slots=1 / 192.168.0.111 slots=1
                               # (multi-node DeepSpeed launch).
```

## End-to-end workflow

```bash
# 1. Train tokenizer
python train_tokenizer.py --data-dir ./data --vocab-size 32000 --out-dir ./tokenizer

# 2. Build pretraining corpus
python build_dataset.py --target-size 5GB

# 3. Pack tokens
python pack_dataset.py --data-dir ./data --tokenizer ./tokenizer

# 4a. Pretrain (single GPU)
python train.py --model-size 0.6B --data-dir ./packed --out-dir ./checkpoints
# 4a. Pretrain (multi-GPU)
torchrun --nproc_per_node=4 train.py --model-size 0.6B --data-dir ./packed
# 4b. Pretrain (DeepSpeed)
deepspeed --num_gpus 4 train_deepspeed.py --model-size 1.7B --data-dir ./packed
# 4c. Pretrain (multi-node)
deepspeed --hostfile hostfile train_deepspeed.py --model-size 8B --data-dir ./packed

# 5. SFT data
python download_sft_data.py --target-size 2GB

# 6. Pack SFT
python pack_sft_data.py --data-dir ./sft_data --tokenizer ./tokenizer \
    --cache-dir ./sft_packed

# 7. SFT
python train_sft.py --checkpoint ./checkpoints/latest.pt \
    --tokenizer ./tokenizer --cache-dir ./sft_packed --out-dir ./sft_checkpoints

# 7b. LoRA SFT (recommended for 1B+ on a single GPU)
python train_sft.py --checkpoint ./checkpoints/latest.pt \
    --tokenizer ./tokenizer --cache-dir ./sft_packed \
    --lora --lora-rank 64 --lora-alpha 128 --out-dir ./sft_checkpoints

# 8. Convert DeepSpeed checkpoint before downstream use
python deepspeed_shard_consolidator.py \
    --ds-dir ./checkpoints_ds/latest_ds --out ./checkpoints/pretrained.pt

# 9. Merge LoRA back into base model
python train_sft.py --merge-lora \
    --checkpoint ./sft_checkpoints/latest.pt --out-dir ./sft_merged
```

## Common development commands

```bash
# Sanity-check the model: build, forward, generate with KV-cache
python model.py --target-size 0.6B

# Single-GPU smoke test (runs if ./packed/train.bin is missing)
python train.py --model-size 0.3B

# Quick GPU check
python test.py
```

## Architecture notes worth knowing

- **Tied embeddings.** `Qwen3ForCausalLM` always allocates `lm_head` as an `nn.Linear`, but when `tie_word_embeddings=True` it points `lm_head.weight` at `embed_tokens.weight` immediately in `__init__`. After any `load_state_dict(...)` the tie is broken (the loader replaces the tensor), so `train.py` and `train_sft.py` both call `tie_weights()` after resume. This dance exists specifically so `torch.compile` / CUDAGraphs doesn't trip on tied parameters.
- **CUDAGraphs interaction.** When using `--compile-mode reduce-overhead`, `_use_cudagraphs` is set and every forward (training **and** validation) must be preceded by `torch.compiler.cudagraph_mark_step_begin()`, otherwise you get "tensor overwritten by a subsequent run" errors.
- **Gradient checkpointing is incompatible with KV-cache**; `Qwen3Model.forward` forces `use_cache=False` when `gradient_checkpointing` is on during training. Don't enable cache in train mode.
- **`Qwen3Attention` defends against Q/K dtype drift.** `RMSNorm` runs in fp32 inside autocast-disabled scope; the defensive `q = q.to(v.dtype)` after RoPE ensures SDPA doesn't see mixed dtypes.
- **`PackedDataLoader`** shards the memmap across ranks (`rank * shard_size : (rank+1) * shard_size`), prebuilds the next batch on CPU while the GPU runs (`prime()` + `_prefetched`), and uses `pin_memory()` + `non_blocking=True` for the H2D copy.
- **`SFT` loss masking** lives in the `mask.bin` produced by `pack_sft_data.py` (1 = compute loss, 0 = mask to -100). Position-level mask also prevents loss from crossing sample boundaries inside packed windows.
- **`train_deepspeed.py` auto-selects ZeRO** from a hardware probe (VRAM/GPU, NVLink/PCIe, IB/Ethernet, CPU RAM). Force a stage with `--zero-stage` / `--cpu-offload-optimizer` to override. The generated `ds_config.json` is written to `--out-dir`.
- **MFU estimation** uses the standard `6 * N * tokens/sec` Chinchilla/PaLM formula over non-embedding parameters. The `GPU_PEAK_TFLOPS` table is duplicated in `train.py` and `train_deepspeed.py`; add new GPUs there if they aren't recognized (unknown GPUs fall back to a conservative 100 TFLOP/s with a warning).

## Environment

Hardware: training has been running on RTX 4090s (24 GB). Two-node multi-GPU cluster per `hostfile`. `.deepspeed_env` forces NCCL over `enp6s18`/`enp7s0` with IB disabled — keep that file when launching multi-node DeepSpeed jobs on the same network.

`requirements.txt` is pinned: `torch==2.8.0`, `transformers==4.56.1`, `deepspeed==0.18.4`, `tokenizers==0.22.0`, `datasets==4.1.1`, plus CUDA 12.8 nvidia-* packages.