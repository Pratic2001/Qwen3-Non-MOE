"""
Run this in your actual training environment (needs your tokenizer + data_dir).
Sanity-checks that val.bin is decodable, in-vocab, and not degenerate.

Usage: python check_val_sanity.py /path/to/data_dir
"""
import sys, json
import numpy as np

data_dir = sys.argv[1]
with open(f"{data_dir}/meta.json") as f:
    meta = json.load(f)

dtype = np.uint16 if meta["dtype"] == "uint16" else np.uint32
vocab_size = meta["vocab_size"]

for name in ("train.bin", "val.bin"):
    data = np.memmap(f"{data_dir}/{name}", dtype=dtype, mode="r")
    sample = data[:2000].astype(np.int64)
    n_oov = (sample >= vocab_size).sum()
    n_unique = len(np.unique(sample))
    print(f"{name}: {len(data):,} tokens | dtype={dtype.__name__} | "
          f"out-of-vocab in first 2000 toks: {n_oov} | unique in first 2000: {n_unique}")
    # A real dtype mismatch shows up as a wall of out-of-vocab ids or
    # near-zero unique tokens (everything aliasing to a couple of huge ints).

print("\nIf val.bin shows lots of out-of-vocab ids, or far fewer unique")
print("tokens than train.bin, the dtype/tokenizer used to pack it doesn't")
print("match meta.json — that's your bug, and it's in pack_dataset.py,")
print("not the train_*.py scripts.")
