#!/usr/bin/env python3
"""Write the actual patch script to _apply_seq_length.py"""
import textwrap

code = textwrap.dedent("""\
#!/usr/bin/env python3
"Apply --seq-length changes to pack_sft_data.py and pack_grpo_data.py."

# =====================================================================
# PART 1: pack_sft_data.py
# =====================================================================

with open("pack_sft_data.py") as f:
    lines = f.readlines()

# Find format_and_tokenise function boundaries (0-indexed)
func_start = None
func_end = None
for i, line in enumerate(lines):
    if line.startswith("def format_and_tokenise("):
        func_start = i
    if func_start is not None and i > func_start and line and not line[0].isspace() and line.strip():
        func_end = i
        break
if func_end is None:
    func_end = len(lines)

print(f"  format_and_tokenise: lines {func_start+1}-{func_end}")

# Build the replacement function
new_func = [
    'def format_and_tokenise(\\n',
    '    record: dict,\\n',
    '    tokenizer: Tokenizer,\\n',
    '    max_len: int = 2048,\\n',
    '    seq_length: Optional[int] = None,\\n',
    ') -> Optional[Tuple[List[int], List[int]]]:\\n',
]

# We need to write the actual content. Let's do it differently.
# Write lines 1 through func_start (keep preamble)
# Then write new function
# Then write lines func_end through end

with open("pack_sft_data.py.new", "w") as out:
    # Write everything before the function
    out.writelines(lines[:func_start])
    # Write new function (will be populated below)
    # ... this approach is too complex for inline generation
    # Let's use a simpler sed-like approach instead

print("Script generated - will use different approach")
""")

with open("_apply_seq_length.py", "w") as f:
    f.write(code)

print("Wrote _apply_seq_length.py")
