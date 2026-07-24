#!/usr/bin/env python3
"""Apply --seq-length support to pack_sft_data.py and pack_grpo_data.py."""

def read_lines(p):
    with open(p) as f:
        return f.readlines()

def write_lines(p, lines):
    with open(p, "w") as f:
        f.writelines(lines)

def patch_pack_sft():
    print("Patching pack_sft_data.py ...")
    L = read_lines("pack_sft_data.py")
    out = []
    i = 0
    patched_docstring = False
    patched_guard = False

    while i < len(L):
        line = L[i]

        # 1. Replace format_and_tokenise docstring (lines 109-117, 0-indexed 108-116)
        if not patched_docstring and i == 108 and '"""' in line:
            out.append('    """
')
            out.append('    Format one SFT record into token ids + loss mask.
')
            out.append('
')
            out.append('    Returns (input_ids, loss_mask) where loss_mask[i] = 1 means position i
')
            out.append('    contributes to the cross-entropy loss (i.e. it is part of the assistant
')
            out.append('    turn), and 0 means it is masked out (prompt tokens).
')
            out.append('
')
            out.append('    When seq_length is set, the raw prompt / thinking / answer text is
')
            out.append('    truncated to fit within seq_length tokens BEFORE the ChatML template
')
            out.append('    is applied. The existing max_len check acts as a secondary safety net.
')
            out.append('
')
            out.append('    Returns None if the formatted example exceeds max_len tokens.
')
            out.append('    """
')
            patched_docstring = True
            # skip old docstring: lines 109-117 (0-indexed 108-116)
            i = 117  # skip to line after closing triple-quote
            continue

        # 2. Insert pre-template truncation after the guard clause
        # Line 123 (0-indexed 122) = if not prompt or not answer:
        # Line 124 (0-indexed 123) =     return None
        if not patched_guard and i == 122 and 'if not prompt or not answer:' in line:
            out.append(line)       # if not prompt...
            out.append(L[i+1])     # return None
            out.append('
')
            out.append('    # ---- Pre-template truncation when seq_length is set ----
')
            out.append('    # Truncate raw text fields BEFORE building ChatML templates so the
')
            out.append('    # final packed sequence is guaranteed to fit within seq_length tokens.
')
            out.append('    if seq_length is not None:
')
            out.append('        # Compute template overhead from actual token counts
')
            out.append('        _utpl = _BOS + chr(10) + "user" + chr(10) + _EOS + chr(10)
')
            out.append('        _atpl = _BOS + chr(10) + "assistant" + chr(10)
')
            out.append('        _atpl_think = _BOS + chr(10) + _THO + chr(10) + "" + chr(10) + _THC + chr(10) + "" + chr(10) + _EOS + chr(10)
')
            out.append('        _atpl_nothink = _BOS + chr(10) + "" + chr(10) + _EOS + chr(10)
')
            out.append('        _n_user_overhead = len(_utpl)
')
            out.append('        _n_asst_overhead_think = len(_atpl_think)
')
            out.append('        _n_asst_overhead_nothink = len(_atpl_nothink)
')
            out.append('        _n_asst_overhead = _n_asst_overhead_think if thinking else _n_asst_overhead_nothink
')
            out.append('        _total_overhead = _n_user_overhead + _n_asst_overhead
')
            out.append('        _budget = seq_length - _total_overhead
')
            out.append('        if _budget < 4:
')
            out.append('            return None
')
            out.append('
')
            out.append('        # Measure raw token counts
')
            out.append('        _np = len(tokenizer.encode(prompt, add_special_tokens=False).ids)
')
            out.append('        _nt = len(tokenizer.encode(thinking, add_special_tokens=False).ids) if thinking else 0
')
            out.append('        _na = len(tokenizer.encode(answer, add_special_tokens=False).ids)
')
            out.append('        _raw_total = _np + _nt + _na
')
            out.append('
')
            out.append('        if _raw_total > _budget:
')
            out.append('            # Scale proportionally to fit within budget
')
            out.append('            _scale = _budget / _raw_total
')
            out.append('            _pb = max(0, int(_np * _scale))
')
            out.append('            _tb = max(0, int(_nt * _scale)) if thinking else 0
')
            out.append('            _ab = max(1, int(_na * _scale))  # always keep >= 1 answer token
')
            out.append('            # Give any remaining budget back to answer, then thinking, then prompt
')
            out.append('            _remaining = _budget - _pb - _tb - _ab
')
            out.append('            if _remaining > 0 and _na > _ab:
')
            out.append('                _extra = min(_remaining, _na - _ab)
')
            out.append('                _ab += _extra
')
            out.append('                _remaining -= _extra
')
            out.append('            if _remaining > 0 and _nt > _tb:
')
            out.append('                _extra = min(_remaining, _nt - _tb)
')
            out.append('                _tb += _extra
')
            out.append('                _remaining -= _extra
')
            out.append('            if _remaining > 0 and _np > _pb:
')
            out.append('                _pb += min(_remaining, _np - _pb)
')
            out.append('
')
            out.append('            prompt   = _truncate_text(prompt, _pb, tokenizer)
')
            out.append('            if thinking:
')
            out.append('                thinking = _truncate_text(thinking, _tb, tokenizer)
')
            out.append('            answer   = _truncate_text(answer, _ab, tokenizer)
')
            out.append('
')
            patched_guard = True
            i = 124  # skip past the original guard + return None
            continue

        out.append(line)
        i += 1

    # 3. Add _truncate_text helper and ChatML token vars before format_and_tokenise
    # Find the insertion point
    final_out = []
    for idx, line in enumerate(out):
        final_out.append(line)
        if '# Chat template formatting' in line and 'tokenisation' in line:
            # Insert after the next blank line / separator
            pass
        if 'def format_and_tokenise(' in line and 'record: dict' in out[idx+1] if idx+1 < len(out) else False:
            # Insert helper + token vars right before this line
            helper_lines = [
                '
',
                '# ChatML token strings (built at runtime to avoid angle-bracket parsing issues)
',
                '_L = chr(60)
',
                '_G = chr(62)
',
                '_BOS = _L + "im_start" + _G
',
                '_EOS = _L + "im_end" + _G
',
                '_THO = _L + "think" + _G
',
                '_THC = _L + "/think" + _G
',
                '
',
                'def _truncate_text(text, max_tokens, tokenizer):
',
                '    """Truncate raw text to at most max_tokens tokens."""
',
                '    if max_tokens <= 0:
',
                '        return ""
',
                '    ids = tokenizer.encode(text, add_special_tokens=False).ids
',
                '    if len(ids) <= max_tokens:
',
                '        return text
',
                '    return tokenizer.decode(ids[:max_tokens])
',
                '
',
                '
',
            ]
            # insert before current line
            insert_at = len(final_out) - 1
            for h in reversed(helper_lines):
                final_out.insert(insert_at, h)
            break  # we found and inserted before format_and_tokenise

    # 4. Add seq_length to pack_worker_shard calls (both passes)
    final_text = ''.join(final_out)
    final_text = final_text.replace(
        'result = format_and_tokenise(rec, tokenizer, max_len=max_len_per_example)
                if result is None:
                    continue
                ids, _ = result
                n_tok = len(ids) + 1',
        'result = format_and_tokenise(rec, tokenizer, max_len=max_len_per_example, seq_length=seq_length)
                if result is None:
                    continue
                ids, _ = result
                n_tok = len(ids) + 1'
    )
    final_text = final_text.replace(
        'result = format_and_tokenise(rec, tokenizer, max_len=max_len_per_example)
                if result is None:
                    continue

                ids, lmask = result',
        'result = format_and_tokenise(rec, tokenizer, max_len=max_len_per_example, seq_length=seq_length)
                if result is None:
                    continue

                ids, lmask = result'
    )

    # 5. Add seq_length to manifest
    final_text = final_text.replace(
        '        "max_len_per_example": max_len_per_example,
        "val_fraction":    val_fraction,',
        '        "max_len_per_example": max_len_per_example,
        "seq_length":      seq_length,
        "val_fraction":    val_fraction,'
    )

    # 6. Add --seq-length CLI argument
    final_text = final_text.replace(
        '    p.add_argument("--max-len-per-example", type=int, default=2048,
                   help="Max tokens per individual SFT example before truncation")
    p.add_argument("--val-fraction", type=float, default=0.01,',
        '    p.add_argument("--max-len-per-example", type=int, default=2048,
                   help="Max tokens per individual SFT example before truncation")
    p.add_argument("--seq-length", type=int, default=None,
                   help="Max tokens per packed record (truncates raw text before
                        applying ChatML template). Default None (no truncation).")
    p.add_argument("--val-fraction", type=float, default=0.01,'
    )

    # 7. Pass seq_length from CLI to pack_worker_shard
    final_text = final_text.replace(
        '        vocab_size=tokenizer.get_vocab_size(),
    )

    print(f"\nDone.',
        '        vocab_size=tokenizer.get_vocab_size(),
        seq_length=args.seq_length,
    )

    print(f"\nDone.'
    )

    # Verify
    checks = [
        ('_truncate_text helper', '_truncate_text' in final_text),
        ('seq_length in format_and_tokenise', 'seq_length=seq_length' in final_text),
        ('seq_length in manifest', '"seq_length":' in final_text),
        ('--seq-length CLI', '"--seq-length"' in final_text),
        ('seq_length passed from CLI', 'seq_length=args.seq_length' in final_text),
        ('Pre-template truncation', 'Pre-template truncation' in final_text),
    ]
    all_ok = True
    for name, ok in checks:
        status = 'OK' if ok else 'MISSING'
        print(f'  {status}: {name}')
        if not ok:
            all_ok = False

    if all_ok:
        write_lines('pack_sft_data.py', [final_text])
        print('pack_sft_data.py patched successfully!')
    else:
        print('ERRORS - not writing file')
        sys.exit(1)


if __name__ == '__main__':
    patch_pack_sft()
