"""
test_gen.py

Writes and executes a small, deterministic pytest suite against a
just-generated map_row() function, using the exact sample rows the model
was shown during codegen. This is the pipeline's own "write tests, then
run them" step -- but the tests themselves are templated by this module,
not invented by the LLM, so their correctness doesn't depend on the same
model that might have a bug in map_row() also getting the test right.

Why test map_row() against samples instead of just py_compiling it and
hoping: a script can compile perfectly and still be wrong in exactly the
way that matters -- reading the wrong column, mapping it to the wrong
field, returning None for every row, or crashing on the first real row
that has a null field the sample happened not to show. All of those pass
`python -m py_compile` and only show up once you actually call the
function. Running it here, before the full (potentially hours-long)
streaming download/crawl starts, turns that class of failure into a
sub-second local check with a specific, repair-able error message instead
of a wasted multi-hour run.

The generated test file is disposable (rewritten every attempt) but kept
on disk next to the generated script for inspection/debugging.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Optional


_TEST_TEMPLATE = '''"""Auto-generated pre-flight test for {module_name}.

Not written by the LLM -- templated by agent/test_gen.py using the real
sample rows fetched during discovery, and run before any full-scale
download/crawl starts. Rewritten on every codegen attempt; safe to delete.
"""
import importlib.util
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from schema_check import validate_record

_SAMPLE_ROWS = {sample_rows_json}
_MODE = {mode!r}

spec = importlib.util.spec_from_file_location("_under_test", {module_path!r})
_under_test = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_under_test)


def test_map_row_is_defined_and_callable():
    assert hasattr(_under_test, "map_row"), "module must define map_row(row)"
    assert callable(_under_test.map_row)


@pytest.mark.parametrize("row", _SAMPLE_ROWS)
def test_map_row_does_not_raise(row):
    _under_test.map_row(row)  # any exception fails this test with a real traceback


@pytest.mark.parametrize("row", _SAMPLE_ROWS)
def test_map_row_output_matches_schema_or_is_none(row):
    result = _under_test.map_row(row)
    if result is None:
        return  # an intentional skip is always valid
    ok, reason = validate_record(result, _MODE)
    assert ok, f"map_row returned a record that fails schema validation: {{reason}} -- got: {{result!r}}"


def test_map_row_maps_at_least_one_sample_row():
    """If every single sample row maps to None, either the column mapping
    is wrong or the function is unconditionally rejecting -- both are bugs
    worth catching here rather than discovering after a multi-hour run
    produced zero output."""
    mapped = [r for r in _SAMPLE_ROWS if _under_test.map_row(r) is not None]
    assert mapped, (
        "map_row(row) returned None for every single sample row -- this "
        "almost certainly means the column mapping is wrong (check the "
        "dataset's real column names) rather than every sample row being "
        "genuinely low-quality."
    )
'''


def write_pytest_file(module_path: str, sample_rows: list, mode: str, test_path: str,
                       agent_dir: Optional[str] = None) -> None:
    """Writes a pytest file at test_path that exercises map_row() (loaded
    from module_path) against sample_rows. agent_dir is added to the test's
    sys.path so `from schema_check import validate_record` resolves;
    defaults to this file's own directory."""
    agent_dir = agent_dir or os.path.dirname(os.path.abspath(__file__))
    # Keep the embedded sample small and JSON-safe; codegen's own
    # sample_hf_dataset()/sample_raw_batch() already guarantee this, but
    # default=str is a last-ditch safety net so a stray non-serializable
    # value degrades to a string instead of breaking test generation itself.
    sample_rows_json = json.dumps(sample_rows, ensure_ascii=False, default=str)
    content = _TEST_TEMPLATE.format(
        module_name=os.path.basename(module_path),
        sample_rows_json=sample_rows_json,
        mode=mode,
        module_path=os.path.abspath(module_path),
    )
    os.makedirs(os.path.dirname(test_path) or ".", exist_ok=True)
    with open(test_path, "w", encoding="utf-8") as f:
        f.write(content)
    # Point the generated test at agent_dir for its own imports without
    # relying on PYTHONPATH being set correctly by whoever runs pytest.
    with open(test_path, "r", encoding="utf-8") as f:
        body = f.read()
    body = body.replace(
        'sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))',
        f'sys.path.insert(0, {agent_dir!r})',
        1,
    )
    with open(test_path, "w", encoding="utf-8") as f:
        f.write(body)


def run_pytest_file(test_path: str, env: dict, timeout: float = 60.0) -> tuple:
    """Runs the generated test file with pytest in a subprocess (isolation:
    a genuinely broken map_row -- infinite loop, segfault-prone C extension,
    etc. -- can't take the whole pipeline process down with it, and a
    timeout guarantees the pipeline can't hang forever on one dataset).

    Returns (passed: bool, output: str)."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", test_path, "-q", "--no-header", "-x"],
            capture_output=True, text=True, timeout=timeout, env=env,
        )
        output = (result.stdout or "") + (result.stderr or "")
        return result.returncode == 0, output
    except subprocess.TimeoutExpired:
        return False, (f"pytest run on {test_path} exceeded {timeout}s timeout -- "
                        f"map_row likely contains a hang (infinite loop / blocking call) "
                        f"rather than a normal bug.")
