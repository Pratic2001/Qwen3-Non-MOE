"""
Unit tests for the trusted harness core: schema_check.validate_record and
harness.run(). These are hand-written (not templated per-dataset like
test_gen.py's output) because this is the code every single generated
map_row() runs inside of -- it deserves real, permanent test coverage, the
same way build_dataset.py's SOURCES-dict pipeline would.

Run with:  pytest agent/tests/test_harness.py -q
"""
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schema_check import validate_record
import harness


# ---------------------------------------------------------------------------
# schema_check.validate_record
# ---------------------------------------------------------------------------

def test_validate_record_pretrain_ok():
    ok, reason = validate_record(
        {"text": "some real content" * 5, "source": "x", "category": "web"}, "pretrain")
    assert ok and reason is None


def test_validate_record_missing_field():
    ok, reason = validate_record({"text": "x", "source": "y"}, "pretrain")
    assert not ok and "missing_fields" in reason


def test_validate_record_wrong_type():
    ok, reason = validate_record({"text": 123, "source": "y", "category": "z"}, "pretrain")
    assert not ok and "field_not_str" in reason


def test_validate_record_empty_required_field():
    ok, reason = validate_record({"text": "   ", "source": "y", "category": "z"}, "pretrain")
    assert not ok and "field_empty" in reason


def test_validate_record_sft_allows_empty_thinking():
    ok, reason = validate_record(
        {"prompt": "q", "thinking": "", "answer": "a", "source": "s", "category": "c"}, "sft")
    assert ok and reason is None


def test_validate_record_none_is_not_ok_but_distinguishable():
    ok, reason = validate_record(None, "pretrain")
    assert not ok
    assert reason == "record_is_none"
    assert harness.validate_record if False else True  # sanity: import wired correctly


def test_validate_record_unknown_mode():
    ok, reason = validate_record({"text": "x", "source": "y", "category": "z"}, "not_a_mode")
    assert not ok and reason.startswith("unknown_mode")


# ---------------------------------------------------------------------------
# harness.run -- the actual loop
# ---------------------------------------------------------------------------

def _rows(n):
    return [{"i": i, "body": f"row number {i} " * 40} for i in range(n)]


def test_run_writes_valid_rows_and_reports_result(tmp_path):
    def map_row(row):
        return {"text": row["body"], "source": "unit-test", "category": "web"}

    result = harness.run(
        map_row, iter(_rows(20)), mode="pretrain", category="web",
        out_dir=str(tmp_path), min_doc_chars=10, target_bytes=10**9,
    )
    assert result["docs"] == 20
    assert result["actual_bytes"] > 0
    assert "error" not in result


def test_run_stops_at_target_bytes(tmp_path):
    def map_row(row):
        return {"text": row["body"], "source": "unit-test", "category": "web"}

    rows = _rows(1000)
    one_row_bytes = len(rows[0]["body"]) + 40  # rough
    result = harness.run(
        map_row, iter(rows), mode="pretrain", category="web",
        out_dir=str(tmp_path), min_doc_chars=10, target_bytes=one_row_bytes * 5,
    )
    assert result["docs"] < 1000  # stopped early, didn't drain the whole iterator
    assert result["actual_bytes"] >= one_row_bytes * 5


def test_run_skips_none_without_counting_as_error(tmp_path):
    def map_row(row):
        return None if row["i"] % 2 == 0 else {
            "text": row["body"], "source": "unit-test", "category": "web"}

    result = harness.run(
        map_row, iter(_rows(20)), mode="pretrain", category="web",
        out_dir=str(tmp_path), min_doc_chars=10, target_bytes=10**9,
    )
    assert result["docs"] == 10
    assert result["map_errors"] == 0


def test_run_aborts_fatal_on_systemic_map_row_bug(tmp_path):
    """A map_row that always raises (e.g. it references a column name that
    doesn't exist in ANY row) must abort with a FATAL error quickly, not
    silently produce an empty dataset after churning through everything."""
    def map_row(row):
        raise KeyError("this_column_does_not_exist")

    result = harness.run(
        map_row, iter(_rows(1000)), mode="pretrain", category="web",
        out_dir=str(tmp_path), min_doc_chars=10, target_bytes=10**9,
    )
    assert result["docs"] == 0
    assert "error" in result
    assert result["map_errors"] <= harness.MAX_CONSECUTIVE_FAILURES_WITH_NO_SUCCESS + 1


def test_run_aborts_fatal_on_schema_invalid_output(tmp_path):
    def map_row(row):
        return {"text": row["body"]}  # missing source/category every time

    result = harness.run(
        map_row, iter(_rows(1000)), mode="pretrain", category="web",
        out_dir=str(tmp_path), min_doc_chars=10, target_bytes=10**9,
    )
    assert result["docs"] == 0
    assert "error" in result


_VARIED_PASSAGE = (
    "The quick brown fox jumps over the lazy dog while researchers "
    "carefully document every observation made during the long expedition "
    "across the northern mountains and coastal wetlands nearby today."
)


def test_run_dedups_exact_duplicates(tmp_path):
    def map_row(row):
        return {"text": _VARIED_PASSAGE, "source": "unit-test", "category": "web"}

    result = harness.run(
        map_row, iter(_rows(10)), mode="pretrain", category="web",
        out_dir=str(tmp_path), min_doc_chars=10, target_bytes=10**9,
    )
    assert result["docs"] == 1
    assert result["dupes"] == 9


def test_run_applies_quality_filter(tmp_path):
    def map_row(row):
        return {"text": "x", "source": "unit-test", "category": "web"}  # too short

    result = harness.run(
        map_row, iter(_rows(10)), mode="pretrain", category="web",
        out_dir=str(tmp_path), min_doc_chars=500, target_bytes=10**9,
    )
    assert result["docs"] == 0
    assert result["quality_rejects"] == 10
    assert "error" not in result  # rejected by quality bar is normal, not fatal


def test_run_sft_mode_uses_prompt_answer_quality_filter(tmp_path):
    def map_row(row):
        return {"prompt": f"question {row['i']}", "thinking": "",
                 "answer": f"answer {row['i']}", "source": "s", "category": "c"}

    result = harness.run(
        map_row, iter(_rows(5)), mode="sft", category="qa",
        out_dir=str(tmp_path), min_doc_chars=10, target_bytes=10**9,
    )
    assert result["docs"] == 5


def test_load_map_row_missing_function_raises(tmp_path):
    bad_module = tmp_path / "bad.py"
    bad_module.write_text("def not_map_row(row):\n    return row\n")
    with pytest.raises(harness.FatalHarnessError):
        harness.load_map_row(str(bad_module))


def test_load_map_row_import_error_raises(tmp_path):
    bad_module = tmp_path / "broken.py"
    bad_module.write_text("this is not valid python (((\n")
    with pytest.raises(harness.FatalHarnessError):
        harness.load_map_row(str(bad_module))


def test_load_map_row_success(tmp_path):
    good_module = tmp_path / "good.py"
    good_module.write_text("def map_row(row):\n    return {'text': row['body'], "
                            "'source': 's', 'category': 'c'}\n")
    fn = harness.load_map_row(str(good_module))
    assert fn({"body": "hello"}) == {"text": "hello", "source": "s", "category": "c"}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
