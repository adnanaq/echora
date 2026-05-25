"""Unit tests for common.utils.jsonl_utils."""

import json
import logging
from pathlib import Path
from unittest.mock import patch

import pytest
from common.utils.jsonl_utils import append_jsonl, write_jsonl  # noqa: E402

# =============================================================================
# append_jsonl
# =============================================================================


def test_append_jsonl_writes_record(tmp_path: Path) -> None:
    out = tmp_path / "out.jsonl"
    append_jsonl(str(out), {"key": "value"})
    assert out.read_text(encoding="utf-8").strip() == '{"key": "value"}'


def test_append_jsonl_appends_multiple_records(tmp_path: Path) -> None:
    out = tmp_path / "out.jsonl"
    append_jsonl(str(out), {"n": 1})
    append_jsonl(str(out), {"n": 2})
    lines = out.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == [{"n": 1}, {"n": 2}]


def test_append_jsonl_creates_parent_dirs(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "dir" / "out.jsonl"
    append_jsonl(str(out), {"x": 1})
    assert out.exists()


def test_append_jsonl_logs_on_write_failure(caplog: pytest.LogCaptureFixture) -> None:
    with patch("os.makedirs"), patch("builtins.open", side_effect=OSError("disk full")):
        with caplog.at_level(logging.WARNING, logger="common.utils.jsonl_utils"):
            append_jsonl("/bad/path.jsonl", {"key": "value"})
    assert "disk full" in caplog.text


# =============================================================================
# write_jsonl
# =============================================================================


def test_write_jsonl_writes_all_records(tmp_path: Path) -> None:
    out = tmp_path / "out.jsonl"
    write_jsonl(str(out), [{"a": 1}, {"b": 2}])
    lines = out.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == [{"a": 1}, {"b": 2}]


def test_write_jsonl_overwrites_existing(tmp_path: Path) -> None:
    out = tmp_path / "out.jsonl"
    write_jsonl(str(out), [{"first": True}])
    write_jsonl(str(out), [{"second": True}])
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == {"second": True}


def test_write_jsonl_creates_parent_dirs(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "out.jsonl"
    write_jsonl(str(out), [{"x": 1}])
    assert out.exists()
