from unittest.mock import patch

import pytest
from work_matching import file_replacement
from work_matching.file_replacement import replace_file_lines


def test_replace_file_lines_existing_file_holds_only_new_lines(tmp_path) -> None:
    path = tmp_path / "corrections.jsonl"
    path.write_text("old line\n", encoding="utf-8")
    replace_file_lines(path, ["first\n", "second\n"])
    assert path.read_text(encoding="utf-8") == "first\nsecond\n"


def test_replace_file_lines_missing_folder_created(tmp_path) -> None:
    path = tmp_path / "seed_cleaning" / "corrections.jsonl"
    replace_file_lines(path, ["line\n"])
    assert path.read_text(encoding="utf-8") == "line\n"


def test_replace_file_lines_failed_write_keeps_old_file_and_removes_temporary_file(
    tmp_path,
) -> None:
    path = tmp_path / "corrections.jsonl"
    path.write_text("old line\n", encoding="utf-8")
    with (
        patch.object(file_replacement.os, "fsync", autospec=True, side_effect=OSError),
        pytest.raises(OSError),
    ):
        replace_file_lines(path, ["new line\n"])
    assert (path.read_text(encoding="utf-8"), list(tmp_path.iterdir())) == (
        "old line\n",
        [path],
    )


def test_replace_file_lines_existing_file_keeps_mode(tmp_path) -> None:
    path = tmp_path / "corrections.jsonl"
    path.write_text("old line\n", encoding="utf-8")
    path.chmod(0o640)
    replace_file_lines(path, ["new line\n"])
    assert path.stat().st_mode & 0o777 == 0o640
