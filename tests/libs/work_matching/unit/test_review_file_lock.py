import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from work_matching.errors import ReviewFileLockedError
from work_matching.review_file_lock import (
    hold_review_file_lock,
    is_review_file_locked,
)


@pytest.fixture
def review_file_path(tmp_path) -> Path:
    path = tmp_path / "review.jsonl"
    path.write_text("", encoding="utf-8")
    return path


@pytest.fixture
def lock_held_by_other_process(review_file_path) -> Iterator[None]:
    holder = subprocess.Popen(  # noqa: S603  (fixed test arguments)
        [
            sys.executable,
            "-c",
            "import sys, time, filelock\n"
            "with filelock.FileLock(sys.argv[1]):\n"
            "    print('held', flush=True); time.sleep(5)",
            f"{review_file_path}.lock",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert holder.stdout is not None
    assert holder.stdout.readline().strip() == "held"
    yield
    holder.kill()
    holder.wait()


def test_is_review_file_locked_free_file_reports_false(review_file_path) -> None:
    assert not is_review_file_locked(review_file_path)


def test_is_review_file_locked_held_by_other_process_reports_true(
    review_file_path, lock_held_by_other_process
) -> None:
    assert is_review_file_locked(review_file_path)


def test_hold_review_file_lock_held_by_other_process_raises(
    review_file_path, lock_held_by_other_process
) -> None:
    with pytest.raises(ReviewFileLockedError), hold_review_file_lock(review_file_path):
        pass


def test_hold_review_file_lock_while_held_reports_locked_to_others(
    review_file_path,
) -> None:
    with hold_review_file_lock(review_file_path):
        checker = subprocess.run(  # noqa: S603  (fixed test arguments)
            [
                sys.executable,
                "-c",
                "import sys, filelock\n"
                "try:\n"
                "    filelock.FileLock(sys.argv[1]).acquire(timeout=0)\n"
                "    print('free')\n"
                "except filelock.Timeout:\n"
                "    print('locked')",
                f"{review_file_path}.lock",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    assert checker.stdout.strip() == "locked"


def test_hold_review_file_lock_released_after_block(review_file_path) -> None:
    with hold_review_file_lock(review_file_path):
        pass
    assert not is_review_file_locked(review_file_path)
