"""Keep one writer at a time on a review file, across processes.

A seed-cleaning run rewrites the review file whole, and the review page saves
answers into it. Both write through ``replace_file_lines``, whose temporary
file has a fixed name, and a save made while a run rewrites the file would be
lost. So every writer holds one lock first:

- The lock is an operating-system file lock on ``<review file>.lock``
  (``filelock``), so it holds between separate processes, such as a run and
  the review page's server.
- A writer that cannot get the lock in time gets ``ReviewFileLockedError`` and
  writes nothing, so the caller can tell the reviewer a run is going.
- The lock file stays after release; only the operating-system lock on it
  matters, and an empty file is harmless.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from filelock import FileLock, Timeout


class ReviewFileLockedError(RuntimeError):
    """Report that another process holds the review file's lock, such as a running seed cleaning."""

    def __init__(self, review_file_path: Path) -> None:
        """Name the locked review file.

        Args:
            review_file_path: The review file whose lock is held.
        """
        super().__init__(
            f"Another process is writing the review file: {review_file_path}"
        )


@contextmanager
def hold_review_file_lock(
    review_file_path: str | Path, *, seconds_to_wait: float = 0.0
) -> Iterator[None]:
    """Hold the review file's lock for the block, waiting at most the given time.

    Args:
        review_file_path: The review file to lock.
        seconds_to_wait: How long to wait for another writer to finish; 0
            gives up at once.

    Yields:
        Nothing; the lock is held while the block runs.

    Raises:
        ReviewFileLockedError: The lock was still held after the wait.
    """
    review_file_path = Path(review_file_path)
    review_file_path.parent.mkdir(parents=True, exist_ok=True)
    review_file_lock = FileLock(_build_lock_path(review_file_path))
    try:
        review_file_lock.acquire(timeout=seconds_to_wait)
    except Timeout as error:
        raise ReviewFileLockedError(review_file_path) from error
    try:
        yield
    finally:
        review_file_lock.release()


def is_review_file_locked(review_file_path: str | Path) -> bool:
    """Report whether another writer holds the review file's lock right now.

    The lock is taken and released at once to find out, since only taking it
    tells whether another process holds it.

    Args:
        review_file_path: The review file to check.

    Returns:
        True when another writer, such as a running seed cleaning, holds it.
    """
    try:
        with hold_review_file_lock(review_file_path):
            return False
    except ReviewFileLockedError:
        return True


def _build_lock_path(review_file_path: Path) -> Path:
    """Return the lock file beside the review file: ``<review file>.lock``.

    Args:
        review_file_path: Path to the review file.

    Returns:
        Path to the sibling ``.lock`` file.
    """
    return review_file_path.with_name(f"{review_file_path.name}.lock")
