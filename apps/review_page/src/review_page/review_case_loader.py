"""Load the review cases in ``review.jsonl``, converting them once per file version.

The case list, toolbar and Corrections tab need every case on every request.
Each rule reads ``<rule>: <reason>``.

- One conversion per file version: the cases and their messages are kept until
  the file changes, because parsing and converting every case is the slowest
  step of a request.
- Version by inode, modification time and size: saves and runs replace the file
  through ``replace_file_lines``, which gives it a new inode, so a change made
  within one clock tick is still noticed. A hand edit changes the time or size.
- The file stays the only source of truth: nothing is kept that the file does
  not hold, and a changed file is always read again.
- One reader at a time: a lock keeps two requests from parsing the same version
  twice.
- Shared results: callers receive the kept tuples and must not change the
  messages in them.
"""

import threading
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError
from review_page.review_case_messages import convert_case_to_message
from shared_proto.v1 import review_cases_pb2
from work_matching import ReviewCase

# The inode, modification time in nanoseconds, and size that identify one
# version of the review file.
type ReviewFileVersion = tuple[int, int, int]


class UnreadableReviewFileError(RuntimeError):
    """Indicate that a line in the review file failed JSON or schema validation."""

    def __init__(self, review_file_path: Path, line_number: int) -> None:
        """Initialize the error with the corrupted file name and 1-based line number.

        Args:
            review_file_path: Path to the unreadable review JSONL file.
            line_number: The 1-based line number where validation failed.
        """
        super().__init__(
            f"Line {line_number} of {review_file_path.name} is not a valid review "
            "case. When the file comes from an older version, run "
            "scripts/clean_seed.py --no-fetch from the terminal, which rebuilds "
            "the file and keeps its saved answers. Otherwise fix or remove that "
            "line, then reload the page."
        )


@dataclass(frozen=True, slots=True)
class ReviewFileCases:
    """Hold one version of the review file as case models and as their messages."""

    cases: tuple[ReviewCase, ...]
    # The same cases converted for the review case source contract, in file order.
    case_messages: tuple[review_cases_pb2.ReviewCase, ...]


class ReviewCaseLoader:
    """Load the review file's cases, loading again only when the file changes."""

    def __init__(self, review_file_path: Path) -> None:
        """Keep the review file's path, with nothing read yet.

        Args:
            review_file_path: The review file, ``review.jsonl``.
        """
        self._review_file_path = review_file_path
        self._read_lock = threading.Lock()
        self._read_version: ReviewFileVersion | None = None
        self._review_file_cases = ReviewFileCases(cases=(), case_messages=())

    def load_cases(self) -> ReviewFileCases:
        """Load the review file's cases, reloading them only after the file changes.

        Returns:
            The cases and their messages in file order; none when the file does
            not exist.

        Raises:
            UnreadableReviewFileError: A line is not a valid review case; the
                error names the line.
        """
        with self._read_lock:
            file_version = self._find_file_version()
            if file_version is None:
                self._read_version = None
                return ReviewFileCases(cases=(), case_messages=())
            if file_version != self._read_version:
                cases = self._parse_cases()
                self._review_file_cases = ReviewFileCases(
                    cases=cases,
                    case_messages=tuple(
                        convert_case_to_message(case) for case in cases
                    ),
                )
                self._read_version = file_version
            return self._review_file_cases

    def _find_file_version(self) -> ReviewFileVersion | None:
        """Return the review file's current version, or None when it does not exist.

        Returns:
            The file's inode, modification time in nanoseconds and size.
        """
        try:
            file_status = self._review_file_path.stat()
        except FileNotFoundError:
            return None
        return (file_status.st_ino, file_status.st_mtime_ns, file_status.st_size)

    def _parse_cases(self) -> tuple[ReviewCase, ...]:
        """Parse every case in the review file, in file order.

        Returns:
            The parsed cases; blank lines are skipped.

        Raises:
            UnreadableReviewFileError: A line is not a valid review case.
        """
        cases: list[ReviewCase] = []
        with self._review_file_path.open(encoding="utf-8") as review_file:
            for line_number, line in enumerate(review_file, start=1):
                if not line.strip():
                    continue
                try:
                    cases.append(ReviewCase.model_validate_json(line))
                except ValidationError as error:
                    raise UnreadableReviewFileError(
                        self._review_file_path, line_number
                    ) from error
        return tuple(cases)
