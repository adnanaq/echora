"""Load the review cases in ``review.jsonl``, converting each line once.

The case list, toolbar and Corrections tab need every case on every request.
Each rule reads ``<rule>: <reason>``.

- One conversion per file version: the cases and their messages are kept until
  the file changes, because parsing and converting every case is the slowest
  step of a request.
- One conversion per line: a save rewrites only the line of the case it
  changes, and a run writes an unchanged case as the same text. So a reload
  reuses the case and message of every line whose text it has seen, and
  parses and converts only new or changed lines. A case depends only on its
  line, so the same text always gives the same case. Only the current
  version's lines are kept.
- Version by inode, modification time and size: saves and runs replace the file
  through ``replace_file_lines``, which gives it a new inode, so a change made
  within one clock tick is still noticed. A hand edit changes the time or size.
- The file stays the only source of truth: nothing is kept that the file does
  not hold, and a changed file is always read again.
- One reader at a time: a lock keeps two requests from parsing the same version
  twice.
- Shared results: callers receive the kept tuples and must not change the
  cases or messages in them, which later versions of the file can share.
"""

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

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


class _ParsedReviewLine(NamedTuple):
    """Hold one review file line's case and its message."""

    case: ReviewCase
    case_message: review_cases_pb2.ReviewCase


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
        # The case and message of each line of the version read last, by the
        # line's text, so an unchanged line is not parsed again.
        self._parsed_line_by_text: dict[str, _ParsedReviewLine] = {}

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
                self._parsed_line_by_text = {}
                return ReviewFileCases(cases=(), case_messages=())
            if file_version != self._read_version:
                parsed_lines = self._parse_lines()
                self._review_file_cases = ReviewFileCases(
                    cases=tuple(parsed_line.case for parsed_line in parsed_lines),
                    case_messages=tuple(
                        parsed_line.case_message for parsed_line in parsed_lines
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

    def _parse_lines(self) -> list[_ParsedReviewLine]:
        """Parse every case in the review file, reusing the lines read before.

        Returns:
            Each case with its message, in file order; blank lines are skipped.

        Raises:
            UnreadableReviewFileError: A line is not a valid review case.
        """
        parsed_lines: list[_ParsedReviewLine] = []
        parsed_line_by_text: dict[str, _ParsedReviewLine] = {}
        with self._review_file_path.open(encoding="utf-8") as review_file:
            for line_number, line in enumerate(review_file, start=1):
                line_text = line.rstrip("\n")
                if not line_text.strip():
                    continue
                parsed_line = parsed_line_by_text.get(
                    line_text
                ) or self._parsed_line_by_text.get(line_text)
                if parsed_line is None:
                    parsed_line = self._parse_line(line_text, line_number)
                parsed_line_by_text[line_text] = parsed_line
                parsed_lines.append(parsed_line)
        self._parsed_line_by_text = parsed_line_by_text
        return parsed_lines

    def _parse_line(self, line_text: str, line_number: int) -> _ParsedReviewLine:
        """Parse one line of the review file into its case and message.

        Args:
            line_text: The line, without its line break.
            line_number: The line's number in the file, counted from 1.

        Returns:
            The line's case and its message.

        Raises:
            UnreadableReviewFileError: The line is not a valid review case.
        """
        try:
            case = ReviewCase.model_validate_json(line_text)
        except ValidationError as error:
            raise UnreadableReviewFileError(
                self._review_file_path, line_number
            ) from error
        return _ParsedReviewLine(case, convert_case_to_message(case))
