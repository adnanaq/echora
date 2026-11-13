"""Read and write review cases in one JSON Lines file across runs.

A run writes every case that could not be decided into the file, with the answer
kept for the case. A person, or the review page, answers cases in the file, and
the next run reads the answers back. The rules for the cases themselves are in
``review_cases``; this module only reads and writes the file:

Each item gives the rule, then the reason for it.

- A run rewrites the file whole (``write_cases``), because people edit lines in
  place; only each line's case name and answer are read back.
- The review page saves answers (``save_answers``) by rewriting only the named
  cases' lines, so a save changes nothing else.
- Both hold the file's lock, so a save never lands between a run's read and
  rewrite.
"""

import logging
import os
import shutil
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path

from filelock import FileLock, Timeout
from pydantic import BaseModel, JsonValue, ValidationError
from work_matching.errors import (
    DuplicateCaseError,
    ReviewFileLockedError,
    SavedAnswerChangedError,
    UnknownCaseError,
)
from work_matching.review_cases import ReviewCase

logger = logging.getLogger(__name__)

# How long a save waits for another save to finish. Saves take milliseconds;
# a run holds the lock for seconds, and a save during it should fail fast.
_SECONDS_TO_WAIT_FOR_ANOTHER_SAVE = 2.0


class _KeptAnswer(BaseModel):
    """The part of a review file line read back: the case name and its answer."""

    case_name: str
    answer: JsonValue = None


class ReviewFile:
    """The review file: review cases kept in one JSON Lines file across runs.

    A person edits lines in place, so the file is rewritten whole rather than
    appended to. Only each line's case name and answer are read back; the rest
    of a line is rebuilt by the run, so anything else written in the line is not
    kept. One process at a time writes the file: ``save_answers`` holds the
    file's lock (``review_file_lock``), and a run holds the lock around
    ``write_cases``, so an answer is never saved between a run's read and
    rewrite.
    """

    def __init__(self, path: str | Path) -> None:
        """Read the answers already written, if the file exists.

        Args:
            path: The JSON Lines review file.

        Raises:
            ValidationError: A line is not a case. The error note names the file
                and line.
        """
        self._path = Path(path)
        self._answers_and_lines_by_case_name = self._read_answers_in_file()

    def attach_kept_answer(self, case: ReviewCase) -> ReviewCase:
        """Return the case with the answer kept for the case, or unchanged if none is.

        The kept answer comes back even when the answer is not valid, so the
        answer stays in the file to be corrected.

        Args:
            case: The review case to attach a kept answer to.

        Returns:
            The review case with the kept answer attached, or unchanged when no
            answer is kept.
        """
        kept_answer = self._answers_and_lines_by_case_name.get(case.case_name)
        return (
            case.model_copy(update={"answer": kept_answer[0]}) if kept_answer else case
        )

    def get_answers(self) -> dict[str, JsonValue]:
        """Return every answer in the file as last read, by case name.

        After ``write_cases`` these are the answers the run wrote, so a caller
        can keep those answers as each case's baseline value.

        Returns:
            A mapping from case name to the saved answer value.
        """
        return {
            case_name: answer
            for case_name, (answer, _) in self._answers_and_lines_by_case_name.items()
        }

    def list_kept_cases(self) -> tuple[ReviewCase, ...]:
        """Return every answered case in the file as last read, its line read whole.

        A run finds among them the previous case a renamed case replaces
        (``previous_answers``), which needs each case's entries, not only its
        answer.

        Returns:
            The answered cases, in the file's order.
        """
        return tuple(
            ReviewCase.model_validate_json(line)
            for _, line in self._answers_and_lines_by_case_name.values()
        )

    def write_cases(self, cases: Iterable[ReviewCase]) -> tuple[str, ...]:
        """Write this run's cases, with their kept answers, in place of the file.

        Each item gives the rule, then the reason for it.

        - The answers are read again first, so one saved since the file was
          opened is kept.
        - A kept answer whose case is not among the cases is written back as
          it was, never applied by its own name, so no answer is lost.
        - A kept answer a case carries as its previous answer is written back
          too, since the next run reads it again, but it is not reported as
          unused. Once that case has an accepted answer of its own, the
          previous answer is dropped, since the new answer replaces it.
        - An unanswered case that did not come up is dropped.

        Args:
            cases: The review cases of this run, with their previous answers.

        Returns:
            The case names of kept answers whose case did not come up and that
            no case carries as its previous answer.

        Raises:
            DuplicateCaseError: Two cases have the same case name.
        """
        cases = tuple(cases)
        case_name_counts = Counter(case.case_name for case in cases)
        if repeated_case_names := [
            case_name
            for case_name, case_count in case_name_counts.items()
            if case_count > 1
        ]:
            raise DuplicateCaseError(repeated_case_names)
        self._answers_and_lines_by_case_name = self._read_answers_in_file()
        cases_with_answers = sorted(
            map(self.attach_kept_answer, cases), key=lambda case: case.case_name
        )
        for case in cases_with_answers:
            if case.answer is not None and case.accepted_answer is None:
                logger.warning(
                    f"Review case answer not accepted, left open: {case.case_name}"
                )
        previous_case_names = {
            case.previous_answer.case_name
            for case in cases_with_answers
            if case.previous_answer is not None
        }
        replaced_case_names = {
            case.previous_answer.case_name
            for case in cases_with_answers
            if case.previous_answer is not None and case.accepted_answer is not None
        }
        kept_case_names = sorted(
            self._answers_and_lines_by_case_name.keys()
            - case_name_counts.keys()
            - replaced_case_names
        )
        replace_file_lines(
            self._path,
            [case.model_dump_json() + "\n" for case in cases_with_answers]
            + [
                self._answers_and_lines_by_case_name[case_name][1]
                for case_name in kept_case_names
            ],
        )
        return tuple(
            case_name
            for case_name in kept_case_names
            if case_name not in previous_case_names
        )

    def save_answers(
        self,
        answers_by_case_name: Mapping[str, JsonValue],
        *,
        seen_saved_answers: Mapping[str, JsonValue] | None = None,
    ) -> None:
        """Write answers into their cases' lines, keeping every other line unchanged.

        The review page calls this method to save user edits. Each item gives
        the rule, then the reason for the rule.

        - The file's lock is held while saving, and the save fails while
          another process, such as a run, holds the lock. A save between a run's
          read and rewrite would otherwise be lost.
        - The file is read again under the lock, so answers saved by another
          writer since this writer opened the file are kept.
        - Only the named cases' lines are rewritten, with the new answer;
          every other line keeps its bytes, so a save changes nothing else.
        - A None answer clears the case's answer, which leaves the case open.
        - Nothing is written when a case name is not in the file, so a
          mistyped name never loses the other answers.
        - Nothing is written when a case's saved answer differs from the
          baseline answer the edit started from, preventing optimistic
          concurrency conflicts when another session modified the file
          concurrently. The check runs under the lock, so no save slips in
          between.

        Args:
            answers_by_case_name: Each case's new answer, by case name.
            seen_saved_answers: The saved answer each edit started from, by
                case name (None for no answer); cases left out are not checked.

        Raises:
            ReviewFileLockedError: Another process holds the file's lock.
            UnknownCaseError: A case name is not in the file.
            SavedAnswerChangedError: A case's saved answer differs from the
                baseline answer the edit started from.
        """
        with hold_review_file_lock(
            self._path, seconds_to_wait=_SECONDS_TO_WAIT_FOR_ANOTHER_SAVE
        ):
            # Lines end at "\n" only: a value may hold U+2028 or U+0085
            # unescaped, which str.splitlines would also split at.
            lines = []
            if self._path.exists():
                with self._path.open(encoding="utf-8") as review_file:
                    lines = review_file.readlines()
            kept_answers = [
                _KeptAnswer.model_validate_json(line) if line.strip() else None
                for line in lines
            ]
            _check_answers_against_file(
                kept_answers, answers_by_case_name, seen_saved_answers or {}
            )
            replace_file_lines(
                self._path,
                (
                    _write_line_answer(
                        line, answers_by_case_name[kept_answer.case_name]
                    )
                    if kept_answer is not None
                    and kept_answer.case_name in answers_by_case_name
                    else line
                    for line, kept_answer in zip(lines, kept_answers, strict=True)
                ),
            )
        self._answers_and_lines_by_case_name = self._read_answers_in_file()

    def _read_answers_in_file(self) -> dict[str, tuple[JsonValue, str]]:
        """Return each answered case's answer and line as written.

        Returns:
            A mapping from case name to a tuple of the saved answer and raw JSON
            line.

        Raises:
            ValidationError: A line is not a valid review case object.
        """
        answers_and_lines_by_case_name: dict[str, tuple[JsonValue, str]] = {}
        if not self._path.exists():
            return answers_and_lines_by_case_name
        with self._path.open(encoding="utf-8") as review_file:
            for line_number, line in enumerate(review_file, start=1):
                if not line.strip():
                    continue
                try:
                    kept_answer = _KeptAnswer.model_validate_json(line)
                except ValidationError as error:
                    error.add_note(f"{self._path}, line {line_number}")
                    raise
                if kept_answer.answer is None:
                    continue
                if kept_answer.case_name in answers_and_lines_by_case_name:
                    logger.warning(
                        "Review case answered twice, the last kept: "
                        f"{kept_answer.case_name}"
                    )
                answers_and_lines_by_case_name[kept_answer.case_name] = (
                    kept_answer.answer,
                    line.rstrip("\n") + "\n",
                )
        return answers_and_lines_by_case_name


def _check_answers_against_file(
    kept_answers: Sequence[_KeptAnswer | None],
    answers_by_case_name: Mapping[str, JsonValue],
    seen_saved_answers: Mapping[str, JsonValue],
) -> None:
    """Refuse answers for cases the file lacks, or whose saved answer changed.

    Args:
        kept_answers: Each line's case name and answer, None for a blank line.
        answers_by_case_name: Each case's new answer, by its case name.
        seen_saved_answers: The saved answer each edit started from, by case
            name.

    Raises:
        UnknownCaseError: A case name is not in the file.
        SavedAnswerChangedError: A case's saved answer is not the one its
            edit started from.
    """
    answers_in_file = {
        kept_answer.case_name: kept_answer.answer
        for kept_answer in kept_answers
        if kept_answer
    }
    if missing_case_names := answers_by_case_name.keys() - answers_in_file.keys():
        raise UnknownCaseError(sorted(missing_case_names))
    if changed_case_names := sorted(
        case_name
        for case_name, seen_answer in seen_saved_answers.items()
        if answers_in_file.get(case_name) != seen_answer
    ):
        raise SavedAnswerChangedError(changed_case_names)


def _write_line_answer(line: str, answer: JsonValue) -> str:
    """Return a review file line rewritten with a new answer.

    Args:
        line: One case's line of the review file, ending with its line break.
        answer: The case's new answer; None clears it.

    Returns:
        The case with its new answer, in the order ``write_cases`` writes
        fields, ending with a line break.
    """
    case = ReviewCase.model_validate_json(line)
    return case.model_copy(update={"answer": answer}).model_dump_json() + "\n"


def replace_file_lines(path: str | Path, lines: Iterable[str]) -> None:
    """Write the lines in place of the file, in one step.

    Args:
        path: The file to replace; it and its folder are created if missing.
        lines: The lines to write, each ending with its line break.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as temporary_file:
            temporary_file.writelines(lines)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        if path.exists():
            shutil.copymode(path, temporary_path)
        temporary_path.replace(path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    _sync_folder(path.parent)


def _sync_folder(folder: Path) -> None:
    """Write a folder's entries to disk, so a rename in it survives a power cut.

    Systems that cannot open a folder, such as Windows, are skipped.

    Args:
        folder: The directory whose directory entries must be flushed to disk.
    """
    if not hasattr(os, "O_DIRECTORY"):
        return
    folder_descriptor = os.open(folder, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(folder_descriptor)
    finally:
        os.close(folder_descriptor)


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
