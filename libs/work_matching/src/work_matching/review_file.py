"""The review file: review cases kept in one JSON Lines file across runs.

A run writes every case it could not decide into the file, with the answer
kept for it. A person, or the review page, answers cases in the file, and the
next run reads the answers back. The rules for the cases themselves are in
``review_cases``; this module only reads and writes the file:

Each item gives the rule, then the reason for it.

- A run rewrites the file whole (``write_cases``), because people edit lines in
  place; only each line's case name and answer are read back.
- The review page saves answers (``save_answers``) by rewriting only the named
  cases' lines, so a save changes nothing else.
- Both hold the file's lock, so a save never lands between a run's read and its
  rewrite.
"""

import logging
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path

from pydantic import BaseModel, JsonValue, ValidationError
from work_matching.file_replacement import replace_file_lines
from work_matching.review_cases import ReviewCase
from work_matching.review_file_lock import hold_review_file_lock

logger = logging.getLogger(__name__)

# How long a save waits for another save to finish. Saves take milliseconds;
# a run holds the lock for seconds, and a save during it should fail fast.
_SECONDS_TO_WAIT_FOR_ANOTHER_SAVE = 2.0


class _KeptAnswer(BaseModel):
    """The part of a review file line read back: the case name and its answer."""

    case_name: str
    answer: JsonValue = None


class DuplicateCaseError(ValueError):
    """Two review cases written in one run have the same case name."""

    def __init__(self, case_names: Iterable[str]) -> None:
        """Name the cases written more than once."""
        super().__init__(f"Review cases given more than once: {', '.join(case_names)}")


class UnknownCaseError(KeyError):
    """Answers were given for case names the review file does not hold."""

    def __init__(self, case_names: Iterable[str]) -> None:
        """Name the cases the review file does not hold."""
        super().__init__(
            f"Review cases not in the review file: {', '.join(case_names)}"
        )


class ReviewFile:
    """The review file: review cases kept in one JSON Lines file across runs.

    A person edits lines in place, so the file is rewritten whole rather than
    appended to. Only each line's case name and answer are read back; the rest
    of a line is rebuilt by the run, so anything else written in it is not kept.
    One process at a time writes the file: ``save_answers`` holds the file's
    lock (``review_file_lock``), and a run holds it around ``write_cases``, so
    an answer is never saved between a run's read and its rewrite.
    """

    def __init__(self, path: str | Path) -> None:
        """Read the answers already written, if the file exists.

        Args:
            path: The JSON Lines review file.

        Raises:
            ValidationError: A line is not a case. Its note names the file and
                line.
        """
        self._path = Path(path)
        self._answers_and_lines_by_case_name = self._read_answers_in_file()

    def attach_kept_answer(self, case: ReviewCase) -> ReviewCase:
        """Return the case with the answer kept for it, or unchanged if none is.

        The kept answer comes back even when it is not valid, so it stays in
        the file to be corrected.
        """
        kept_answer = self._answers_and_lines_by_case_name.get(case.case_name)
        return (
            case.model_copy(update={"answer": kept_answer[0]}) if kept_answer else case
        )

    def get_answers(self) -> dict[str, JsonValue]:
        """Return every answer in the file as last read, by case name.

        After ``write_cases`` these are the answers the run wrote, so a caller
        can keep them as each case's original value.
        """
        return {
            case_name: answer
            for case_name, (answer, _) in self._answers_and_lines_by_case_name.items()
        }

    def write_cases(self, cases: Iterable[ReviewCase]) -> tuple[str, ...]:
        """Write this run's cases, with their kept answers, in place of the file.

        The answers are read again first, so one saved since the file was
        opened is kept. A kept answer whose case is not among the cases is
        written back as it was, never applied. An unanswered case that did not
        come up is dropped.

        Args:
            cases: The review cases of this run.

        Returns:
            The case names of kept answers whose case did not come up.

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
        unused_case_names = tuple(
            sorted(
                self._answers_and_lines_by_case_name.keys() - case_name_counts.keys()
            )
        )
        replace_file_lines(
            self._path,
            [case.model_dump_json() + "\n" for case in cases_with_answers]
            + [
                self._answers_and_lines_by_case_name[case_name][1]
                for case_name in unused_case_names
            ],
        )
        return unused_case_names

    def save_answers(self, answers_by_case_name: Mapping[str, JsonValue]) -> None:
        """Write answers into their cases' lines, keeping every other line as it is.

        This is how the review page saves. Each item gives the rule, then the
        reason for it.

        - The file's lock is held while saving, and the save fails while
          another process, such as a run, holds it. A save between a run's
          read and its rewrite would otherwise be lost.
        - The file is read again under the lock, so answers saved by another
          writer since this one opened the file are kept.
        - Only the named cases' lines are rewritten, with the new answer;
          every other line keeps its bytes, so a save changes nothing else.
        - A None answer clears the case's answer, which leaves it open.
        - Nothing is written when a case name is not in the file, so a
          mistyped name never loses the other answers.

        Args:
            answers_by_case_name: Each case's new answer, by its case name.

        Raises:
            ReviewFileLockedError: Another process holds the file's lock.
            UnknownCaseError: A case name is not in the file.
        """
        with hold_review_file_lock(
            self._path, seconds_to_wait=_SECONDS_TO_WAIT_FOR_ANOTHER_SAVE
        ):
            lines = (
                self._path.read_text(encoding="utf-8").splitlines(keepends=True)
                if self._path.exists()
                else []
            )
            saved_lines = [
                _replace_line_answer(line, answers_by_case_name) for line in lines
            ]
            missing_case_names = answers_by_case_name.keys() - {
                _KeptAnswer.model_validate_json(line).case_name
                for line in lines
                if line.strip()
            }
            if missing_case_names:
                raise UnknownCaseError(sorted(missing_case_names))
            replace_file_lines(self._path, saved_lines)
        self._answers_and_lines_by_case_name = self._read_answers_in_file()

    def _read_answers_in_file(self) -> dict[str, tuple[JsonValue, str]]:
        """Return each answered case's answer and its line as written."""
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


def _replace_line_answer(
    line: str, answers_by_case_name: Mapping[str, JsonValue]
) -> str:
    """Return the line with its case's new answer, or unchanged when not named.

    Args:
        line: One line of the review file, ending with its line break.
        answers_by_case_name: Each case's new answer, by its case name.

    Returns:
        The case rewritten with its new answer, in the order ``write_cases``
        writes fields; the line as it was otherwise.
    """
    if not line.strip():
        return line
    case_name = _KeptAnswer.model_validate_json(line).case_name
    if case_name not in answers_by_case_name:
        return line
    case = ReviewCase.model_validate_json(line)
    return (
        case.model_copy(
            update={"answer": answers_by_case_name[case_name]}
        ).model_dump_json()
        + "\n"
    )
