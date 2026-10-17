"""Review cases: decisions the matching rules leave to a person, with their answers.

Every decision with outcome ``REVIEW`` becomes a review case, one JSON line in the
review file. A person, or later the review agent, writes the answer into the line,
and the next run hands it back to the caller, who applies it. The rules:

- Case name: a case is named by its kind and its sorted page keys, never by an
  entry's position, so the same case in a later seed release or discovery run
  meets its kept answer.
- Changed pages: a case whose pages changed gets a new name, so it is asked again.
- Answers: a doubled provider's answer places every page of the case (``entry``,
  ``other work`` or ``work 2``, ``work 3``…); a split work's answer is
  ``one work`` or ``separate works``. Any other answer leaves the case open.
- Kept answers: an answer whose case no longer comes up is kept as written,
  never applied, and reported.
- Rewrite: the file is rewritten whole on each run, because people edit lines in
  place; only each line's case name and answer are read back.
"""

import logging
import os
import re
import shutil
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

from work_matching.doubled_provider_check import DoubledProviderDecision
from work_matching.provider_page import ProviderPage, pages_by_current_key
from work_matching.split_work_check import SplitWorkDecision
from work_matching.work_entry import WorkEntry

logger = logging.getLogger(__name__)

# Where a doubled provider's page goes: to this entry, to another work, or to
# the second, third or later work of a mixed entry.
_PAGE_PLACEMENT_ANSWER = re.compile(r"entry|other work|work (?:[2-9]|[1-9][0-9]+)")
_SPLIT_WORK_ANSWERS = frozenset({"one work", "separate works"})


class ReviewCase(BaseModel):
    """One decision left to a person: what the rules found, and the answer given.

    Fields are written in this order, so ``case`` and ``answer`` open each line
    of the review file and a person finds them first.
    """

    model_config = ConfigDict(frozen=True)

    # The case name: its kind and its sorted page keys.
    case: str
    # As written in the file; ``accepted_answer`` says whether it is valid.
    answer: JsonValue = None
    # The name of the decision's reason, which stays when its wording changes.
    reason: str
    # Each entry involved, by its reference page, with its title.
    entries: dict[str, str]
    decision: DoubledProviderDecision | SplitWorkDecision
    pages: tuple[ProviderPage, ...] = ()

    @property
    def accepted_answer(self) -> dict[str, str] | str | None:
        """The answer when it is a valid answer to this case, otherwise None.

        A doubled provider's answer places every page of the case: ``entry``,
        ``other work`` or ``work 2``, ``work 3`` and so on. A split work's
        answer is ``one work`` or ``separate works``.
        """
        answer = self.answer
        if isinstance(self.decision, SplitWorkDecision):
            is_valid = isinstance(answer, str) and answer in _SPLIT_WORK_ANSWERS
            return answer if is_valid else None
        if not isinstance(answer, dict) or set(answer) != set(self.decision.pages):
            return None
        page_placements = {
            page_key: placement
            for page_key, placement in answer.items()
            if isinstance(placement, str)
            and _PAGE_PLACEMENT_ANSWER.fullmatch(placement)
        }
        return page_placements if len(page_placements) == len(answer) else None

    @classmethod
    def from_doubled_provider(
        cls,
        entry: WorkEntry,
        decision: DoubledProviderDecision,
        pages: Iterable[ProviderPage],
    ) -> Self:
        """Make the review case for a doubled provider the rules could not settle.

        Args:
            entry: The work entry the decision is about.
            decision: The decision ``decide_doubled_providers`` made.
            pages: The fetched pages; the details of the case's pages are kept.
        """
        return cls(
            case=_case_name("doubled provider", decision.pages),
            reason=decision.reason.name,
            entries={entry.reference_page: entry.title},
            decision=decision,
            pages=_page_details(pages, frozenset(decision.pages)),
        )

    @classmethod
    def from_split_work(
        cls, decision: SplitWorkDecision, pages: Iterable[ProviderPage]
    ) -> Self:
        """Make the review case for split-work candidates the rules could not settle.

        Args:
            decision: The decision ``decide_split_work`` made.
            pages: The fetched pages; the details of the entries' pages are kept.
        """
        entries = {entry.reference_page: entry.title for entry in decision.entries}
        return cls(
            case=_case_name("split work", entries),
            reason=decision.reason.name,
            entries=entries,
            decision=decision,
            pages=_page_details(
                pages,
                frozenset(key for entry in decision.entries for key in entry.pages),
            ),
        )


class _KeptAnswer(BaseModel):
    """The part of a review file line read back: the case name and its answer."""

    case: str
    answer: JsonValue = None


class DuplicateCaseError(ValueError):
    """Two review cases written in one run have the same case name."""

    def __init__(self, case_names: Iterable[str]) -> None:
        """Name the cases written more than once."""
        super().__init__(f"Review cases given more than once: {', '.join(case_names)}")


class ReviewFile:
    """The review file: review cases kept in one JSON Lines file across runs.

    A person edits lines in place, so the file is rewritten whole rather than
    appended to. Only each line's case name and answer are read back; the rest
    of a line is rebuilt by the run, so anything else written in it is not kept.
    One process at a time writes the file.
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
        self._answers_by_case = self._answers_in_file()

    def with_kept_answer(self, case: ReviewCase) -> ReviewCase:
        """Return the case with the answer kept for it, or unchanged if none is.

        The kept answer comes back even when it is not valid, so it stays in
        the file to be corrected.
        """
        kept_answer = self._answers_by_case.get(case.case)
        return (
            case.model_copy(update={"answer": kept_answer[0]}) if kept_answer else case
        )

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
        case_name_counts = Counter(case.case for case in cases)
        if repeated_case_names := [
            name for name, count in case_name_counts.items() if count > 1
        ]:
            raise DuplicateCaseError(repeated_case_names)
        self._answers_by_case = self._answers_in_file()
        cases_with_answers = sorted(
            map(self.with_kept_answer, cases), key=lambda case: case.case
        )
        for case in cases_with_answers:
            if case.answer is not None and case.accepted_answer is None:
                logger.warning(
                    f"Review case answer not accepted, left open: {case.case}"
                )
        unused_case_names = tuple(
            sorted(self._answers_by_case.keys() - case_name_counts.keys())
        )
        self._replace_file(
            [case.model_dump_json() + "\n" for case in cases_with_answers]
            + [self._answers_by_case[name][1] for name in unused_case_names]
        )
        return unused_case_names

    def _answers_in_file(self) -> dict[str, tuple[JsonValue, str]]:
        """Return each answered case's answer and its line as written."""
        answers_by_case: dict[str, tuple[JsonValue, str]] = {}
        if not self._path.exists():
            return answers_by_case
        with self._path.open(encoding="utf-8") as review_lines:
            for line_number, line in enumerate(review_lines, start=1):
                if not line.strip():
                    continue
                try:
                    kept_answer = _KeptAnswer.model_validate_json(line)
                except ValidationError as error:
                    error.add_note(f"{self._path}, line {line_number}")
                    raise
                if kept_answer.answer is None:
                    continue
                if kept_answer.case in answers_by_case:
                    logger.warning(
                        f"Review case answered twice, the last kept: {kept_answer.case}"
                    )
                answers_by_case[kept_answer.case] = (
                    kept_answer.answer,
                    line.rstrip("\n") + "\n",
                )
        return answers_by_case

    def _replace_file(self, lines: list[str]) -> None:
        """Write the lines next to the review file, then rename over it in one step.

        An interrupted run leaves the old file, and an existing file keeps its
        permissions.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self._path.with_name(f".{self._path.name}.tmp")
        try:
            with temporary_path.open("w", encoding="utf-8") as output:
                output.writelines(lines)
                output.flush()
                os.fsync(output.fileno())
            if self._path.exists():
                shutil.copymode(self._path, temporary_path)
            temporary_path.replace(self._path)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise


def _case_name(case_kind: str, page_keys: Iterable[str]) -> str:
    """Name a case by its kind and its sorted page keys."""
    return f"{case_kind}: {' '.join(sorted(page_keys))}"


def _page_details(
    pages: Iterable[ProviderPage], page_keys: frozenset[str]
) -> tuple[ProviderPage, ...]:
    """Return the details of the pages with these current keys, sorted by key."""
    pages_by_key = pages_by_current_key(pages)
    return tuple(pages_by_key[key] for key in sorted(page_keys) if key in pages_by_key)
