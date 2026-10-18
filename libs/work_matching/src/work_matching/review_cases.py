"""Review cases: decisions the matching rules leave to a person, with their answers.

Every entry the rules cannot place, and every split-work group they cannot
settle, becomes a review case: one JSON line in the review file. A person, or
later the review agent, writes the answer into the line, and the next run hands
it back to the caller, who applies it. The rules:

- One case per entry: all of an entry's doubled providers are reviewed together,
  because telling which works a mixed entry holds needs all its pages at once,
  and because an answer's ``work 2`` then names the same work on every page.
- Case name: a case is named by its kind and its sorted page keys, never by an
  entry's position, so the same case in a later seed release or discovery run
  meets its kept answer. A doubled-provider case also names its entry's
  reference page, because its answer places pages relative to that entry.
- Changed pages: a case whose pages changed, or whose doubled pages moved to
  another entry, gets a new name, so it is asked again.
- Answers: a doubled-provider case's answer places pages with the words in
  ``page_placement`` (``entry``, ``other work``, ``work 2``…, and ``, main``);
  a split work's answer is ``one work`` or ``separate works``. Any other answer
  leaves the case open.
- Suggested answer: a doubled-provider case carries the places its links point
  to, so a person starts from them rather than from nothing.
- Kept answers: an answer whose case no longer comes up is kept as written,
  never applied, and reported.
- Rewrite: the file is rewritten whole on each run, because people edit lines in
  place; only each line's case name and answer are read back.
"""

import logging
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

from work_matching.doubled_provider_check import DoubledProviderDecision
from work_matching.file_replacement import replace_file_lines
from work_matching.page_placement import (
    PagePlacement,
    read_answer_placements,
    suggest_page_placements,
)
from work_matching.provider_page import ProviderPage, index_pages_by_current_key
from work_matching.split_work_check import SplitWorkDecision
from work_matching.work_entry import WorkEntry

logger = logging.getLogger(__name__)

_SPLIT_WORK_ANSWERS = frozenset({"one work", "separate works"})


class ReviewCase(BaseModel):
    """One decision left to a person: what the rules found, and the answer given.

    Fields are written in this order, so ``case``, ``answer`` and
    ``suggested_answer`` open each line of the review file and a person finds
    them first.
    """

    model_config = ConfigDict(frozen=True)

    # The case name: its kind and its sorted page keys.
    case: str
    # As written in the file; ``accepted_answer`` says whether it is valid.
    answer: JsonValue = None
    # For a doubled-provider case, the place the links point to for each page
    # they tie to a work; empty for a split-work case.
    suggested_answer: dict[str, str] = {}
    # The names of the decisions' reasons, which stay when their wording changes.
    reasons: tuple[str, ...]
    # Each entry involved, by its reference page.
    entries: dict[str, WorkEntry]
    # Every doubled-provider decision of the entry, or the one split-work
    # decision of the group.
    decisions: tuple[DoubledProviderDecision, ...] | tuple[SplitWorkDecision]
    # The entries' fetched pages and the pages found through them, which hold
    # the links behind the decisions.
    pages: tuple[ProviderPage, ...] = ()

    @property
    def accepted_answer(self) -> dict[str, PagePlacement] | str | None:
        """The answer when it is a valid answer to this case, otherwise None.

        A doubled-provider answer comes back as each placed page's place; a
        split-work answer as ``one work`` or ``separate works``.
        """
        answer = self.answer
        doubled_provider_decisions = [
            decision
            for decision in self.decisions
            if isinstance(decision, DoubledProviderDecision)
        ]
        if not doubled_provider_decisions:
            is_valid = isinstance(answer, str) and answer in _SPLIT_WORK_ANSWERS
            return answer if is_valid else None
        [entry] = self.entries.values()
        return read_answer_placements(answer, entry, doubled_provider_decisions)

    @classmethod
    def from_doubled_provider_entry(
        cls,
        entry: WorkEntry,
        decisions: Sequence[DoubledProviderDecision],
        pages: Iterable[ProviderPage],
    ) -> Self:
        """Make the review case for an entry whose pages the rules could not place.

        Args:
            entry: The work entry the decisions are about.
            decisions: Every decision ``decide_doubled_providers`` made for it.
            pages: The fetched pages; the details of the entry's pages are kept.
        """
        doubled_pages = [key for decision in decisions for key in decision.pages]
        return cls(
            case=(
                f"{_build_case_name('doubled provider', doubled_pages)}"
                f" in {entry.reference_page}"
            ),
            suggested_answer=suggest_page_placements(entry, decisions),
            reasons=tuple(decision.reason.name for decision in decisions),
            entries={entry.reference_page: entry},
            decisions=tuple(decisions),
            pages=_select_page_details(
                pages, frozenset(entry.pages) | set(doubled_pages)
            ),
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
        entries = {entry.reference_page: entry for entry in decision.entries}
        return cls(
            case=_build_case_name("split work", entries),
            reasons=(decision.reason.name,),
            entries=entries,
            decisions=(decision,),
            pages=_select_page_details(
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
    One process at a time writes the file, and a person does not edit it during
    a run: an answer saved after ``write_cases`` reads the file again is lost.
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
        self._answers_by_case = self._read_answers_in_file()

    def attach_kept_answer(self, case: ReviewCase) -> ReviewCase:
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
            case_name
            for case_name, case_count in case_name_counts.items()
            if case_count > 1
        ]:
            raise DuplicateCaseError(repeated_case_names)
        self._answers_by_case = self._read_answers_in_file()
        cases_with_answers = sorted(
            map(self.attach_kept_answer, cases), key=lambda case: case.case
        )
        for case in cases_with_answers:
            if case.answer is not None and case.accepted_answer is None:
                logger.warning(
                    f"Review case answer not accepted, left open: {case.case}"
                )
        unused_case_names = tuple(
            sorted(self._answers_by_case.keys() - case_name_counts.keys())
        )
        replace_file_lines(
            self._path,
            [case.model_dump_json() + "\n" for case in cases_with_answers]
            + [self._answers_by_case[case_name][1] for case_name in unused_case_names],
        )
        return unused_case_names

    def _read_answers_in_file(self) -> dict[str, tuple[JsonValue, str]]:
        """Return each answered case's answer and its line as written."""
        answers_by_case: dict[str, tuple[JsonValue, str]] = {}
        if not self._path.exists():
            return answers_by_case
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
                if kept_answer.case in answers_by_case:
                    logger.warning(
                        f"Review case answered twice, the last kept: {kept_answer.case}"
                    )
                answers_by_case[kept_answer.case] = (
                    kept_answer.answer,
                    line.rstrip("\n") + "\n",
                )
        return answers_by_case


def _build_case_name(case_kind: str, page_keys: Iterable[str]) -> str:
    """Name a case by its kind and its sorted page keys."""
    return f"{case_kind}: {' '.join(sorted(page_keys))}"


def _select_page_details(
    pages: Iterable[ProviderPage], page_keys: frozenset[str]
) -> tuple[ProviderPage, ...]:
    """Return the details of these pages and of pages found through them.

    A page is kept when its key or its current key is one of ``page_keys``, or
    when it was found through one of them (an AnimeSchedule page found by MAL
    id), so a reviewer sees every link the decision read. Sorted by current key.
    """
    return tuple(
        page
        for key, page in sorted(index_pages_by_current_key(pages).items())
        if {key, page.key, *page.found_by} & page_keys
    )
