"""Decisions the matching rules leave to a person, and the answers given to them.

Each review decision becomes a case, one JSON line in the review file. A person,
or later the review agent, writes the answer into the case's ``answer`` field;
the next run reads it back and hands it to the caller, who applies it. A case is
named by its kind and its pages, so the same case in a later seed release or
discovery run meets its kept answer, and a case whose pages changed is asked
again.
"""

import re
from collections.abc import Iterable
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Self

from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

from work_matching.doubled_provider_check import DoubledProviderDecision
from work_matching.provider_page import ProviderPage, pages_by_current_key
from work_matching.split_work_check import SplitWorkDecision
from work_matching.work_entry import WorkEntry

# Where a doubled provider's page goes: to the entry, to another work, or to
# the second, third or later work of a mixed entry.
_PLACEMENT = re.compile(r"entry|elsewhere|work (?:[2-9]|[1-9]\d+)")
_SPLIT_WORK_ANSWERS = frozenset({"one work", "separate works"})


class ReviewCase(BaseModel):
    """One decision left to a person: what the rules found, and the answer given.

    Fields are written in this order, so ``case`` and ``answer`` open each line
    of the review file.
    """

    model_config = ConfigDict(frozen=True)

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
        ``elsewhere`` or ``work 2``, ``work 3`` and so on. A split work's answer
        is ``one work`` or ``separate works``.
        """
        answer = self.answer
        if isinstance(self.decision, SplitWorkDecision):
            is_valid = isinstance(answer, str) and answer in _SPLIT_WORK_ANSWERS
            return answer if is_valid else None
        if not isinstance(answer, dict) or set(answer) != set(self.decision.pages):
            return None
        placements = {
            page: placement
            for page, placement in answer.items()
            if isinstance(placement, str) and _PLACEMENT.fullmatch(placement)
        }
        return placements if len(placements) == len(answer) else None

    @classmethod
    def from_doubled_provider(
        cls,
        entry: WorkEntry,
        decision: DoubledProviderDecision,
        pages: Iterable[ProviderPage],
    ) -> Self:
        """Make the case for a doubled provider sent to review.

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
            pages=_details_of(pages, frozenset(decision.pages)),
        )

    @classmethod
    def from_split_work(
        cls, decision: SplitWorkDecision, pages: Iterable[ProviderPage]
    ) -> Self:
        """Make the case for split-work candidates sent to review.

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
            pages=_details_of(
                pages,
                frozenset(key for entry in decision.entries for key in entry.pages),
            ),
        )


class _KeptAnswer(BaseModel):
    """The part of a review file line read back: the case and its answer."""

    case: str
    answer: JsonValue = None


class ReviewFile:
    """Review cases kept in one JSON Lines file, rewritten on every run.

    A person edits lines in place, so the file is rewritten whole rather than
    appended to. Opening it reads only each line's case and answer; the rest of
    a line is rebuilt by the run. One process at a time writes the file.
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
        # Each answered case's answer and its line as written.
        self._answered: dict[str, tuple[JsonValue, str]] = {}
        if self._path.exists():
            self._read_answers()

    def with_kept_answer(self, case: ReviewCase) -> ReviewCase:
        """Return the case with the answer kept for it, or unchanged if none is."""
        kept = self._answered.get(case.case)
        return case.model_copy(update={"answer": kept[0]}) if kept else case

    def write_cases(self, cases: Iterable[ReviewCase]) -> tuple[str, ...]:
        """Write this run's cases, with their kept answers, in place of the file.

        A kept answer whose case is not among them is written back as it was,
        never applied. An unanswered case that did not come up is dropped. The
        file is replaced in one step, so an interrupted run leaves the old one.

        Args:
            cases: The review cases of this run.

        Returns:
            The names of kept answers whose case did not come up.
        """
        lines = {
            case.case: self.with_kept_answer(case).model_dump_json() + "\n"
            for case in cases
        }
        unused = tuple(sorted(self._answered.keys() - lines.keys()))
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=self._path.parent,
            prefix=f".{self._path.name}.",
            delete=False,
        )
        try:
            with temporary as output:
                output.writelines(lines[name] for name in sorted(lines))
                output.writelines(self._answered[name][1] for name in unused)
            Path(temporary.name).replace(self._path)
        except BaseException:
            Path(temporary.name).unlink(missing_ok=True)
            raise
        return unused

    def _read_answers(self) -> None:
        """Keep the answer and line of every answered case in the file."""
        with self._path.open(encoding="utf-8") as lines:
            for number, line in enumerate(lines, start=1):
                if not line.strip():
                    continue
                try:
                    kept = _KeptAnswer.model_validate_json(line)
                except ValidationError as error:
                    error.add_note(f"{self._path}, line {number}")
                    raise
                if kept.answer is not None:
                    self._answered[kept.case] = (kept.answer, line.rstrip("\n") + "\n")


def _case_name(kind: str, keys: Iterable[str]) -> str:
    """Name a case by its kind and its sorted page keys."""
    return f"{kind}: {' '.join(sorted(keys))}"


def _details_of(
    pages: Iterable[ProviderPage], keys: frozenset[str]
) -> tuple[ProviderPage, ...]:
    """Return the details of the pages with these current keys, sorted by key."""
    by_key = pages_by_current_key(pages)
    return tuple(by_key[key] for key in sorted(keys) if key in by_key)
