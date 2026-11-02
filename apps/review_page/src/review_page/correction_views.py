"""Build the Corrections tab: the entries the last run corrected, and each entry's pages.

The tab reads the corrections only as the review case source's messages, so it
shows what any source reports. Each item gives the rule, then the reason for it.

- One list row per corrected entry, in the source's order, because a reviewer
  checks an entry at a time, as on the Cases tab.
- An entry is titled from the run's checked entries, then from a review case
  holding it, then by its reference page. A correction names its entry only by
  reference page.
- Filters within one group (changes, sources) widen the list; groups narrow
  each other, as the Cases tab's filters do. A change filter keeps an entry
  with at least one page making that change; "unchanged" keeps an entry with a
  page that stays as it was.
- A page's destination names where it goes: the seed entry it joins (by
  reference page, linked to that page), a new entry, its current address, or
  that it stays. A target entry is often not one the run checked, so its title
  is not known.
- The panel links the cases holding the entry, so a reviewer goes from a
  correction to the answer behind it.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

from review_page.case_views import find_case_title
from shared_proto.v1 import review_cases_pb2
from work_matching import (
    LinkedPageDecision,
    ReviewCase,
    build_page_address,
)

# The change filter value for a page that stays as it was.
UNCHANGED = "unchanged"
# The source filter values: a reviewer's answer, or the rules alone.
ANSWER_SOURCE = "your answer"
RULES_SOURCE = "rules"


@dataclass(frozen=True, slots=True)
class CorrectedEntryRow:
    """One corrected entry as the Corrections list shows it."""

    reference_page: str
    title: str
    # Every change the entry's pages make, UNCHANGED included.
    changes: frozenset[str]
    page_count: int
    # The pages that change, leaving out pages that stay as they were.
    changed_page_count: int
    has_answer_correction: bool
    has_rule_correction: bool
    page_keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CorrectionFilter:
    """The Corrections tab's filters; an empty group lets every entry through."""

    changes: frozenset[str] = frozenset()
    sources: frozenset[str] = frozenset()
    search_text: str = ""


@dataclass(frozen=True, slots=True)
class CorrectionRowView:
    """One page of a corrected entry, as the entry's table shows it."""

    page_key: str
    address: str
    # The change as the run summary names it, or "Unchanged".
    change_label: str
    # The page's work and mark in answer words: "work 1, part".
    place: str
    destination_label: str
    destination_address: str | None
    reason_sentence: str
    is_from_answer: bool


@dataclass(frozen=True, slots=True)
class CorrectionPanelView:
    """One corrected entry, as the Corrections panel shows it."""

    reference_page: str
    reference_address: str | None
    title: str
    rows: tuple[CorrectionRowView, ...]
    # Each case holding the entry: its name and its list title.
    case_links: tuple[tuple[str, str], ...] = field(default=())


def build_corrected_entry_rows(
    corrections: Sequence[review_cases_pb2.SeedCorrection],
    checked_entries: Sequence[review_cases_pb2.CheckedEntry],
    cases: Sequence[ReviewCase],
) -> list[CorrectedEntryRow]:
    """Build one row per corrected entry, in the order the corrections name them.

    Args:
        corrections: Every correction of the last run.
        checked_entries: The entries the last run checked, for titles.
        cases: The review cases, for the titles of entries no check lists.

    Returns:
        The rows, one per entry reference page.
    """
    title_by_reference_page = _map_entry_titles(checked_entries, cases)
    corrections_by_entry: dict[str, list[review_cases_pb2.SeedCorrection]] = {}
    for correction in corrections:
        corrections_by_entry.setdefault(correction.entry_reference_page, []).append(
            correction
        )
    return [
        CorrectedEntryRow(
            reference_page=reference_page,
            title=title_by_reference_page.get(reference_page, reference_page),
            changes=frozenset(map(_get_change, entry_corrections)),
            page_count=len(entry_corrections),
            changed_page_count=sum(
                correction.HasField("change") for correction in entry_corrections
            ),
            has_answer_correction=any(
                correction.is_from_answer for correction in entry_corrections
            ),
            has_rule_correction=not all(
                correction.is_from_answer for correction in entry_corrections
            ),
            page_keys=tuple(correction.page_key for correction in entry_corrections),
        )
        for reference_page, entry_corrections in corrections_by_entry.items()
    ]


def filter_corrected_entry_rows(
    rows: Sequence[CorrectedEntryRow], correction_filter: CorrectionFilter
) -> list[CorrectedEntryRow]:
    """Return the rows every chosen filter group lets through, in their order."""
    search_text = correction_filter.search_text.strip().casefold()
    return [
        row
        for row in rows
        if (not correction_filter.changes or row.changes & correction_filter.changes)
        and (
            not correction_filter.sources
            or _list_row_sources(row) & correction_filter.sources
        )
        and (
            not search_text
            or search_text in row.title.casefold()
            or search_text in row.reference_page
            or any(search_text in page_key for page_key in row.page_keys)
        )
    ]


def count_corrected_entry_rows(rows: Sequence[CorrectedEntryRow]) -> dict[str, int]:
    """Count the entries each filter chip would let through, over every row.

    Returns:
        Each change seen, UNCHANGED, ANSWER_SOURCE and RULES_SOURCE mapped to
        the number of entries matching it.
    """
    counts = {UNCHANGED: 0, ANSWER_SOURCE: 0, RULES_SOURCE: 0}
    for row in rows:
        for chip_value in (*row.changes, *_list_row_sources(row)):
            counts[chip_value] = counts.get(chip_value, 0) + 1
    return counts


def build_correction_panel(
    reference_page: str,
    corrections: Sequence[review_cases_pb2.SeedCorrection],
    checked_entries: Sequence[review_cases_pb2.CheckedEntry],
    cases: Sequence[ReviewCase],
) -> CorrectionPanelView | None:
    """Build the panel for one corrected entry.

    Args:
        reference_page: The entry's reference page.
        corrections: Every correction of the last run.
        checked_entries: The entries the last run checked, for titles.
        cases: The review cases, for titles and the links to them.

    Returns:
        The panel, or None when no correction names the entry.
    """
    entry_corrections = [
        correction
        for correction in corrections
        if correction.entry_reference_page == reference_page
    ]
    if not entry_corrections:
        return None
    return CorrectionPanelView(
        reference_page=reference_page,
        reference_address=build_page_address(reference_page),
        title=_map_entry_titles(checked_entries, cases).get(
            reference_page, reference_page
        ),
        rows=tuple(map(_build_correction_row, entry_corrections)),
        case_links=tuple(
            (case.case_name, find_case_title(case))
            for case in cases
            if reference_page in _list_case_entry_reference_pages(case)
        ),
    )


def list_case_correction_links(
    case: ReviewCase, corrections: Sequence[review_cases_pb2.SeedCorrection]
) -> tuple[tuple[str, str, int], ...]:
    """Return each of a case's entries the last run corrected, for the case panel's links.

    Args:
        case: The open review case.
        corrections: Every correction of the last run.

    Returns:
        Each corrected entry's reference page, title and number of corrections,
        in the case's entry order.
    """
    title_by_reference_page = _map_entry_titles((), (case,))
    correction_counts: dict[str, int] = {}
    for correction in corrections:
        correction_counts[correction.entry_reference_page] = (
            correction_counts.get(correction.entry_reference_page, 0) + 1
        )
    return tuple(
        (
            reference_page,
            title_by_reference_page.get(reference_page, reference_page),
            correction_counts[reference_page],
        )
        for reference_page in _list_case_entry_reference_pages(case)
        if reference_page in correction_counts
    )


def _build_correction_row(
    correction: review_cases_pb2.SeedCorrection,
) -> CorrectionRowView:
    """Describe one correction as a row of its entry's table."""
    change = _get_change(correction)
    destination_label, destination_address = _describe_destination(correction)
    mark = ", main" if correction.is_main_page else ""
    mark += ", part" if correction.is_part_page else ""
    return CorrectionRowView(
        page_key=correction.page_key,
        address=correction.address,
        change_label=change[:1].upper() + change[1:],
        place=correction.placement_word + mark,
        destination_label=destination_label,
        destination_address=destination_address,
        reason_sentence=correction.reason_sentence,
        is_from_answer=correction.is_from_answer,
    )


def _describe_destination(
    correction: review_cases_pb2.SeedCorrection,
) -> tuple[str, str | None]:
    """Return where a corrected page goes, and the address to open, if any."""
    if correction.is_added_page:
        return "added to this entry", correction.address
    if correction.HasField("renamed_to"):
        return (
            f"current address {correction.renamed_to}",
            build_page_address(correction.renamed_to),
        )
    if correction.HasField("target_entry_reference_page"):
        return (
            f"seed entry {correction.target_entry_reference_page}",
            build_page_address(correction.target_entry_reference_page),
        )
    if correction.HasField("new_entry_key"):
        return "a new entry", None
    return "stays in this entry", None


def _get_change(correction: review_cases_pb2.SeedCorrection) -> str:
    """Return a correction's change, or UNCHANGED for a page that stays as it was."""
    return correction.change if correction.HasField("change") else UNCHANGED


def _list_row_sources(row: CorrectedEntryRow) -> set[str]:
    """Return the source filter values an entry matches."""
    sources = set()
    if row.has_answer_correction:
        sources.add(ANSWER_SOURCE)
    if row.has_rule_correction:
        sources.add(RULES_SOURCE)
    return sources


def _map_entry_titles(
    checked_entries: Sequence[review_cases_pb2.CheckedEntry],
    cases: Sequence[ReviewCase],
) -> dict[str, str]:
    """Map entry reference pages to titles: the run's checked entries over the cases'."""
    title_by_reference_page: dict[str, str] = {}
    for case in cases:
        for entry in case.entries.values():
            title_by_reference_page[entry.reference_page] = entry.title
        for decision in case.decisions:
            if isinstance(decision, LinkedPageDecision):
                for linking_entry in decision.linking_entries:
                    title_by_reference_page[linking_entry.reference_page] = (
                        linking_entry.title
                    )
    for checked_entry in checked_entries:
        title_by_reference_page[checked_entry.reference_page] = checked_entry.title
    return title_by_reference_page


def _list_case_entry_reference_pages(case: ReviewCase) -> list[str]:
    """Return the reference pages of the entries a case decides about, in its order."""
    reference_pages = list(case.entries)
    for decision in case.decisions:
        if isinstance(decision, LinkedPageDecision):
            reference_pages.extend(
                entry.reference_page
                for entry in decision.linking_entries
                if entry.reference_page not in reference_pages
            )
    return reference_pages
