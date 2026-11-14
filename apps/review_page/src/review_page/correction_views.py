"""Build presentation models for the Corrections tab from review case source messages.

Constructs corrected entry summary rows, filter chip counts, and per-entry page
correction panels exclusively from ``ReviewCaseSource`` protobuf messages so the
tab operates independently of the underlying storage backend.

Each presentation rule reads `<rule>: <reason>`:
- Entry-grouped rows: The list displays one row per corrected entry in source
  order because reviewers inspect corrections one work entry at a time.
- Title resolution precedence: Entry titles resolve from the run's checked
  entries first, then from any review case containing the entry, and finally
  fall back to the entry reference page key.
- Filter combination: Selecting multiple chips within a single filter category
  (changes or sources) widens matches with an OR condition, whereas selecting
  across categories narrows matches with an AND condition.
- Destination formatting: Each page row describes its destination (target seed
  entry reference page with link, new entry, renamed current address, or
  unchanged) without requiring the target entry's title.
- Case cross-linking: The entry panel links every review case referencing that
  entry so reviewers can navigate directly between a correction and its answer.
"""

from collections import Counter
from collections.abc import Sequence

from review_page.case_views import find_case_title
from review_page.models import (
    CorrectedEntryRow,
    CorrectionFilter,
    CorrectionPanelView,
    CorrectionRowView,
)
from review_page.template_functions import capitalize_first_letter
from shared_proto.v1 import review_cases_pb2
from work_matching import (
    ReviewCase,
    WorkEntry,
    build_page_address,
)

# Filter chip value matching entries with at least one unchanged page.
UNCHANGED = "unchanged"
# Filter chip values distinguishing answer-driven placements from rule-only placements.
ANSWER_SOURCE = "your answer"
RULES_SOURCE = "rules"


def build_corrected_entry_rows(
    corrections: Sequence[review_cases_pb2.SeedCorrection],
    checked_entries: Sequence[review_cases_pb2.CheckedEntry],
    cases: Sequence[ReviewCase],
) -> list[CorrectedEntryRow]:
    """Build one summary row per corrected entry in the order corrections appear.

    Args:
        corrections: All seed correction messages from the most recent run.
        checked_entries: Checked entry messages from the last run for resolving titles.
        cases: Review cases used as a fallback source for entry titles.

    Returns:
        A list of ``CorrectedEntryRow`` models, one per corrected entry reference page.
    """
    title_by_reference_page = _map_entry_titles(checked_entries, cases)
    corrections_by_entry: dict[str, list[review_cases_pb2.SeedCorrection]] = {}
    for correction in corrections:
        corrections_by_entry.setdefault(correction.entry_reference_page, []).append(
            correction
        )
    return [
        CorrectedEntryRow(
            entry_reference_page=reference_page,
            title=title_by_reference_page.get(reference_page, reference_page),
            changes=frozenset(map(_read_change_name, entry_corrections)),
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
    """Filter corrected entry rows by selected change chips, source chips, and search text.

    Args:
        rows: All corrected entry summary rows before filtering.
        correction_filter: Active change chips, source chips, and search query.

    Returns:
        The subset of ``CorrectedEntryRow`` models matching all active filter groups.
    """
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
            or search_text in row.entry_reference_page
            or any(search_text in page_key for page_key in row.page_keys)
        )
    ]


def count_corrected_entry_rows(rows: Sequence[CorrectedEntryRow]) -> dict[str, int]:
    """Count how many corrected entry rows match each change and source filter chip.

    Args:
        rows: All corrected entry summary rows before filtering.

    Returns:
        A dictionary mapping each change label, ``UNCHANGED``, ``ANSWER_SOURCE``,
        and ``RULES_SOURCE`` to the number of matching entry rows.
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
    """Build the detail panel view for a single corrected seed entry.

    Args:
        reference_page: The reference page key identifying the seed entry.
        corrections: All seed correction messages from the most recent run.
        checked_entries: Checked entry messages from the last run for resolving titles.
        cases: Review cases used to resolve fallback titles and build cross-links.

    Returns:
        The assembled ``CorrectionPanelView``, or None when no corrections exist
        for ``reference_page``.
    """
    entry_corrections = [
        correction
        for correction in corrections
        if correction.entry_reference_page == reference_page
    ]
    if not entry_corrections:
        return None
    return CorrectionPanelView(
        entry_reference_page=reference_page,
        entry_reference_address=build_page_address(reference_page),
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
    """List the case's entries the last run changed, for cross-linking in the case panel.

    Only corrections that change the seed are counted. A page an answer
    places where it already is still gets a correction, but that correction
    changes nothing, so counting it would overstate what the run did.

    Args:
        case: The active review case being inspected.
        corrections: All seed correction messages from the most recent run.

    Returns:
        A tuple of ``(reference_page, title, change_count)`` triples in the
        case's entry order, for entries with at least one change.
    """
    title_by_reference_page = _map_entry_titles((), (case,))
    change_counts = Counter(
        correction.entry_reference_page
        for correction in corrections
        if correction.HasField("change")
    )
    return tuple(
        (
            reference_page,
            title_by_reference_page.get(reference_page, reference_page),
            change_counts[reference_page],
        )
        for reference_page in _list_case_entry_reference_pages(case)
        if change_counts[reference_page]
    )


def _build_correction_row(
    correction: review_cases_pb2.SeedCorrection,
) -> CorrectionRowView:
    """Convert a single ``SeedCorrection`` message into a ``CorrectionRowView`` model.

    Args:
        correction: A single seed correction protobuf message.

    Returns:
        The populated ``CorrectionRowView`` with formatted change, placement,
        destination, and provenance fields.
    """
    change_name = _read_change_name(correction)
    destination_label, destination_address = _describe_destination(correction)
    partial_targets = [
        *correction.partial_entry_reference_pages,
        *correction.partial_new_entry_keys,
    ]
    partial_suffix = (
        f", partial source of {', '.join(partial_targets)}" if partial_targets else ""
    )
    return CorrectionRowView(
        page_key=correction.page_key,
        address=correction.address,
        change_label=capitalize_first_letter(change_name),
        place=correction.placement_word + partial_suffix
        if correction.HasField("placement_word")
        else partial_suffix.removeprefix(", ") or "—",
        destination_label=destination_label,
        destination_address=destination_address,
        reason_sentence=correction.reason_sentence,
        is_from_answer=correction.is_from_answer,
    )


def _describe_destination(
    correction: review_cases_pb2.SeedCorrection,
) -> tuple[str, str | None]:
    """Format the destination label and optional external link URL for a correction.

    Args:
        correction: A single seed correction protobuf message.

    Returns:
        A tuple of ``(destination_label, destination_address)`` where
        ``destination_address`` is None when no external link applies.
    """
    if correction.is_dead_page:
        return "dropped, since the provider no longer has it", None
    if correction.is_dropped_page:
        return "dropped, since it duplicates another page of its provider", None
    if correction.is_added_page:
        return (
            "added to this entry's partial sources"
            if correction.partial_entry_reference_pages
            and not correction.HasField("placement_word")
            else "added to this entry",
            correction.address,
        )
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


def _read_change_name(correction: review_cases_pb2.SeedCorrection) -> str:
    """Read a correction's change category name, defaulting to ``UNCHANGED`` when unset.

    Args:
        correction: A single seed correction protobuf message.

    Returns:
        The change category string, or ``UNCHANGED`` when the page is unchanged.
    """
    return correction.change if correction.HasField("change") else UNCHANGED


def _list_row_sources(row: CorrectedEntryRow) -> set[str]:
    """Identify which source filter chip values apply to a corrected entry row.

    Args:
        row: A corrected entry summary row.

    Returns:
        A set containing ``ANSWER_SOURCE``, ``RULES_SOURCE``, or both.
    """
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
    """Build a lookup mapping entry reference page keys to display titles.

    Titles from ``checked_entries`` take precedence over titles extracted from
    review cases.

    Args:
        checked_entries: Checked entry messages from the most recent pipeline run.
        cases: All review cases for fallback title extraction.

    Returns:
        A dictionary mapping each entry reference page key to its resolved title.
    """
    title_by_reference_page: dict[str, str] = {}
    for case in cases:
        for entry in _list_case_entries(case):
            title_by_reference_page[entry.reference_page_key] = entry.title
    for checked_entry in checked_entries:
        title_by_reference_page[checked_entry.entry_reference_page] = (
            checked_entry.title
        )
    return title_by_reference_page


def _list_case_entry_reference_pages(case: ReviewCase) -> list[str]:
    """Collect the reference page keys of all entries involved in a review case.

    Args:
        case: The review case to inspect.

    Returns:
        A deduplicated list of entry reference page keys in the case's entry order.
    """
    return list(
        dict.fromkeys(entry.reference_page_key for entry in _list_case_entries(case))
    )


def _list_case_entries(case: ReviewCase) -> list[WorkEntry]:
    """List every seed entry a review case involves.

    Args:
        case: The review case to inspect.

    Returns:
        The case's entries, followed by the entries linking to its linked pages.
    """
    linking_entries = (
        case.linked_page_decision.linking_entries if case.linked_page_decision else ()
    )
    return [*case.entries.values(), *linking_entries]
