"""Build the case panel's tables: entries, page rows, comparison rows and evidence.

``case_views.build_case_view`` assembles the panel from these parts. Each item
gives the rule, then the reason for it.

- An entry group's case shows one row per page an answer can place, in
  ``list_placeable_case_page_keys`` order. Each row names the entry listing
  the page (A, B, ...) and whether the page is that entry's reference page.
- A rule another page's place breaks is stated once under the dropdown, with
  every choice it disables, so a case with many MAL pages does not repeat it.
- Pages that were only searched, such as AnimeSchedule's, are shown as
  evidence, since an answer cannot place them.
- A linked-page case compares the page's details with the linking entry's,
  row by row, so the reviewer sees where they differ.
- Show only differences keeps only the detail columns whose values differ
  between rows, and only the comparison rows where the entry and the page
  differ.
"""

from collections.abc import Mapping, Sequence

from pydantic import JsonValue
from review_page.models import (
    ComparisonRowView,
    DetailColumn,
    EntryView,
    PageLink,
    PageRowView,
)
from work_matching import (
    GroupPageChoices,
    LinkedPageDecision,
    PlacementRule,
    ProviderPage,
    ReviewCase,
    build_page_address,
    list_group_page_choices,
)

# Rules that disable a page's choice because of where other pages are placed.
_RULES_FROM_OTHER_PAGES = frozenset({PlacementRule.WORK_WITH_TWO_MAL_PAGES})


def build_page_rows(
    case: ReviewCase,
    chosen_answer: JsonValue,
    original: JsonValue,
    *,
    is_rejected: bool,
) -> tuple[PageRowView, ...]:
    """Build one row per placeable page, with its details, choices and marks.

    Args:
        case: An entry group's case.
        chosen_answer: The answer the case's choices show now.
        original: The case's answer as the last run wrote it.
        is_rejected: Whether the case's last save broke a rule.

    Returns:
        One row per placeable page, in ``list_placeable_case_page_keys`` order.
    """
    answer = chosen_answer if isinstance(chosen_answer, dict) else {}
    original_answer = original if isinstance(original, dict) else {}
    pages_by_key = {
        key: page for page in case.pages for key in (page.key, page.current_key)
    }
    entry_label_by_page_key, reference_page_keys = _label_entry_pages(case)
    return tuple(
        PageRowView(
            page_key=choices.page_key,
            address=_find_page_address(case, choices.page_key, pages_by_key),
            entry_label=entry_label_by_page_key.get(choices.page_key, ""),
            is_reference_page=choices.page_key in reference_page_keys,
            details=_describe_page(pages_by_key.get(choices.page_key)),
            choices=choices,
            is_changed=answer.get(choices.page_key)
            != original_answer.get(choices.page_key),
            disabled_choice_notes=_explain_disabled_choices(choices),
            is_place_missing=is_rejected
            and choices.is_required
            and choices.work is None,
            linked_pages=_list_page_links(pages_by_key.get(choices.page_key)),
        )
        for choices in list_group_page_choices(case, answer)
    )


def _explain_disabled_choices(choices: GroupPageChoices) -> tuple[str, ...]:
    """Explain the choices other pages' places disable, one line per rule.

    Each item gives the rule, then the reason for it.

    - A rule disabling several choices, such as a work for each other MAL
      page, is stated once with every choice it disables. A case with many
      MAL pages then does not repeat the same sentence on every row.
    - A choice the page's own place disables, such as ``main`` for a
      provider's only page, is explained by the option's tooltip alone. Most
      rows then stay quiet.

    Args:
        choices: One page's choices, with the rule disabling each one.

    Returns:
        One note per rule another page's place breaks, naming the choices the
        rule disables.
    """
    choices_by_rule: dict[PlacementRule, list[str]] = {}
    for option in (*choices.work_options, *choices.mark_options):
        if option.disabling_rule in _RULES_FROM_OTHER_PAGES:
            choices_by_rule.setdefault(option.disabling_rule, []).append(
                option.value or "—"
            )
    return tuple(
        f"{_join_choice_names(names)} {'is' if len(names) == 1 else 'are'} not "
        f"offered: {rule.value}"
        for rule, names in choices_by_rule.items()
    )


def _join_choice_names(names: list[str]) -> str:
    """Join choice names the way a sentence lists them: "a, b and c".

    Args:
        names: One or more choice names.

    Returns:
        The names joined with commas and a final "and".
    """
    if len(names) == 1:
        return names[0]
    return f"{', '.join(names[:-1])} and {names[-1]}"


def _label_entry_pages(case: ReviewCase) -> tuple[dict[str, str], set[str]]:
    """Return each page's entry label (A, B, ...) and the entries' reference pages.

    Args:
        case: The review case.

    Returns:
        Each page key mapped to the label of the first entry listing the page,
        and the set of the entries' reference pages.
    """
    entry_label_by_page_key: dict[str, str] = {}
    for label, entry in zip(_entry_labels(), case.entries.values(), strict=False):
        for page_key in entry.addresses_by_page_key:
            entry_label_by_page_key.setdefault(page_key, label)
    return entry_label_by_page_key, set(case.entries)


def build_entry_views(case: ReviewCase) -> tuple[EntryView, ...]:
    """Describe each entry of the case, labelled A, B, ... in the case's order.

    Args:
        case: The review case.

    Returns:
        One heading per entry, with its label, title, summary and reference
        page.
    """
    return tuple(
        EntryView(
            label=label,
            title=entry.title,
            summary=" · ".join(
                str(value) for value in (entry.type, entry.year, entry.season) if value
            ),
            reference_page=reference_page,
        )
        for label, (reference_page, entry) in zip(
            _entry_labels(), case.entries.items(), strict=False
        )
    )


def build_comparison_rows(
    case: ReviewCase, show_only_differences: bool
) -> tuple[ComparisonRowView, ...]:
    """Compare a linked page's details with its one linking entry's.

    Args:
        case: The review case; only a linked-page case has comparison rows.
        show_only_differences: Whether to keep only the rows that differ.

    Returns:
        One row per detail, marking where the entry and the page differ; none
        for an entry group's case.
    """
    [decision] = case.decisions
    if not isinstance(decision, LinkedPageDecision):
        return ()
    page = next(
        (page for page in case.pages if page.current_key == decision.page_key), None
    )
    entry = decision.linking_entries[0]
    pairs = (
        ("Type", entry.type, page.type if page else None),
        ("Year", entry.year, page.year if page else None),
        ("Season", entry.season, page.season if page else None),
        (
            "Episodes",
            decision.entry_episode_count,
            page.episode_count if page else None,
        ),
        ("Titles", entry.title, ", ".join(page.titles) if page else None),
    )
    rows = (
        ComparisonRowView(
            label=label,
            entry_value=_show_value(entry_value),
            page_value=_show_value(page_value),
            is_different=_show_value(entry_value) != _show_value(page_value),
        )
        for label, entry_value, page_value in pairs
    )
    return tuple(row for row in rows if row.is_different or not show_only_differences)


def list_shown_columns(
    page_rows: Sequence[PageRowView], show_only_differences: bool
) -> tuple[DetailColumn, ...]:
    """Return the detail columns to show; with the switch on, the ones that differ.

    Args:
        page_rows: The case's page rows.
        show_only_differences: Whether the "Show only differences" switch is on.

    Returns:
        Every column with the switch off; otherwise the columns whose values
        differ between rows.
    """
    if not show_only_differences:
        return tuple(DetailColumn)
    return tuple(
        column
        for column in DetailColumn
        if len({row.details[column] for row in page_rows}) > 1
    )


def list_evidence_pages(
    case: ReviewCase, page_rows: Sequence[PageRowView]
) -> tuple[ProviderPage, ...]:
    """Return the case's pages no answer places, such as pages found by a search.

    Args:
        case: The review case.
        page_rows: The case's page rows, the pages an answer places.

    Returns:
        The case's pages that no row, linked page or entry holds.
    """
    placed_page_keys = {row.page_key for row in page_rows}
    linked_page_keys = {
        decision.page_key
        for decision in case.decisions
        if isinstance(decision, LinkedPageDecision)
    }
    entry_page_keys = {
        key for entry in case.entries.values() for key in entry.addresses_by_page_key
    }
    return tuple(
        page
        for page in case.pages
        if page.key not in placed_page_keys | linked_page_keys | entry_page_keys
        and page.current_key not in placed_page_keys | linked_page_keys
    )


def _describe_page(page: ProviderPage | None) -> dict[DetailColumn, str]:
    """Return a page's details as the table shows them; blank for an unfetched page.

    Args:
        page: The page's fetched details, or None for an unfetched page.

    Returns:
        Each detail column's text; an em dash where a value is missing.
    """
    if page is None:
        return dict.fromkeys(DetailColumn, "—")
    return {
        DetailColumn.TITLES: ", ".join(page.titles) or "—",
        DetailColumn.TYPE: _show_value(page.type),
        DetailColumn.EPISODE_COUNT: _show_value(page.episode_count),
        DetailColumn.AIRED_FROM: _show_value(
            page.aired_from and page.aired_from.date()
        ),
        DetailColumn.AIRED_TO: _show_value(page.aired_to and page.aired_to.date()),
        DetailColumn.LINKS: ", ".join(page.linked_page_keys) or "—",
    }


def find_linked_page_address(case: ReviewCase) -> str:
    """Return the address of a linked-page case's page, from its fetched details.

    Args:
        case: The review case.

    Returns:
        The linked page's address, or an empty string for an entry group's
        case.
    """
    [decision] = case.decisions
    if not isinstance(decision, LinkedPageDecision):
        return ""
    return _find_page_address(
        case, decision.page_key, {page.key: page for page in case.pages}
    )


def _list_page_links(page: ProviderPage | None) -> tuple[PageLink, ...]:
    """Return the pages a fetched page links to, each with its provider address.

    Args:
        page: The page's fetched details, or None for an unfetched page.

    Returns:
        Each linked page key with its provider address; none for an unfetched
        page.
    """
    if page is None:
        return ()
    return tuple(
        PageLink(page_key, build_page_address(page_key))
        for page_key in page.linked_page_keys
    )


def _find_page_address(
    case: ReviewCase, page_key: str, pages_by_key: Mapping[str, ProviderPage]
) -> str:
    """Return a page's address from its entry, else from its fetched details.

    Args:
        case: The review case.
        page_key: The page whose address to find.
        pages_by_key: The case's fetched pages, by page key and current key.

    Returns:
        The page's address, or an empty string when neither source has it.
    """
    for entry in case.entries.values():
        if page_key in entry.addresses_by_page_key:
            return entry.addresses_by_page_key[page_key]
    page = pages_by_key.get(page_key)
    return page.address if page else ""


def _entry_labels() -> list[str]:
    """Return entry labels A to Z, then numbers for larger groups.

    Returns:
        The labels in order: A to Z, then 27 to 999.
    """
    return [chr(code) for code in range(ord("A"), ord("Z") + 1)] + [
        str(number) for number in range(27, 1000)
    ]


def _show_value(value: object) -> str:
    """Write a detail for the page: an em dash for a missing value.

    Args:
        value: A page detail, or None.

    Returns:
        The value as text, or an em dash when the value is missing.
    """
    return "—" if value in (None, "") else str(value)
