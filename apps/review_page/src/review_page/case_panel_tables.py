"""Construct presentation tables and comparison models for the case review panel.

Assembles entry headings, candidate provider page rows, side-by-side metadata
comparisons, and background evidence pages consumed by ``case_views.build_case_view``.

Key presentation and constraint rules include:
- Placeable pages: An entry group case presents one row for each placeable
  provider page in deterministic ``list_placeable_case_page_keys`` order,
  annotating the entry label (Work A, Work B) and whether the page serves as
  that entry's reference page.
- Constraint explanations: When placing one page disables choices on another
  (such as assigning two MyAnimeList pages to the same work), the disabling
  rule is displayed once beneath the dropdown with all disabled options.
- Merge targets: Rows offer candidate seed entries for "other work" movements
  only when the target entry does not already hold a page from that provider,
  preventing the move from introducing duplicate providers.
- Evidence pages: Pages found via search without placement obligations (such
  as AnimeSchedule entries) are displayed in an evidence table for context.
- Linked page comparisons: A linked-page case presents side-by-side attribute
  comparisons against its linking entry to help reviewers decide whether to
  keep, split, or ignore the link.
- Difference filtering: The "Show only differences" mode hides detail columns
  and comparison rows whose values are identical across all candidate pages.
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
    get_page_provider,
    list_group_page_choices,
)

# Rules that disable a page's choice because of where other pages are placed.
_RULES_FROM_OTHER_PAGES = frozenset({PlacementRule.WORK_WITH_TWO_MAL_PAGES})


def build_page_rows(
    case: ReviewCase,
    chosen_answer: JsonValue,
    original_answer: JsonValue = None,
    *,
    is_rejected: bool,
) -> tuple[PageRowView, ...]:
    """Build candidate provider page rows with details, placement options, and flags.

    Args:
        case: The review case containing candidate provider pages.
        chosen_answer: The placement mappings currently selected in the UI.
        original_answer: The baseline answer payload from the previous pipeline run.
        is_rejected: Whether the last save attempt failed constraint validation.

    Returns:
        A tuple of PageRowView instances in deterministic placeable key order.
    """
    answer = chosen_answer if isinstance(chosen_answer, dict) else {}
    base_answer = original_answer if isinstance(original_answer, dict) else {}
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
            != base_answer.get(choices.page_key),
            disabled_choice_notes=_explain_disabled_choices(choices),
            is_place_missing=is_rejected
            and choices.is_required
            and choices.work is None,
            linked_pages=_list_page_links(pages_by_key.get(choices.page_key)),
            target_entry_options=tuple(
                suggestion
                for suggestion in case.target_entry_suggestions
                if get_page_provider(choices.page_key) not in suggestion.providers
            ),
        )
        for choices in list_group_page_choices(case, answer)
    )


def _explain_disabled_choices(choices: GroupPageChoices) -> tuple[str, ...]:
    """Explain placement options disabled by placement constraints on other pages.

    Aggregates choices disabled by multi-page rules (such as allowing only one
    MyAnimeList page per work) into a single concise note per broken rule.

    Args:
        choices: Placement options for a single page, including any disabling rule.

    Returns:
        A tuple of explanatory notes, one per violated rule, listing all disabled options.
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
    """Join option names into a natural grammatical list ("a, b and c").

    Args:
        names: One or more placement option names to join.

    Returns:
        The option names joined with commas and a final "and".
    """
    if len(names) == 1:
        return names[0]
    return f"{', '.join(names[:-1])} and {names[-1]}"


def _label_entry_pages(case: ReviewCase) -> tuple[dict[str, str], set[str]]:
    """Map each page key to its entry label (A, B, ...) and collect reference page keys.

    Args:
        case: The review case containing work entries.

    Returns:
        A tuple containing a dictionary mapping each page key to the label of
        the first entry holding that page, and the set of reference page keys.
    """
    entry_label_by_page_key: dict[str, str] = {}
    for label, entry in zip(_build_entry_labels(), case.entries.values(), strict=False):
        for page_key in entry.addresses_by_page_key:
            entry_label_by_page_key.setdefault(page_key, label)
    return entry_label_by_page_key, set(case.entries)


def build_entry_views(case: ReviewCase) -> tuple[EntryView, ...]:
    """Build display headings for each work entry in the review case.

    Labels entries sequentially (A, B, ...) according to their order in the case.

    Args:
        case: The review case containing work entries.

    Returns:
        A tuple of EntryView instances with labels, titles, metadata summaries,
        and reference page keys.
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
            _build_entry_labels(), case.entries.items(), strict=False
        )
    )


def build_comparison_rows(
    case: ReviewCase, show_only_differences: bool
) -> tuple[ComparisonRowView, ...]:
    """Compare a candidate linked page's metadata against its linking work entry.

    Args:
        case: The review case; only linked-page cases produce comparison rows.
        show_only_differences: Whether to filter out rows where both values match.

    Returns:
        A tuple of ComparisonRowView instances for each compared attribute,
        or an empty tuple for entry group cases.
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
    """Determine which detail columns to display in the candidate page table.

    Args:
        page_rows: The candidate page rows rendered for the review case.
        show_only_differences: Whether to hide columns with identical values across rows.

    Returns:
        All DetailColumn members when difference filtering is off; otherwise only
        columns whose values vary across two or more rows.
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
    """List background provider pages that require no placement decision.

    Args:
        case: The review case containing all fetched provider pages.
        page_rows: The candidate page rows that require placement decisions.

    Returns:
        A tuple of ProviderPage instances not present in candidate rows, linked
        page decisions, or work entries.
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
    """Format a provider page's metadata attributes into table column strings.

    Args:
        page: The fetched provider page details, or None for an unfetched page.

    Returns:
        A dictionary mapping each DetailColumn to its formatted display string,
        using an em dash for missing values.
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
    """Resolve the external URL for the candidate page in a linked-page case.

    Args:
        case: The review case under inspection.

    Returns:
        The external URL of the linked page, or an empty string for entry group cases.
    """
    [decision] = case.decisions
    if not isinstance(decision, LinkedPageDecision):
        return ""
    return _find_page_address(
        case, decision.page_key, {page.key: page for page in case.pages}
    )


def _list_page_links(page: ProviderPage | None) -> tuple[PageLink, ...]:
    """Build external hyperlink views for all pages referenced by a provider page.

    Args:
        page: The fetched provider page details, or None for an unfetched page.

    Returns:
        A tuple of PageLink instances pairing each linked page key with its
        resolved provider URL, or an empty tuple when unfetched.
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
    """Resolve a provider page's external URL from case entries or fetched details.

    Args:
        case: The review case containing work entries.
        page_key: The provider page key to look up.
        pages_by_key: Fetched provider pages indexed by page key and current key.

    Returns:
        The external URL string, or an empty string when neither source holds it.
    """
    for entry in case.entries.values():
        if page_key in entry.addresses_by_page_key:
            return entry.addresses_by_page_key[page_key]
    page = pages_by_key.get(page_key)
    return page.address if page else ""


def _build_entry_labels() -> list[str]:
    """Generate entry labels A to Z, followed by numbers for larger entry groups.

    Returns:
        A list of sequential work entry labels starting with A through Z,
        then numeric strings 27 through 999.
    """
    return [chr(code) for code in range(ord("A"), ord("Z") + 1)] + [
        str(number) for number in range(27, 1000)
    ]


def _show_value(value: object) -> str:
    """Format a detail value for display, returning an em dash for missing values.

    Args:
        value: The metadata attribute value to display, or None.

    Returns:
        The string representation of the value, or an em dash when empty or None.
    """
    return "—" if value in (None, "") else str(value)
