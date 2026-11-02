"""Turn review cases into what the page's templates show.

The templates only lay out what these views hold; every decision is made
here or in the libraries, so it is tested without a browser. Each item gives
the rule, then the reason for it.

- An entry group's case shows one row per page an answer can place, in
  ``list_placeable_case_page_keys`` order, with the entry listing it (A, B,
  ...) and whether it is that entry's reference page. Pages that were only
  searched, such as AnimeSchedule's, are shown as evidence, since an answer
  cannot place them.
- A case starts on its unsaved edit, else its saved answer, else its
  suggestion, which is then marked as not yet answered.
- A page whose place differs from the case's original value is marked
  changed, so the reviewer sees what a save would change.
- Show only differences hides each detail column that is the same on every
  row, and each comparison row where the entry and the linked page agree. It
  never hides a page or its choices.
- A list row's title is the combined entry's title for an entry group (the
  most trusted entry's), and the linking entry's for a linked page.
"""

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from pydantic import JsonValue
from review_page.page_state import (
    CaseButtons,
    CaseEditState,
    CaseRow,
    find_case_buttons,
    find_case_edit_state,
    find_saved_change,
)
from work_matching import (
    ChoiceOption,
    GroupPageChoices,
    LinkedPageDecision,
    PlacementRule,
    ProviderPage,
    ReviewCase,
    ReviewCaseKind,
    build_page_address,
    combine_group_entries,
    describe_case_reasons,
    list_group_page_choices,
    list_linked_page_options,
)
from work_matching.answer_words import PLACEMENT_WORD_MEANINGS

# Rules that disable a page's choice because of where other pages are placed.
_RULES_FROM_OTHER_PAGES = frozenset({PlacementRule.WORK_WITH_TWO_MAL_PAGES})


class DetailColumn(StrEnum):
    """A page detail shown in an entry group's page table, by its heading."""

    TITLES = "Titles"
    TYPE = "Type"
    EPISODE_COUNT = "Eps"
    AIRED_FROM = "Aired from"
    AIRED_TO = "Aired to"
    LINKS = "Links"


@dataclass(frozen=True, slots=True)
class PageRowView:
    """One page of an entry group's case, with its details and choices."""

    page_key: str
    address: str
    entry_label: str
    is_reference_page: bool
    details: dict[DetailColumn, str]
    choices: GroupPageChoices
    # Whether the page's place differs from the case's original value.
    is_changed: bool
    # Why choices that other pages' places disable are not offered.
    disabled_choice_notes: tuple[str, ...] = ()
    # Whether a rejected save left this page, which the answer must place,
    # without a place, so the reviewer sees which page the rule is about.
    is_place_missing: bool = False
    # The pages this page links to, so the Links column opens each one.
    linked_pages: tuple[PageLink, ...] = ()


@dataclass(frozen=True, slots=True)
class PageLink:
    """A page key a page links to, with the provider address it opens."""

    page_key: str
    # None for a key that names no known provider's page; shown as text.
    address: str | None


@dataclass(frozen=True, slots=True)
class ComparisonRowView:
    """One detail of a linked-page case: the entry's value against the page's."""

    label: str
    entry_value: str
    page_value: str
    is_different: bool


@dataclass(frozen=True, slots=True)
class EntryView:
    """One entry of a case, as its heading shows it."""

    label: str
    title: str
    summary: str
    reference_page: str


@dataclass(frozen=True, slots=True)
class CaseView:
    """Everything the case panel shows for one case."""

    case: ReviewCase
    # The title the case list shows, so the heading matches the list.
    title: str
    edit_state: CaseEditState
    chosen_answer: JsonValue
    is_suggestion_shown: bool
    reason_sentences: tuple[str, ...]
    entries: tuple[EntryView, ...]
    page_rows: tuple[PageRowView, ...]
    shown_columns: tuple[DetailColumn, ...]
    evidence_pages: tuple[ProviderPage, ...]
    linked_page_options: tuple[ChoiceOption, ...]
    comparison_rows: tuple[ComparisonRowView, ...]
    # The linked page's address for a linked-page case, so its key opens it.
    linked_page_address: str
    # Each placement word and its meaning, listed under an entry group's pages.
    choice_meanings: tuple[tuple[str, str], ...]
    buttons: CaseButtons
    show_only_differences: bool


def find_original_answer(
    case_name: str,
    saved_answer: JsonValue,
    original_answers: Mapping[str, JsonValue] | None,
) -> JsonValue:
    """Return a case's original value: its answer as the last run wrote it.

    Args:
        case_name: The case.
        saved_answer: Its answer in the file now.
        original_answers: The last run's answers, or None before any run wrote
            a result, when the file's answers stand as originals.
    """
    if original_answers is None:
        return saved_answer
    return original_answers.get(case_name)


def build_case_rows(
    cases: Sequence[ReviewCase],
    *,
    original_answers: Mapping[str, JsonValue] | None,
    new_case_names: Collection[str],
    unused_answer_case_names: Collection[str] = (),
) -> list[CaseRow]:
    """Build the case list's rows, in the cases' order.

    Args:
        cases: The review cases.
        original_answers: The last run's answers, or None before any run.
        new_case_names: The cases the last run brought up for the first time.
        unused_answer_case_names: The kept answers whose case the last run no
            longer brought up.
    """
    return [
        CaseRow(
            case_name=case.case_name,
            kind=case.kind,
            title=find_case_title(case),
            saved_change=find_saved_change(
                find_original_answer(case.case_name, case.answer, original_answers),
                case.answer,
            ),
            is_open=case.answer in (None, {}),
            is_new=case.case_name in new_case_names,
            has_suggestion=bool(case.suggested_answer),
            is_answer_unused=case.case_name in unused_answer_case_names,
        )
        for case in cases
    ]


def build_case_view(
    case: ReviewCase,
    *,
    original: JsonValue,
    unsaved: JsonValue,
    is_rejected: bool,
    is_run_going: bool,
    show_only_differences: bool,
) -> CaseView:
    """Build the case panel's view, starting on the edit, saved answer or suggestion.

    Args:
        case: The review case, with its saved answer.
        original: Its answer as the last run wrote it.
        unsaved: The browser's unsaved answer, or None.
        is_rejected: Whether the case's last save broke a rule.
        is_run_going: Whether a run is rewriting the cases.
        show_only_differences: Whether to hide details every page shares.
    """
    is_suggestion_shown = unsaved is None and case.answer in (None, {})
    chosen_answer: JsonValue = (
        unsaved
        if unsaved is not None
        else (_copy_suggested_answer(case) if is_suggestion_shown else case.answer)
    )
    is_linked_page = case.kind is ReviewCaseKind.LINKED_PAGE
    page_rows = (
        ()
        if is_linked_page
        else _build_page_rows(
            case,
            chosen_answer,
            # An unanswered case starts from its suggestion, so rows are
            # marked where they differ from what the reviewer was first shown.
            original
            if original not in (None, {}) or not case.suggested_answer
            else _copy_suggested_answer(case),
            is_rejected=is_rejected,
        )
    )
    comparison_rows = _build_comparison_rows(case) if is_linked_page else ()
    return CaseView(
        case=case,
        title=find_case_title(case),
        edit_state=find_case_edit_state(
            original=original,
            saved=case.answer,
            unsaved=unsaved,
            has_unsaved_edit=unsaved is not None,
            is_rejected=is_rejected,
        ),
        chosen_answer=chosen_answer,
        is_suggestion_shown=is_suggestion_shown and bool(case.suggested_answer),
        reason_sentences=describe_case_reasons(case),
        entries=_build_entry_views(case),
        page_rows=page_rows,
        shown_columns=_list_shown_columns(page_rows, show_only_differences),
        evidence_pages=_list_evidence_pages(case, page_rows),
        linked_page_options=list_linked_page_options(case) if is_linked_page else (),
        linked_page_address=_find_linked_page_address(case) if is_linked_page else "",
        choice_meanings=() if is_linked_page else PLACEMENT_WORD_MEANINGS,
        comparison_rows=tuple(
            row
            for row in comparison_rows
            if row.is_different or not show_only_differences
        ),
        buttons=find_case_buttons(
            case,
            original=original,
            saved=case.answer,
            chosen=chosen_answer,
            is_run_going=is_run_going,
        ),
        show_only_differences=show_only_differences,
    )


def _build_page_rows(
    case: ReviewCase,
    chosen_answer: JsonValue,
    original: JsonValue,
    *,
    is_rejected: bool,
) -> tuple[PageRowView, ...]:
    """Build one row per placeable page, with its details, choices and marks."""
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

    A rule disabling several choices, such as a work for each other MAL page,
    is stated once with every choice it disables, so a case with many MAL
    pages does not repeat the same sentence on every row. Choices a page's own
    place disables, such as ``main`` for a provider's only page, are explained
    by the option's tooltip alone, so most rows stay quiet.
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
    """Join choice names the way a sentence lists them: "a, b and c"."""
    if len(names) == 1:
        return names[0]
    return f"{', '.join(names[:-1])} and {names[-1]}"


def _label_entry_pages(case: ReviewCase) -> tuple[dict[str, str], set[str]]:
    """Return each page's entry label (A, B, ...) and the entries' reference pages."""
    entry_label_by_page_key: dict[str, str] = {}
    for label, entry in zip(_entry_labels(), case.entries.values(), strict=False):
        for page_key in entry.addresses_by_page_key:
            entry_label_by_page_key.setdefault(page_key, label)
    return entry_label_by_page_key, set(case.entries)


def _build_entry_views(case: ReviewCase) -> tuple[EntryView, ...]:
    """Describe each entry of the case, labelled A, B, ... in the case's order."""
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


def _build_comparison_rows(case: ReviewCase) -> tuple[ComparisonRowView, ...]:
    """Compare a linked page's details with its one linking entry's."""
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
    return tuple(
        ComparisonRowView(
            label=label,
            entry_value=_show_value(entry_value),
            page_value=_show_value(page_value),
            is_different=_show_value(entry_value) != _show_value(page_value),
        )
        for label, entry_value, page_value in pairs
    )


def _list_shown_columns(
    page_rows: Sequence[PageRowView], show_only_differences: bool
) -> tuple[DetailColumn, ...]:
    """Return the detail columns to show; with the switch on, the ones that differ."""
    if not show_only_differences:
        return tuple(DetailColumn)
    return tuple(
        column
        for column in DetailColumn
        if len({row.details[column] for row in page_rows}) > 1
    )


def _list_evidence_pages(
    case: ReviewCase, page_rows: Sequence[PageRowView]
) -> tuple[ProviderPage, ...]:
    """Return the case's pages no answer places, such as pages found by a search."""
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
    """Return a page's details as the table shows them; blank for an unfetched page."""
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


def _find_linked_page_address(case: ReviewCase) -> str:
    """Return the address of a linked-page case's page, from its fetched details."""
    [decision] = case.decisions
    if not isinstance(decision, LinkedPageDecision):
        return ""
    return _find_page_address(
        case, decision.page_key, {page.key: page for page in case.pages}
    )


def _list_page_links(page: ProviderPage | None) -> tuple[PageLink, ...]:
    """Return the pages a fetched page links to, each with its provider address."""
    if page is None:
        return ()
    return tuple(
        PageLink(page_key, build_page_address(page_key))
        for page_key in page.linked_page_keys
    )


def _find_page_address(
    case: ReviewCase, page_key: str, pages_by_key: Mapping[str, ProviderPage]
) -> str:
    """Return a page's address from its entry, else from its fetched details."""
    for entry in case.entries.values():
        if page_key in entry.addresses_by_page_key:
            return entry.addresses_by_page_key[page_key]
    page = pages_by_key.get(page_key)
    return page.address if page else ""


def find_case_title(case: ReviewCase) -> str:
    """Return the title a case is listed under."""
    if case.kind is ReviewCaseKind.LINKED_PAGE:
        [decision] = case.decisions
        if isinstance(decision, LinkedPageDecision):
            return decision.linking_entries[0].title
    return combine_group_entries(list(case.entries.values())).title


def _copy_suggested_answer(case: ReviewCase) -> dict[str, JsonValue]:
    """Return the case's suggestion as an answer the reviewer can change."""
    return {page_key: place for page_key, place in case.suggested_answer.items()}


def _entry_labels() -> list[str]:
    """Return entry labels A to Z, then numbers for larger groups."""
    return [chr(code) for code in range(ord("A"), ord("Z") + 1)] + [
        str(number) for number in range(27, 1000)
    ]


def _show_value(value: object) -> str:
    """Write a detail for the page: an em dash for a missing value."""
    return "—" if value in (None, "") else str(value)
