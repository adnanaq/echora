from datetime import UTC, datetime

import pytest
from review_page.case_views import (
    build_case_rows,
    build_case_view,
    find_original_answer,
    resolve_shown_answer,
)
from review_page.models import (
    CaseEditState,
    DetailColumn,
    PageLink,
    RunSummary,
    SavedChange,
)
from work_matching import (
    KnownWorkEntriesFromSeed,
    PreviousAnswer,
    ReviewCase,
    ReviewCaseKind,
    TargetEntrySuggestion,
    WorkEntry,
    build_page_address,
    combine_group_entries,
    decide_doubled_providers,
    describe_answer_word,
    resolve_page_key,
)
from work_matching.answer_words import LATER_WORK_MEANING

NAMI_ANSWER = {"anilist:anime:102289": "skip"}

ATASHINCHI_ANSWER = {
    "mal:anime:63013": "work 1",
    "anilist:anime:203146": "work 1",
    "anidb:anime:19895": "work 1",
}


def _build_case_view(case, **options):
    return build_case_view(
        case,
        original_answer=options.get("original_answer"),
        unsaved_answer=options.get("unsaved_answer"),
        is_rejected=options.get("is_rejected", False),
        is_run_going=options.get("is_run_going", False),
        show_only_differences=options.get("show_only_differences", False),
        invalid_named_page_key_reasons=options.get(
            "invalid_named_page_key_reasons", {}
        ),
    )


def _build_run_summary(**fields) -> RunSummary:
    return RunSummary(
        seed_file="assets/seed_data/seed.json",
        started_at=datetime(2026, 11, 1, 13, 58, 37, tzinfo=UTC),
        finished_at=datetime(2026, 11, 1, 13, 58, 39, tzinfo=UTC),
        correction_counts={},
        corrected_entry_counts={},
        linked_page_counts={},
        rejected_answer_case_names=fields.get("rejected_answer_case_names", ()),
        conflicting_linked_page_keys=(),
        unused_answer_case_names=fields.get("unused_answer_case_names", ()),
        case_names=(),
        new_case_names=fields.get("new_case_names", ()),
        original_answers=fields.get("original_answers", {}),
    )


def _build_mal_only_case(*mal_numbers: int) -> ReviewCase:
    entries = [
        WorkEntry(
            title=f"Work {number}",
            type="TV",
            year=2020,
            addresses_by_page_key={
                f"mal:anime:{number}": f"https://myanimelist.net/anime/{number}"
            },
        )
        for number in mal_numbers
    ]
    decisions = decide_doubled_providers(
        combine_group_entries(entries), [], KnownWorkEntriesFromSeed([])
    )
    return ReviewCase.build_entry_group_case(entries, decisions, [], [])


def test_build_case_view_split_work_rows_are_placeable_pages(review_cases) -> None:
    case_view = _build_case_view(review_cases["atashinchi_next"])
    assert [row.page_key for row in case_view.page_rows] == [
        "anilist:anime:203146",
        "mal:anime:63013",
        "anidb:anime:19895",
    ]


def test_build_case_view_searched_page_shown_as_evidence_only(review_cases) -> None:
    case_view = _build_case_view(review_cases["atashinchi_next"])
    assert [page.address for page in case_view.evidence_pages] == [
        "https://animeschedule.net/anime/atashin-chi-next-2nd-season"
    ]


def test_build_case_view_all_columns_shown_without_switch(review_cases) -> None:
    assert _build_case_view(review_cases["atashinchi_next"]).shown_columns == tuple(
        DetailColumn
    )


def test_build_case_view_show_only_differences_hides_shared_details(
    review_cases,
) -> None:
    case_view = _build_case_view(
        review_cases["atashinchi_next"], show_only_differences=True
    )
    assert case_view.shown_columns == (
        DetailColumn.TITLES,
        DetailColumn.AIRED_TO,
        DetailColumn.LINKS,
    )


def test_build_case_view_unanswered_case_starts_on_suggestion(review_cases) -> None:
    case_view = _build_case_view(review_cases["atashinchi_next"])
    assert (case_view.chosen_answer, case_view.is_suggestion_shown) == (
        dict(review_cases["atashinchi_next"].suggested_answer),
        True,
    )


def test_build_case_view_untouched_suggestion_marks_no_row_changed(
    review_cases,
) -> None:
    case_view = _build_case_view(review_cases["atashinchi_next"])
    assert [row.page_key for row in case_view.page_rows if row.is_changed] == []


def test_build_case_view_unsaved_edit_shown_and_marked(review_cases) -> None:
    edit = {**ATASHINCHI_ANSWER, "anidb:anime:19895": "work 2"}
    case_view = _build_case_view(review_cases["atashinchi_next"], unsaved_answer=edit)
    changed_page_keys = {row.page_key for row in case_view.page_rows if row.is_changed}
    assert (case_view.chosen_answer, case_view.edit_state, changed_page_keys) == (
        edit,
        CaseEditState.EDITED_NOT_SAVED,
        {"anidb:anime:19895"},
    )


def test_build_case_view_answered_case_marks_rows_differing_from_original(
    review_cases,
) -> None:
    original_answer = {**ATASHINCHI_ANSWER, "mal:anime:63013": "work 2"}
    case_view = _build_case_view(
        review_cases["atashinchi_next"],
        original_answer=original_answer,
        unsaved_answer=ATASHINCHI_ANSWER,
    )
    assert [row.page_key for row in case_view.page_rows if row.is_changed] == [
        "mal:anime:63013"
    ]


def test_build_case_view_rows_name_entry_and_reference_page(review_cases) -> None:
    case_view = _build_case_view(review_cases["atashinchi_next"])
    assert [
        (row.page_key, row.entry_label, row.is_reference_page)
        for row in case_view.page_rows
    ] == [
        ("anilist:anime:203146", "A", False),
        ("mal:anime:63013", "A", True),
        ("anidb:anime:19895", "B", True),
    ]


def test_build_case_view_rows_offer_target_entries_lacking_page_provider(
    review_cases,
) -> None:
    case = review_cases["gochuumon_movie"].model_copy(
        update={
            "target_entry_suggestions": (
                TargetEntrySuggestion(
                    entry_reference_page="mal:anime:21273",
                    title="Gochuumon wa Usagi desu ka?",
                    providers=("anidb", "mal"),
                ),
                TargetEntrySuggestion(
                    entry_reference_page="mal:anime:29787",
                    title="Gochuumon wa Usagi desu ka??",
                    providers=("mal",),
                ),
            )
        }
    )
    options_by_page_key = {
        row.page_key: [entry.entry_reference_page for entry in row.target_entry_options]
        for row in _build_case_view(case).page_rows
    }
    assert {
        page_key: options_by_page_key[page_key]
        for page_key in ("anidb:anime:19188", "anidb:anime:19792", "mal:anime:61115")
    } == {
        "anidb:anime:19188": ["mal:anime:29787"],
        "anidb:anime:19792": ["mal:anime:29787"],
        "mal:anime:61115": [],
    }


def test_build_case_view_reasons_read_as_sentences(review_cases) -> None:
    case_view = _build_case_view(review_cases["gochuumon_movie"])
    assert len(case_view.reason_sentences) == 2
    assert "PAGES_OF_OTHER_WORKS" not in case_view.reason_sentences


def test_build_case_view_linked_page_offers_answers(review_cases) -> None:
    case_view = _build_case_view(review_cases["nami"])
    assert [option.value for option in case_view.linked_page_options] == [
        "add",
        "add as partial",
        "skip",
    ]


def test_build_case_view_linked_page_compares_entry_and_page(review_cases) -> None:
    case_view = _build_case_view(review_cases["nami"])
    differing_rows = {
        row.label: (row.entry_value, row.page_value)
        for row in case_view.comparison_rows
        if row.is_different
    }
    assert differing_rows["Type"] == ("MOVIE", "ONA")
    assert "Year" not in differing_rows


def test_build_case_view_linked_show_only_differences_keeps_differing_rows(
    review_cases,
) -> None:
    case_view = _build_case_view(review_cases["nami"], show_only_differences=True)
    assert all(row.is_different for row in case_view.comparison_rows)
    assert "Type" in [row.label for row in case_view.comparison_rows]


def test_build_case_view_run_going_disables_buttons(review_cases) -> None:
    case_view = _build_case_view(review_cases["atashinchi_next"], is_run_going=True)
    assert not case_view.buttons.is_accept_enabled


def test_find_original_answer_no_run_summary_uses_saved_answer(review_cases) -> None:
    case = review_cases["nami"].model_copy(update={"answer": NAMI_ANSWER})
    assert find_original_answer(case, None) == NAMI_ANSWER


def test_find_original_answer_run_summary_without_answer_gives_none(
    review_cases,
) -> None:
    case = review_cases["nami"].model_copy(update={"answer": NAMI_ANSWER})
    assert find_original_answer(case, _build_run_summary()) is None


def test_find_original_answer_run_summary_gives_run_answer(review_cases) -> None:
    case = review_cases["nami"]
    run_summary = _build_run_summary(original_answers={case.case_name: NAMI_ANSWER})
    assert find_original_answer(case, run_summary) == NAMI_ANSWER


def test_build_case_rows_reads_titles_kinds_and_new_cases(review_cases) -> None:
    cases = [review_cases["nami"], review_cases["atashinchi_next"]]
    rows = build_case_rows(
        cases, _build_run_summary(new_case_names=(review_cases["nami"].case_name,))
    )
    assert [(row.title, row.kind, row.is_new, row.saved_change) for row in rows] == [
        ("Nami", ReviewCaseKind.LINKED_PAGE, True, SavedChange.UNCHANGED),
        (
            "Atashin'chi Next 2nd Season",
            ReviewCaseKind.SPLIT_WORK,
            False,
            SavedChange.UNCHANGED,
        ),
    ]


@pytest.mark.parametrize(
    ("original_answers", "rejected_answer_case_names", "is_answer_applied"),
    [
        ({"nami": NAMI_ANSWER}, (), True),
        ({}, (), False),
        ({"nami": NAMI_ANSWER}, ("nami",), False),
        (None, (), False),
    ],
)
def test_build_case_rows_answer_applied_only_when_last_run_used_answer(
    review_cases, original_answers, rejected_answer_case_names, is_answer_applied
) -> None:
    case = review_cases["nami"].model_copy(update={"answer": NAMI_ANSWER})
    case_name_by_label = {"nami": case.case_name}
    run_summary = (
        None
        if original_answers is None
        else _build_run_summary(
            original_answers={
                case_name_by_label[label]: answer
                for label, answer in original_answers.items()
            },
            rejected_answer_case_names=tuple(
                case_name_by_label[label] for label in rejected_answer_case_names
            ),
        )
    )
    [row] = build_case_rows([case], run_summary)
    assert row.is_answer_applied is is_answer_applied


def test_build_case_view_unplaced_rows_name_mal_page_holding_each_closed_work() -> None:
    case_view = _build_case_view(
        _build_mal_only_case(1, 2, 3),
        unsaved_answer={"mal:anime:1": "work 1", "mal:anime:2": "work 2"},
    )
    closed_works = {
        row.page_key: row.pages_by_closed_work for row in case_view.page_rows
    }
    assert closed_works == {
        "mal:anime:1": {},
        "mal:anime:2": {},
        "mal:anime:3": {
            "work 1": PageLink("mal:anime:1", "https://myanimelist.net/anime/1"),
            "work 2": PageLink("mal:anime:2", "https://myanimelist.net/anime/2"),
        },
    }


def test_build_case_view_rows_explain_choices_other_pages_disable() -> None:
    entries = [
        WorkEntry(
            title=title,
            type="TV",
            year=2020,
            addresses_by_page_key={resolve_page_key(address): address},
        )
        for title, address in (
            ("First", "https://myanimelist.net/anime/1"),
            ("Second", "https://myanimelist.net/anime/2"),
        )
    ]
    decisions = decide_doubled_providers(
        combine_group_entries(entries), [], KnownWorkEntriesFromSeed([])
    )
    case = ReviewCase.build_entry_group_case(entries, decisions, [], [])
    case_view = _build_case_view(case, unsaved_answer={"mal:anime:1": "work 1"})
    closed_works = {
        row.page_key: row.pages_by_closed_work for row in case_view.page_rows
    }
    assert closed_works["mal:anime:2"] == {
        "work 1": PageLink("mal:anime:1", "https://myanimelist.net/anime/1")
    }


def test_build_case_view_rows_leave_lone_page_main_to_tooltip(review_cases) -> None:
    case_view = _build_case_view(review_cases["atashinchi_next"])
    assert all(row.pages_by_closed_work == {} for row in case_view.page_rows)


def test_build_case_view_title_matches_list_title(review_cases) -> None:
    case = review_cases["atashinchi_next"]
    [row] = build_case_rows([case], None)
    assert _build_case_view(case).title == row.title == "Atashin'chi Next 2nd Season"


def test_build_case_view_rejected_case_marks_required_pages_without_place(
    review_cases,
) -> None:
    edit = {"anidb:anime:19895": "work 1", "anilist:anime:203146": "work 1"}
    case_view = _build_case_view(
        review_cases["atashinchi_next"], unsaved_answer=edit, is_rejected=True
    )
    assert [row.page_key for row in case_view.page_rows if row.is_place_missing] == [
        "mal:anime:63013"
    ]


def test_build_case_view_unrejected_case_marks_no_missing_place(review_cases) -> None:
    edit = {"anidb:anime:19895": "work 1", "anilist:anime:203146": "work 1"}
    case_view = _build_case_view(review_cases["atashinchi_next"], unsaved_answer=edit)
    assert not any(row.is_place_missing for row in case_view.page_rows)


def test_build_case_view_rows_link_linked_page_keys_to_addresses(review_cases) -> None:
    case_view = _build_case_view(review_cases["atashinchi_next"])
    rows = {row.page_key: row for row in case_view.page_rows}
    assert rows["anidb:anime:19895"].linked_pages == (
        PageLink("ann:anime:31487", build_page_address("ann:anime:31487")),
        PageLink("mal:anime:63013", "https://myanimelist.net/anime/63013"),
    )


def test_build_case_view_linked_page_links_entry_and_linked_page(review_cases) -> None:
    case_view = _build_case_view(review_cases["nami"])
    assert (case_view.linking_entry_link, case_view.linked_page_link) == (
        PageLink("kitsu:anime:10561", "https://kitsu.app/anime/10561"),
        PageLink("anilist:anime:102289", "https://anilist.co/anime/102289"),
    )


def test_build_case_view_entry_group_has_no_linked_page_links(review_cases) -> None:
    case_view = _build_case_view(review_cases["atashinchi_next"])
    assert (case_view.linking_entry_link, case_view.linked_page_link) == (None, None)


def test_build_case_rows_marks_answers_last_run_left_unused(review_cases) -> None:
    cases = [review_cases["nami"], review_cases["atashinchi_next"]]
    rows = build_case_rows(
        cases,
        _build_run_summary(
            unused_answer_case_names=(review_cases["atashinchi_next"].case_name,)
        ),
    )
    assert [row.is_answer_unused for row in rows] == [False, True]


def test_build_case_rows_marks_open_cases_with_previous_answer(review_cases) -> None:
    previous_answer = PreviousAnswer(
        case_name="split work: mal:anime:1",
        answer={"mal:anime:1": "work 1"},
        is_applied=True,
    )
    cases = [
        review_cases["nami"].model_copy(
            update={"answer": None, "previous_answer": previous_answer}
        ),
        review_cases["atashinchi_next"].model_copy(
            update={
                "answer": {"mal:anime:63013": "work 1"},
                "previous_answer": previous_answer,
            }
        ),
        review_cases["gochuumon_movie"],
    ]
    assert [row.has_previous_answer for row in build_case_rows(cases, None)] == [
        True,
        False,
        False,
    ]


def test_build_case_view_group_case_explains_every_choice(review_cases) -> None:
    meanings = dict(_build_case_view(review_cases["atashinchi_next"]).choice_meanings)
    assert (
        meanings["work 1"],
        meanings["work 2, work 3, …"],
        meanings["dropped"],
    ) == (
        describe_answer_word("work 1"),
        LATER_WORK_MEANING,
        describe_answer_word("dropped"),
    )


def test_build_case_view_linked_page_case_has_no_placement_legend(
    review_cases,
) -> None:
    assert _build_case_view(review_cases["nami"]).choice_meanings == ()


def test_resolve_shown_answer_unsaved_edit_wins(review_cases) -> None:
    case = review_cases["atashinchi_next"].model_copy(
        update={"answer": ATASHINCHI_ANSWER}
    )
    edit = {**ATASHINCHI_ANSWER, "anidb:anime:19895": "work 2"}
    assert resolve_shown_answer(case, edit) == edit


def test_resolve_shown_answer_saved_answer_wins_over_suggestion(
    review_cases,
) -> None:
    case = review_cases["atashinchi_next"].model_copy(
        update={"answer": {**ATASHINCHI_ANSWER, "anidb:anime:19895": "work 2"}}
    )
    assert resolve_shown_answer(case, None) == case.answer


def test_resolve_shown_answer_unanswered_case_returns_suggestion(
    review_cases,
) -> None:
    case = review_cases["atashinchi_next"]
    assert resolve_shown_answer(case, None) == dict(case.suggested_answer)
