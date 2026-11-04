from review_page.case_views import (
    build_case_rows,
    build_case_view,
    find_original_answer,
)
from review_page.models import CaseEditState, DetailColumn, PageLink, SavedChange
from work_matching import (
    PART_MARK,
    KnownWorkEntriesFromSeed,
    PlacementRule,
    ReviewCase,
    ReviewCaseKind,
    TargetEntrySuggestion,
    WorkEntry,
    build_page_address,
    combine_group_entries,
    decide_doubled_providers,
    describe_answer_word,
    get_page_key,
)
from work_matching.answer_words import LATER_WORK_MEANING

ATASHINCHI_ANSWER = {
    "mal:anime:63013": "work 1",
    "anilist:anime:203146": "work 1",
    "anidb:anime:19895": "work 1",
}


def _view(case, **options):
    return build_case_view(
        case,
        original_answer=options.get("original_answer"),
        unsaved_answer=options.get("unsaved_answer"),
        is_rejected=options.get("is_rejected", False),
        is_run_going=options.get("is_run_going", False),
        show_only_differences=options.get("show_only_differences", False),
    )


def test_build_case_view_split_work_rows_are_placeable_pages(review_cases) -> None:
    view = _view(review_cases["atashinchi_next"])
    assert [row.page_key for row in view.page_rows] == [
        "anilist:anime:203146",
        "mal:anime:63013",
        "anidb:anime:19895",
    ]


def test_build_case_view_searched_page_shown_as_evidence_only(review_cases) -> None:
    view = _view(review_cases["atashinchi_next"])
    assert [page.address for page in view.evidence_pages] == [
        "https://animeschedule.net/anime/atashin-chi-next-2nd-season"
    ]


def test_build_case_view_all_columns_shown_without_switch(review_cases) -> None:
    assert _view(review_cases["atashinchi_next"]).shown_columns == tuple(DetailColumn)


def test_build_case_view_show_only_differences_hides_shared_details(
    review_cases,
) -> None:
    view = _view(review_cases["atashinchi_next"], show_only_differences=True)
    assert view.shown_columns == (
        DetailColumn.TITLES,
        DetailColumn.AIRED_TO,
        DetailColumn.LINKS,
    )


def test_build_case_view_unanswered_case_starts_on_suggestion(review_cases) -> None:
    view = _view(review_cases["atashinchi_next"])
    assert (view.chosen_answer, view.is_suggestion_shown) == (
        dict(review_cases["atashinchi_next"].suggested_answer),
        True,
    )


def test_build_case_view_untouched_suggestion_marks_no_row_changed(
    review_cases,
) -> None:
    view = _view(review_cases["atashinchi_next"])
    assert [row.page_key for row in view.page_rows if row.is_changed] == []


def test_build_case_view_unsaved_edit_shown_and_marked(review_cases) -> None:
    edit = {**ATASHINCHI_ANSWER, "anidb:anime:19895": "work 2"}
    view = _view(review_cases["atashinchi_next"], unsaved_answer=edit)
    changed_page_keys = {row.page_key for row in view.page_rows if row.is_changed}
    assert (view.chosen_answer, view.edit_state, changed_page_keys) == (
        edit,
        CaseEditState.EDITED_NOT_SAVED,
        {"anidb:anime:19895"},
    )


def test_build_case_view_answered_case_marks_rows_differing_from_original(
    review_cases,
) -> None:
    original_answer = {**ATASHINCHI_ANSWER, "mal:anime:63013": "work 2"}
    view = _view(
        review_cases["atashinchi_next"],
        original_answer=original_answer,
        unsaved_answer=ATASHINCHI_ANSWER,
    )
    assert [row.page_key for row in view.page_rows if row.is_changed] == [
        "mal:anime:63013"
    ]


def test_build_case_view_rows_name_entry_and_reference_page(review_cases) -> None:
    view = _view(review_cases["atashinchi_next"])
    assert [
        (row.page_key, row.entry_label, row.is_reference_page) for row in view.page_rows
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
                    reference_page="mal:anime:21273",
                    title="Gochuumon wa Usagi desu ka?",
                    providers=("anidb", "mal"),
                ),
                TargetEntrySuggestion(
                    reference_page="mal:anime:29787",
                    title="Gochuumon wa Usagi desu ka??",
                    providers=("mal",),
                ),
            )
        }
    )
    options_by_page_key = {
        row.page_key: [entry.reference_page for entry in row.target_entry_options]
        for row in _view(case).page_rows
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
    view = _view(review_cases["gochuumon_movie"])
    assert len(view.reason_sentences) == 2
    assert "PAGES_OF_OTHER_WORKS" not in view.reason_sentences


def test_build_case_view_linked_page_offers_answers(review_cases) -> None:
    view = _view(review_cases["nami"])
    assert [option.value for option in view.linked_page_options] == [
        "add",
        "add as part",
        "skip",
    ]


def test_build_case_view_linked_page_compares_entry_and_page(review_cases) -> None:
    view = _view(review_cases["nami"])
    differing_rows = {
        row.label: (row.entry_value, row.page_value)
        for row in view.comparison_rows
        if row.is_different
    }
    assert differing_rows["Type"] == ("MOVIE", "ONA")
    assert "Year" not in differing_rows


def test_build_case_view_linked_show_only_differences_keeps_differing_rows(
    review_cases,
) -> None:
    view = _view(review_cases["nami"], show_only_differences=True)
    assert all(row.is_different for row in view.comparison_rows)
    assert "Type" in [row.label for row in view.comparison_rows]


def test_build_case_view_run_going_disables_buttons(review_cases) -> None:
    view = _view(review_cases["atashinchi_next"], is_run_going=True)
    assert not view.buttons.is_accept_enabled


def test_find_original_answer_no_run_result_uses_saved_answer() -> None:
    assert find_original_answer("a", {"x": "work 1"}, None) == {"x": "work 1"}


def test_find_original_answer_run_result_without_answer_gives_none() -> None:
    assert find_original_answer("a", {"x": "work 1"}, {}) is None


def test_build_case_rows_reads_titles_kinds_and_new_cases(review_cases) -> None:
    cases = [review_cases["nami"], review_cases["atashinchi_next"]]
    rows = build_case_rows(
        cases,
        original_answers={},
        new_case_names=frozenset({review_cases["nami"].case_name}),
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


def _mal_only_case(*mal_numbers: int) -> ReviewCase:
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
    return ReviewCase.from_entry_group(entries, decisions, [], [])


def test_build_case_view_rows_join_works_one_rule_disables_into_one_note() -> None:
    view = _view(
        _mal_only_case(1, 2, 3),
        unsaved_answer={"mal:anime:1": "work 1", "mal:anime:2": "work 2"},
    )
    notes = {row.page_key: row.disabled_choice_notes for row in view.page_rows}
    assert notes["mal:anime:3"] == (
        "work 1 and work 2 are not offered: "
        f"{PlacementRule.WORK_WITH_TWO_MAL_PAGES.value}",
    )


def test_build_case_view_rows_explain_choices_other_pages_disable() -> None:
    entries = [
        WorkEntry(
            title=title,
            type="TV",
            year=2020,
            addresses_by_page_key={get_page_key(address): address},
        )
        for title, address in (
            ("First", "https://myanimelist.net/anime/1"),
            ("Second", "https://myanimelist.net/anime/2"),
        )
    ]
    decisions = decide_doubled_providers(
        combine_group_entries(entries), [], KnownWorkEntriesFromSeed([])
    )
    case = ReviewCase.from_entry_group(entries, decisions, [], [])
    view = _view(case, unsaved_answer={"mal:anime:1": "work 1"})
    notes = {row.page_key: row.disabled_choice_notes for row in view.page_rows}
    assert notes["mal:anime:2"] == (
        f"work 1 is not offered: {PlacementRule.WORK_WITH_TWO_MAL_PAGES.value}",
    )


def test_build_case_view_rows_leave_lone_page_main_to_tooltip(review_cases) -> None:
    view = _view(review_cases["atashinchi_next"])
    assert all(row.disabled_choice_notes == () for row in view.page_rows)


def test_build_case_view_title_matches_list_title(review_cases) -> None:
    case = review_cases["atashinchi_next"]
    [row] = build_case_rows([case], original_answers=None, new_case_names=())
    assert _view(case).title == row.title == "Atashin'chi Next 2nd Season"


def test_build_case_view_rejected_case_marks_required_pages_without_place(
    review_cases,
) -> None:
    edit = {"anidb:anime:19895": "work 1", "anilist:anime:203146": "work 1"}
    view = _view(review_cases["atashinchi_next"], unsaved_answer=edit, is_rejected=True)
    assert [row.page_key for row in view.page_rows if row.is_place_missing] == [
        "mal:anime:63013"
    ]


def test_build_case_view_unrejected_case_marks_no_missing_place(review_cases) -> None:
    edit = {"anidb:anime:19895": "work 1", "anilist:anime:203146": "work 1"}
    view = _view(review_cases["atashinchi_next"], unsaved_answer=edit)
    assert not any(row.is_place_missing for row in view.page_rows)


def test_build_case_view_rows_link_linked_page_keys_to_addresses(review_cases) -> None:
    view = _view(review_cases["atashinchi_next"])
    rows = {row.page_key: row for row in view.page_rows}
    assert rows["anidb:anime:19895"].linked_pages == (
        PageLink("ann:anime:31487", build_page_address("ann:anime:31487")),
        PageLink("mal:anime:63013", "https://myanimelist.net/anime/63013"),
    )


def test_build_case_view_linked_page_gives_page_address(review_cases) -> None:
    assert _view(review_cases["nami"]).linked_page_address == (
        "https://anilist.co/anime/102289"
    )


def test_build_case_rows_marks_answers_last_run_left_unused(review_cases) -> None:
    cases = [review_cases["nami"], review_cases["atashinchi_next"]]
    rows = build_case_rows(
        cases,
        original_answers={},
        new_case_names=(),
        unused_answer_case_names={review_cases["atashinchi_next"].case_name},
    )
    assert [row.is_answer_unused for row in rows] == [False, True]


def test_build_case_view_group_case_explains_every_choice(review_cases) -> None:
    meanings = dict(_view(review_cases["atashinchi_next"]).choice_meanings)
    assert (
        meanings["work 1"],
        meanings["work 2, work 3, …"],
        meanings["part"],
    ) == (
        describe_answer_word("work 1"),
        LATER_WORK_MEANING,
        describe_answer_word(PART_MARK),
    )


def test_build_case_view_linked_page_case_has_no_placement_legend(
    review_cases,
) -> None:
    assert _view(review_cases["nami"]).choice_meanings == ()
