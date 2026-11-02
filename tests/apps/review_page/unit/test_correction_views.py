from review_page.correction_views import (
    ANSWER_SOURCE,
    RULES_SOURCE,
    UNCHANGED,
    build_corrected_entry_rows,
    build_correction_panel,
    count_corrected_entry_rows,
    filter_corrected_entry_rows,
)
from review_page.models import CorrectionFilter
from shared_proto.v1 import review_cases_pb2

RENAMED_ENTRIES = [
    "mal:anime:2090",
    "mal:anime:40815",
    "mal:anime:42429",
    "mal:anime:61202",
    "mal:anime:61418",
    "mal:anime:62193",
]


def _rows(correction_messages, checked_entry_messages, cases=()):
    return build_corrected_entry_rows(
        correction_messages, checked_entry_messages, list(cases)
    )


def _gochuumon_correction() -> review_cases_pb2.SeedCorrection:
    return review_cases_pb2.SeedCorrection(
        entry_reference_page="mal:anime:61115",
        page_key="anidb:anime:19188",
        address="https://anidb.net/anime/19188",
        placement_word="other work",
        new_entry_key="mal:anime:61115 other work",
        reason="review answer",
        change="pages moved to a new entry",
        reason_sentence="a reviewer's answer placed the page",
        is_from_answer=True,
    )


def test_build_corrected_entry_rows_one_row_per_entry_in_file_order(
    correction_messages, checked_entry_messages
) -> None:
    reference_pages = list(
        dict.fromkeys(
            correction.entry_reference_page for correction in correction_messages
        )
    )
    rows = _rows(correction_messages, checked_entry_messages)
    assert [row.reference_page for row in rows] == reference_pages


def test_build_corrected_entry_rows_titles_from_checked_entries(
    correction_messages, checked_entry_messages
) -> None:
    title_by_reference_page = {
        entry.reference_page: entry.title for entry in checked_entry_messages
    }
    rows = _rows(correction_messages, checked_entry_messages)
    assert all(row.title == title_by_reference_page[row.reference_page] for row in rows)


def test_build_corrected_entry_rows_unknown_entry_titled_from_case(
    review_cases,
) -> None:
    [row] = _rows([_gochuumon_correction()], [], [review_cases["gochuumon_movie"]])
    assert row.title == "Gochuumon wa Usagi desu ka? Movie: We Are Family!"


def test_build_corrected_entry_rows_untitled_entry_shows_reference_page() -> None:
    [row] = _rows([_gochuumon_correction()], [])
    assert row.title == "mal:anime:61115"


def test_build_corrected_entry_rows_counts_changed_pages_and_answer(
    correction_messages, checked_entry_messages
) -> None:
    rows = {
        row.reference_page: row
        for row in _rows(correction_messages, checked_entry_messages)
    }
    answered = rows["mal:anime:57822"]
    assert (
        answered.page_count,
        answered.changed_page_count,
        answered.has_answer_correction,
        rows["mal:anime:2090"].has_answer_correction,
    ) == (13, 7, True, False)


def test_filter_corrected_entry_rows_change_keeps_entries_with_change(
    correction_messages, checked_entry_messages
) -> None:
    rows = _rows(correction_messages, checked_entry_messages)
    kept = filter_corrected_entry_rows(
        rows, CorrectionFilter(changes=frozenset({"old addresses dropped"}))
    )
    assert [row.reference_page for row in kept] == RENAMED_ENTRIES


def test_filter_corrected_entry_rows_answer_source_keeps_answered_entry(
    correction_messages, checked_entry_messages
) -> None:
    rows = _rows(correction_messages, checked_entry_messages)
    kept = filter_corrected_entry_rows(
        rows, CorrectionFilter(sources=frozenset({ANSWER_SOURCE}))
    )
    assert [row.reference_page for row in kept] == ["mal:anime:57822"]


def test_filter_corrected_entry_rows_groups_narrow_each_other(
    correction_messages, checked_entry_messages
) -> None:
    rows = _rows(correction_messages, checked_entry_messages)
    kept = filter_corrected_entry_rows(
        rows,
        CorrectionFilter(
            changes=frozenset({"old addresses dropped"}),
            sources=frozenset({ANSWER_SOURCE}),
        ),
    )
    assert kept == []


def test_filter_corrected_entry_rows_search_matches_page_key_and_title(
    correction_messages, checked_entry_messages
) -> None:
    rows = _rows(correction_messages, checked_entry_messages)
    by_page_key = filter_corrected_entry_rows(
        rows, CorrectionFilter(search_text="kitsu:anime:50921")
    )
    by_title = filter_corrected_entry_rows(rows, CorrectionFilter(search_text="orguss"))
    assert (
        [row.reference_page for row in by_page_key],
        [row.reference_page for row in by_title],
    ) == (["mal:anime:39476"], ["mal:anime:2090"])


def test_count_corrected_entry_rows_counts_entries_per_chip(
    correction_messages, checked_entry_messages
) -> None:
    counts = count_corrected_entry_rows(
        _rows(correction_messages, checked_entry_messages)
    )
    assert (
        counts["pages moved to a seed entry"],
        counts["old addresses dropped"],
        counts[ANSWER_SOURCE],
        counts[RULES_SOURCE],
        counts[UNCHANGED],
    ) == (4, 6, 1, 10, 11)


def test_build_correction_panel_moved_page_links_target_entry(
    correction_messages, checked_entry_messages
) -> None:
    panel = build_correction_panel(
        "mal:anime:37196", correction_messages, checked_entry_messages, []
    )
    assert panel is not None
    moved = {row.page_key: row for row in panel.rows}["anilist:anime:199801"]
    assert (
        moved.change_label,
        moved.place,
        moved.destination_label,
        moved.destination_address,
    ) == (
        "Pages moved to a seed entry",
        "other work",
        "seed entry mal:anime:62648",
        "https://myanimelist.net/anime/62648",
    )


def test_build_correction_panel_renamed_page_links_current_address(
    correction_messages, checked_entry_messages
) -> None:
    panel = build_correction_panel(
        "mal:anime:2090", correction_messages, checked_entry_messages, []
    )
    assert panel is not None
    renamed = {row.page_key: row for row in panel.rows}["animeplanet:anime:orguss-02"]
    assert (renamed.destination_label, renamed.destination_address) == (
        "current address animeplanet:anime:super-dimension-century-orguss-02",
        "https://anime-planet.com/anime/super-dimension-century-orguss-02",
    )


def test_build_correction_panel_answer_rows_show_mark_and_source(
    correction_messages, checked_entry_messages
) -> None:
    panel = build_correction_panel(
        "mal:anime:57822", correction_messages, checked_entry_messages, []
    )
    assert panel is not None
    rows = {row.page_key: row for row in panel.rows}
    assert (
        rows["animeplanet:anime:atashinchi-next"].place,
        rows["ann:anime:31487"].place,
        rows["mal:anime:57822"].change_label,
        rows["mal:anime:57822"].destination_label,
        all(row.is_from_answer for row in panel.rows),
    ) == ("work 1, main", "work 1, part", "Unchanged", "stays in this entry", True)


def test_build_correction_panel_lists_cases_holding_entry(review_cases) -> None:
    panel = build_correction_panel(
        "mal:anime:61115",
        [_gochuumon_correction()],
        [],
        list(review_cases.values()),
    )
    assert panel is not None
    assert [case_name for case_name, _ in panel.case_links] == [
        review_cases["gochuumon_movie"].case_name
    ]


def test_build_correction_panel_unknown_entry_gives_none(
    correction_messages, checked_entry_messages
) -> None:
    assert (
        build_correction_panel(
            "mal:anime:1", correction_messages, checked_entry_messages, []
        )
        is None
    )
