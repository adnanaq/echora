import json
import re

import pytest
from google.protobuf import json_format
from jinja2 import Template
from review_page.models import CaseEditState, CaseState, InvalidNamedPageKeyReason
from review_page.template_functions import build_element_id, build_static_url
from review_page.tooltips import Tooltip
from shared_proto.v1 import review_cases_pb2
from work_matching import (
    PreviousAnswer,
    ReviewCaseKind,
    TargetEntrySuggestion,
    describe_answer_word,
)

ATASHINCHI_ANSWER = {
    "mal:anime:63013": "work 1",
    "anilist:anime:203146": "work 1",
    "anidb:anime:19895": "work 1",
}
GOCHUUMON_DOUBLED_PAGES = ("anidb:anime:19188", "anidb:anime:19792")


def escape(text: str) -> str:
    return Template("{{ text }}", autoescape=True).render(text=text)


def _state(**fields) -> dict[str, str]:
    return {"browser_state": json.dumps(fields)}


def _events(response) -> dict:
    return json.loads(response.headers["HX-Trigger"])


def test_health_answers_ok(client) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_page_holds_every_region_and_script(client) -> None:
    page = client.get("/").text
    assert all(
        region in page
        for region in (
            'id="toolbar"',
            'id="case-list"',
            'id="case-panel"',
            'id="run-panel"',
            "htmx-2.0.11.min.js",
            "alpinejs-persist-3.17.5.min.js",
        )
    )


def test_page_offers_system_light_and_dark_themes(client) -> None:
    page = client.get("/").text
    theme_select = page[page.index('aria-label="Theme"') :]
    theme_select = theme_select[: theme_select.index("</select>")]
    assert (
        [
            value in theme_select
            for value in ('value="system"', 'value="light"', 'value="dark"')
        ],
        "setTheme($event.target.value)" in theme_select,
        page.index("applyReviewPageTheme(") < page.index("review_page.css"),
    ) == ([True, True, True], True, True)


def test_page_case_list_can_close_and_resize(client) -> None:
    page = client.get("/").text
    assert (
        'role="separator"' in page,
        'aria-label="Resize the case list"' in page,
        "toggleSidebar()" in page,
        "is-sidebar-closed" in page,
        "Hide the case list" in page and "Show the case list" in page,
        "<svg" in page,
        "Hide cases" in page,
    ) == (True, True, True, True, True, True, False)


def test_page_case_list_button_sits_beside_search_and_in_rail(client) -> None:
    page = client.get("/").text
    header = page[page.index("<header") : page.index("</header>")]
    list_top = page[page.index('class="list-panel-top"') : page.index('id="case-list"')]
    rail = page[page.index('class="layout-panel layout-rail"') :]
    rail = rail[: rail.index("</div>")]
    assert (
        "toggleSidebar()" in header,
        list_top.index("toggleSidebar()") < list_top.index('type="search"'),
        "toggleSidebar()" in rail,
        'aria-controls="case-list-panel"' in list_top
        and 'aria-controls="case-list-panel"' in rail,
    ) == (False, True, True, True)


def test_static_parts_stylesheet_served(client) -> None:
    assert client.get("/static/review_page.css").status_code == 200


def test_case_list_lists_every_case_with_counts(client, review_cases) -> None:
    case_list = client.post("/fragments/case-list", data=_state()).text
    assert "Nami" in case_list
    assert "Viewing 1–4 of 4" in case_list


def test_case_list_kind_filter_keeps_kind(client) -> None:
    case_list = client.post(
        "/fragments/case-list", data=_state(kinds=["linked page"])
    ).text
    assert ("Nami" in case_list, "Viewing 1–1 of 1" in case_list) == (True, True)


def test_case_list_malformed_state_lists_every_case(client) -> None:
    case_list = client.post(
        "/fragments/case-list", data={"browser_state": "{not json"}
    ).text
    assert "Viewing 1–4 of 4" in case_list


def test_case_panel_shows_open_case_pages_and_choices(client, review_cases) -> None:
    panel = client.post(
        "/fragments/case-panel",
        data=_state(open_case_name=review_cases["atashinchi_next"].case_name),
    ).text
    assert all(page_key in panel for page_key in ATASHINCHI_ANSWER)
    assert 'name="work"' in panel


def test_case_panel_show_only_differences_hides_shared_columns(
    client, review_cases
) -> None:
    case_name = review_cases["atashinchi_next"].case_name
    full_panel = client.post(
        "/fragments/case-panel", data=_state(open_case_name=case_name)
    ).text
    short_panel = client.post(
        "/fragments/case-panel",
        data=_state(open_case_name=case_name, show_only_differences=True),
    ).text
    assert ("<th>Eps</th>" in full_panel, "<th>Eps</th>" in short_panel) == (
        True,
        False,
    )


def _render_case_list_with_nami_answer(client, source, review_cases, *, is_run_done):
    case = review_cases["nami"].model_copy(
        update={"answer": {"anilist:anime:102289": "skip"}}
    )
    source.cases[case.case_name] = case
    source.run_result = json_format.ParseDict(
        {
            "started_at": "2026-11-01T00:00:00Z",
            "finished_at": "2026-11-01T00:00:02Z",
            "original_answers": {case.case_name: case.answer} if is_run_done else {},
        },
        review_cases_pb2.SeedCleaningRunResult(),
    )
    case_list = client.post("/fragments/case-list", data=_state()).text
    linked_page_chip = case_list[case_list.index('id="chip-kinds-linked-page"') :]
    return case_list, linked_page_chip[: linked_page_chip.index("</button>")]


def test_case_list_every_answer_applied_shows_icon_and_plain_kind_total(
    client, source, review_cases
) -> None:
    case_list, linked_page_chip = _render_case_list_with_nami_answer(
        client, source, review_cases, is_run_done=True
    )
    assert (
        case_list.count('class="control-icon tone-applied"'),
        '<span class="control-chip-count">(1)</span>' in linked_page_chip,
        Tooltip.RESOLVED_CASE_KIND_COUNT.value in linked_page_chip,
    ) == (1, True, True)


def test_case_list_answer_saved_before_run_shows_no_icon(
    client, source, review_cases
) -> None:
    case_list, linked_page_chip = _render_case_list_with_nami_answer(
        client, source, review_cases, is_run_done=False
    )
    assert (
        case_list.count('class="control-icon tone-applied"'),
        '<span class="control-chip-count">(0/1)</span>' in linked_page_chip,
        Tooltip.UNRESOLVED_CASE_KIND_COUNT.value in linked_page_chip,
        "saved, applies on Run" in case_list,
    ) == (0, True, True, True)


def test_page_search_row_holds_reset_filter_icon_buttons(client) -> None:
    page = client.get("/").text
    search_row = page[page.index('class="list-panel-top"') :]
    search_row = search_row[: search_row.index('id="case-list"')]
    case_list = client.post("/fragments/case-list", data=_state()).text
    assert (
        f'aria-label="{Tooltip.RESET_CASE_FILTERS.value}"' in search_row,
        f'aria-label="{Tooltip.RESET_CORRECTION_FILTERS.value}"' in search_row,
        "Reset filters" in case_list,
    ) == (True, True, False)


def test_case_panel_without_open_case_asks_to_choose(client) -> None:
    assert "Choose a case" in client.post("/fragments/case-panel", data=_state()).text


def test_choice_keeps_partial_works_and_reports_unsaved_answer(
    client, review_cases
) -> None:
    case_name = review_cases["gochuumon_movie"].case_name
    response = client.post(
        "/fragments/choice",
        data={
            **_state(
                open_case_name=case_name,
                edits={
                    case_name: {
                        "anidb:anime:19188": "work 1, partial in work 2",
                        "anidb:anime:19792": "work 2",
                    }
                },
            ),
            "page_key": "anidb:anime:19188",
            "work": "work 3",
        },
    )
    answer_changed = _events(response)["answerchanged"]
    assert answer_changed == {
        "case_name": case_name,
        "answer": {
            "anidb:anime:19188": "work 3, partial in work 2",
            "anidb:anime:19792": "work 2",
        },
        "is_unsaved": True,
        "saved_answer": None,
    }


def test_choice_back_to_suggestion_reports_no_unsaved_edit(
    client, review_cases
) -> None:
    case = review_cases["atashinchi_next"]
    changed_answer = {**case.suggested_answer, "anidb:anime:19895": "work 2"}
    response = client.post(
        "/fragments/choice",
        data={
            **_state(
                open_case_name=case.case_name,
                edits={case.case_name: changed_answer},
            ),
            "page_key": "anidb:anime:19895",
            "work": "work 1",
        },
    )
    answer_changed = _events(response)["answerchanged"]
    assert (
        answer_changed["answer"],
        answer_changed["is_unsaved"],
        "edited, not saved" in response.text,
    ) == (dict(case.suggested_answer), False, False)


@pytest.mark.parametrize(
    "named_entry",
    ["mal:anime:21273", "https://myanimelist.net/anime/21273", " mal:anime:21273 "],
)
def test_choice_other_work_named_entry_written_as_page_key(
    client, review_cases, named_entry
) -> None:
    case_name = review_cases["gochuumon_movie"].case_name
    response = client.post(
        "/fragments/choice",
        data={
            **_state(open_case_name=case_name),
            "page_key": "anidb:anime:19188",
            "work": "other work",
            "named_entry": named_entry,
        },
    )
    answer = _events(response)["answerchanged"]["answer"]
    assert answer["anidb:anime:19188"] == "other work mal:anime:21273"


def test_choice_other_work_unreadable_named_entry_kept_as_written(
    client, review_cases
) -> None:
    case_name = review_cases["gochuumon_movie"].case_name
    response = client.post(
        "/fragments/choice",
        data={
            **_state(open_case_name=case_name),
            "page_key": "anidb:anime:19188",
            "work": "other work",
            "named_entry": "Gochuumon",
        },
    )
    answer = _events(response)["answerchanged"]["answer"]
    assert answer["anidb:anime:19188"] == "other work Gochuumon"


GOCHUUMON_SERIES_SUGGESTION = TargetEntrySuggestion(
    entry_reference_page="mal:anime:21273",
    title="Gochuumon wa Usagi desu ka?",
    providers=("mal",),
)


def _render_other_work_panel(client, source, review_cases, place: str) -> str:
    case = review_cases["gochuumon_movie"].model_copy(
        update={"target_entry_suggestions": (GOCHUUMON_SERIES_SUGGESTION,)}
    )
    source.cases[case.case_name] = case
    return client.post(
        "/fragments/case-panel",
        data=_state(
            open_case_name=case.case_name,
            edits={case.case_name: {"anidb:anime:19188": place}},
        ),
    ).text


def _cut_named_entry_controls(panel: str) -> str:
    start = panel.index('class="control-picker')
    return panel[start : panel.index("</td>", start)]


def _read_trigger_text(controls: str) -> str:
    trigger_text = re.search(r'class="control-picker-text">([^<]*)<', controls)
    assert trigger_text is not None
    return trigger_text.group(1)


def test_case_panel_other_work_offers_new_entry_and_suggestions(
    client, source, review_cases
) -> None:
    controls = _cut_named_entry_controls(
        _render_other_work_panel(client, source, review_cases, "other work")
    )
    assert (
        _read_trigger_text(controls),
        re.findall(r'"named_entry": "([^"]*)"', controls),
        re.findall(r'class="control-picker-option-title">([^<]*)<', controls),
        re.search(
            r'aria-selected="true"\s+class="control-picker-option is-chosen"', controls
        )
        is not None,
        'style="display: none"' in controls,
        'class="page-key"' in controls,
    ) == (
        "New entry",
        ["mal:anime:21273"],
        ["New entry", "Gochuumon wa Usagi desu ka?"],
        True,
        True,
        False,
    )


def test_case_panel_other_work_suggested_entry_named_shows_its_title(
    client, source, review_cases
) -> None:
    controls = _cut_named_entry_controls(
        _render_other_work_panel(
            client, source, review_cases, "other work mal:anime:21273"
        )
    )
    assert (
        _read_trigger_text(controls),
        controls.count("is-chosen"),
        '<a class="page-key" href="https://myanimelist.net/anime/21273"' in controls,
    ) == ("Gochuumon wa Usagi desu ka?", 1, True)


def test_case_panel_other_work_typed_entry_named_shows_its_key(
    client, source, review_cases
) -> None:
    controls = _cut_named_entry_controls(
        _render_other_work_panel(
            client, source, review_cases, "other work mal:anime:29787"
        )
    )
    assert (
        _read_trigger_text(controls),
        "is-chosen" in controls,
        '<a class="page-key" href="https://myanimelist.net/anime/29787"' in controls,
    ) == ("Other entry", False, True)


@pytest.mark.parametrize("invalid_reason", list(InvalidNamedPageKeyReason))
def test_case_panel_other_work_invalid_named_key_marked_with_reason(
    client, source, review_cases, invalid_reason
) -> None:
    source.invalid_named_page_key_reasons = {"mal:anime:29787": invalid_reason}
    controls = _cut_named_entry_controls(
        _render_other_work_panel(
            client, source, review_cases, "other work mal:anime:29787"
        )
    )
    assert (
        'class="control-picker is-invalid"' in controls,
        'aria-invalid="true"' in controls,
        re.search(
            rf'class="control-error-line">.*?{invalid_reason.value}</p>', controls
        )
        is not None,
    ) == (True, True, True)


def test_case_panel_other_work_valid_named_key_not_marked(
    client, source, review_cases
) -> None:
    controls = _cut_named_entry_controls(
        _render_other_work_panel(
            client, source, review_cases, "other work mal:anime:29787"
        )
    )
    assert ("is-invalid" in controls, "control-error-line" in controls) == (
        False,
        False,
    )


def test_choice_linked_page_answer_reported(client, review_cases) -> None:
    case_name = review_cases["nami"].case_name
    response = client.post(
        "/fragments/choice",
        data={
            **_state(open_case_name=case_name),
            "page_key": "anilist:anime:102289",
            "linked_answer": "skip",
        },
    )
    assert _events(response)["answerchanged"]["answer"] == {
        "anilist:anime:102289": "skip"
    }


def test_answers_save_valid_edit_reports_saved(client, source, review_cases) -> None:
    case_name = review_cases["atashinchi_next"].case_name
    response = client.post(
        "/answers",
        data={
            **_state(edits={case_name: ATASHINCHI_ANSWER}),
            "action": "save",
            "case_name": case_name,
        },
    )
    assert (
        response.status_code,
        _events(response)["answerssaved"]["saved_cases"],
    ) == (
        204,
        [case_name],
    )
    assert source.cases[case_name].answer == ATASHINCHI_ANSWER


def test_answers_save_rule_breaking_edit_reports_rule(client, review_cases) -> None:
    case_name = review_cases["atashinchi_next"].case_name
    response = client.post(
        "/answers",
        data={
            **_state(edits={case_name: {"mal:anime:63013": "work 1"}}),
            "action": "save",
            "case_name": case_name,
        },
    )
    assert _events(response)["answerssaved"]["rejected"] == {
        case_name: ["every required page must be assigned to a work"]
    }


def test_answers_save_next_names_next_case_in_filtered_list(
    client, review_cases
) -> None:
    case_names = list(review_cases)
    first_case_name = review_cases[case_names[0]].case_name
    response = client.post(
        "/answers",
        data={
            **_state(edits={first_case_name: ATASHINCHI_ANSWER}),
            "action": "save_next",
            "case_name": first_case_name,
        },
    )
    assert (
        _events(response)["answerssaved"]["next_case_name"]
        == review_cases[case_names[1]].case_name
    )


def test_answers_save_next_rejected_save_stays(client, review_cases) -> None:
    case_name = review_cases["atashinchi_next"].case_name
    response = client.post(
        "/answers",
        data={
            **_state(edits={case_name: {"mal:anime:63013": "work 1"}}),
            "action": "save_next",
            "case_name": case_name,
        },
    )
    assert _events(response)["answerssaved"]["next_case_name"] is None


def test_answers_save_edit_after_other_tab_saved_reports_saved_elsewhere(
    client, source, review_cases
) -> None:
    case_name = review_cases["atashinchi_next"].case_name
    other_tab_answer = {**ATASHINCHI_ANSWER, "anidb:anime:19895": "work 2"}
    source.cases[case_name] = source.cases[case_name].model_copy(
        update={"answer": other_tab_answer}
    )
    response = client.post(
        "/answers",
        data={
            **_state(
                edits={case_name: ATASHINCHI_ANSWER},
                seen_saved_answers={case_name: None},
            ),
            "action": "save",
            "case_name": case_name,
        },
    )
    answers_saved = _events(response)["answerssaved"]
    assert (
        answers_saved["saved_cases"],
        answers_saved["cases_saved_elsewhere"],
    ) == (
        [],
        [case_name],
    )
    assert source.cases[case_name].answer == other_tab_answer


def test_answers_save_edit_with_unchanged_seen_answer_saves(
    client, source, review_cases
) -> None:
    case_name = review_cases["atashinchi_next"].case_name
    response = client.post(
        "/answers",
        data={
            **_state(
                edits={case_name: ATASHINCHI_ANSWER},
                seen_saved_answers={case_name: None},
            ),
            "action": "save",
            "case_name": case_name,
        },
    )
    assert _events(response)["answerssaved"]["saved_cases"] == [case_name]
    assert source.cases[case_name].answer == ATASHINCHI_ANSWER


def test_answers_save_while_other_save_changes_answer_answers_409(
    client, source, review_cases
) -> None:
    source.is_answer_changed_during_save = True
    case_name = review_cases["atashinchi_next"].case_name
    response = client.post(
        "/answers",
        data={
            **_state(edits={case_name: ATASHINCHI_ANSWER}),
            "action": "save",
            "case_name": case_name,
        },
    )
    assert response.status_code == 409
    assert "nothing was saved" in _events(response)["notice"]["text"]


def test_answers_accept_saves_suggestion(client, source, review_cases) -> None:
    case = review_cases["atashinchi_next"]
    client.post(
        "/answers",
        data={**_state(), "action": "accept", "case_name": case.case_name},
    )
    assert source.cases[case.case_name].answer == dict(case.suggested_answer)


def test_answers_reset_writes_original_back(client, source, review_cases) -> None:
    case_name = review_cases["atashinchi_next"].case_name
    source.cases[case_name] = source.cases[case_name].model_copy(
        update={"answer": ATASHINCHI_ANSWER}
    )
    client.post(
        "/answers", data={**_state(), "action": "reset", "case_name": case_name}
    )
    assert source.cases[case_name].answer == ATASHINCHI_ANSWER


def test_answers_save_all_saves_every_edit(client, source, review_cases) -> None:
    edits = {
        review_cases["atashinchi_next"].case_name: ATASHINCHI_ANSWER,
        review_cases["nami"].case_name: {"anilist:anime:102289": "skip"},
    }
    response = client.post(
        "/answers", data={**_state(edits=edits), "action": "save_all"}
    )
    assert sorted(_events(response)["answerssaved"]["saved_cases"]) == sorted(edits)


@pytest.mark.parametrize(
    ("source_flag", "flag_value"),
    [("is_locked", True), ("is_seed_file_readable", False)],
)
def test_answers_refused_by_source_answers_409_with_notice(
    client, source, review_cases, source_flag, flag_value
) -> None:
    setattr(source, source_flag, flag_value)
    response = client.post(
        "/answers",
        data={
            **_state(),
            "action": "save",
            "case_name": review_cases["atashinchi_next"].case_name,
        },
    )
    assert response.status_code == 409
    assert "notice" in _events(response)


def test_answers_after_run_drops_case_answers_409(client, source, review_cases) -> None:
    source.is_case_dropped_by_run = True
    response = client.post(
        "/answers",
        data={
            **_state(),
            "action": "save",
            "case_name": review_cases["atashinchi_next"].case_name,
        },
    )
    assert response.status_code == 409
    assert "nothing was saved" in _events(response)["notice"]["text"]


def test_run_started_reports_event(client, source) -> None:
    response = client.post("/run", data=_state())
    assert (response.status_code, _events(response), source.is_running) == (
        204,
        {"runstarted": True},
        True,
    )


def test_run_without_seed_file_answers_409(client, source) -> None:
    source.is_seed_file_known = False
    assert client.post("/run", data=_state()).status_code == 409


def test_toolbar_counts_unsaved_cases(client, review_cases) -> None:
    toolbar = client.post(
        "/fragments/toolbar",
        data=_state(
            edits={review_cases["nami"].case_name: {"anilist:anime:102289": "skip"}}
        ),
    ).text
    assert "Save all (1)" in toolbar


def test_run_panel_reports_run_state(client, source) -> None:
    source.is_running = True
    response = client.post("/fragments/run-panel", data=_state())
    assert _events(response)["runstate"]["is_running"] is True
    assert "Run going" in response.text


def test_page_links_static_files_with_content_hash(client) -> None:
    page = client.get("/").text
    assert build_static_url("review_page.css") in page
    assert "?v=" in build_static_url("review_page.css")


def test_static_file_served_at_hashed_address(client) -> None:
    assert client.get(build_static_url("review_page.js")).status_code == 200


def test_case_list_and_panel_mark_answer_last_run_left_unused(
    client, source, review_cases
) -> None:
    case_name = review_cases["nami"].case_name
    source.run_result = json_format.ParseDict(
        {
            "started_at": "2026-11-01T00:00:00Z",
            "finished_at": "2026-11-01T00:00:02Z",
            "unused_answer_case_names": [case_name],
        },
        review_cases_pb2.SeedCleaningRunResult(),
    )
    case_list = client.post("/fragments/case-list", data=_state()).text
    panel = client.post(
        "/fragments/case-panel", data=_state(open_case_name=case_name)
    ).text
    assert (case_list.count("Unused answer"), "Unused answer" in panel) == (
        1,
        True,
    )


def test_case_list_and_panel_show_previous_answer_not_applied_with_added_page(
    client, source, review_cases
) -> None:
    case = review_cases["atashinchi_next"]
    added_page_key = next(iter(sorted(case.entry_page_keys)))
    source.cases[case.case_name] = case.model_copy(
        update={
            "answer": None,
            "previous_answer": PreviousAnswer(
                case_name="split work: mal:anime:1",
                answer={"mal:anime:63013": "work 1"},
                added_page_keys=(added_page_key,),
                is_applied=False,
            ),
        }
    )
    case_list = client.post("/fragments/case-list", data=_state()).text
    panel = client.post(
        "/fragments/case-panel", data=_state(open_case_name=case.case_name)
    ).text
    assert "Previous answer" in case_list
    assert escape(Tooltip.PREVIOUS_ANSWER_NOT_APPLIED) in panel
    assert f"Added pages: {added_page_key}." in panel
    assert "The previous answer placed mal:anime:63013 as work 1." in panel
    assert escape(Tooltip.ADDED_PAGE_PILL) in panel


def test_case_panel_dropdowns_explain_chosen_and_offered_choices(
    client, review_cases
) -> None:
    panel = client.post(
        "/fragments/case-panel",
        data=_state(open_case_name=review_cases["atashinchi_next"].case_name),
    ).text
    assert (
        f'title="{escape(describe_answer_word("work 1"))}"' in panel,
        escape(describe_answer_word("dropped")) in panel,
        "What the choices mean" in panel,
    ) == (True, True, True)


def test_case_panel_linked_page_lists_what_each_answer_does(
    client, review_cases
) -> None:
    panel = client.post(
        "/fragments/case-panel",
        data=_state(open_case_name=review_cases["nami"].case_name),
    ).text
    assert all(
        escape(describe_answer_word(answer)) in panel
        for answer in ("add", "add as partial", "skip")
    )


def test_case_list_filters_and_tags_explain_themselves(client) -> None:
    case_list = client.post("/fragments/case-list", data=_state()).text
    assert all(
        f'title="{escape(description)}"' in case_list
        for description in (
            ReviewCaseKind.LINKED_PAGE.description,
            CaseState.OPEN.description,
            CaseEditState.REJECTED.description,
        )
    )


def test_case_list_pager_offers_first_previous_next_and_last_page(client) -> None:
    case_list = client.post("/fragments/case-list", data=_state()).text
    pager = case_list[case_list.index('class="pager"') :]
    assert (
        [
            label in pager
            for label in ("First page", "Previous page", "Next page", "Last page")
        ],
        ">Previous<" in pager or ">Next<" in pager,
        pager.count("disabled") >= 4,
    ) == ([True, True, True, True], False, True)


def test_redrawn_controls_carry_stable_ids(client, review_cases) -> None:
    case_name = review_cases["atashinchi_next"].case_name
    state = _state(open_case_name=case_name)
    first_panel = client.post("/fragments/case-panel", data=state).text
    second_panel = client.post("/fragments/case-panel", data=state).text
    case_list = client.post("/fragments/case-list", data=_state()).text
    assert (
        'id="choice-work-Work-of-mal-anime-63013"' in first_panel,
        'id="choice-work-Work-of-mal-anime-63013"' in second_panel,
        'id="chip-kinds-split-work"' in case_list,
        f'id="{build_element_id("case", case_name)}"' in case_list,
        'id="icon-Next-page' in case_list,
    ) == (True, True, True, True, True)


def test_redrawn_regions_never_anchor_scrolling(client) -> None:
    stylesheet = client.get("/static/review_page.css").text
    rule = stylesheet[stylesheet.index("#case-list, #case-panel") :]
    assert "overflow-anchor: none" in rule[: rule.index("}")]


def test_case_list_chosen_chips_drawn_on(client) -> None:
    case_list = client.post(
        "/fragments/case-list", data=_state(kinds=["linked page"], case_states=["open"])
    ).text

    def _read_chip_classes(chip_id: str) -> str:
        chip = case_list[case_list.index(f'id="{chip_id}"') :]
        return chip[
            chip.index('class="') + 7 : chip.index('"', chip.index('class="') + 7)
        ]

    assert (
        _read_chip_classes("chip-kinds-linked-page"),
        _read_chip_classes("chip-case_states-open"),
        _read_chip_classes("chip-kinds-split-work"),
    ) == ("control-chip is-on", "control-chip is-on", "control-chip")


def test_case_list_chip_count_shown_in_brackets(client) -> None:
    case_list = client.post("/fragments/case-list", data=_state()).text
    chip = case_list[case_list.index('id="chip-case_states-open"') :]
    chip = chip[: chip.index("</button>")]
    assert re.search(r'<span class="control-chip-count">\(\d+\)</span>', chip)


def test_case_panel_linked_page_heading_links_entry_and_page_below_titles(
    client, review_cases
) -> None:
    panel = client.post(
        "/fragments/case-panel",
        data=_state(open_case_name=review_cases["nami"].case_name),
    ).text
    heading = panel[panel.index("<thead>") : panel.index("</thead>")]
    assert (
        '<br><a class="page-key" href="https://kitsu.app/anime/10561"' in heading,
        '<br><a class="page-key" href="https://anilist.co/anime/102289"' in heading,
    ) == (True, True)


def test_run_state_reports_run_started_elsewhere(client, source) -> None:
    source.is_running = True
    response = client.post("/run-state", data=_state())
    assert (response.status_code, _events(response)) == (
        204,
        {"runstate": {"is_running": True}},
    )


def test_case_list_unreadable_review_file_answers_notice_naming_line(
    client, source
) -> None:
    source.is_review_file_unreadable = True
    response = client.post("/fragments/case-list", data=_state())
    assert response.status_code == 500
    assert "Line 2 of review.jsonl" in _events(response)["notice"]["text"]


@pytest.mark.parametrize(
    ("is_partial_source", "placement_text"),
    [("true", "work 1, partial in work 2"), ("false", "work 1")],
)
def test_choice_partial_source_adds_or_removes_work(
    client, review_cases, is_partial_source, placement_text
) -> None:
    case_name = review_cases["gochuumon_movie"].case_name
    start_placement = (
        "work 1" if is_partial_source == "true" else "work 1, partial in work 2"
    )
    response = client.post(
        "/fragments/choice",
        data={
            **_state(
                open_case_name=case_name,
                edits={
                    case_name: {
                        "anidb:anime:19188": start_placement,
                        "anidb:anime:19792": "work 2",
                    }
                },
            ),
            "page_key": "anidb:anime:19188",
            "partial_source_work": "work 2",
            "is_partial_source": is_partial_source,
        },
    )
    assert (
        _events(response)["answerchanged"]["answer"]["anidb:anime:19188"]
        == placement_text
    )


def test_case_panel_works_section_shows_partial_sources_and_picker(
    client, review_cases
) -> None:
    case_name = review_cases["gochuumon_movie"].case_name
    panel = client.post(
        "/fragments/case-panel",
        data=_state(
            open_case_name=case_name,
            edits={
                case_name: {
                    "anidb:anime:19188": "work 1, partial in work 2",
                    "anidb:anime:19792": "work 2",
                }
            },
        ),
    ).text
    works_section = panel.split(">Works</h3>", 1)[1]
    assert (
        'class="partial-source-chip"' in works_section,
        "anidb:anime:19188" in works_section.split("partial-source-chip", 2)[1],
        "Add partial source" in works_section,
        'page-key">anidb:anime:19792</span>' in works_section,
        "in work 2" in works_section.split('page-key">anidb:anime:19792', 1)[1][:400],
    ) == (True, True, True, True, True)


def test_choice_partial_sources_adds_every_ticked_page(client, review_cases) -> None:
    case_name = review_cases["gochuumon_movie"].case_name
    response = client.post(
        "/fragments/choice",
        data={
            **_state(
                open_case_name=case_name,
                edits={
                    case_name: {
                        "anidb:anime:19188": "work 1",
                        "mal:anime:61115": "work 1",
                        "anidb:anime:19792": "work 2",
                    }
                },
            ),
            "partial_source_work": "work 2",
            "is_partial_source": "true",
            "partial_source_page_keys": ["anidb:anime:19188", "mal:anime:61115"],
        },
    )
    answer = _events(response)["answerchanged"]["answer"]
    assert (answer["anidb:anime:19188"], answer["mal:anime:61115"]) == (
        "work 1, partial in work 2",
        "work 1, partial in work 2",
    )
