import json

from google.protobuf import json_format
from jinja2 import Template
from review_page.page_state import CaseEditState, CaseState
from review_page.template_functions import build_element_id, build_static_url
from shared_proto.v1 import review_cases_pb2
from work_matching import (
    PART_MARK,
    ReviewCaseKind,
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
        page.index("applyReviewPageTheme(") < page.index("parts.css"),
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
    assert client.get("/static/parts.css").status_code == 200


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
    assert all(key in panel for key in ATASHINCHI_ANSWER)
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


def test_case_panel_without_open_case_asks_to_choose(client) -> None:
    assert "Choose a case" in client.post("/fragments/case-panel", data=_state()).text


def test_choice_part_spreads_and_reports_unsaved_answer(client, review_cases) -> None:
    case_name = review_cases["gochuumon_movie"].case_name
    response = client.post(
        "/fragments/choice",
        data={
            **_state(
                open_case_name=case_name,
                edits={case_name: dict.fromkeys(GOCHUUMON_DOUBLED_PAGES, "work 1")},
            ),
            "page_key": "anidb:anime:19792",
            "work": "work 1",
            "mark": "part",
        },
    )
    answer_changed = _events(response)["answerchanged"]
    assert answer_changed == {
        "case_name": case_name,
        "answer": dict.fromkeys(GOCHUUMON_DOUBLED_PAGES, "work 1, part"),
        "is_unsaved": True,
    }


def test_choice_back_to_suggestion_reports_no_unsaved_edit(
    client, review_cases
) -> None:
    case = review_cases["atashinchi_next"]
    changed = {**case.suggested_answer, "anidb:anime:19895": "work 2"}
    response = client.post(
        "/fragments/choice",
        data={
            **_state(open_case_name=case.case_name, edits={case.case_name: changed}),
            "page_key": "anidb:anime:19895",
            "work": "work 1",
            "mark": "",
        },
    )
    answer_changed = _events(response)["answerchanged"]
    assert (
        answer_changed["answer"],
        answer_changed["is_unsaved"],
        "edited, not saved" in response.text,
    ) == (dict(case.suggested_answer), False, False)


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
    assert (response.status_code, _events(response)["answerssaved"]["saved"]) == (
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
        case_name: ["a page this answer must place has no place"]
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
    assert sorted(_events(response)["answerssaved"]["saved"]) == sorted(edits)


def test_answers_while_run_holds_lock_answers_409(client, source, review_cases) -> None:
    source.is_locked = True
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
    assert build_static_url("parts.css") in page
    assert "?v=" in build_static_url("parts.css")


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
    assert (case_list.count("No longer comes up"), "No longer comes up" in panel) == (
        1,
        True,
    )


def test_case_panel_dropdowns_explain_chosen_and_offered_choices(
    client, review_cases
) -> None:
    panel = client.post(
        "/fragments/case-panel",
        data=_state(open_case_name=review_cases["atashinchi_next"].case_name),
    ).text
    assert (
        f'title="{escape(describe_answer_word("work 1"))}"' in panel,
        f'title="{escape(describe_answer_word(PART_MARK))}"' in panel,
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
        for answer in ("add", "add as part", "skip")
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
    stylesheet = client.get("/static/parts.css").text
    rule = stylesheet[stylesheet.index("#case-list, #case-panel") :]
    assert "overflow-anchor: none" in rule[: rule.index("}")]


def test_case_list_chosen_chips_drawn_on(client) -> None:
    case_list = client.post(
        "/fragments/case-list", data=_state(kinds=["linked page"], case_states=["open"])
    ).text

    def chip_classes(chip_id: str) -> str:
        chip = case_list[case_list.index(f'id="{chip_id}"') :]
        return chip[
            chip.index('class="') + 7 : chip.index('"', chip.index('class="') + 7)
        ]

    assert (
        chip_classes("chip-kinds-linked-page"),
        chip_classes("chip-case_states-open"),
        chip_classes("chip-kinds-split-work"),
    ) == ("part-chip is-on", "part-chip is-on", "part-chip")
