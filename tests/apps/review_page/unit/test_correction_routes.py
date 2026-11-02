import json

import pytest
from jinja2 import Template
from shared_proto.v1 import review_cases_pb2


@pytest.fixture
def corrected_source(source, correction_messages, checked_entry_messages):
    source.corrections = list(correction_messages)
    source.checked_entries = list(checked_entry_messages)
    return source


def escape(text: str) -> str:
    return Template("{{ text }}", autoescape=True).render(text=text)


def _state(**fields) -> dict[str, str]:
    return {"browser_state": json.dumps(fields)}


def test_correction_list_lists_every_corrected_entry_with_counts(
    client, corrected_source
) -> None:
    correction_list = client.post("/fragments/correction-list", data=_state()).text
    assert (
        "Viewing 1–11 of 11" in correction_list,
        "Choujikuu Seiki Orguss 02" in correction_list,
        "Pages moved to a seed entry" in correction_list,
    ) == (True, True, True)


def test_correction_list_change_filter_keeps_matching_entries(
    client, corrected_source
) -> None:
    correction_list = client.post(
        "/fragments/correction-list",
        data=_state(correction_changes=["pages moved to a seed entry"]),
    ).text
    assert "Viewing 1–4 of 4" in correction_list


def test_correction_list_answer_source_marks_answered_entry(
    client, corrected_source
) -> None:
    correction_list = client.post(
        "/fragments/correction-list", data=_state(correction_sources=["your answer"])
    ).text
    assert ("Viewing 1–1 of 1" in correction_list, "Atashin" in correction_list) == (
        True,
        True,
    )


def test_correction_list_before_any_run_says_so(client) -> None:
    correction_list = client.post("/fragments/correction-list", data=_state()).text
    assert "No corrections yet" in correction_list


def test_correction_panel_shows_entry_pages_and_destinations(
    client, corrected_source
) -> None:
    panel = client.post(
        "/fragments/correction-panel",
        data=_state(open_correction_entry="mal:anime:37196"),
    ).text
    assert (
        "anilist:anime:199801" in panel,
        "seed entry mal:anime:62648" in panel,
        'href="https://myanimelist.net/anime/62648"' in panel,
        str(escape("at most one page belongs to the entry's work")) in panel,
    ) == (True, True, True, True)


def test_correction_panel_links_case_holding_entry(
    client, corrected_source, review_cases
) -> None:
    case = review_cases["gochuumon_movie"]
    corrected_source.corrections.append(
        review_cases_pb2.SeedCorrection(
            entry_reference_page="mal:anime:61115",
            page_key="anidb:anime:19188",
            address="https://anidb.net/anime/19188",
            placement_word="other work",
            reason="review answer",
            reason_sentence="a reviewer's answer placed the page",
            is_from_answer=True,
        )
    )
    panel = client.post(
        "/fragments/correction-panel",
        data=_state(open_correction_entry="mal:anime:61115"),
    ).text
    assert f"$store.review.showCase({escape(json.dumps(case.case_name))})" in panel


def test_correction_panel_without_open_entry_asks_to_choose(client) -> None:
    panel = client.post("/fragments/correction-panel", data=_state()).text
    assert "Choose an entry" in panel


def test_case_panel_links_corrections_of_case_entries(
    client, corrected_source, review_cases
) -> None:
    corrected_source.corrections.append(
        review_cases_pb2.SeedCorrection(
            entry_reference_page="mal:anime:63013",
            page_key="anidb:anime:19895",
            address="https://anidb.net/anime/19895",
            placement_word="work 1",
            reason="review answer",
            reason_sentence="a reviewer's answer placed the page",
            is_from_answer=True,
        )
    )
    panel = client.post(
        "/fragments/case-panel",
        data=_state(open_case_name=review_cases["atashinchi_next"].case_name),
    ).text
    assert ("showCorrection(" in panel, "1 correction" in panel) == (True, True)


def test_page_offers_cases_and_corrections_tabs(client) -> None:
    page = client.get("/").text
    tabs = page[page.index('role="tablist"') :]
    tabs = tabs[: tabs.index("</nav>")]
    assert (
        tabs.count('role="tab"'),
        "Cases" in tabs and "Corrections" in tabs,
        'id="correction-list"' in page and 'id="correction-panel"' in page,
    ) == (2, True, True)


def test_correction_list_chosen_chips_drawn_on(client, corrected_source) -> None:
    correction_list = client.post(
        "/fragments/correction-list", data=_state(correction_sources=["rules"])
    ).text
    chip = correction_list[
        correction_list.index('id="chip-correction_sources-rules"') :
    ]
    assert 'class="part-chip is-on"' in chip[: chip.index(">")]
