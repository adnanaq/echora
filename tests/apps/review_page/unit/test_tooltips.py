import json
import re
from pathlib import Path

import pytest
from review_page.correction_views import ANSWER_SOURCE, RULES_SOURCE
from review_page.models import CaseEditState, CaseState
from review_page.tooltips import (
    CASE_STATE_TOOLTIPS,
    CORRECTION_CHANGE_TOOLTIPS,
    CORRECTION_SOURCE_TOOLTIPS,
    EDIT_STATE_TOOLTIPS,
    THEME_TOOLTIPS,
    Tooltip,
)
from seed_cleaning import CORRECTION_COUNT_NAMES

PAGE_SOURCES = (
    Path(__file__).parents[4] / "apps" / "review_page" / "src" / "review_page"
)
CASE_COUNT_NAMES = {"review cases answered", "review cases open"}


def test_edit_state_tooltips_cover_every_case_edit_state() -> None:
    assert set(EDIT_STATE_TOOLTIPS) == {state.value for state in CaseEditState}


def test_case_state_tooltips_cover_every_case_state() -> None:
    assert set(CASE_STATE_TOOLTIPS) == {state.value for state in CaseState}


def test_correction_change_tooltips_cover_every_change_and_unchanged() -> None:
    assert set(CORRECTION_CHANGE_TOOLTIPS) == (
        set(CORRECTION_COUNT_NAMES) - CASE_COUNT_NAMES
    ) | {"unchanged"}


def test_correction_source_tooltips_cover_every_source() -> None:
    assert set(CORRECTION_SOURCE_TOOLTIPS) == {ANSWER_SOURCE, RULES_SOURCE}


def test_theme_tooltips_cover_every_theme_choice() -> None:
    script = (PAGE_SOURCES / "static" / "review_page.js").read_text(encoding="utf-8")
    assert set(THEME_TOOLTIPS) == set(re.findall(r'"(system|light|dark)"', script))


@pytest.mark.parametrize("tooltip", list(Tooltip), ids=lambda tooltip: tooltip.name)
def test_tooltip_used_by_page(tooltip: Tooltip) -> None:
    sources = "".join(
        path.read_text(encoding="utf-8")
        for path in PAGE_SOURCES.rglob("*")
        if path.suffix in (".html", ".py") and path.name != "tooltips.py"
    )
    assert re.search(rf"\b(tooltips|Tooltip)\.{tooltip.name}\b", sources)


def test_rendered_fragments_have_no_empty_tooltip(
    client, source, review_cases, correction_messages, checked_entry_messages
) -> None:
    source.corrections = list(correction_messages)
    source.checked_entries = list(checked_entry_messages)
    case_names = [case.case_name for case in review_cases.values()]
    state = {"edits": {case_names[0]: {}}}
    rendered = [
        client.get("/").text,
        client.post(
            "/fragments/toolbar", data={"browser_state": json.dumps(state)}
        ).text,
        client.post("/fragments/case-list", data={"browser_state": "{}"}).text,
        client.post("/fragments/correction-list", data={"browser_state": "{}"}).text,
        *(
            client.post(
                "/fragments/case-panel",
                data={"browser_state": json.dumps({"open_case_name": case_name})},
            ).text
            for case_name in case_names
        ),
        *(
            client.post(
                "/fragments/correction-panel",
                data={
                    "browser_state": json.dumps(
                        {"open_correction_entry": correction.entry_reference_page}
                    )
                },
            ).text
            for correction in correction_messages
        ),
    ]
    assert not [text for text in rendered if 'title=""' in text or "title=''" in text]
