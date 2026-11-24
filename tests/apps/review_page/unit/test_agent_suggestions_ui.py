"""Unit tests for AgentService suggestion pre-selection and score display in review_page."""

import re
from typing import Any

from fastapi.testclient import TestClient
from review_page.app import create_app
from review_page.models import AgentPageSuggestion
from work_matching import ProviderPage, ReviewCase, ReviewCaseKind, WorkEntry


class FakeAgentSuggestionsClient:
    """Return fixed Jev suggestions keyed by case_name for unit tests."""

    def __init__(
        self, suggestions_by_case: dict[str, tuple[AgentPageSuggestion, ...]]
    ) -> None:
        self._suggestions_by_case = suggestions_by_case

    def suggest_page_answers(self, case: ReviewCase) -> tuple[AgentPageSuggestion, ...]:
        """Return configured suggestions for the given case."""
        return self._suggestions_by_case.get(case.case_name, ())


def _build_split_case(answer: dict[str, Any] | None = None) -> ReviewCase:
    return ReviewCase(
        case_name="Monster [TV]",
        kind=ReviewCaseKind.SPLIT_WORK,
        reasons=(),
        decisions=(),
        entries={
            "mal:anime:10": WorkEntry(
                title="Monster",
                type="TV",
                year=2004,
                addresses_by_page_key={
                    "mal:anime:10": "https://myanimelist.net/anime/10"
                },
            ),
            "anilist:anime:19": WorkEntry(
                title="Monster",
                type="TV",
                year=2004,
                addresses_by_page_key={
                    "anilist:anime:19": "https://anilist.co/anime/19"
                },
            ),
        },
        pages=(
            ProviderPage(
                key="mal:anime:10",
                address="https://myanimelist.net/anime/10",
                titles=["Monster"],
                type="TV",
                year=2004,
            ),
            ProviderPage(
                key="anilist:anime:19",
                address="https://anilist.co/anime/19",
                titles=["Monster"],
                type="TV",
                year=2004,
            ),
        ),
        placeable_page_keys=("mal:anime:10", "anilist:anime:19"),
        suggested_answer={"mal:anime:10": "work 1"},
        answer=answer,
    )


def test_review_page_preselects_agent_suggestion_and_shows_score(
    source: Any,
) -> None:
    """Verify open cases pre-select Jev's confident placement and display its score."""
    case = _build_split_case()
    source.cases = {case.case_name: case}
    suggestions_client = FakeAgentSuggestionsClient(
        {
            case.case_name: (
                AgentPageSuggestion(
                    page_key="anilist:anime:19",
                    answer_word="work 1",
                    score=0.92,
                ),
            )
        }
    )
    client = TestClient(
        create_app(source, suggestion_client=suggestions_client),
        base_url="http://127.0.0.1:8003",
        headers={"Origin": "http://127.0.0.1:8003", "HX-Request": "true"},
    )

    response = client.post(
        "/fragments/case-panel",
        data={"browser_state": '{"open_case_name": "Monster [TV]"}'},
    )

    assert response.status_code == 200
    assert "Jev 92%" in response.text
    assert re.search(
        r'aria-label="Work of anilist:anime:19".*?<option value="work 1"\s+selected',
        response.text,
        re.DOTALL,
    )


def test_review_page_hides_score_after_reviewer_changes_choice(
    source: Any,
) -> None:
    """Verify changing a page's dropdown hides Jev's confidence score for that page."""
    case = _build_split_case()
    source.cases = {case.case_name: case}
    suggestions_client = FakeAgentSuggestionsClient(
        {
            case.case_name: (
                AgentPageSuggestion(
                    page_key="anilist:anime:19",
                    answer_word="work 1",
                    score=0.92,
                ),
            )
        }
    )
    client = TestClient(
        create_app(source, suggestion_client=suggestions_client),
        base_url="http://127.0.0.1:8003",
        headers={"Origin": "http://127.0.0.1:8003", "HX-Request": "true"},
    )

    response = client.post(
        "/fragments/choice",
        data={
            "browser_state": '{"open_case_name": "Monster [TV]"}',
            "page_key": "anilist:anime:19",
            "work": "work 2",
        },
    )

    assert response.status_code == 200
    assert "92%" not in response.text


def test_review_page_never_changes_saved_answer(source: Any) -> None:
    """Verify saved answers are never overwritten or annotated by agent suggestions."""
    saved_case = _build_split_case(
        answer={"mal:anime:10": "work 1", "anilist:anime:19": "work 2"}
    )
    source.cases = {saved_case.case_name: saved_case}
    suggestions_client = FakeAgentSuggestionsClient(
        {
            saved_case.case_name: (
                AgentPageSuggestion(
                    page_key="anilist:anime:19",
                    answer_word="work 1",
                    score=0.95,
                ),
            )
        }
    )
    client = TestClient(
        create_app(source, suggestion_client=suggestions_client),
        base_url="http://127.0.0.1:8003",
        headers={"Origin": "http://127.0.0.1:8003", "HX-Request": "true"},
    )

    response = client.post(
        "/fragments/case-panel",
        data={"browser_state": '{"open_case_name": "Monster [TV]"}'},
    )

    assert response.status_code == 200
    assert "95%" not in response.text
    assert re.search(
        r'aria-label="Work of anilist:anime:19".*?<option value="work 2"\s+selected',
        response.text,
        re.DOTALL,
    )


def test_review_page_accept_saves_preselected_agent_suggestion(source: Any) -> None:
    """Verify Accept persists Jev's pre-selected placement for an open case."""
    case = _build_split_case()
    source.cases = {case.case_name: case}
    suggestions_client = FakeAgentSuggestionsClient(
        {
            case.case_name: (
                AgentPageSuggestion(
                    page_key="anilist:anime:19",
                    answer_word="work 1",
                    score=0.92,
                ),
            )
        }
    )
    client = TestClient(
        create_app(source, suggestion_client=suggestions_client),
        base_url="http://127.0.0.1:8003",
        headers={"Origin": "http://127.0.0.1:8003", "HX-Request": "true"},
    )

    response = client.post(
        "/answers",
        data={
            "browser_state": '{"open_case_name": "Monster [TV]"}',
            "action": "accept",
            "case_name": "Monster [TV]",
        },
    )

    assert response.status_code == 204
    assert source.cases["Monster [TV]"].answer == {
        "mal:anime:10": "work 1",
        "anilist:anime:19": "work 1",
    }
