from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import create_autospec

from match_suggestions.case_judging import (
    estimate_input_tokens,
    judge_pair_requests,
    list_pair_requests,
)
from match_suggestions.model_pair_judgment import PairJudgment
from match_suggestions.pair_judgment_store import JsonlPairJudgmentStore
from match_suggestions.pair_questions import QUESTION_VERSION
from typesafe_sdk import (
    AsyncTypeSafeClient,
    SystemOneResponse,
    TypeSafeAPIConnectionError,
)
from work_matching import ProviderPage, ReviewCase, WorkEntry

MAL = "mal:anime:1"
ANILIST = "anilist:anime:2"
KITSU = "kitsu:anime:3"
SCHEDULE = "animeschedule:anime:show"
ADDRESSES = {
    MAL: "https://myanimelist.net/anime/1",
    ANILIST: "https://anilist.co/anime/2",
    KITSU: "https://kitsu.app/anime/3",
    SCHEDULE: "https://animeschedule.net/anime/show",
}


def _case(
    page_keys: list[str],
    suggested_answer: dict[str, str],
    answer: dict[str, str] | None = None,
) -> ReviewCase:
    return ReviewCase(
        case_name=f"split work: {' '.join(page_keys)}",
        answer=answer,
        suggested_answer=suggested_answer,
        reasons=(),
        entries={
            page_key: WorkEntry(
                title="Show",
                type="TV",
                addresses_by_page_key={page_key: ADDRESSES[page_key]},
            )
            for page_key in page_keys
        },
        decisions=(),
        pages=(
            *(
                ProviderPage(key=page_key, address=ADDRESSES[page_key], year=2020)
                for page_key in page_keys
            ),
            ProviderPage(key=SCHEDULE, address=ADDRESSES[SCHEDULE], searched_by=(MAL,)),
        ),
    )


async def _read_synopsis(page: ProviderPage) -> str | None:
    return f"Synopsis of {page.key}."


def _answer_every_question(is_same_work: float) -> SystemOneResponse:
    return SystemOneResponse.model_validate(
        {
            "model": "jev-1.14.0",
            "usage": {"input_tokens": 400, "output_tokens": 4},
            "answers": {
                "is_same_franchise": {"type": "noul", "noul": 0.9},
                "is_same_work": {"type": "noul", "noul": is_same_work},
                "is_first_work_part_of_second_work": {"type": "noul", "noul": 0.1},
                "is_second_work_part_of_first_work": {"type": "noul", "noul": 0.2},
            },
        }
    )


async def test_list_pair_requests_pairs_open_pages_with_case_pages(
    tmp_path: Path,
) -> None:
    store = JsonlPairJudgmentStore(tmp_path / "judgments.jsonl")
    review_cases = [
        _case([MAL, ANILIST, KITSU], {MAL: "work 1"}),
        _case(
            [MAL, ANILIST], {MAL: "work 1"}, answer={MAL: "work 1", ANILIST: "work 2"}
        ),
    ]

    pair_requests = await list_pair_requests(review_cases, store, _read_synopsis)

    assert [pair_request.page_key_pair for pair_request in pair_requests] == [
        (ANILIST, KITSU),
        (ANILIST, MAL),
        (KITSU, MAL),
    ]
    assert (
        pair_requests[0].pair_state["page_a"]["synopsis"] == f"Synopsis of {ANILIST}."
    )


async def test_list_pair_requests_pair_judged_with_same_request_left_out(
    tmp_path: Path,
) -> None:
    store = JsonlPairJudgmentStore(tmp_path / "judgments.jsonl")
    review_cases = [_case([MAL, ANILIST], {MAL: "work 1"})]
    (pair_request,) = await list_pair_requests(review_cases, store, _read_synopsis)
    store.add_judgments(
        [
            PairJudgment(
                first_page_key=ANILIST,
                second_page_key=MAL,
                question_version=QUESTION_VERSION,
                request_fingerprint=pair_request.request_fingerprint,
                is_same_franchise=0.9,
                is_same_work=0.9,
                is_first_work_part_of_second_work=0.1,
                is_second_work_part_of_first_work=0.1,
                judged_at=datetime(2026, 11, 23, tzinfo=UTC),
            )
        ]
    )

    async def read_changed_synopsis(page: ProviderPage) -> str | None:
        return "A new synopsis."

    assert await list_pair_requests(review_cases, store, _read_synopsis) == ()
    assert (
        len(await list_pair_requests(review_cases, store, read_changed_synopsis)) == 1
    )


async def test_judge_pair_requests_stores_answers(tmp_path: Path) -> None:
    store = JsonlPairJudgmentStore(tmp_path / "judgments.jsonl")
    (pair_request,) = await list_pair_requests(
        [_case([MAL, ANILIST], {MAL: "work 1"})], store, _read_synopsis
    )
    client = create_autospec(AsyncTypeSafeClient, instance=True)
    client.system_one.return_value = _answer_every_question(0.85)

    judging_summary = await judge_pair_requests([pair_request], store, client)

    stored_judgment = store.find_judgments([(ANILIST, MAL)])[(ANILIST, MAL)]
    assert (
        judging_summary.judged_pair_count,
        judging_summary.failed_pair_count,
        judging_summary.input_token_count,
    ) == (1, 0, 400)
    assert (
        stored_judgment.is_same_work,
        stored_judgment.is_second_work_part_of_first_work,
    ) == (0.85, 0.2)
    assert stored_judgment.request_fingerprint == pair_request.request_fingerprint


async def test_judge_pair_requests_failed_request_stores_nothing(
    tmp_path: Path,
) -> None:
    store = JsonlPairJudgmentStore(tmp_path / "judgments.jsonl")
    (pair_request,) = await list_pair_requests(
        [_case([MAL, ANILIST], {MAL: "work 1"})], store, _read_synopsis
    )
    client = create_autospec(AsyncTypeSafeClient, instance=True)
    client.system_one.side_effect = TypeSafeAPIConnectionError("connection refused")

    judging_summary = await judge_pair_requests([pair_request], store, client)

    assert (
        judging_summary.judged_pair_count,
        judging_summary.failed_pair_count,
        judging_summary.input_token_count,
    ) == (0, 1, 0)
    assert store.find_judgments([(ANILIST, MAL)]) == {}


async def test_estimate_input_tokens_adds_each_request(tmp_path: Path) -> None:
    (pair_request,) = await list_pair_requests(
        [_case([MAL, ANILIST], {MAL: "work 1"})],
        JsonlPairJudgmentStore(tmp_path / "judgments.jsonl"),
        _read_synopsis,
    )

    single_request_tokens = estimate_input_tokens([pair_request])

    assert single_request_tokens > 0
    assert (
        estimate_input_tokens([pair_request, pair_request]) == 2 * single_request_tokens
    )
