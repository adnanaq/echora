from datetime import UTC, datetime

import pytest
from match_suggestions.model_pair_judgment import PairJudgment
from match_suggestions.pair_relation import PairRelation, decide_pair_relation
from work_matching import ProviderPage

DATED_PAGE = ProviderPage(
    key="mal:anime:1", address="https://myanimelist.net/anime/1", year=2020
)
OTHER_DATED_PAGE = ProviderPage(
    key="anilist:anime:2", address="https://anilist.co/anime/2", year=2020
)
UNDATED_PAGE = ProviderPage(key="kitsu:anime:3", address="https://kitsu.app/anime/3")


def _judgment(
    is_same_work: float, first_part: float, second_part: float
) -> PairJudgment:
    return PairJudgment(
        first_page_key="anilist:anime:2",
        second_page_key="mal:anime:1",
        question_version="version",
        request_fingerprint="fingerprint",
        is_same_franchise=0.9,
        is_same_work=is_same_work,
        is_first_work_part_of_second_work=first_part,
        is_second_work_part_of_first_work=second_part,
        judged_at=datetime(2026, 11, 23, tzinfo=UTC),
    )


@pytest.mark.parametrize(
    ("is_same_work", "first_part", "second_part", "expected_relation"),
    [
        (0.9, 0.1, 0.2, PairRelation.SAME_WORK),
        (0.9, 0.5, 0.1, PairRelation.UNSETTLED),
        (0.4, 0.8, 0.1, PairRelation.PARTIAL_IN),
        (0.4, 0.1, 0.8, PairRelation.PARTIAL_IN),
        (0.2, 0.1, 0.1, PairRelation.DIFFERENT_WORKS),
        (0.5, 0.5, 0.1, PairRelation.UNSETTLED),
    ],
)
def test_decide_pair_relation_reads_thresholds(
    is_same_work: float,
    first_part: float,
    second_part: float,
    expected_relation: PairRelation,
) -> None:
    assert (
        decide_pair_relation(
            _judgment(is_same_work, first_part, second_part),
            DATED_PAGE,
            OTHER_DATED_PAGE,
        )
        is expected_relation
    )


def test_decide_pair_relation_same_work_without_year_stays_unsettled() -> None:
    assert (
        decide_pair_relation(_judgment(0.95, 0.05, 0.05), DATED_PAGE, UNDATED_PAGE)
        is PairRelation.UNSETTLED
    )
