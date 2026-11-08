import pytest
from work_matching.model_doubled_provider_decision import (
    DoubledProviderDecision,
    DoubledProviderOutcome,
)
from work_matching.part_page_splits import (
    find_splitting_decision,
    list_duplicate_page_keys,
    suggest_part_page_splits,
)
from work_matching.provider_page import ProviderPage


def _build_dated_page(
    page_key: str, episode_count: int, aired_from: str, aired_to: str | None
) -> ProviderPage:
    return ProviderPage(
        key=page_key,
        address=f"https://example.org/{page_key}",
        episode_count=episode_count,
        aired_from=aired_from,
        aired_to=aired_to,
    )


def _build_decision(
    outcome: DoubledProviderOutcome, *page_keys: str
) -> DoubledProviderDecision:
    return DoubledProviderDecision(
        provider=page_keys[0].split(":")[0],
        page_keys=page_keys,
        outcome=outcome,
        review_reasons=(),
    )


SAINT_SEIYA_PAGES = (
    _build_dated_page(
        "anilist:anime:114448", 6, "2020-01-22T00:00:00Z", "2020-01-22T00:00:00Z"
    ),
    _build_dated_page(
        "anilist:anime:102427", 6, "2019-07-18T00:00:00Z", "2019-07-18T00:00:00Z"
    ),
    _build_dated_page(
        "mal:anime:36040", 12, "2019-07-18T15:00:00Z", "2020-01-22T15:00:00Z"
    ),
    _build_dated_page("kitsu:anime:1", 6, "2019-07-19T00:00:00Z", None),
    _build_dated_page("kitsu:anime:2", 6, "2019-07-18T00:00:00Z", None),
    _build_dated_page("anisearch:anime:1", 30, "2015-01-01T00:00:00Z", None),
)
SAINT_SEIYA_DECISION = _build_decision(
    DoubledProviderOutcome.PART_PAGES, "anilist:anime:114448", "anilist:anime:102427"
)


def test_suggest_part_page_splits_one_work_per_part_and_partial_sources() -> None:
    placeable_page_keys = [page.key for page in SAINT_SEIYA_PAGES]
    assert suggest_part_page_splits(
        placeable_page_keys, SAINT_SEIYA_DECISION, SAINT_SEIYA_PAGES
    ) == {
        "anilist:anime:114448": "work 2",
        "anilist:anime:102427": "work 1",
        "mal:anime:36040": "work 1, partial in work 2",
    }


@pytest.mark.parametrize(
    ("decisions", "splitting_page_keys"),
    [
        ([SAINT_SEIYA_DECISION], SAINT_SEIYA_DECISION.page_keys),
        (
            [
                _build_decision(
                    DoubledProviderOutcome.PART_PAGES, "kitsu:anime:1", "kitsu:anime:2"
                ),
                _build_decision(
                    DoubledProviderOutcome.PART_PAGES,
                    "anidb:anime:1",
                    "anidb:anime:2",
                    "anidb:anime:3",
                ),
            ],
            ("anidb:anime:1", "anidb:anime:2", "anidb:anime:3"),
        ),
        (
            [
                _build_decision(
                    DoubledProviderOutcome.DUPLICATE_PAGES,
                    "kitsu:anime:46351",
                    "kitsu:anime:50917",
                )
            ],
            None,
        ),
    ],
)
def test_find_splitting_decision_most_part_pages_wins(
    decisions: list[DoubledProviderDecision],
    splitting_page_keys: tuple[str, ...] | None,
) -> None:
    splitting_decision = find_splitting_decision(decisions)
    assert (
        splitting_decision.page_keys if splitting_decision else None
    ) == splitting_page_keys


def test_list_duplicate_page_keys_lists_duplicate_page_decisions_only() -> None:
    decisions = [
        SAINT_SEIYA_DECISION,
        _build_decision(
            DoubledProviderOutcome.DUPLICATE_PAGES,
            "kitsu:anime:46351",
            "kitsu:anime:50917",
        ),
    ]
    assert list_duplicate_page_keys(decisions) == {
        "kitsu:anime:46351",
        "kitsu:anime:50917",
    }
