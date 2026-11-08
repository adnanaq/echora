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
NARARA_PAGES = tuple(
    _build_dated_page(
        page_key, episode_count, f"{aired_from}T15:00:00Z", f"{aired_to}T15:00:00Z"
    )
    for page_key, episode_count, aired_from, aired_to in (
        ("anidb:anime:10799", 28, "1990-08-20", "1993-03-18"),
        ("anidb:anime:10873", 13, "1998-07-02", "1998-09-24"),
        ("anidb:anime:10874", 13, "2001-10-18", "2002-01-10"),
        ("anilist:anime:205608", 2, "1990-08-14", "1990-08-14"),
        ("anilist:anime:205615", 13, "1991-10-17", "1992-01-09"),
        ("animeplanet:anime:narara-superboard", 28, "1990-08-20", "1993-01-10"),
        ("animeplanet:anime:narara-superboard-1998", 13, "1998-07-02", "1998-09-24"),
        ("animeplanet:anime:narara-superboard-2001", 13, "2001-10-18", "2002-01-10"),
        ("anisearch:anime:9933", 54, "1990-08-20", "2002-01-10"),
        ("kitsu:anime:12176", 54, "1990-08-20", "2002-01-10"),
        ("mal:anime:32136", 54, "1990-08-20", "2002-01-10"),
    )
)
NARARA_DECISIONS = [
    _build_decision(
        DoubledProviderOutcome.PART_PAGES,
        "anidb:anime:10799",
        "anidb:anime:10873",
        "anidb:anime:10874",
    ),
    _build_decision(
        DoubledProviderOutcome.PART_PAGES,
        "anilist:anime:205608",
        "anilist:anime:205615",
    ),
    _build_decision(
        DoubledProviderOutcome.UNDECIDED,
        "animeplanet:anime:narara-superboard",
        "animeplanet:anime:narara-superboard-1998",
        "animeplanet:anime:narara-superboard-2001",
    ),
]


@pytest.mark.parametrize(
    ("pages", "splitting_decision", "suggested_placements"),
    [
        (
            SAINT_SEIYA_PAGES,
            SAINT_SEIYA_DECISION,
            {
                "anilist:anime:114448": "work 2",
                "anilist:anime:102427": "work 1",
                "mal:anime:36040": "work 1, partial in work 2",
            },
        ),
        (
            NARARA_PAGES,
            NARARA_DECISIONS[0],
            {
                "anidb:anime:10799": "work 1",
                "anidb:anime:10873": "work 2",
                "anidb:anime:10874": "work 3",
                "animeplanet:anime:narara-superboard-1998": "work 2",
                "animeplanet:anime:narara-superboard-2001": "work 3",
                "anisearch:anime:9933": "work 1, partial in work 2, work 3",
                "kitsu:anime:12176": "work 1, partial in work 2, work 3",
                "mal:anime:32136": "work 1, partial in work 2, work 3",
            },
        ),
    ],
)
def test_suggest_part_page_splits_one_work_per_part_and_partial_sources(
    pages: tuple[ProviderPage, ...],
    splitting_decision: DoubledProviderDecision,
    suggested_placements: dict[str, str],
) -> None:
    placeable_page_keys = [page.key for page in pages]
    assert (
        suggest_part_page_splits(placeable_page_keys, splitting_decision, pages)
        == suggested_placements
    )


@pytest.mark.parametrize(
    ("decisions", "splitting_page_keys"),
    [
        ([SAINT_SEIYA_DECISION], SAINT_SEIYA_DECISION.page_keys),
        (NARARA_DECISIONS, NARARA_DECISIONS[0].page_keys),
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
