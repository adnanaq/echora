import pytest
from work_matching.covered_part_pages import find_covered_part_pages
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


SEITOKAI_PARTS = (
    _build_dated_page(
        "anidb:anime:10761", 9, "2014-05-15T15:00:00Z", "2020-09-16T15:00:00Z"
    ),
    _build_dated_page(
        "anidb:anime:10810", 1, "2014-10-21T15:00:00Z", "2014-10-21T15:00:00Z"
    ),
)
DOULUO_ARCS = (
    _build_dated_page(
        "anilist:anime:157127", 4, "2022-11-18T00:00:00Z", "2022-12-09T00:00:00Z"
    ),
    _build_dated_page(
        "anilist:anime:163783", 2, "2022-12-16T00:00:00Z", "2022-12-30T00:00:00Z"
    ),
    _build_dated_page(
        "anilist:anime:163784", 3, "2023-01-05T00:00:00Z", "2023-01-19T00:00:00Z"
    ),
)
SAINT_SEIYA_PARTS = (
    _build_dated_page(
        "anilist:anime:102427", 6, "2019-07-18T00:00:00Z", "2019-07-18T00:00:00Z"
    ),
    _build_dated_page(
        "anilist:anime:114448", 6, "2020-01-22T00:00:00Z", "2020-01-22T00:00:00Z"
    ),
)


@pytest.mark.parametrize(
    ("page", "part_pages", "covered_page_keys"),
    [
        (
            _build_dated_page(
                "mal:anime:36040", 12, "2019-07-18T15:00:00Z", "2020-01-22T15:00:00Z"
            ),
            SAINT_SEIYA_PARTS,
            ["anilist:anime:102427", "anilist:anime:114448"],
        ),
        (
            _build_dated_page("kitsu:anime:43641", 6, "2019-07-19T00:00:00Z", None),
            SAINT_SEIYA_PARTS,
            ["anilist:anime:102427"],
        ),
        (
            _build_dated_page(
                "mal:anime:26123", 10, "2014-05-15T15:00:00Z", "2020-09-16T15:00:00Z"
            ),
            SEITOKAI_PARTS,
            ["anidb:anime:10761", "anidb:anime:10810"],
        ),
        (
            _build_dated_page(
                "kitsu:anime:47595", 3, "2022-12-16T00:00:00Z", "2022-12-30T00:00:00Z"
            ),
            DOULUO_ARCS,
            [],
        ),
        (
            ProviderPage(key="mal:anime:1", address="https://example.org/mal"),
            SAINT_SEIYA_PARTS,
            [],
        ),
        (
            _build_dated_page("kitsu:anime:43641", 6, "2019-07-20T00:00:00Z", None),
            SAINT_SEIYA_PARTS,
            ["anilist:anime:102427"],
        ),
        (
            _build_dated_page(
                "mal:anime:1", 6, "2019-07-18T00:00:00Z", "2019-08-22T00:00:00Z"
            ),
            (
                _build_dated_page(
                    "anilist:anime:1", 6, "2019-07-18T00:00:00Z", "2019-08-22T00:00:00Z"
                ),
                _build_dated_page(
                    "anilist:anime:2", 6, "2019-07-18T00:00:00Z", "2019-08-22T00:00:00Z"
                ),
            ),
            [],
        ),
        (
            _build_dated_page("kitsu:anime:43641", 6, "2019-07-19T00:00:00Z", None),
            (
                ProviderPage(
                    key="anilist:anime:3",
                    address="https://example.org/anilist:anime:3",
                    episode_count=6,
                ),
                *SAINT_SEIYA_PARTS,
            ),
            ["anilist:anime:102427"],
        ),
    ],
)
def test_find_covered_part_pages_reads_episodes_and_air_dates(
    page: ProviderPage,
    part_pages: tuple[ProviderPage, ...],
    covered_page_keys: list[str],
) -> None:
    assert [
        part_page.key for part_page in find_covered_part_pages(page, part_pages)
    ] == covered_page_keys
