import pytest
from work_matching.page_details_comparison import (
    MalPageComparison,
    compare_page_with_mal_page,
    get_seed_type,
    is_page_details_matching,
)
from work_matching.provider_page import ProviderPage


@pytest.mark.parametrize(
    ("page_type", "page_year", "expected"),
    [
        ("TV SHORT", 2020, True),
        ("TV", None, True),
        ("MOVIE", 2020, False),
        ("TV", 2019, False),
    ],
)
def test_is_page_details_matching_compares_year_and_seed_type(
    work_entry, page_type: str, page_year: int | None, expected: bool
) -> None:
    page = ProviderPage(
        key="anidb:anime:1",
        address="https://anidb.net/anime/1",
        type=page_type,
        year=page_year,
    )
    entry = work_entry("https://myanimelist.net/anime/1", work_type="TV", year=2020)
    assert is_page_details_matching(page, entry) is expected


@pytest.mark.parametrize(
    ("page_type", "seed_type"),
    [("TV SHORT", "TV"), ("MUSIC", "SPECIAL"), ("UNKNOWN", None), (None, None)],
)
def test_get_seed_type_provider_type_falls_under_seed_type(
    page_type: str | None, seed_type: str | None
) -> None:
    page = ProviderPage(key="mal:anime:1", address="https://x", type=page_type)
    assert get_seed_type(page) == seed_type


def _dated_page(
    key: str,
    page_type: str,
    episode_count: int | None,
    aired_from: str | None = None,
    aired_to: str | None = None,
) -> ProviderPage:
    return ProviderPage(
        key=key,
        address=f"https://{key}",
        type=page_type,
        episode_count=episode_count,
        aired_from=aired_from,
        aired_to=aired_to,
    )


@pytest.mark.parametrize(
    ("page", "mal_page", "expected"),
    [
        pytest.param(
            _dated_page("kitsu:anime:12168", "TV", 230, "2016-04-07", "2022-03-03"),
            _dated_page("mal:anime:32956", "TV", 230, "2016-04-07", "2022-03-03"),
            MalPageComparison.SAME_WORK,
            id="aware-meisaku-kun-kitsu",
        ),
        pytest.param(
            _dated_page("anilist:anime:128872", "TV", 25, "2021-01-09", "2021-06-26"),
            _dated_page("mal:anime:42653", "TV", 24, "2021-01-09", "2021-06-26"),
            MalPageComparison.SAME_WORK,
            id="aikatsu-planet-one-episode-more",
        ),
        pytest.param(
            _dated_page("anisearch:anime:11255", "TV", 39, "2016-04-07", "2017-02-16"),
            _dated_page("mal:anime:32956", "TV", 230, "2016-04-07", "2022-03-03"),
            MalPageComparison.PART_OF_WORK,
            id="aware-meisaku-kun-first-39-episodes",
        ),
        pytest.param(
            _dated_page("anilist:anime:114448", "ONA", 6, "2020-01-22", "2020-01-22"),
            _dated_page("mal:anime:36040", "ONA", 12, "2019-07-19", "2020-01-23"),
            MalPageComparison.PART_OF_WORK,
            id="saint-seiya-part-2",
        ),
        pytest.param(
            _dated_page("anilist:anime:129347", "ONA", 13, "2019-11-22", "2020-02-14"),
            _dated_page("mal:anime:37822", "ONA", 237, "2018-11-30", "2023-06-09"),
            MalPageComparison.PART_OF_WORK,
            id="douluo-dalu-arc",
        ),
        pytest.param(
            _dated_page("anisearch:anime:1296", "OVA", 16, "1994-01-20", "1996-04-24"),
            _dated_page("mal:anime:1802", "OVA", 6, "1994-01-20", "1994-06-20"),
            MalPageComparison.SEVERAL_WORKS,
            id="dirty-pair-flash-three-ova-series",
        ),
        pytest.param(
            _dated_page("anidb:anime:13263", "TV", 25, "2017-10-02", "2018-03-26"),
            _dated_page("mal:anime:36838", "TV", 12, "2018-01-07", "2018-03-25"),
            MalPageComparison.SEVERAL_WORKS,
            id="gintama-2017-two-mal-works",
        ),
        pytest.param(
            _dated_page("anisearch:anime:21707", "TV", 7, "2024-11-23", "2025-01-04"),
            _dated_page("mal:anime:57101", "TV", 13, "2024-11-23", "2025-01-04"),
            MalPageComparison.DETAILS_DIFFER,
            id="1-cho-fewer-episodes-same-dates",
        ),
        pytest.param(
            _dated_page("kitsu:anime:47792", "ONA", 6, "2023-07-22", "2023-07-24"),
            _dated_page("mal:anime:63548", "PV", 1, "2023-07-30"),
            MalPageComparison.DETAILS_DIFFER,
            id="holoadvent-linking-breach",
        ),
        pytest.param(
            _dated_page("anilist:anime:179872", "TV", 7, "2023-10-06", "2023-11-17"),
            _dated_page("mal:anime:58630", "TV", 15, "2024-04-06", "2024-11-16"),
            MalPageComparison.DETAILS_DIFFER,
            id="oshiri-tantei-7-part-2-before-8th-season",
        ),
        pytest.param(
            _dated_page("kitsu:anime:14129", "MOVIE", 1, "2018-11-09", "2018-11-09"),
            _dated_page("mal:anime:36999", "TV", 6),
            MalPageComparison.DETAILS_DIFFER,
            id="zoku-owarimonogatari-movie-against-tv",
        ),
        pytest.param(
            _dated_page("animeplanet:anime:aware-meisaku-kun", "TV", 39, "2016-04-07"),
            _dated_page("mal:anime:32956", "TV", 230, "2016-04-07", "2022-03-03"),
            MalPageComparison.DETAILS_UNKNOWN,
            id="aware-meisaku-kun-anime-planet-end-date-unknown",
        ),
        pytest.param(
            _dated_page("anilist:anime:102416", "MUSIC", 1, "2013-10-22", "2013-10-22"),
            _dated_page("mal:anime:20707", "MUSIC", 1, "2013-09-10"),
            MalPageComparison.DETAILS_DIFFER,
            id="0-music-video-same-count-start-apart",
        ),
        pytest.param(
            _dated_page("anilist:anime:126186", "OVA", 3, "2021-09-14", "2021-09-14"),
            _dated_page("mal:anime:43506", "OVA", 1, "2021-09-14"),
            MalPageComparison.DETAILS_DIFFER,
            id="alice-gear-aegis-more-episodes-same-day",
        ),
        pytest.param(
            _dated_page("anilist:anime:8723", "OVA", 6, "1989-11-21"),
            _dated_page("mal:anime:8723", "OVA", 8, "1992-07-04"),
            MalPageComparison.DETAILS_UNKNOWN,
            id="anime-rakugo-kan-fewer-episodes-mal-end-unknown",
        ),
        pytest.param(
            _dated_page("anilist:anime:18227", "TV SHORT", 6, "2013-04-05"),
            _dated_page("mal:anime:18227", "TV", 3, "2013-04-05"),
            MalPageComparison.DETAILS_UNKNOWN,
            id="bakujuu-gasshin-ziguru-hazeru-more-episodes-ends-unknown",
        ),
        pytest.param(
            _dated_page("kitsu:anime:50804", "TV", None, "2026-07-05"),
            _dated_page("mal:anime:63219", "TV", 12, "2026-07-05", "2026-09-20"),
            MalPageComparison.DETAILS_UNKNOWN,
            id="aware-meisaku-kun-2026-count-unknown",
        ),
    ],
)
def test_compare_page_with_mal_page_reads_type_episode_count_and_dates(
    page: ProviderPage, mal_page: ProviderPage, expected: MalPageComparison
) -> None:
    assert compare_page_with_mal_page(page, mal_page) is expected
