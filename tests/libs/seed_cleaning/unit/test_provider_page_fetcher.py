from typing import Any

from enrichment.pipeline.api_fetcher import ApiFetcher
from seed_cleaning.provider_page_fetcher import FetchOutcome, ProviderPageFetcher
from work_matching import ProviderPageStore, WorkEntry

MAL = "https://myanimelist.net/anime/21"
KITSU = "https://kitsu.app/anime/12"
LIVECHART = "https://livechart.me/anime/321"
SCHEDULE = "https://animeschedule.net/anime/one-piece"


class ScriptedApiFetcher(ApiFetcher):
    def __init__(self, answers: dict[tuple[str, str], dict | Exception | None]) -> None:
        self.answers = answers
        self.requests: list[tuple[str, dict[str, str], dict[str, Any]]] = []
        self.api_errors: dict[str, str] = {}
        self.api_timings: dict[str, float] = {}

    async def fetch_all_data(
        self,
        ids: dict[str, str],
        offline_data: dict[str, Any],
        temp_dir: str | None = None,
        skip_services: list[str] | None = None,
        only_services: list[str] | None = None,
        *,
        fetch_characters: bool = True,
        fetch_episodes: bool = True,
    ) -> dict[str, Any]:
        [service] = only_services or []
        self.requests.append((service, ids, offline_data))
        self.api_errors = {}
        answer = self.answers.get((service, next(iter(ids.values()))))
        if isinstance(answer, Exception):
            self.api_errors[service] = str(answer)
            return {service: None}
        return {service: None if answer is None else {"anime": answer}}


def _one_piece(*addresses: str) -> WorkEntry:
    return WorkEntry.from_seed_entry(
        {"title": "One Piece", "type": "TV", "sources": list(addresses)}
    )


def _fetcher(tmp_path, answers) -> tuple[ProviderPageFetcher, ScriptedApiFetcher]:
    api_fetcher = ScriptedApiFetcher(answers)
    store = ProviderPageStore(tmp_path / "pages.jsonl")
    return (
        ProviderPageFetcher(
            lambda: api_fetcher, store, seconds_between_found_page_lookups=0
        ),
        api_fetcher,
    )


async def test_fetch_entry_pages_record_stored_as_page_with_links(tmp_path) -> None:
    fetcher, _ = _fetcher(
        tmp_path,
        {("kitsu", KITSU): {"title": "One Piece", "sources": [KITSU, MAL]}},
    )
    summary = await fetcher.fetch_entry_pages([_one_piece(KITSU)])
    page = ProviderPageStore(tmp_path / "pages.jsonl").get_page("kitsu:anime:12")
    assert page is not None
    assert (page.titles, page.links) == (("One Piece",), ("mal:anime:21",))
    assert summary.outcomes == {FetchOutcome.FETCHED: 1}


async def test_fetch_entry_pages_missing_page_stored_as_dead(tmp_path) -> None:
    fetcher, _ = _fetcher(tmp_path, {})
    summary = await fetcher.fetch_entry_pages([_one_piece(KITSU)])
    page = ProviderPageStore(tmp_path / "pages.jsonl").get_page("kitsu:anime:12")
    assert page is not None
    assert page.dead
    assert summary.outcomes == {FetchOutcome.DEAD: 1}


async def test_fetch_entry_pages_failed_fetch_not_stored_and_reported(
    tmp_path,
) -> None:
    fetcher, _ = _fetcher(tmp_path, {("kitsu", KITSU): RuntimeError("HTTP 503")})
    summary = await fetcher.fetch_entry_pages([_one_piece(KITSU)])
    assert "kitsu:anime:12" not in ProviderPageStore(tmp_path / "pages.jsonl")
    assert summary.outcomes == {FetchOutcome.FAILED: 1}
    assert summary.failed_pages == {"kitsu:anime:12": "HTTP 503"}


async def test_fetch_entry_pages_stored_page_not_fetched_again(tmp_path) -> None:
    fetcher, api_fetcher = _fetcher(tmp_path, {})
    await fetcher.fetch_entry_pages([_one_piece(KITSU)])
    summary = await fetcher.fetch_entry_pages([_one_piece(KITSU)])
    assert len(api_fetcher.requests) == 1
    assert summary.outcomes == {FetchOutcome.ALREADY_STORED: 1}


async def test_fetch_entry_pages_provider_without_helper_skipped(tmp_path) -> None:
    fetcher, api_fetcher = _fetcher(tmp_path, {})
    summary = await fetcher.fetch_entry_pages([_one_piece(LIVECHART)])
    assert api_fetcher.requests == []
    assert summary.outcomes == {}


async def test_fetch_entry_pages_animeschedule_found_through_mal_page(
    tmp_path,
) -> None:
    fetcher, api_fetcher = _fetcher(
        tmp_path,
        {
            ("mal", MAL): {"title": "One Piece", "sources": [MAL]},
            ("animeschedule", MAL): {"title": "One Piece", "sources": [SCHEDULE, MAL]},
        },
    )
    await fetcher.fetch_entry_pages([_one_piece(MAL, KITSU)])
    found_page = ProviderPageStore(tmp_path / "pages.jsonl").get_page(SCHEDULE)
    assert found_page is not None
    assert found_page.found_by == ("mal:anime:21",)
    assert (
        "animeschedule",
        {"mal_url": MAL},
        {"title": "One Piece", "sources": [MAL]},
    ) in (api_fetcher.requests)


async def test_fetch_entry_pages_animeschedule_lookup_done_not_repeated(
    tmp_path,
) -> None:
    fetcher, api_fetcher = _fetcher(
        tmp_path,
        {("animeschedule", MAL): {"title": "One Piece", "sources": [SCHEDULE, MAL]}},
    )
    await fetcher.fetch_entry_pages([_one_piece(MAL)])
    requests_before = len(api_fetcher.requests)
    await fetcher.fetch_entry_pages([_one_piece(MAL)])
    assert len(api_fetcher.requests) == requests_before


async def test_fetch_entry_pages_animeschedule_without_anime_stores_nothing(
    tmp_path,
) -> None:
    fetcher, _ = _fetcher(tmp_path, {("mal", MAL): {"title": "One Piece"}})
    summary = await fetcher.fetch_entry_pages([_one_piece(MAL)])
    assert len(ProviderPageStore(tmp_path / "pages.jsonl")) == 1
    assert summary.outcomes == {
        FetchOutcome.FETCHED: 1,
        FetchOutcome.NOT_LISTED: 1,
    }


async def test_fetch_entry_pages_page_in_two_entries_fetched_once(tmp_path) -> None:
    fetcher, api_fetcher = _fetcher(tmp_path, {})
    await fetcher.fetch_entry_pages([_one_piece(KITSU), _one_piece(KITSU)])
    assert len(api_fetcher.requests) == 1


async def test_fetch_entry_pages_entry_with_two_mal_pages_looks_up_animeschedule_twice(
    tmp_path,
) -> None:
    other_mal = "https://myanimelist.net/anime/22"
    fetcher, api_fetcher = _fetcher(tmp_path, {})
    await fetcher.fetch_entry_pages([_one_piece(MAL, other_mal)])
    lookups = [
        ids for service, ids, _ in api_fetcher.requests if service == "animeschedule"
    ]
    assert lookups == [{"mal_url": MAL}, {"mal_url": other_mal}]


class KitsuRaisingApiFetcher(ScriptedApiFetcher):
    async def fetch_all_data(
        self,
        ids: dict[str, str],
        offline_data: dict[str, Any],
        temp_dir: str | None = None,
        skip_services: list[str] | None = None,
        only_services: list[str] | None = None,
        *,
        fetch_characters: bool = True,
        fetch_episodes: bool = True,
    ) -> dict[str, Any]:
        if only_services == ["kitsu"]:
            raise RuntimeError("store disk full")
        return await super().fetch_all_data(
            ids, offline_data, temp_dir, skip_services, only_services
        )


async def test_fetch_entry_pages_unexpected_error_fails_page_and_other_queues_finish(
    tmp_path,
) -> None:
    api_fetcher = KitsuRaisingApiFetcher({("mal", MAL): {"title": "One Piece"}})
    fetcher = ProviderPageFetcher(
        lambda: api_fetcher,
        ProviderPageStore(tmp_path / "pages.jsonl"),
        seconds_between_found_page_lookups=0,
    )
    summary = await fetcher.fetch_entry_pages([_one_piece(MAL, KITSU)])
    assert summary.failed_pages == {"kitsu:anime:12": "RuntimeError: store disk full"}
    assert "mal:anime:21" in ProviderPageStore(tmp_path / "pages.jsonl")


async def test_fetch_entry_pages_animeschedule_lookup_finding_nothing_repeated(
    tmp_path,
) -> None:
    fetcher, api_fetcher = _fetcher(tmp_path, {})
    await fetcher.fetch_entry_pages([_one_piece(MAL)])
    await fetcher.fetch_entry_pages([_one_piece(MAL)])
    lookups = [
        ids for service, ids, _ in api_fetcher.requests if service == "animeschedule"
    ]
    assert lookups == [{"mal_url": MAL}, {"mal_url": MAL}]
