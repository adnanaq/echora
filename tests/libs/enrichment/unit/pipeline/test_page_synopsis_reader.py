from typing import Any
from unittest.mock import create_autospec

from enrichment.pipeline.api_fetcher import ApiFetcher
from enrichment.pipeline.page_synopsis_reader import PageSynopsisReader
from work_matching import ProviderPage

MAL_PAGE = ProviderPage(
    key="mal:anime:21",
    address="https://myanimelist.net/anime/21",
    titles=("One Piece",),
)


def _api_fetcher(records_by_service: dict, api_errors: dict | None = None) -> Any:
    api_fetcher = create_autospec(ApiFetcher, instance=True)
    api_fetcher.fetch_all_data.return_value = records_by_service
    api_fetcher.api_errors = api_errors or {}
    return api_fetcher


async def test_read_synopsis_fetched_record_gives_synopsis() -> None:
    api_fetcher = _api_fetcher({"mal": {"anime": {"synopsis": "Pirates."}}})

    assert await PageSynopsisReader(api_fetcher).read_synopsis(MAL_PAGE) == "Pirates."
    assert api_fetcher.fetch_all_data.call_args.kwargs == {
        "only_services": ["mal"],
        "fetch_characters": False,
        "fetch_episodes": False,
    }


async def test_read_synopsis_provider_without_helper_sends_nothing() -> None:
    api_fetcher = _api_fetcher({})
    ann_page = ProviderPage(
        key="ann:anime:836",
        address="https://animenewsnetwork.com/encyclopedia/anime.php?id=836",
    )

    assert await PageSynopsisReader(api_fetcher).read_synopsis(ann_page) is None
    api_fetcher.fetch_all_data.assert_not_called()


async def test_read_synopsis_failed_fetch_gives_none() -> None:
    api_fetcher = _api_fetcher({}, api_errors={"mal": "timeout"})

    assert await PageSynopsisReader(api_fetcher).read_synopsis(MAL_PAGE) is None


async def test_read_synopsis_page_without_record_gives_none() -> None:
    assert (
        await PageSynopsisReader(_api_fetcher({"mal": None})).read_synopsis(MAL_PAGE)
        is None
    )
