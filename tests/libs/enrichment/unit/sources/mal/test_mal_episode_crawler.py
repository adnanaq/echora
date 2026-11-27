import sys
from unittest.mock import call, create_autospec, patch

import pytest
import zendriver
from enrichment.sources.base.framework import NullRepository
from enrichment.sources.mal import mal_episode_crawler
from enrichment.sources.mal.mal_episode_crawler import (
    MalEpisodeCrawler,
    _fetch_episode_html,
    _fetch_mal_episode_data,
    fetch_mal_episode,
    fetch_mal_episodes,
    main,
)
from enrichment.sources.mal.mal_episode_parser import MAL_EPISODE_XPATHS

pytestmark = pytest.mark.asyncio

_EP1_URL = "https://myanimelist.net/anime/21/One_Piece/episode/1"
_EP50_URL = "https://myanimelist.net/anime/21/One_Piece/episode/50"
_EP1152_URL = "https://myanimelist.net/anime/21/One_Piece/episode/1152"
_REDIRECT_URL = "https://myanimelist.net/anime/21/redirect"
_LUFFY_TITLE = "I'm Luffy! The Man Who's Gonna Be King of the Pirates!"
_PAGE_WITHOUT_EPISODE_TITLE = '<html><body><h2 class="fs18">Episode</h2></body></html>'


@pytest.fixture
def episode_browser(open_browser):
    return lambda browser: open_browser(mal_episode_crawler, browser)


@pytest.fixture
def cached():
    def install(values: list[dict | None], missing: list[int]):
        return patch.object(
            _fetch_mal_episode_data,
            "cache_batch_get",
            autospec=True,
            return_value=(values, missing),
        )

    return install


@pytest.fixture
def cache_writes():
    with patch.object(
        _fetch_mal_episode_data, "cache_batch_set", autospec=True
    ) as cache_batch_set:
        yield cache_batch_set


def _browser_redirecting(tab_showing, html: str, final_url: str) -> zendriver.Browser:
    browser = create_autospec(zendriver.Browser, instance=True)
    browser.get.return_value = tab_showing(html, final_url)
    return browser


def _episode_page(url: str) -> str:
    number = url.rstrip("/").split("/")[-1]
    return f"""<html><body>
    <h2 class="fs18">#{number} - Title</h2>
    <div class="di-tc ar"><p>Aired: Oct 20, 1999</p></div>
    </body></html>"""


async def test_fetch_episode_html_episode_page_returns_content_and_address(
    browser_serving, mal_episode_html
) -> None:
    browser = browser_serving({_EP1_URL: mal_episode_html})

    assert await _fetch_episode_html(browser, _EP1_URL) == (mal_episode_html, _EP1_URL)


async def test_fetch_episode_html_page_without_title_returns_none(
    browser_serving,
) -> None:
    browser = browser_serving({_EP1_URL: "<html><body></body></html>"})

    assert await _fetch_episode_html(browser, _EP1_URL) is None


async def test_fetch_mal_episode_data_episode_page_returns_raw_with_address(
    episode_browser, browser_serving, mal_episode_html
) -> None:
    with episode_browser(browser_serving({_EP1_URL: mal_episode_html})):
        raw = await _fetch_mal_episode_data(_EP1_URL)

    assert raw is not None
    assert raw["title_header"].startswith("#1")
    assert raw["_url"] == _EP1_URL


@pytest.mark.parametrize("html", ["", _PAGE_WITHOUT_EPISODE_TITLE])
async def test_fetch_mal_episode_data_unreadable_page_returns_none(
    episode_browser, browser_serving, html: str
) -> None:
    with episode_browser(browser_serving({_EP1_URL: html})):
        assert await _fetch_mal_episode_data(_EP1_URL) is None


def test_mal_episode_crawler_get_extraction_schema_returns_xpaths() -> None:
    assert MalEpisodeCrawler(NullRepository()).get_extraction_schema() == {
        "xpaths": MAL_EPISODE_XPATHS
    }


def test_mal_episode_crawler_normalize_identifier_returns_url_unchanged() -> None:
    assert MalEpisodeCrawler(NullRepository()).normalize_identifier("foo") == "foo"


async def test_mal_episode_crawler_fetch_raw_data_returns_page_data(
    episode_browser, browser_serving, mal_episode_html
) -> None:
    with episode_browser(browser_serving({_EP1_URL: mal_episode_html})):
        raw = await MalEpisodeCrawler(NullRepository()).fetch_raw_data(_EP1_URL)

    assert raw is not None
    assert raw["_url"] == _EP1_URL


def test_mal_episode_crawler_map_to_canonical_returns_episode_title(
    mal_episode_extracted,
) -> None:
    crawler = MalEpisodeCrawler(NullRepository())
    episode = crawler.build_source_model(dict(mal_episode_extracted), _EP1_URL)

    assert episode.episode_number == 1
    assert crawler.map_to_canonical(episode)["title"] == _LUFFY_TITLE


async def test_fetch_mal_episode_episode_page_returns_canonical_episode(
    episode_browser, browser_serving, mal_episode_html
) -> None:
    with episode_browser(browser_serving({_EP1_URL: mal_episode_html})):
        episode = await fetch_mal_episode(_EP1_URL)

    assert episode is not None
    assert episode["title"] == _LUFFY_TITLE


async def test_fetch_mal_episodes_without_urls_returns_empty_list() -> None:
    assert await fetch_mal_episodes([]) == []


async def test_fetch_mal_episodes_all_cached_returns_episodes_without_fetching(
    episode_browser,
    browser_serving,
    cached,
    mal_episode_extracted,
    mal_episode_filler_extracted,
) -> None:
    raw_values = [
        {**mal_episode_extracted, "_url": _EP1_URL},
        {**mal_episode_filler_extracted, "_url": _EP50_URL},
    ]
    with cached(raw_values, []), episode_browser(browser_serving({})):
        episodes = await fetch_mal_episodes([_EP1_URL, _EP50_URL])

    assert [episode["title"] for episode in episodes] == [
        _LUFFY_TITLE,
        "Usopp vs. Daddy the Parent! Showdown at High!",
    ]
    assert episodes[1]["is_filler"] is True


async def test_fetch_mal_episodes_cache_miss_fetches_page_and_caches_raw(
    episode_browser,
    browser_serving,
    cached,
    cache_writes,
    mal_episode_extracted,
    mal_episode_filler_html,
) -> None:
    with (
        cached([{**mal_episode_extracted, "_url": _EP1_URL}, None], [1]),
        episode_browser(browser_serving({_EP50_URL: mal_episode_filler_html})),
    ):
        episodes = await fetch_mal_episodes([_EP1_URL, _EP50_URL])

    assert episodes[0]["title"] == _LUFFY_TITLE
    assert episodes[1]["is_filler"] is True
    [written] = cache_writes.await_args_list
    urls, raw_values = written.args
    assert urls == [_EP50_URL]
    assert raw_values[0]["title_header"].startswith("#50")


async def test_fetch_mal_episodes_several_misses_wait_between_requests(
    episode_browser, browser_serving, cached, cache_writes
) -> None:
    pages = {url: _episode_page(url) for url in (_EP1_URL, _EP50_URL)}
    with (
        cached([None, None], [0, 1]),
        episode_browser(browser_serving(pages)),
        patch.object(mal_episode_crawler.asyncio, "sleep", autospec=True) as sleep,
    ):
        await fetch_mal_episodes([_EP1_URL, _EP50_URL])

    assert sleep.await_args_list == [call(mal_episode_crawler._INTER_REQUEST_DELAY)]


async def test_fetch_mal_episodes_cached_raw_without_episode_number_refetches_page(
    episode_browser, browser_serving, cached, cache_writes, mal_episode_filler_html
) -> None:
    unusable = {"_url": _REDIRECT_URL, "title_header": "#1 - X"}
    with (
        cached([unusable], []),
        episode_browser(browser_serving({_EP50_URL: mal_episode_filler_html})),
    ):
        episodes = await fetch_mal_episodes([_EP50_URL])

    assert episodes[0] is not None
    assert episodes[0]["is_filler"] is True


@pytest.mark.parametrize("html", ["", _PAGE_WITHOUT_EPISODE_TITLE])
async def test_fetch_mal_episodes_unreadable_page_returns_none(
    episode_browser, browser_serving, cached, cache_writes, html: str
) -> None:
    with cached([None], [0]), episode_browser(browser_serving({_EP1_URL: html})):
        assert await fetch_mal_episodes([_EP1_URL]) == [None]


async def test_fetch_mal_episodes_redirect_without_episode_number_returns_none(
    episode_browser, tab_showing, cached, cache_writes, mal_episode_filler_html
) -> None:
    browser = _browser_redirecting(tab_showing, mal_episode_filler_html, _REDIRECT_URL)
    with cached([None], [0]), episode_browser(browser):
        assert await fetch_mal_episodes([_EP50_URL]) == [None]


async def test_fetch_mal_episodes_episode_without_synopsis_returns_no_synopsis(
    episode_browser,
    browser_serving,
    cached,
    cache_writes,
    mal_episode_no_synopsis_html,
) -> None:
    with (
        cached([None], [0]),
        episode_browser(browser_serving({_EP1152_URL: mal_episode_no_synopsis_html})),
    ):
        episodes = await fetch_mal_episodes([_EP1152_URL])

    assert episodes[0] is not None
    assert episodes[0].get("synopsis") is None


@pytest.mark.parametrize(("episode", "exit_code"), [({"title": "Test"}, 0), (None, 1)])
async def test_main_fetched_episode_returns_exit_code(
    tmp_path, episode: dict | None, exit_code: int
) -> None:
    arguments = ["mal_episode_crawler", _EP1_URL, "--output", str(tmp_path / "out")]
    with (
        patch.object(sys, "argv", arguments),
        patch.object(
            mal_episode_crawler,
            "fetch_mal_episode",
            autospec=True,
            return_value=episode,
        ),
    ):
        assert await main() == exit_code
