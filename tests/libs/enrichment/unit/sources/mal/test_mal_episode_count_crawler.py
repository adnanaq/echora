import pytest
from enrichment.sources.mal import mal_episode_count_crawler
from enrichment.sources.mal.mal_episode_count_crawler import (
    _EPISODE_COUNT_XPATH,
    _extract_episode_count,
    _fetch_episode_count_data,
    _fetch_episode_count_html,
    fetch_mal_episode_count,
)

pytestmark = pytest.mark.asyncio

ANIME_URL = "https://myanimelist.net/anime/21/One_Piece"
EPISODE_LIST_URL = f"{ANIME_URL}/episode"
_PAGE_WITHOUT_COUNTER = (
    "<html><body><h2 class='h2_overwrite'>Episodes</h2></body></html>"
)


def _episode_list_page(counter: str) -> str:
    return f"""<html><body>
  <h2 class="h2_overwrite">Episodes</h2>
  <span class="di-ib pl4 fw-n fs10">{counter}</span>
</body></html>"""


@pytest.fixture
def episode_list_browser(open_browser, browser_serving):
    def serve(html: str):
        return open_browser(
            mal_episode_count_crawler, browser_serving({EPISODE_LIST_URL: html})
        )

    return serve


def test_episode_count_xpath_targets_counter_after_episodes_heading() -> None:
    assert "h2_overwrite" in _EPISODE_COUNT_XPATH
    assert "Episodes" in _EPISODE_COUNT_XPATH
    assert "following-sibling::span" in _EPISODE_COUNT_XPATH


@pytest.mark.parametrize("counter", ["(1,155/Unknown)", "(26/26)"])
def test_extract_episode_count_episode_list_page_returns_counter(counter: str) -> None:
    assert _extract_episode_count(_episode_list_page(counter)) == counter


@pytest.mark.parametrize(
    "html",
    [
        "",
        _PAGE_WITHOUT_COUNTER,
        '<?xml version="1.0" encoding="utf-8"?><html><body><p>x</p></body></html>',
    ],
)
def test_extract_episode_count_page_without_counter_returns_none(html: str) -> None:
    assert _extract_episode_count(html) is None


async def test_fetch_episode_count_html_episode_list_page_returns_content(
    episode_list_browser,
) -> None:
    html = _episode_list_page("(1,155/Unknown)")

    with episode_list_browser(html):
        assert await _fetch_episode_count_html(EPISODE_LIST_URL) == html


async def test_fetch_episode_count_html_page_without_heading_returns_none(
    episode_list_browser,
) -> None:
    with episode_list_browser("<html><body><p>no heading</p></body></html>"):
        assert await _fetch_episode_count_html(EPISODE_LIST_URL) is None


async def test_fetch_episode_count_data_episode_list_page_returns_counter(
    episode_list_browser,
) -> None:
    with episode_list_browser(_episode_list_page("(1,155/Unknown)")):
        assert await _fetch_episode_count_data(EPISODE_LIST_URL) == "(1,155/Unknown)"


@pytest.mark.parametrize(
    "html", ["<html><body><p>no heading</p></body></html>", _PAGE_WITHOUT_COUNTER]
)
async def test_fetch_episode_count_data_unreadable_page_returns_none(
    episode_list_browser, html: str
) -> None:
    with episode_list_browser(html):
        assert await _fetch_episode_count_data(EPISODE_LIST_URL) is None


@pytest.mark.parametrize(
    ("counter", "expected"), [("(12/12)", 12), ("(1,155/Unknown)", 1155)]
)
async def test_fetch_mal_episode_count_episode_list_counter_returns_aired_count(
    episode_list_browser, counter: str, expected: int
) -> None:
    with episode_list_browser(_episode_list_page(counter)):
        assert await fetch_mal_episode_count(ANIME_URL) == expected


@pytest.mark.parametrize(
    "html", [_PAGE_WITHOUT_COUNTER, _episode_list_page("(?/Unknown)")]
)
async def test_fetch_mal_episode_count_without_readable_counter_returns_zero(
    episode_list_browser, html: str
) -> None:
    with episode_list_browser(html):
        assert await fetch_mal_episode_count(ANIME_URL) == 0
