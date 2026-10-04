import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest
import zendriver
from enrichment.sources.base import cloudflare_challenge
from enrichment.sources.base.cloudflare_challenge import (
    CloudflareChallengeError,
    is_cloudflare_challenge,
    wait_through_challenge,
)

SOURCES_TESTS = Path(__file__).resolve().parents[1]
INTERSTITIAL = (
    SOURCES_TESTS / "anidb" / "fixtures" / "anidb_cloudflare_interstitial.html"
).read_text()
ANIME_PLANET_PAGES = sorted(
    (SOURCES_TESTS / "anime_planet" / "fixtures").glob("ap_char_*.html")
)
CHARACTER = '<html><body><div id="tab_1_pane">Luffy</div></body></html>'
ANTILEECH = "<html><head><title>AniDB AntiLeech</title></head><body></body></html>"
_URL = "https://anidb.net/character/474"


class ScriptedTab(zendriver.Tab):
    def __init__(self, pages: list[str]) -> None:
        self.pages = list(pages)
        self.html = ""

    async def get_content(self, **_: object) -> str:
        self.html = self.pages.pop(0) if len(self.pages) > 1 else self.pages[0]
        return self.html

    async def query_selector(self, selector: str, **_: object) -> zendriver.Tab | None:
        return self if 'id="tab_1_pane"' in self.html else None


class UnreadableTab(ScriptedTab):
    async def get_content(self, **_: object) -> str:
        raise ConnectionError("connection closed")


@pytest.fixture
def short_challenge_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cloudflare_challenge, "CHALLENGE_WAIT_SECONDS", 0.05)
    monkeypatch.setattr(cloudflare_challenge, "_POLL_SECONDS", 0.01)


def test_is_cloudflare_challenge_saved_anidb_interstitial_returns_true() -> None:
    assert is_cloudflare_challenge(INTERSTITIAL)


@pytest.mark.parametrize("page", ANIME_PLANET_PAGES, ids=lambda path: path.name)
def test_is_cloudflare_challenge_saved_anime_planet_page_returns_false(
    page: Path,
) -> None:
    html = page.read_text()

    assert "challenges.cloudflare.com" in html
    assert not is_cloudflare_challenge(html)


async def test_wait_through_challenge_interstitial_clears_by_itself_skips_solving() -> (
    None
):
    tab = ScriptedTab([INTERSTITIAL, INTERSTITIAL, CHARACTER])

    with patch.object(cloudflare_challenge, "verify_cf", autospec=True) as verify_cf:
        read = await wait_through_challenge(
            tab, _URL, site="AniDB", selector="#tab_1_pane"
        )

    assert (read.html, read.ready, read.challenged) == (CHARACTER, True, True)
    verify_cf.assert_not_awaited()


@pytest.mark.usefixtures("short_challenge_wait")
async def test_wait_through_challenge_interstitial_persists_solves_once_returns_page() -> (
    None
):
    tab = ScriptedTab([INTERSTITIAL])

    async def solve(page: ScriptedTab, **_: object) -> None:
        page.pages = [CHARACTER]

    with patch.object(
        cloudflare_challenge, "verify_cf", autospec=True, side_effect=solve
    ) as verify_cf:
        read = await wait_through_challenge(
            tab, _URL, site="AniDB", selector="#tab_1_pane"
        )

    assert (read.html, read.ready, read.challenged) == (CHARACTER, True, True)
    verify_cf.assert_awaited_once()


@pytest.mark.usefixtures("short_challenge_wait")
@pytest.mark.parametrize("solve_error", [None, TimeoutError("checkbox not found")])
async def test_wait_through_challenge_interstitial_never_clears_raises_after_one_solve(
    solve_error: Exception | None, caplog: pytest.LogCaptureFixture
) -> None:
    tab = ScriptedTab([INTERSTITIAL])

    with (
        patch.object(
            cloudflare_challenge, "verify_cf", autospec=True, side_effect=solve_error
        ) as verify_cf,
        pytest.raises(CloudflareChallengeError, match=_URL),
    ):
        async with asyncio.timeout(2):
            await wait_through_challenge(
                tab, _URL, site="AniDB", selector="#tab_1_pane"
            )

    verify_cf.assert_awaited_once()
    assert "AniDB: Cloudflare challenge did not clear" in caplog.text
    assert _URL in caplog.text


async def test_wait_through_challenge_stop_on_page_returns_ready_without_element() -> (
    None
):
    tab = ScriptedTab([ANTILEECH])

    read = await wait_through_challenge(
        tab,
        _URL,
        site="AniDB",
        selector="#tab_1_pane",
        stop_on=lambda html: "<title>AniDB AntiLeech" in html,
    )

    assert (read.html, read.ready, read.challenged) == (ANTILEECH, True, False)


async def test_wait_through_challenge_content_never_arrives_returns_not_ready_at_deadline() -> (
    None
):
    tab = ScriptedTab(["<html><body>Not found</body></html>"])

    async with asyncio.timeout(2):
        read = await wait_through_challenge(
            tab, _URL, site="AniDB", selector="#tab_1_pane", timeout=0.1
        )

    assert (read.html, read.ready) == ("<html><body>Not found</body></html>", False)


async def test_wait_through_challenge_unreadable_page_returns_no_html_at_deadline() -> (
    None
):
    async with asyncio.timeout(2):
        read = await wait_through_challenge(
            UnreadableTab([]), _URL, site="AniDB", selector="#tab_1_pane", timeout=0.1
        )

    assert (read.html, read.ready) == (None, False)
