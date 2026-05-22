import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from enrichment.sources.mal.mal_helper import MalHelper

# ---------------------------------------------------------------------------
# MalHelper — unit tests (crawler-based implementation)
# ---------------------------------------------------------------------------

_ANIME_URL = "https://myanimelist.net/anime/21/One_Piece"
_CHAR_URL = "https://myanimelist.net/character/40/Monkey_D_Luffy"


@pytest.mark.asyncio
async def test_context_manager_protocol() -> None:
    """async with MalHelper covers close/__aenter__/__aexit__."""
    async with MalHelper() as helper:
        assert isinstance(helper, MalHelper)


@pytest.mark.asyncio
async def test_fetch_anime_returns_data_payload():
    """_fetch_anime delegates to fetch_mal_anime and returns canonical dict."""
    mapped = {
        "mal_id": 1,
        "title": "X",
        "synopsis": None,
        "episode_count": 12,
        "sources": [_ANIME_URL],
    }

    with patch(
        "enrichment.sources.mal.mal_helper.fetch_mal_anime",
        new=AsyncMock(return_value=mapped),
    ):
        data = await MalHelper()._fetch_anime(_ANIME_URL)

    assert data is not None
    assert data["mal_id"] == 1
    assert data["title"] == "X"
    assert "synopsis" in data


@pytest.mark.asyncio
async def test_fetch_anime_resolves_unknown_episode_count():
    """_fetch_anime patches episode_count via fetch_mal_episode_count when anime page returns 0."""
    mapped = {
        "mal_id": 21,
        "title": "One Piece",
        "episode_count": 0,
    }

    with (
        patch(
            "enrichment.sources.mal.mal_helper.fetch_mal_anime",
            new=AsyncMock(return_value=mapped),
        ),
        patch(
            "enrichment.sources.mal.mal_helper.fetch_mal_episode_count",
            new=AsyncMock(return_value=1155),
        ),
    ):
        data = await MalHelper()._fetch_anime(_ANIME_URL)

    assert data["episode_count"] == 1155


@pytest.mark.asyncio
async def test_fetch_anime_skips_episode_count_resolution_when_known():
    """_fetch_anime does not call fetch_mal_episode_count when episode_count is already set."""
    mapped = {
        "mal_id": 57334,
        "title": "Dandadan",
        "episode_count": 12,
        "sources": ["https://myanimelist.net/anime/57334/Dandadan"],
    }
    mock_count = AsyncMock(return_value=99)

    with (
        patch(
            "enrichment.sources.mal.mal_helper.fetch_mal_anime",
            new=AsyncMock(return_value=mapped),
        ),
        patch(
            "enrichment.sources.mal.mal_helper.fetch_mal_episode_count", new=mock_count
        ),
    ):
        data = await MalHelper()._fetch_anime(
            "https://myanimelist.net/anime/57334/Dandadan"
        )

    mock_count.assert_not_awaited()
    assert data["episode_count"] == 12


@pytest.mark.asyncio
async def test_fetch_anime_returns_none_on_crawler_failure():
    """_fetch_anime returns None when the crawler returns None."""
    with patch(
        "enrichment.sources.mal.mal_helper.fetch_mal_anime",
        new=AsyncMock(return_value=None),
    ):
        data = await MalHelper()._fetch_anime(_ANIME_URL)

    assert data is None


@pytest.mark.asyncio
async def test_fetch_character_urls_returns_list():
    """_fetch_character_urls delegates to fetch_mal_character_refs and returns URLs."""
    urls = [
        "https://myanimelist.net/character/40/Monkey_D_Luffy",
        "https://myanimelist.net/character/62/Roronoa_Zoro",
    ]

    with patch(
        "enrichment.sources.mal.mal_helper.fetch_mal_character_refs",
        new=AsyncMock(return_value=urls),
    ):
        items = await MalHelper()._fetch_character_urls(_ANIME_URL)

    assert len(items) == 2
    assert items[0] == "https://myanimelist.net/character/40/Monkey_D_Luffy"
    assert items[1] == "https://myanimelist.net/character/62/Roronoa_Zoro"


@pytest.mark.asyncio
async def test_fetch_character_urls_returns_empty_on_failure():
    """_fetch_character_urls returns [] when crawler returns empty list."""
    with patch(
        "enrichment.sources.mal.mal_helper.fetch_mal_character_refs",
        new=AsyncMock(return_value=[]),
    ):
        items = await MalHelper()._fetch_character_urls(_ANIME_URL)

    assert items == []


@pytest.mark.asyncio
async def test_fetch_episodes_maps_minimal_fields(tmp_path: Path):
    """_fetch_episodes uses fetch_mal_episodes; crawler returns canonical dicts directly."""
    ep1 = {"mal_id": 1, "title": "E1", "episode_number": 1}
    ep2 = {"mal_id": 2, "title": "E2", "episode_number": 2}

    out = tmp_path / "eps.jsonl"
    with patch(
        "enrichment.sources.mal.mal_helper.fetch_mal_episodes",
        new=AsyncMock(return_value=[ep1, ep2]),
    ) as mock_fetch:
        episodes = await MalHelper()._fetch_episodes(
            _ANIME_URL, 2, output_path=str(out)
        )

    assert len(episodes) == 2
    assert episodes[0]["mal_id"] == 1
    assert episodes[0]["title"] == "E1"
    assert episodes[0]["episode_number"] == 1
    assert episodes[1]["mal_id"] == 2
    assert episodes[1]["episode_number"] == 2
    mock_fetch.assert_awaited_once_with(
        [f"{_ANIME_URL}/episode/1", f"{_ANIME_URL}/episode/2"],
        output_path=str(out),
    )


@pytest.mark.asyncio
async def test_fetch_episodes_skips_none_results():
    """_fetch_episodes skips None entries returned by the batch crawler (failed episodes)."""
    ep1 = {"mal_id": 1, "title": "E1", "episode_number": 1}

    with patch(
        "enrichment.sources.mal.mal_helper.fetch_mal_episodes",
        new=AsyncMock(return_value=[ep1, None]),
    ):
        episodes = await MalHelper()._fetch_episodes(_ANIME_URL, 2)

    assert len(episodes) == 1
    assert episodes[0]["mal_id"] == 1


@pytest.mark.asyncio
async def test_fetch_episodes_returns_empty_when_count_is_zero():
    """_fetch_episodes returns [] immediately when episode_count is 0."""
    assert await MalHelper()._fetch_episodes(_ANIME_URL, 0) == []


@pytest.mark.asyncio
async def test_fetch_character_returns_mapped_data():
    """_fetch_character delegates to fetch_mal_character and returns canonical dict."""
    mapped = {
        "name": "Luffy",
        "sources": [_CHAR_URL],
    }

    with patch(
        "enrichment.sources.mal.mal_helper.fetch_mal_character",
        new=AsyncMock(return_value=mapped),
    ):
        result = await MalHelper()._fetch_character(_CHAR_URL)

    assert result is not None
    assert result["name"] == "Luffy"


@pytest.mark.asyncio
async def test_fetch_character_returns_none_on_crawler_failure():
    """_fetch_character returns None when crawler returns None."""
    with patch(
        "enrichment.sources.mal.mal_helper.fetch_mal_character",
        new=AsyncMock(return_value=None),
    ):
        result = await MalHelper()._fetch_character(_CHAR_URL)

    assert result is None


@pytest.mark.asyncio
async def test_fetch_character_with_output_path_passes_path_to_crawler(tmp_path: Path):
    """_fetch_character passes output_path to fetch_mal_character (FileRepository handles write)."""
    out = tmp_path / "chars.jsonl"
    mock_crawler = AsyncMock(return_value={"name": "Luffy", "sources": [_CHAR_URL]})

    with patch(
        "enrichment.sources.mal.mal_helper.fetch_mal_character", new=mock_crawler
    ):
        await MalHelper()._fetch_character(_CHAR_URL, output_path=str(out))

    mock_crawler.assert_awaited_once_with(_CHAR_URL, output_path=str(out))


@pytest.mark.asyncio
async def test_fetch_characters_returns_full_character_data(tmp_path: Path):
    """_fetch_characters fetches refs then batch-fetches details."""
    char_urls = ["https://myanimelist.net/character/100/A"]
    mapped = {
        "name": "A",
        "role": "MAIN",
        "sources": ["https://myanimelist.net/character/100/A"],
    }
    out = tmp_path / "chars.jsonl"

    with (
        patch(
            "enrichment.sources.mal.mal_helper.fetch_mal_character_refs",
            new=AsyncMock(return_value=char_urls),
        ),
        patch(
            "enrichment.sources.mal.mal_helper.fetch_mal_characters",
            new=AsyncMock(return_value=[mapped]),
        ) as mock_fetch,
    ):
        chars = await MalHelper()._fetch_characters(_ANIME_URL, output_path=str(out))
    mock_fetch.assert_awaited_once_with(char_urls, output_path=str(out))

    assert len(chars) == 1
    assert chars[0]["name"] == "A"


@pytest.mark.asyncio
async def test_fetch_characters_returns_empty_when_no_refs():
    """_fetch_characters returns [] when no character refs are found."""
    with patch(
        "enrichment.sources.mal.mal_helper.fetch_mal_character_refs",
        new=AsyncMock(return_value=[]),
    ):
        chars = await MalHelper()._fetch_characters(_ANIME_URL)

    assert chars == []


@pytest.mark.asyncio
async def test_fetch_characters_skips_failed_fetches():
    """_fetch_characters skips characters whose batch fetch returns None."""
    char_urls = [
        "https://myanimelist.net/character/100/A",
        "https://myanimelist.net/character/101/B",
    ]

    with (
        patch(
            "enrichment.sources.mal.mal_helper.fetch_mal_character_refs",
            new=AsyncMock(return_value=char_urls),
        ),
        patch(
            "enrichment.sources.mal.mal_helper.fetch_mal_characters",
            new=AsyncMock(return_value=[None, None]),
        ),
    ):
        chars = await MalHelper()._fetch_characters(_ANIME_URL)

    assert len(chars) == 0


@pytest.mark.asyncio
async def test_fetch_all_standardized(tmp_path: Path):
    """fetch_all orchestrates anime + episodes + characters; extracts canonical URL from sources."""
    ids = {"mal_url": _ANIME_URL}

    helper = MalHelper()
    helper._fetch_anime = AsyncMock(
        return_value={"mal_id": 1, "episode_count": 2, "title": "Test"}
    )
    helper._fetch_episodes = AsyncMock(
        return_value=[{"episode_number": 1}, {"episode_number": 2}]
    )
    helper._fetch_characters = AsyncMock(return_value=[{"mal_id": 10, "name": "A"}])

    result = await helper.fetch_all(ids, {}, temp_dir=str(tmp_path))

    assert result is not None
    assert result["anime"]["mal_id"] == 1
    assert len(result["episodes"]) == 2
    assert result["characters"] == [{"mal_id": 10, "name": "A"}]
    helper._fetch_anime.assert_awaited_once_with(
        _ANIME_URL, output_path=str(tmp_path / "mal_anime.jsonl")
    )
    helper._fetch_episodes.assert_awaited_once_with(
        _ANIME_URL, 2, output_path=str(tmp_path / "mal_episodes.jsonl")
    )
    helper._fetch_characters.assert_awaited_once_with(
        _ANIME_URL, output_path=str(tmp_path / "mal_characters.jsonl")
    )


@pytest.mark.asyncio
async def test_fetch_all_passes_episode_count_from_anime_page():
    """fetch_all passes episode_count from the anime page to _fetch_episodes."""
    ids = {"mal_url": _ANIME_URL}
    helper = MalHelper()
    helper._fetch_anime = AsyncMock(
        return_value={"mal_id": 1, "episode_count": 12, "title": "Test"}
    )
    helper._fetch_episodes = AsyncMock(return_value=[{"episode_number": 1}])
    helper._fetch_characters = AsyncMock(return_value=[])

    result = await helper.fetch_all(ids, {})

    assert result is not None
    assert result["episodes"] == [{"episode_number": 1}]
    helper._fetch_episodes.assert_awaited_once_with(_ANIME_URL, 12, output_path=None)


@pytest.mark.asyncio
async def test_fetch_all_returns_empty_when_detailed_empty():
    """fetch_all returns empty characters when detailed fetch returns empty."""
    ids = {"mal_url": _ANIME_URL}
    helper = MalHelper()
    helper._fetch_anime = AsyncMock(
        return_value={"mal_id": 1, "episode_count": None, "title": "Test"}
    )
    helper._fetch_episodes = AsyncMock(return_value=[])
    helper._fetch_characters = AsyncMock(return_value=[])

    result = await helper.fetch_all(ids, {})

    assert result is not None
    assert result["characters"] == []


@pytest.mark.asyncio
async def test_fetch_all_continues_when_episodes_raise():
    """fetch_all continues with empty episodes when episode fetch raises."""
    ids = {"mal_url": _ANIME_URL}
    helper = MalHelper()
    helper._fetch_anime = AsyncMock(
        return_value={"mal_id": 1, "episode_count": 2, "title": "Test"}
    )
    helper._fetch_episodes = AsyncMock(side_effect=Exception("network error"))
    helper._fetch_characters = AsyncMock(return_value=[])

    result = await helper.fetch_all(ids, {})

    assert result is not None
    assert result["episodes"] == []


@pytest.mark.asyncio
async def test_fetch_all_continues_when_characters_raise():
    """fetch_all continues with empty characters when character fetch raises."""
    ids = {"mal_url": _ANIME_URL}
    helper = MalHelper()
    helper._fetch_anime = AsyncMock(
        return_value={"mal_id": 1, "episode_count": 0, "title": "Test"}
    )
    helper._fetch_episodes = AsyncMock(return_value=[])
    helper._fetch_characters = AsyncMock(side_effect=Exception("network error"))

    result = await helper.fetch_all(ids, {})

    assert result is not None
    assert result["characters"] == []


@pytest.mark.asyncio
async def test_fetch_all_returns_none_when_anime_fails():
    """fetch_all returns None when the anime fetch fails."""
    ids = {"mal_url": _ANIME_URL}
    helper = MalHelper()
    helper._fetch_anime = AsyncMock(return_value=None)

    result = await helper.fetch_all(ids, {})

    assert result is None


@pytest.mark.asyncio
async def test_fetch_all_returns_none_when_mal_url_missing():
    """fetch_all returns None when ids has no mal_url key."""
    result = await MalHelper().fetch_all({}, {})
    assert result is None


@pytest.mark.asyncio
async def test_fetch_all_returns_none_on_invalid_url():
    """fetch_all returns None when mal_url is not a valid MAL anime URL."""
    result = await MalHelper().fetch_all({"mal_url": "https://example.com/not-mal"}, {})
    assert result is None


@pytest.mark.asyncio
async def test_fetch_all_returns_none_on_slugless_url():
    """fetch_all returns None when mal_url has no slug (numeric-only URL)."""
    result = await MalHelper().fetch_all(
        {"mal_url": "https://myanimelist.net/anime/21"}, {}
    )
    assert result is None


# ---------------------------------------------------------------------------
# CLI — unit tests
# ---------------------------------------------------------------------------


class TestMALHelperCli:
    @pytest.mark.asyncio
    async def test_main_anime_writes_output_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "anime.json"

            helper = AsyncMock()
            helper.fetch_all = AsyncMock(
                return_value={"anime": {"mal_id": 1, "title": "X"}}
            )

            with (
                patch(
                    "enrichment.sources.mal.mal_helper.MalHelper",
                    return_value=helper,
                ),
                patch(
                    "sys.argv",
                    ["mal_helper", "anime", _ANIME_URL, str(out)],
                ),
            ):
                from enrichment.sources.mal.mal_helper import main

                rc = await main()

            assert rc == 0
            assert out.exists()
            assert json.loads(out.read_text(encoding="utf-8"))["anime"]["mal_id"] == 1

    @pytest.mark.asyncio
    async def test_main_episodes_writes_output_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "episodes.json"

            helper = AsyncMock()
            helper._fetch_episodes = AsyncMock(
                return_value=[{"episode_number": 1}, {"episode_number": 2}]
            )

            with (
                patch(
                    "enrichment.sources.mal.mal_helper.MalHelper",
                    return_value=helper,
                ),
                patch(
                    "sys.argv",
                    ["mal_helper", "episodes", _ANIME_URL, "2", str(out)],
                ),
            ):
                from enrichment.sources.mal.mal_helper import main

                rc = await main()

            assert rc == 0
            assert out.exists()
            assert len(json.loads(out.read_text(encoding="utf-8"))) == 2
            helper._fetch_episodes.assert_awaited_once_with(_ANIME_URL, 2)

    @pytest.mark.asyncio
    async def test_main_characters_writes_output_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "characters.json"

            helper = AsyncMock()
            helper._fetch_characters = AsyncMock(
                return_value=[{"mal_id": 10, "name": "Luffy"}]
            )

            with (
                patch(
                    "enrichment.sources.mal.mal_helper.MalHelper",
                    return_value=helper,
                ),
                patch(
                    "sys.argv",
                    ["mal_helper", "characters", _ANIME_URL, str(out)],
                ),
            ):
                from enrichment.sources.mal.mal_helper import main

                rc = await main()

            assert rc == 0
            assert out.exists()
            assert json.loads(out.read_text(encoding="utf-8"))[0]["mal_id"] == 10
            helper._fetch_characters.assert_awaited_once_with(_ANIME_URL)
