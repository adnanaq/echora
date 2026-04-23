import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from enrichment.sources.mal.mal_helper import MalHelper

# ---------------------------------------------------------------------------
# MalHelper — unit tests (crawler-based implementation)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_context_manager_protocol() -> None:
    """async with MalHelper covers close/__aenter__/__aexit__."""
    async with MalHelper() as helper:
        assert isinstance(helper, MalHelper)


@pytest.mark.asyncio
async def test_fetch_anime_returns_data_payload():
    """fetch_anime delegates to fetch_mal_anime; crawler now returns canonical dict directly."""
    mapped = {
        "mal_id": 1,
        "title": "X",
        "synopsis": None,
        "episode_count": 12,
        "sources": ["https://myanimelist.net/anime/1/X"],
    }

    with patch(
        "enrichment.sources.mal.mal_helper.fetch_mal_anime",
        new=AsyncMock(return_value=mapped),
    ):
        helper = MalHelper()
        helper._mal_source = "https://myanimelist.net/anime/1"
        data = await helper.fetch_anime()

    assert data is not None
    assert data["mal_id"] == 1
    assert data["title"] == "X"
    assert "synopsis" in data


@pytest.mark.asyncio
async def test_fetch_anime_resolves_unknown_episode_count():
    """fetch_anime patches episode_count via fetch_mal_episode_count when anime page returns 0."""
    mapped = {
        "mal_id": 21,
        "title": "One Piece",
        "episode_count": 0,
        "sources": ["https://myanimelist.net/anime/21/One_Piece"],
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
        helper = MalHelper()
        helper._mal_source = "https://myanimelist.net/anime/21/One_Piece"
        data = await helper.fetch_anime()

    assert data["episode_count"] == 1155


@pytest.mark.asyncio
async def test_fetch_anime_skips_episode_count_resolution_when_known():
    """fetch_anime does not call fetch_mal_episode_count when episode_count is already set."""
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
        helper = MalHelper()
        helper._mal_source = "https://myanimelist.net/anime/57334/Dandadan"
        data = await helper.fetch_anime()

    mock_count.assert_not_awaited()
    assert data["episode_count"] == 12


@pytest.mark.asyncio
async def test_fetch_anime_returns_none_on_crawler_failure():
    """fetch_anime returns None when the crawler returns None."""
    with patch(
        "enrichment.sources.mal.mal_helper.fetch_mal_anime",
        new=AsyncMock(return_value=None),
    ):
        helper = MalHelper()
        helper._mal_source = "https://myanimelist.net/anime/1"
        data = await helper.fetch_anime()

    assert data is None


@pytest.mark.asyncio
async def test_fetch_character_urls_returns_list():
    """fetch_character_urls delegates to fetch_mal_character_refs and returns URLs."""
    urls = [
        "https://myanimelist.net/character/40/Monkey_D_Luffy",
        "https://myanimelist.net/character/62/Roronoa_Zoro",
    ]

    with patch(
        "enrichment.sources.mal.mal_helper.fetch_mal_character_refs",
        new=AsyncMock(return_value=urls),
    ):
        helper = MalHelper()
        helper._anime_url = "https://myanimelist.net/anime/21/One_Piece"
        items = await helper.fetch_character_urls()

    assert len(items) == 2
    assert items[0] == "https://myanimelist.net/character/40/Monkey_D_Luffy"
    assert items[1] == "https://myanimelist.net/character/62/Roronoa_Zoro"


@pytest.mark.asyncio
async def test_fetch_character_urls_returns_empty_on_failure():
    """fetch_character_urls returns [] when crawler returns empty list."""
    with patch(
        "enrichment.sources.mal.mal_helper.fetch_mal_character_refs",
        new=AsyncMock(return_value=[]),
    ):
        helper = MalHelper()
        helper._anime_url = "https://myanimelist.net/anime/21/One_Piece"
        items = await helper.fetch_character_urls()

    assert items == []


@pytest.mark.asyncio
async def test_fetch_character_urls_returns_empty_when_anime_url_not_set():
    """fetch_character_urls returns [] and logs error if called before fetch_anime."""
    helper = MalHelper()
    helper._anime_url = ""
    items = await helper.fetch_character_urls()
    assert items == []


@pytest.mark.asyncio
async def test_fetch_episodes_maps_minimal_fields(tmp_path: Path):
    """fetch_episodes uses fetch_mal_episodes; crawler now returns canonical dicts directly."""
    ep1 = {"mal_id": 1, "title": "E1", "episode_number": 1}
    ep2 = {"mal_id": 2, "title": "E2", "episode_number": 2}

    async def _mock_fetch_episodes(urls, *, on_result=None):
        results = [ep1, ep2]
        if on_result:
            for ep in results:
                on_result(ep)
        return results

    out = tmp_path / "eps.jsonl"
    with patch(
        "enrichment.sources.mal.mal_helper.fetch_mal_episodes",
        side_effect=_mock_fetch_episodes,
    ):
        helper = MalHelper()
        helper._anime_url = "https://myanimelist.net/anime/21/One_Piece"
        episodes = await helper.fetch_episodes(2, output_path=str(out))

    assert len(episodes) == 2
    assert episodes[0]["mal_id"] == 1
    assert episodes[0]["title"] == "E1"
    assert episodes[0]["episode_number"] == 1
    assert episodes[1]["mal_id"] == 2
    assert episodes[1]["episode_number"] == 2
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["mal_id"] == 1
    assert json.loads(lines[1])["mal_id"] == 2


@pytest.mark.asyncio
async def test_fetch_episodes_skips_none_results():
    """fetch_episodes skips None entries returned by the batch crawler (failed episodes)."""
    ep1 = {"mal_id": 1, "title": "E1", "episode_number": 1}

    with patch(
        "enrichment.sources.mal.mal_helper.fetch_mal_episodes",
        new=AsyncMock(return_value=[ep1, None]),
    ):
        helper = MalHelper()
        helper._anime_url = "https://myanimelist.net/anime/21/One_Piece"
        episodes = await helper.fetch_episodes(2)

    assert len(episodes) == 1
    assert episodes[0]["mal_id"] == 1


@pytest.mark.asyncio
async def test_fetch_episodes_returns_empty_when_count_is_zero():
    """fetch_episodes returns [] immediately when episode_count is 0."""
    helper = MalHelper()
    helper._anime_url = "https://myanimelist.net/anime/21/One_Piece"
    assert await helper.fetch_episodes(0) == []


@pytest.mark.asyncio
async def test_fetch_character_returns_mapped_data():
    """fetch_character delegates to fetch_mal_character; crawler now returns canonical dict directly."""
    mapped = {
        "name": "Luffy",
        "sources": ["https://myanimelist.net/character/40/Luffy"],
    }

    with patch(
        "enrichment.sources.mal.mal_helper.fetch_mal_character",
        new=AsyncMock(return_value=mapped),
    ):
        helper = MalHelper()
        result = await helper.fetch_character(
            "https://myanimelist.net/character/40/Luffy"
        )

    assert result is not None
    assert result["name"] == "Luffy"


@pytest.mark.asyncio
async def test_fetch_character_returns_none_on_crawler_failure():
    """fetch_character returns None when crawler returns None."""
    with patch(
        "enrichment.sources.mal.mal_helper.fetch_mal_character",
        new=AsyncMock(return_value=None),
    ):
        helper = MalHelper()
        result = await helper.fetch_character(
            "https://myanimelist.net/character/40/Luffy"
        )

    assert result is None


@pytest.mark.asyncio
async def test_fetch_character_with_output_path_appends_jsonl(tmp_path: Path):
    """fetch_character appends to JSONL file when output_path is provided."""
    mapped = {
        "name": "Luffy",
        "sources": ["https://myanimelist.net/character/40/Luffy"],
    }
    out = tmp_path / "chars.jsonl"

    with patch(
        "enrichment.sources.mal.mal_helper.fetch_mal_character",
        new=AsyncMock(return_value=mapped),
    ):
        helper = MalHelper()
        await helper.fetch_character(
            "https://myanimelist.net/character/40/Luffy", output_path=str(out)
        )

    assert out.exists()
    assert json.loads(out.read_text(encoding="utf-8").strip())["name"] == "Luffy"


@pytest.mark.asyncio
async def test_fetch_characters_returns_full_character_data(tmp_path: Path):
    """fetch_characters batch-fetches character details; crawler now returns canonical dicts."""
    urls = ["https://myanimelist.net/character/100/A"]
    mapped = {
        "name": "A",
        "role": "MAIN",
        "sources": ["https://myanimelist.net/character/100/A"],
    }
    out = tmp_path / "chars.jsonl"

    async def _mock_fetch_characters(urls, *, on_result=None):
        if on_result:
            on_result(mapped)
        return [mapped]

    with patch(
        "enrichment.sources.mal.mal_helper.fetch_mal_characters",
        side_effect=_mock_fetch_characters,
    ):
        helper = MalHelper()
        chars = await helper.fetch_characters(urls, output_path=str(out))

    assert len(chars) == 1
    assert chars[0]["name"] == "A"
    assert out.exists()
    assert json.loads(out.read_text(encoding="utf-8").strip())["name"] == "A"


@pytest.mark.asyncio
async def test_fetch_characters_skips_failed_fetches():
    """fetch_characters skips characters whose batch fetch returns None."""
    urls = [
        "https://myanimelist.net/character/100/A",
        "https://myanimelist.net/character/101/B",
    ]

    with patch(
        "enrichment.sources.mal.mal_helper.fetch_mal_characters",
        new=AsyncMock(return_value=[None, None]),
    ):
        helper = MalHelper()
        chars = await helper.fetch_characters(urls)

    assert len(chars) == 0


@pytest.mark.asyncio
async def test_fetch_all_standardized(tmp_path: Path):
    """fetch_all standardized interface orchestrates anime + episodes + character detail fetches."""
    char_urls = ["https://myanimelist.net/character/10/A"]
    ids = {"mal_url": "https://myanimelist.net/anime/1"}

    helper = MalHelper()
    helper.fetch_anime = AsyncMock(
        return_value={"mal_id": 1, "episode_count": 2, "title": "Test"}
    )
    helper.fetch_character_urls = AsyncMock(return_value=char_urls)
    helper.fetch_episodes = AsyncMock(
        return_value=[{"episode_number": 1}, {"episode_number": 2}]
    )
    helper.fetch_characters = AsyncMock(return_value=[{"mal_id": 10, "name": "A"}])

    result = await helper.fetch_all(ids, {}, temp_dir=str(tmp_path))

    assert result is not None
    assert result["anime"]["mal_id"] == 1
    assert len(result["episodes"]) == 2
    assert result["characters"] == [{"mal_id": 10, "name": "A"}]
    helper.fetch_episodes.assert_awaited_once_with(
        2, output_path=str(tmp_path / "mal_episodes.jsonl")
    )
    helper.fetch_characters.assert_awaited_once_with(
        char_urls, output_path=str(tmp_path / "mal_characters.jsonl")
    )
    assert (tmp_path / "mal_anime.jsonl").exists()


@pytest.mark.asyncio
async def test_fetch_all_passes_episode_count_from_anime_page():
    """fetch_all passes episode_count from the anime page to fetch_episodes."""
    ids = {"mal_url": "https://myanimelist.net/anime/1"}
    helper = MalHelper()
    helper.fetch_anime = AsyncMock(
        return_value={"mal_id": 1, "episode_count": 12, "title": "Test"}
    )
    helper.fetch_character_urls = AsyncMock(return_value=[])
    helper.fetch_episodes = AsyncMock(return_value=[{"episode_number": 1}])
    helper.fetch_characters = AsyncMock(return_value=[])

    result = await helper.fetch_all(ids, {})

    assert result is not None
    assert result["episodes"] == [{"episode_number": 1}]
    helper.fetch_episodes.assert_awaited_once_with(12, output_path=None)


@pytest.mark.asyncio
async def test_fetch_all_returns_empty_when_detailed_empty():
    """fetch_all returns empty characters when detailed fetch returns empty."""
    ids = {"mal_url": "https://myanimelist.net/anime/1"}
    helper = MalHelper()
    helper.fetch_anime = AsyncMock(
        return_value={"mal_id": 1, "episode_count": None, "title": "Test"}
    )
    helper.fetch_character_urls = AsyncMock(
        return_value=["https://myanimelist.net/character/10/A"]
    )
    helper.fetch_episodes = AsyncMock(return_value=[])
    helper.fetch_characters = AsyncMock(return_value=[])

    result = await helper.fetch_all(ids, {})

    assert result is not None
    assert result["characters"] == []


@pytest.mark.asyncio
async def test_fetch_all_continues_when_episodes_raise():
    """fetch_all continues with empty episodes when episode fetch raises."""
    ids = {"mal_url": "https://myanimelist.net/anime/1"}
    helper = MalHelper()
    helper.fetch_anime = AsyncMock(
        return_value={"mal_id": 1, "episode_count": 2, "title": "Test"}
    )
    helper.fetch_character_urls = AsyncMock(return_value=[])
    helper.fetch_episodes = AsyncMock(side_effect=Exception("network error"))
    helper.fetch_characters = AsyncMock(return_value=[])

    result = await helper.fetch_all(ids, {})

    assert result is not None
    assert result["episodes"] == []


@pytest.mark.asyncio
async def test_fetch_all_continues_when_characters_raise():
    """fetch_all continues with empty characters when character fetch raises."""
    ids = {"mal_url": "https://myanimelist.net/anime/1"}
    helper = MalHelper()
    helper.fetch_anime = AsyncMock(
        return_value={"mal_id": 1, "episode_count": 0, "title": "Test"}
    )
    helper.fetch_character_urls = AsyncMock(
        return_value=["https://myanimelist.net/character/10/A"]
    )
    helper.fetch_episodes = AsyncMock(return_value=[])
    helper.fetch_characters = AsyncMock(side_effect=Exception("network error"))

    result = await helper.fetch_all(ids, {})

    assert result is not None
    assert result["characters"] == []


@pytest.mark.asyncio
async def test_fetch_all_returns_none_when_anime_fails():
    """fetch_all returns None when the anime fetch fails."""
    ids = {"mal_url": "https://myanimelist.net/anime/1"}
    helper = MalHelper()
    helper.fetch_anime = AsyncMock(return_value=None)

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
async def test_fetch_episodes_returns_empty_when_anime_url_not_set():
    """fetch_episodes returns [] and logs error when _anime_url is empty."""
    helper = MalHelper()
    helper._anime_url = ""
    result = await helper.fetch_episodes(12)
    assert result == []


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
                    [
                        "mal_helper",
                        "anime",
                        "https://myanimelist.net/anime/1",
                        str(out),
                    ],
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
            helper.fetch_episodes = AsyncMock(
                return_value=[{"episode_number": 1}, {"episode_number": 2}]
            )

            with (
                patch(
                    "enrichment.sources.mal.mal_helper.MalHelper",
                    return_value=helper,
                ),
                patch(
                    "sys.argv",
                    [
                        "mal_helper",
                        "episodes",
                        "https://myanimelist.net/anime/21",
                        "2",
                        str(out),
                    ],
                ),
            ):
                from enrichment.sources.mal.mal_helper import main

                rc = await main()

            assert rc == 0
            assert out.exists()
            assert len(json.loads(out.read_text(encoding="utf-8"))) == 2
            helper.fetch_episodes.assert_awaited_once_with(2)

    @pytest.mark.asyncio
    async def test_main_characters_writes_output_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "characters.json"

            helper = AsyncMock()
            helper.fetch_character_urls = AsyncMock(
                return_value=["https://myanimelist.net/character/10/A"]
            )
            helper.fetch_characters = AsyncMock(
                return_value=[{"mal_id": 10, "name": "Luffy"}]
            )

            with (
                patch(
                    "enrichment.sources.mal.mal_helper.MalHelper",
                    return_value=helper,
                ),
                patch(
                    "sys.argv",
                    [
                        "mal_helper",
                        "characters",
                        "https://myanimelist.net/anime/21",
                        str(out),
                    ],
                ),
            ):
                from enrichment.sources.mal.mal_helper import main

                rc = await main()

            assert rc == 0
            assert out.exists()
            assert json.loads(out.read_text(encoding="utf-8"))[0]["mal_id"] == 10
            helper.fetch_character_urls.assert_awaited_once_with()
            helper.fetch_characters.assert_awaited_once()
