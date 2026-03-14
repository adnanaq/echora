import json
import tempfile
from pathlib import Path
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

from enrichment.api_helpers.mal_helper import MalEnrichmentHelper
from enrichment.crawlers.mal_crawler.mal_models import CharacterRef


# ---------------------------------------------------------------------------
# MalEnrichmentHelper — unit tests (crawler-based implementation)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_anime_returns_data_payload():
    """fetch_anime delegates to fetch_mal_anime crawler + anime_from_mal mapper."""
    fake_anime = MagicMock()
    fake_anime.url = "https://myanimelist.net/anime/1/X"
    mapped = {"mal_id": 1, "title": "X", "synopsis": None}

    with (
        patch("enrichment.api_helpers.mal_helper.fetch_mal_anime", new=AsyncMock(return_value=fake_anime)),
        patch("enrichment.api_helpers.mal_helper.anime_from_mal", return_value=mapped),
    ):
        helper = MalEnrichmentHelper("1")
        data = await helper.fetch_anime()

    assert data is not None
    assert data["mal_id"] == 1
    assert data["title"] == "X"
    assert "synopsis" in data


@pytest.mark.asyncio
async def test_fetch_anime_returns_none_on_crawler_failure():
    """fetch_anime returns None when the crawler returns None."""
    with patch("enrichment.api_helpers.mal_helper.fetch_mal_anime", new=AsyncMock(return_value=None)):
        helper = MalEnrichmentHelper("1")
        data = await helper.fetch_anime()

    assert data is None


@pytest.mark.asyncio
async def test_fetch_characters_basic_returns_list():
    """fetch_characters_basic delegates to fetch_mal_character_refs and dumps CharacterRefs."""
    refs = [
        CharacterRef(char_id=10, name="Luffy", role="Main"),
        CharacterRef(char_id=11, name="Zoro", role="Supporting"),
    ]

    with patch("enrichment.api_helpers.mal_helper.fetch_mal_character_refs", new=AsyncMock(return_value=refs)):
        helper = MalEnrichmentHelper("21")
        helper._anime_url = "https://myanimelist.net/anime/21/One_Piece"
        items = await helper.fetch_characters_basic()

    assert len(items) == 2
    assert items[0]["char_id"] == 10
    assert items[0]["role"] == "Main"
    assert items[1]["char_id"] == 11


@pytest.mark.asyncio
async def test_fetch_characters_basic_returns_empty_on_failure():
    """fetch_characters_basic returns [] when crawler returns empty list."""
    with patch("enrichment.api_helpers.mal_helper.fetch_mal_character_refs", new=AsyncMock(return_value=[])):
        helper = MalEnrichmentHelper("21")
        helper._anime_url = "https://myanimelist.net/anime/21/One_Piece"
        items = await helper.fetch_characters_basic()

    assert items == []


@pytest.mark.asyncio
async def test_fetch_episodes_maps_minimal_fields():
    """fetch_episodes uses fetch_mal_episodes and maps each result via episode_from_mal."""
    from enrichment.crawlers.mal_crawler.mal_models import MalScrapedEpisode

    ep1 = MagicMock(spec=MalScrapedEpisode)
    ep2 = MagicMock(spec=MalScrapedEpisode)
    mapped1 = {"mal_id": 1, "title": "E1", "episode_number": 1}
    mapped2 = {"mal_id": 2, "title": "E2", "episode_number": 2}

    with (
        patch("enrichment.api_helpers.mal_helper.fetch_mal_episodes", new=AsyncMock(return_value=[ep1, ep2])),
        patch("enrichment.api_helpers.mal_helper.episode_from_mal", side_effect=[mapped1, mapped2]),
    ):
        helper = MalEnrichmentHelper("21")
        helper._anime_url = "https://myanimelist.net/anime/21/One_Piece"
        episodes = await helper.fetch_episodes(2)

    assert len(episodes) == 2
    assert episodes[0]["mal_id"] == 1
    assert episodes[0]["title"] == "E1"
    assert episodes[0]["episode_number"] == 1
    assert episodes[1]["mal_id"] == 2
    assert episodes[1]["episode_number"] == 2


@pytest.mark.asyncio
async def test_fetch_episodes_skips_none_results():
    """fetch_episodes skips None entries returned by the batch crawler (failed episodes)."""
    from enrichment.crawlers.mal_crawler.mal_models import MalScrapedEpisode

    ep1 = MagicMock(spec=MalScrapedEpisode)
    mapped1 = {"mal_id": 1, "title": "E1", "episode_number": 1}

    with (
        patch("enrichment.api_helpers.mal_helper.fetch_mal_episodes", new=AsyncMock(return_value=[ep1, None])),
        patch("enrichment.api_helpers.mal_helper.episode_from_mal", return_value=mapped1),
    ):
        helper = MalEnrichmentHelper("21")
        helper._anime_url = "https://myanimelist.net/anime/21/One_Piece"
        episodes = await helper.fetch_episodes(2)

    assert len(episodes) == 1
    assert episodes[0]["mal_id"] == 1


@pytest.mark.asyncio
async def test_fetch_characters_detailed_returns_full_character_data():
    """fetch_characters_detailed batch-fetches character details and injects name/role."""
    from enrichment.crawlers.mal_crawler.mal_models import MalScrapedCharacter

    basic = [
        {"char_id": 100, "name": "A", "role": "Main", "favorites": 0},
    ]
    char = MalScrapedCharacter(mal_id=100, url="https://myanimelist.net/character/100", name="A")
    mapped = {"name": "A", "role": "MAIN", "sources": ["https://myanimelist.net/character/100"]}

    with (
        patch("enrichment.api_helpers.mal_helper.fetch_mal_characters", new=AsyncMock(return_value=[char])),
        patch("enrichment.api_helpers.mal_helper.character_from_mal", return_value=mapped),
    ):
        helper = MalEnrichmentHelper("21")
        chars = await helper.fetch_characters_detailed(basic)

    assert len(chars) == 1
    assert chars[0]["name"] == "A"


@pytest.mark.asyncio
async def test_fetch_characters_detailed_skips_failed_fetches():
    """fetch_characters_detailed skips characters whose batch fetch returns None."""
    basic = [
        {"char_id": 100, "name": "A", "role": "Main", "favorites": 0},
        {"char_id": 101, "name": "B", "role": "Supporting", "favorites": 0},
    ]

    with patch("enrichment.api_helpers.mal_helper.fetch_mal_characters", new=AsyncMock(return_value=[None, None])):
        helper = MalEnrichmentHelper("21")
        chars = await helper.fetch_characters_detailed(basic)

    assert len(chars) == 0


@pytest.mark.asyncio
async def test_fetch_all_data_fetches_episodes_and_characters():
    """fetch_all_data orchestrates anime + episodes + character detail fetches."""
    helper = MalEnrichmentHelper("1")
    helper.fetch_anime = AsyncMock(return_value={"mal_id": 1, "episode_count": 2})
    helper.fetch_characters_basic = AsyncMock(
        return_value=[{"mal_id": 10, "name": "A", "role": "Main"}]
    )
    helper.fetch_episodes = AsyncMock(return_value=[{"episode_number": 1}, {"episode_number": 2}])
    helper.fetch_characters_detailed = AsyncMock(return_value=[{"mal_id": 10, "name": "A"}])

    result = await helper.fetch_all_data(fallback_episode_count=0)

    assert result is not None
    assert result["anime"]["mal_id"] == 1
    assert len(result["episodes"]) == 2
    assert result["characters"] == [{"mal_id": 10, "name": "A"}]
    helper.fetch_episodes.assert_awaited_once_with(2, output_path=None)
    helper.fetch_characters_detailed.assert_awaited_once_with(
        [{"mal_id": 10, "name": "A", "role": "Main"}], output_path=None
    )


@pytest.mark.asyncio
async def test_fetch_all_data_uses_fallback_episode_count():
    """fetch_all_data uses fallback_episode_count when anime page lacks episode count."""
    helper = MalEnrichmentHelper("1")
    helper.fetch_anime = AsyncMock(return_value={"mal_id": 1, "episode_count": None})
    helper.fetch_characters_basic = AsyncMock(return_value=[])
    helper.fetch_episodes = AsyncMock(return_value=[{"episode_number": 1}])
    helper.fetch_characters_detailed = AsyncMock(return_value=[])

    result = await helper.fetch_all_data(fallback_episode_count=3)

    assert result is not None
    assert result["episodes"] == [{"episode_number": 1}]
    helper.fetch_episodes.assert_awaited_once_with(3, output_path=None)


@pytest.mark.asyncio
async def test_fetch_all_data_keeps_basic_when_detailed_empty():
    """fetch_all_data falls back to basic character list when detailed fetch returns empty."""
    helper = MalEnrichmentHelper("1")
    helper.fetch_anime = AsyncMock(return_value={"mal_id": 1, "episode_count": None})
    helper.fetch_characters_basic = AsyncMock(
        return_value=[{"mal_id": 10, "name": "A", "role": "Main"}]
    )
    helper.fetch_episodes = AsyncMock(return_value=[])
    helper.fetch_characters_detailed = AsyncMock(return_value=[])

    result = await helper.fetch_all_data(fallback_episode_count=0)

    assert result is not None
    # Falls back to basic when detailed is empty
    assert result["characters"] == [{"mal_id": 10, "name": "A", "role": "Main"}]


@pytest.mark.asyncio
async def test_fetch_all_data_returns_none_when_anime_fails():
    """fetch_all_data returns None when the anime fetch fails."""
    helper = MalEnrichmentHelper("1")
    helper.fetch_anime = AsyncMock(return_value=None)

    result = await helper.fetch_all_data()

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
            helper.fetch_anime = AsyncMock(return_value={"mal_id": 1, "title": "X"})

            helper_cm = AsyncMock()
            helper_cm.__aenter__ = AsyncMock(return_value=helper)
            helper_cm.__aexit__ = AsyncMock(return_value=False)

            with patch(
                "enrichment.api_helpers.mal_helper.MalEnrichmentHelper",
                return_value=helper_cm,
            ):
                with patch(
                    "sys.argv",
                    ["mal_helper", "anime", "1", str(out)],
                ):
                    from enrichment.api_helpers.mal_helper import main

                    rc = await main()

            assert rc == 0
            assert out.exists()
            assert json.loads(out.read_text(encoding="utf-8"))["mal_id"] == 1

    @pytest.mark.asyncio
    async def test_main_episodes_writes_output_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "episodes.json"

            helper = AsyncMock()
            helper.fetch_episodes = AsyncMock(
                return_value=[{"episode_number": 1}, {"episode_number": 2}]
            )

            helper_cm = AsyncMock()
            helper_cm.__aenter__ = AsyncMock(return_value=helper)
            helper_cm.__aexit__ = AsyncMock(return_value=False)

            with patch(
                "enrichment.api_helpers.mal_helper.MalEnrichmentHelper",
                return_value=helper_cm,
            ):
                with patch(
                    "sys.argv",
                    ["mal_helper", "episodes", "21", "2", str(out)],
                ):
                    from enrichment.api_helpers.mal_helper import main

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
            helper.fetch_characters_basic = AsyncMock(
                return_value=[{"mal_id": 10, "name": "Luffy", "role": "Main"}]
            )
            helper.fetch_characters_detailed = AsyncMock(
                return_value=[{"mal_id": 10, "name": "Luffy", "role": "Main"}]
            )

            helper_cm = AsyncMock()
            helper_cm.__aenter__ = AsyncMock(return_value=helper)
            helper_cm.__aexit__ = AsyncMock(return_value=False)

            with patch(
                "enrichment.api_helpers.mal_helper.MalEnrichmentHelper",
                return_value=helper_cm,
            ):
                with patch(
                    "sys.argv",
                    ["mal_helper", "characters", "21", str(out)],
                ):
                    from enrichment.api_helpers.mal_helper import main

                    rc = await main()

            assert rc == 0
            assert out.exists()
            assert json.loads(out.read_text(encoding="utf-8"))[0]["mal_id"] == 10
            helper.fetch_characters_basic.assert_awaited_once_with()
            helper.fetch_characters_detailed.assert_awaited_once()
