"""Unit tests for AnimePlanetEnrichmentHelper."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from enrichment.crawlers.anime_planet.anime_planet_character_models import AnimePlanetCharacter
from enrichment.crawlers.anime_planet.anime_planet_models import AnimePlanetAnime

pytestmark = pytest.mark.asyncio

_AP_WWW = "https://www.anime-planet.com"
_AP_ANIME_URL = f"{_AP_WWW}/anime/dandadan"
_AP_ANIME_URL_NO_WWW = "https://anime-planet.com/anime/dandadan"

_ANIME_OBJ = AnimePlanetAnime(name="Dandadan", slug="dandadan")
_ANIME_CANONICAL = {"title": "Dandadan", "slug": "dandadan"}


# ---------------------------------------------------------------------------
# Method naming contract
# ---------------------------------------------------------------------------


def test_helper_exposes_fetch_anime_not_fetch_anime_data():
    from enrichment.api_helpers.anime_planet_helper import AnimePlanetEnrichmentHelper

    helper = AnimePlanetEnrichmentHelper()
    assert hasattr(helper, "fetch_anime")
    assert not hasattr(helper, "fetch_anime_data")


def test_helper_exposes_fetch_characters_not_fetch_character_data():
    from enrichment.api_helpers.anime_planet_helper import AnimePlanetEnrichmentHelper

    helper = AnimePlanetEnrichmentHelper()
    assert hasattr(helper, "fetch_characters")
    assert not hasattr(helper, "fetch_character_data")


# ---------------------------------------------------------------------------
# fetch_anime — URL normalization + passthrough
# ---------------------------------------------------------------------------


async def test_fetch_anime_passes_www_url_to_crawler():
    """Already-www URL is forwarded unchanged to fetch_animeplanet_anime."""
    from enrichment.api_helpers.anime_planet_helper import AnimePlanetEnrichmentHelper

    helper = AnimePlanetEnrichmentHelper()
    with patch(
        "enrichment.api_helpers.anime_planet_helper.fetch_animeplanet_anime",
        new=AsyncMock(return_value=_ANIME_OBJ),
    ) as mock_crawl, patch(
        "enrichment.api_helpers.anime_planet_helper.anime_from_animeplanet",
        return_value=_ANIME_CANONICAL,
    ):
        await helper.fetch_anime(_AP_ANIME_URL)

    mock_crawl.assert_awaited_once_with(_AP_ANIME_URL)


async def test_fetch_anime_normalizes_non_www_url_before_passing_to_crawler():
    """Non-www AP URL is normalized to www before being passed to the crawler."""
    from enrichment.api_helpers.anime_planet_helper import AnimePlanetEnrichmentHelper

    helper = AnimePlanetEnrichmentHelper()
    with patch(
        "enrichment.api_helpers.anime_planet_helper.fetch_animeplanet_anime",
        new=AsyncMock(return_value=_ANIME_OBJ),
    ) as mock_crawl, patch(
        "enrichment.api_helpers.anime_planet_helper.anime_from_animeplanet",
        return_value=_ANIME_CANONICAL,
    ):
        await helper.fetch_anime(_AP_ANIME_URL_NO_WWW)

    mock_crawl.assert_awaited_once_with(_AP_ANIME_URL)


async def test_fetch_anime_returns_none_when_crawler_returns_none():
    from enrichment.api_helpers.anime_planet_helper import AnimePlanetEnrichmentHelper

    helper = AnimePlanetEnrichmentHelper()
    with patch(
        "enrichment.api_helpers.anime_planet_helper.fetch_animeplanet_anime",
        new=AsyncMock(return_value=None),
    ):
        result = await helper.fetch_anime(_AP_ANIME_URL)

    assert result is None


async def test_fetch_anime_returns_mapped_canonical_dict():
    from enrichment.api_helpers.anime_planet_helper import AnimePlanetEnrichmentHelper

    helper = AnimePlanetEnrichmentHelper()
    with patch(
        "enrichment.api_helpers.anime_planet_helper.fetch_animeplanet_anime",
        new=AsyncMock(return_value=_ANIME_OBJ),
    ), patch(
        "enrichment.api_helpers.anime_planet_helper.anime_from_animeplanet",
        return_value=_ANIME_CANONICAL,
    ):
        result = await helper.fetch_anime(_AP_ANIME_URL)

    assert result is not None
    assert result["title"] == "Dandadan"


# ---------------------------------------------------------------------------
# fetch_characters — full URL expansion
# ---------------------------------------------------------------------------


async def test_fetch_characters_expands_relative_ref_urls_to_full():
    """Relative /characters/slug refs are expanded to absolute URLs before the
    crawler receives them — the crawler must never see relative paths."""
    from enrichment.api_helpers.anime_planet_helper import AnimePlanetEnrichmentHelper

    refs = [
        {"url": "/characters/monkey-d-luffy"},
        {"url": "/characters/nami"},
    ]
    with patch(
        "enrichment.api_helpers.anime_planet_helper.fetch_animeplanet_character_refs",
        new=AsyncMock(return_value=refs),
    ), patch(
        "enrichment.api_helpers.anime_planet_helper.fetch_animeplanet_characters",
        new=AsyncMock(return_value=[]),
    ) as mock_chars:
        await AnimePlanetEnrichmentHelper().fetch_characters(f"{_AP_WWW}/anime/dandadan")

    passed_urls = mock_chars.await_args[0][0]
    assert passed_urls == [
        f"{_AP_WWW}/characters/monkey-d-luffy",
        f"{_AP_WWW}/characters/nami",
    ]


async def test_fetch_characters_returns_empty_list_when_no_refs():
    from enrichment.api_helpers.anime_planet_helper import AnimePlanetEnrichmentHelper

    with patch(
        "enrichment.api_helpers.anime_planet_helper.fetch_animeplanet_character_refs",
        new=AsyncMock(return_value=[]),
    ):
        result = await AnimePlanetEnrichmentHelper().fetch_characters(f"{_AP_WWW}/anime/dandadan")

    assert result == []


async def test_fetch_characters_builds_correct_refs_url():
    """refs crawl is called with the full /anime/{slug}/characters URL."""
    from enrichment.api_helpers.anime_planet_helper import AnimePlanetEnrichmentHelper

    with patch(
        "enrichment.api_helpers.anime_planet_helper.fetch_animeplanet_character_refs",
        new=AsyncMock(return_value=[]),
    ) as mock_refs:
        await AnimePlanetEnrichmentHelper().fetch_characters(f"{_AP_WWW}/anime/one-piece")

    mock_refs.assert_awaited_once_with(f"{_AP_WWW}/anime/one-piece/characters")


# ---------------------------------------------------------------------------
# fetch_all
# ---------------------------------------------------------------------------


async def test_fetch_all_returns_split_anime_and_characters():
    """fetch_all returns {"anime": ..., "characters": [...]}."""
    from enrichment.api_helpers.anime_planet_helper import AnimePlanetEnrichmentHelper

    helper = AnimePlanetEnrichmentHelper()
    characters = [{"name": "Okarun"}]

    with patch(
        "enrichment.api_helpers.anime_planet_helper.fetch_animeplanet_anime",
        new=AsyncMock(return_value=_ANIME_OBJ),
    ), patch(
        "enrichment.api_helpers.anime_planet_helper.anime_from_animeplanet",
        return_value=_ANIME_CANONICAL,
    ), patch.object(
        helper, "fetch_characters", new=AsyncMock(return_value=characters)
    ):
        result = await helper.fetch_all({"anime_planet_url": _AP_ANIME_URL}, {})

    assert result == {"anime": _ANIME_CANONICAL, "characters": characters}


async def test_fetch_all_writes_anime_before_characters(tmp_path):
    """Anime output file is written before character fetch begins."""
    from enrichment.api_helpers.anime_planet_helper import AnimePlanetEnrichmentHelper

    write_calls: list[str] = []

    async def slow_char_fetch(slug: str, *, output_path: str | None = None):
        write_calls.append("char_fetch_started")
        return []

    helper = AnimePlanetEnrichmentHelper()

    def tracking_append(path: str, data: object) -> None:
        write_calls.append(f"write:{os.path.basename(path)}")

    with patch(
        "enrichment.api_helpers.anime_planet_helper.fetch_animeplanet_anime",
        new=AsyncMock(return_value=_ANIME_OBJ),
    ), patch(
        "enrichment.api_helpers.anime_planet_helper.anime_from_animeplanet",
        return_value=_ANIME_CANONICAL,
    ), patch(
        "enrichment.api_helpers.anime_planet_helper.append_jsonl",
        side_effect=tracking_append,
    ), patch.object(helper, "fetch_characters", side_effect=slow_char_fetch):
        await helper.fetch_all(
            {"anime_planet_url": _AP_ANIME_URL},
            {},
            temp_dir=str(tmp_path),
        )

    assert write_calls.index("write:anime_planet_anime.jsonl") < write_calls.index(
        "char_fetch_started"
    )


async def test_fetch_all_returns_none_when_anime_missing():
    from enrichment.api_helpers.anime_planet_helper import AnimePlanetEnrichmentHelper

    with patch(
        "enrichment.api_helpers.anime_planet_helper.fetch_animeplanet_anime",
        new=AsyncMock(return_value=None),
    ):
        result = await AnimePlanetEnrichmentHelper().fetch_all({"anime_planet_url": _AP_ANIME_URL}, {})

    assert result is None


async def test_fetch_all_survives_character_fetch_failure():
    from enrichment.api_helpers.anime_planet_helper import AnimePlanetEnrichmentHelper

    helper = AnimePlanetEnrichmentHelper()
    with patch(
        "enrichment.api_helpers.anime_planet_helper.fetch_animeplanet_anime",
        new=AsyncMock(return_value=_ANIME_OBJ),
    ), patch(
        "enrichment.api_helpers.anime_planet_helper.anime_from_animeplanet",
        return_value=_ANIME_CANONICAL,
    ), patch.object(
        helper, "fetch_characters", new=AsyncMock(side_effect=RuntimeError("network error"))
    ):
        result = await helper.fetch_all({"anime_planet_url": _AP_ANIME_URL}, {})

    assert result is not None
    assert result["anime"] == _ANIME_CANONICAL
    assert result["characters"] == []


# ---------------------------------------------------------------------------
# Context manager protocol
# ---------------------------------------------------------------------------


async def test_context_manager_protocol():
    from enrichment.api_helpers.anime_planet_helper import AnimePlanetEnrichmentHelper

    async with AnimePlanetEnrichmentHelper() as helper:
        assert helper is not None
        assert isinstance(helper, AnimePlanetEnrichmentHelper)


async def test_context_manager_close_is_callable():
    from enrichment.api_helpers.anime_planet_helper import AnimePlanetEnrichmentHelper

    helper = AnimePlanetEnrichmentHelper()
    assert hasattr(helper, "close") and callable(helper.close)
    await helper.close()


async def test_context_manager_cleanup_on_exception():
    from enrichment.api_helpers.anime_planet_helper import AnimePlanetEnrichmentHelper

    with pytest.raises(ValueError, match="Test error"):
        async with AnimePlanetEnrichmentHelper():
            raise ValueError("Test error")


# ---------------------------------------------------------------------------
# main() CLI — subcommand interface
# ---------------------------------------------------------------------------


@patch("enrichment.api_helpers.anime_planet_helper.AnimePlanetEnrichmentHelper")
async def test_main_anime_subcommand_success(mock_helper_class, tmp_path):
    from enrichment.api_helpers.anime_planet_helper import main

    mock_helper = AsyncMock()
    mock_helper.fetch_anime = AsyncMock(return_value={"title": "Dandadan"})
    mock_helper_class.return_value = mock_helper
    mock_helper.__aenter__ = AsyncMock(return_value=mock_helper)
    mock_helper.__aexit__ = AsyncMock(return_value=False)
    out = tmp_path / "out.jsonl"

    with patch("sys.argv", ["prog", "anime", _AP_ANIME_URL, "--output", str(out)]), patch(
        "enrichment.api_helpers.anime_planet_helper.append_jsonl"
    ):
        code = await main()

    assert code == 0
    mock_helper.fetch_anime.assert_awaited_once_with(_AP_ANIME_URL)


@patch("enrichment.api_helpers.anime_planet_helper.AnimePlanetEnrichmentHelper")
async def test_main_anime_subcommand_no_data_returns_1(mock_helper_class):
    from enrichment.api_helpers.anime_planet_helper import main

    mock_helper = AsyncMock()
    mock_helper.fetch_anime = AsyncMock(return_value=None)
    mock_helper_class.return_value = mock_helper
    mock_helper.__aenter__ = AsyncMock(return_value=mock_helper)
    mock_helper.__aexit__ = AsyncMock(return_value=False)

    with patch("sys.argv", ["prog", "anime", _AP_ANIME_URL, "--output", "out.jsonl"]):
        code = await main()

    assert code == 1


async def test_main_characters_subcommand_streams_to_jsonl(tmp_path):
    from enrichment.api_helpers.anime_planet_helper import main

    char_url = f"{_AP_WWW}/characters/monkey-d-luffy"
    out = tmp_path / "chars.jsonl"
    char_obj = AnimePlanetCharacter(
        name="Luffy", slug="monkey-d-luffy", url=char_url
    )

    async def fake_fetch(urls: list[str], *, on_result=None) -> list[AnimePlanetCharacter]:
        if on_result:
            on_result(char_obj)
        return [char_obj]

    with patch(
        "enrichment.api_helpers.anime_planet_helper.fetch_animeplanet_characters",
        side_effect=fake_fetch,
    ), patch(
        "enrichment.api_helpers.anime_planet_helper.character_from_animeplanet",
        return_value={"name": "Luffy"},
    ), patch(
        "enrichment.api_helpers.anime_planet_helper.append_jsonl"
    ) as mock_append, patch(
        "sys.argv", ["prog", "characters", char_url, "--output", str(out)]
    ):
        code = await main()

    assert code == 0
    mock_append.assert_called_once_with(str(out), {"name": "Luffy"})


async def test_main_characters_subcommand_passes_urls_to_crawler(tmp_path):
    from enrichment.api_helpers.anime_planet_helper import main

    url1 = f"{_AP_WWW}/characters/monkey-d-luffy"
    url2 = f"{_AP_WWW}/characters/nami"
    out = tmp_path / "chars.jsonl"

    with patch(
        "enrichment.api_helpers.anime_planet_helper.fetch_animeplanet_characters",
        new=AsyncMock(return_value=[]),
    ) as mock_crawl, patch(
        "sys.argv", ["prog", "characters", url1, url2, "--output", str(out)]
    ):
        await main()

    passed_urls = mock_crawl.await_args[0][0]
    assert passed_urls == [url1, url2]


@patch("enrichment.api_helpers.anime_planet_helper.AnimePlanetEnrichmentHelper")
async def test_main_all_subcommand_success(mock_helper_class, tmp_path):
    from enrichment.api_helpers.anime_planet_helper import main

    mock_helper = AsyncMock()
    mock_helper.fetch_all = AsyncMock(return_value={"anime": _ANIME_CANONICAL, "characters": []})
    mock_helper_class.return_value = mock_helper
    mock_helper.__aenter__ = AsyncMock(return_value=mock_helper)
    mock_helper.__aexit__ = AsyncMock(return_value=False)
    anime_out = tmp_path / "anime.jsonl"
    chars_out = tmp_path / "chars.jsonl"

    with patch(
        "sys.argv",
        ["prog", "all", _AP_ANIME_URL, "--anime-output", str(anime_out), "--chars-output", str(chars_out)],
    ):
        code = await main()

    assert code == 0
    mock_helper.fetch_all.assert_awaited_once_with(
        {"anime_planet_url": _AP_ANIME_URL},
        {},
        None,
    )


@patch("enrichment.api_helpers.anime_planet_helper.AnimePlanetEnrichmentHelper")
async def test_main_all_subcommand_no_data_returns_1(mock_helper_class):
    from enrichment.api_helpers.anime_planet_helper import main

    mock_helper = AsyncMock()
    mock_helper.fetch_all = AsyncMock(return_value=None)
    mock_helper_class.return_value = mock_helper
    mock_helper.__aenter__ = AsyncMock(return_value=mock_helper)
    mock_helper.__aexit__ = AsyncMock(return_value=False)

    with patch("sys.argv", ["prog", "all", _AP_ANIME_URL]):
        code = await main()

    assert code == 1


async def test_main_no_subcommand_returns_1():
    from enrichment.api_helpers.anime_planet_helper import main

    with patch("sys.argv", ["prog"]):
        code = await main()

    assert code == 1
