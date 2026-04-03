"""Unit tests for kitsu_helper.py."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cm(response):
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=response)
    cm.__aexit__ = AsyncMock(return_value=None)
    return cm


# ---------------------------------------------------------------------------
# main() function
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("enrichment.api_helpers.kitsu_helper.KitsuEnrichmentHelper")
async def test_main_function_success(mock_helper_class):
    from enrichment.api_helpers.kitsu_helper import main

    mock_helper = AsyncMock()
    mock_helper.fetch_all = AsyncMock(return_value={"anime": {"title": "Test"}, "episodes": [], "characters": []})
    mock_helper.close = AsyncMock()
    mock_helper_class.return_value = mock_helper

    with patch("sys.argv", ["script.py", "123", "/tmp/output.json"]):
        with patch("builtins.open", MagicMock()):
            exit_code = await main()

    assert exit_code == 0
    mock_helper_class.assert_called_once()
    mock_helper.fetch_all.assert_awaited_once_with(123)


@pytest.mark.asyncio
@patch("enrichment.api_helpers.kitsu_helper.KitsuEnrichmentHelper")
async def test_main_function_no_data_found(mock_helper_class):
    from enrichment.api_helpers.kitsu_helper import main

    mock_helper = AsyncMock()
    mock_helper.fetch_all = AsyncMock(return_value=None)
    mock_helper.close = AsyncMock()
    mock_helper_class.return_value = mock_helper

    with patch("sys.argv", ["script.py", "99999", "/tmp/output.json"]):
        exit_code = await main()

    assert exit_code == 1


@pytest.mark.asyncio
@patch("enrichment.api_helpers.kitsu_helper.KitsuEnrichmentHelper")
async def test_main_function_error_handling(mock_helper_class):
    from enrichment.api_helpers.kitsu_helper import main

    mock_helper = AsyncMock()
    mock_helper.fetch_all = AsyncMock(side_effect=Exception("API error"))
    mock_helper.close = AsyncMock()
    mock_helper_class.return_value = mock_helper

    with patch("sys.argv", ["script.py", "123", "/tmp/output.json"]):
        exit_code = await main()

    assert exit_code == 1


@pytest.mark.asyncio
async def test_main_function_invalid_arguments():
    from enrichment.api_helpers.kitsu_helper import main

    with patch("sys.argv", ["script.py"]):
        exit_code = await main()

    assert exit_code == 1


# ---------------------------------------------------------------------------
# Context manager protocol
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_context_manager_protocol():
    from enrichment.api_helpers.kitsu_helper import KitsuEnrichmentHelper

    async with KitsuEnrichmentHelper() as helper:
        assert helper is not None
        assert isinstance(helper, KitsuEnrichmentHelper)


@pytest.mark.asyncio
async def test_context_manager_close_method_exists():
    from enrichment.api_helpers.kitsu_helper import KitsuEnrichmentHelper

    helper = KitsuEnrichmentHelper()
    assert hasattr(helper, "close")
    assert callable(helper.close)
    await helper.close()


@pytest.mark.asyncio
async def test_context_manager_cleanup_on_exception():
    from enrichment.api_helpers.kitsu_helper import KitsuEnrichmentHelper

    with pytest.raises(ValueError, match="Test error"):
        async with KitsuEnrichmentHelper() as helper:
            raise ValueError("Test error")


# ---------------------------------------------------------------------------
# Cache-aware pagination throttling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_anime_episodes_skips_sleep_on_cache_hit():
    """Cached pages should not trigger artificial throttle sleep."""
    from enrichment.api_helpers.kitsu_helper import KitsuEnrichmentHelper

    helper = KitsuEnrichmentHelper()

    page1 = [{"id": str(i)} for i in range(1, 21)]
    page2 = [{"id": str(i)} for i in range(21, 41)]
    helper._make_request = AsyncMock(
        side_effect=[
            {"data": page1, "meta": {"count": 40}, "_from_cache": True},
            {"data": page2, "meta": {"count": 40}, "_from_cache": True},
        ]
    )

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        episodes = await helper.get_anime_episodes(12)

    assert len(episodes) == 40
    mock_sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_anime_categories_skips_sleep_on_cache_hit():
    """Cached category pages should not trigger artificial throttle sleep."""
    from enrichment.api_helpers.kitsu_helper import KitsuEnrichmentHelper

    helper = KitsuEnrichmentHelper()

    page1 = [
        {"attributes": {"title": f"Theme {i}", "description": f"Desc {i}"}}
        for i in range(1, 21)
    ]
    page2 = [{"attributes": {"title": "Theme 21", "description": "Desc 21"}}]
    helper._make_request = AsyncMock(
        side_effect=[
            {"data": page1, "meta": {"count": 21}, "_from_cache": True},
            {"data": page2, "meta": {"count": 21}, "_from_cache": True},
        ]
    )

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        categories = await helper.get_anime_categories(12)

    assert len(categories) == 21
    mock_sleep.assert_not_awaited()


# ---------------------------------------------------------------------------
# get_anime_characters — sideloaded character resolution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_anime_characters_resolves_from_included():
    """Characters sideloaded in included[] should be attached to each mediaChar."""
    from enrichment.api_helpers.kitsu_helper import KitsuEnrichmentHelper

    helper = KitsuEnrichmentHelper()

    media_char_item = {
        "id": "mc1",
        "type": "mediaCharacters",
        "attributes": {"role": "main"},
        "relationships": {
            "character": {"data": {"id": "c1", "type": "characters"}}
        },
    }
    char_item = {
        "id": "c1",
        "type": "characters",
        "attributes": {"canonicalName": "Luffy", "slug": "luffy"},
    }

    helper._make_request = AsyncMock(
        return_value={
            "data": [media_char_item],
            "included": [char_item],
            "meta": {"count": 1},
            "_from_cache": True,
        }
    )

    result = await helper.get_anime_characters(12)

    assert len(result) == 1
    assert result[0].id == "mc1"
    assert result[0].character is not None
    assert result[0].character.id == "c1"
    assert result[0].character.attributes.canonicalName == "Luffy"


@pytest.mark.asyncio
async def test_get_anime_characters_missing_included_yields_none_character():
    """When a character is not found in included[], media_char.character stays None."""
    from enrichment.api_helpers.kitsu_helper import KitsuEnrichmentHelper

    helper = KitsuEnrichmentHelper()

    media_char_item = {
        "id": "mc1",
        "type": "mediaCharacters",
        "attributes": {"role": "supporting"},
        "relationships": {
            "character": {"data": {"id": "c999", "type": "characters"}}
        },
    }
    helper._make_request = AsyncMock(
        return_value={
            "data": [media_char_item],
            "included": [],  # character not sideloaded
            "meta": {"count": 1},
            "_from_cache": True,
        }
    )

    result = await helper.get_anime_characters(12)
    assert len(result) == 1
    assert result[0].character is None


# ---------------------------------------------------------------------------
# get_character_voices
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_character_voices_resolves_person():
    """People sideloaded in included[] should be attached to each voice."""
    from enrichment.api_helpers.kitsu_helper import KitsuEnrichmentHelper

    helper = KitsuEnrichmentHelper()

    voice_item = {
        "id": "v1",
        "type": "characterVoices",
        "attributes": {"locale": "ja_jp"},
        "relationships": {"person": {"data": {"id": "p1", "type": "people"}}},
    }
    person_item = {
        "id": "p1",
        "type": "people",
        "attributes": {"name": "Mayumi Tanaka"},
    }

    helper._make_request = AsyncMock(
        return_value={
            "data": [voice_item],
            "included": [person_item],
            "meta": {"count": 1},
            "_from_cache": True,
        }
    )

    result = await helper.get_character_voices("mc1")

    assert len(result) == 1
    assert result[0].attributes.locale == "ja_jp"
    assert result[0].person is not None
    assert result[0].person.attributes.name == "Mayumi Tanaka"


# ---------------------------------------------------------------------------
# get_character_animeography
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_character_animeography_splits_anime_and_manga():
    """media_type should be populated from relationship data.type."""
    from enrichment.api_helpers.kitsu_helper import KitsuEnrichmentHelper

    helper = KitsuEnrichmentHelper()

    anime_entry = {
        "id": "entry1",
        "type": "mediaCharacters",
        "attributes": {"role": "main"},
        "relationships": {"media": {"data": {"id": "m1", "type": "anime"}}},
    }
    manga_entry = {
        "id": "entry2",
        "type": "mediaCharacters",
        "attributes": {"role": "main"},
        "relationships": {"media": {"data": {"id": "m2", "type": "manga"}}},
    }
    anime_media = {
        "id": "m1",
        "type": "anime",
        "attributes": {"canonicalTitle": "One Piece", "slug": "one-piece"},
    }
    manga_media = {
        "id": "m2",
        "type": "manga",
        "attributes": {"canonicalTitle": "One Piece Manga", "slug": "one-piece-manga"},
    }

    helper._make_request = AsyncMock(
        return_value={
            "data": [anime_entry, manga_entry],
            "included": [anime_media, manga_media],
            "meta": {"count": 2},
            "_from_cache": True,
        }
    )

    result = await helper.get_character_animeography("c1")

    assert len(result) == 2
    anime_r = next(r for r in result if r.media_type == "anime")
    manga_r = next(r for r in result if r.media_type == "manga")
    assert anime_r.media is not None
    assert anime_r.media.attributes.canonicalTitle == "One Piece"
    assert manga_r.media is not None
    assert manga_r.media.attributes.canonicalTitle == "One Piece Manga"


# ---------------------------------------------------------------------------
# fetch_all — canonical shape and single-session reuse
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_all_reuses_one_session_for_anime_and_episodes():
    """fetch_all should open a single cached session for anime + episodes."""
    from enrichment.api_helpers.kitsu_helper import KitsuEnrichmentHelper

    helper = KitsuEnrichmentHelper()

    def get_side_effect(url, **kwargs):  # type: ignore[no-untyped-def]
        resp = AsyncMock()
        resp.status = 200
        if url.endswith("/anime/12"):
            payload = {
                "data": {
                    "id": "12",
                    "attributes": {
                        "canonicalTitle": "One Piece",
                        "subtype": "TV",
                        "status": "current",
                    },
                }
            }
        elif "/episodes" in url:
            payload = {"data": [{"id": "e1", "attributes": {"number": 1, "canonicalTitle": "Ep 1"}}], "meta": {"count": 1}}
        elif "/genres" in url:
            payload = {"data": [{"id": "g1", "attributes": {"name": "Action"}}], "meta": {"count": 1}}
        elif "/categories" in url:
            payload = {"data": [{"id": "cat1", "attributes": {"title": "Pirates", "description": "Pirate stuff"}}], "meta": {"count": 1}}
        else:
            payload = {}
        resp.json = AsyncMock(return_value=payload)
        return _cm(resp)

    mock_session = MagicMock()
    mock_session.get = MagicMock(side_effect=get_side_effect)

    def make_session_cm(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        return _cm(mock_session)

    # Patch fetch_characters to avoid character sub-requests in this test
    with patch(
        "enrichment.api_helpers.kitsu_helper._cache_manager.get_aiohttp_session",
        side_effect=make_session_cm,
    ) as mock_get_session:
        with patch.object(helper, "fetch_characters", new=AsyncMock(return_value=[])):
            result = await helper.fetch_all(12)

    assert result is not None
    assert result["anime"]["title"] == "One Piece"
    assert isinstance(result["episodes"], list)
    assert isinstance(result["characters"], list)
    # One session opened for anime + episodes + genres + categories
    assert mock_get_session.call_count == 1


@pytest.mark.asyncio
async def test_fetch_all_returns_none_when_anime_not_found():
    """fetch_all should return None when get_anime_by_id returns None."""
    from enrichment.api_helpers.kitsu_helper import KitsuEnrichmentHelper

    helper = KitsuEnrichmentHelper()

    with patch.object(helper, "fetch_anime", new=AsyncMock(return_value=None)):
        with patch.object(helper, "fetch_episodes", new=AsyncMock(return_value=[])):
            # We need to patch at the _cache_manager level too
            def make_session_cm(*_args, **_kwargs):  # type: ignore[no-untyped-def]
                return _cm(MagicMock())

            with patch(
                "enrichment.api_helpers.kitsu_helper._cache_manager.get_aiohttp_session",
                side_effect=make_session_cm,
            ):
                result = await helper.fetch_all(99999)

    assert result is None

