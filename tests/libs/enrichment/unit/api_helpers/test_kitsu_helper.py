"""Unit tests for kitsu_helper.py."""

import aiohttp
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


def _make_media_char(mc_id: str, char_id: str, char_name: str = "Test"):
    """Build a minimal KitsuMediaCharacter with a character attached."""
    from enrichment.api_helpers.kitsu.kitsu_models import (
        KitsuCharacter,
        KitsuMediaCharacter,
    )

    char = KitsuCharacter.model_validate(
        {"id": char_id, "attributes": {"canonicalName": char_name}}
    )
    mc = KitsuMediaCharacter.model_validate(
        {"id": mc_id, "attributes": {"role": "main"}}
    )
    mc.character = char
    return mc


# ---------------------------------------------------------------------------
# main() function
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("enrichment.api_helpers.kitsu_helper.KitsuEnrichmentHelper")
async def test_main_function_success(mock_helper_class):
    from enrichment.api_helpers.kitsu_helper import main

    mock_helper = AsyncMock()
    mock_helper.fetch_all = AsyncMock(
        return_value={"anime": {"title": "Test"}, "episodes": [], "characters": []}
    )
    mock_helper.close = AsyncMock()
    mock_helper_class.return_value = mock_helper

    with patch("sys.argv", ["script.py", "123", "/tmp/output.json"]):
        with patch("builtins.open", MagicMock()):
            exit_code = await main()

    assert exit_code == 0
    mock_helper_class.assert_called_once()
    mock_helper.fetch_all.assert_awaited_once_with({"kitsu_id": "123"}, {})


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
# _make_request — HTTP error paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_make_request_raises_service_not_found_on_404():
    """HTTP 404 from the Kitsu API → ServiceNotFoundError propagated."""
    from enrichment.api_helpers.kitsu_helper import KitsuEnrichmentHelper
    from enrichment.exceptions import ServiceNotFoundError

    helper = KitsuEnrichmentHelper()
    mock_resp = AsyncMock()
    mock_resp.status = 404
    mock_sess = MagicMock()
    mock_sess.get = MagicMock(return_value=_cm(mock_resp))

    with patch(
        "enrichment.api_helpers.kitsu_helper._cache_manager.get_aiohttp_session",
        return_value=_cm(mock_sess),
    ):
        with pytest.raises(ServiceNotFoundError):
            await helper._make_request("/anime/1")


@pytest.mark.asyncio
async def test_make_request_raises_on_non_200():
    """HTTP 500 → raise_for_status propagates ClientResponseError."""
    from enrichment.api_helpers.kitsu_helper import KitsuEnrichmentHelper

    helper = KitsuEnrichmentHelper()
    mock_resp = AsyncMock()
    mock_resp.status = 500
    mock_resp.raise_for_status = MagicMock(
        side_effect=aiohttp.ClientResponseError(MagicMock(), MagicMock(), status=500)
    )
    mock_sess = MagicMock()
    mock_sess.get = MagicMock(return_value=_cm(mock_resp))

    with patch(
        "enrichment.api_helpers.kitsu_helper._cache_manager.get_aiohttp_session",
        return_value=_cm(mock_sess),
    ):
        with pytest.raises(aiohttp.ClientResponseError):
            await helper._make_request("/anime/1")


# ---------------------------------------------------------------------------
# _paginate — edge cases
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


@pytest.mark.asyncio
async def test_paginate_breaks_when_response_has_no_data_key():
    """_make_request returning {} (no 'data' key) → loop breaks immediately."""
    from enrichment.api_helpers.kitsu_helper import KitsuEnrichmentHelper

    helper = KitsuEnrichmentHelper()
    helper._make_request = AsyncMock(return_value={})

    items, included = await helper._paginate("/anime/1/episodes")

    assert items == []
    assert included == []
    helper._make_request.assert_awaited_once()


@pytest.mark.asyncio
async def test_paginate_breaks_when_items_empty():
    """data=[] → loop breaks without fetching further pages."""
    from enrichment.api_helpers.kitsu_helper import KitsuEnrichmentHelper

    helper = KitsuEnrichmentHelper()
    helper._make_request = AsyncMock(
        return_value={"data": [], "meta": {"count": 0}, "_from_cache": True}
    )

    items, _ = await helper._paginate("/anime/1/episodes")

    assert items == []
    helper._make_request.assert_awaited_once()


@pytest.mark.asyncio
async def test_paginate_sleeps_on_live_multi_page():
    """Non-cached first page with more pages remaining → asyncio.sleep(0.1) called."""
    from enrichment.api_helpers.kitsu_helper import KitsuEnrichmentHelper

    helper = KitsuEnrichmentHelper()
    page1 = [{"id": str(i)} for i in range(20)]
    page2 = [{"id": "99"}]
    helper._make_request = AsyncMock(
        side_effect=[
            {"data": page1, "meta": {"count": 21}, "_from_cache": False},
            {"data": page2, "meta": {"count": 21}, "_from_cache": True},
        ]
    )

    with patch(
        "enrichment.api_helpers.kitsu_helper.asyncio.sleep", new_callable=AsyncMock
    ) as mock_sleep:
        items, _ = await helper._paginate("/anime/1/episodes")

    assert len(items) == 21
    mock_sleep.assert_awaited_once_with(0.1)


@pytest.mark.asyncio
async def test_paginate_handles_request_exception():
    """Exception from _make_request → caught, returns empty collections."""
    from enrichment.api_helpers.kitsu_helper import KitsuEnrichmentHelper

    helper = KitsuEnrichmentHelper()
    helper._make_request = AsyncMock(side_effect=RuntimeError("network failure"))

    items, included = await helper._paginate("/anime/1/episodes")

    assert items == []
    assert included == []


# ---------------------------------------------------------------------------
# get_anime_by_id — error paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_anime_by_id_returns_none_on_service_not_found():
    """ServiceNotFoundError from _make_request → None."""
    from enrichment.api_helpers.kitsu_helper import KitsuEnrichmentHelper
    from enrichment.exceptions import ServiceNotFoundError

    helper = KitsuEnrichmentHelper()
    helper._make_request = AsyncMock(
        side_effect=ServiceNotFoundError("not found", service="kitsu")
    )

    assert await helper.get_anime_by_id(99999) is None


@pytest.mark.asyncio
async def test_get_anime_by_id_returns_none_on_client_error():
    """aiohttp.ClientError from _make_request → None."""
    from enrichment.api_helpers.kitsu_helper import KitsuEnrichmentHelper

    helper = KitsuEnrichmentHelper()
    helper._make_request = AsyncMock(side_effect=aiohttp.ClientError())

    assert await helper.get_anime_by_id(99999) is None


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
        "relationships": {"character": {"data": {"id": "c1", "type": "characters"}}},
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
        "relationships": {"character": {"data": {"id": "c999", "type": "characters"}}},
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
# fetch_anime
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_anime_returns_none_when_get_anime_raises():
    """get_anime_by_id raising inside gather(return_exceptions=True) → anime_raw is Exception → None."""
    from enrichment.api_helpers.kitsu_helper import KitsuEnrichmentHelper

    helper = KitsuEnrichmentHelper()
    with patch.object(
        helper, "get_anime_by_id", new=AsyncMock(side_effect=RuntimeError("fail"))
    ):
        with patch.object(helper, "get_anime_genres", new=AsyncMock(return_value=[])):
            with patch.object(
                helper, "get_anime_categories", new=AsyncMock(return_value=[])
            ):
                result = await helper.fetch_anime(12, session=MagicMock())

    assert result is None


@pytest.mark.asyncio
async def test_fetch_anime_writes_output_path(tmp_path):
    """output_path provided → _write_jsonl called with the mapped result."""
    from enrichment.api_helpers.kitsu_helper import KitsuEnrichmentHelper

    helper = KitsuEnrichmentHelper()
    raw_anime = {
        "id": "12",
        "attributes": {
            "canonicalTitle": "Test Anime",
            "subtype": "TV",
            "status": "current",
        },
    }
    out = str(tmp_path / "kitsu_anime.jsonl")

    with patch.object(helper, "get_anime_by_id", new=AsyncMock(return_value=raw_anime)):
        with patch.object(helper, "get_anime_genres", new=AsyncMock(return_value=[])):
            with patch.object(
                helper, "get_anime_categories", new=AsyncMock(return_value=[])
            ):
                with patch(
                    "enrichment.api_helpers.kitsu_helper._write_jsonl"
                ) as mock_write:
                    result = await helper.fetch_anime(
                        12, output_path=out, session=MagicMock()
                    )

    assert result is not None
    mock_write.assert_called_once_with(out, [result])


# ---------------------------------------------------------------------------
# fetch_episodes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_episodes_writes_each_episode_to_output_path(tmp_path):
    """output_path provided → _append_jsonl called once per episode."""
    from enrichment.api_helpers.kitsu_helper import KitsuEnrichmentHelper

    helper = KitsuEnrichmentHelper()
    raw_eps = [
        {
            "id": "e1",
            "type": "episodes",
            "attributes": {"number": 1, "canonicalTitle": "Episode 1"},
        },
        {
            "id": "e2",
            "type": "episodes",
            "attributes": {"number": 2, "canonicalTitle": "Episode 2"},
        },
    ]
    out = str(tmp_path / "kitsu_episodes.jsonl")

    with patch.object(
        helper, "get_anime_episodes", new=AsyncMock(return_value=raw_eps)
    ):
        with patch("enrichment.api_helpers.kitsu_helper._append_jsonl") as mock_append:
            results = await helper.fetch_episodes(12, output_path=out)

    assert len(results) == 2
    assert mock_append.call_count == 2


# ---------------------------------------------------------------------------
# fetch_characters
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_characters_returns_empty_when_no_characters():
    """get_anime_characters returns [] → fetch_characters returns []."""
    from enrichment.api_helpers.kitsu_helper import KitsuEnrichmentHelper

    helper = KitsuEnrichmentHelper()
    with patch.object(helper, "get_anime_characters", new=AsyncMock(return_value=[])):
        result = await helper.fetch_characters(12)

    assert result == []


@pytest.mark.asyncio
async def test_fetch_characters_maps_character_with_voices_and_animeography():
    """Happy path: character mapped via character_from_kitsu."""
    from enrichment.api_helpers.kitsu_helper import KitsuEnrichmentHelper

    helper = KitsuEnrichmentHelper()
    mc = _make_media_char("mc1", "c1", "Luffy")

    with patch.object(helper, "get_anime_characters", new=AsyncMock(return_value=[mc])):
        with patch.object(
            helper, "get_character_voices", new=AsyncMock(return_value=[])
        ):
            with patch.object(
                helper, "get_character_animeography", new=AsyncMock(return_value=[])
            ):
                with patch(
                    "enrichment.api_helpers.kitsu_helper.character_from_kitsu",
                    return_value={"name": "Luffy"},
                ) as mock_map:
                    result = await helper.fetch_characters(12)

    assert len(result) == 1
    assert result[0]["name"] == "Luffy"
    mock_map.assert_called_once()


@pytest.mark.asyncio
async def test_fetch_characters_skips_chars_without_character_object():
    """media_chars with character=None are excluded from the resolved list."""
    from enrichment.api_helpers.kitsu_helper import KitsuEnrichmentHelper
    from enrichment.api_helpers.kitsu.kitsu_models import KitsuMediaCharacter

    helper = KitsuEnrichmentHelper()
    mc_no_char = KitsuMediaCharacter.model_validate(
        {"id": "mc2", "attributes": {}}
    )  # character=None

    with patch.object(
        helper, "get_anime_characters", new=AsyncMock(return_value=[mc_no_char])
    ):
        result = await helper.fetch_characters(12)

    assert result == []


@pytest.mark.asyncio
async def test_fetch_characters_handles_voices_exception():
    """get_character_voices raising → voices treated as [] and char still mapped."""
    from enrichment.api_helpers.kitsu_helper import KitsuEnrichmentHelper

    helper = KitsuEnrichmentHelper()
    mc = _make_media_char("mc1", "c1")

    with patch.object(helper, "get_anime_characters", new=AsyncMock(return_value=[mc])):
        with patch.object(
            helper,
            "get_character_voices",
            new=AsyncMock(side_effect=RuntimeError("voices fail")),
        ):
            with patch.object(
                helper, "get_character_animeography", new=AsyncMock(return_value=[])
            ):
                with patch(
                    "enrichment.api_helpers.kitsu_helper.character_from_kitsu",
                    return_value={"name": "X"},
                ):
                    result = await helper.fetch_characters(12)

    assert len(result) == 1


@pytest.mark.asyncio
async def test_fetch_characters_handles_animeography_exception():
    """get_character_animeography raising → animeography treated as [] and char still mapped."""
    from enrichment.api_helpers.kitsu_helper import KitsuEnrichmentHelper

    helper = KitsuEnrichmentHelper()
    mc = _make_media_char("mc1", "c1")

    with patch.object(helper, "get_anime_characters", new=AsyncMock(return_value=[mc])):
        with patch.object(
            helper, "get_character_voices", new=AsyncMock(return_value=[])
        ):
            with patch.object(
                helper,
                "get_character_animeography",
                new=AsyncMock(side_effect=RuntimeError("anim fail")),
            ):
                with patch(
                    "enrichment.api_helpers.kitsu_helper.character_from_kitsu",
                    return_value={"name": "X"},
                ):
                    result = await helper.fetch_characters(12)

    assert len(result) == 1


@pytest.mark.asyncio
async def test_fetch_characters_handles_mapper_exception():
    """character_from_kitsu raising → that character filtered out of results."""
    from enrichment.api_helpers.kitsu_helper import KitsuEnrichmentHelper

    helper = KitsuEnrichmentHelper()
    mc = _make_media_char("mc1", "c1")

    with patch.object(helper, "get_anime_characters", new=AsyncMock(return_value=[mc])):
        with patch.object(
            helper, "get_character_voices", new=AsyncMock(return_value=[])
        ):
            with patch.object(
                helper, "get_character_animeography", new=AsyncMock(return_value=[])
            ):
                with patch(
                    "enrichment.api_helpers.kitsu_helper.character_from_kitsu",
                    side_effect=ValueError("bad data"),
                ):
                    result = await helper.fetch_characters(12)

    assert result == []


@pytest.mark.asyncio
async def test_fetch_characters_writes_to_output_path(tmp_path):
    """output_path triggers _append_jsonl for each successfully mapped character."""
    from enrichment.api_helpers.kitsu_helper import KitsuEnrichmentHelper

    helper = KitsuEnrichmentHelper()
    mc = _make_media_char("mc1", "c1", "Luffy")
    out = str(tmp_path / "chars.jsonl")

    with patch.object(helper, "get_anime_characters", new=AsyncMock(return_value=[mc])):
        with patch.object(
            helper, "get_character_voices", new=AsyncMock(return_value=[])
        ):
            with patch.object(
                helper, "get_character_animeography", new=AsyncMock(return_value=[])
            ):
                with patch(
                    "enrichment.api_helpers.kitsu_helper.character_from_kitsu",
                    return_value={"name": "Luffy"},
                ):
                    with patch(
                        "enrichment.api_helpers.kitsu_helper._append_jsonl"
                    ) as mock_append:
                        await helper.fetch_characters(12, output_path=out)

    mock_append.assert_called_once_with(out, {"name": "Luffy"})


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
            payload = {
                "data": [
                    {"id": "e1", "attributes": {"number": 1, "canonicalTitle": "Ep 1"}}
                ],
                "meta": {"count": 1},
            }
        elif "/genres" in url:
            payload = {
                "data": [{"id": "g1", "attributes": {"name": "Action"}}],
                "meta": {"count": 1},
            }
        elif "/categories" in url:
            payload = {
                "data": [
                    {
                        "id": "cat1",
                        "attributes": {
                            "title": "Pirates",
                            "description": "Pirate stuff",
                        },
                    }
                ],
                "meta": {"count": 1},
            }
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
            result = await helper.fetch_all({"kitsu_id": "12"}, {})

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
                result = await helper.fetch_all({"kitsu_id": "99999"}, {})

    assert result is None


@pytest.mark.asyncio
async def test_fetch_all_returns_none_when_no_kitsu_id():
    """ids dict without 'kitsu_id' → immediate None."""
    from enrichment.api_helpers.kitsu_helper import KitsuEnrichmentHelper

    assert await KitsuEnrichmentHelper().fetch_all({}, {}) is None


@pytest.mark.asyncio
async def test_fetch_all_slug_resolution_http_failure():
    """Non-200 from Kitsu slug endpoint → None before any fetch attempt."""
    from enrichment.api_helpers.kitsu_helper import KitsuEnrichmentHelper

    mock_resp = AsyncMock()
    mock_resp.status = 503
    mock_sess = AsyncMock()
    mock_sess.get = MagicMock(return_value=_cm(mock_resp))

    with patch(
        "enrichment.api_helpers.kitsu_helper.aiohttp.ClientSession",
        return_value=_cm(mock_sess),
    ):
        result = await KitsuEnrichmentHelper().fetch_all({"kitsu_id": "one-piece"}, {})

    assert result is None


@pytest.mark.asyncio
async def test_fetch_all_slug_resolution_no_data():
    """Slug endpoint returns 200 but data[] is empty → None."""
    from enrichment.api_helpers.kitsu_helper import KitsuEnrichmentHelper

    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.json = AsyncMock(return_value={"data": []})
    mock_sess = AsyncMock()
    mock_sess.get = MagicMock(return_value=_cm(mock_resp))

    with patch(
        "enrichment.api_helpers.kitsu_helper.aiohttp.ClientSession",
        return_value=_cm(mock_sess),
    ):
        result = await KitsuEnrichmentHelper().fetch_all({"kitsu_id": "one-piece"}, {})

    assert result is None


@pytest.mark.asyncio
async def test_fetch_all_slug_resolved_to_numeric_id():
    """Slug resolves to numeric ID and fetch proceeds normally."""
    from enrichment.api_helpers.kitsu_helper import KitsuEnrichmentHelper

    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.json = AsyncMock(return_value={"data": [{"id": "42"}]})
    mock_sess = AsyncMock()
    mock_sess.get = MagicMock(return_value=_cm(mock_resp))

    helper = KitsuEnrichmentHelper()
    with patch(
        "enrichment.api_helpers.kitsu_helper.aiohttp.ClientSession",
        return_value=_cm(mock_sess),
    ):
        with patch(
            "enrichment.api_helpers.kitsu_helper._cache_manager.get_aiohttp_session",
            return_value=_cm(AsyncMock()),
        ):
            with patch.object(
                helper,
                "fetch_anime",
                new=AsyncMock(return_value={"title": "Test", "sources": []}),
            ):
                with patch.object(
                    helper, "fetch_episodes", new=AsyncMock(return_value=[])
                ):
                    with patch.object(
                        helper, "fetch_characters", new=AsyncMock(return_value=[])
                    ):
                        result = await helper.fetch_all({"kitsu_id": "one-piece"}, {})

    assert result is not None
    assert result["anime"]["title"] == "Test"


@pytest.mark.asyncio
async def test_fetch_all_episodes_exception_yields_empty_list():
    """fetch_episodes raising inside gather → canonical_episodes = []."""
    from enrichment.api_helpers.kitsu_helper import KitsuEnrichmentHelper

    helper = KitsuEnrichmentHelper()
    with patch(
        "enrichment.api_helpers.kitsu_helper._cache_manager.get_aiohttp_session",
        return_value=_cm(AsyncMock()),
    ):
        with patch.object(
            helper,
            "fetch_anime",
            new=AsyncMock(return_value={"title": "X", "sources": []}),
        ):
            with patch.object(
                helper,
                "fetch_episodes",
                new=AsyncMock(side_effect=RuntimeError("ep fail")),
            ):
                with patch.object(
                    helper, "fetch_characters", new=AsyncMock(return_value=[])
                ):
                    result = await helper.fetch_all({"kitsu_id": "1"}, {})

    assert result is not None
    assert result["episodes"] == []


@pytest.mark.asyncio
async def test_fetch_all_characters_exception_yields_empty_list():
    """fetch_characters raising inside gather → canonical_characters = []."""
    from enrichment.api_helpers.kitsu_helper import KitsuEnrichmentHelper

    helper = KitsuEnrichmentHelper()
    with patch(
        "enrichment.api_helpers.kitsu_helper._cache_manager.get_aiohttp_session",
        return_value=_cm(AsyncMock()),
    ):
        with patch.object(
            helper,
            "fetch_anime",
            new=AsyncMock(return_value={"title": "X", "sources": []}),
        ):
            with patch.object(helper, "fetch_episodes", new=AsyncMock(return_value=[])):
                with patch.object(
                    helper,
                    "fetch_characters",
                    new=AsyncMock(side_effect=RuntimeError("char fail")),
                ):
                    result = await helper.fetch_all({"kitsu_id": "1"}, {})

    assert result is not None
    assert result["characters"] == []


@pytest.mark.asyncio
async def test_fetch_all_outer_exception_returns_none():
    """Exception during session setup → fetch_all returns None."""
    from enrichment.api_helpers.kitsu_helper import KitsuEnrichmentHelper

    helper = KitsuEnrichmentHelper()
    with patch(
        "enrichment.api_helpers.kitsu_helper._cache_manager.get_aiohttp_session",
        side_effect=RuntimeError("session failed"),
    ):
        result = await helper.fetch_all({"kitsu_id": "1"}, {})

    assert result is None
