from unittest.mock import AsyncMock, MagicMock

import pytest

from enrichment.api_helpers.mal_enrichment_helper import MalEnrichmentHelper


def _cm(response):
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=response)
    cm.__aexit__ = AsyncMock(return_value=None)
    return cm


@pytest.mark.asyncio
async def test_fetch_anime_returns_data_payload():
    resp = AsyncMock()
    resp.status = 200
    resp.json = AsyncMock(return_value={"data": {"mal_id": 1, "title": "X"}})

    session = MagicMock()
    session.get = MagicMock(return_value=_cm(resp))

    limiter = AsyncMock()
    helper = MalEnrichmentHelper("1", session=session, limiter=limiter)

    data = await helper.fetch_anime()
    assert data["mal_id"] == 1
    assert data["title"] == "X"
    # Pydantic dump includes default fields like synopsis (None)
    assert "synopsis" in data


@pytest.mark.asyncio
async def test_fetch_characters_basic_returns_list():
    resp = AsyncMock()
    resp.status = 200
    resp.json = AsyncMock(
        return_value={
            "data": [
                {
                    "character": {"mal_id": 10, "name": "Luffy", "url": "url"},
                    "role": "Main",
                }
            ]
        }
    )

    session = MagicMock()
    session.get = MagicMock(return_value=_cm(resp))

    helper = MalEnrichmentHelper("1", session=session, limiter=AsyncMock())
    items = await helper.fetch_characters_basic()
    assert len(items) == 1
    assert items[0]["character"]["mal_id"] == 10
    assert items[0]["role"] == "Main"


@pytest.mark.asyncio
async def test_fetch_episodes_maps_minimal_fields():
    def make_episode_response(title: str, eid: int):
        resp = AsyncMock()
        resp.status = 200
        resp.json = AsyncMock(
            return_value={"data": {"mal_id": eid, "title": title, "synopsis": "S"}}
        )
        return resp

    session = MagicMock()
    session.get = MagicMock(
        side_effect=[
            _cm(make_episode_response("E1", 1)),
            _cm(make_episode_response("E2", 2)),
        ]
    )

    helper = MalEnrichmentHelper("21", session=session, limiter=AsyncMock())
    episodes = await helper.fetch_episodes(2)

    assert len(episodes) == 2
    assert episodes[0]["mal_id"] == 1
    assert episodes[0]["title"] == "E1"
    # Verify episode_number injection
    assert episodes[0]["episode_number"] == 1
    
    assert episodes[1]["mal_id"] == 2
    assert episodes[1]["title"] == "E2"
    assert episodes[1]["episode_number"] == 2


@pytest.mark.asyncio
async def test_fetch_characters_detailed_returns_full_character_data():
    basic = [
        {"character": {"mal_id": 100}, "role": "Main", "voice_actors": []},
    ]

    def make_char_response(character_id: int, name: str):
        resp = AsyncMock()
        resp.status = 200
        resp.json = AsyncMock(
            return_value={
                "data": {
                    "mal_id": character_id,
                    "name": name,
                    "nicknames": ["nick"],
                    "favorites": 123,
                    "about": "About",
                    "anime": [{"role": "Main", "anime": {"mal_id": 21, "title": "X"}}],
                    "voices": [],
                }
            }
        )
        return resp

    session = MagicMock()
    session.get = MagicMock(
        side_effect=[
            _cm(make_char_response(100, "A")),
        ]
    )

    helper = MalEnrichmentHelper("21", session=session, limiter=AsyncMock())
    chars = await helper.fetch_characters_detailed(basic)

    assert len(chars) == 1
    assert chars[0]["mal_id"] == 100
    assert chars[0]["name"] == "A"
    assert "nicknames" in chars[0]
    # Verify full data (animeography) is present
    assert len(chars[0]["anime"]) == 1
    assert chars[0]["anime"][0]["role"] == "Main"
    # No character_id or role merging from basic!
    assert "character_id" not in chars[0]


@pytest.mark.asyncio
async def test_fetch_all_data_without_details_keeps_basic_characters():
    helper = MalEnrichmentHelper("1", session=MagicMock(), limiter=AsyncMock())
    helper.fetch_anime = AsyncMock(return_value={"mal_id": 1, "episodes": 12})
    helper.fetch_characters_basic = AsyncMock(
        return_value=[{"character": {"mal_id": 10}, "role": "Main"}]
    )
    helper.fetch_episodes = AsyncMock(return_value=[{"episode_number": 1}])
    helper.fetch_characters_detailed = AsyncMock(return_value=[{"mal_id": 10}])

    result = await helper.fetch_all_data(
        include_details=False,
        fallback_episode_count=0,
    )

    assert result is not None
    assert result["anime"]["mal_id"] == 1
    assert result["episodes"] == []
    assert result["characters"] == [{"character": {"mal_id": 10}, "role": "Main"}]
    helper.fetch_episodes.assert_not_awaited()
    helper.fetch_characters_detailed.assert_not_awaited()


@pytest.mark.asyncio
async def test_fetch_all_data_with_details_uses_fallback_and_character_fallback():
    helper = MalEnrichmentHelper("1", session=MagicMock(), limiter=AsyncMock())
    helper.fetch_anime = AsyncMock(return_value={"mal_id": 1, "episodes": None})
    helper.fetch_characters_basic = AsyncMock(
        return_value=[{"character": {"mal_id": 10}, "role": "Main"}]
    )
    helper.fetch_episodes = AsyncMock(return_value=[{"episode_number": 1}])
    # If detailed fetch fails/empty, helper should keep basic characters.
    helper.fetch_characters_detailed = AsyncMock(return_value=[])

    result = await helper.fetch_all_data(
        include_details=True,
        fallback_episode_count=3,
    )

    assert result is not None
    assert result["anime"]["mal_id"] == 1
    assert result["episodes"] == [{"episode_number": 1}]
    assert result["characters"] == [{"character": {"mal_id": 10}, "role": "Main"}]
    helper.fetch_episodes.assert_awaited_once_with(3)
    helper.fetch_characters_detailed.assert_awaited_once_with(
        [{"character": {"mal_id": 10}, "role": "Main"}]
    )
