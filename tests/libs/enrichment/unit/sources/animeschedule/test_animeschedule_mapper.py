"""Unit tests for animeschedule_mapper.py — calendar field mapping.

AnimeSchedule is the only provider that reports a premiere month, so anything
the mapper drops here is lost for good.
"""

import json

import pytest
from enrichment.sources.animeschedule.animeschedule_mapper import (
    anime_from_animeschedule,
    map_animeschedule_anime_responses,
)
from enrichment.sources.animeschedule.animeschedule_models import AnimeScheduleAnime
from shared_proto.v1.merge_pb2 import PageResponse


def _anime(**fields: object) -> AnimeScheduleAnime:
    defaults = {
        "id": "one-piece",
        "route": "one-piece",
        "title": "One Piece",
        "month": "October",
        "year": 1999,
    }
    return AnimeScheduleAnime.model_validate({**defaults, **fields})


def test_calendar_fields_reach_the_canonical_record() -> None:
    mapped = anime_from_animeschedule(_anime(season={"season": "Fall", "year": "1999"}))
    assert mapped["month"] == 10
    assert mapped["year"] == 1999
    assert mapped["season"] == "FALL"


def test_month_unknown_name_is_absent() -> None:
    assert "month" not in anime_from_animeschedule(_anime(month="Spring"))


def test_month_absent_when_the_api_omits_it() -> None:
    assert "month" not in anime_from_animeschedule(_anime(month=None))


def test_studios_become_companies() -> None:
    mapped = anime_from_animeschedule(
        _anime(studios=[{"name": "Toei Animation", "route": "toei-animation"}])
    )
    assert mapped["companies"] == [
        {
            "name": "Toei Animation",
            "roles": ["STUDIO"],
            "sources": ["https://animeschedule.net/studios/toei-animation"],
        }
    ]
    assert "studios" not in mapped


_ZERO = "0001-01-01T00:00:00Z"


def test_anime_from_animeschedule_zero_dates_left_out() -> None:
    mapped = anime_from_animeschedule(
        _anime(premier=_ZERO, jpnTime=_ZERO, subPremier=_ZERO, delayedFrom=_ZERO)
    )

    assert "aired_dates" not in mapped
    assert "broadcast" not in mapped
    assert "hiatus" not in mapped


def test_anime_from_animeschedule_premiere_sets_aired_dates_and_broadcast() -> None:
    mapped = anime_from_animeschedule(
        _anime(premier="1999-10-20T00:00:00Z", subPremier=_ZERO)
    )

    assert mapped["aired_dates"]["aired_from"] == "1999-10-19T15:00:00Z"
    assert mapped["broadcast"] == {"premiere_jp": "1999-10-19T15:00:00Z"}


def test_anime_from_animeschedule_delay_sets_hiatus_without_zero_end() -> None:
    mapped = anime_from_animeschedule(
        _anime(
            delayedFrom="2024-01-07T00:00:00Z",
            delayedUntil=_ZERO,
            delayedDesc="Production issues",
        )
    )

    assert mapped["hiatus"] == {
        "reason": "Production issues",
        "hiatus_from": "2024-01-07",
    }


def test_anime_from_animeschedule_broadcast_times_map_to_time_of_day() -> None:
    mapped = anime_from_animeschedule(
        _anime(
            jpnTime="2025-04-01T23:15:00+09:00",
            subTime="2025-04-01T15:30:00Z",
            dubTime="2025-04-08T18:00:00Z",
        )
    )

    assert mapped["broadcast"] == {
        "jp_time": "23:15:00",
        "sub_time": "15:30:00",
        "dub_time": "18:00:00",
    }


def test_anime_from_animeschedule_description_entities_decoded() -> None:
    mapped = anime_from_animeschedule(
        _anime(description="see &#34;black entities&#34;.<br>Next")
    )

    assert mapped["synopsis"] == 'see "black entities". Next'


def test_anime_from_animeschedule_missing_status_gives_unknown() -> None:
    assert anime_from_animeschedule(_anime(status=None))["status"] == "UNKNOWN"


def test_anime_from_animeschedule_year_only_season_left_out() -> None:
    assert "season" not in anime_from_animeschedule(
        _anime(season={"season": "", "year": "2026"})
    )


def test_anime_from_animeschedule_own_page_listed_first_in_sources() -> None:
    mapped = anime_from_animeschedule(
        _anime(
            websites={
                "mal": "myanimelist.net/anime/21",
                "aniList": "anilist.co/anime/21",
            }
        )
    )

    assert mapped["sources"] == [
        "https://animeschedule.net/anime/one-piece",
        "https://myanimelist.net/anime/21",
        "https://anilist.co/anime/21",
    ]


def test_map_animeschedule_anime_responses_valid_json_returns_canonical_anime() -> None:
    payload = {
        "id": "as-1",
        "title": "Cowboy Bebop",
        "route": "cowboy-bebop",
        "episodes": 26,
        "websites": {"mal": "myanimelist.net/anime/1"},
    }
    canonical = map_animeschedule_anime_responses(
        [
            PageResponse(
                request_name="main",
                content_type="application/json",
                body=json.dumps(payload).encode("utf-8"),
            )
        ]
    )

    assert canonical["title"] == "Cowboy Bebop"
    assert canonical["episode_count"] == 26
    assert canonical["sources"] == [
        "https://animeschedule.net/anime/cowboy-bebop",
        "https://myanimelist.net/anime/1",
    ]


def test_map_animeschedule_anime_responses_missing_main_raises_value_error() -> None:
    with pytest.raises(ValueError, match="main"):
        map_animeschedule_anime_responses(
            [
                PageResponse(
                    request_name="other", content_type="application/json", body=b"{}"
                )
            ]
        )
