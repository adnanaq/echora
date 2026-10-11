"""Unit tests for animeschedule_mapper.py — calendar field mapping.

AnimeSchedule is the only provider that reports a premiere month, so anything
the mapper drops here is lost for good.
"""

from enrichment.sources.animeschedule.animeschedule_mapper import (
    anime_from_animeschedule,
)
from enrichment.sources.animeschedule.animeschedule_models import AnimScheduleAnime


def _anime(**fields: object) -> AnimScheduleAnime:
    defaults = {
        "id": "one-piece",
        "route": "one-piece",
        "title": "One Piece",
        "month": "October",
        "year": 1999,
    }
    return AnimScheduleAnime.model_validate({**defaults, **fields})


def test_calendar_fields_reach_the_canonical_record() -> None:
    mapped = anime_from_animeschedule(_anime(season={"season": "Fall", "year": "1999"}))
    assert mapped["month"] == "October"
    assert mapped["year"] == 1999
    assert mapped["season"] == "FALL"


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
        "hiatus_from": "2024-01-07T00:00:00Z",
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
