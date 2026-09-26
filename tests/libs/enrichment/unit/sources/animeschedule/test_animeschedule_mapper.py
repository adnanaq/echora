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
    return AnimScheduleAnime(**{**defaults, **fields})


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
