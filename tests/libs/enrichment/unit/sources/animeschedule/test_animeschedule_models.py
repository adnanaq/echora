import pytest
from enrichment.sources.animeschedule.animeschedule_models import (
    AnimeScheduleAnime,
    AnimeScheduleNames,
    AnimeScheduleSeason,
)

_ZERO = "0001-01-01T00:00:00Z"


def _anime(**fields: object) -> AnimeScheduleAnime:
    return AnimeScheduleAnime.model_validate(
        {"id": "1", "route": "one-piece", "title": "One Piece", **fields}
    )


@pytest.mark.parametrize(
    "field",
    [
        "premier",
        "subPremier",
        "dubPremier",
        "jpnTime",
        "subTime",
        "dubTime",
        "delayedFrom",
        "delayedUntil",
    ],
)
def test_anim_schedule_anime_zero_date_becomes_none(field: str) -> None:
    anime = _anime(**{field: _ZERO})

    assert anime.model_dump(by_alias=True)[field] is None


@pytest.mark.parametrize("field", ["premier", "delayedFrom"])
def test_anim_schedule_anime_empty_date_becomes_none(field: str) -> None:
    assert _anime(**{field: ""}).model_dump(by_alias=True)[field] is None


def test_anim_schedule_anime_real_date_is_kept() -> None:
    assert _anime(premier="1999-10-20T00:00:00Z").premier == "1999-10-20T00:00:00Z"


def test_anim_schedule_anime_description_html_becomes_plain_text() -> None:
    anime = _anime(description="see &#34;black entities&#34;.<br><br>(Source: ANN)")

    assert anime.description == 'see "black entities". (Source: ANN)'


def test_anim_schedule_anime_delayed_description_is_delayed_reason() -> None:
    assert _anime(delayedDesc="Production issues").delayed_reason == "Production issues"


def test_anim_schedule_season_empty_season_becomes_none() -> None:
    assert (
        AnimeScheduleSeason.model_validate({"season": "", "year": "2026"}).season
        is None
    )


@pytest.mark.parametrize("field", ["native", "romaji", "english", "abbreviation"])
def test_anim_schedule_names_empty_name_becomes_none(field: str) -> None:
    assert getattr(AnimeScheduleNames.model_validate({field: " "}), field) is None
