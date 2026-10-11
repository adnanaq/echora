import logging
from enum import StrEnum

import pytest
from common.models.anime import (
    Anime,
    AnimeRating,
    AnimeRecord,
    AnimeRelationType,
    AnimeStatus,
    AnimeType,
    CharacterRole,
    ContextualRank,
    Episode,
    EpisodeRange,
    ProductionStaff,
    RelatedSourceMaterial,
    SourceMaterialRelationType,
    SourceMaterialType,
    StaffMember,
    Statistics,
    StreamingEntry,
)
from pydantic import ValidationError


def _anime(**fields: object) -> Anime:
    return Anime.model_validate(
        {
            "title": "One Piece",
            "status": "FINISHED",
            "type": "TV",
            "sources": [],
            **fields,
        }
    )


@pytest.mark.parametrize(
    ("provider_value", "expected"),
    [
        ("Currently Airing", AnimeStatus.ONGOING),
        ("NOT_YET_RELEASED", AnimeStatus.UPCOMING),
        ("tba", AnimeStatus.UPCOMING),
        ("Completed", AnimeStatus.FINISHED),
        ("FINISHED", AnimeStatus.FINISHED),
    ],
)
def test_anime_status_provider_value_maps_to_member(
    provider_value: str, expected: AnimeStatus
) -> None:
    assert AnimeStatus(provider_value) is expected


def test_anime_status_unmapped_value_returns_unknown_and_logs_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="common.models.base"):
        assert AnimeStatus("Sleeping") is AnimeStatus.UNKNOWN

    assert "AnimeStatus" in caplog.text
    assert "'Sleeping'" in caplog.text


def test_anime_status_empty_value_returns_unknown_without_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="common.models.base"):
        assert AnimeStatus("") is AnimeStatus.UNKNOWN

    assert caplog.text == ""


def test_anime_status_non_string_returns_unknown() -> None:
    assert AnimeStatus(None) is AnimeStatus.UNKNOWN


def test_anime_type_unmapped_value_returns_unknown_and_logs_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="common.models.base"):
        assert AnimeType("Hologram") is AnimeType.UNKNOWN

    assert "AnimeType" in caplog.text


def test_anime_relation_type_unmapped_value_returns_other_and_logs_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="common.models.base"):
        assert AnimeRelationType("Distant cousin") is AnimeRelationType.OTHER

    assert "AnimeRelationType" in caplog.text


def test_character_role_secondary_maps_to_supporting() -> None:
    assert CharacterRole("Secondary") is CharacterRole.SUPPORTING


def test_source_material_type_web_novel_value_is_plain_string() -> None:
    assert SourceMaterialType.WEB_NOVEL.value == "WEB NOVEL"


def test_anime_minimal_fields_dump_defaults() -> None:
    assert _anime().model_dump(mode="json", exclude_none=True) == {
        "episode_count": 0,
        "entity_type": "anime",
        "rating": "UNKNOWN",
        "status": "FINISHED",
        "title": "One Piece",
        "type": "TV",
        "companies": [],
        "content_warnings": [],
        "demographics": [],
        "ending_themes": [],
        "external_sources": [],
        "genres": [],
        "opening_themes": [],
        "related_anime": {},
        "related_source_material": {},
        "sources": [],
        "streaming_sources": [],
        "synonyms": [],
        "tags": [],
        "themes": [],
        "trailers": [],
        "images": {"covers": [], "posters": [], "banners": []},
        "statistics": {},
        "titles": {},
    }


@pytest.mark.parametrize("missing", ["title", "status", "type", "sources"])
def test_anime_missing_required_field_fails(missing: str) -> None:
    fields = {"title": "One Piece", "status": "FINISHED", "type": "TV", "sources": []}
    del fields[missing]

    with pytest.raises(ValidationError):
        Anime.model_validate(fields)


def test_anime_unknown_key_fails() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden|Extra inputs"):
        _anime(studios=["Toei Animation"])


def test_anime_unknown_nested_key_fails() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        _anime(
            streaming_sources=[
                {
                    "platform": "Crunchyroll",
                    "source": "https://crunchyroll.com",
                    "language": "en",
                }
            ]
        )


def test_anime_negative_episode_count_fails() -> None:
    with pytest.raises(ValidationError):
        _anime(episode_count=-1)


def test_anime_negative_duration_fails() -> None:
    with pytest.raises(ValidationError):
        _anime(duration=-60)


def test_anime_provider_statistics_parse_from_dict() -> None:
    anime = _anime(statistics={"mal": {"score": "8.7", "scored_by": 1200, "rank": 50}})

    assert anime.statistics["mal"] == Statistics(score=8.7, scored_by=1200, rank=50)


@pytest.mark.parametrize("score", [-0.1, 10.1, float("nan"), float("inf")])
def test_statistics_score_outside_zero_to_ten_fails(score: float) -> None:
    with pytest.raises(ValidationError):
        Statistics(score=score)


@pytest.mark.parametrize("score", [0, 10])
def test_statistics_score_at_bound_is_kept(score: float) -> None:
    assert Statistics(score=score).score == score


@pytest.mark.parametrize("field", ["favorites", "members", "scored_by"])
def test_statistics_negative_count_fails(field: str) -> None:
    with pytest.raises(ValidationError):
        Statistics.model_validate({field: -1})


@pytest.mark.parametrize("field", ["rank", "popularity"])
def test_statistics_zero_rank_fails(field: str) -> None:
    with pytest.raises(ValidationError):
        Statistics.model_validate({field: 0})


def test_contextual_rank_zero_rank_fails() -> None:
    with pytest.raises(ValidationError):
        ContextualRank(context="highest rated all time", rank=0)


def test_related_source_material_negative_chapters_fails() -> None:
    with pytest.raises(ValidationError):
        RelatedSourceMaterial(
            title="One Piece", type=SourceMaterialType.MANGA, chapters=-1
        )


def test_episode_range_negative_start_fails() -> None:
    with pytest.raises(ValidationError):
        EpisodeRange(start=-1)


def test_episode_zero_episode_number_is_kept() -> None:
    assert Episode(episode_number=0, title="Prologue").episode_number == 0


def test_episode_negative_episode_number_fails() -> None:
    with pytest.raises(ValidationError):
        Episode(episode_number=-1, title="Prologue")


def test_episode_negative_score_fails() -> None:
    with pytest.raises(ValidationError):
        Episode(episode_number=1, title="Romance Dawn", score=-1)


def test_episode_zero_season_number_fails() -> None:
    with pytest.raises(ValidationError):
        Episode(episode_number=1, title="Romance Dawn", season_number=0)


def test_production_staff_get_all_roles_returns_staff_members() -> None:
    staff = ProductionStaff.model_validate(
        {"directors": [{"name": "Konosuke Uda", "role": "Director"}], "producers": []}
    )

    roles = staff.get_all_roles()

    assert roles == {"directors": [StaffMember(name="Konosuke Uda", role="Director")]}


def test_production_staff_role_not_list_fails() -> None:
    with pytest.raises(ValidationError):
        ProductionStaff.model_validate({"notes": "not a list"})


def test_streaming_entry_unknown_key_fails() -> None:
    with pytest.raises(ValidationError):
        StreamingEntry.model_validate(
            {"platform": "Netflix", "source": "https://netflix.com", "language": "en"}
        )


def test_anime_record_round_trip_keeps_values() -> None:
    record = AnimeRecord(
        anime=_anime(statistics={"mal": {"score": 8.7}}, episode_count=1155),
        episodes=[Episode(episode_number=1, title="Romance Dawn", duration=1440)],
    )

    dumped = record.model_dump(mode="json", exclude_none=True)

    assert AnimeRecord.model_validate(dumped) == record


@pytest.mark.parametrize(
    ("enum", "provider_value", "expected"),
    [
        (SourceMaterialType, "Original Work", SourceMaterialType.ORIGINAL),
        (
            SourceMaterialRelationType,
            "Original Work",
            SourceMaterialRelationType.ADAPTATION,
        ),
        (SourceMaterialRelationType, "CHARACTER", SourceMaterialRelationType.CHARACTER),
        (AnimeRelationType, "Main Story", AnimeRelationType.PARENT_STORY),
        (
            AnimeRelationType,
            "Alternative Environment",
            AnimeRelationType.ALTERNATIVE_SETTING,
        ),
        (AnimeRelationType, "Shared Universe", AnimeRelationType.SIDE_STORY),
        (AnimeRelationType, "Same origin", AnimeRelationType.SIDE_STORY),
        (AnimeRelationType, "?", AnimeRelationType.OTHER),
        (SourceMaterialRelationType, "?", SourceMaterialRelationType.OTHER),
        (AnimeRating, "None", AnimeRating.UNKNOWN),
    ],
)
def test_provider_value_maps_to_member_without_warning(
    enum: type[StrEnum],
    provider_value: str,
    expected: StrEnum,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="common.models.base"):
        assert enum(provider_value) is expected

    assert caplog.text == ""


@pytest.mark.parametrize(
    ("provider_value", "expected"),
    [
        ("TV", AnimeType.TV),
        ("tv short", AnimeType.TV_SHORT),
        ("MANGA", None),
        ("", None),
        (None, None),
    ],
)
def test_anime_type_from_provider_value_returns_member_or_none_without_warning(
    provider_value: str | None,
    expected: AnimeType | None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        assert AnimeType.from_provider_value(provider_value) is expected

    assert caplog.text == ""


@pytest.mark.parametrize(
    ("enum", "provider_value", "expected"),
    [
        (AnimeType, "Other", AnimeType.OTHER),
        (AnimeRelationType, "Original Work", AnimeRelationType.ADAPTATION),
        (AnimeRelationType, "Adaptation: Incomplete", AnimeRelationType.ADAPTATION),
        (AnimeRelationType, "Complete Story", AnimeRelationType.FULL_STORY),
        (AnimeRelationType, "Full Version", AnimeRelationType.FULL_STORY),
        (AnimeRelationType, "Original", AnimeRelationType.PARENT_STORY),
        (
            SourceMaterialRelationType,
            "Same origin",
            SourceMaterialRelationType.SIDE_STORY,
        ),
        (
            SourceMaterialRelationType,
            "SIDE_STORY",
            SourceMaterialRelationType.SIDE_STORY,
        ),
        (SourceMaterialRelationType, "SEQUEL", SourceMaterialRelationType.SEQUEL),
        (
            SourceMaterialRelationType,
            "Alternative version",
            SourceMaterialRelationType.ALTERNATIVE,
        ),
        (SourceMaterialRelationType, "PREQUEL", SourceMaterialRelationType.PREQUEL),
        (SourceMaterialRelationType, "PARENT", SourceMaterialRelationType.PARENT_STORY),
        (
            SourceMaterialRelationType,
            "Adaptation: Incomplete",
            SourceMaterialRelationType.ADAPTATION,
        ),
        (
            SourceMaterialRelationType,
            "Adaptation: Spin-off",
            SourceMaterialRelationType.SPIN_OFF,
        ),
        (SourceMaterialType, "Webtoon", SourceMaterialType.WEBTOON),
        (SourceMaterialType, "Fairy Tale", SourceMaterialType.FAIRY_TALE),
        (SourceMaterialType, "Short Story", SourceMaterialType.SHORT_STORY),
        (SourceMaterialType, "Otome Game", SourceMaterialType.VISUAL_NOVEL),
        (SourceMaterialType, "Eroge", SourceMaterialType.VISUAL_NOVEL),
        (SourceMaterialType, "Religious Text", SourceMaterialType.BOOK),
        (SourceMaterialType, "Cartoon", SourceMaterialType.WESTERN_MEDIA),
        (AnimeRating, "13 teens or older", AnimeRating.PG13),
        (AnimeRating, "13", AnimeRating.PG13),
        (AnimeRating, "violence & profanity", AnimeRating.R),
        (AnimeRating, "17+ (Violence and Profanity)", AnimeRating.R),
        (AnimeRating, "Violence and Profanity ", AnimeRating.R),
        (AnimeRating, "Violence", AnimeRating.R),
    ],
)
def test_provider_value_found_in_live_data_maps_to_member(
    enum: type[StrEnum],
    provider_value: str,
    expected: StrEnum,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="common.models.base"):
        assert enum(provider_value) is expected

    assert caplog.text == ""
