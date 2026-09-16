"""Unit tests for anidb_models.py — Pydantic source model validation."""

import pytest
from enrichment.sources.anidb.anidb_models import (
    AniDBAnime,
    AniDBCategory,
    AniDBCharacter,
    AniDBCharacterPage,
    AniDBCreator,
    AniDBEpisode,
    AniDBExternalResource,
    AniDBRatings,
    AniDBRelatedAnime,
    AniDBSeiyuu,
)
from pydantic import ValidationError

# =============================================================================
# AniDBSeiyuu
# =============================================================================


def test_seiyuu_minimal() -> None:
    s = AniDBSeiyuu(name="Mayumi Tanaka")
    assert s.name == "Mayumi Tanaka"
    assert s.id is None
    assert s.picture is None


def test_seiyuu_full() -> None:
    s = AniDBSeiyuu(id=95, name="Mayumi Tanaka", picture="95.jpg")
    assert s.id == 95
    assert s.picture == "95.jpg"


# =============================================================================
# AniDBCharacter
# =============================================================================


def test_character_minimal() -> None:
    c = AniDBCharacter()
    assert c.id is None
    assert c.seiyuu == []
    assert c.rating_votes == 0


def test_character_multiple_seiyuu() -> None:
    c = AniDBCharacter(
        id=40,
        name="Luffy",
        seiyuu=[
            AniDBSeiyuu(id=95, name="Mayumi Tanaka"),
            AniDBSeiyuu(id=200, name="Colleen Clinkenbeard"),
        ],
    )
    assert len(c.seiyuu) == 2
    assert c.seiyuu[0].name == "Mayumi Tanaka"


def test_character_instances_have_independent_seiyuu_lists() -> None:
    """Pydantic v2 must not share mutable defaults between instances."""
    a = AniDBCharacter(id=1, name="Luffy")
    b = AniDBCharacter(id=2, name="Zoro")
    a.seiyuu.append(AniDBSeiyuu(name="Test VA"))
    assert len(b.seiyuu) == 0
    assert a.seiyuu is not b.seiyuu


# =============================================================================
# AniDBEpisode
# =============================================================================


def test_episode_minimal() -> None:
    ep = AniDBEpisode()
    assert ep.id is None
    assert ep.titles == {}
    assert ep.streaming == {}
    assert ep.rating_votes == 0


def test_episode_regular() -> None:
    ep = AniDBEpisode(
        id=1001,
        episode_number=1,
        episode_type=1,
        length=24,
        airdate="1999-10-20",
        titles={"en": "Romance Dawn", "romaji": "Romance Dawn"},
    )
    assert ep.length == 24
    assert ep.titles["en"] == "Romance Dawn"


def test_episode_special_string_number() -> None:
    ep = AniDBEpisode(episode_number="S1", episode_type=2)
    assert ep.episode_number == "S1"


# =============================================================================
# AniDBCreator
# =============================================================================


def test_creator_minimal() -> None:
    c = AniDBCreator()
    assert c.id is None
    assert c.name is None
    assert c.role is None


def test_creator_full() -> None:
    c = AniDBCreator(id=10, name="Eiichiro Oda", role="Original Work")
    assert c.role == "Original Work"


# =============================================================================
# AniDBRelatedAnime
# =============================================================================


def test_related_anime_required_fields() -> None:
    r = AniDBRelatedAnime(id=2, relation_type="Sequel")
    assert r.id == 2
    assert r.title is None


def test_related_anime_missing_id_raises() -> None:
    with pytest.raises(ValidationError):
        AniDBRelatedAnime(relation_type="Sequel")  # id is required


# =============================================================================
# AniDBCategory
# =============================================================================


def test_category_defaults() -> None:
    c = AniDBCategory(name="Action")
    assert c.hentai is False
    assert c.weight == 0
    assert c.id is None


def test_category_hentai_flag() -> None:
    c = AniDBCategory(name="Ecchi", hentai=True, weight=400)
    assert c.hentai is True


# =============================================================================
# AniDBRatings
# =============================================================================


def test_ratings_all_none() -> None:
    r = AniDBRatings()
    assert r.permanent is None
    assert r.permanent_count == 0
    assert r.review is None


def test_ratings_full() -> None:
    r = AniDBRatings(
        permanent=8.33,
        permanent_count=9547,
        temporary=8.58,
        temporary_count=10282,
        review=8.68,
        review_count=20,
    )
    assert r.permanent == 8.33
    assert r.review_count == 20


# =============================================================================
# AniDBExternalResource
# =============================================================================


def test_external_resource_urls_only() -> None:
    r = AniDBExternalResource(type="4", urls=["http://example.com"])
    assert r.type == "4"
    assert r.identifiers == []


def test_external_resource_identifiers_only() -> None:
    r = AniDBExternalResource(type="2", identifiers=["21"])
    assert r.urls == []
    assert r.identifiers == ["21"]


# =============================================================================
# AniDBAnime
# =============================================================================


def test_anime_minimal() -> None:
    a = AniDBAnime(id=69)
    assert a.id == 69
    assert a.restricted is False
    assert a.episode_count == 0
    assert a.synonyms == []
    assert a.title_others == {}
    assert a.ratings is None


def test_anime_instances_independent_lists() -> None:
    """Pydantic v2 must not share mutable list defaults between instances."""
    a = AniDBAnime(id=1)
    b = AniDBAnime(id=2)
    a.tags.append("action")
    assert b.tags == []


def test_anime_restricted_flag() -> None:
    a = AniDBAnime(id=99, restricted=True)
    assert a.restricted is True


# =============================================================================
# AniDBCharacterPage
# =============================================================================


def test_character_page_minimal() -> None:
    p = AniDBCharacterPage()
    assert p.name_main is None
    assert p.abilities == []


def test_character_page_extra_forbidden() -> None:
    import pytest

    with pytest.raises(Exception):
        AniDBCharacterPage(name_main="Luffy", unknown_field="extra")
