import pytest
from enrichment.sources.anilist.anilist_anime_models import AniListAnime, AniListTitle


@pytest.mark.parametrize("field", ["romaji", "english", "native"])
def test_anilist_title_empty_title_becomes_none(field: str) -> None:
    assert getattr(AniListTitle.model_validate({field: " "}), field) is None


def test_anilist_title_title_is_stripped() -> None:
    assert (
        AniListTitle.model_validate({"native": " ワンピース "}).native == "ワンピース"
    )


def test_anilist_anime_description_html_becomes_plain_text() -> None:
    anime = AniListAnime.model_validate(
        {"id": 21, "description": "Luffy&rsquo;s crew.<br><br>\n(Source: Crunchyroll)"}
    )

    assert anime.description == "Luffy’s crew.\n(Source: Crunchyroll)"
