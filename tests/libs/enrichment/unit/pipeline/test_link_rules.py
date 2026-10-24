import pytest
from enrichment.pipeline.link_rules import find_anime_id_page_key, merged_anime_id


@pytest.mark.parametrize(
    ("sources", "page_key"),
    [
        (
            [
                "https://kitsu.app/anime/12",
                "https://myanimelist.net/anime/21/One_Piece",
                "https://anilist.co/anime/21",
            ],
            "mal:anime:21",
        ),
        (
            ["https://kitsu.app/anime/12", "https://anilist.co/anime/21"],
            "anilist:anime:21",
        ),
        (
            ["https://myanimelist.net/anime/9", "https://myanimelist.net/anime/10"],
            "mal:anime:10",
        ),
        (["https://animenewsnetwork.com/encyclopedia/anime.php?id=836"], None),
        ([], None),
    ],
)
def test_find_anime_id_page_key_takes_first_provider_in_id_order(
    sources: list[str], page_key: str | None
) -> None:
    assert find_anime_id_page_key(sources) == page_key


def test_merged_anime_id_same_page_key_gives_same_id() -> None:
    assert merged_anime_id(
        ["https://kitsu.app/anime/12", "https://myanimelist.net/anime/21"]
    ) == merged_anime_id(["https://myanimelist.net/anime/21/One_Piece"])


def test_merged_anime_id_slug_decoration_gives_same_id() -> None:
    assert merged_anime_id(
        ["https://myanimelist.net/anime/21/One_Piece"]
    ) == merged_anime_id(["https://myanimelist.net/anime/21"])


def test_merged_anime_id_ann_only_work_ignores_address_spelling() -> None:
    assert merged_anime_id(
        ["https://animenewsnetwork.com/encyclopedia/anime.php?id=836"]
    ) == merged_anime_id(
        ["http://www.animenewsnetwork.com/encyclopedia/anime.php?id=836"]
    )


def test_merged_anime_id_no_source_raises() -> None:
    with pytest.raises(ValueError, match="Cannot derive an anime id"):
        merged_anime_id([])
