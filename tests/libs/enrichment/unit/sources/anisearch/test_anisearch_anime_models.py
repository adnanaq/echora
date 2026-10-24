import pytest
from enrichment.sources.anisearch.anisearch_anime_models import AniSearchAnime


def _block(language: str, title: str, *native_titles: str) -> dict:
    return {"language": language, "title": title, "native_titles": list(native_titles)}


def _title_fields(*blocks: dict) -> tuple:
    anime = AniSearchAnime.model_validate({"title_blocks": list(blocks)})
    return anime.title, anime.title_english, anime.title_japanese, anime.titles


def test_anisearch_anime_japanese_work_takes_romaji_title_and_kanji_japanese_title() -> (
    None
):
    assert _title_fields(
        _block("ja", "Kami no Tou (2024)", "Kami no Tō (2024)", "神之塔 （2024）"),
        _block("en", "Tower of God: Season 2"),
        _block("de", "Tower of God: Staffel 2"),
    ) == (
        "Kami no Tou (2024)",
        "Tower of God: Season 2",
        "神之塔 （2024）",
        {"de": "Tower of God: Staffel 2"},
    )


def test_anisearch_anime_chinese_work_files_native_script_under_its_language() -> None:
    assert _title_fields(
        _block("zh", "Ba La La Xiao Mo Xian Zhi: Qiji Wubu", "巴啦啦小魔仙之奇迹舞步"),
        _block("en", "Balala the Fairies 2"),
    ) == (
        "Ba La La Xiao Mo Xian Zhi: Qiji Wubu",
        "Balala the Fairies 2",
        None,
        {"zh": "巴啦啦小魔仙之奇迹舞步"},
    )


def test_anisearch_anime_chinese_work_with_japanese_release_files_japanese_romaji_under_x_jat() -> (
    None
):
    assert _title_fields(
        _block("zh", "Mo Dao Zu Shi: Qianchen Pian", "魔道祖师 前尘篇"),
        _block(
            "ja",
            "Madou Soshi: Zenjin Hen",
            "Madō Soshi: Zenjin Hen",
            "魔道祖師 前塵編",
        ),
        _block("en", "The Founder of Diabolism"),
        _block("it", "The Founder of Diabolism"),
    ) == (
        "Mo Dao Zu Shi: Qianchen Pian",
        "The Founder of Diabolism",
        "魔道祖師 前塵編",
        {"zh": "魔道祖师 前尘篇", "x-jat": "Madou Soshi: Zenjin Hen"},
    )


def test_anisearch_anime_korean_language_code_becomes_ko() -> None:
    assert _title_fields(
        _block("kr", "Pororo Daemoheom", "뽀로로의 대모험"),
        _block("en", "Pororo to the Cookie Castle"),
        _block("fr", "Pororo au Royaume des Friandises"),
    )[3] == {"ko": "뽀로로의 대모험", "fr": "Pororo au Royaume des Friandises"}


def test_anisearch_anime_hong_kong_code_becomes_zh_hk() -> None:
    assert _title_fields(
        _block("hk", "Siu Sin", "小倩"),
        _block("en", "A Chinese Ghost Story"),
    )[3] == {"zh-HK": "小倩"}


def test_anisearch_anime_titles_repeating_another_title_left_out() -> None:
    assert _title_fields(
        _block(
            "ja", "RWBY: Hyousetsu Teikoku", "RWBY: Hyōsetsu Teikoku", "RWBY 氷雪帝国"
        ),
        _block("en", "RWBY: Ice Queendom"),
        _block("de", "RWBY: Ice Queendom"),
        _block("fr", "RWBY : Ice Queendom"),
        _block("es", "Rwby Ice Queendom 2"),
        _block("it", "rwby ice queendom 2"),
    )[3] == {"es": "Rwby Ice Queendom 2"}


def test_anisearch_anime_empty_title_blocks_gives_no_titles() -> None:
    assert _title_fields() == (None, None, None, {})


def test_anisearch_anime_without_title_blocks_keeps_given_titles() -> None:
    anime = AniSearchAnime(title="One Piece", title_japanese="ワンピース")
    assert (anime.title, anime.title_japanese, anime.titles) == (
        "One Piece",
        "ワンピース",
        {},
    )


@pytest.mark.parametrize(
    ("type_line", "episode_count"),
    [
        ("Type: TV-Series, 52 (~24 min, Total: ~20 h 48 min)", 52),
        ("Type: Web, 15 (~26 min)", 15),
        ("Type: TV-Series, 1200 (~24 min, Total: ~20 d)", 1200),
        ("Type: TV-Series, ? (~24 min)", None),
        ("Type: TV-Series", None),
        (None, None),
    ],
)
def test_anisearch_anime_type_line_gives_episode_count(
    type_line: str | None, episode_count: int | None
) -> None:
    assert AniSearchAnime.model_validate({"type": type_line}).episode_count == (
        episode_count
    )


def test_anisearch_anime_type_line_with_episode_count_keeps_type() -> None:
    anime = AniSearchAnime.model_validate({"type": "Type: Web, 15 (~26 min)"})
    assert (anime.type, anime.episode_count) == ("Web", 15)


def test_anisearch_anime_given_episode_count_kept() -> None:
    anime = AniSearchAnime.model_validate(
        {"type": "Type: TV-Series, 52 (~24 min)", "episode_count": 12}
    )
    assert anime.episode_count == 12
