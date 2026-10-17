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
