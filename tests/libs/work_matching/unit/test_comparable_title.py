import pytest
from work_matching.comparable_title import comparable_title


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("CUE!", "Cue!"),
        ("Ranma 1/2", "Ranma ½"),
        ("Pokémon", "Pokemon"),
        ("Fullmetal  Alchemist:\tBrotherhood", "fullmetal alchemist brotherhood"),
        ("Shingeki no Kyojin 2nd Season", "Shingeki no Kyojin Season 2"),
        ("Re:Zero 3rd season", "re zero season 3"),
        ("ＯＮＥ ＰＩＥＣＥ", "One Piece"),
    ],
)
def test_comparable_title_cosmetic_difference_gives_same_title(
    first: str, second: str
) -> None:
    assert comparable_title(first) == comparable_title(second)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("ガンダム", "カンタム"),
        ("魔道祖师", "魔道祖師"),
        ("Season 2", "Season 3"),
        ("Hikari", "Hikaru"),
    ],
)
def test_comparable_title_different_titles_stay_different(
    first: str, second: str
) -> None:
    assert comparable_title(first) != comparable_title(second)


def test_comparable_title_non_latin_title_keeps_letters() -> None:
    assert comparable_title("魔道祖师 前尘篇") == "魔道祖师 前尘篇"


def test_comparable_title_punctuation_only_gives_empty_title() -> None:
    assert comparable_title("!?…") == ""
