import pytest
from work_matching import ReviewCaseKind
from work_matching.answer_words import describe_answer_word


@pytest.mark.parametrize(
    "answer_word",
    [
        "work 1",
        "work 2",
        "work 12",
        "other work",
        "dropped",
        "add",
        "add as partial",
        "skip",
    ],
)
def test_describe_answer_word_known_word_has_meaning(answer_word) -> None:
    assert describe_answer_word(answer_word)


def test_describe_answer_word_every_later_work_has_one_meaning() -> None:
    assert describe_answer_word("work 2") == describe_answer_word("work 12")


def test_describe_answer_word_named_entry_names_reference_page() -> None:
    assert "mal:anime:1" in describe_answer_word("add as partial to mal:anime:1")


def test_describe_answer_word_unknown_word_has_no_meaning() -> None:
    assert describe_answer_word("merge everything") == ""


def test_review_case_kind_description_every_kind_has_meaning() -> None:
    assert all(kind.description for kind in ReviewCaseKind)
