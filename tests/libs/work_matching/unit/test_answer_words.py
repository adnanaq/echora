import pytest
from work_matching import (
    ReviewCaseKind,
    list_group_page_choices,
    list_linked_page_options,
)
from work_matching.answer_words import (
    MAIN_MARK,
    NO_MARK,
    PART_MARK,
    describe_answer_word,
)


@pytest.mark.parametrize(
    "answer_word",
    [
        "work 1",
        "work 2",
        "work 12",
        "other work",
        NO_MARK,
        MAIN_MARK,
        PART_MARK,
        "add",
        "add as part",
        "skip",
    ],
)
def test_describe_answer_word_known_word_has_meaning(answer_word) -> None:
    assert describe_answer_word(answer_word)


def test_describe_answer_word_every_later_work_has_one_meaning() -> None:
    assert describe_answer_word("work 2") == describe_answer_word("work 12")


def test_describe_answer_word_named_entry_names_reference_page() -> None:
    assert "mal:anime:1" in describe_answer_word("add as part to mal:anime:1")


def test_describe_answer_word_unknown_word_has_no_meaning() -> None:
    assert describe_answer_word("merge everything") == ""


def test_choice_option_description_reads_answer_word_meaning(review_cases) -> None:
    options = [
        option
        for choices in list_group_page_choices(review_cases["atashinchi_next"], {})
        for option in (*choices.work_options, *choices.mark_options)
    ] + list(list_linked_page_options(review_cases["nami"]))
    assert all(
        option.description == describe_answer_word(option.value) for option in options
    )
    assert all(option.description for option in options)


def test_review_case_kind_description_every_kind_has_meaning() -> None:
    assert all(kind.description for kind in ReviewCaseKind)
