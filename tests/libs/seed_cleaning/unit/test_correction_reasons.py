from seed_cleaning.correction_reasons import (
    ANSWER_REASON_SENTENCE,
    describe_correction_reason,
)
from seed_cleaning.seed_corrections import REVIEW_ANSWER_REASON, SeedCorrection
from work_matching import DoubledProviderOutcome, LinkedPageReason


def _correction(reason: str, *, is_added_page: bool = False) -> SeedCorrection:
    return SeedCorrection(
        entry_reference_page="mal:anime:1",
        page_key="kitsu:anime:1",
        address="https://kitsu.app/anime/1",
        placement_word="work 1",
        is_added_page=is_added_page,
        reason=reason,
    )


def test_describe_correction_reason_answer_says_answer() -> None:
    assert describe_correction_reason(_correction(REVIEW_ANSWER_REASON)) == (
        ANSWER_REASON_SENTENCE
    )


def test_describe_correction_reason_kept_page_gives_doubled_provider_outcome() -> None:
    assert describe_correction_reason(_correction("SAME_PAGE")) == (
        DoubledProviderOutcome.SAME_PAGE.value
    )


def test_describe_correction_reason_added_page_gives_linked_page_reason() -> None:
    assert (
        describe_correction_reason(_correction("CHECKS_PASSED", is_added_page=True))
        == LinkedPageReason.CHECKS_PASSED.value
    )


def test_describe_correction_reason_unknown_name_shown_as_written() -> None:
    assert describe_correction_reason(_correction("NEWER_RULE")) == "NEWER_RULE"
