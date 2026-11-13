import pytest
from work_matching.models import (
    DoubledProviderOutcome,
    DoubledProviderReviewReason,
    LinkedPageReason,
)
from work_matching.page_details_comparison import MalPageComparison
from work_matching.review_case_reasons import describe_case_reasons
from work_matching.split_work_check import SplitWorkReviewReason


def test_describe_case_reasons_doubled_provider_gives_decision_sentences(
    review_cases,
) -> None:
    provider = review_cases["gochuumon_movie"].decisions[0].provider
    assert describe_case_reasons(review_cases["gochuumon_movie"]) == (
        f"{provider}: {DoubledProviderOutcome.PAGES_OF_OTHER_WORKS.value}",
        f"{provider}: {DoubledProviderReviewReason.MOVE_NOT_SURE.value}",
    )


def test_describe_case_reasons_several_doubled_providers_name_provider(
    review_cases,
) -> None:
    sentences = describe_case_reasons(review_cases["aware_meisaku_kun"])
    assert len(set(sentences)) == len(sentences)


def test_describe_case_reasons_linked_page_gives_decision_reason(
    review_cases,
) -> None:
    assert describe_case_reasons(review_cases["nami"]) == (
        LinkedPageReason.DETAILS_DIFFER.value,
    )


def test_describe_case_reasons_split_work_gives_split_work_sentence_last(
    review_cases,
) -> None:
    sentences = describe_case_reasons(review_cases["aware_meisaku_kun"])
    assert sentences[-1] == SplitWorkReviewReason.YEAR_DIFFERS.value


@pytest.mark.parametrize(
    "case_name", ["atashinchi_next", "gochuumon_movie", "nami", "aware_meisaku_kun"]
)
def test_describe_case_reasons_one_sentence_per_reason(
    review_cases, case_name: str
) -> None:
    case = review_cases[case_name]
    sentences = describe_case_reasons(case)
    assert len(sentences) == len(case.reasons)
    assert all(sentence not in case.reasons for sentence in sentences)


def test_describe_case_reasons_split_work_and_mal_reason_names_never_overlap() -> None:
    assert not {reason.name for reason in SplitWorkReviewReason} & {
        comparison.name for comparison in MalPageComparison
    }
