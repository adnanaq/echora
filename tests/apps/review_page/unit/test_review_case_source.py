import operator

from review_page.file_review_case_source import convert_run_result_to_message
from review_page.review_case_messages import convert_case_to_message
from review_page.review_case_source import (
    ConvertedReviewCases,
    load_cases,
    load_last_run_summary,
)
from seed_cleaning import SeedCleaningRunResult


def test_load_cases_returns_source_cases_as_models(source) -> None:
    assert load_cases(source) == list(source.cases.values())


def test_converted_review_cases_same_messages_reuse_models(review_cases) -> None:
    case_messages = [convert_case_to_message(case) for case in review_cases.values()]
    converted_review_cases = ConvertedReviewCases()
    first_cases = converted_review_cases.convert_cases(case_messages)
    same_message_cases = converted_review_cases.convert_cases(list(case_messages))
    new_message_cases = converted_review_cases.convert_cases(
        [convert_case_to_message(case) for case in review_cases.values()]
    )
    assert (
        first_cases == list(review_cases.values()),
        all(map(operator.is_, first_cases, same_message_cases)),
        any(map(operator.is_, first_cases, new_message_cases)),
        new_message_cases == first_cases,
    ) == (True, True, False, True)


def test_load_last_run_summary_before_first_run_returns_none(source) -> None:
    assert load_last_run_summary(source) is None


def test_load_last_run_summary_reads_run_result(source) -> None:
    source.run_result = convert_run_result_to_message(
        SeedCleaningRunResult.model_validate(
            {
                "seed_file": "assets/seed_data/seed.json",
                "started_at": "2026-11-01T13:58:37+00:00",
                "finished_at": "2026-11-01T13:58:39+00:00",
                "correction_counts": {},
                "corrected_entry_counts": {},
                "linked_page_counts": {},
                "rejected_answer_case_names": [],
                "conflicting_linked_page_keys": [],
                "unused_answer_case_names": [],
                "case_names": ["split work: b"],
                "new_case_names": ["split work: b"],
                "original_answers": {},
            }
        )
    )

    run_summary = load_last_run_summary(source)

    assert run_summary is not None
    assert run_summary.new_case_names == ("split work: b",)
