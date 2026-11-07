from review_page.file_review_case_source import convert_run_result_to_message
from review_page.review_case_source import load_cases, load_last_run_summary
from seed_cleaning import SeedCleaningRunResult


def test_load_cases_returns_source_cases_as_models(source) -> None:
    assert load_cases(source) == list(source.cases.values())


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
