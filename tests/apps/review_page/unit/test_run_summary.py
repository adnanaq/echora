import dataclasses
from datetime import UTC, datetime

from review_page.file_review_case_source import convert_run_result_to_message
from review_page.models import RunSummary
from review_page.run_summary import build_run_summary
from seed_cleaning import SeedCleaningRunResult


def _run_result() -> SeedCleaningRunResult:
    return SeedCleaningRunResult(
        seed_file="assets/seed_data/seed.json",
        started_at=datetime(2026, 11, 1, 13, 58, 37, tzinfo=UTC),
        finished_at=datetime(2026, 11, 1, 13, 58, 39, tzinfo=UTC),
        correction_counts={"pages moved to a seed entry": 4, "review cases open": 0},
        corrected_entry_counts={"entries removed, every page moved out": 0},
        linked_page_counts={"linked pages not found": 376},
        rejected_answer_case_names=("linked page: a",),
        conflicting_linked_page_keys=(),
        unused_answer_case_names=("split work: c",),
        case_names=("linked page: a", "split work: b"),
        new_case_names=("split work: b",),
        original_answers={"linked page: a": {"anilist:anime:1": "skip"}},
    )


def test_build_run_summary_reads_every_field_of_run_result() -> None:
    result = _run_result()
    summary = build_run_summary(convert_run_result_to_message(result))
    assert dataclasses.asdict(summary) == result.model_dump()


def test_build_run_summary_keeps_count_order() -> None:
    summary = build_run_summary(convert_run_result_to_message(_run_result()))
    assert list(summary.correction_counts) == [
        "pages moved to a seed entry",
        "review cases open",
    ]


def test_run_summary_fields_match_run_result_fields() -> None:
    assert [field.name for field in dataclasses.fields(RunSummary)] == list(
        SeedCleaningRunResult.model_fields
    )
