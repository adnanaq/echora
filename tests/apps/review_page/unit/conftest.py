from collections.abc import Sequence
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from google.protobuf import json_format
from review_page.app import create_app
from review_page.file_review_case_source import (
    FileReviewCaseSource,
    RunSeedFileUnknownError,
)
from review_page.review_case_messages import convert_case_to_message
from review_page.review_case_source import ReviewCaseSource
from shared_proto.v1 import review_cases_pb2
from work_matching import ReviewCase, ReviewFileLockedError, UnknownCaseError

_REVIEW_CASE_FIXTURE = (
    Path(__file__).parents[3]
    / "libs"
    / "work_matching"
    / "unit"
    / "fixtures"
    / "review_cases.jsonl"
)


_FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def correction_lines() -> list[str]:
    return (
        (_FIXTURES / "corrections.jsonl")
        .read_text(encoding="utf-8")
        .splitlines(keepends=True)
    )


@pytest.fixture(scope="session")
def entries_to_check_text() -> str:
    return (_FIXTURES / "entries_to_check.json").read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def fixture_source(tmp_path_factory, correction_lines, entries_to_check_text):
    data_dir = tmp_path_factory.mktemp("seed_cleaning")
    (data_dir / "corrections.jsonl").write_text(
        "".join(correction_lines), encoding="utf-8"
    )
    (data_dir / "entries_to_check.json").write_text(
        entries_to_check_text, encoding="utf-8"
    )
    return FileReviewCaseSource(data_dir, data_dir / "clean_seed.py")


@pytest.fixture(scope="session")
def correction_messages(fixture_source):
    return fixture_source.list_corrections()


@pytest.fixture(scope="session")
def checked_entry_messages(fixture_source):
    return fixture_source.list_checked_entries()


@pytest.fixture(scope="session")
def review_case_lines() -> list[str]:
    return _REVIEW_CASE_FIXTURE.read_text(encoding="utf-8").splitlines(keepends=True)


@pytest.fixture(scope="session")
def review_cases(review_case_lines) -> dict[str, ReviewCase]:
    cases = [ReviewCase.model_validate_json(line) for line in review_case_lines]
    page_key_by_name = {
        "atashinchi_next": "mal:anime:63013",
        "gochuumon_movie": "mal:anime:61115",
        "nami": "kitsu:anime:10561",
        "aware_meisaku_kun": "mal:anime:32956",
    }
    return {
        name: next(case for case in cases if page_key in case.case_name)
        for name, page_key in page_key_by_name.items()
    }


class FakeReviewCaseSource(ReviewCaseSource):
    def __init__(self, cases: Sequence[ReviewCase]) -> None:
        self.cases = {case.case_name: case for case in cases}
        self.is_locked = False
        self.is_case_dropped_by_run = False
        self.is_seed_file_known = True
        self.is_running = False
        self.run_result: review_cases_pb2.SeedCleaningRunResult | None = None
        self.corrections: list[review_cases_pb2.SeedCorrection] = []
        self.checked_entries: list[review_cases_pb2.CheckedEntry] = []

    def list_cases(self) -> list[review_cases_pb2.ReviewCase]:
        return [convert_case_to_message(case) for case in self.cases.values()]

    def get_case(self, case_name: str) -> review_cases_pb2.ReviewCase | None:
        case = self.cases.get(case_name)
        return None if case is None else convert_case_to_message(case)

    def get_last_run_result(self) -> review_cases_pb2.SeedCleaningRunResult | None:
        return self.run_result

    def list_corrections(self) -> list[review_cases_pb2.SeedCorrection]:
        return list(self.corrections)

    def list_checked_entries(self) -> list[review_cases_pb2.CheckedEntry]:
        return list(self.checked_entries)

    def save_answers(
        self, case_answers: Sequence[review_cases_pb2.CaseAnswer]
    ) -> list[review_cases_pb2.RuleCheckResult]:
        if self.is_locked:
            raise ReviewFileLockedError(__file__)
        if self.is_case_dropped_by_run:
            raise UnknownCaseError([answer.case_name for answer in case_answers])
        results = []
        for case_answer in case_answers:
            answer = (
                json_format.MessageToDict(case_answer.answer)
                if case_answer.HasField("answer")
                else None
            )
            case = self.cases[case_answer.case_name]
            broken_rules = (
                () if answer is None else case.list_broken_answer_rules(answer)
            )
            if not broken_rules:
                self.cases[case.case_name] = case.model_copy(update={"answer": answer})
            results.append(
                review_cases_pb2.RuleCheckResult(
                    case_name=case.case_name,
                    is_saved=not broken_rules,
                    broken_rule_sentences=[rule.value for rule in broken_rules],
                )
            )
        return results

    def start_run(self) -> review_cases_pb2.RunState:
        if not self.is_seed_file_known:
            raise RunSeedFileUnknownError
        self.is_running = True
        return review_cases_pb2.RunState(is_running=True)

    def get_run_state(self) -> review_cases_pb2.RunState:
        return review_cases_pb2.RunState(is_running=self.is_running)


@pytest.fixture
def source(review_cases) -> FakeReviewCaseSource:
    return FakeReviewCaseSource(list(review_cases.values()))


@pytest.fixture
def client(source) -> TestClient:
    return TestClient(
        create_app(source),
        base_url="http://127.0.0.1:8003",
        headers={"HX-Request": "true"},
    )
