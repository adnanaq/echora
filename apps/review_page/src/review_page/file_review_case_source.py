"""Read and persist review cases in local seed-cleaning files and manage pipeline runs.

Implements ``ReviewCaseSource`` over the local ``review.jsonl`` file, correction
outputs, and the ``scripts/clean_seed.py`` subprocess.

Each operational rule reads `<rule>: <reason>`:
- File-backed reads: Cases come from ``review.jsonl`` through ``ReviewCaseLoader``,
  which converts them once per file version and reads the file again whenever
  it changes, so the page always shows what the file holds.
- Pre-save rule validation: Every submitted answer is validated through
  ``answer_checks`` using domain placement rules before writing to disk.
- Seed entry resolution: Answers referencing an external seed entry via
  ``other work`` are resolved against the active seed release via ``seed_entries``
  so unknown or conflicting target entries are rejected before saving.
- Dead pages: A key whose page the last run dropped as dead names no entry,
  as in the run, so saving refuses it. The dead keys come from the last run's
  corrections, so the large page store is never loaded.
- Atomic line updates: Validated answers are persisted together through
  ``ReviewFile.save_answers``, which acquires the review file lock and rewrites
  only modified lines.
- Provider conflict warnings: Linked-page answers that attach whole pages from
  the same provider to a single work entry are saved with a warning so the
  reviewer can resolve the conflict before running the pipeline.
- Pipeline runs: Starting a run and reporting whether one is active belong to
  ``SeedCleaningRunProcess``, which keeps the run process apart from the web
  server.
- Consistent seed release selection: Runs use the explicitly configured seed file
  or fall back to the seed file recorded by the previous run so releases never
  switch unexpectedly.
"""

import json
import sys
from collections.abc import Sequence
from pathlib import Path

from google.protobuf import json_format
from pydantic import JsonValue
from review_page.answer_checks import (
    check_case_answer,
    list_conflict_warnings,
    read_case_answer,
    read_seen_saved_answers,
)
from review_page.models import InvalidNamedPageKeyReason
from review_page.review_case_loader import ReviewCaseLoader
from review_page.review_case_messages import (
    list_ordered_mappings,
)
from review_page.review_case_source import ReviewCaseSource, SeedFileUnavailableError
from review_page.seed_cleaning_run_process import SeedCleaningRunProcess
from review_page.seed_entries import load_seed_entries
from seed_cleaning import (
    CORRECTIONS_FILE_NAME,
    ENTRIES_TO_CHECK_FILE_NAME,
    REVIEW_ANSWER_REASON,
    REVIEW_FILE_NAME,
    SeedCleaningRunResult,
    SeedCorrection,
    describe_correction_reason,
    find_correction_change,
    read_seed_cleaning_run_result,
)
from shared_proto.v1 import review_cases_pb2
from work_matching import (
    KnownWorkEntriesFromSeed,
    ReviewCase,
    ReviewFile,
    build_target_entry_suggestion,
    list_named_entry_page_keys,
    resolve_named_entries,
)


class RunSeedFileUnknownError(RuntimeError):
    """Indicate that no seed file is configured or recorded for a pipeline run."""

    def __init__(self) -> None:
        """Initialize the error with instructions for configuring a seed file."""
        super().__init__(
            "No seed file to clean: set REVIEW_PAGE_SEED_FILE, or run "
            "scripts/clean_seed.py once so its result names the file"
        )


class FileReviewCaseSource(ReviewCaseSource):
    """Manage review cases, saved answers, and pipeline runs using local files."""

    def __init__(
        self,
        data_dir: Path,
        clean_seed_script: Path,
        *,
        seed_file: Path | None = None,
        python_executable: str = sys.executable,
    ) -> None:
        """Initialize the file-backed review case source with paths and execution settings.

        Args:
            data_dir: Directory containing ``review.jsonl`` and run state files.
            clean_seed_script: Path to ``scripts/clean_seed.py`` executed on run.
            seed_file: Explicit seed release file path, or None to use the path
                recorded by the previous pipeline run.
            python_executable: Python interpreter executable used to launch runs.
        """
        self._data_dir = data_dir
        self._seed_file = seed_file
        self._review_file_path = data_dir / REVIEW_FILE_NAME
        self._review_case_loader = ReviewCaseLoader(self._review_file_path)
        self._run_process = SeedCleaningRunProcess(
            data_dir, self._review_file_path, clean_seed_script, python_executable
        )

    def list_cases(self) -> list[review_cases_pb2.ReviewCase]:
        """Read all review cases from the local review file in file order.

        Returns:
            All review cases converted to protobuf messages.
        """
        return list(self._review_case_loader.load_cases().case_messages)

    def get_case(self, case_name: str) -> review_cases_pb2.ReviewCase | None:
        """Read a single review case by name from the local review file.

        Args:
            case_name: The unique identifier of the review case to look up.

        Returns:
            The matching review case message, or None when absent from the file.
        """
        return next(
            (
                case_message
                for case_message in self._review_case_loader.load_cases().case_messages
                if case_message.case_name == case_name
            ),
            None,
        )

    def get_last_run_result(self) -> review_cases_pb2.SeedCleaningRunResult | None:
        """Read the ``SeedCleaningRunResult`` recorded by the most recent pipeline run.

        Returns:
            The last run result message, or None before the first run completes.
        """
        run_result = read_seed_cleaning_run_result(self._data_dir)
        return None if run_result is None else convert_run_result_to_message(run_result)

    def save_answers(
        self, case_answers: Sequence[review_cases_pb2.CaseAnswer]
    ) -> list[review_cases_pb2.RuleCheckResult]:
        """Validate submitted answers against placement rules and write valid answers to disk.

        Args:
            case_answers: Submitted case answers to validate and save; an unset
                answer clears the existing saved answer.

        Returns:
            One validation check result per submitted answer in request order.

        Raises:
            ReviewFileLockedError: A pipeline run holds the review file lock.
            UnknownCaseError: A concurrent run removed a target case between
                validation and persistence.
            SavedAnswerChangedError: Another session modified a case's saved
                answer between validation and persistence.
            SeedFileUnavailableError: An answer names a seed entry and the seed
                file is missing or unknown.
        """
        cases_by_name = {
            case.case_name: case for case in self._review_case_loader.load_cases().cases
        }
        seen_saved_answers = read_seen_saved_answers(case_answers)
        answers_to_save: dict[str, JsonValue] = {}
        check_results: list[review_cases_pb2.RuleCheckResult] = []
        for case_answer in case_answers:
            answer = read_case_answer(case_answer)
            case = cases_by_name.get(case_answer.case_name)
            if case is not None and list_named_entry_page_keys(answer):
                case = self._resolve_live_named_entries(case, answer)
            check_result = check_case_answer(
                case, case_answer.case_name, answer, seen_saved_answers
            )
            if check_result.is_saved:
                answers_to_save[case_answer.case_name] = answer
            check_results.append(check_result)
        if answers_to_save:
            ReviewFile(self._review_file_path).save_answers(
                answers_to_save,
                seen_saved_answers={
                    case_name: seen_answer
                    for case_name, seen_answer in seen_saved_answers.items()
                    if case_name in answers_to_save
                },
            )
            warnings_by_case_name = list_conflict_warnings(
                cases_by_name, answers_to_save
            )
            for check_result in check_results:
                check_result.warnings.extend(
                    warnings_by_case_name.get(check_result.case_name, ())
                )
        return check_results

    def start_run(self) -> review_cases_pb2.RunState:
        """Launch ``clean_seed.py --no-fetch`` in a background subprocess if no run is active.

        Returns:
            The active run state marked as running.

        Raises:
            RunSeedFileUnknownError: Neither configured settings nor a previous
                run result specifies a seed file path.
        """
        return self._run_process.start_run(self._find_seed_file)

    def get_run_state(self) -> review_cases_pb2.RunState:
        """Report whether a pipeline run is active and the exit code of the last run.

        Each tracking rule reads `<rule>: <reason>`:
        - Child process polling: Runs spawned by this server process are tracked
          by PID; once finished, the exit code is persisted and the PID file removed.
        - File lock fallback: Runs started externally or before a server restart
          are detected while they hold the review file lock.
        - Foreign PID preservation: A PID file written by another server process
          is left untouched so the originating process can collect its exit code.

        Returns:
            The current run state message with active status and last exit code.
        """
        return self._run_process.read_run_state()

    def find_invalid_named_page_keys(
        self, page_keys: Sequence[str]
    ) -> dict[str, InvalidNamedPageKeyReason]:
        """Find the page keys an answer cannot name after ``other work``, and why.

        Without a seed file, only dead keys are found, since unknown keys cannot
        be told apart. Saving then refuses the answer with a notice instead.

        Args:
            page_keys: The page keys an answer names after ``other work``.

        Returns:
            Each invalid key mapped to the reason it is invalid. Valid keys are left out.
        """
        if not page_keys:
            return {}
        dead_page_keys = self._list_dead_page_keys()
        try:
            seed_entries: KnownWorkEntriesFromSeed | None = self._load_seed_entries()
        except SeedFileUnavailableError:
            seed_entries = None
        invalid_reasons: dict[str, InvalidNamedPageKeyReason] = {}
        for page_key in page_keys:
            if page_key in dead_page_keys:
                invalid_reasons[page_key] = InvalidNamedPageKeyReason.DEAD_PAGE
            elif seed_entries is not None and not seed_entries.find_by_page_key(
                page_key
            ):
                invalid_reasons[page_key] = InvalidNamedPageKeyReason.UNKNOWN_PAGE
        return invalid_reasons

    def find_named_entry_suggestions(
        self, page_keys: Sequence[str]
    ) -> dict[str, review_cases_pb2.TargetEntrySuggestion]:
        """Find the seed entry holding each page key, from the active seed file.

        Args:
            page_keys: The page keys to look up.

        Returns:
            Each key a seed entry holds, mapped to that entry's suggestion
            message; empty when the seed file cannot be read.
        """
        try:
            seed_entries = self._load_seed_entries()
        except SeedFileUnavailableError:
            return {}
        return {
            page_key: review_cases_pb2.TargetEntrySuggestion(
                **build_target_entry_suggestion(work_entry).model_dump()
            )
            for page_key in page_keys
            if (work_entry := seed_entries.find_by_page_key(page_key))
        }

    def list_corrections(self) -> list[review_cases_pb2.SeedCorrection]:
        """Read all seed corrections from the last run with change labels and reasons.

        Returns:
            All seed correction messages, or an empty list before the first run.
        """
        corrections_path = self._data_dir / CORRECTIONS_FILE_NAME
        if not corrections_path.exists():
            return []
        # Lines end at "\n" only: a value may hold U+2028 or U+0085
        # unescaped, which str.splitlines would also split at.
        with corrections_path.open(encoding="utf-8") as corrections_file:
            return [
                convert_correction_to_message(SeedCorrection.model_validate_json(line))
                for line in corrections_file
                if line.strip()
            ]

    def list_checked_entries(self) -> list[review_cases_pb2.CheckedEntry]:
        """Read deduplicated checked entries from ``entries_to_check.json``.

        Entries from split-work candidates include anime type and year metadata
        and therefore supersede doubled-provider entries for the same reference page.

        Returns:
            One checked entry message per reference page key, or an empty list
            before the first run.
        """
        entries_path = self._data_dir / ENTRIES_TO_CHECK_FILE_NAME
        if not entries_path.exists():
            return []
        document = json.loads(entries_path.read_text(encoding="utf-8"))
        entries_by_reference_page = {
            entry["reference_page"]: review_cases_pb2.CheckedEntry(
                entry_reference_page=entry["reference_page"], title=entry["title"]
            )
            for entry in document.get("doubled_provider_entries", [])
        }
        for candidate in document.get("split_work_candidates", []):
            for entry in candidate["entries"]:
                entries_by_reference_page[entry["reference_page"]] = (
                    review_cases_pb2.CheckedEntry(
                        entry_reference_page=entry["reference_page"],
                        title=entry["title"],
                        type=entry.get("type"),
                        year=entry.get("year"),
                    )
                )
        return list(entries_by_reference_page.values())

    def _resolve_live_named_entries(
        self, case: ReviewCase, answer: JsonValue
    ) -> ReviewCase:
        """Resolve an answer's named keys against the seed without its dead pages.

        Args:
            case: The review case the answer is for.
            answer: The answer as submitted.

        Returns:
            The case, each named key that a seed entry holds and that is not
            dead mapped to that entry's reference page.

        Raises:
            SeedFileUnavailableError: The seed file is unknown or missing.
        """
        resolved_case = resolve_named_entries(case, self._load_seed_entries(), answer)
        dead_page_keys = self._list_dead_page_keys()
        return resolved_case.model_copy(
            update={
                "named_entry_reference_pages": {
                    page_key: reference_page
                    for page_key, reference_page in (
                        resolved_case.named_entry_reference_pages.items()
                    )
                    if page_key not in dead_page_keys
                }
            }
        )

    def _list_dead_page_keys(self) -> set[str]:
        """List the page keys the last run dropped because their provider no longer has them.

        Returns:
            The dead page keys, or none before the first run.
        """
        return {
            correction.page_key
            for correction in self.list_corrections()
            if correction.is_dead_page
        }

    def _load_seed_entries(self) -> KnownWorkEntriesFromSeed:
        """Load the seed entries that named entries are looked up in.

        Returns:
            The seed's work entries, loaded once per seed file version.

        Raises:
            SeedFileUnavailableError: The seed file is unknown or missing.
        """
        try:
            return load_seed_entries(self._find_seed_file())
        except (RunSeedFileUnknownError, FileNotFoundError) as error:
            raise SeedFileUnavailableError from error

    def _find_seed_file(self) -> Path:
        """Resolve the seed file path from configuration or the previous run result.

        Returns:
            The resolved seed release file path.

        Raises:
            RunSeedFileUnknownError: No seed file is configured or recorded.
        """
        if self._seed_file is not None:
            return self._seed_file
        last_run_result = read_seed_cleaning_run_result(self._data_dir)
        if last_run_result is None:
            raise RunSeedFileUnknownError
        return Path(last_run_result.seed_file)


def convert_run_result_to_message(
    run_result: SeedCleaningRunResult,
) -> review_cases_pb2.SeedCleaningRunResult:
    """Convert a ``SeedCleaningRunResult`` model into its protobuf message.

    Args:
        run_result: The run result model written by the seed-cleaning pipeline.

    Returns:
        The corresponding protobuf message with ordered count entries preserved.
    """
    return json_format.ParseDict(
        list_ordered_mappings(run_result.model_dump(mode="json")),
        review_cases_pb2.SeedCleaningRunResult(),
    )


def convert_correction_to_message(
    correction: SeedCorrection,
) -> review_cases_pb2.SeedCorrection:
    """Convert a seed correction model into a protobuf message with display metadata.

    Populates the change category label, human-readable reason sentence, and
    whether the placement originated from a reviewer answer.

    Args:
        correction: A single seed correction model from the corrections file.

    Returns:
        The populated ``SeedCorrection`` protobuf message.
    """
    fields = {
        field: value
        for field, value in correction.model_dump(mode="json").items()
        if value is not None
    }
    message = json_format.ParseDict(fields, review_cases_pb2.SeedCorrection())
    if change := find_correction_change(correction):
        message.change = change
    message.reason_sentence = describe_correction_reason(correction)
    message.is_from_answer = correction.reason == REVIEW_ANSWER_REASON
    return message
