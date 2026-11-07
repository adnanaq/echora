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
- Atomic line updates: Validated answers are persisted together through
  ``ReviewFile.save_answers``, which acquires the review file lock and rewrites
  only modified lines.
- Provider conflict warnings: Linked-page answers that attach whole pages from
  the same provider to a single work entry are saved with a warning so the
  reviewer can resolve the conflict before running the pipeline.
- Isolated run subprocess: Starting a run spawns ``clean_seed.py --no-fetch`` in
  an independent process group so pipeline memory usage stays outside the web
  server and server restarts do not terminate active runs.
- Bounded run logs: Subprocess output appends to ``logs/review_page_runs.log``,
  rotating to ``review_page_runs.log.1`` once file size exceeds
  ``RUN_LOG_LIMIT_BYTES``.
- Synchronized run lifecycle: A reentrant lock serializes run startup, process ID
  tracking in ``review_run.pid``, and exit code collection in
  ``review_run.exit_code`` so concurrent requests never launch duplicate runs.
- Cross-process run detection: Only the server process that spawned a run polls
  the child PID via ``waitpid``, while external or pre-restart runs are detected
  by checking whether ``review.jsonl.lock`` is held.
- Consistent seed release selection: Runs use the explicitly configured seed file
  or fall back to the seed file recorded by the previous run so releases never
  switch unexpectedly.
"""

import json
import os
import subprocess
import sys
import threading
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
from review_page.review_case_loader import ReviewCaseLoader
from review_page.review_case_messages import (
    list_ordered_mappings,
)
from review_page.review_case_source import ReviewCaseSource
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
    ReviewFile,
    is_review_file_locked,
    list_named_entry_page_keys,
    replace_file_lines,
    resolve_named_entries,
)

_PROCESS_ID_FILE_NAME = "review_run.pid"
_EXIT_CODE_FILE_NAME = "review_run.exit_code"
_RUN_LOG_PATH = Path("logs") / "review_page_runs.log"
# A run log past this size moves to ``review_page_runs.log.1`` before the next
# run, replacing the older copy, so the logs never take more than about twice
# this. One run writes a few kilobytes.
RUN_LOG_LIMIT_BYTES = 1_000_000


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
        self._clean_seed_script = clean_seed_script
        self._seed_file = seed_file
        self._python_executable = python_executable
        self._review_file_path = data_dir / REVIEW_FILE_NAME
        self._review_case_loader = ReviewCaseLoader(self._review_file_path)
        self._run_lock = threading.RLock()

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
                case = resolve_named_entries(
                    case, load_seed_entries(self._find_seed_file()), answer
                )
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
        with self._run_lock:
            if self.get_run_state().is_running:
                return review_cases_pb2.RunState(is_running=True)
            run_process = self._start_run_process(self._find_seed_file())
            (self._data_dir / _EXIT_CODE_FILE_NAME).unlink(missing_ok=True)
            replace_file_lines(
                self._data_dir / _PROCESS_ID_FILE_NAME, [str(run_process.pid)]
            )
        return review_cases_pb2.RunState(is_running=True)

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
        with self._run_lock:
            return self._read_run_state()

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
                reference_page=entry["reference_page"], title=entry["title"]
            )
            for entry in document.get("doubled_provider_entries", [])
        }
        for candidate in document.get("split_work_candidates", []):
            for entry in candidate["entries"]:
                entries_by_reference_page[entry["reference_page"]] = (
                    review_cases_pb2.CheckedEntry(
                        reference_page=entry["reference_page"],
                        title=entry["title"],
                        type=entry.get("type"),
                        year=entry.get("year"),
                    )
                )
        return list(entries_by_reference_page.values())

    def _read_run_state(self) -> review_cases_pb2.RunState:
        """Inspect the active run process or lock file and record completed exit codes.

        Assumes the caller holds ``self._run_lock``.

        Returns:
            The current run state with active status and any recorded exit code.
        """
        process_id_path = self._data_dir / _PROCESS_ID_FILE_NAME
        if process_id_path.exists():
            is_running, exit_code = _check_process(
                int(process_id_path.read_text(encoding="utf-8"))
            )
            if is_running:
                return review_cases_pb2.RunState(is_running=True)
            if exit_code is not None:
                replace_file_lines(
                    self._data_dir / _EXIT_CODE_FILE_NAME, [str(exit_code)]
                )
                process_id_path.unlink(missing_ok=True)
        run_state = review_cases_pb2.RunState(
            is_running=is_review_file_locked(self._review_file_path)
        )
        exit_code_path = self._data_dir / _EXIT_CODE_FILE_NAME
        if exit_code_path.exists():
            run_state.last_exit_code = int(exit_code_path.read_text(encoding="utf-8"))
        return run_state

    def _start_run_process(self, seed_file: Path) -> subprocess.Popen[bytes]:
        """Spawn ``clean_seed.py --no-fetch`` for the given seed file with log rotation.

        Rotates ``review_page_runs.log`` to ``review_page_runs.log.1`` when its
        size exceeds ``RUN_LOG_LIMIT_BYTES``.

        Args:
            seed_file: Path to the seed release file to clean.

        Returns:
            The spawned subprocess handle.
        """
        run_log_path = self._data_dir / _RUN_LOG_PATH
        run_log_path.parent.mkdir(parents=True, exist_ok=True)
        if run_log_path.exists() and run_log_path.stat().st_size > RUN_LOG_LIMIT_BYTES:
            run_log_path.replace(run_log_path.with_name(f"{run_log_path.name}.1"))
        with run_log_path.open("a", encoding="utf-8") as run_log:
            # The script and its arguments are fixed; no shell runs them.
            return subprocess.Popen(  # noqa: S603
                [
                    self._python_executable,
                    str(self._clean_seed_script),
                    str(seed_file),
                    "--no-fetch",
                    "--data-dir",
                    str(self._data_dir),
                ],
                stdout=run_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

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


def _check_process(process_id: int) -> tuple[bool, int | None]:
    """Check whether a child run process spawned by this server is still executing.

    Reaps finished child processes via ``os.waitpid`` to collect their exit code
    and avoid zombie processes. Process IDs not spawned by the current server
    process raise ``ChildProcessError`` and are treated as not running here so
    the caller falls back to checking the review file lock.

    Args:
        process_id: The process ID read from ``review_run.pid``.

    Returns:
        A tuple of ``(is_running, exit_code)`` where ``exit_code`` is None while
        running or when the process is not a child of this server.
    """
    try:
        ended_process_id, wait_status = os.waitpid(process_id, os.WNOHANG)
    except ChildProcessError:
        return False, None
    if ended_process_id == 0:
        return True, None
    return False, os.waitstatus_to_exitcode(wait_status)
