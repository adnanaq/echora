"""The first review case source: seed cleaning's local review file and its run script.

Each item gives the rule, then the reason for it.

- Cases are read from ``review.jsonl`` on every call, not kept in memory, so
  the server holds no state and always shows what the file holds. Reading all
  635 cases of a release takes about 0.06 s.
- An answer is checked with ``ReviewCase.list_broken_answer_rules``, the run's
  own checks, before it is saved; answers that break a rule are not saved and
  come back with the rules.
- Answers that follow the rules are saved together through
  ``ReviewFile.save_answers``, which holds the file's lock and rewrites only
  their lines.
- Two linked-page answers adding whole pages of one provider to one entry are
  saved, with a warning, because the run adds neither page; the reviewer
  decides which to change.
- Run starts ``clean_seed.py --no-fetch`` as its own process, so its memory
  (about 0.9 GB for a release) is never the server's, and a server restart
  does not stop it. Its output is appended to ``logs/review_page_runs.log``.
- One run starts at a time: a lock covers the check for a running run, the
  start and the process id file, so two clicks or two tabs cannot start two
  runs. The same lock covers reading the run state, so when a run ends only
  one request collects its exit code, and every other request reads the exit
  code that request wrote.
- The run's process id is kept in ``review_run.pid``, and how it ended in
  ``review_run.exit_code``; both files are written in one step, so a reader
  never sees one half written. Only the server that started the run trusts
  the process id: after a restart the id may belong to another process. Any
  other run, and a run seen from another server, is followed through the
  review file's lock, which a run holds while it works.
- Run uses the configured seed file, else the one the last run read, so a Run
  never switches seed releases on its own.
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
from review_page.review_case_messages import (
    convert_case_to_message,
    list_ordered_mappings,
)
from review_page.review_case_source import ReviewCaseSource
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
    ReviewCase,
    ReviewFile,
    is_review_file_locked,
    replace_file_lines,
)

_PROCESS_ID_FILE_NAME = "review_run.pid"
_EXIT_CODE_FILE_NAME = "review_run.exit_code"
_RUN_LOG_PATH = Path("logs") / "review_page_runs.log"


class RunSeedFileUnknownError(RuntimeError):
    """Run has no seed file: none is configured, and no run has recorded one."""

    def __init__(self) -> None:
        """Say how to give Run a seed file."""
        super().__init__(
            "No seed file to clean: set REVIEW_PAGE_SEED_FILE, or run "
            "scripts/clean_seed.py once so its result names the file"
        )


class FileReviewCaseSource(ReviewCaseSource):
    """Read and save cases in the local review file, and run the seed-cleaning script."""

    def __init__(
        self,
        data_dir: Path,
        clean_seed_script: Path,
        *,
        seed_file: Path | None = None,
        python_executable: str = sys.executable,
    ) -> None:
        """Keep where the files are and how to start a run.

        Args:
            data_dir: Seed cleaning's data folder, holding ``review.jsonl``.
            clean_seed_script: The seed-cleaning script Run starts.
            seed_file: The seed release file Run cleans; None uses the file
                of the last run.
            python_executable: The Python that runs the script.
        """
        self._data_dir = data_dir
        self._clean_seed_script = clean_seed_script
        self._seed_file = seed_file
        self._python_executable = python_executable
        self._review_file_path = data_dir / REVIEW_FILE_NAME
        self._run_lock = threading.RLock()

    def list_cases(self) -> list[review_cases_pb2.ReviewCase]:
        """Return every case in the review file, in file order."""
        return [convert_case_to_message(case) for case in self._read_cases()]

    def get_case(self, case_name: str) -> review_cases_pb2.ReviewCase | None:
        """Return the named case in the review file, or None when it is not there."""
        case = next(
            (case for case in self._read_cases() if case.case_name == case_name), None
        )
        return None if case is None else convert_case_to_message(case)

    def get_last_run_result(self) -> review_cases_pb2.SeedCleaningRunResult | None:
        """Return the result the last run wrote, or None before the first run."""
        result = read_seed_cleaning_run_result(self._data_dir)
        return None if result is None else convert_run_result_to_message(result)

    def save_answers(
        self, case_answers: Sequence[review_cases_pb2.CaseAnswer]
    ) -> list[review_cases_pb2.RuleCheckResult]:
        """Check each answer with the run's rules and save the ones that follow them.

        Args:
            case_answers: Each case's new answer; an unset answer clears it.

        Returns:
            One result per answer, in the given order.

        Raises:
            ReviewFileLockedError: A run is writing the review file; nothing
                is saved.
            UnknownCaseError: A run removed a named case between the check
                and the save; nothing is saved.
            SavedAnswerChangedError: Another save changed a case's answer
                between the check and the save; nothing is saved.
        """
        cases_by_name = {case.case_name: case for case in self._read_cases()}
        seen_saved_answers = read_seen_saved_answers(case_answers)
        answers_to_save: dict[str, JsonValue] = {}
        results: list[review_cases_pb2.RuleCheckResult] = []
        for case_answer in case_answers:
            answer = read_case_answer(case_answer)
            result = check_case_answer(
                cases_by_name.get(case_answer.case_name),
                case_answer.case_name,
                answer,
                seen_saved_answers,
            )
            if result.is_saved:
                answers_to_save[case_answer.case_name] = answer
            results.append(result)
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
            for result in results:
                result.warnings.extend(warnings_by_case_name.get(result.case_name, ()))
        return results

    def start_run(self) -> review_cases_pb2.RunState:
        """Start ``clean_seed.py --no-fetch`` in its own process, unless a run is going.

        Returns:
            The run state: running.

        Raises:
            RunSeedFileUnknownError: No seed file is configured or recorded.
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
        """Return whether a run is going, and how the last run started here ended.

        A run started here is followed by its process id; when it has ended,
        its exit code is kept and the process id removed. Any other run is
        seen while it holds the review file's lock. A process id file this
        server did not write is left alone, so the server that started that
        run still records its exit code.
        """
        with self._run_lock:
            return self._read_run_state()

    def list_corrections(self) -> list[review_cases_pb2.SeedCorrection]:
        """Return the corrections file's lines, each with its change and reason sentence."""
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
        """Return both passes' entries from ``entries_to_check.json``, once each.

        A split-work candidate's entry carries its type and year, so it stands
        in for the same entry listed by the doubled-provider pass with its title
        alone.
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
        for group in document.get("split_work_candidates", []):
            for entry in group["entries"]:
                entries_by_reference_page[entry["reference_page"]] = (
                    review_cases_pb2.CheckedEntry(
                        reference_page=entry["reference_page"],
                        title=entry["title"],
                        type=entry.get("type"),
                        year=entry.get("year"),
                    )
                )
        return list(entries_by_reference_page.values())

    def _read_cases(self) -> list[ReviewCase]:
        """Read every case in the review file; none when it does not exist yet."""
        if not self._review_file_path.exists():
            return []
        with self._review_file_path.open(encoding="utf-8") as review_file:
            return [
                ReviewCase.model_validate_json(line)
                for line in review_file
                if line.strip()
            ]

    def _read_run_state(self) -> review_cases_pb2.RunState:
        """Return the run state, recording a finished run's exit code; the caller holds the run lock."""
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
        """Start ``clean_seed.py --no-fetch`` on the seed file, its output in the run log."""
        run_log_path = self._data_dir / _RUN_LOG_PATH
        run_log_path.parent.mkdir(parents=True, exist_ok=True)
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
        """Return the configured seed file, else the one the last run read."""
        if self._seed_file is not None:
            return self._seed_file
        last_run_result = read_seed_cleaning_run_result(self._data_dir)
        if last_run_result is None:
            raise RunSeedFileUnknownError
        return Path(last_run_result.seed_file)


def convert_run_result_to_message(
    result: SeedCleaningRunResult,
) -> review_cases_pb2.SeedCleaningRunResult:
    """Convert a seed-cleaning run's result to the message the page reads.

    Args:
        result: The run's result, as the run wrote it.

    Returns:
        The message; its counts keep the run's order.
    """
    return json_format.ParseDict(
        list_ordered_mappings(result.model_dump(mode="json")),
        review_cases_pb2.SeedCleaningRunResult(),
    )


def convert_correction_to_message(
    correction: SeedCorrection,
) -> review_cases_pb2.SeedCorrection:
    """Convert one seed correction to the message the page reads.

    The message also carries what the correction changes, its reason as a
    sentence, and whether an answer placed the page, so the page shows them
    without the seed-cleaning rules.

    Args:
        correction: One line of the corrections file.

    Returns:
        The message.
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
    """Report whether a run this server started is still going, and its exit code once it ended.

    The process is collected with ``waitpid``, which gives its exit code and
    keeps it from staying as a finished child. A process id this server did
    not start counts as not running here: after a restart the id may belong
    to an unrelated process. The caller then follows the run through the
    review file's lock.

    Args:
        process_id: The run's process id, from the process id file.

    Returns:
        Whether it runs, and its exit code when it ended; None otherwise.
    """
    try:
        ended_process_id, wait_status = os.waitpid(process_id, os.WNOHANG)
    except ChildProcessError:
        return False, None
    if ended_process_id == 0:
        return True, None
    return False, os.waitstatus_to_exitcode(wait_status)
