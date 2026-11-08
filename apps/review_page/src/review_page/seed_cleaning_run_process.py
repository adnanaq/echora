"""Start ``clean_seed.py --no-fetch`` from the review page and report whether a run is active.

Each operational rule reads `<rule>: <reason>`:
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
"""

import os
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path

from shared_proto.v1 import review_cases_pb2
from work_matching import is_review_file_locked, replace_file_lines

_PROCESS_ID_FILE_NAME = "review_run.pid"
_EXIT_CODE_FILE_NAME = "review_run.exit_code"
_RUN_LOG_PATH = Path("logs") / "review_page_runs.log"
# A run log past this size moves to ``review_page_runs.log.1`` before the next
# run, replacing the older copy, so the run logs never take more than about
# twice this size on disk.
RUN_LOG_LIMIT_BYTES = 1_000_000


class SeedCleaningRunProcess:
    """Start the seed-cleaning run process and track its state in the data folder."""

    def __init__(
        self,
        data_dir: Path,
        review_file_path: Path,
        clean_seed_script: Path,
        python_executable: str,
    ) -> None:
        """Initialize the run process tracker for one seed-cleaning data folder.

        Args:
            data_dir: Directory containing ``review.jsonl`` and run state files.
            review_file_path: Path to ``review.jsonl``, whose lock a run holds.
            clean_seed_script: Path to ``scripts/clean_seed.py`` executed on run.
            python_executable: Python interpreter executable used to launch runs.
        """
        self._data_dir = data_dir
        self._clean_seed_script = clean_seed_script
        self._python_executable = python_executable
        self._review_file_path = review_file_path
        self._run_lock = threading.RLock()

    def start_run(
        self, find_seed_file: Callable[[], Path]
    ) -> review_cases_pb2.RunState:
        """Launch ``clean_seed.py --no-fetch`` in a background subprocess if no run is active.

        The seed file is resolved only when no run is active, so asking for a
        run while one is active never fails on a missing seed file.

        Args:
            find_seed_file: Resolves the seed release file to clean.

        Returns:
            The active run state marked as running.
        """
        with self._run_lock:
            if self._resolve_run_state().is_running:
                return review_cases_pb2.RunState(is_running=True)
            run_process = self._start_process(find_seed_file())
            (self._data_dir / _EXIT_CODE_FILE_NAME).unlink(missing_ok=True)
            replace_file_lines(
                self._data_dir / _PROCESS_ID_FILE_NAME, [str(run_process.pid)]
            )
        return review_cases_pb2.RunState(is_running=True)

    def read_run_state(self) -> review_cases_pb2.RunState:
        """Report whether a pipeline run is active and the exit code of the last run.

        Returns:
            The current run state message with active status and last exit code.
        """
        with self._run_lock:
            return self._resolve_run_state()

    def _resolve_run_state(self) -> review_cases_pb2.RunState:
        """Resolve whether a run is active from its process or the lock file, and record its exit code.

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

    def _start_process(self, seed_file: Path) -> subprocess.Popen[bytes]:
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
