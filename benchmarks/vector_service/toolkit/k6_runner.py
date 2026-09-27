"""Run the k6 search load test (``load/vector_search.js``) in the pinned k6 image.

Per run, the results folder gets:

- ``<run_id>-samples.csv.gz``  every sample, for per-window numbers
- ``<run_id>.json``            end-of-run summary
- ``<run_id>-report.html``     k6 dashboard report with time-series graphs
- ``<run_id>-resources.csv``   CPU, memory and GPU use, when sampled

Results are also pushed to Prometheus, tagged with ``run_id``.
"""

import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from benchmarks.vector_service.toolkit.resources import ResourceSampler
from benchmarks.vector_service.toolkit.settings import REPOSITORY_ROOT, Environment

LOAD_DIR = REPOSITORY_ROOT / "benchmarks" / "vector_service" / "load"
TREND_STATS = "p(50),p(95),p(99),max"


@dataclass(frozen=True)
class LoadRun:
    run_id: str
    results_dir: Path

    @property
    def samples_file(self) -> Path:
        return self.results_dir / f"{self.run_id}-samples.csv.gz"

    @property
    def summary_file(self) -> Path:
        return self.results_dir / f"{self.run_id}.json"

    @property
    def resources_file(self) -> Path:
        return self.results_dir / f"{self.run_id}-resources.csv"

    @property
    def html_report_file(self) -> Path:
        return self.results_dir / f"{self.run_id}-report.html"

    @property
    def console_file(self) -> Path:
        return self.results_dir / f"{self.run_id}-console.txt"


def new_run_id(test_type: str, now: datetime | None = None) -> str:
    return f"{test_type}-{(now or datetime.now()):%Y%m%d-%H%M%S}"


def k6_command(
    environment: Environment,
    run_id: str,
    test_type: str,
    k6_arguments: list[str],
    user: str,
    target: str | None = None,
) -> list[str]:
    """The ``docker run`` command for one k6 run.

    ``target`` sets TARGET and TLS from the environment; without it the k6
    script's own default or a ``-e TARGET=...`` in ``k6_arguments`` applies.
    """
    mounts, results_path = results_mount(environment.results_dir)
    k6_settings = environment.k6
    docker_env = [
        f"K6_PROMETHEUS_RW_SERVER_URL={k6_settings.prometheus_url}",
        f"K6_PROMETHEUS_RW_TREND_STATS={TREND_STATS}",
        f"K6_CSV_TIME_FORMAT={k6_settings.csv_time_format}",
    ]
    if k6_settings.html_report:
        docker_env += [
            "K6_WEB_DASHBOARD=true",
            "K6_WEB_DASHBOARD_PORT=-1",
            f"K6_WEB_DASHBOARD_EXPORT={results_path}/{run_id}-report.html",
        ]
    target_arguments = []
    if target is not None:
        tls = "true" if environment.service_target.tls else "false"
        target_arguments = ["-e", f"TARGET={target}", "-e", f"TLS={tls}"]
    return [
        "docker", "run", "--rm", "--name", f"k6-{run_id}",
        "--network", "host", "--user", user,
        "-v", f"{REPOSITORY_ROOT}:/repo", *mounts,
        "-w", "/repo/benchmarks/vector_service/load",
        *[item for value in docker_env for item in ("-e", value)],
        k6_settings.image, "run",
        "--out", "experimental-prometheus-rw",
        "--out", f"csv={results_path}/{run_id}-samples.csv.gz",
        "--tag", f"run_id={run_id}",
        "--new-machine-readable-summary",
        "--summary-export", f"{results_path}/{run_id}.json",
        "-e", f"TEST_TYPE={test_type}",
        *k6_arguments,
        *target_arguments,
        "vector_search.js",
    ]  # fmt: skip


def results_mount(results_dir: Path) -> tuple[list[str], str]:
    """Docker mounts and the in-container path for the results folder."""
    if results_dir.is_relative_to(REPOSITORY_ROOT):
        return [], os.path.relpath(results_dir, LOAD_DIR)
    return ["-v", f"{results_dir}:/results"], "/results"


def run_load_test(
    environment: Environment,
    test_type: str,
    k6_arguments: list[str],
    target: str | None = None,
    sampler_factory: Callable[[LoadRun], ResourceSampler | None] | None = None,
    runner: Callable[[list[str], Path | None], int] | None = None,
    console_to_file: bool = False,
) -> tuple[LoadRun, int]:
    """Run k6 once, sampling resources meanwhile; return the run and k6's exit code.

    k6's console output goes to the terminal, or to ``<run_id>-console.txt``
    with ``console_to_file``.
    """
    run = LoadRun(new_run_id(test_type), environment.results_dir)
    environment.results_dir.mkdir(parents=True, exist_ok=True)
    command = k6_command(
        environment,
        run.run_id,
        test_type,
        k6_arguments,
        user=f"{os.getuid()}:{os.getgid()}",
        target=target,
    )
    sampler = sampler_factory(run) if sampler_factory else None
    if sampler is not None:
        sampler.start()
    try:
        console_file = run.console_file if console_to_file else None
        exit_code = (runner or run_and_wait)(command, console_file)
    finally:
        if sampler is not None:
            sampler.stop()
    return run, exit_code


def run_and_wait(command: list[str], console_file: Path | None) -> int:
    if console_file is None:
        return subprocess.run(command, check=False).returncode  # noqa: S603 - built above
    with console_file.open("w") as console:
        return subprocess.run(  # noqa: S603 - built above
            command, stdout=console, stderr=subprocess.STDOUT, check=False
        ).returncode
