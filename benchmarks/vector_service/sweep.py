#!/usr/bin/env python3
"""Compare service settings: run the same load test once per setting combination.

For each variant in a sweep file the sweep starts a benchmark-owned service
instance with the variant's environment variables, waits until it is SERVING
(and, when it asks for the GPU, checks it is on the GPU), runs k6 while
sampling CPU, memory and GPU use, then stops the service. It writes one
comparison table (``sweep-<name>-<time>.md``) and every variant's settings and
numbers (``.json``) to the environment's results folder.

Usage:
    uv run python -m benchmarks.vector_service.sweep SWEEP [--environment laptop]
        [--set section.field=value] [--timed]

``SWEEP`` is a file or a name in ``benchmarks/vector_service/sweeps/``.
``--timed`` runs the service through ``diagnostics/run_timed_vector_service.py``
so each variant's log has per-stage timing.
"""

import argparse
import json
import re
import sys
import tomllib
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmarks.vector_service.toolkit.k6_runner import LoadRun, run_load_test
from benchmarks.vector_service.toolkit.resources import (
    ContainerCgroupReader,
    DockerStatsReader,
    NvidiaGpuReader,
    ProcessReader,
    ResourceReader,
    ResourceSampler,
)
from benchmarks.vector_service.toolkit.results import (
    ResourceColumns,
    SteadySummary,
    mean_resources,
    read_windows,
    steady_summary,
)
from benchmarks.vector_service.toolkit.service_control import (
    DockerController,
    GpuProcessReader,
    LocalProcessController,
    RunningService,
    ServiceController,
    check_gpu_use,
    wait_until_serving,
)
from benchmarks.vector_service.toolkit.settings import Environment, load_environment

SWEEPS_DIR = Path(__file__).resolve().parent / "sweeps"
WINDOW_SECONDS = 20
# The ``load`` test type in load/vector_search.js: 2 min ramp, HOLD, 1 min down.
LOAD_RAMP_SECONDS = 120
SETTLE_SECONDS = 15
DURATION_UNITS = {"s": 1, "m": 60, "h": 3600}


@dataclass(frozen=True)
class Variant:
    name: str
    service_env: dict[str, str]
    k6_settings: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Sweep:
    name: str
    test_type: str
    k6_settings: dict[str, str]
    variants: list[Variant]


@dataclass
class VariantResult:
    name: str
    service_env: dict[str, str]
    run_id: str | None = None
    summary: SteadySummary | None = None
    highest_completed_per_second: float | None = None
    resources: ResourceColumns | None = None
    error: str | None = None
    notes: list[str] = field(default_factory=list)


def load_sweep(path_or_name: Path | str) -> Sweep:
    path = Path(path_or_name)
    if not path.exists():
        path = SWEEPS_DIR / f"{path_or_name}.toml"
    data = tomllib.loads(path.read_text())
    base_env = {key: str(value) for key, value in data.get("service_env", {}).items()}
    load = data.get("load", {})
    return Sweep(
        name=str(data.get("name", path.stem)),
        test_type=str(load.get("test_type", "load")),
        k6_settings={key: str(value) for key, value in load.get("k6", {}).items()},
        variants=[
            Variant(
                name=str(variant["name"]),
                service_env={
                    **base_env,
                    **{k: str(v) for k, v in variant.get("service_env", {}).items()},
                },
                k6_settings={k: str(v) for k, v in variant.get("k6_env", {}).items()},
            )
            for variant in data.get("variant", [])
        ],
    )


def k6_arguments(sweep: Sweep, variant: Variant) -> list[str]:
    """k6 ``-e`` arguments: the sweep's load settings, then the variant's own."""
    settings = {**sweep.k6_settings, **variant.k6_settings}
    return [
        item for key, value in settings.items() for item in ("-e", f"{key}={value}")
    ]


def seconds_from_duration(duration: str) -> int:
    """Seconds in a k6 duration such as ``60s``, ``2m`` or ``1h30m``."""
    parts = re.findall(r"(\d+)([smh]?)", duration)
    return sum(
        int(number) * DURATION_UNITS.get(unit or "s", 1) for number, unit in parts
    )


def steady_range(sweep: Sweep) -> tuple[float, float | None]:
    """Seconds since the run's start that count for the comparison."""
    if sweep.test_type == "load":
        hold = seconds_from_duration(sweep.k6_settings.get("HOLD", "10m"))
        return LOAD_RAMP_SECONDS + SETTLE_SECONDS, LOAD_RAMP_SECONDS + hold
    return 0, None


def controller_for(environment: Environment, timed: bool) -> ServiceController:
    if environment.service_start.kind == "docker":
        return DockerController(environment)
    return LocalProcessController(environment, timed=timed)


def readers_for(
    environment: Environment, run: LoadRun, service: RunningService
) -> list[ResourceReader]:
    readers: list[ResourceReader] = []
    if environment.service_start.kind == "docker":
        readers.append(DockerStatsReader(environment.service_start.docker_container))
    else:
        readers.append(ProcessReader(min(service.process_ids)))
    readers.append(
        DockerStatsReader(f"k6-{run.run_id}", cpu_column="k6_cpu", memory_column=None)
    )
    if environment.resources.gpu:
        readers.append(NvidiaGpuReader())
    if environment.resources.qdrant:
        readers.append(ContainerCgroupReader(environment.qdrant.container))
    return readers


def run_variant(
    environment: Environment,
    sweep: Sweep,
    variant: Variant,
    controller: ServiceController,
    load_runner: Callable[..., tuple[LoadRun, int]],
    wait: Callable[..., None],
    gpu_reader: GpuProcessReader,
) -> VariantResult:
    result = VariantResult(name=variant.name, service_env=variant.service_env)
    run_env = {**environment.service_start.env, **variant.service_env}
    log_file = environment.results_dir / f"sweep-{sweep.name}-{slug(variant.name)}.log"
    try:
        service = controller.start(variant.service_env, log_file)
        wait(
            environment.service_target.address,
            environment.service_start.startup_timeout_seconds,
            is_alive=controller.is_alive,
        )
        check_gpu_use(service, run_env, gpu_reader)
        run, exit_code = load_runner(
            environment,
            sweep.test_type,
            k6_arguments(sweep, variant),
            target=environment.service_target.address,
            console_to_file=True,
            sampler_factory=lambda load_run: ResourceSampler(
                load_run.resources_file,
                readers_for(environment, load_run, service),
                environment.resources.sample_seconds,
            ),
        )
    except Exception as error:  # noqa: BLE001 - one variant's failure must not stop the sweep
        result.error = f"{type(error).__name__}: {error}"
        return result
    finally:
        controller.stop()
    if exit_code != 0:
        result.notes.append(f"k6 exited with {exit_code} (a threshold or abort)")
    summarize_variant(result, run, sweep)
    return result


def summarize_variant(result: VariantResult, run: LoadRun, sweep: Sweep) -> None:
    start_seconds, end_seconds = steady_range(sweep)
    windows = read_windows(run.samples_file, WINDOW_SECONDS)
    last_window_end = (max(windows, default=0) + 1) * WINDOW_SECONDS
    result.run_id = run.run_id
    result.summary = steady_summary(
        windows, WINDOW_SECONDS, start_seconds, end_seconds or last_window_end
    )
    result.highest_completed_per_second = max(
        (window.completed / WINDOW_SECONDS for window in windows.values()), default=0.0
    )
    if run.resources_file.exists():
        result.resources = mean_resources(
            run.resources_file, start_seconds, end_seconds
        )


def run_sweep(
    environment: Environment,
    sweep: Sweep,
    controller_factory: Callable[[Environment], ServiceController],
    load_runner: Callable[..., tuple[LoadRun, int]] = run_load_test,
    wait: Callable[..., None] = wait_until_serving,
    gpu_reader: GpuProcessReader | None = None,
) -> list[VariantResult]:
    environment.results_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for variant in sweep.variants:
        print(f"== {variant.name}", file=sys.stderr, flush=True)
        results.append(
            run_variant(
                environment,
                sweep,
                variant,
                controller_factory(environment),
                load_runner,
                wait,
                gpu_reader or NvidiaGpuReader(),
            )
        )
    return results


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def number(value: float | None, digits: int = 0, suffix: str = "") -> str:
    return "–" if value is None else f"{value:.{digits}f}{suffix}"


def report_table(results: list[VariantResult]) -> str:
    lines = [
        "| Variant | Done/s (steady) | Highest done/s | p50 ms | p95 ms | p99 ms"
        " | Failed | Dropped/s | Service CPU | Event loop thread | Qdrant CPU | GPU | Run |",
        "| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |",
    ]
    for result in results:
        if result.error or result.summary is None:
            lines.append(f"| {result.name} | failed: {result.error} |" + " |" * 11)
            continue
        summary, resources = (
            result.summary,
            result.resources or ResourceColumns(None, None, None, None),
        )
        lines.append(
            f"| {result.name} | {number(summary.completed_per_second)}"
            f" | {number(result.highest_completed_per_second)}"
            f" | {number(summary.p50_ms)} | {number(summary.p95_ms)} | {number(summary.p99_ms)}"
            f" | {summary.failed} | {number(summary.dropped_per_second, 1)}"
            f" | {number(resources.service_cpu, 0, '%')}"
            f" | {number(resources.service_main_thread_cpu, 0, '%')}"
            f" | {number(resources.qdrant_cpu, 0, '%')} | {number(resources.gpu_util, 0, '%')}"
            f" | {result.run_id} |"
        )
    return "\n".join(lines)


def write_report(
    environment: Environment, sweep: Sweep, results: list[VariantResult]
) -> Path:
    stem = f"sweep-{sweep.name}-{datetime.now():%Y%m%d-%H%M%S}"
    start_seconds, end_seconds = steady_range(sweep)
    steady = (
        f"{start_seconds:.0f} s to {end_seconds:.0f} s" if end_seconds else "whole run"
    )
    k6 = " ".join(f"{key}={value}" for key, value in sweep.k6_settings.items())
    notes = [f"- {r.name}: {note}" for r in results for note in r.notes]
    text = "\n".join(
        [
            f"# Sweep {sweep.name} ({environment.name})",
            "",
            f"Test `{sweep.test_type}` with {k6}; steady part {steady}.",
            "",
            report_table(results),
            *(["", *notes] if notes else []),
            "",
        ]
    )
    report = environment.results_dir / f"{stem}.md"
    report.write_text(text)
    data: dict[str, Any] = {
        "sweep": asdict(sweep),
        "environment": environment.name,
        "results": [asdict(result) for result in results],
    }
    (environment.results_dir / f"{stem}.json").write_text(json.dumps(data, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], allow_abbrev=False
    )
    parser.add_argument("sweep")
    parser.add_argument("--environment", default="laptop")
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--timed", action="store_true")
    arguments = parser.parse_args()
    environment = load_environment(arguments.environment, arguments.overrides)
    sweep = load_sweep(arguments.sweep)
    results = run_sweep(
        environment,
        sweep,
        controller_factory=lambda env: controller_for(env, arguments.timed),
    )
    report = write_report(environment, sweep, results)
    print(report.read_text())
    print(f"report: {report}", file=sys.stderr)
    return 0 if all(result.error is None for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
