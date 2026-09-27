import gzip
from pathlib import Path

import pytest

from benchmarks.vector_service.sweep import (
    k6_arguments,
    load_sweep,
    report_table,
    run_sweep,
    seconds_from_duration,
    steady_range,
)
from benchmarks.vector_service.toolkit.k6_runner import LoadRun
from benchmarks.vector_service.toolkit.service_control import (
    RunningService,
    ServiceExitedError,
)
from benchmarks.vector_service.toolkit.settings import load_environment

SWEEP = """
name = "embed_concurrency"

[service_env]
EMBED_BATCH_MAX_SIZE = "64"

[load]
test_type = "load"

[load.k6]
RATE = "500"
HOLD = "60s"

[[variant]]
name = "one call"
[variant.service_env]
EMBED_MAX_CONCURRENCY = "1"

[[variant]]
name = "two calls"
[variant.service_env]
EMBED_MAX_CONCURRENCY = "2"
"""


class FakeController:
    def __init__(self, log: list[str]) -> None:
        self.log = log

    def start(self, run_env: dict[str, str], log_file: Path) -> RunningService:
        self.log.append(f"start {run_env['EMBED_MAX_CONCURRENCY']}")
        return RunningService(frozenset({1}), log_file)

    def is_alive(self) -> bool:
        return True

    def stop(self) -> None:
        self.log.append("stop")


class FakeGpu:
    def compute_process_ids(self) -> set[int]:
        return {1}


def write_sweep(tmp_path: Path) -> Path:
    path = tmp_path / "sweep.toml"
    path.write_text(SWEEP)
    return path


def fake_load_runner(results_dir: Path, latencies: list[float]):
    remaining = iter(latencies)

    def run(
        environment, test_type, k6_arguments, target, sampler_factory, console_to_file
    ):
        latency = next(remaining)
        run = LoadRun(run_id=f"load-{latency:.0f}", results_dir=results_dir)
        with gzip.open(run.samples_file, "wt") as samples:
            samples.write("metric_name,timestamp,metric_value\n")
            for second in range(0, 200):
                samples.write(f"iterations,{1000 + second},5\n")
                samples.write(f"grpc_req_duration,{1000 + second},{latency}\n")
        rows = [
            f"10:{second // 60:02d}:{second % 60:02d},100.0%,1MiB,1%,40,1,500.0%,42.0%"
            for second in range(0, 200, 2)
        ]
        run.resources_file.write_text(
            "time,service_cpu,service_memory,k6_cpu,gpu_util,gpu_memory_mib,"
            "qdrant_cpu,service_main_thread_cpu\n" + "\n".join(rows) + "\n"
        )
        return run, 0

    return run


def test_sweep_file_gives_each_variant_its_settings(tmp_path: Path) -> None:
    sweep = load_sweep(write_sweep(tmp_path))

    assert sweep.name == "embed_concurrency"
    assert sweep.test_type == "load"
    assert k6_arguments(sweep, sweep.variants[0]) == [
        "-e",
        "RATE=500",
        "-e",
        "HOLD=60s",
    ]
    assert [variant.name for variant in sweep.variants] == ["one call", "two calls"]
    assert sweep.variants[1].service_env == {
        "EMBED_BATCH_MAX_SIZE": "64",
        "EMBED_MAX_CONCURRENCY": "2",
    }


@pytest.mark.parametrize(
    ("duration", "seconds"), [("60s", 60), ("2m", 120), ("1h", 3600), ("90", 90)]
)
def test_durations_are_read_like_k6(duration: str, seconds: int) -> None:
    assert seconds_from_duration(duration) == seconds


def test_steady_range_of_a_load_test_skips_ramp_and_settle(tmp_path: Path) -> None:
    sweep = load_sweep(write_sweep(tmp_path))

    assert steady_range(sweep) == (135, 180)


def test_sweep_runs_every_variant_and_stops_each_service(tmp_path: Path) -> None:
    environment = load_environment("laptop", [f"results_dir={tmp_path}"])
    sweep = load_sweep(write_sweep(tmp_path))
    log: list[str] = []

    results = run_sweep(
        environment,
        sweep,
        controller_factory=lambda env: FakeController(log),
        load_runner=fake_load_runner(tmp_path, [80.0, 100.0]),
        wait=lambda *args, **kwargs: None,
        gpu_reader=FakeGpu(),
    )

    assert log == ["start 1", "stop", "start 2", "stop"]
    summaries = [result.summary for result in results]
    assert all(summary is not None for summary in summaries)
    assert [summary.p50_ms for summary in summaries if summary] == [80.0, 100.0]
    first_summary, first_resources = results[0].summary, results[0].resources
    assert first_summary is not None
    assert first_summary.completed_per_second == 5.0
    assert first_resources is not None
    assert first_resources.service_main_thread_cpu == 42.0
    table = report_table(results)
    assert "| one call |" in table
    assert "| two calls |" in table


def test_a_variant_that_fails_to_start_is_reported_and_the_sweep_goes_on(
    tmp_path: Path,
) -> None:
    environment = load_environment("laptop", [f"results_dir={tmp_path}"])
    sweep = load_sweep(write_sweep(tmp_path))
    log: list[str] = []
    waits = iter([ServiceExitedError(), None])

    def wait(*args, **kwargs) -> None:
        failure = next(waits)
        if failure is not None:
            raise failure

    results = run_sweep(
        environment,
        sweep,
        controller_factory=lambda env: FakeController(log),
        load_runner=fake_load_runner(tmp_path, [80.0]),
        wait=wait,
        gpu_reader=FakeGpu(),
    )

    assert log == ["start 1", "stop", "start 2", "stop"]
    assert results[0].error is not None
    assert results[1].error is None
    assert "failed" in report_table(results)


def test_a_variant_can_change_k6_settings(tmp_path: Path) -> None:
    path = tmp_path / "sweep.toml"
    path.write_text(
        SWEEP.replace(
            'name = "two calls"\n',
            'name = "two calls"\nk6_env = { HOLD = "30s", WITH_PAYLOAD = "true" }\n',
        )
    )
    sweep = load_sweep(path)

    assert k6_arguments(sweep, sweep.variants[0]) == [
        "-e",
        "RATE=500",
        "-e",
        "HOLD=60s",
    ]
    assert k6_arguments(sweep, sweep.variants[1]) == [
        "-e", "RATE=500", "-e", "HOLD=30s", "-e", "WITH_PAYLOAD=true",
    ]  # fmt: skip
