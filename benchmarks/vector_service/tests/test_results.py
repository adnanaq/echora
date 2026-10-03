import gzip
from pathlib import Path

from benchmarks.vector_service.toolkit.results import (
    ResourceColumns,
    mean_resources,
    read_profile,
    read_windows,
    steady_summary,
)

HEADER = "metric_name,timestamp,metric_value,check\n"


def write_samples(tmp_path: Path, rows: list[tuple[str, int, float]]) -> Path:
    path = tmp_path / "run-samples.csv.gz"
    with gzip.open(path, "wt") as samples:
        samples.write(HEADER)
        for metric, timestamp, value in rows:
            samples.write(f"{metric},{timestamp},{value},\n")
    return path


def test_read_windows_groups_samples_by_seconds_since_start(tmp_path: Path) -> None:
    path = write_samples(
        tmp_path,
        [
            ("grpc_req_duration", 1000, 10.0),
            ("iterations", 1000, 1.0),
            ("grpc_req_duration", 1019, 30.0),
            ("iterations", 1019, 1.0),
            ("grpc_req_duration", 1020, 50.0),
            ("iterations", 1020, 1.0),
            ("dropped_iterations", 1021, 2.0),
            ("search_errors", 1022, 1.0),
        ],
    )

    windows = read_windows(path, 20)

    assert sorted(windows) == [0, 1]
    assert windows[0].latencies_ms == [10.0, 30.0]
    assert windows[0].completed == 2
    assert windows[1].completed == 1
    assert windows[1].dropped == 2
    assert windows[1].failed == 1


def test_read_windows_millisecond_timestamps_returns_same_windows(
    tmp_path: Path,
) -> None:
    path = write_samples(
        tmp_path,
        [
            ("iterations", 1_790_000_000_000, 1.0),
            ("iterations", 1_790_000_019_999, 1.0),
            ("iterations", 1_790_000_020_000, 1.0),
        ],
    )

    windows = read_windows(path, 20)

    assert windows[0].completed == 2
    assert windows[1].completed == 1


def test_steady_summary_covers_only_chosen_windows(tmp_path: Path) -> None:
    rows = []
    for second in range(60):
        rows.append(("iterations", 1000 + second, 1.0))
        rows.append(("grpc_req_duration", 1000 + second, float(second)))
    windows = read_windows(write_samples(tmp_path, rows), 20)

    summary = steady_summary(windows, 20, start_seconds=20, end_seconds=40)

    assert summary.completed_per_second == 1.0
    assert summary.p50_ms == 30.0
    assert summary.searches == 20


def test_mean_resources_reads_percent_and_mebibyte_columns(tmp_path: Path) -> None:
    path = tmp_path / "run-resources.csv"
    path.write_text(
        "time,service_cpu,service_memory,k6_cpu,gpu_util,gpu_memory_mib,qdrant_cpu\n"
        "10:00:00,,5000MiB,1%,10,100,\n"
        "10:00:02,120.0%,5100MiB,2%,30,100,400.0%\n"
        "10:00:04,140.0%,5200MiB,3%,50,100,600.0%\n"
    )

    means = mean_resources(path)

    assert means == ResourceColumns(
        service_cpu=130.0, qdrant_cpu=500.0, gpu_util=30.0, k6_cpu=2.0
    )


def test_mean_resources_file_without_qdrant_column_returns_no_qdrant_cpu(
    tmp_path: Path,
) -> None:
    path = tmp_path / "run-resources.csv"
    path.write_text(
        "time,service_cpu,service_memory,k6_cpu,gpu_util,gpu_memory_mib\n"
        "10:00:02,120.0%,5100MiB,2%,30,100\n"
    )

    assert mean_resources(path).qdrant_cpu is None


def test_read_profile_splits_event_loop_from_other_threads(tmp_path: Path) -> None:
    path = tmp_path / "profile.txt"
    path.write_text(
        "_run_module_as_main (runpy.py:198);_run (asyncio/events.py:94) 30\n"
        "_run_module_as_main (runpy.py:198);query (qdrant_client/async.py:10) 10\n"
        "_bootstrap (threading.py:1);forward (torch/nn/linear.py:134) 20\n"
    )

    profile = read_profile(path)

    assert profile.loop_areas == {"asyncio": 30, "Qdrant client": 10}
    assert profile.worker_areas == {"model forward pass": 20}
