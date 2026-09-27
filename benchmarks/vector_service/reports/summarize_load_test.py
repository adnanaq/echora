#!/usr/bin/env python3
"""Print a load test run's completed searches and latency per time window.

k6's Prometheus output reports latency percentiles over everything since the
run started, so late in a ramp they mostly reflect the earlier, lighter load.
This reads the run's raw samples (``benchmarks/vector_service/load/results/<run_id>-samples.csv.gz``,
written by ``run_vector_search.sh``) and computes each window on its own:
searches completed and dropped per second, failed searches, and p50/p95/p99
latency of the searches that finished in that window.

Usage: uv run python benchmarks/vector_service/reports/summarize_load_test.py RUN_ID_OR_FILE [--window 20]
"""

import argparse
import csv
import gzip
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

RESULTS_DIR = Path("benchmarks/vector_service/load/results")


@dataclass
class WindowSamples:
    latencies_ms: list[float] = field(default_factory=list)
    completed: int = 0
    dropped: int = 0
    failed: int = 0


def samples_file(run: str) -> Path:
    path = Path(run)
    if path.exists():
        return path
    return RESULTS_DIR / f"{run}-samples.csv.gz"


def read_windows(path: Path, window_seconds: int) -> dict[int, WindowSamples]:
    windows: dict[int, WindowSamples] = defaultdict(WindowSamples)
    first_timestamp: int | None = None
    with gzip.open(path, "rt", newline="") as samples:
        for row in csv.DictReader(samples):
            timestamp = int(row["timestamp"])
            if first_timestamp is None:
                first_timestamp = timestamp
            window = windows[(timestamp - first_timestamp) // window_seconds]
            metric = row["metric_name"]
            value = float(row["metric_value"])
            if metric == "grpc_req_duration":
                window.latencies_ms.append(value)
            elif metric == "iterations":
                window.completed += int(value)
            elif metric == "dropped_iterations":
                window.dropped += int(value)
            elif metric == "search_errors":
                window.failed += int(value)
    return windows


def percentile(sorted_values: list[float], fraction: float) -> float:
    index = min(len(sorted_values) - 1, round(fraction * (len(sorted_values) - 1)))
    return sorted_values[index]


def print_windows(windows: dict[int, WindowSamples], window_seconds: int) -> None:
    print(
        f"{'from':>6} {'done/s':>7} {'dropped/s':>9} {'failed':>6}"
        f" {'p50 ms':>7} {'p95 ms':>7} {'p99 ms':>7}"
    )
    for index in sorted(windows):
        window = windows[index]
        latencies = sorted(window.latencies_ms)
        if latencies:
            spread = " ".join(
                f"{percentile(latencies, fraction):7.0f}"
                for fraction in (0.50, 0.95, 0.99)
            )
        else:
            spread = f"{'-':>7} {'-':>7} {'-':>7}"
        print(
            f"{index * window_seconds:5d}s {window.completed / window_seconds:7.0f}"
            f" {window.dropped / window_seconds:9.1f} {window.failed:6d} {spread}"
        )
    all_latencies = [
        latency for window in windows.values() for latency in window.latencies_ms
    ]
    if all_latencies:
        print(
            f"whole run: {len(all_latencies)} searches,"
            f" median {statistics.median(all_latencies):.0f} ms"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "run", help="run id (e.g. breakpoint-20260927-090527) or a file"
    )
    parser.add_argument(
        "--window", type=int, default=20, help="window length in seconds"
    )
    arguments = parser.parse_args()
    print_windows(
        read_windows(samples_file(arguments.run), arguments.window), arguments.window
    )


if __name__ == "__main__":
    main()
