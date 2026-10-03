#!/usr/bin/env python3
"""Print a load test run's completed searches and latency per time window.

k6's Prometheus output reports latency percentiles over everything since the
run started, so late in a ramp they mostly reflect the earlier, lighter load.
This reads the run's raw samples (``<results_dir>/<run_id>-samples.csv.gz``,
written by the load runner) and computes each window on its own: searches
completed and dropped per second, failed searches, and p50/p95/p99 latency of
the searches that finished in that window.

Usage: uv run python -m benchmarks.vector_service.reports.summarize_load_test RUN_ID_OR_FILE [--window 20] [--environment laptop]
"""

import argparse

from benchmarks.vector_service.toolkit.results import (
    print_windows,
    read_windows,
    samples_file,
)
from benchmarks.vector_service.toolkit.settings import load_environment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "run", help="run id (e.g. breakpoint-20260927-090527) or a file"
    )
    parser.add_argument(
        "--window", type=int, default=20, help="window length in seconds"
    )
    parser.add_argument(
        "--environment", default="laptop", help="environment name or file"
    )
    arguments = parser.parse_args()
    results_dir = load_environment(arguments.environment).results_dir
    print_windows(
        read_windows(samples_file(arguments.run, results_dir), arguments.window),
        arguments.window,
    )


if __name__ == "__main__":
    main()
