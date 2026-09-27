"""Read load test samples, resource samples and profiles into numbers.

k6's Prometheus output reports latency percentiles over everything since the
run started, so late in a ramp they mostly reflect the earlier, lighter load.
The functions here read a run's raw samples (``<run_id>-samples.csv.gz``) and
compute each time window on its own.
"""

import csv
import gzip
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

MILLISECOND_TIMESTAMP_FLOOR = 10**11
RESOURCE_MEAN_COLUMNS = (
    "service_cpu",
    "qdrant_cpu",
    "gpu_util",
    "k6_cpu",
    "service_main_thread_cpu",
)
EVENT_LOOP_ROOT = "_run_module_as_main"
EVENT_LOOP_AREAS = (
    ("type_inspector", "Qdrant client: inference check"),
    ("qdrant_client/", "Qdrant client"),
    ("grpc/", "gRPC"),
    ("request_batcher", "request batcher"),
    ("routes/search.py", "search route"),
    ("text_processor", "text processor"),
    ("opentelemetry", "OpenTelemetry"),
    ("asyncio/", "asyncio"),
)
WORKER_AREAS = (
    ("tokenization", "tokenizer"),
    ("torch/", "model forward pass"),
    ("transformers/", "model forward pass"),
    ("_convert_to_numpy", "copying results off the GPU"),
    ("flagembedding_model", "embedding code"),
    ("grpc/", "gRPC"),
)


@dataclass
class WindowSamples:
    latencies_ms: list[float] = field(default_factory=list)
    completed: int = 0
    dropped: int = 0
    failed: int = 0


@dataclass(frozen=True)
class SteadySummary:
    completed_per_second: float
    dropped_per_second: float
    failed: int
    searches: int
    p50_ms: float | None
    p95_ms: float | None
    p99_ms: float | None


@dataclass(frozen=True)
class ResourceColumns:
    service_cpu: float | None
    qdrant_cpu: float | None
    gpu_util: float | None
    k6_cpu: float | None
    service_main_thread_cpu: float | None = None


@dataclass
class ProfileCounts:
    loop_areas: Counter[str] = field(default_factory=Counter)
    loop_leaves: Counter[str] = field(default_factory=Counter)
    worker_areas: Counter[str] = field(default_factory=Counter)
    worker_leaves: Counter[str] = field(default_factory=Counter)


def samples_file(run: str, results_dir: Path) -> Path:
    path = Path(run)
    if path.exists():
        return path
    return results_dir / f"{run}-samples.csv.gz"


def read_windows(path: Path, window_seconds: int) -> dict[int, WindowSamples]:
    """Group a run's raw samples into windows of ``window_seconds``."""
    windows: dict[int, WindowSamples] = defaultdict(WindowSamples)
    first_timestamp: int | None = None
    window_length = window_seconds
    with gzip.open(path, "rt", newline="") as samples:
        for row in csv.DictReader(samples):
            timestamp = int(row["timestamp"])
            if first_timestamp is None:
                first_timestamp = timestamp
                if timestamp >= MILLISECOND_TIMESTAMP_FLOOR:
                    window_length = window_seconds * 1000
            add_sample(
                windows[(timestamp - first_timestamp) // window_length],
                row["metric_name"],
                float(row["metric_value"]),
            )
    return windows


def add_sample(window: WindowSamples, metric: str, value: float) -> None:
    if metric == "grpc_req_duration":
        window.latencies_ms.append(value)
    elif metric == "iterations":
        window.completed += int(value)
    elif metric == "dropped_iterations":
        window.dropped += int(value)
    elif metric == "search_errors":
        window.failed += int(value)


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


def steady_summary(
    windows: dict[int, WindowSamples],
    window_seconds: int,
    start_seconds: float,
    end_seconds: float,
) -> SteadySummary:
    """Combine the windows lying fully between ``start_seconds`` and ``end_seconds``."""
    chosen = [
        windows[index]
        for index in sorted(windows)
        if index * window_seconds >= start_seconds
        and (index + 1) * window_seconds <= end_seconds
    ]
    seconds = len(chosen) * window_seconds or 1
    latencies = sorted(latency for window in chosen for latency in window.latencies_ms)
    spread = [
        percentile(latencies, fraction) if latencies else None
        for fraction in (0.50, 0.95, 0.99)
    ]
    return SteadySummary(
        completed_per_second=sum(window.completed for window in chosen) / seconds,
        dropped_per_second=sum(window.dropped for window in chosen) / seconds,
        failed=sum(window.failed for window in chosen),
        searches=len(latencies),
        p50_ms=spread[0],
        p95_ms=spread[1],
        p99_ms=spread[2],
    )


def mean_resources(
    path: Path, start_seconds: float = 0.0, end_seconds: float | None = None
) -> ResourceColumns:
    """Mean of each resource column, over seconds since the first sample."""
    columns: dict[str, list[float]] = defaultdict(list)
    with path.open(newline="") as resources:
        rows = list(csv.DictReader(resources))
    first_second = clock_seconds(rows[0]["time"]) if rows else 0
    for row in rows:
        elapsed = (clock_seconds(row["time"]) - first_second) % 86400
        if elapsed < start_seconds or (
            end_seconds is not None and elapsed > end_seconds
        ):
            continue
        for name in RESOURCE_MEAN_COLUMNS:
            number = parse_number(row.get(name))
            if number is not None:
                columns[name].append(number)
    means = {
        name: statistics.mean(columns[name]) if columns[name] else None
        for name in RESOURCE_MEAN_COLUMNS
    }
    return ResourceColumns(**means)


def clock_seconds(clock: str) -> int:
    hours, minutes, seconds = (int(part) for part in clock.split(":"))
    return hours * 3600 + minutes * 60 + seconds


def parse_number(text: str | None) -> float | None:
    cleaned = (text or "").strip().rstrip("%").removesuffix("MiB")
    return float(cleaned) if cleaned else None


def area_of(joined_stack: str, areas: tuple[tuple[str, str], ...]) -> str:
    return next((name for key, name in areas if key in joined_stack), "other")


def read_profile(path: Path) -> ProfileCounts:
    """Count py-spy folded-stack samples by thread group and area of the code."""
    counts = ProfileCounts()
    for line in path.read_text().splitlines():
        stack, _, count_text = line.rpartition(" ")
        frames = stack.split(";")
        count = int(count_text)
        joined = ";".join(frames)
        leaf = frames[-1].split(" (")[0] + " (" + frames[-1].rsplit("/", 1)[-1]
        if frames[0].startswith(EVENT_LOOP_ROOT):
            counts.loop_areas[area_of(joined, EVENT_LOOP_AREAS)] += count
            counts.loop_leaves[leaf] += count
        else:
            counts.worker_areas[area_of(joined, WORKER_AREAS)] += count
            counts.worker_leaves[leaf] += count
    return counts


def print_profile_group(title: str, areas: Counter[str], leaves: Counter[str]) -> None:
    total = sum(areas.values())
    if not total:
        return
    print(f"-- {title}")
    for name, count in areas.most_common():
        print(f"  {count / total:6.1%}  {name}")
    print("  top functions:")
    for name, count in leaves.most_common(8):
        print(f"  {count / total:6.1%}  {name[:100]}")
