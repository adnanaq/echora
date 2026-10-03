from pathlib import Path

from benchmarks.vector_service.toolkit.resources import (
    RESOURCE_COLUMNS,
    ContainerCgroupReader,
    DockerStatsReader,
    NvidiaGpuReader,
    ProcessReader,
    sample_row,
)


class FakeClock:
    def __init__(self) -> None:
        self.now_ns = 0

    def __call__(self) -> int:
        return self.now_ns


def write_process(proc_root: Path, pid: int, ticks: int, rss_kib: int) -> None:
    directory = proc_root / str(pid)
    (directory / "task" / str(pid)).mkdir(parents=True, exist_ok=True)
    fields = ["S"] + ["0"] * 10 + [str(ticks), "0"] + ["0"] * 30
    stat = f"{pid} (python) " + " ".join(fields)
    (directory / "stat").write_text(stat)
    (directory / "task" / str(pid) / "stat").write_text(stat)
    (directory / "status").write_text(f"Name:\tpython\nVmRSS:\t{rss_kib} kB\n")


def test_process_reader_two_reads_returns_cpu_between_them(tmp_path: Path) -> None:
    clock = FakeClock()
    write_process(tmp_path, 42, ticks=100, rss_kib=2048 * 1024)
    reader = ProcessReader(42, proc_root=tmp_path, clock=clock, ticks_per_second=100)

    first = reader.read()
    clock.now_ns = 2_000_000_000
    write_process(tmp_path, 42, ticks=400, rss_kib=2048 * 1024)
    second = reader.read()

    assert first["service_cpu"] == ""
    assert second["service_cpu"] == "150.0%"
    assert second["service_memory"] == "2048MiB"
    assert second["service_main_thread_cpu"] == "150.0%"


def test_docker_stats_reader_returns_docker_formats() -> None:
    reader = DockerStatsReader(
        "service", runner=lambda command: "31.25%,1.2GiB / 15GiB\n"
    )

    assert reader.read() == {"service_cpu": "31.25%", "service_memory": "1.2GiB"}


def test_container_cgroup_reader_two_reads_returns_cpu_percent(tmp_path: Path) -> None:
    clock = FakeClock()
    stat_file = tmp_path / "docker-abc.scope" / "cpu.stat"
    stat_file.parent.mkdir()
    stat_file.write_text("usage_usec 1000000\nuser_usec 5\n")
    reader = ContainerCgroupReader(
        "qdrant", runner=lambda command: "abc\n", cgroup_root=tmp_path, clock=clock
    )

    first = reader.read()
    clock.now_ns = 1_000_000_000
    stat_file.write_text("usage_usec 5000000\nuser_usec 5\n")
    second = reader.read()

    assert first == {"qdrant_cpu": ""}
    assert second == {"qdrant_cpu": "400.0%"}
    assert reader.cpu_seconds() == 5.0


def test_container_cgroup_reader_without_container_returns_empty(
    tmp_path: Path,
) -> None:
    reader = ContainerCgroupReader(
        "missing", runner=lambda command: "", cgroup_root=tmp_path
    )

    assert reader.read() == {"qdrant_cpu": ""}
    assert reader.cpu_seconds() is None


def test_nvidia_gpu_reader_returns_utilisation_and_process_ids() -> None:
    outputs = {
        "--query-gpu=utilization.gpu,memory.used": "34, 5089\n",
        "--query-compute-apps=pid": "8128\n1693722\n",
    }

    def runner(command: list[str]) -> str:
        return next(text for flag, text in outputs.items() if flag in command)

    reader = NvidiaGpuReader(runner=runner)

    assert reader.read() == {"gpu_util": "34", "gpu_memory_mib": "5089"}
    assert reader.compute_process_ids() == {8128, 1693722}


def test_sample_row_keeps_original_columns_first() -> None:
    row = sample_row(
        "10:00:00",
        {"service_cpu": "12.5%", "gpu_util": "40", "qdrant_cpu": "300.0%"},
    )

    assert RESOURCE_COLUMNS == (
        "service_cpu",
        "service_memory",
        "k6_cpu",
        "gpu_util",
        "gpu_memory_mib",
        "qdrant_cpu",
        "service_main_thread_cpu",
    )
    assert row == "10:00:00,12.5%,,,40,,300.0%,"
