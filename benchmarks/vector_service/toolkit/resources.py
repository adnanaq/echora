"""Read CPU, memory and GPU use of the service, Qdrant and k6 during a run.

Each reader returns the resource CSV columns it knows, already formatted.
Readers that measure CPU as a rate (process ticks, cgroup usage) report
nothing on their first read and the rate since the previous read afterwards;
a cumulative figure such as ``ps %cpu`` would average over the whole process
life instead. Commands and ``/proc`` / cgroup roots are injectable for tests.
"""

import os
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

CommandRunner = Callable[[list[str]], str]
Clock = Callable[[], int]

RESOURCE_COLUMNS = (
    "service_cpu",
    "service_memory",
    "k6_cpu",
    "gpu_util",
    "gpu_memory_mib",
    "qdrant_cpu",
    "service_main_thread_cpu",
)
CGROUP_ROOT = Path("/sys/fs/cgroup/system.slice")


def run_command(command: list[str]) -> str:
    """Run a command and return its output, or "" when it fails or is missing."""
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argument lists built here
            command, capture_output=True, text=True, check=False, timeout=10
        )
    except OSError, subprocess.TimeoutExpired:
        return ""
    return completed.stdout if completed.returncode == 0 else ""


class ResourceReader(Protocol):
    def read(self) -> dict[str, str]: ...


class CpuRate:
    """CPU use as a percentage of one core, from a cumulative counter in seconds."""

    def __init__(self, clock: Clock) -> None:
        self._clock = clock
        self._previous: tuple[float, int] | None = None

    def update(self, cpu_seconds: float) -> str:
        now_ns = self._clock()
        previous, self._previous = self._previous, (cpu_seconds, now_ns)
        if previous is None or now_ns <= previous[1]:
            return ""
        elapsed = (now_ns - previous[1]) / 1e9
        return f"{100 * (cpu_seconds - previous[0]) / elapsed:.1f}%"


class ProcessReader:
    """A local process: CPU from ``/proc/<pid>/stat`` ticks, memory from VmRSS."""

    def __init__(
        self,
        pid: int,
        proc_root: Path = Path("/proc"),
        clock: Clock = time.monotonic_ns,
        ticks_per_second: int | None = None,
    ) -> None:
        self._directory = proc_root / str(pid)
        self._pid = pid
        self._ticks_per_second = ticks_per_second or os.sysconf("SC_CLK_TCK")
        self._process_rate = CpuRate(clock)
        self._main_thread_rate = CpuRate(clock)

    def read(self) -> dict[str, str]:
        try:
            process_seconds = self._cpu_seconds(self._directory / "stat")
            thread_stat = self._directory / "task" / str(self._pid) / "stat"
            main_thread_seconds = self._cpu_seconds(thread_stat)
            memory = self._memory()
        except FileNotFoundError, ProcessLookupError:
            return {}
        return {
            "service_cpu": self._process_rate.update(process_seconds),
            "service_memory": memory,
            "service_main_thread_cpu": self._main_thread_rate.update(
                main_thread_seconds
            ),
        }

    def _cpu_seconds(self, stat_file: Path) -> float:
        fields = stat_file.read_text().rsplit(")", 1)[1].split()
        return (int(fields[11]) + int(fields[12])) / self._ticks_per_second

    def _memory(self) -> str:
        for line in (self._directory / "status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return f"{int(line.split()[1]) / 1024:.0f}MiB"
        return ""


class DockerStatsReader:
    """A container through ``docker stats``, in Docker's own number formats."""

    def __init__(
        self,
        container: str,
        cpu_column: str = "service_cpu",
        memory_column: str | None = "service_memory",
        runner: CommandRunner = run_command,
    ) -> None:
        self._container = container
        self._cpu_column = cpu_column
        self._memory_column = memory_column
        self._runner = runner

    def read(self) -> dict[str, str]:
        output = self._runner(
            [
                "docker", "stats", "--no-stream",
                "--format", "{{.CPUPerc}},{{.MemUsage}}", self._container,
            ]
        ).strip()  # fmt: skip
        if not output:
            return {}
        cpu, _, memory = output.partition(",")
        values = {self._cpu_column: cpu}
        if self._memory_column:
            values[self._memory_column] = memory.split(" / ")[0]
        return values


class ContainerCgroupReader:
    """A Docker container's CPU from its cgroup v2 ``cpu.stat`` counter."""

    def __init__(
        self,
        container: str,
        runner: CommandRunner = run_command,
        cgroup_root: Path = CGROUP_ROOT,
        clock: Clock = time.monotonic_ns,
    ) -> None:
        self._container = container
        self._runner = runner
        self._cgroup_root = cgroup_root
        self._rate = CpuRate(clock)
        self._stat_file: Path | None = None

    def cpu_seconds(self) -> float | None:
        """Total CPU time the container has used, or None when it cannot be read."""
        stat_file = self._find_stat_file()
        if stat_file is None:
            return None
        for line in stat_file.read_text().splitlines():
            if line.startswith("usage_usec"):
                return int(line.split()[1]) / 1_000_000
        return None

    def read(self) -> dict[str, str]:
        seconds = self.cpu_seconds()
        return {"qdrant_cpu": self._rate.update(seconds) if seconds is not None else ""}

    def _find_stat_file(self) -> Path | None:
        if self._stat_file is None:
            container_id = self._runner(
                ["docker", "inspect", "-f", "{{.Id}}", self._container]
            ).strip()
            candidate = self._cgroup_root / f"docker-{container_id}.scope" / "cpu.stat"
            if container_id and candidate.exists():
                self._stat_file = candidate
        return self._stat_file


class NvidiaGpuReader:
    """GPU use and the processes on the GPU, from ``nvidia-smi``."""

    def __init__(self, runner: CommandRunner = run_command) -> None:
        self._runner = runner

    def read(self) -> dict[str, str]:
        output = self._runner(
            [
                "nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ]
        )  # fmt: skip
        first_line = output.strip().splitlines()[:1]
        if not first_line:
            return {}
        utilisation, _, memory = first_line[0].replace(" ", "").partition(",")
        return {"gpu_util": utilisation, "gpu_memory_mib": memory}

    def compute_process_ids(self) -> set[int]:
        output = self._runner(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"]
        )
        return {int(line) for line in output.split() if line.strip().isdigit()}


def sample_row(clock_text: str, values: dict[str, str]) -> str:
    return ",".join([clock_text, *(values.get(name, "") for name in RESOURCE_COLUMNS)])


class ResourceSampler:
    """Write one CSV row of every reader's values every ``sample_seconds``."""

    def __init__(
        self, path: Path, readers: list[ResourceReader], sample_seconds: float
    ) -> None:
        self._path = path
        self._readers = readers
        self._sample_seconds = sample_seconds
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._path.write_text(",".join(["time", *RESOURCE_COLUMNS]) + "\n")
        self._thread.start()

    def stop(self) -> None:
        self._stopped.set()
        self._thread.join(timeout=self._sample_seconds + 15)

    def _run(self) -> None:
        with self._path.open("a") as csv_file:
            while not self._stopped.is_set():
                values: dict[str, str] = {}
                for reader in self._readers:
                    values.update(reader.read())
                csv_file.write(sample_row(time.strftime("%H:%M:%S"), values) + "\n")
                csv_file.flush()
                self._stopped.wait(self._sample_seconds)
