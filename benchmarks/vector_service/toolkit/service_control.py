"""Start and stop a benchmark-owned vector service instance with given settings.

The benchmark never touches the dev stack: a local process listens on the
environment's own port, and the Docker kind runs its own container. A run's
service settings are passed as environment variables to that instance only;
nothing is written to the project's configuration.
"""

import os
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Protocol

from benchmarks.vector_service.toolkit.resources import CommandRunner, run_command
from benchmarks.vector_service.toolkit.settings import (
    REPOSITORY_ROOT,
    Environment,
    ProtectedCollectionError,
)

DEV_CONTAINERS = frozenset({"echora-dev-vector-service"})
TIMED_LAUNCHER = "benchmarks/vector_service/diagnostics/run_timed_vector_service.py"
TRUE_WORDS = frozenset({"true", "1", "yes"})


class ServiceStartError(RuntimeError):
    """Raised when the service does not reach SERVING."""


class ServiceExitedError(ServiceStartError):
    def __init__(self) -> None:
        super().__init__("benchmark service exited while starting; see its log")


class ServiceTimeoutError(ServiceStartError):
    def __init__(self, timeout_seconds: float) -> None:
        super().__init__(f"benchmark service not SERVING after {timeout_seconds:.0f} s")


class DockerRunError(ServiceStartError):
    def __init__(self, container: str) -> None:
        super().__init__(f"docker run of benchmark container {container} failed")


class DevContainerRefusedError(ValueError):
    """Raised when a benchmark would replace a dev stack container."""

    def __init__(self, container: str) -> None:
        super().__init__(f"refusing to replace dev container {container}")


class GpuNotUsedError(RuntimeError):
    """Raised when a run asks for the GPU but the service is not on it."""

    def __init__(self) -> None:
        super().__init__(
            "ENABLE_GPU=true but the service is not among nvidia-smi's compute"
            " processes; it would silently run on the CPU"
        )


class MissingSettingError(ValueError):
    """Raised when a service kind needs a setting the environment leaves empty."""

    def __init__(self, name: str) -> None:
        super().__init__(f"benchmark setting {name} must be set for this service kind")


@dataclass(frozen=True)
class RunningService:
    process_ids: frozenset[int]
    log_file: Path


class GpuProcessReader(Protocol):
    def compute_process_ids(self) -> set[int]: ...


class ServiceController(Protocol):
    def start(self, run_env: dict[str, str], log_file: Path) -> RunningService: ...

    def is_alive(self) -> bool: ...

    def stop(self) -> None: ...


def service_environment(
    environment: Environment, run_env: dict[str, str]
) -> dict[str, str]:
    """The service's environment variables: environment defaults, then the run's."""
    start = environment.service_start
    variables = {
        "VECTOR_SERVICE_PORT": str(start.port),
        "QDRANT_COLLECTION_NAME": environment.collections.load,
        "QDRANT_URL": start.qdrant_url or environment.qdrant.url,
    }
    api_key = environment.qdrant.api_key()
    if api_key:
        variables["QDRANT_API_KEY"] = api_key
    variables.update(start.env)
    variables.update(run_env)
    if variables["QDRANT_COLLECTION_NAME"] in environment.collections.protected:
        raise ProtectedCollectionError(variables["QDRANT_COLLECTION_NAME"])
    return variables


class LocalProcessController:
    """The service as a local process, started from the repository's virtualenv."""

    def __init__(
        self,
        environment: Environment,
        timed: bool = False,
        popen: Callable[..., subprocess.Popen] = subprocess.Popen,
    ) -> None:
        self._environment = environment
        self._timed = timed
        self._popen = popen
        self._process: subprocess.Popen | None = None
        self._log: IO[str] | None = None

    def command(self) -> list[str]:
        python = str(REPOSITORY_ROOT / self._environment.service_start.python)
        if self._timed:
            return [python, str(REPOSITORY_ROOT / TIMED_LAUNCHER)]
        return [python, "-m", "vector_service.main"]

    def process_environment(self, run_env: dict[str, str]) -> dict[str, str]:
        paths = [
            str(path)
            for pattern in self._environment.service_start.pythonpath_globs
            for path in sorted(REPOSITORY_ROOT.glob(pattern))
        ]
        return {
            **os.environ,
            "PYTHONPATH": ":".join(paths),
            **service_environment(self._environment, run_env),
        }

    def start(self, run_env: dict[str, str], log_file: Path) -> RunningService:
        self._log = log_file.open("w")
        # Started in the results folder rather than the repository root, so the
        # service does not read the developer's .env: only the environment file
        # and the run's settings apply, and results stay comparable.
        self._process = self._popen(
            self.command(),
            cwd=self._environment.results_dir,
            env=self.process_environment(run_env),
            stdout=self._log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        return RunningService(frozenset({self._process.pid}), log_file)

    def is_alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def stop(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
        if self._log is not None:
            self._log.close()


class DockerController:
    """The service in its own container from the dev image, removed on stop."""

    def __init__(
        self, environment: Environment, runner: CommandRunner = run_command
    ) -> None:
        start = environment.service_start
        if not start.qdrant_url:
            raise MissingSettingError("service_start.qdrant_url")
        if start.docker_container in DEV_CONTAINERS:
            raise DevContainerRefusedError(start.docker_container)
        self._environment = environment
        self._runner = runner
        self._log_file: Path | None = None

    def docker_command(self, run_env: dict[str, str]) -> list[str]:
        start = self._environment.service_start
        limits = [
            item
            for flag, value in (
                ("--cpus", start.docker_cpus),
                ("--memory", start.docker_memory),
                ("--gpus", start.docker_gpus),
            )
            if value
            for item in (flag, value)
        ]
        variables = service_environment(self._environment, run_env)
        return [
            "docker", "run", "-d", "--name", start.docker_container,
            "--network", start.docker_network, "-p", f"{start.port}:{start.port}",
            *limits,
            *[item for name, value in variables.items() for item in ("-e", f"{name}={value}")],
            start.docker_image,
        ]  # fmt: skip

    def start(self, run_env: dict[str, str], log_file: Path) -> RunningService:
        container = self._environment.service_start.docker_container
        self._runner(["docker", "rm", "-f", container])
        if not self._runner(self.docker_command(run_env)).strip():
            self._runner(["docker", "rm", "-f", container])
            raise DockerRunError(container)
        self._log_file = log_file
        return RunningService(self._container_process_ids(), log_file)

    def is_alive(self) -> bool:
        container = self._environment.service_start.docker_container
        state = self._runner(
            ["docker", "inspect", "-f", "{{.State.Running}}", container]
        )
        return state.strip() == "true"

    def stop(self) -> None:
        container = self._environment.service_start.docker_container
        if self._log_file is not None:
            with self._log_file.open("w") as log:
                subprocess.run(  # noqa: S603 - fixed docker command
                    ["docker", "logs", container],  # noqa: S607 - docker from PATH
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
        self._runner(["docker", "rm", "-f", container])

    def _container_process_ids(self) -> frozenset[int]:
        container = self._environment.service_start.docker_container
        output = self._runner(["docker", "top", container, "-eo", "pid"])
        return frozenset(int(line) for line in output.split() if line.isdigit())


def grpc_health_probe(address: str) -> bool:
    """True when the service's gRPC health check reports SERVING."""
    import grpc
    from grpc_health.v1 import health_pb2, health_pb2_grpc

    with grpc.insecure_channel(address) as channel:
        stub = health_pb2_grpc.HealthStub(channel)
        try:
            response = stub.Check(health_pb2.HealthCheckRequest(), timeout=2)
        except grpc.RpcError:
            return False
    return response.status == health_pb2.HealthCheckResponse.SERVING


def wait_until_serving(
    address: str,
    timeout_seconds: float,
    probe: Callable[[str], bool] = grpc_health_probe,
    is_alive: Callable[[], bool] = lambda: True,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> None:
    deadline = clock() + timeout_seconds
    while clock() < deadline:
        if not is_alive():
            raise ServiceExitedError()
        if probe(address):
            return
        sleep(2)
    raise ServiceTimeoutError(timeout_seconds)


def check_gpu_use(
    service: RunningService, run_env: dict[str, str], gpu_reader: GpuProcessReader
) -> None:
    """Fail when a run asks for the GPU but no service process is on it."""
    if run_env.get("ENABLE_GPU", "").lower() not in TRUE_WORDS:
        return
    if not service.process_ids & gpu_reader.compute_process_ids():
        raise GpuNotUsedError()
