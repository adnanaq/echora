from pathlib import Path
from unittest.mock import MagicMock

import pytest

from benchmarks.vector_service.toolkit.service_control import (
    DockerController,
    GpuNotUsedError,
    LocalProcessController,
    MissingSettingError,
    RunningService,
    ServiceStartError,
    check_gpu_use,
    service_environment,
    wait_until_serving,
)
from benchmarks.vector_service.toolkit.settings import (
    REPOSITORY_ROOT,
    ProtectedCollectionError,
    load_environment,
)


class FakeGpuReader:
    def __init__(self, process_ids: set[int]) -> None:
        self.process_ids = process_ids

    def compute_process_ids(self) -> set[int]:
        return self.process_ids


def test_service_environment_adds_port_collection_and_qdrant() -> None:
    environment = load_environment("laptop")

    variables = service_environment(environment, {"EMBED_MAX_CONCURRENCY": "1"})

    assert variables["ENABLE_GPU"] == "true"
    assert variables["EMBED_MAX_CONCURRENCY"] == "1"
    assert variables["VECTOR_SERVICE_PORT"] == "8011"
    assert variables["QDRANT_COLLECTION_NAME"] == "anime_load_test"
    assert variables["QDRANT_URL"] == "http://localhost:6333"


def test_run_settings_override_the_environment() -> None:
    environment = load_environment("laptop")

    variables = service_environment(environment, {"ENABLE_GPU": "false"})

    assert variables["ENABLE_GPU"] == "false"


def test_a_run_cannot_point_the_service_at_a_protected_collection() -> None:
    environment = load_environment("laptop")

    with pytest.raises(ProtectedCollectionError):
        service_environment(environment, {"QDRANT_COLLECTION_NAME": "anime_database"})


def test_local_process_runs_the_service_module_with_library_paths() -> None:
    controller = LocalProcessController(load_environment("laptop"))

    command = controller.command()
    variables = controller.process_environment({})

    assert command == [
        str(REPOSITORY_ROOT / ".venv/bin/python"),
        "-m",
        "vector_service.main",
    ]
    paths = variables["PYTHONPATH"].split(":")
    assert str(REPOSITORY_ROOT / "libs/common/src") in paths
    assert str(REPOSITORY_ROOT / "apps/vector_service/src") in paths
    assert variables["VECTOR_SERVICE_PORT"] == "8011"


def test_timed_local_process_runs_the_diagnostic_launcher() -> None:
    controller = LocalProcessController(load_environment("laptop"), timed=True)

    assert controller.command()[-1].endswith("diagnostics/run_timed_vector_service.py")


def test_docker_needs_the_qdrant_address_seen_from_the_container() -> None:
    with pytest.raises(MissingSettingError):
        DockerController(load_environment("laptop", ["service_start.kind=docker"]))


def test_docker_command_uses_its_own_container_and_limits() -> None:
    environment = load_environment(
        "laptop",
        [
            "service_start.kind=docker",
            "service_start.qdrant_url=http://qdrant:6333",
            "service_start.docker_cpus=2",
            "service_start.docker_memory=4g",
            "service_start.env.ENABLE_GPU=false",
        ],
    )

    command = DockerController(environment).docker_command(
        {"EMBED_BATCH_MAX_SIZE": "64"}
    )

    assert command[:8] == [
        "docker", "run", "-d", "--name", "echora-bench-vector-service",
        "--network", "echora-dev_echora-network", "-p",
    ]  # fmt: skip
    assert command[8] == "8011:8011"
    assert ["--cpus", "2"] == command[9:11]
    assert ["--memory", "4g"] == command[11:13]
    assert "--gpus" not in command
    assert "QDRANT_URL=http://qdrant:6333" in command
    assert "EMBED_BATCH_MAX_SIZE=64" in command
    assert command[-1] == "echora-vector-service:dev"


def test_docker_refuses_the_dev_container_name() -> None:
    with pytest.raises(ValueError):
        DockerController(
            load_environment(
                "laptop",
                [
                    "service_start.kind=docker",
                    "service_start.qdrant_url=http://qdrant:6333",
                    "service_start.docker_container=echora-dev-vector-service",
                ],
            )
        )


def test_wait_returns_once_the_service_is_serving() -> None:
    answers = iter([False, False, True])

    wait_until_serving(
        "localhost:8011",
        timeout_seconds=10,
        probe=lambda address: next(answers),
        is_alive=lambda: True,
        sleep=lambda seconds: None,
    )


def test_wait_fails_when_the_service_exits(tmp_path: Path) -> None:
    with pytest.raises(ServiceStartError):
        wait_until_serving(
            "localhost:8011",
            timeout_seconds=10,
            probe=lambda address: False,
            is_alive=lambda: False,
            sleep=lambda seconds: None,
        )


def test_wait_fails_after_the_timeout() -> None:
    clock = iter(range(100))

    with pytest.raises(ServiceStartError):
        wait_until_serving(
            "localhost:8011",
            timeout_seconds=5,
            probe=lambda address: False,
            is_alive=lambda: True,
            sleep=lambda seconds: None,
            clock=lambda: next(clock),
        )


def test_gpu_check_passes_when_the_service_is_on_the_gpu(tmp_path: Path) -> None:
    service = RunningService(process_ids=frozenset({42}), log_file=tmp_path / "log")

    check_gpu_use(service, {"ENABLE_GPU": "true"}, FakeGpuReader({7, 42}))


def test_gpu_check_fails_when_the_service_is_not_on_the_gpu(tmp_path: Path) -> None:
    service = RunningService(process_ids=frozenset({42}), log_file=tmp_path / "log")

    with pytest.raises(GpuNotUsedError):
        check_gpu_use(service, {"ENABLE_GPU": "true"}, FakeGpuReader({7}))


def test_gpu_check_is_skipped_for_cpu_runs(tmp_path: Path) -> None:
    service = RunningService(process_ids=frozenset({42}), log_file=tmp_path / "log")

    check_gpu_use(service, {"ENABLE_GPU": "false"}, FakeGpuReader(set()))


def test_local_service_starts_outside_the_repository_root(tmp_path: Path) -> None:
    environment = load_environment("laptop", [f"results_dir={tmp_path}"])
    popen = MagicMock(return_value=MagicMock(pid=42))
    controller = LocalProcessController(environment, popen=popen)

    controller.start({}, tmp_path / "service.log")

    assert popen.call_args.kwargs["cwd"] == tmp_path
    assert popen.call_args.kwargs["cwd"] != REPOSITORY_ROOT
