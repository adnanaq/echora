#!/usr/bin/env python3
"""Run the vector search load test once, sampling CPU, memory and GPU use.

Usage:
    uv run python -m benchmarks.vector_service.load.run_load_test TEST_TYPE [k6 args...]
        [--environment laptop] [--set section.field=value] [--target host:port]
        [--service-pid PID | --service-container NAME] [--no-resources]

Anything not recognised is passed to k6 in order, for example
``-e RATE=40 -e HOLD=5m``. The service's resources are read from a local
process (``--service-pid``) or a container (``--service-container``, default
``echora-dev-vector-service``); ``SERVICE_PID``, ``SERVICE_CONTAINER``,
``SAMPLE_RESOURCES=false`` and ``K6_PROMETHEUS_URL`` work as environment
variables too. Without ``--target`` the k6 script's default target applies,
unless ``-e TARGET=...`` is given.
"""

import argparse
import os
import sys

from benchmarks.vector_service.toolkit.k6_runner import LoadRun, run_load_test
from benchmarks.vector_service.toolkit.resources import (
    ContainerCgroupReader,
    DockerStatsReader,
    NvidiaGpuReader,
    ProcessReader,
    ResourceReader,
    ResourceSampler,
)
from benchmarks.vector_service.toolkit.settings import Environment, load_environment


def parse_arguments(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], allow_abbrev=False
    )
    parser.add_argument("test_type")
    parser.add_argument("--environment", default="laptop")
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--target")
    parser.add_argument("--service-pid", type=int, default=env_int("SERVICE_PID"))
    parser.add_argument(
        "--service-container",
        default=os.environ.get("SERVICE_CONTAINER", "echora-dev-vector-service"),
    )
    parser.add_argument(
        "--no-resources",
        action="store_true",
        default=os.environ.get("SAMPLE_RESOURCES", "true") != "true",
    )
    return parser.parse_known_args(argv)


def env_int(name: str) -> int | None:
    value = os.environ.get(name, "")
    return int(value) if value else None


def resource_readers(
    environment: Environment,
    run: LoadRun,
    service_pid: int | None,
    service_container: str,
) -> list[ResourceReader]:
    readers: list[ResourceReader] = []
    if service_pid is not None:
        readers.append(ProcessReader(service_pid))
    else:
        readers.append(DockerStatsReader(service_container))
    readers.append(
        DockerStatsReader(f"k6-{run.run_id}", cpu_column="k6_cpu", memory_column=None)
    )
    if environment.resources.gpu:
        readers.append(NvidiaGpuReader())
    if environment.resources.qdrant:
        readers.append(ContainerCgroupReader(environment.qdrant.container))
    return readers


def main(argv: list[str] | None = None) -> int:
    arguments, k6_arguments = parse_arguments(sys.argv[1:] if argv is None else argv)
    overrides = list(arguments.overrides)
    if os.environ.get("K6_PROMETHEUS_URL"):
        overrides.append(f"k6.prometheus_url={os.environ['K6_PROMETHEUS_URL']}")
    environment = load_environment(arguments.environment, overrides)

    def sampler_factory(run: LoadRun) -> ResourceSampler | None:
        if arguments.no_resources:
            return None
        readers = resource_readers(
            environment, run, arguments.service_pid, arguments.service_container
        )
        return ResourceSampler(
            run.resources_file, readers, environment.resources.sample_seconds
        )

    run, exit_code = run_load_test(
        environment,
        arguments.test_type,
        k6_arguments,
        target=arguments.target,
        sampler_factory=sampler_factory,
    )
    print(f"run id: {run.run_id}", file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
