"""Benchmark environments: where the service, Qdrant and results are, and how to run them.

One TOML file per environment (``environments/<name>.toml``). Every value can
be overridden for a single run with ``section.field=value``, and a service
environment variable with ``service_start.env.NAME=value``. Secrets are never
stored in the file: ``qdrant.api_key_env`` names the environment variable that
holds the Qdrant API key.
"""

import tomllib
from dataclasses import dataclass, field, fields
from os import environ
from pathlib import Path
from typing import TYPE_CHECKING, Any, get_origin

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
ENVIRONMENTS_DIR = Path(__file__).resolve().parents[1] / "environments"
TRUE_WORDS = frozenset({"true", "1", "yes"})


class UnknownSettingError(KeyError):
    """Raised for a setting name that no environment section defines."""

    def __init__(self, name: str) -> None:
        super().__init__(f"unknown benchmark setting: {name}")


class ProtectedCollectionError(ValueError):
    """Raised when a benchmark would use a collection that must not be touched."""

    def __init__(self, collection: str) -> None:
        super().__init__(f"collection {collection} is protected from benchmarks")


@dataclass(frozen=True)
class ServiceTarget:
    address: str = "localhost:8011"
    tls: bool = False


@dataclass(frozen=True)
class ServiceStart:
    kind: str = "local_process"
    port: int = 8011
    python: str = ".venv/bin/python"
    pythonpath_globs: tuple[str, ...] = ("libs/*/src", "apps/*/src")
    qdrant_url: str = ""
    docker_image: str = "echora-vector-service:dev"
    docker_container: str = "echora-bench-vector-service"
    docker_network: str = "echora-dev_echora-network"
    docker_cpus: str = ""
    docker_memory: str = ""
    docker_gpus: str = ""
    startup_timeout_seconds: float = 600.0
    env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class QdrantTarget:
    url: str = "http://localhost:6333"
    api_key_env: str = "QDRANT_API_KEY"
    container: str = "echora-dev-qdrant"

    def api_key(self) -> str | None:
        return environ.get(self.api_key_env) or None


@dataclass(frozen=True)
class Collections:
    load: str = "anime_load_test"
    accuracy: str = "anime_accuracy_test"
    protected: tuple[str, ...] = ("anime_database",)

    def __post_init__(self) -> None:
        for collection in (self.load, self.accuracy):
            if collection in self.protected:
                raise ProtectedCollectionError(collection)


@dataclass(frozen=True)
class ResourceSources:
    service: str = "process"
    service_container: str = ""
    qdrant: bool = True
    gpu: bool = True
    sample_seconds: float = 2.0


@dataclass(frozen=True)
class K6Settings:
    image: str = "grafana/k6:2.3.0"
    prometheus_url: str = "http://localhost:9090/api/v1/write"
    html_report: bool = True
    csv_time_format: str = "unix_milli"


SECTIONS: dict[str, type[DataclassInstance]] = {
    "service_target": ServiceTarget,
    "service_start": ServiceStart,
    "qdrant": QdrantTarget,
    "collections": Collections,
    "resources": ResourceSources,
    "k6": K6Settings,
}


@dataclass(frozen=True)
class Environment:
    name: str
    results_dir: Path
    service_target: ServiceTarget
    service_start: ServiceStart
    qdrant: QdrantTarget
    collections: Collections
    resources: ResourceSources
    k6: K6Settings


def load_environment(
    path_or_name: Path | str, overrides: list[str] | None = None
) -> Environment:
    """Load an environment file, then apply ``section.field=value`` overrides."""
    path = environment_path(path_or_name)
    data = tomllib.loads(path.read_text())
    for override in overrides or []:
        apply_override(data, override)
    return build_environment(data, default_name=path.stem)


def environment_path(path_or_name: Path | str) -> Path:
    path = Path(path_or_name)
    if path.exists():
        return path
    return ENVIRONMENTS_DIR / f"{path_or_name}.toml"


def apply_override(data: dict[str, Any], override: str) -> None:
    name, _, value = override.partition("=")
    parts = name.strip().split(".")
    if len(parts) < 2:
        data[parts[0]] = value
        return
    target = data.setdefault(parts[0], {})
    for part in parts[1:-1]:
        target = target.setdefault(part, {})
    target[parts[-1]] = value


def build_environment(data: dict[str, Any], default_name: str) -> Environment:
    unknown = set(data) - set(SECTIONS) - {"name", "results_dir"}
    if unknown:
        raise UnknownSettingError(sorted(unknown)[0])
    sections = {
        section: build_section(section, section_type, data.get(section, {}))
        for section, section_type in SECTIONS.items()
    }
    results_dir = Path(
        data.get("results_dir", "benchmarks/vector_service/load/results")
    )
    if not results_dir.is_absolute():
        results_dir = REPOSITORY_ROOT / results_dir
    return Environment(
        name=str(data.get("name", default_name)),
        results_dir=results_dir,
        **sections,
    )


def build_section(
    section: str, section_type: type[DataclassInstance], values: dict[str, Any]
) -> Any:
    field_types = {item.name: item.type for item in fields(section_type)}
    converted = {}
    for key, value in values.items():
        if key not in field_types:
            raise UnknownSettingError(f"{section}.{key}")
        converted[key] = convert(value, field_types[key])
    return section_type(**converted)


def convert(value: Any, target_type: Any) -> Any:
    origin = get_origin(target_type)
    if origin is tuple:
        items = value.split(",") if isinstance(value, str) else value
        return tuple(str(item).strip() for item in items)
    if origin is dict:
        return {str(key): str(item) for key, item in value.items()}
    if target_type is bool:
        return value if isinstance(value, bool) else str(value).lower() in TRUE_WORDS
    if target_type in (int, float, str):
        return target_type(value)
    return value
