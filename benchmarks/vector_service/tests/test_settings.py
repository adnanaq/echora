from pathlib import Path

import pytest

from benchmarks.vector_service.toolkit.settings import (
    ProtectedCollectionError,
    UnknownSettingError,
    load_environment,
)

ENVIRONMENT = """
name = "test"
results_dir = "results"

[service_target]
address = "localhost:9000"

[service_start]
kind = "local_process"
port = 9000

[service_start.env]
ENABLE_GPU = "true"

[qdrant]
url = "http://qdrant.example:6333"
api_key_env = "BENCHMARK_QDRANT_KEY"
container = "qdrant-test"

[collections]
load = "load_collection"
accuracy = "accuracy_collection"
"""


def write_environment(tmp_path: Path, text: str = ENVIRONMENT) -> Path:
    path = tmp_path / "test.toml"
    path.write_text(text)
    return path


def test_load_environment_reads_every_section(tmp_path: Path) -> None:
    environment = load_environment(write_environment(tmp_path))

    assert environment.name == "test"
    assert environment.service_target.address == "localhost:9000"
    assert environment.service_target.tls is False
    assert environment.service_start.port == 9000
    assert environment.service_start.env == {"ENABLE_GPU": "true"}
    assert environment.qdrant.url == "http://qdrant.example:6333"
    assert environment.qdrant.container == "qdrant-test"
    assert environment.collections.load == "load_collection"


def test_load_environment_missing_sections_use_defaults(tmp_path: Path) -> None:
    environment = load_environment(write_environment(tmp_path, 'name = "bare"\n'))

    assert environment.k6.image == "grafana/k6:2.3.0"
    assert environment.resources.sample_seconds == 2.0
    assert environment.collections.protected == ("anime_database",)


def test_load_environment_results_dir_is_relative_to_repository_root(
    tmp_path: Path,
) -> None:
    environment = load_environment(write_environment(tmp_path))

    assert environment.results_dir.is_absolute()
    assert environment.results_dir.name == "results"


def test_load_environment_overrides_are_converted_to_field_type(tmp_path: Path) -> None:
    environment = load_environment(
        write_environment(tmp_path),
        [
            "service_start.port=9100",
            "service_target.tls=true",
            "resources.sample_seconds=0.5",
            "collections.protected=one,two",
        ],
    )

    assert environment.service_start.port == 9100
    assert environment.service_target.tls is True
    assert environment.resources.sample_seconds == 0.5
    assert environment.collections.protected == ("one", "two")


def test_load_environment_override_sets_service_environment_variable(
    tmp_path: Path,
) -> None:
    environment = load_environment(
        write_environment(tmp_path), ["service_start.env.EMBED_MAX_CONCURRENCY=1"]
    )

    assert environment.service_start.env == {
        "ENABLE_GPU": "true",
        "EMBED_MAX_CONCURRENCY": "1",
    }


def test_load_environment_unknown_override_raises_unknown_setting(
    tmp_path: Path,
) -> None:
    with pytest.raises(UnknownSettingError):
        load_environment(write_environment(tmp_path), ["service_start.prot=1"])


def test_load_environment_unknown_key_in_file_raises_unknown_setting(
    tmp_path: Path,
) -> None:
    with pytest.raises(UnknownSettingError):
        load_environment(
            write_environment(tmp_path, "[qdrant]\nadress = 'x'\n"),
        )


def test_load_environment_protected_collection_raises_protected_collection(
    tmp_path: Path,
) -> None:
    with pytest.raises(ProtectedCollectionError):
        load_environment(
            write_environment(tmp_path), ["collections.load=anime_database"]
        )


def test_load_environment_protected_image_collection_raises_protected_collection(
    tmp_path: Path,
) -> None:
    with pytest.raises(ProtectedCollectionError):
        load_environment(
            write_environment(tmp_path), ["collections.image_load=anime_database"]
        )


def test_load_environment_image_collection_has_own_default_name(tmp_path: Path) -> None:
    environment = load_environment(write_environment(tmp_path))

    assert environment.collections.image_load == "anime_image_load_test"


def test_load_environment_api_key_read_from_named_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = load_environment(write_environment(tmp_path))

    monkeypatch.delenv("BENCHMARK_QDRANT_KEY", raising=False)
    assert environment.qdrant.api_key() is None
    monkeypatch.setenv("BENCHMARK_QDRANT_KEY", "secret-value")
    assert environment.qdrant.api_key() == "secret-value"
