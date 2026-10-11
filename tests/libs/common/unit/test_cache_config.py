import pytest
from common.config import CacheConfig
from common.config.cache_config import DEFAULT_TTL
from pydantic import ValidationError

SERVICE_TTLS = {
    "mal": 604800,
    "anilist": 604800,
    "anidb": 604800,
    "kitsu": 604800,
    "anime_planet": 604800,
    "anisearch": 604800,
    "animeschedule": 86400,
}


def test_cache_config_returns_documented_defaults() -> None:
    config = CacheConfig()

    assert config.cache_enabled is True
    assert config.storage_type == "redis"
    assert config.force_cache is True
    assert config.always_revalidate is False
    assert config.max_cache_key_length == 200
    assert config.cloudflare_clearance_max_ttl == 86400


@pytest.mark.parametrize(("service", "expected"), SERVICE_TTLS.items())
def test_ttl_for_known_service_returns_its_ttl(service: str, expected: int) -> None:
    assert CacheConfig().ttl_for(service) == expected


def test_ttl_for_unknown_service_returns_default_ttl() -> None:
    assert CacheConfig().ttl_for("unknown") == DEFAULT_TTL


def test_ttl_for_custom_ttl_returns_custom_value() -> None:
    assert CacheConfig(ttl_mal=3600).ttl_for("mal") == 3600


@pytest.mark.parametrize(
    ("variable", "value", "field", "expected"),
    [
        ("CACHE_ENABLED", "false", "cache_enabled", False),
        ("CACHE_ENABLED", "TrUe", "cache_enabled", True),
        ("FORCE_CACHE", "false", "force_cache", False),
        ("ALWAYS_REVALIDATE", "true", "always_revalidate", True),
        ("MAX_CACHE_KEY_LENGTH", "120", "max_cache_key_length", 120),
        ("TTL_ANIMESCHEDULE", "3600", "ttl_animeschedule", 3600),
        ("CLOUDFLARE_CLEARANCE_MAX_TTL", "600", "cloudflare_clearance_max_ttl", 600),
    ],
)
def test_cache_config_environment_variable_sets_field(
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
    value: str,
    field: str,
    expected: object,
) -> None:
    monkeypatch.setenv(variable, value)

    assert getattr(CacheConfig(), field) == expected


def test_cache_config_value_given_in_code_wins_over_environment_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CACHE_ENABLED", "false")

    assert CacheConfig(cache_enabled=True).cache_enabled is True


def test_cache_config_invalid_cache_enabled_raises_validation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CACHE_ENABLED", "invalid")

    with pytest.raises(ValidationError, match="cache_enabled"):
        CacheConfig()


def test_cache_config_invalid_storage_type_raises_validation_error() -> None:
    with pytest.raises(ValidationError, match="storage_type"):
        CacheConfig(storage_type="invalid")  # type: ignore[arg-type]


@pytest.mark.parametrize("ttl", [0, 31_536_000])
def test_cache_config_zero_or_positive_ttl_is_accepted(ttl: int) -> None:
    assert CacheConfig(ttl_anidb=ttl).ttl_anidb == ttl


def test_cache_config_negative_ttl_raises_validation_error() -> None:
    with pytest.raises(ValidationError, match="ttl_anidb"):
        CacheConfig(ttl_anidb=-1)


@pytest.mark.parametrize(
    "field", ["max_cache_key_length", "cloudflare_clearance_max_ttl"]
)
def test_cache_config_zero_limit_raises_validation_error(field: str) -> None:
    with pytest.raises(ValidationError, match=field):
        CacheConfig(**{field: 0})
