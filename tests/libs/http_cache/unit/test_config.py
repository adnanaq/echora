import pytest
from http_cache.config import CacheConfig, get_cache_config
from pydantic import ValidationError

SERVICE_TTL_FIELDS = (
    "ttl_mal",
    "ttl_anilist",
    "ttl_anidb",
    "ttl_kitsu",
    "ttl_anime_planet",
    "ttl_anisearch",
    "ttl_animeschedule",
)


@pytest.fixture
def fresh_cache_config():
    get_cache_config.cache_clear()
    yield
    get_cache_config.cache_clear()


def test_cache_config_returns_documented_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("REDIS_URL", raising=False)
    config = CacheConfig()

    assert config.cache_enabled is True
    assert config.storage_type == "redis"
    assert config.redis_url == "redis://localhost:6379/0"
    assert config.ttl_mal == 604800
    assert config.ttl_anilist == 604800
    assert config.ttl_anidb == 604800
    assert config.ttl_kitsu == 604800
    assert config.ttl_anime_planet == 604800
    assert config.ttl_anisearch == 604800
    assert config.ttl_animeschedule == 86400


def test_cache_config_custom_redis_values_are_kept() -> None:
    config = CacheConfig(
        cache_enabled=True,
        storage_type="redis",
        redis_url="redis://custom-host:6380/1",
        ttl_mal=3600,
        ttl_anilist=7200,
    )

    assert config.cache_enabled is True
    assert config.storage_type == "redis"
    assert config.redis_url == "redis://custom-host:6380/1"
    assert config.ttl_mal == 3600
    assert config.ttl_anilist == 7200


def test_cache_config_cache_disabled_keeps_redis_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("REDIS_URL", raising=False)
    config = CacheConfig(cache_enabled=False)

    assert config.cache_enabled is False
    assert config.storage_type == "redis"
    assert config.redis_url == "redis://localhost:6379/0"


def test_cache_config_custom_service_ttls_are_kept() -> None:
    custom_ttls = {
        field: 1800 * (index + 1) for index, field in enumerate(SERVICE_TTL_FIELDS)
    }

    config = CacheConfig(**custom_ttls)

    assert {field: getattr(config, field) for field in SERVICE_TTL_FIELDS} == (
        custom_ttls
    )


def test_cache_config_invalid_storage_type_raises_validation_error() -> None:
    with pytest.raises(ValidationError) as exc_info:
        CacheConfig(storage_type="invalid")  # type: ignore

    errors = exc_info.value.errors()
    assert len(errors) > 0
    assert "storage_type" in str(errors[0])


def test_cache_config_non_boolean_cache_enabled_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        CacheConfig(cache_enabled="not_a_bool")  # type: ignore


def test_cache_config_non_integer_ttl_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        CacheConfig(ttl_mal="not_an_int")  # type: ignore


@pytest.mark.parametrize("ttl", [-1, 0, 31_536_000])
def test_cache_config_any_integer_ttl_is_accepted(ttl: int) -> None:
    assert CacheConfig(ttl_anidb=ttl).ttl_anidb == ttl


@pytest.mark.parametrize(
    "redis_url",
    [
        "redis://localhost:6379/0",
        "redis://:password@localhost:6379/0",
        "redis://user:password@host:6379/0",
        "redis+sentinel://localhost:26379/mymaster/0",
    ],
    ids=["standard", "password", "username_and_password", "sentinel"],
)
def test_cache_config_redis_url_format_is_kept(redis_url: str) -> None:
    assert CacheConfig(redis_url=redis_url).redis_url == redis_url


@pytest.mark.usefixtures("fresh_cache_config")
def test_get_cache_config_no_environment_variables_returns_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CACHE_ENABLED", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)

    config = get_cache_config()

    assert config.cache_enabled is True
    assert config.storage_type == "redis"
    assert config.redis_url == "redis://localhost:6379/0"


@pytest.mark.usefixtures("fresh_cache_config")
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("true", True),
        ("false", False),
        ("TRUE", True),
        ("False", False),
        ("TrUe", True),
    ],
)
def test_get_cache_config_cache_enabled_reads_boolean_in_any_case(
    monkeypatch: pytest.MonkeyPatch, value: str, expected: bool
) -> None:
    monkeypatch.setenv("CACHE_ENABLED", value)

    assert get_cache_config().cache_enabled is expected


@pytest.mark.usefixtures("fresh_cache_config")
def test_get_cache_config_invalid_cache_enabled_raises_validation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CACHE_ENABLED", "invalid")

    with pytest.raises(ValidationError) as exc_info:
        get_cache_config()

    errors = exc_info.value.errors()
    assert any(error["loc"] == ("cache_enabled",) for error in errors)


@pytest.mark.usefixtures("fresh_cache_config")
def test_get_cache_config_storage_type_is_redis() -> None:
    assert get_cache_config().storage_type == "redis"


@pytest.mark.usefixtures("fresh_cache_config")
@pytest.mark.parametrize(
    "redis_url",
    ["redis://prod-redis:6379/2", "redis://:mypassword@secure-redis:6379/0", ""],
    ids=["custom_host", "password", "empty_string"],
)
def test_get_cache_config_redis_url_environment_variable_is_used_as_given(
    monkeypatch: pytest.MonkeyPatch, redis_url: str
) -> None:
    monkeypatch.setenv("REDIS_URL", redis_url)

    assert get_cache_config().redis_url == redis_url


@pytest.mark.usefixtures("fresh_cache_config")
def test_get_cache_config_all_environment_variables_set_reads_each(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CACHE_ENABLED", "false")
    monkeypatch.setenv("REDIS_URL", "redis://custom:6380/1")

    config = get_cache_config()

    assert config.cache_enabled is False
    assert config.storage_type == "redis"
    assert config.redis_url == "redis://custom:6380/1"


@pytest.mark.usefixtures("fresh_cache_config")
def test_get_cache_config_one_environment_variable_set_keeps_other_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.setenv("CACHE_ENABLED", "false")

    config = get_cache_config()

    assert config.cache_enabled is False
    assert config.storage_type == "redis"
    assert config.redis_url == "redis://localhost:6379/0"
    assert config.ttl_mal == 604800
    assert config.ttl_anilist == 604800


@pytest.mark.usefixtures("fresh_cache_config")
def test_get_cache_config_repeated_calls_return_same_instance_until_cleared() -> None:
    first = get_cache_config()
    second = get_cache_config()
    get_cache_config.cache_clear()
    after_clear = get_cache_config()

    assert first is second
    assert after_clear is not first


@pytest.mark.usefixtures("fresh_cache_config")
@pytest.mark.parametrize(
    ("variable", "value", "field", "expected"),
    [
        ("REDIS_MAX_CONNECTIONS", "50", "redis_max_connections", 50),
        ("REDIS_SOCKET_KEEPALIVE", "false", "redis_socket_keepalive", False),
        ("REDIS_SOCKET_CONNECT_TIMEOUT", "10", "redis_socket_connect_timeout", 10),
        ("REDIS_SOCKET_TIMEOUT", "20", "redis_socket_timeout", 20),
        ("REDIS_RETRY_ON_TIMEOUT", "false", "redis_retry_on_timeout", False),
        ("REDIS_HEALTH_CHECK_INTERVAL", "60", "redis_health_check_interval", 60),
    ],
)
def test_get_cache_config_redis_pool_environment_variable_sets_field(
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
    value: str,
    field: str,
    expected: object,
) -> None:
    monkeypatch.setenv(variable, value)

    assert getattr(get_cache_config(), field) == expected


@pytest.mark.usefixtures("fresh_cache_config")
def test_get_cache_config_all_redis_pool_environment_variables_set_reads_each(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REDIS_URL", "redis://prod:6379/0")
    monkeypatch.setenv("REDIS_MAX_CONNECTIONS", "200")
    monkeypatch.setenv("REDIS_SOCKET_KEEPALIVE", "true")
    monkeypatch.setenv("REDIS_SOCKET_CONNECT_TIMEOUT", "3")
    monkeypatch.setenv("REDIS_SOCKET_TIMEOUT", "15")
    monkeypatch.setenv("REDIS_RETRY_ON_TIMEOUT", "true")
    monkeypatch.setenv("REDIS_HEALTH_CHECK_INTERVAL", "45")

    config = get_cache_config()

    assert config.redis_url == "redis://prod:6379/0"
    assert config.redis_max_connections == 200
    assert config.redis_socket_keepalive is True
    assert config.redis_socket_connect_timeout == 3
    assert config.redis_socket_timeout == 15
    assert config.redis_retry_on_timeout is True
    assert config.redis_health_check_interval == 45
