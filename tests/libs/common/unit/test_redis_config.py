import pytest
from common.config import RedisConfig
from pydantic import ValidationError


def test_redis_config_returns_documented_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("REDIS_URL", raising=False)

    config = RedisConfig()

    assert config.redis_url == "redis://localhost:6379/0"
    assert config.redis_max_connections == 100
    assert config.redis_socket_keepalive is True
    assert config.redis_socket_connect_timeout == 5
    assert config.redis_socket_timeout == 10
    assert config.redis_retry_on_timeout is True
    assert config.redis_health_check_interval == 30


@pytest.mark.parametrize(
    "redis_url",
    [
        "redis://prod-redis:6379/2",
        "redis://:password@localhost:6379/0",
        "redis://user:password@host:6379/0",
        "redis+sentinel://localhost:26379/mymaster/0",
    ],
    ids=["custom_host", "password", "username_and_password", "sentinel"],
)
def test_redis_config_redis_url_environment_variable_is_used_as_given(
    monkeypatch: pytest.MonkeyPatch, redis_url: str
) -> None:
    monkeypatch.setenv("REDIS_URL", redis_url)

    assert RedisConfig().redis_url == redis_url


@pytest.mark.parametrize(
    ("variable", "value", "field", "expected"),
    [
        ("REDIS_MAX_CONNECTIONS", "50", "redis_max_connections", 50),
        ("REDIS_SOCKET_KEEPALIVE", "false", "redis_socket_keepalive", False),
        ("REDIS_SOCKET_CONNECT_TIMEOUT", "3", "redis_socket_connect_timeout", 3),
        ("REDIS_SOCKET_TIMEOUT", "20", "redis_socket_timeout", 20),
        ("REDIS_RETRY_ON_TIMEOUT", "false", "redis_retry_on_timeout", False),
        ("REDIS_HEALTH_CHECK_INTERVAL", "0", "redis_health_check_interval", 0),
    ],
)
def test_redis_config_pool_environment_variable_sets_field(
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
    value: str,
    field: str,
    expected: object,
) -> None:
    monkeypatch.setenv(variable, value)

    assert getattr(RedisConfig(), field) == expected


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("redis_max_connections", 0),
        ("redis_socket_connect_timeout", 0),
        ("redis_socket_timeout", 0),
        ("redis_health_check_interval", -1),
    ],
)
def test_redis_config_out_of_range_value_raises_validation_error(
    field: str, value: int
) -> None:
    with pytest.raises(ValidationError, match=field):
        RedisConfig(**{field: value})


def test_redis_config_redis_url_none_turns_redis_off() -> None:
    assert RedisConfig(redis_url=None).redis_url is None
