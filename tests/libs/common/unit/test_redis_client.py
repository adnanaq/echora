from common.config import RedisConfig
from common.redis_client import create_redis_client


def test_create_redis_client_without_redis_url_returns_none() -> None:
    assert create_redis_client(RedisConfig(redis_url=None)) is None


def test_create_redis_client_applies_pool_settings() -> None:
    config = RedisConfig(
        redis_url="redis://cache-host:6380/3",
        redis_max_connections=7,
        redis_socket_keepalive=False,
        redis_socket_connect_timeout=2,
        redis_socket_timeout=4,
        redis_retry_on_timeout=False,
        redis_health_check_interval=11,
    )

    client = create_redis_client(config)

    assert client is not None
    pool = client.connection_pool
    connection = pool.connection_kwargs
    assert pool.max_connections == 7
    assert (connection["host"], connection["port"], connection["db"]) == (
        "cache-host",
        6380,
        3,
    )
    assert connection["socket_keepalive"] is False
    assert connection["socket_connect_timeout"] == 2
    assert connection["socket_timeout"] == 4
    assert connection["health_check_interval"] == 11
    assert connection["decode_responses"] is False


def test_create_redis_client_decode_responses_returns_text_client() -> None:
    client = create_redis_client(
        RedisConfig(redis_url="redis://localhost:6379/1"), decode_responses=True
    )

    assert client is not None
    assert client.connection_pool.connection_kwargs["decode_responses"] is True
