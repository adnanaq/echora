"""Build Redis clients from ``RedisConfig``, the same way in every service."""

from redis.asyncio import Redis

from common.config.redis_config import RedisConfig


def create_redis_client(
    config: RedisConfig, *, decode_responses: bool = False
) -> Redis | None:
    """Create an async Redis client with the configured connection pool.

    Creating the client opens no connection; the first command does.

    Args:
        config: Where Redis is and how to size its connection pool.
        decode_responses: Return ``str`` instead of ``bytes``. Callers that store
            raw response bodies keep it off; callers that store JSON turn it on.

    Returns:
        The client, or None when no ``redis_url`` is configured.

    Examples:
        >>> create_redis_client(RedisConfig()) is None
        True
        >>> client = create_redis_client(RedisConfig(redis_url="redis://localhost:6379/1"))
        >>> client.connection_pool.max_connections
        100
    """
    if not config.redis_url:
        return None
    return Redis.from_url(
        config.redis_url,
        decode_responses=decode_responses,
        max_connections=config.redis_max_connections,
        socket_keepalive=config.redis_socket_keepalive,
        socket_connect_timeout=config.redis_socket_connect_timeout,
        socket_timeout=config.redis_socket_timeout,
        retry_on_timeout=config.redis_retry_on_timeout,
        health_check_interval=config.redis_health_check_interval,
    )
