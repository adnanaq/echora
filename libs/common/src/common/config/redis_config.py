"""Redis connection configuration, shared by every service that uses Redis."""

from pydantic import Field

from .env_settings import EnvSettings


class RedisConfig(EnvSettings):
    """How to reach Redis and size its connection pool.

    Databases 0-15 belong to the services: enrichment uses 0 and the vector
    service 1. Tests use 16-31 on the dev Redis. Setting ``redis_url`` to None
    turns Redis off: the vector service skips its embedding cache and
    enrichment runs without its HTTP and result caches.
    """

    redis_url: str | None = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL; database 0 is enrichment's, 1 the vector service's",
    )
    redis_max_connections: int = Field(
        default=100,
        ge=1,
        description=(
            "Most connections in one pool; enrichment runs 20 agents with about "
            "10 operations each"
        ),
    )
    redis_socket_keepalive: bool = Field(
        default=True,
        description="TCP keepalive, so idle connections dropped by NAT or firewalls are noticed",
    )
    redis_socket_connect_timeout: int = Field(
        default=5, gt=0, description="Seconds to connect before giving up"
    )
    redis_socket_timeout: int = Field(
        default=10, gt=0, description="Seconds a read or write may take"
    )
    redis_retry_on_timeout: bool = Field(
        default=True,
        description="Retry an operation that timed out; cache operations are safe to repeat",
    )
    redis_health_check_interval: int = Field(
        default=30,
        ge=0,
        description="Seconds between checks that a pooled connection still works (0 = never)",
    )
