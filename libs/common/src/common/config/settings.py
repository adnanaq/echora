"""Settings for every service, built once per process by ``get_settings``."""

from enum import StrEnum
from functools import lru_cache
from itertools import combinations
from typing import Any

from pydantic import Field, field_validator

from .cache_config import CacheConfig
from .embedding_config import EmbeddingConfig
from .enrichment_config import EnrichmentConfig
from .env_settings import EnvSettings
from .observability_config import ObservabilityConfig
from .qdrant_config import QdrantConfig
from .redis_config import RedisConfig
from .service_config import ServiceConfig


class Environment(StrEnum):
    """Application environment types."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class Settings(EnvSettings):
    """All configuration, one sub-config per concern.

    Every sub-config reads its own environment variables, so ``QDRANT_URL``
    sets ``settings.qdrant.qdrant_url``.
    """

    environment: Environment = Field(
        description="Application environment (development/staging/production)"
    )
    debug: bool = Field(default=True, description="Enable debug mode")

    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    service: ServiceConfig = Field(default_factory=ServiceConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    enrichment: EnrichmentConfig = Field(default_factory=EnrichmentConfig)

    @field_validator("environment", mode="before")
    @classmethod
    def lowercase_environment(cls, value: Any) -> Any:
        """Accept ``ENVIRONMENT`` in any case, such as ``PRODUCTION``."""
        return value.lower() if isinstance(value, str) else value

    def model_post_init(self, __context: Any) -> None:
        """Apply environment-specific overrides after initialization."""
        self.apply_environment_settings()

    def apply_environment_settings(self) -> None:
        """Apply the defaults of each environment.

        Development and staging only change values nobody set, whether in an
        environment variable, ``.env`` or code. Production always enforces
        debug off, WARNING logs, the Qdrant write-ahead log and model warm-up.
        """
        match self.environment:
            case Environment.DEVELOPMENT:
                self._default_unless_set(debug=True, log_level="DEBUG")
            case Environment.STAGING:
                self._default_unless_set(
                    debug=True, log_level="INFO", qdrant_enable_wal=True
                )
            case Environment.PRODUCTION:
                self.debug = False
                self.service.log_level = "WARNING"
                self.qdrant.qdrant_enable_wal = True
                self.embedding.model_warm_up = True

    def _default_unless_set(
        self,
        *,
        debug: bool,
        log_level: str,
        qdrant_enable_wal: bool | None = None,
    ) -> None:
        """Set each given value on fields that were not set explicitly."""
        if "debug" not in self.model_fields_set:
            self.debug = debug
        if "log_level" not in self.service.model_fields_set:
            self.service.log_level = log_level
        if (
            qdrant_enable_wal is not None
            and "qdrant_enable_wal" not in self.qdrant.model_fields_set
        ):
            self.qdrant.qdrant_enable_wal = qdrant_enable_wal


# The fields each class reads from the environment: Settings' own fields and
# every sub-config's.
_ENV_FIELDS: dict[str, set[str]] = {"Settings": set()}
for _name, _field in Settings.model_fields.items():
    _annotation = _field.annotation
    if isinstance(_annotation, type) and issubclass(_annotation, EnvSettings):
        _ENV_FIELDS[_name] = set(_annotation.model_fields)
    else:
        _ENV_FIELDS["Settings"].add(_name)

# One environment variable must feed exactly one field.
# Note: Using assert for import-time structural invariants (acceptable despite -O stripping)
for (_first, _first_fields), (_second, _second_fields) in combinations(
    _ENV_FIELDS.items(), 2
):
    _overlap = _first_fields & _second_fields
    assert not _overlap, f"Field overlap {_first}/{_second}: {sorted(_overlap)}"  # noqa: S101


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance.

    Returns a singleton Settings instance that is cached for the lifetime of the process.
    This is efficient for production use but has implications for testing:

    - Tests should construct Settings() directly, not use get_settings()
    - If get_settings() is used in tests, the cache must be cleared between tests:
      `get_settings.cache_clear()`
    - Changing environment variables after the first call has no effect

    Returns:
        Cached Settings instance
    """
    return Settings()
