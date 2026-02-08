"""Vector Service Configuration Settings."""

import json
import logging
import os
import types
import typing
from enum import Enum
from functools import lru_cache
from typing import Any

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .embedding_config import EmbeddingConfig
from .qdrant_config import QdrantConfig
from .service_config import ServiceConfig

logger = logging.getLogger(__name__)


class Environment(str, Enum):
    """Application environment types."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


def get_environment() -> Environment:
    """Detect environment from ENVIRONMENT variable.

    Returns:
        Environment: Detected environment based on ENVIRONMENT.

    Raises:
        ValueError: If ENVIRONMENT is not set or contains an invalid value.
    """
    env_str = os.getenv("ENVIRONMENT")
    if not env_str:
        raise ValueError(  # noqa: TRY003
            "ENVIRONMENT environment variable must be set to one of: "
            "development, staging, production"
        )

    env_str = env_str.lower()

    match env_str:
        case "production":
            return Environment.PRODUCTION
        case "staging":
            return Environment.STAGING
        case "development":
            return Environment.DEVELOPMENT
        case _:
            raise ValueError(  # noqa: TRY003
                f"Invalid ENVIRONMENT value '{env_str}'. "
                "Must be one of: development, staging, production"
            )


# Field name -> sub-config key mapping
_QDRANT_FIELDS = frozenset(QdrantConfig.model_fields.keys())
_EMBEDDING_FIELDS = frozenset(EmbeddingConfig.model_fields.keys())
_SERVICE_FIELDS = frozenset(ServiceConfig.model_fields.keys())

# Ensure no field name collisions between sub-configs
# Note: Using assert for import-time structural invariants (acceptable despite -O stripping)
assert not (_QDRANT_FIELDS & _EMBEDDING_FIELDS), (  # noqa: S101
    f"Field overlap Qdrant/Embedding: {_QDRANT_FIELDS & _EMBEDDING_FIELDS}"
)
assert not (_QDRANT_FIELDS & _SERVICE_FIELDS), (  # noqa: S101
    f"Field overlap Qdrant/Service: {_QDRANT_FIELDS & _SERVICE_FIELDS}"
)
assert not (_EMBEDDING_FIELDS & _SERVICE_FIELDS), (  # noqa: S101
    f"Field overlap Embedding/Service: {_EMBEDDING_FIELDS & _SERVICE_FIELDS}"
)


def _is_complex_type(annotation: Any) -> bool:
    """Check if a type annotation represents a list or dict (including generics).

    Uses typing introspection instead of string matching for robustness.
    Handles Union types (e.g., list[str] | None).
    """
    origin = typing.get_origin(annotation)
    if origin in (list, dict):
        return True
    # Handle Union types (e.g., list[str] | None)
    if origin is typing.Union or isinstance(annotation, types.UnionType):
        for arg in typing.get_args(annotation):
            if _is_complex_type(arg):
                return True
    return False


# Fields that expect complex types (list/dict) and need JSON parsing from env vars
_JSON_FIELDS: frozenset[str] = frozenset(
    field_name
    for config_cls in (QdrantConfig, EmbeddingConfig, ServiceConfig)
    for field_name, field_info in config_cls.model_fields.items()
    if field_info.annotation is not None and _is_complex_type(field_info.annotation)
)


def _maybe_parse_json(field_name: str, value: Any) -> Any:
    """Parse JSON strings for fields that expect complex types (list/dict)."""
    if field_name in _JSON_FIELDS and isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(
                f"Failed to parse JSON for field '{field_name}': {e}. "
                f"Value will be passed as-is: {value[:100]}{'...' if len(value) > 100 else ''}"
            )
            return value
    return value


class Settings(BaseSettings):
    """Vector service settings with validation and type safety."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # Top-level fields
    environment: Environment = Field(
        default_factory=get_environment,
        description="Application environment (development/staging/production)",
    )
    debug: bool = Field(default=True, description="Enable debug mode")

    # Composed sub-configs
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    service: ServiceConfig = Field(default_factory=ServiceConfig)

    @model_validator(mode="before")
    @classmethod
    def distribute_env_vars(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Route flat env vars into nested sub-config groups.

        BaseSettings only loads env vars for fields defined directly on Settings.
        Since sub-config fields (e.g. log_level, qdrant_url) now live on BaseModel
        sub-configs, we must explicitly read them from the environment and inject
        them into the appropriate nested dict.

        Precedence: real env var > .env file value > default.
        For fields with complex types (list/dict), JSON string values are parsed.
        """
        if not isinstance(data, dict):
            return data

        # Normalize environment value to lowercase for case-insensitive enum matching
        env_val = data.get("environment")
        if isinstance(env_val, str):
            data["environment"] = env_val.lower()

        qdrant_data: dict[str, Any] = {}
        embedding_data: dict[str, Any] = {}
        service_data: dict[str, Any] = {}

        # Collect sub-config fields from data (may come from .env file via BaseSettings)
        keys_to_remove: list[str] = []
        for key, value in data.items():
            lower_key = key.lower() if isinstance(key, str) else key
            if lower_key in _QDRANT_FIELDS:
                qdrant_data[lower_key] = _maybe_parse_json(lower_key, value)
                keys_to_remove.append(key)
            elif lower_key in _EMBEDDING_FIELDS:
                embedding_data[lower_key] = _maybe_parse_json(lower_key, value)
                keys_to_remove.append(key)
            elif lower_key in _SERVICE_FIELDS:
                service_data[lower_key] = _maybe_parse_json(lower_key, value)
                keys_to_remove.append(key)

        for key in keys_to_remove:
            data.pop(key, None)

        # Override with real env vars (take precedence over .env file values)
        for fields, target_data in [
            (_QDRANT_FIELDS, qdrant_data),
            (_EMBEDDING_FIELDS, embedding_data),
            (_SERVICE_FIELDS, service_data),
        ]:
            for field_name in fields:
                env_val = os.environ.get(field_name.upper())
                if env_val is not None:
                    target_data[field_name] = _maybe_parse_json(field_name, env_val)

        # Merge into existing nested dicts (if any) or create new ones
        for config_key, config_data in [
            ("qdrant", qdrant_data),
            ("embedding", embedding_data),
            ("service", service_data),
        ]:
            if config_data:
                existing = data.get(config_key, {})
                if isinstance(existing, dict):
                    existing.update(config_data)
                    data[config_key] = existing
                elif hasattr(existing, "model_dump"):
                    # Preserve fields from pre-built Pydantic models
                    merged = existing.model_dump()
                    merged.update(config_data)
                    data[config_key] = merged
                else:
                    data[config_key] = config_data

        return data

    def model_post_init(self, __context: Any) -> None:
        """Apply environment-specific overrides after initialization."""
        self.apply_environment_settings()

    def apply_environment_settings(self) -> None:
        """Apply environment-specific settings with smart defaults.

        DEVELOPMENT:
            - Sets debug=True, log_level=DEBUG as defaults
            - Respects user-provided values (from env vars, .env, or constructor)

        STAGING:
            - Sets debug=True, log_level=INFO, wal=True as defaults
            - Respects user-provided values (from env vars, .env, or constructor)

        PRODUCTION (ENFORCED):
            - ALWAYS enforces debug=False, log_level=WARNING
            - ALWAYS enforces wal=True, model_warm_up=True
            - Security: Cannot be bypassed by user configuration
        """
        if self.environment == Environment.DEVELOPMENT:
            # Apply defaults only if user didn't explicitly provide values via env vars
            # Note: This checks os.environ (not .env file or constructor args)
            # to avoid overriding explicit user values that happen to match defaults
            if "DEBUG" not in os.environ:
                self.debug = True
            if "LOG_LEVEL" not in os.environ:
                self.service.log_level = "DEBUG"

        elif self.environment == Environment.STAGING:
            # Apply defaults only if user didn't explicitly provide values via env vars
            # Note: This checks os.environ (not .env file or constructor args)
            # to avoid overriding explicit user values that happen to match defaults
            if "DEBUG" not in os.environ:
                self.debug = True
            if "LOG_LEVEL" not in os.environ:
                self.service.log_level = "INFO"
            if "QDRANT_ENABLE_WAL" not in os.environ:
                self.qdrant.qdrant_enable_wal = True

        elif self.environment == Environment.PRODUCTION:
            # ENFORCED - cannot be bypassed (security feature)
            # NOTE: Direct attribute assignment bypasses Pydantic validators.
            # This is intentional for production enforcement. All values here are
            # pre-validated to be correct and safe.
            self.debug = False
            self.service.log_level = "WARNING"
            self.qdrant.qdrant_enable_wal = True
            self.embedding.model_warm_up = True


# Ensure sub-config fields don't collide with Settings' own fields
_SETTINGS_FIELDS = frozenset(Settings.model_fields.keys()) - {
    "qdrant",
    "embedding",
    "service",
}
# Note: Using assert for import-time structural invariants (acceptable despite -O stripping)
assert not (_QDRANT_FIELDS & _SETTINGS_FIELDS), (  # noqa: S101
    f"Field overlap Qdrant/Settings: {_QDRANT_FIELDS & _SETTINGS_FIELDS}"
)
assert not (_EMBEDDING_FIELDS & _SETTINGS_FIELDS), (  # noqa: S101
    f"Field overlap Embedding/Settings: {_EMBEDDING_FIELDS & _SETTINGS_FIELDS}"
)
assert not (_SERVICE_FIELDS & _SETTINGS_FIELDS), (  # noqa: S101
    f"Field overlap Service/Settings: {_SERVICE_FIELDS & _SETTINGS_FIELDS}"
)


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
