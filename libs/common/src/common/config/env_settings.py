"""Base class for configuration read from environment variables."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class EnvSettings(BaseSettings):
    """Configuration whose fields are read from environment variables.

    Each field is read from the environment variable of the same name in upper
    case, then from ``.env``, then falls back to its default. List and dict
    fields take JSON, such as ``ALLOWED_ORIGINS='["https://example.com"]'``.
    Values passed in code win over both.
    """

    # validate_default=False keeps pydantic's usual rule that defaults are
    # trusted; BaseSettings would otherwise check them against values given
    # alongside, such as the default multivector names against custom vectors.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        validate_default=False,
    )
