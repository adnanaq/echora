"""Base classes for the canonical anime, character and episode models."""

import logging
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)


class StrictFieldsModel(BaseModel):
    """Model that rejects any field it does not declare.

    A misspelled or leftover key in the data fails validation instead of being
    dropped without notice.
    """

    model_config = ConfigDict(extra="forbid")


class ProviderMappedEnum(StrEnum):
    """Set of canonical values that also accepts the words providers use for them.

    MAL sends "Currently Airing" and AniList "RELEASING" for the same airing
    status; both become ``AnimeStatus.ONGOING``. Each subclass lists these
    provider words in ``provider_value_map`` and names the value to use for an
    unknown word (``UNKNOWN`` or ``OTHER``) in ``_missing_``.
    """

    @classmethod
    def provider_value_map(cls) -> dict[str, Self]:
        """Return each provider word, in lower case, with the canonical value it means."""
        return {}

    @classmethod
    def from_provider_value(cls, value: object) -> Self | None:
        """Return the canonical value for a provider word, or None when it is unknown.

        A canonical value itself ("ONGOING") is returned as is. Unlike calling
        the enum, an unknown word gives None and logs nothing, so callers can
        ask whether a word belongs to this set.
        """
        if not isinstance(value, str):
            return None
        for canonical in cls:
            if canonical.value == value:
                return canonical
        return cls.provider_value_map().get(value.strip().lower())

    @classmethod
    def provider_value_or_fallback(cls, value: object, fallback: Self) -> Self:
        """Return the canonical value for a provider word, or ``fallback``.

        A non-empty word with no mapping is logged, so new provider words show
        up instead of silently becoming the fallback.
        """
        canonical = cls.from_provider_value(value)
        if canonical is not None:
            return canonical
        if isinstance(value, str) and value.strip():
            logger.warning(
                f"{cls.__name__} has no canonical value for {value!r}; using {fallback}"
            )
        return fallback
