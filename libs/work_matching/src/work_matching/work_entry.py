"""Model a work entry with its title, type, year, season, and provider pages."""

from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field
from work_matching.provider_page_key import (
    REFERENCE_ORDER,
    read_page_provider,
    resolve_page_key,
)

WorkType = Literal["TV", "MOVIE", "OVA", "ONA", "SPECIAL", "UNKNOWN"]
Season = Literal["WINTER", "SPRING", "SUMMER", "FALL"]


def is_year_matching(*years: int | None) -> bool:
    """Report whether the release years agree, with a missing year matching any year.

    Args:
        *years: Release years to compare, where None indicates an unknown year.

    Returns:
        True when all known years are equal or at most one known year is given.

    Examples:
        >>> is_year_matching(2020, None, 2020)
        True
        >>> is_year_matching(2020, 2021)
        False
    """
    return len({year for year in years if year is not None}) <= 1


def is_type_matching(*work_types: WorkType | None) -> bool:
    """Report whether the work types agree, with ``UNKNOWN`` or None matching any type.

    Args:
        *work_types: Work types to compare.

    Returns:
        True when all known non-``UNKNOWN`` work types are equal.

    Examples:
        >>> is_type_matching("TV", "UNKNOWN", None)
        True
        >>> is_type_matching("TV", "MOVIE")
        False
    """
    return (
        len(
            {
                work_type
                for work_type in work_types
                if work_type not in (None, "UNKNOWN")
            }
        )
        <= 1
    )


def _read_known_season(seed_season: object) -> object:
    """Convert the seed's ``UNDEFINED`` season marker to None.

    Args:
        seed_season: Raw season value from a seed entry or caller.

    Returns:
        None when ``seed_season`` is ``"UNDEFINED"``, or ``seed_season`` unchanged.

    Examples:
        >>> _read_known_season("UNDEFINED") is None
        True
        >>> _read_known_season("FALL")
        'FALL'
    """
    return None if seed_season == "UNDEFINED" else seed_season


class WorkEntry(BaseModel):
    """Represent one work and the provider pages that belong to it.

    Work matching inspects only these fields because other seed fields such as
    episodes, pictures, and tags do not determine page identity.
    """

    model_config = ConfigDict(frozen=True)

    title: str
    type: WorkType
    year: int | None = None
    season: Annotated[Season | None, BeforeValidator(_read_known_season)] = None
    # Provider page key -> the address as the source wrote it, in source order.
    addresses_by_page_key: dict[str, str] = Field(min_length=1)

    @property
    def reference_page_key(self) -> str:
        """Return the key of the page that names this entry across releases.

        An entry's position in the seed file changes every release, so an entry
        is named by its most trusted provider page instead.

        Returns:
            The page key of the highest-trusted provider present in the entry,
            or the first page key when no provider is in ``REFERENCE_ORDER``.
        """
        page_key_by_provider = {
            read_page_provider(page_key): page_key
            for page_key in reversed(self.addresses_by_page_key)
        }
        return next(
            (
                page_key_by_provider[provider]
                for provider in REFERENCE_ORDER
                if provider in page_key_by_provider
            ),
            next(iter(self.addresses_by_page_key)),
        )

    @classmethod
    def read_seed_entry(cls, seed_entry: dict[str, Any]) -> Self:
        """Read a work entry from one entry of an anime-offline-database file.

        Args:
            seed_entry: One item of the seed file's ``data`` list.

        Returns:
            The work entry with its title, type, year, season, and page
            addresses keyed by page key.
        """
        anime_season = seed_entry.get("animeSeason") or {}
        addresses_by_page_key: dict[str, str] = {}
        for address in seed_entry["sources"]:
            addresses_by_page_key.setdefault(resolve_page_key(address), address)
        return cls(
            title=seed_entry["title"],
            type=seed_entry["type"],
            year=anime_season.get("year"),
            season=anime_season.get("season"),
            addresses_by_page_key=addresses_by_page_key,
        )

    def has_page_of_same_provider(self, page_key: str) -> bool:
        """Report whether the entry holds a page of the same provider as a page key.

        A page joining an entry that already holds a page of that provider
        creates a doubled provider requiring human review.

        Args:
            page_key: The key of the page that would join the entry.

        Returns:
            True when one of the entry's pages shares the key's provider, or
            False when the key has no known provider or no matching page.
        """
        provider = read_page_provider(page_key)
        return provider is not None and any(
            read_page_provider(existing_key) == provider
            for existing_key in self.addresses_by_page_key
        )
