"""One work as handed to enrichment: its title, type, year, season and provider pages."""

from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field
from work_matching.provider_page_key import (
    REFERENCE_ORDER,
    get_page_key,
    get_page_provider,
)

WorkType = Literal["TV", "MOVIE", "OVA", "ONA", "SPECIAL", "UNKNOWN"]
Season = Literal["WINTER", "SPRING", "SUMMER", "FALL"]


def is_year_matching(*years: int | None) -> bool:
    """Report whether the years agree, a missing year agreeing with any.

    Examples:
        >>> is_year_matching(2020, None, 2020)
        True
        >>> is_year_matching(2020, 2021)
        False
    """
    return len({year for year in years if year is not None}) <= 1


def is_type_matching(*work_types: WorkType | None) -> bool:
    """Report whether the work types agree, ``UNKNOWN`` or none agreeing with any.

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
    """Read the seed's ``UNDEFINED`` season as no season.

    Examples:
        >>> _read_known_season("UNDEFINED") is None
        True
        >>> _read_known_season("FALL")
        'FALL'
    """
    return None if seed_season == "UNDEFINED" else seed_season


class WorkEntry(BaseModel):
    """One work and the provider pages that belong to it.

    Matching reads only these fields; the seed's other fields (episodes,
    pictures, tags) say nothing about which pages belong together.
    """

    model_config = ConfigDict(frozen=True)

    title: str
    type: WorkType
    year: int | None = None
    season: Annotated[Season | None, BeforeValidator(_read_known_season)] = None
    # Provider page key -> the address as the source wrote it, in source order.
    pages: dict[str, str] = Field(min_length=1)

    @property
    def reference_page(self) -> str:
        """The key of the page that names this entry across releases.

        An entry's position in the seed file changes every release, so an entry
        is named by its most trusted provider page instead.
        """
        page_key_by_provider = {
            get_page_provider(key): key for key in reversed(self.pages)
        }
        return next(
            (
                page_key_by_provider[provider]
                for provider in REFERENCE_ORDER
                if provider in page_key_by_provider
            ),
            next(iter(self.pages)),
        )

    def has_page_of_same_provider(self, page_key: str) -> bool:
        """Report whether the entry holds a page of the provider this page key belongs to.

        A page joining an entry that already holds a page of its provider would
        make a doubled provider, which needs a person to decide. A key of no
        known provider has no provider to share, so it never counts.

        Args:
            page_key: The key of the page that would join the entry.
        """
        provider = get_page_provider(page_key)
        return provider is not None and any(
            get_page_provider(key) == provider for key in self.pages
        )

    @classmethod
    def from_seed_entry(cls, entry: dict[str, Any]) -> Self:
        """Build a work entry from one entry of an anime-offline-database file.

        Args:
            entry: One item of the seed file's ``data`` list.

        Returns:
            The entry's title, type, year, season and pages keyed by page.
        """
        anime_season = entry.get("animeSeason") or {}
        pages: dict[str, str] = {}
        for address in entry["sources"]:
            pages.setdefault(get_page_key(address), address)
        return cls(
            title=entry["title"],
            type=entry["type"],
            year=anime_season.get("year"),
            season=anime_season.get("season"),
            pages=pages,
        )
