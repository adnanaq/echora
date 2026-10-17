"""One work as handed to enrichment: its title, type, year, season and provider pages."""

from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from work_matching.provider_page_key import (
    REFERENCE_ORDER,
    provider_of_key,
    provider_page_key,
)

WorkType = Literal["TV", "MOVIE", "OVA", "ONA", "SPECIAL", "UNKNOWN"]
Season = Literal["WINTER", "SPRING", "SUMMER", "FALL"]


def _known_season(value: object) -> object:
    """Read the seed's ``UNDEFINED`` season as no season.

    Examples:
        >>> _known_season("UNDEFINED") is None
        True
        >>> _known_season("FALL")
        'FALL'
    """
    return None if value == "UNDEFINED" else value


class WorkEntry(BaseModel):
    """One work and the provider pages that belong to it.

    Matching reads only these fields; the seed's other fields (episodes,
    pictures, tags) say nothing about which pages belong together.
    """

    model_config = ConfigDict(frozen=True)

    title: str
    type: WorkType
    year: int | None = None
    season: Annotated[Season | None, BeforeValidator(_known_season)] = None
    # Provider page key -> the address as the source wrote it, in source order.
    pages: dict[str, str] = Field(min_length=1)

    @property
    def reference_page(self) -> str:
        """The key of the page that names this entry across releases.

        An entry's position in the seed file changes every release, so an entry
        is named by its most trusted provider page instead.
        """
        by_provider = {provider_of_key(key): key for key in reversed(self.pages)}
        return next(
            (
                by_provider[provider]
                for provider in REFERENCE_ORDER
                if provider in by_provider
            ),
            next(iter(self.pages)),
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
            pages.setdefault(provider_page_key(address), address)
        return cls(
            title=entry["title"],
            type=entry["type"],
            year=anime_season.get("year"),
            season=anime_season.get("season"),
            pages=pages,
        )
