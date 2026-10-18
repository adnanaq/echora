"""One provider page's details, as work matching compares them.

The details come from the canonical record an enrichment helper returns for the
page, fetched with characters and episodes switched off. Reading that record,
rather than fetching the page a second way, lets seed cleaning and enrichment
share one fetch and one cache.
"""

from collections.abc import Iterable, Iterator
from typing import Any, Self

from pydantic import BaseModel, ConfigDict
from work_matching.provider_page_key import get_page_key, get_page_provider


class ProviderPage(BaseModel):
    """What one provider says about one of its pages."""

    # Unknown keys are rejected: a stored page written under an old field name
    # would otherwise load with that field silently empty.
    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str
    address: str
    # Every title the provider gives, its main title first, each once.
    titles: tuple[str, ...] = ()
    type: str | None = None
    year: int | None = None
    season: str | None = None
    # Keys of other providers' pages this page links to, sorted. Shared
    # websites are left out: two different works can share an official site.
    links: tuple[str, ...] = ()
    # The key the address was renamed to. A record lists other providers' pages
    # and its own page, so a second page of its own provider is the page's new
    # address; on the fetched pages this is so only for Anime-Planet renames.
    renamed_to: str | None = None
    # The keys of the lookup pages this found page was found through, sorted:
    # AnimeSchedule is found by MAL id. One found page can be reached through
    # pages of several entries, such as both entries of a split work, so every
    # lookup page is kept. Empty for a page the seed lists itself.
    found_by: tuple[str, ...] = ()
    is_dead: bool = False

    @classmethod
    def from_canonical_record(
        cls, address: str, record: dict[str, Any], found_by: str | None = None
    ) -> Self:
        """Read a page's details from the canonical record fetched for it.

        Args:
            address: The page address that was fetched.
            record: The canonical anime record the provider's helper returned.
            found_by: The address of the lookup page, when the page was found
                through another provider's page.

        Returns:
            The page's titles, type, year, season, links to other providers'
            pages, and the key the page was renamed to, if it was renamed.
        """
        key = get_page_key(address)
        own_provider = get_page_provider(key)
        linked_keys = {
            get_page_key(linked_address)
            for linked_address in _read_linked_addresses(record)
        }
        renamed_keys = sorted(
            linked_key
            for linked_key in linked_keys - {key}
            if own_provider and get_page_provider(linked_key) == own_provider
        )
        return cls(
            key=key,
            address=address,
            titles=tuple(dict.fromkeys(_read_record_titles(record))),
            type=record.get("type"),
            year=record.get("year"),
            season=record.get("season"),
            links=tuple(
                sorted(
                    linked_key
                    for linked_key in linked_keys
                    if get_page_provider(linked_key) not in (None, own_provider)
                )
            ),
            renamed_to=renamed_keys[0] if renamed_keys else None,
            found_by=_read_lookup_keys(found_by),
        )

    @classmethod
    def from_dead_address(cls, address: str, found_by: str | None = None) -> Self:
        """Record a dead page: one the provider no longer serves."""
        return cls(
            key=get_page_key(address),
            address=address,
            is_dead=True,
            found_by=_read_lookup_keys(found_by),
        )


def index_pages_by_current_key(
    pages: Iterable[ProviderPage],
) -> dict[str, ProviderPage]:
    """Index pages by the current key each resolves to.

    A renamed page resolves to the key it was renamed to, unless the page was also
    fetched at that key.

    Args:
        pages: Fetched pages.

    Returns:
        Each page by its current key.
    """
    pages_by_key: dict[str, ProviderPage] = {}
    for page in pages:
        key = page.renamed_to or page.key
        if key not in pages_by_key or page.key == key:
            pages_by_key[key] = page
    return pages_by_key


def _read_lookup_keys(found_by: str | None) -> tuple[str, ...]:
    """Return the lookup page's key as ``found_by`` holds it; empty without one."""
    return (get_page_key(found_by),) if found_by else ()


def _read_record_titles(record: dict[str, Any]) -> Iterator[str]:
    """Yield a record's titles, its main title first, skipping empty ones."""
    for title_field in ("title", "title_english", "title_japanese"):
        if title := record.get(title_field):
            yield title
    yield from (title for title in (record.get("titles") or {}).values() if title)


def _read_linked_addresses(record: dict[str, Any]) -> Iterator[str]:
    """Yield every page address a record lists in its two link fields.

    The fields are ``sources`` and ``external_sources``.
    """
    yield from (address for address in record.get("sources") or [] if address)
    for link in record.get("external_sources") or []:
        if address := link.get("source"):
            yield address
