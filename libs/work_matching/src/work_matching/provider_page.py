"""Model a provider page's details and cross-provider links for work matching.

The details are extracted from the canonical record returned by an enrichment
helper with characters and episodes disabled. Reading that canonical record
allows seed cleaning and the enrichment pipeline to share a single fetch and
HTTP cache.
"""

from collections.abc import Iterable, Iterator
from datetime import datetime
from typing import Any, Self

from pydantic import BaseModel, ConfigDict
from work_matching.provider_page_key import read_page_provider, resolve_page_key


class ProviderPage(BaseModel):
    """Represent what one provider reports about one of its pages."""

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
    # The number of episodes the provider lists, or None when it gives none.
    # Episode counts and air dates never decide a match: they show a reviewer
    # a provider dividing a work differently (one season against the whole run),
    # and they stop an automatic change.
    episode_count: int | None = None
    aired_from: datetime | None = None
    aired_to: datetime | None = None
    # Keys of other providers' pages this page links to, sorted. Shared
    # websites are left out: two different works can share an official site.
    linked_page_keys: tuple[str, ...] = ()
    # The key the address was renamed to. A record lists other providers' pages
    # and its own page, so a second page of its own provider is the page's
    # current address: an Anime-Planet rename, or a Kitsu page fetched by slug,
    # which lists itself by number.
    renamed_to: str | None = None
    # The keys of the pages this searched page was searched by, sorted:
    # AnimeSchedule is searched by MAL id. One searched page can be reached
    # through pages of several entries, such as both entries of a split work, so
    # every one is kept. Empty for a page the seed lists itself.
    searched_by: tuple[str, ...] = ()
    is_dead: bool = False

    @classmethod
    def read_canonical_record(
        cls,
        address: str,
        record: dict[str, Any],
        searched_by_address: str | None = None,
    ) -> Self:
        """Read a page's details from the canonical record fetched for it.

        Args:
            address: The page address that was fetched.
            record: The canonical anime record the provider's helper returned.
            searched_by_address: The address of the page the search used, when
                the page was found by searching its provider with another
                page's identifier.

        Returns:
            The provider page with its titles, type, year, season, episode
            count, air dates, cross-provider links, and any renamed target key.
        """
        page_key = resolve_page_key(address)
        aired_dates = record.get("aired_dates") or {}
        own_provider = read_page_provider(page_key)
        linked_page_keys = {
            resolve_page_key(linked_address)
            for linked_address in _read_linked_addresses(record)
        }
        renamed_page_keys = sorted(
            linked_page_key
            for linked_page_key in linked_page_keys - {page_key}
            if own_provider and read_page_provider(linked_page_key) == own_provider
        )
        return cls(
            key=page_key,
            address=address,
            titles=tuple(dict.fromkeys(_read_record_titles(record))),
            type=record.get("type"),
            year=record.get("year"),
            season=record.get("season"),
            # The canonical record holds 0 when the provider gives no count,
            # such as for an anime still airing, so 0 is read as unknown.
            episode_count=record.get("episode_count") or None,
            aired_from=aired_dates.get("aired_from"),
            aired_to=aired_dates.get("aired_to"),
            linked_page_keys=tuple(
                sorted(
                    linked_page_key
                    for linked_page_key in linked_page_keys
                    if read_page_provider(linked_page_key) not in (None, own_provider)
                )
            ),
            renamed_to=renamed_page_keys[0] if renamed_page_keys else None,
            searched_by=_read_searched_by_page_keys(searched_by_address),
        )

    @classmethod
    def build_dead_page(
        cls, address: str, searched_by_address: str | None = None
    ) -> Self:
        """Build a dead provider page that the provider no longer serves.

        Args:
            address: The page address that returned not found.
            searched_by_address: The address of the page used to search for
                this page, if any.

        Returns:
            A dead ``ProviderPage`` instance with ``is_dead=True``.
        """
        return cls(
            key=resolve_page_key(address),
            address=address,
            is_dead=True,
            searched_by=_read_searched_by_page_keys(searched_by_address),
        )

    @property
    def current_key(self) -> str:
        """Return the key the page is found under now.

        A page reached through an old or alternate spelling of its address
        resolves to the page at its current key. An example is a Kitsu page
        linked by slug while the seed lists the page by number.

        Returns:
            The key the page was renamed to, or the page's own key.
        """
        return self.renamed_to or self.key


def index_pages_by_current_page_key(
    pages: Iterable[ProviderPage],
) -> dict[str, ProviderPage]:
    """Index pages by the current page key each resolves to.

    A renamed page resolves to the key it was renamed to unless the page was
    also fetched directly at that current key.

    Args:
        pages: Fetched provider pages to index.

    Returns:
        A dictionary mapping each current page key to its ``ProviderPage``.
    """
    pages_by_current_page_key: dict[str, ProviderPage] = {}
    for page in pages:
        current_page_key = page.current_key
        if (
            current_page_key not in pages_by_current_page_key
            or page.key == current_page_key
        ):
            pages_by_current_page_key[current_page_key] = page
    return pages_by_current_page_key


def _read_searched_by_page_keys(searched_by_address: str | None) -> tuple[str, ...]:
    """Resolve the searched-by page address into a singleton key tuple or empty tuple.

    Args:
        searched_by_address: The address of the page used for the search, or None.

    Returns:
        A one-element tuple holding the resolved page key, or an empty tuple.
    """
    return (resolve_page_key(searched_by_address),) if searched_by_address else ()


def _read_record_titles(record: dict[str, Any]) -> Iterator[str]:
    """Yield a canonical record's non-empty titles with the main title first.

    Args:
        record: The canonical anime record dictionary.

    Yields:
        Non-empty title strings starting with ``title``, ``title_english``,
        ``title_japanese``, and remaining entries in ``titles``.
    """
    for title_field in ("title", "title_english", "title_japanese"):
        if title := record.get(title_field):
            yield title
    yield from (title for title in (record.get("titles") or {}).values() if title)


def _read_linked_addresses(record: dict[str, Any]) -> Iterator[str]:
    """Yield every non-empty page address from ``sources`` and ``external_sources``.

    Args:
        record: The canonical anime record dictionary.

    Yields:
        Non-empty URL strings from ``sources`` followed by ``external_sources``.
    """
    yield from (address for address in record.get("sources") or [] if address)
    for external_source in record.get("external_sources") or []:
        if address := external_source.get("source"):
            yield address
