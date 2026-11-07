"""Build bidirectional links between candidate anime provider pages.

Cross-provider links are evaluated to determine whether candidate pages belong
to the same work (such as deciding doubled providers in an entry, or checking
whether candidate entries represent a split work).

Each rule reads ``<rule>: <behavior>``:

- Bidirectional: A link between Page A and Page B ties them together symmetrically,
  whichever page holds the link.
- Candidate scope: Only links between pages within the candidate set count;
  external links (such as sequels or spin-offs) are ignored.
- Renamed pages: When a provider renames a page, the old address resolves to
  the current key.
- Searched pages: A searched page (such as AnimeSchedule searched by MAL ID) is
  credited to each page that searched for the page: the links count as links of
  every such page, except the link back to that page, which only shows how the
  searched page was found.
- Multi-work pages: A page naming multiple pages from the same provider (such
  as an anthology) does not prove identity for any single work. Multi-work page
  links are set aside.
- Several-works pages: A page linked with two or more MAL pages, in either
  direction, covers several works, since a work is what MAL lists (such as
  AniDB's single page for three films MAL lists apart). Several-works page
  links are set aside too, tying none of those works together.
"""

from collections import defaultdict
from collections.abc import Iterable
from itertools import chain

from work_matching.provider_page import ProviderPage
from work_matching.provider_page_key import has_repeated_provider, read_page_provider

# A work is what MAL lists, so a page linked with two of its pages covers
# several works.
_MAL_PROVIDER = "mal"


class ProviderPageLinks:
    """Undirected graph of links between candidate pages of a work or entry.

    Renamed pages resolve to the current key, and searched pages fold into
    the pages that searched for them. Multi-work pages (pages referencing
    multiple entries of one provider) and same-provider links are excluded from
    matching links.
    """

    def __init__(
        self, pages: Iterable[ProviderPage], page_keys: Iterable[str] = ()
    ) -> None:
        """Collect links between the candidate pages of an entry or split work.

        Args:
            pages: Details of the fetched candidate pages whose links form the edges.
            page_keys: Keys of additional candidate pages in the entry or work,
                including pages from providers that were not fetched.
        """
        fetched_pages = tuple(pages)
        self._current_page_key_by_renamed_page_key = {
            page.key: page.renamed_to for page in fetched_pages if page.renamed_to
        }
        self._searched_by_page_keys_by_searched_page_key = {
            page.key: tuple(map(self.get_current_page_key, page.searched_by))
            for page in fetched_pages
            if page.searched_by
        }
        candidate_page_keys = set(
            chain.from_iterable(
                map(
                    self._get_current_page_keys,
                    chain(page_keys, (page.key for page in fetched_pages)),
                )
            )
        )
        self._linked_page_keys_by_page_key: defaultdict[str, set[str]] = defaultdict(
            set
        )
        self._multi_work_link_page_keys_by_page_key: defaultdict[str, set[str]] = (
            defaultdict(set)
        )
        for page in fetched_pages:
            for source_page_key in self._get_current_page_keys(page.key):
                self._add_links(page, source_page_key, candidate_page_keys)
        self._remove_multi_mal_page_links()

    def get_current_page_key(self, page_key: str) -> str:
        """Return the current key a renamed page resolves to, or the key unchanged.

        Args:
            page_key: The page key to resolve.

        Returns:
            The current page key after following any provider rename.
        """
        return self._current_page_key_by_renamed_page_key.get(page_key, page_key)

    def get_linked_page_keys(self, page_key: str) -> frozenset[str]:
        """Return the keys of candidate pages linked with a page, in either direction.

        A searched page's key gives the pages linked with any page that searched
        for the searched page.

        Args:
            page_key: The page key whose linked candidate pages are requested.

        Returns:
            The keys of candidate pages linked with ``page_key``.
        """
        return frozenset(
            chain.from_iterable(
                self._linked_page_keys_by_page_key.get(current_page_key, ())
                for current_page_key in self._get_current_page_keys(page_key)
            )
        )

    def get_multi_work_link_page_keys(self, page_key: str) -> frozenset[str]:
        """Return the keys of multi-work pages that link to this page.

        Args:
            page_key: The page key whose inbound multi-work links are requested.

        Returns:
            The keys of multi-work pages that reference ``page_key``.
        """
        return frozenset(
            chain.from_iterable(
                self._multi_work_link_page_keys_by_page_key.get(current_page_key, ())
                for current_page_key in self._get_current_page_keys(page_key)
            )
        )

    def _add_links(
        self, page: ProviderPage, source_page_key: str, candidate_page_keys: set[str]
    ) -> None:
        """Record one page's links, as links of ``source_page_key``, within the candidates.

        ``source_page_key`` is the page's current key, or for a searched page
        one of the pages that searched for the page. A multi-work page's links
        are recorded in ``_multi_work_link_page_keys_by_page_key`` rather than
        added as bidirectional candidate links.

        Args:
            page: The fetched provider page whose links are inspected.
            source_page_key: The current page key credited with the page's links.
            candidate_page_keys: All candidate page keys in the entry or group.
        """
        target_page_keys = {
            self.get_current_page_key(linked_page_key)
            for linked_page_key in page.linked_page_keys
        } - {source_page_key}
        referenced_page_keys = (
            target_page_keys | {source_page_key}
            if page.searched_by
            else target_page_keys
        )
        if has_repeated_provider(referenced_page_keys):
            for target_page_key in target_page_keys & candidate_page_keys:
                self._multi_work_link_page_keys_by_page_key[target_page_key].add(
                    source_page_key
                )
            return

        source_provider = read_page_provider(source_page_key)
        for target_page_key in target_page_keys & candidate_page_keys:
            if read_page_provider(target_page_key) != source_provider:
                self._linked_page_keys_by_page_key[source_page_key].add(target_page_key)
                self._linked_page_keys_by_page_key[target_page_key].add(source_page_key)

    def _remove_multi_mal_page_links(self) -> None:
        """Remove every link of a page linked with two or more MAL pages.

        Such a page covers several works, so a link through that page would tie
        separate works together.
        """
        multi_mal_page_keys = [
            page_key
            for page_key, linked_page_keys in self._linked_page_keys_by_page_key.items()
            if sum(
                read_page_provider(linked_page_key) == _MAL_PROVIDER
                for linked_page_key in linked_page_keys
            )
            > 1
        ]
        for page_key in multi_mal_page_keys:
            for linked_page_key in self._linked_page_keys_by_page_key.pop(page_key):
                self._linked_page_keys_by_page_key[linked_page_key].discard(page_key)

    def _get_current_page_keys(self, page_key: str) -> tuple[str, ...]:
        """Return the current keys a page key stands for.

        A searched page stands for the pages that searched for the page, and a
        renamed page resolves to the current key.

        Args:
            page_key: The page key to expand into current keys.

        Returns:
            The current keys represented by ``page_key``.
        """
        return self._searched_by_page_keys_by_searched_page_key.get(page_key) or (
            self.get_current_page_key(page_key),
        )
