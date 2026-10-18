"""Bidirectional links between candidate anime provider pages.

Cross-provider links are evaluated to determine whether candidate pages belong
to the same work (such as deciding doubled providers in an entry, or checking
whether candidate entries represent a split work).

Key rules from the work matching glossary:
- Bidirectional: A link between Page A and Page B ties them together symmetrically,
  whichever page holds the link.
- Candidate scope: Only links between pages within the candidate set count;
  external links (such as sequels or spin-offs) are ignored.
- Renamed pages: When a provider renames a page, its old address resolves to
  its current key.
- Found pages: A found page (such as AnimeSchedule found by MAL ID) is credited
  to its lookup page; its links count as links of that lookup page.
- Multi-work pages: A page naming multiple pages from the same provider (e.g. an
  anthology) does not prove identity for any single work. Its links are set aside.
"""

from collections import defaultdict
from collections.abc import Iterable
from itertools import chain

from work_matching.provider_page import ProviderPage
from work_matching.provider_page_key import has_repeated_provider, provider_of_key


class ProviderPageLinks:
    """Undirected graph of links between candidate pages of a work or entry.

    Renamed pages resolve to their current key, and found pages fold into their
    lookup page. Multi-work pages (pages referencing multiple entries of one
    provider) and same-provider links are excluded from matching links.
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
        self._renamed_keys = {
            page.key: page.lands_on for page in fetched_pages if page.lands_on
        }
        self._lookup_pages = {
            page.key: self.same_page(page.found_by)
            for page in fetched_pages
            if page.found_by
        }
        candidate_keys = set(
            map(
                self._current_key,
                chain(page_keys, (page.key for page in fetched_pages)),
            )
        )
        self._linked: defaultdict[str, set[str]] = defaultdict(set)
        self._linked_by_multi_work: defaultdict[str, set[str]] = defaultdict(set)
        for page in fetched_pages:
            self._add_links(page, candidate_keys)

    def same_page(self, key: str) -> str:
        """Return the current key a renamed page resolves to, or the key unchanged."""
        return self._renamed_keys.get(key, key)

    def linked_with(self, key: str) -> frozenset[str]:
        """Return candidate pages linked with a page, in either direction."""
        return frozenset(self._linked.get(self._current_key(key), ()))

    def linked_by_multi_work_pages(self, key: str) -> frozenset[str]:
        """Return multi-work pages that linked to this page."""
        return frozenset(self._linked_by_multi_work.get(self._current_key(key), ()))

    def _add_links(self, page: ProviderPage, candidate_keys: set[str]) -> None:
        """Record one page's links within the candidate keys.

        A multi-work page's links are set apart rather than recorded.
        """
        source_key = self._current_key(page.key)
        target_keys = {self._current_key(link) for link in page.links} - {source_key}
        referenced_keys = target_keys | {source_key} if page.found_by else target_keys
        if has_repeated_provider(referenced_keys):
            for target_key in target_keys & candidate_keys:
                self._linked_by_multi_work[target_key].add(source_key)
            return

        source_provider = provider_of_key(source_key)
        for target_key in target_keys & candidate_keys:
            if provider_of_key(target_key) != source_provider:
                self._linked[source_key].add(target_key)
                self._linked[target_key].add(source_key)

    def _current_key(self, key: str) -> str:
        """Return the current canonical key for a page key.

        A found page's links are credited to its lookup page, and a renamed
        page resolves to its current key.
        """
        return self._lookup_pages.get(key) or self.same_page(key)
