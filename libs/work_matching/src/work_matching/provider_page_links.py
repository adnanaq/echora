"""Which provider pages of a group link to which, counted in both directions.

A link from page A to page B ties the two together whichever page holds it, so
a link is kept as an undirected edge. Only links between pages of the group
count: a link to a page outside it says nothing about the group's pages.

A page found through another provider's page, as AnimeSchedule is found by MAL
id, is that provider's record of the same work. Its links count as links of the
page it was found through, so the two stay one page here. This is followed one
step: the page a page was found through is a page of the provider looked up by,
never a found page itself.
"""

from collections import defaultdict
from collections.abc import Iterable
from itertools import chain

from work_matching.provider_page import ProviderPage
from work_matching.provider_page_key import has_repeated_provider


class ProviderPageLinks:
    """The links between a group's provider pages, renamed and found pages folded.

    A page that names two or more pages of one provider, such as an anthology
    page covering several works, is no evidence that any of them is its work.
    Its links are kept apart and do not count. A found page names the page it
    was found through as well as the pages it links.
    """

    def __init__(
        self, pages: Iterable[ProviderPage], page_keys: Iterable[str] = ()
    ) -> None:
        """Collect the links between the group's pages.

        Args:
            pages: Details of the fetched pages; their links are the edges.
            page_keys: Keys of further pages in the group, such as the entry's
                pages of providers that are not fetched.
        """
        pages = tuple(pages)
        self._lands_on = {page.key: page.lands_on for page in pages if page.lands_on}
        self._found_through = {
            page.key: self.same_page(page.found_by) for page in pages if page.found_by
        }
        group = set(map(self._node, chain(page_keys, (page.key for page in pages))))
        self._linked: defaultdict[str, set[str]] = defaultdict(set)
        self._linked_by_multi_work: defaultdict[str, set[str]] = defaultdict(set)
        for page in pages:
            source = self._node(page.key)
            targets = {self._node(link) for link in page.links} - {source}
            named = targets | {source} if page.found_by else targets
            if has_repeated_provider(named):
                for target in targets & group:
                    self._linked_by_multi_work[target].add(source)
                continue
            for target in targets & group:
                self._linked[source].add(target)
                self._linked[target].add(source)

    def same_page(self, key: str) -> str:
        """Return the key a page answers to now: its new key if it was renamed."""
        return self._lands_on.get(key, key)

    def linked_with(self, key: str) -> frozenset[str]:
        """Return the group's pages linked with a page, in either direction."""
        return frozenset(self._linked.get(self._node(key), ()))

    def linked_by_multi_work_pages(self, key: str) -> frozenset[str]:
        """Return the pages linking a page whose links were set aside."""
        return frozenset(self._linked_by_multi_work.get(self._node(key), ()))

    def _node(self, key: str) -> str:
        """Return the key a page's links are kept under.

        A found page's links are kept under the page it was found through, and
        a renamed page's under its new key.
        """
        return self._found_through.get(key) or self.same_page(key)
