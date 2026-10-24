"""Decide what an entry's pages from one provider are, from the links between pages.

A doubled provider is an entry holding two or more pages of one provider. Each
decision covers one doubled provider and records two things
(``DoubledProviderDecision``): what the pages are (``Outcome``) and why a
person must check it (``ReviewReason``). The rules:

| Rule | Reason for the rule |
| -- | -- |
| Links decide the outcome; titles never do. | Providers listing one work twice often vary the titles. |
| Air dates and episode counts only tell duplicate pages from part pages. | Both outcomes always go to a person, so the details only shape the suggestion. |
| Every review reason that applies is recorded, not only the first. | A reviewer sees every problem at once. |
| A move to another seed entry is checked here (``MOVE_NOT_SURE``). | The decision then says by itself whether the rules can apply it. |
"""

from collections.abc import Iterable, Mapping
from dataclasses import replace

from work_matching.comparable_title import fold_title_for_comparison
from work_matching.doubled_provider_decision import (
    DoubledProviderDecision,
    Outcome,
    ReviewReason,
    UnrelatedPage,
)
from work_matching.doubled_provider_pages import (
    find_doubled_provider_page_keys,
    get_decided_provider,
    has_several_page_ids,
)
from work_matching.known_work_entries import KnownWorkEntries
from work_matching.page_details_comparison import (
    get_seed_type,
    is_page_details_matching,
)
from work_matching.provider_page import ProviderPage, index_pages_by_current_key
from work_matching.provider_page_key import get_page_provider
from work_matching.provider_page_links import ProviderPageLinks
from work_matching.work_entry import WorkEntry, is_type_matching, is_year_matching


def decide_doubled_providers(
    entry: WorkEntry,
    pages: Iterable[ProviderPage],
    known_work_entries: KnownWorkEntries,
) -> list[DoubledProviderDecision]:
    """Decide every doubled provider of one entry.

    Args:
        entry: The work entry.
        pages: Details of the entry's fetched pages, and of searched pages
            searched by one of them (such as AnimeSchedule).
        known_work_entries: The work entries already known, to identify the
            entries an unrelated page links to.

    Returns:
        One decision per provider with two or more pages in the entry, in the
        order the entry lists them.
    """
    entry_check = _EntryCheck(entry, pages, known_work_entries)
    return [
        entry_check.decide_provider(provider, list(page_keys))
        for provider, page_keys in find_doubled_provider_page_keys(
            entry.addresses_by_page_key
        ).items()
    ]


class _EntryCheck:
    """Evaluates links and metadata for one entry's doubled provider pages."""

    def __init__(
        self,
        entry: WorkEntry,
        pages: Iterable[ProviderPage],
        known_work_entries: KnownWorkEntries,
    ) -> None:
        self._entry = entry
        fetched_pages = tuple(pages)
        self._links = ProviderPageLinks(fetched_pages, entry.addresses_by_page_key)
        self._pages_by_current_key = index_pages_by_current_key(fetched_pages)
        self._known_work_entries = known_work_entries
        self._entry_page_keys = frozenset(
            map(self._links.get_current_key, entry.addresses_by_page_key)
        )
        self._entry_providers = frozenset(map(get_page_provider, self._entry_page_keys))

    def decide_provider(
        self, provider: str, page_keys: list[str]
    ) -> DoubledProviderDecision:
        """Decide what one doubled provider's pages are, and why a person must check it.

        Args:
            provider: The provider decided, as ``get_decided_provider`` names it.
            page_keys: The provider's page keys, as the entry lists them.

        Returns:
            The decision. A decision a person must check also carries the pages
            matching the entry's title, year and type, and season.
        """
        current_page_keys = tuple(
            dict.fromkeys(map(self._links.get_current_key, page_keys))
        )
        if not has_several_page_ids(current_page_keys):
            is_current_address_listed = (
                current_page_keys[0] in self._entry.addresses_by_page_key
            )
            return DoubledProviderDecision(
                provider,
                current_page_keys,
                Outcome.SAME_PAGE,
                ()
                if is_current_address_listed
                else (ReviewReason.CURRENT_ADDRESS_MISSING,),
            )
        decision = self._sort_pages_by_links(provider, current_page_keys)
        decision = replace(decision, review_reasons=self._list_review_reasons(decision))
        if not decision.needs_review:
            return decision
        return replace(
            decision,
            title_matches=self._find_title_matches(provider, current_page_keys),
            year_type_matches=self._find_year_type_matches(current_page_keys),
            season_matches=self._find_season_matches(current_page_keys),
        )

    def _sort_pages_by_links(
        self, provider: str, page_keys: tuple[str, ...]
    ) -> DoubledProviderDecision:
        """Sort a provider's pages by what their links show, and name the outcome.

        Each page falls in one group:

        | Group | The page's links |
        | -- | -- |
        | belonging | link the entry's pages and no other work |
        | unrelated | link another work and none of the entry's pages |
        | contradicting | link the entry's pages and another work |
        | unlinked | link neither |

        Args:
            provider: The provider decided.
            page_keys: The current keys of the provider's pages.

        Returns:
            The decision with its outcome and groups, without review reasons.
        """
        linked_page_keys_by_page_key = {
            key: self._links.get_linked_page_keys(key) for key in page_keys
        }
        other_work_page_keys_by_page_key = {
            key: self._find_linked_page_keys_of_other_works(key) for key in page_keys
        }
        belonging_page_keys = {
            key: linked_page_keys_by_page_key[key]
            for key in page_keys
            if linked_page_keys_by_page_key[key]
            and not other_work_page_keys_by_page_key[key]
        }
        contradicting_page_keys = tuple(
            key
            for key in page_keys
            if linked_page_keys_by_page_key[key]
            and other_work_page_keys_by_page_key[key]
        )
        unlinked_page_keys = tuple(
            key
            for key in page_keys
            if not linked_page_keys_by_page_key[key]
            and not other_work_page_keys_by_page_key[key]
        )
        return DoubledProviderDecision(
            provider,
            page_keys,
            self._choose_outcome(
                belonging_page_keys, contradicting_page_keys + unlinked_page_keys
            ),
            belonging_page_keys=belonging_page_keys,
            unrelated_pages={
                key: self._describe_unrelated_page(
                    key, other_work_page_keys_by_page_key[key]
                )
                for key in page_keys
                if other_work_page_keys_by_page_key[key]
                and not linked_page_keys_by_page_key[key]
            },
            contradicting_page_keys=contradicting_page_keys,
            unlinked_page_keys=unlinked_page_keys,
        )

    def _choose_outcome(
        self,
        belonging_page_keys: Mapping[str, frozenset[str]],
        unplaced_page_keys: tuple[str, ...],
    ) -> Outcome:
        """Name what the pages are, from the entry's pages each belonging page links.

        Args:
            belonging_page_keys: Each belonging page's key, mapped to the keys
                of the entry's pages linked with it.
            unplaced_page_keys: Keys of the contradicting and unlinked pages,
                which the links do not place.

        Returns:
            ``UNDECIDED`` while a page is unplaced; ``PAGES_OF_OTHER_WORKS`` for
            at most one belonging page; then, by how the belonging pages' linked
            pages meet: a shared page gives ``DUPLICATE_PAGES`` or
            ``PART_PAGES``, separate sets give ``SEPARATE_WORKS``, and partly
            shared sets give ``UNDECIDED``.
        """
        if unplaced_page_keys:
            return Outcome.UNDECIDED
        if len(belonging_page_keys) < 2:
            return Outcome.PAGES_OF_OTHER_WORKS
        linked_page_key_sets = list(belonging_page_keys.values())
        if frozenset.intersection(*linked_page_key_sets):
            if self._is_each_page_whole_work(list(belonging_page_keys)):
                return Outcome.DUPLICATE_PAGES
            return Outcome.PART_PAGES
        if sum(map(len, linked_page_key_sets)) == len(
            frozenset.union(*linked_page_key_sets)
        ):
            return Outcome.SEPARATE_WORKS
        return Outcome.UNDECIDED

    def _list_review_reasons(
        self, decision: DoubledProviderDecision
    ) -> tuple[ReviewReason, ...]:
        """List every reason a person must check a decision, in ``ReviewReason`` order.

        Args:
            decision: The decision with its outcome and page groups.

        Returns:
            Each review reason that applies; empty when none does.
        """
        is_one_work = decision.outcome in (
            Outcome.DUPLICATE_PAGES,
            Outcome.PART_PAGES,
        )
        has_unplaced_pages = bool(
            decision.contradicting_page_keys or decision.unlinked_page_keys
        )
        reason_applies = {
            ReviewReason.NO_PAGE_LINKED_WITH_ENTRY: (
                decision.outcome is Outcome.PAGES_OF_OTHER_WORKS
                and not decision.belonging_page_keys
            ),
            ReviewReason.MOVE_NOT_SURE: any(
                unrelated_page.move_target_entry is None
                for unrelated_page in decision.unrelated_pages.values()
            ),
            ReviewReason.DETAILS_DIFFER: is_one_work
            and not self._is_details_matching(list(decision.belonging_page_keys)),
            ReviewReason.CONTRADICTING_LINKS: bool(decision.contradicting_page_keys),
            ReviewReason.UNLINKED_PAGES: bool(decision.unlinked_page_keys),
            ReviewReason.OVERLAPPING_LINKS: decision.outcome is Outcome.UNDECIDED
            and not has_unplaced_pages,
            ReviewReason.MULTI_WORK_LINK: any(
                map(self._links.get_linking_multi_work_page_keys, decision.page_keys)
            ),
            ReviewReason.DEAD_PAGE: any(map(self._is_dead, decision.page_keys)),
        }
        return tuple(reason for reason in ReviewReason if reason_applies.get(reason))

    def _describe_unrelated_page(
        self, key: str, linked_page_keys: frozenset[str]
    ) -> UnrelatedPage:
        """Describe the work an unrelated page belongs to, and where the page moves.

        Args:
            key: The unrelated page's current key.
            linked_page_keys: The keys of its linked pages that show another work.

        Returns:
            The linked pages, the known work entries holding them, and the seed
            entry the page moves to when the move is sure.
        """
        other_work_entries = self._find_other_work_entries(linked_page_keys)
        return UnrelatedPage(
            linked_page_keys,
            other_work_entries,
            self._find_move_target(key, other_work_entries),
        )

    def _find_move_target(
        self, key: str, other_work_entries: tuple[WorkEntry, ...]
    ) -> WorkEntry | None:
        """Return the seed entry an unrelated page moves to, when the move is sure.

        The move is sure only when all of these hold, each for its own reason:

        | Rule | Reason for the rule |
        | -- | -- |
        | The page's links name exactly one seed entry. | A link to a page outside the seed most often points to a page the provider has deleted, kept in a stale mapping. |
        | That entry holds no page of the same provider. | The move would make a new doubled provider. |
        | The page's year and type agree with that entry's. | A disagreement shows the mapping itself may be wrong. |

        The same target takes a page a review answer sends to ``other work``,
        so an answer cannot make a doubled provider either.

        Args:
            key: The unrelated page's current key.
            other_work_entries: The known work entries its links name.

        Returns:
            The seed entry to move the page to, or None when no move is sure.
        """
        if len(other_work_entries) != 1:
            return None
        [other_work_entry] = other_work_entries
        if other_work_entry.has_page_of_same_provider(key):
            return None
        # A page links another work only through its own fetched details, so it
        # has them; without them nothing confirms the move.
        page = self._pages_by_current_key.get(key)
        is_confirmed = page is not None and is_page_details_matching(
            page, other_work_entry
        )
        return other_work_entry if is_confirmed else None

    def _is_dead(self, key: str) -> bool:
        """Report whether a page was fetched and the provider no longer serves it."""
        page = self._pages_by_current_key.get(key)
        return page is not None and page.is_dead

    def _find_linked_page_keys_of_other_works(self, key: str) -> frozenset[str]:
        """Return the keys of linked pages showing the page belongs to another work.

        A link points to another work when it points outside this entry to:
        - A provider already present in this entry (indicating a conflicting ID).
        - A page belonging to another known work in the database.

        Links to the page's own provider or unknown providers outside the entry
        do not count as evidence of another work.
        """
        page = self._pages_by_current_key.get(key)
        own_provider = get_page_provider(key)
        return frozenset(
            linked_page_key
            for linked_page_key in map(
                self._links.get_current_key, page.linked_page_keys if page else ()
            )
            if linked_page_key not in self._entry_page_keys
            and (provider := get_page_provider(linked_page_key)) != own_provider
            and (
                provider in self._entry_providers
                or self._known_work_entries.find_by_page_key(linked_page_key)
                is not None
            )
        )

    def _find_other_work_entries(
        self, linked_page_keys: frozenset[str]
    ) -> tuple[WorkEntry, ...]:
        """Return the known work entries holding these linked pages, each once."""
        linked_work_entries = (
            self._known_work_entries.find_by_page_key(linked_page_key)
            for linked_page_key in sorted(linked_page_keys)
        )
        return tuple(
            {
                work_entry.reference_page: work_entry
                for work_entry in linked_work_entries
                if work_entry
            }.values()
        )

    def _find_title_matches(
        self, provider: str, page_keys: tuple[str, ...]
    ) -> tuple[str, ...]:
        """Return the keys of the pages whose main title matches the entry's titles.

        The entry's titles are its own title and the main titles of its pages
        from other providers.
        """
        entry_titles = {fold_title_for_comparison(self._entry.title)} | {
            fold_title_for_comparison(page.titles[0])
            for key in self._entry_page_keys
            if get_decided_provider(key) != provider
            and (page := self._pages_by_current_key.get(key))
            and page.titles
        }
        return tuple(
            key
            for key in page_keys
            if (page := self._pages_by_current_key.get(key))
            and page.titles
            and fold_title_for_comparison(page.titles[0]) in entry_titles
        )

    def _find_year_type_matches(self, page_keys: tuple[str, ...]) -> tuple[str, ...]:
        """Return the keys of the pages whose year and type agree with the entry's.

        Args:
            page_keys: The current keys of one doubled provider's pages.

        Returns:
            The keys of the fetched pages that agree, in the order given.
        """
        return tuple(
            key
            for key in page_keys
            if (page := self._pages_by_current_key.get(key))
            and is_page_details_matching(page, self._entry)
        )

    def _find_season_matches(self, page_keys: tuple[str, ...]) -> tuple[str, ...]:
        """Return the keys of the pages in the entry's season; none without a season."""
        return tuple(
            key
            for key in page_keys
            if self._entry.season
            and (page := self._pages_by_current_key.get(key))
            and page.season == self._entry.season
        )

    def _is_each_page_whole_work(self, page_keys: list[str]) -> bool:
        """Report whether pages of one work each cover all of it, from their details.

        - Pages whose first air dates and episode counts agree are duplicates.
        - Pages whose first air dates or episode counts differ are parts, such
          as two OVAs aired months apart, or a pilot and the series it starts.
        - An unknown date or count agrees with any, because providers fill them
          in at different times.

        Args:
            page_keys: The current keys of the belonging pages.

        Returns:
            True when the details agree, so the pages are duplicates.
        """
        fetched_pages = [
            page for key in page_keys if (page := self._pages_by_current_key.get(key))
        ]
        episode_counts = {
            page.episode_count
            for page in fetched_pages
            if page.episode_count is not None
        }
        first_air_dates = {
            page.aired_from.date()
            for page in fetched_pages
            if page.aired_from is not None
        }
        return len(episode_counts) <= 1 and len(first_air_dates) <= 1

    def _is_details_matching(self, page_keys: list[str]) -> bool:
        """Report whether the pages' years and types agree with each other.

        Pages without details, from providers that are not fetched, agree.
        """
        fetched_pages = [
            page for key in page_keys if (page := self._pages_by_current_key.get(key))
        ]
        return is_year_matching(
            *(page.year for page in fetched_pages)
        ) and is_type_matching(*map(get_seed_type, fetched_pages))
