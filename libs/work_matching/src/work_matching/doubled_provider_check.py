"""Decide which of an entry's pages from one provider belong to the entry's work.

A "doubled provider" occurs when an anime entry contains two or more pages from
the same provider. Cross-provider links decide what these pages represent:
- SAME_PAGE: The addresses resolve to the same canonical page (e.g. after a rename).
- ONE_WORK: The pages belong to this entry's single work, either because only one
  page links to the entry while others link other works, or because the provider
  maintains duplicate pages for the same work that share links to the entry.
- MIXED_ENTRY: The entry accidentally contains pages from multiple distinct works;
  the doubled pages link to completely separate, disjoint sets of the entry's pages.
- REVIEW: The link evidence is inconclusive, conflicting, or missing.

Title, year, and type never decide automatically because providers listing one
work twice often vary titles or release dates; picking by title would mistakenly
drop duplicate pages. These attributes are recorded as metadata for review cases.
"""

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Annotated

from pydantic import PlainSerializer
from work_matching.comparable_title import fold_title_for_comparison
from work_matching.known_work_entries import KnownWorkEntries
from work_matching.provider_page import ProviderPage, index_pages_by_current_key
from work_matching.provider_page_key import get_page_id, get_page_provider
from work_matching.provider_page_links import ProviderPageLinks
from work_matching.work_entry import (
    WorkEntry,
    WorkType,
    is_type_matching,
    is_year_matching,
)

# Simkl and AnimeCountdown number a work the same, so they are decided together,
# under Simkl's name: a Simkl and an AnimeCountdown page with different numbers
# are two works, just as two Simkl pages are.
_PROVIDER_DECIDED_UNDER = {"animecountdown": "simkl"}

# The seed type each provider type falls under. A provider type missing here,
# such as UNKNOWN, agrees with every seed type.
_SEED_TYPE_BY_PROVIDER_TYPE: dict[str, WorkType] = {
    "TV": "TV",
    "TV SHORT": "TV",
    "MOVIE": "MOVIE",
    "OVA": "OVA",
    "ONA": "ONA",
    "SPECIAL": "SPECIAL",
    "TV SPECIAL": "SPECIAL",
    "MUSIC": "SPECIAL",
    "CM": "SPECIAL",
    "PV": "SPECIAL",
    "OTHER": "SPECIAL",
}


class Outcome(StrEnum):
    """The high-level conclusion for a doubled provider's pages."""

    SAME_PAGE = "same page"
    ONE_WORK = "one work"
    MIXED_ENTRY = "mixed entry"
    REVIEW = "review"


class Reason(StrEnum):
    """The specific rule explaining why a decision was reached."""

    SAME_PAGE = "the addresses are one page: an old address leads to the renamed one"
    ONE_LINKED = (
        "exactly one page is linked with the entry; every other page links a "
        "different work"
    )
    SHARED_LINK = (
        "the linked pages are all linked through one same page, with the same year "
        "and type"
    )
    LINKED_APART = (
        "the linked pages are linked through entirely separate pages, so each is a "
        "separate work"
    )
    MULTI_WORK_LINK = "a page naming several pages of one provider links these pages"
    DEAD_PAGE = "a page no longer exists on the provider's site"
    CONTRADICTING_LINKS = (
        "a page is linked with the entry and also links a different work"
    )
    UNDECIDED_PAGES = "a page has no link to the entry and no link to another work"
    DETAILS_DIFFER = (
        "the linked pages are linked through one same page, but their years or "
        "types differ"
    )
    OVERLAPPING_LINKS = (
        "the linked pages share some linked pages, but no page is linked with all "
        "of them"
    )
    NONE_LINKED = "every page links a different work and none is linked with the entry"

    @property
    def outcome(self) -> Outcome:
        """What a decision with this reason says about the pages."""
        return _OUTCOME_BY_REASON[self]


_OUTCOME_BY_REASON: dict[Reason, Outcome] = {
    Reason.SAME_PAGE: Outcome.SAME_PAGE,
    Reason.ONE_LINKED: Outcome.ONE_WORK,
    Reason.SHARED_LINK: Outcome.ONE_WORK,
    Reason.LINKED_APART: Outcome.MIXED_ENTRY,
    Reason.MULTI_WORK_LINK: Outcome.REVIEW,
    Reason.DEAD_PAGE: Outcome.REVIEW,
    Reason.CONTRADICTING_LINKS: Outcome.REVIEW,
    Reason.UNDECIDED_PAGES: Outcome.REVIEW,
    Reason.DETAILS_DIFFER: Outcome.REVIEW,
    Reason.OVERLAPPING_LINKS: Outcome.REVIEW,
    Reason.NONE_LINKED: Outcome.REVIEW,
}


# Page keys held as a set, written sorted: a set's order changes from one process
# to the next, and a review file written in a fixed order lets two runs be
# compared line by line.
PageKeySet = Annotated[
    frozenset[str], PlainSerializer(sorted, return_type=list[str], when_used="json")
]


@dataclass(frozen=True, slots=True)
class UnrelatedPage:
    """A doubled provider page whose links point to another anime work."""

    # Outbound links to other works: pages of providers already in the entry,
    # or pages of known works.
    linked_page_keys: PageKeySet
    # The known work entries corresponding to those links.
    other_work_entries: tuple[WorkEntry, ...]


@dataclass(frozen=True, slots=True)
class DoubledProviderDecision:
    """Which of an entry's pages from one provider belong to its work, and why."""

    # The provider decided; "simkl" for Simkl and AnimeCountdown together.
    provider: str
    # The provider's page keys, with renamed pages resolved to their current key.
    page_keys: tuple[str, ...]
    reason: Reason
    # Each belonging page's key, mapped to the keys of the entry's pages linked
    # with it.
    belonging_page_keys: Mapping[str, PageKeySet] = field(default_factory=dict)
    # Each unrelated page's key, mapped to what its links say about other works.
    unrelated_pages: Mapping[str, UnrelatedPage] = field(default_factory=dict)
    # Keys of pages with no links either to the entry or to another work.
    undecided_page_keys: tuple[str, ...] = ()
    # On review only: keys of pages matching the entry's title, year and type, or
    # season.
    title_matches: tuple[str, ...] = ()
    year_type_matches: tuple[str, ...] = ()
    season_matches: tuple[str, ...] = ()

    @property
    def outcome(self) -> Outcome:
        """What the decision says about the pages; a person checks ``REVIEW``."""
        return self.reason.outcome


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


def find_doubled_provider_page_keys(
    page_keys: Iterable[str],
) -> dict[str, tuple[str, ...]]:
    """Return the page keys of each doubled provider among these page keys.

    A provider is doubled when its pages carry more than one id: two pages of
    one provider with different ids are different pages. Pages of providers
    decided together count as one provider (see ``get_decided_provider``), so a
    Simkl page and an AnimeCountdown page with the same number are one page,
    not a doubled provider. Seed cleaning lists the entries to check with this rule,
    so it checks exactly the pages ``decide_doubled_providers`` decides.

    Args:
        page_keys: Page keys, such as the keys of an entry's pages.

    Returns:
        Each doubled provider, by the name its decision falls under, with its
        page keys in the order given.
    """
    page_keys_by_provider: defaultdict[str, list[str]] = defaultdict(list)
    for key in page_keys:
        if provider := get_decided_provider(key):
            page_keys_by_provider[provider].append(key)
    return {
        provider: tuple(page_keys)
        for provider, page_keys in page_keys_by_provider.items()
        if _has_several_ids(page_keys)
    }


def get_decided_provider(key: str) -> str | None:
    """Return the provider whose decision a page key falls under, if any.

    AnimeCountdown numbers a work the same as Simkl: an entry listing one lists
    the other with the same number. So its pages fall under ``simkl``.

    Args:
        key: A page key, as ``get_page_key`` gives it.

    Returns:
        The provider name, or None when the key is not a known provider's page.
    """
    provider = get_page_provider(key)
    return _PROVIDER_DECIDED_UNDER.get(provider, provider) if provider else None


def get_seed_type(page: ProviderPage) -> WorkType | None:
    """Return the seed type a provider page's type falls under, or None when unknown.

    Providers name types more finely than the seed (``TV SHORT``, ``MUSIC``), so
    comparing a page with an entry, or building an entry from a page, needs the
    seed's coarser type.
    """
    return _SEED_TYPE_BY_PROVIDER_TYPE.get(page.type or "")


def is_page_details_matching(page: ProviderPage, entry: WorkEntry) -> bool:
    """Report whether a page's year and type agree with an entry's.

    A page joining an entry, by a move or as an added linked page, must agree.
    A disagreement shows the page's link may be a wrong mapping. The rules:

    - A missing year agrees with any year, because providers fill years in at
      different times.
    - An unknown type agrees with any type, for the same reason.
    - The page's type is compared as the seed's coarser type (``get_seed_type``).

    Args:
        page: The provider page's fetched details.
        entry: The work entry the page would join.

    Returns:
        True when both the year and the type agree.
    """
    return is_year_matching(page.year, entry.year) and is_type_matching(
        get_seed_type(page), entry.type
    )


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
        """Decide the outcome and belonging pages for one doubled provider."""
        current_page_keys = tuple(
            dict.fromkeys(map(self._links.get_current_key, page_keys))
        )
        if not _has_several_ids(current_page_keys):
            return DoubledProviderDecision(
                provider, current_page_keys, Reason.SAME_PAGE
            )
        linked_page_keys_by_page_key = {
            key: self._links.get_linked_page_keys(key) for key in current_page_keys
        }
        other_work_page_keys_by_page_key = {
            key: self._find_linked_page_keys_of_other_works(key)
            for key in current_page_keys
        }
        reason = self._choose_reason(
            current_page_keys,
            linked_page_keys_by_page_key,
            other_work_page_keys_by_page_key,
        )
        is_review = reason.outcome is Outcome.REVIEW
        return DoubledProviderDecision(
            provider,
            current_page_keys,
            reason,
            belonging_page_keys={
                key: linked_page_keys_by_page_key[key]
                for key in current_page_keys
                if linked_page_keys_by_page_key[key]
                and not other_work_page_keys_by_page_key[key]
            },
            unrelated_pages={
                key: UnrelatedPage(
                    other_work_page_keys_by_page_key[key],
                    self._find_other_work_entries(
                        other_work_page_keys_by_page_key[key]
                    ),
                )
                for key in current_page_keys
                if other_work_page_keys_by_page_key[key]
                and not linked_page_keys_by_page_key[key]
            },
            undecided_page_keys=tuple(
                key
                for key in current_page_keys
                if not (
                    linked_page_keys_by_page_key[key]
                    or other_work_page_keys_by_page_key[key]
                )
            ),
            title_matches=self._find_title_matches(provider, current_page_keys)
            if is_review
            else (),
            year_type_matches=self._find_year_type_matches(current_page_keys)
            if is_review
            else (),
            season_matches=self._find_season_matches(current_page_keys)
            if is_review
            else (),
        )

    def _choose_reason(
        self,
        page_keys: tuple[str, ...],
        linked_page_keys_by_page_key: Mapping[str, frozenset[str]],
        other_work_page_keys_by_page_key: Mapping[str, frozenset[str]],
    ) -> Reason:
        """Return the reason determining the pages' outcome, review triggers first.

        A multi-work link or dead page requires review before evaluating link graphs,
        even if the page is the only one linked to the entry.
        """
        if any(map(self._links.get_linking_multi_work_page_keys, page_keys)):
            return Reason.MULTI_WORK_LINK
        if any(map(self._is_dead, page_keys)):
            return Reason.DEAD_PAGE
        if any(
            linked_page_keys_by_page_key[key] and other_work_page_keys_by_page_key[key]
            for key in page_keys
        ):
            return Reason.CONTRADICTING_LINKS
        if not all(
            linked_page_keys_by_page_key[key] or other_work_page_keys_by_page_key[key]
            for key in page_keys
        ):
            return Reason.UNDECIDED_PAGES
        linked_page_key_sets = [
            linked_page_keys_by_page_key[key]
            for key in page_keys
            if linked_page_keys_by_page_key[key]
        ]
        if not linked_page_key_sets:
            return Reason.NONE_LINKED
        if len(linked_page_key_sets) == 1:
            return Reason.ONE_LINKED
        if frozenset.intersection(*linked_page_key_sets):
            linked_page_keys = [
                key for key in page_keys if linked_page_keys_by_page_key[key]
            ]
            if self._is_details_matching(linked_page_keys):
                return Reason.SHARED_LINK
            return Reason.DETAILS_DIFFER
        if sum(map(len, linked_page_key_sets)) == len(
            frozenset.union(*linked_page_key_sets)
        ):
            return Reason.LINKED_APART
        return Reason.OVERLAPPING_LINKS

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


def _has_several_ids(page_keys: Iterable[str]) -> bool:
    """Report whether a decision group's page keys carry more than one id.

    Pages of one provider with different ids are different pages, and so are
    pages of providers decided together, since they number a work the same.
    """
    return len(set(map(get_page_id, page_keys))) > 1
