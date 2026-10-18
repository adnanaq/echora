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

from work_matching.comparable_title import comparable_title
from work_matching.known_works import KnownWorks
from work_matching.provider_page import ProviderPage, pages_by_current_key
from work_matching.provider_page_key import page_id_of_key, provider_of_key
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
_DECIDED_WITH = {"animecountdown": "simkl"}

# The seed type each provider type falls under. A provider type missing here,
# such as UNKNOWN, agrees with every seed type.
_SEED_TYPE: dict[str, WorkType] = {
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
        return _OUTCOME_OF_REASON[self]


_OUTCOME_OF_REASON: dict[Reason, Outcome] = {
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


@dataclass(frozen=True, slots=True)
class UnrelatedPage:
    """A doubled provider page whose links point to another anime work."""

    # Outbound links to other works: pages of providers already in the entry,
    # or pages of known works.
    links: frozenset[str]
    # The known work entries corresponding to those links.
    other_works: tuple[WorkEntry, ...]


@dataclass(frozen=True, slots=True)
class DoubledProviderDecision:
    """Which of an entry's pages from one provider belong to its work, and why."""

    # The provider decided; "simkl" for Simkl and AnimeCountdown together.
    provider: str
    # The provider's pages, with renamed pages resolved to their current key.
    pages: tuple[str, ...]
    reason: Reason
    # Each page that belongs, mapped to the entry's pages linked with it.
    belonging: Mapping[str, frozenset[str]] = field(default_factory=dict)
    # Pages that link to another work instead of this entry.
    unrelated: Mapping[str, UnrelatedPage] = field(default_factory=dict)
    # Pages with no links either to the entry or to another work.
    undecided: tuple[str, ...] = ()
    # Review helper metadata: pages matching the entry's title, year/type, or season.
    title_matches: tuple[str, ...] = ()
    year_type_matches: tuple[str, ...] = ()
    season_matches: tuple[str, ...] = ()

    @property
    def outcome(self) -> Outcome:
        """What the decision says about the pages; a person checks ``REVIEW``."""
        return self.reason.outcome


def decide_doubled_providers(
    entry: WorkEntry, pages: Iterable[ProviderPage], known_works: KnownWorks
) -> list[DoubledProviderDecision]:
    """Decide every doubled provider of one entry.

    Args:
        entry: The work entry.
        pages: Details of the entry's fetched pages, and of found pages
            whose lookup page is one of them (such as AnimeSchedule).
        known_works: The works already known, to identify works an unrelated
            page links to.

    Returns:
        One decision per provider with two or more pages in the entry, in the
        order the entry lists them.
    """
    check = _EntryCheck(entry, pages, known_works)
    return [
        check.decide_provider(provider, list(keys))
        for provider, keys in doubled_provider_pages(entry.pages).items()
    ]


def doubled_provider_pages(page_keys: Iterable[str]) -> dict[str, tuple[str, ...]]:
    """Return the pages of each doubled provider among these page keys.

    A provider is doubled when its pages carry more than one id: two pages of
    one provider with different ids are different pages. Pages of providers
    decided together count as one provider (see ``decided_provider``), so a
    Simkl page and an AnimeCountdown page with the same number are one page,
    not a doubled provider. Seed cleaning lists the entries to check with this rule,
    so it checks exactly the pages ``decide_doubled_providers`` decides.

    Args:
        page_keys: Page keys, such as an entry's pages.

    Returns:
        Each doubled provider, by the name its decision falls under, with its
        page keys in the order given.
    """
    pages_by_provider: defaultdict[str, list[str]] = defaultdict(list)
    for key in page_keys:
        if provider := decided_provider(key):
            pages_by_provider[provider].append(key)
    return {
        provider: tuple(keys)
        for provider, keys in pages_by_provider.items()
        if _has_several_ids(keys)
    }


def decided_provider(key: str) -> str | None:
    """Return the provider whose decision a page key falls under, if any.

    AnimeCountdown numbers a work the same as Simkl (on release 2026-40, all
    14,649 entries with one list the other with the same number), so its
    pages fall under ``simkl``.

    Args:
        key: A page key, as ``provider_page_key`` gives it.

    Returns:
        The provider name, or None when the key is not a known provider's page.
    """
    provider = provider_of_key(key)
    return _DECIDED_WITH.get(provider, provider) if provider else None


class _EntryCheck:
    """Evaluates links and metadata for one entry's doubled provider pages."""

    def __init__(
        self, entry: WorkEntry, pages: Iterable[ProviderPage], known_works: KnownWorks
    ) -> None:
        self._entry = entry
        fetched = tuple(pages)
        self._links = ProviderPageLinks(fetched, entry.pages)
        self._details = pages_by_current_key(fetched)
        self._known_works = known_works
        self._entry_pages = frozenset(map(self._links.same_page, entry.pages))
        self._entry_providers = frozenset(map(provider_of_key, self._entry_pages))

    def decide_provider(
        self, provider: str, keys: list[str]
    ) -> DoubledProviderDecision:
        """Decide the outcome and belonging pages for one doubled provider."""
        pages = tuple(dict.fromkeys(map(self._links.same_page, keys)))
        if not _has_several_ids(pages):
            return DoubledProviderDecision(provider, pages, Reason.SAME_PAGE)
        linked = {key: self._links.linked_with(key) for key in pages}
        other_work_links = {key: self._links_to_other_works(key) for key in pages}
        reason = self._reason_for(pages, linked, other_work_links)
        is_review = reason.outcome is Outcome.REVIEW
        return DoubledProviderDecision(
            provider,
            pages,
            reason,
            belonging={
                key: linked[key]
                for key in pages
                if linked[key] and not other_work_links[key]
            },
            unrelated={
                key: UnrelatedPage(
                    other_work_links[key], self._other_works(other_work_links[key])
                )
                for key in pages
                if other_work_links[key] and not linked[key]
            },
            undecided=tuple(
                key for key in pages if not (linked[key] or other_work_links[key])
            ),
            title_matches=self._title_matches(provider, pages) if is_review else (),
            year_type_matches=self._year_type_matches(pages) if is_review else (),
            season_matches=self._season_matches(pages) if is_review else (),
        )

    def _reason_for(
        self,
        pages: tuple[str, ...],
        linked: Mapping[str, frozenset[str]],
        other_work_links: Mapping[str, frozenset[str]],
    ) -> Reason:
        """Return the reason determining the pages' outcome, review triggers first.

        A multi-work link or dead page requires review before evaluating link graphs,
        even if the page is the only one linked to the entry.
        """
        if any(map(self._links.linked_by_multi_work_pages, pages)):
            return Reason.MULTI_WORK_LINK
        if any(map(self._is_dead, pages)):
            return Reason.DEAD_PAGE
        if any(linked[key] and other_work_links[key] for key in pages):
            return Reason.CONTRADICTING_LINKS
        if not all(linked[key] or other_work_links[key] for key in pages):
            return Reason.UNDECIDED_PAGES
        linked_sets = [linked[key] for key in pages if linked[key]]
        if not linked_sets:
            return Reason.NONE_LINKED
        if len(linked_sets) == 1:
            return Reason.ONE_LINKED
        if frozenset.intersection(*linked_sets):
            linked_pages = [key for key in pages if linked[key]]
            if self._is_details_matching(linked_pages):
                return Reason.SHARED_LINK
            return Reason.DETAILS_DIFFER
        if sum(map(len, linked_sets)) == len(frozenset.union(*linked_sets)):
            return Reason.LINKED_APART
        return Reason.OVERLAPPING_LINKS

    def _is_dead(self, key: str) -> bool:
        """Report whether a page was fetched and the provider no longer serves it."""
        page = self._details.get(key)
        return page is not None and page.dead

    def _links_to_other_works(self, key: str) -> frozenset[str]:
        """Return a page's links that provide evidence of belonging to another work.

        A link points to another work when it points outside this entry to:
        - A provider already present in this entry (indicating a conflicting ID).
        - A page belonging to another known work in the database.

        Links to the page's own provider or unknown providers outside the entry
        do not count as evidence of another work.
        """
        page = self._details.get(key)
        own_provider = provider_of_key(key)
        return frozenset(
            link
            for link in map(self._links.same_page, page.links if page else ())
            if link not in self._entry_pages
            and (provider := provider_of_key(link)) != own_provider
            and (
                provider in self._entry_providers
                or self._known_works.find_by_page_key(link) is not None
            )
        )

    def _other_works(self, links: frozenset[str]) -> tuple[WorkEntry, ...]:
        """Return the known works the links belong to, each once."""
        works = (self._known_works.find_by_page_key(link) for link in sorted(links))
        return tuple({work.reference_page: work for work in works if work}.values())

    def _title_matches(self, provider: str, pages: tuple[str, ...]) -> tuple[str, ...]:
        """Return the pages whose main title matches the entry's titles.

        The entry's titles are its own title and the main titles of its pages
        from other providers.
        """
        entry_titles = {comparable_title(self._entry.title)} | {
            comparable_title(page.titles[0])
            for key in self._entry_pages
            if decided_provider(key) != provider
            and (page := self._details.get(key))
            and page.titles
        }
        return tuple(
            key
            for key in pages
            if (page := self._details.get(key))
            and page.titles
            and comparable_title(page.titles[0]) in entry_titles
        )

    def _year_type_matches(self, pages: tuple[str, ...]) -> tuple[str, ...]:
        """Return the pages whose year and type agree with the entry's."""
        entry = self._entry
        return tuple(
            key
            for key in pages
            if (page := self._details.get(key))
            and is_year_matching(page.year, entry.year)
            and is_type_matching(_seed_type(page), entry.type)
        )

    def _season_matches(self, pages: tuple[str, ...]) -> tuple[str, ...]:
        """Return the pages whose season is the entry's; none when it has no season."""
        return tuple(
            key
            for key in pages
            if self._entry.season
            and (page := self._details.get(key))
            and page.season == self._entry.season
        )

    def _is_details_matching(self, pages: list[str]) -> bool:
        """Report whether the pages' years and types agree with each other.

        Pages without details, from providers that are not fetched, agree.
        """
        details = [page for key in pages if (page := self._details.get(key))]
        return is_year_matching(*(page.year for page in details)) and is_type_matching(
            *map(_seed_type, details)
        )


def _seed_type(page: ProviderPage) -> WorkType | None:
    """Return the seed type a page's type falls under, or None when unknown."""
    return _SEED_TYPE.get(page.type or "")


def _has_several_ids(keys: Iterable[str]) -> bool:
    """Report whether a decision group's page keys carry more than one id.

    Pages of one provider with different ids are different pages, and so are
    pages of providers decided together, since they number a work the same.
    """
    return len(set(map(page_id_of_key, keys))) > 1
