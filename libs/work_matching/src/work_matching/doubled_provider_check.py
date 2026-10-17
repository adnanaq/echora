"""Decide which of an entry's pages from one provider belong to the entry's work.

An entry with two or more pages of one provider is a doubled provider. Links
decide: a page linked with another provider's page of the entry belongs, and a
page linking another work does not. Title, year and type never decide, because
a provider that lists one work twice gives the two pages different titles, and
picking one by title wrongly drops the other. They are recorded for review.
"""

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from work_matching.comparable_title import comparable_title
from work_matching.known_works import KnownWorks
from work_matching.provider_page import ProviderPage
from work_matching.provider_page_key import provider_of_key
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
    """What a decision says about a doubled provider's pages."""

    SAME_PAGE = "same page"
    ONE_WORK = "one work"
    MIXED_ENTRY = "mixed entry"
    REVIEW = "review"


class Reason(StrEnum):
    """Why a decision came out as it did."""

    SAME_PAGE = "the addresses name one page"
    ONE_LINKED = "one page is linked with the entry and the others link other works"
    SHARED_LINK = "every linked page is linked with one same page of the entry"
    LINKED_APART = "the linked pages are linked with separate pages of the entry"
    MULTI_WORK_LINK = "a page linking several pages of one provider links these pages"
    DEAD_PAGE = "a page is no longer served"
    CONTRADICTING_LINKS = "a page is linked with the entry and links another work"
    UNDECIDED_PAGES = "a page is neither linked with the entry nor links another work"
    DETAILS_DIFFER = (
        "the linked pages share a linked page but their year or type differ"
    )
    OVERLAPPING_LINKS = "the linked pages share some pages but none linked with all"
    NONE_LINKED = "no page is linked with the entry"


_OUTCOME_OF_REASON = {
    Reason.SAME_PAGE: Outcome.SAME_PAGE,
    Reason.ONE_LINKED: Outcome.ONE_WORK,
    Reason.SHARED_LINK: Outcome.ONE_WORK,
    Reason.LINKED_APART: Outcome.MIXED_ENTRY,
}


@dataclass(frozen=True, slots=True)
class UnrelatedPage:
    """A page whose links name another work."""

    # Its links to pages outside the entry: pages of a provider the entry has,
    # or pages of another known work.
    links: frozenset[str]
    # The known works those links belong to.
    other_works: tuple[WorkEntry, ...]


@dataclass(frozen=True, slots=True)
class DoubledProviderDecision:
    """Which of an entry's pages from one provider belong to its work, and why."""

    # The provider decided; "simkl" for Simkl and AnimeCountdown together.
    provider: str
    # The provider's pages, renamed pages folded into their current key.
    pages: tuple[str, ...]
    reason: Reason
    # Each page that belongs, with the entry's pages linked with it. On a mixed
    # entry each page is a separate work, together with its linked pages.
    belonging: Mapping[str, frozenset[str]] = field(default_factory=dict)
    unrelated: Mapping[str, UnrelatedPage] = field(default_factory=dict)
    # Pages without links either way.
    undecided: tuple[str, ...] = ()
    # On review, the pages whose main title matches the entry's titles, those
    # whose year and type agree with the entry's, and those in its season.
    title_matches: tuple[str, ...] = ()
    year_type_matches: tuple[str, ...] = ()
    season_matches: tuple[str, ...] = ()

    @property
    def outcome(self) -> Outcome:
        """What the decision says about the pages; a person checks ``REVIEW``."""
        return _outcome_of(self.reason)


def decide_doubled_providers(
    entry: WorkEntry, pages: Iterable[ProviderPage], known_works: KnownWorks
) -> list[DoubledProviderDecision]:
    """Decide every doubled provider of one entry.

    Args:
        entry: The work entry.
        pages: Details of the entry's fetched pages, and of pages found through
            them, such as AnimeSchedule's.
        known_works: The works already known, to name the work an unrelated
            page links to.

    Returns:
        One decision per provider with two or more pages in the entry, in the
        order the entry lists them.
    """
    groups: defaultdict[str, list[str]] = defaultdict(list)
    for key in entry.pages:
        if group := _decision_group(key):
            groups[group].append(key)
    check = _EntryCheck(entry, pages, known_works)
    return [
        check.decide(group, keys)
        for group, keys in groups.items()
        if _has_several_ids(keys)
    ]


class _EntryCheck:
    """The links and details of one entry's pages, shared by its decisions."""

    def __init__(
        self, entry: WorkEntry, pages: Iterable[ProviderPage], known_works: KnownWorks
    ) -> None:
        self._entry = entry
        self._details = {page.key: page for page in pages}
        self._links = ProviderPageLinks(self._details.values(), entry.pages)
        self._known_works = known_works
        self._entry_pages = frozenset(map(self._links.same_page, entry.pages))
        self._entry_providers = frozenset(map(provider_of_key, self._entry_pages))

    def decide(self, provider: str, keys: list[str]) -> DoubledProviderDecision:
        """Decide one provider's pages."""
        pages = tuple(dict.fromkeys(map(self._links.same_page, keys)))
        if not _has_several_ids(pages):
            return DoubledProviderDecision(provider, pages, Reason.SAME_PAGE)
        linked = {key: self._links.linked_with(key) for key in pages}
        elsewhere = {key: self._links_elsewhere(key) for key in pages}
        reason = self._reason(pages, linked, elsewhere)
        is_review = _outcome_of(reason) is Outcome.REVIEW
        return DoubledProviderDecision(
            provider,
            pages,
            reason,
            belonging={
                key: linked[key] for key in pages if linked[key] and not elsewhere[key]
            },
            unrelated={
                key: UnrelatedPage(elsewhere[key], self._other_works(elsewhere[key]))
                for key in pages
                if elsewhere[key] and not linked[key]
            },
            undecided=tuple(
                key for key in pages if not (linked[key] or elsewhere[key])
            ),
            title_matches=self._title_matches(provider, pages) if is_review else (),
            year_type_matches=self._year_type_matches(pages) if is_review else (),
            season_matches=self._season_matches(pages) if is_review else (),
        )

    def _reason(
        self,
        pages: tuple[str, ...],
        linked: Mapping[str, frozenset[str]],
        elsewhere: Mapping[str, frozenset[str]],
    ) -> Reason:
        """Return the rule that decides the pages, checks that refuse first."""
        if any(map(self._links.linked_by_multi_work_pages, pages)):
            return Reason.MULTI_WORK_LINK
        if any((page := self._details.get(key)) and page.dead for key in pages):
            return Reason.DEAD_PAGE
        if any(linked[key] and elsewhere[key] for key in pages):
            return Reason.CONTRADICTING_LINKS
        if not all(linked[key] or elsewhere[key] for key in pages):
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

    def _links_elsewhere(self, key: str) -> frozenset[str]:
        """Return a page's links naming another work.

        A link names another work when it points outside the entry, at a page
        of a provider the entry has or at a page of another known work.
        """
        page = self._details.get(key)
        return frozenset(
            link
            for link in map(self._links.same_page, page.links if page else ())
            if link not in self._entry_pages
            and (
                provider_of_key(link) in self._entry_providers
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
            if _decision_group(key) != provider
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


def _outcome_of(reason: Reason) -> Outcome:
    """Return what a decision with this reason says about the pages."""
    return _OUTCOME_OF_REASON.get(reason, Outcome.REVIEW)


def _has_several_ids(keys: Iterable[str]) -> bool:
    """Report whether a decision group's page keys carry more than one id.

    Pages of one provider with different ids are different pages, and so are
    pages of providers decided together, since they number a work the same.
    """
    return len({key.rpartition(":")[2] for key in keys}) > 1


def _decision_group(key: str) -> str | None:
    """Return the provider whose decision a page key falls under, if any."""
    provider = provider_of_key(key)
    return _DECIDED_WITH.get(provider, provider) if provider else None
