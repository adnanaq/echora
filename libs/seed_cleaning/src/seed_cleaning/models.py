"""Define the data models and value objects of seed cleaning.

Each model is an immutable record of what a seed-cleaning pass groups, places,
corrects, or reports:

- ``SplitWorkLink`` and ``SplitWorkLinks``: links tying seed entries that may be
  one split work.
- ``SplitWorkCandidates`` and ``EntryGroup``: entries grouped for one review
  case, with the evidence that grouped them.
- ``SeedCorrection`` and ``PlacementCorrections``: where each checked page ends
  up and why, alongside the pass's review cases.
- ``CorrectedEntry``, ``EntryIdentity``, ``PageOrigin``, and ``CorrectedSeed``:
  entries as the first pass leaves them, with how second-pass corrections name
  their seed entries and pages.
- ``PageDestination``, ``CorrectionNames``, ``GroupPlacement``, and
  ``AnswerPlacements``: where a group's placed pages go and which answer placed
  them.
- ``LinkingSeedEntry``, ``LinkedPageGroup``, and ``LinkedPageCorrections``:
  unheld linked pages grouped by current key and their resulting corrections.
- ``SplitWorkPass``, ``SeedFetchSummary``, and ``SeedCleaningRunResult``: the
  second pass's inputs, the page-fetch summary, and the run result written to
  ``review_run.json``.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import NamedTuple, Self

from enrichment.pipeline.fetch_summary import FetchSummary
from pydantic import BaseModel, ConfigDict, JsonValue
from work_matching import (
    KnownWorkEntriesFromSeed,
    LinkingEntryDetails,
    PagePlacement,
    ProviderPage,
    ReviewCase,
    WorkEntry,
)


class SplitWorkLink(NamedTuple):
    """Record one link tying two entries: a page of one links a page the other holds."""

    linking_page_key: str
    linked_page_key: str


class SplitWorkLinks(NamedTuple):
    """Hold the links tying seed entries, by the entries' places in the seed."""

    # Each tied pair of entries, by their places in the seed (the lower first),
    # mapped to every link tying them.
    links_by_entry_pair: dict[tuple[int, int], tuple[SplitWorkLink, ...]]
    # Each linked page that ties entries, mapped to the places of every entry
    # whose pages link it.
    linking_entries_by_page_key: dict[str, frozenset[int]]
    # Each page linked from two or more entries holding their own MAL page,
    # which covers several works, mapped to the places of those entries.
    mal_entries_by_several_works_page_key: dict[str, frozenset[int]] = {}


class SplitWorkCandidates(BaseModel):
    """Hold seed entries that may be one work, for a person to review as one group."""

    model_config = ConfigDict(frozen=True)

    # In seed file order.
    entries: tuple[WorkEntry, ...]
    # The one provider with pages in two or more of the entries, if any. That
    # provider lists the entries as separate pages, which alone does not decide:
    # such groups are sometimes one work and sometimes not.
    shared_provider: str | None = None
    # Why the entries are candidates, shown to the reviewer: each shared title
    # (``same title: 'A' and 'B'``) and each link (``<page key> links <page
    # key>``), titles first.
    candidate_evidence: tuple[str, ...] = ()
    # Whether a page tying the entries is also linked from entries outside the
    # group, so its link is weaker evidence; the review case warns about it.
    has_shared_tying_link: bool = False


class EntryGroup(NamedTuple):
    """Hold entries one review case places, with why they are grouped."""

    # In seed file order.
    entries: tuple[WorkEntry, ...]
    # Why the entries are candidates, shown to the reviewer; empty for a
    # single entry.
    candidate_evidence: tuple[str, ...] = ()
    # Whether a page tying the entries is also linked from entries outside the
    # group, which the review case warns about.
    has_shared_tying_link: bool = False

    @classmethod
    def from_candidates(cls, candidates: SplitWorkCandidates) -> Self:
        """Make the group of one set of split-work candidates, keeping its evidence.

        Args:
            candidates: One group of split-work candidate entries and evidence.

        Returns:
            The entry group built from ``candidates``.
        """
        return cls(
            candidates.entries,
            candidates.candidate_evidence,
            candidates.has_shared_tying_link,
        )


class SeedCorrection(BaseModel):
    """Record where one checked page ends up, and why."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # The reference page of the seed entry that lists the page.
    entry_reference_page: str
    # The page key and address as the seed entry lists them.
    page_key: str
    address: str
    # The page's work in answer words: ``work 1``, ``work 2`` and so on or
    # ``other work``. None for a dead page, which the run drops from every work.
    placement_word: str | None
    # The current page key, when the seed lists the page under an old address.
    renamed_to: str | None = None
    # The reference page of the seed entry the page joins; None when the page
    # stays or starts a new entry.
    target_entry_reference_page: str | None = None
    # The page key an answer named after ``other work``; the page ends up
    # wherever that named page ends up. None when no answer named a page.
    named_entry_page_key: str | None = None
    # The new entry the page joins: pages with the same key form one new entry.
    # None when the page stays or joins a seed entry.
    new_entry_key: str | None = None
    # Whether the page is not in the seed and is added to the entry: a page the
    # entry's pages link that no seed entry holds.
    is_added_page: bool = False
    # Whether the provider says the page does not exist, so the run drops it
    # from its entry.
    is_dead_page: bool = False
    # Whether an answer drops the page as a duplicate page, so it leaves its
    # entry and joins no other.
    is_dropped_page: bool = False
    # The reference pages of the seed entries the page is a partial source of,
    # as corrections name them.
    partial_entry_reference_pages: tuple[str, ...] = ()
    # The new entry keys of the new entries the page is a partial source of.
    partial_new_entry_keys: tuple[str, ...] = ()
    # Whether the anime id of the seed entry this added page ends in now comes
    # from the page (MAL first), so the search entry stored under the entry's
    # old id must be replaced. Set after both passes (``mark_anime_id_pages``).
    is_anime_id_page: bool = False
    # Why the page is placed here: the doubled-provider decision's outcome name,
    # the linked-page decision's reason name, ``review answer``, or
    # ``dead page``.
    reason: str


@dataclass(frozen=True)
class PlacementCorrections:
    """Hold the corrections placing the pages of the entries checked, and their review cases."""

    corrections: tuple[SeedCorrection, ...]
    # Every case of this run, with its kept answer, for the review file.
    review_cases: tuple[ReviewCase, ...]


@dataclass(frozen=True)
class CorrectedEntry:
    """Hold a work entry after corrections, and the original seed entry it stands in for."""

    # The reference page of the original seed entry this corrected entry takes
    # the place of in the cleaned seed; None for a new entry, which takes the
    # place of nothing.
    seed_entry_reference_page: str | None
    # None when every page of the seed entry left it: the entry is removed from
    # the cleaned seed, since an entry without pages is no work.
    entry: WorkEntry | None
    # The key the corrections give a new entry's pages; None for a seed entry.
    new_entry_key: str | None = None


class EntryIdentity(NamedTuple):
    """Identify how seed corrections name an entry: by its seed entry or its new entry key."""

    # The reference page of the seed entry the entry stands in for; None for a
    # new entry.
    seed_entry_reference_page: str | None
    # The key a new entry's pages share in their corrections; None for a seed
    # entry.
    new_entry_key: str | None


class PageOrigin(NamedTuple):
    """Record how seed corrections name a page: the entry listing it and its address."""

    # The seed entry listing the page, or the seed entry it was added to.
    entry_reference_page: str
    address: str
    is_added_page: bool = False


@dataclass(frozen=True)
class CorrectedSeed:
    """Hold every work entry after the first pass, and how corrections name each one."""

    known_work_entries: KnownWorkEntriesFromSeed
    # Each entry's identity, by the entry's reference page.
    identity_by_reference_page: dict[str, EntryIdentity]
    # Each page's origin, by the entry's reference page and the page key.
    origin_by_entry_page: dict[tuple[str, str], PageOrigin]

    def find_page_origin(self, entry_reference_page: str, page_key: str) -> PageOrigin:
        """Return the origin of one page of one entry of the corrected seed.

        Args:
            entry_reference_page: The reference page of the entry in the
                corrected seed.
            page_key: The page key within that entry.

        Returns:
            The recorded ``PageOrigin`` for that entry and page.
        """
        return self.origin_by_entry_page[entry_reference_page, page_key]


class PageDestination(NamedTuple):
    """Hold the entry a placed page ends up in; both fields None when the page stays."""

    target_entry_reference_page: str | None
    new_entry_key: str | None


@dataclass(frozen=True)
class CorrectionNames:
    """Resolve how corrections name pages and entries: by the seed, or by a corrected seed."""

    # None in the first pass, whose entries are the seed's own.
    corrected_seed: CorrectedSeed | None

    def find_page_origin(
        self, entry: WorkEntry, page_key: str, address: str
    ) -> PageOrigin:
        """Return the seed entry listing a page of an entry, its address and whether added.

        Args:
            entry: The entry listing the page in this pass.
            page_key: The key the entry lists the page under.
            address: The page's address in the entry.

        Returns:
            The page's origin in the downloaded seed.
        """
        if self.corrected_seed is None:
            return PageOrigin(entry.reference_page_key, address)
        return self.corrected_seed.find_page_origin(entry.reference_page_key, page_key)

    def find_entry_destination(self, reference_page: str) -> PageDestination:
        """Return how a correction names the entry with this reference page.

        Args:
            reference_page: The reference page of an entry of this pass.

        Returns:
            The seed entry the entry stands for, or its new entry key.
        """
        if self.corrected_seed is None:
            return PageDestination(reference_page, None)
        identity = self.corrected_seed.identity_by_reference_page[reference_page]
        return PageDestination(
            identity.seed_entry_reference_page, identity.new_entry_key
        )


class GroupPlacement(NamedTuple):
    """Hold where an entry group's pages go, and what the group's corrections read."""

    combined_entry: WorkEntry
    # Each placed page's placement, and why it is placed, by its key.
    placements: Mapping[str, PagePlacement]
    reason_by_page_key: Mapping[str, str]
    # Each work's surviving entry, as ``find_surviving_entries`` gives them.
    surviving_reference_page_by_work: Mapping[str, str]
    # Each renamed page's current key, by the key an entry lists it under.
    current_page_key_by_page_key: Mapping[str, str]
    # The seed entry each unrelated page surely moves to, by its current key.
    target_reference_page_by_page_key: Mapping[str, str]
    # The reference page of the seed entry holding each page key an answer
    # names after ``other work``.
    named_entry_reference_pages: Mapping[str, str]
    correction_names: CorrectionNames


class AnswerPlacements(NamedTuple):
    """Hold the placements an answer gives, with the entries it was given for."""

    placements: dict[str, PagePlacement]
    named_entry_reference_pages: dict[str, str]
    # The reference pages of the entries the answer was given for; only these
    # can survive as a work, so an entry a previous answer never saw stays as
    # the seed lists it.
    answered_reference_pages: frozenset[str]


@dataclass(frozen=True, slots=True)
class LinkingSeedEntry:
    """Hold an entry linking an unheld page, and the seed entry it stands for."""

    # The reference page of the seed entry, which a correction names; the
    # corrected entry's own reference page can differ after a page moved out.
    seed_entry_reference_page: str
    entry_details: LinkingEntryDetails


# One linked page's details and every entry linking it, under any spelling.
type LinkedPageGroup = tuple[ProviderPage, tuple[LinkingSeedEntry, ...]]


@dataclass(frozen=True)
class LinkedPageCorrections:
    """Hold the added pages, the review cases, and the linked pages not added."""

    corrections: tuple[SeedCorrection, ...]
    # Every linked-page case of this run, with its kept answer, for the review file.
    review_cases: tuple[ReviewCase, ...]
    # Linked pages not fetched yet, or whose fetch failed, sorted.
    waiting_page_keys: tuple[str, ...]
    # Linked pages the provider says do not exist, sorted.
    not_found_page_keys: tuple[str, ...]
    # Linked pages a seed entry already holds under another address, by their
    # current key, sorted.
    held_page_keys: tuple[str, ...] = ()
    # Linked pages answers add as whole pages next to another page of their
    # provider in one entry, so none of them is added, sorted.
    conflicting_page_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class SplitWorkPass:
    """Hold the second pass's entries and candidates, and how corrections name them."""

    # Every entry as the first pass leaves it, for naming pages and entries.
    corrected_seed: CorrectedSeed
    # The entries the second pass groups: the corrected seed without the
    # entries of open first-pass cases.
    known_work_entries: KnownWorkEntriesFromSeed
    candidates: list[SplitWorkCandidates]


@dataclass(frozen=True)
class SeedFetchSummary:
    """Summarize how fetching a seed release's pages and their linked pages went."""

    seed_page_summary: FetchSummary
    linked_page_summary: FetchSummary
    animeschedule_search_summary: FetchSummary


class SeedCleaningRunResult(BaseModel):
    """Hold what one seed-cleaning run did, as the review page reads it."""

    model_config = ConfigDict(frozen=True)

    # The seed release file the run read, so a later run reads the same one.
    seed_file: str
    started_at: datetime
    finished_at: datetime
    # ``count_corrections`` by count name: pages moved, added, kept, cases.
    correction_counts: dict[str, int]
    # ``count_corrected_entries`` by count name: entries removed, new entries.
    corrected_entry_counts: dict[str, int]
    # Linked pages waiting for a fetch, not found, and held under another address.
    linked_page_counts: dict[str, int]
    # Cases whose answer the rules refused, so they stay open.
    rejected_answer_case_names: tuple[str, ...]
    # Linked pages answers add as whole pages beside another of their provider
    # in one entry, so neither is added.
    conflicting_linked_page_keys: tuple[str, ...]
    # Kept answers whose case did not come up in this run.
    unused_answer_case_names: tuple[str, ...]
    # Every case of this run, sorted.
    case_names: tuple[str, ...]
    # Cases of this run that the previous run did not have, sorted.
    new_case_names: tuple[str, ...]
    # Every answer in the review file as this run wrote it, by case name.
    original_answers: dict[str, JsonValue]
