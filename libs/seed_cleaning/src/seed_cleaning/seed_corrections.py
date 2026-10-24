"""Seed corrections: where each page of an entry with doubled providers ends up.

For every entry with a doubled provider, the work matching rules either place
its pages themselves or leave the entry to a person as one review case. A
correction records one page's final place, from the rules or from an accepted
answer; a correction can also add a page the seed lacks (see
``linked_page_additions``). The cleaned seed is built from the downloaded
release plus these corrections, so the release itself is never edited. The
rules:

- Final places, not steps: each correction says where its page ends up, so the
  order corrections are applied in never matters, even when a page moves into
  an entry that is itself being corrected.
- Rebuilt every run: corrections come from this run's decisions and the kept
  answers, so the file always matches them and needs no clean-up.
- Moves: a page placed in ``other work`` joins the seed entry its links name
  when they name exactly one; otherwise it starts a new entry. Pages a review
  answer puts in the same ``work N`` form one new entry.
- Renamed pages: an old address is dropped, because the entry also lists the
  page under its current address (the rules send it to review otherwise).
- Main page: a work's main page is listed before its other pages, because
  enrichment fetches the first address of each provider.
- New entries take their title, type, year and season from their most trusted
  fetched page. A page with no fetched details only has the entry it came from
  to go by, so it takes that entry's title and an unknown type.
"""

import logging
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from pydantic import BaseModel, ConfigDict
from work_matching import (
    ENTRY_WORK,
    OTHER_WORK,
    DoubledProviderDecision,
    KnownWorkEntries,
    PagePlacement,
    ProviderPage,
    ProviderPageStore,
    ReviewCase,
    ReviewFile,
    WorkEntry,
    decide_doubled_providers,
    find_move_target,
    get_seed_type,
    index_pages_by_current_key,
    place_pages_without_review,
    rank_by_provider_trust,
    replace_file_lines,
)

logger = logging.getLogger(__name__)

# The reason recorded for a page placed by a review answer.
REVIEW_ANSWER_REASON = "review answer"


class SeedCorrection(BaseModel):
    """Where one page of an entry with doubled providers ends up, and why."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # The reference page of the seed entry that lists the page.
    entry_reference_page: str
    # The page key and address as the seed entry lists them.
    page_key: str
    address: str
    # The page's work in answer words: ``entry``, ``other work`` or ``work 2``…
    placement_word: str
    # Whether enrichment fetches this page for its provider in its work.
    is_main_page: bool = False
    # The current page key, when the seed lists the page under an old address.
    renamed_to: str | None = None
    # The reference page of the seed entry an ``other work`` page joins; None
    # when the page starts a new entry.
    target_entry_reference_page: str | None = None
    # Whether the page is not in the seed and is added to the entry: a page the
    # entry's pages link that no seed entry holds.
    is_added_page: bool = False
    # Whether the page covers only part of its work, such as one season of a
    # run MyAnimeList lists whole, so enrichment combines it with the work's
    # other part pages rather than taking it for the whole work.
    is_part_page: bool = False
    # The name of the decision's reason, or ``review answer``.
    reason: str


class UnknownSeedEntryError(LookupError):
    """A correction names a seed entry the known works do not hold.

    Corrections are made from the same seed they are applied to, so this means
    they were made from another seed release.
    """

    def __init__(self, reference_page: str) -> None:
        """Name the reference page no seed entry has."""
        super().__init__(f"No seed entry has the reference page {reference_page}")


@dataclass(frozen=True)
class DoubledProviderCorrections:
    """The corrections for entries with doubled providers, and their review cases."""

    corrections: tuple[SeedCorrection, ...]
    # Every case of this run, with its kept answer, for the review file.
    review_cases: tuple[ReviewCase, ...]


@dataclass(frozen=True)
class CorrectedEntry:
    """A work entry after corrections, and the original seed entry it stands in for."""

    # The reference page of the original seed entry this corrected entry takes
    # the place of in the cleaned seed; None for a new entry, which takes the
    # place of nothing.
    seed_entry_reference_page: str | None
    entry: WorkEntry


class _GroupedCorrections(NamedTuple):
    """The corrections grouped by the entry each one changes or creates."""

    # Each seed entry's own corrections, by its reference page.
    corrections_by_reference_page: dict[str, list[SeedCorrection]]
    # The corrections of pages moving into each seed entry, by its reference page.
    moved_in_corrections_by_reference_page: dict[str, list[SeedCorrection]]
    # The corrections of each new entry, keyed by the seed entry the pages leave
    # and by what makes them one new entry: an ``other work`` page alone forms
    # one, keyed by its page key; the pages of one ``work N`` form one together,
    # keyed by ``work N``.
    new_entry_corrections: dict[tuple[str, str], list[SeedCorrection]]


def correct_doubled_provider_entries(
    entries: Iterable[WorkEntry],
    store: ProviderPageStore,
    known_work_entries: KnownWorkEntries,
    review_file: ReviewFile,
) -> DoubledProviderCorrections:
    """Place the pages of each entry with doubled providers, or ask a person.

    Args:
        entries: The seed entries with a doubled provider.
        store: The fetched pages, including pages searched by the entries' pages.
        known_work_entries: Every seed entry, to find the entries pages move to.
        review_file: The review file, for the answers kept in it.

    Returns:
        The corrections from settled entries and accepted answers, and every
        review case of this run.
    """
    corrections: list[SeedCorrection] = []
    review_cases: list[ReviewCase] = []
    for entry in entries:
        pages = store.get_entry_pages(entry)
        decisions = decide_doubled_providers(entry, pages, known_work_entries)
        placements = place_pages_without_review(entry, decisions, pages)
        reason_by_page_key = {
            key: decision.reason.name
            for decision in decisions
            for key in decision.page_keys
        }
        if placements is None:
            review_case = review_file.attach_kept_answer(
                ReviewCase.from_doubled_provider_entry(entry, decisions, pages)
            )
            review_cases.append(review_case)
            accepted_placements = review_case.accepted_answer
            if not isinstance(accepted_placements, dict):
                continue
            placements = accepted_placements
            reason_by_page_key = dict.fromkeys(
                accepted_placements, REVIEW_ANSWER_REASON
            )
        corrections.extend(
            _build_entry_corrections(
                entry, decisions, placements, reason_by_page_key, pages
            )
        )
    return DoubledProviderCorrections(tuple(corrections), tuple(review_cases))


def build_corrected_entries(
    corrections: Iterable[SeedCorrection],
    known_work_entries: KnownWorkEntries,
    store: ProviderPageStore,
) -> list[CorrectedEntry]:
    """Build every work entry the corrections change or create.

    Args:
        corrections: The seed corrections.
        known_work_entries: Every seed entry, to find the entries the corrections name.
        store: The fetched pages, for the details of new entries.

    Returns:
        The corrected seed entries, in the order the corrections first name
        them (pages moved in come after the entries they leave), then the new
        entries.
    """
    (
        corrections_by_reference_page,
        moved_in_corrections_by_reference_page,
        new_entry_corrections,
    ) = _group_corrections(corrections)
    changed_reference_pages = dict.fromkeys(
        [*corrections_by_reference_page, *moved_in_corrections_by_reference_page]
    )
    return [
        CorrectedEntry(
            seed_entry_reference_page=reference_page,
            entry=_correct_seed_entry(
                _find_seed_entry(known_work_entries, reference_page),
                corrections_by_reference_page[reference_page],
                moved_in_corrections_by_reference_page[reference_page],
            ),
        )
        for reference_page in changed_reference_pages
    ] + [
        CorrectedEntry(
            seed_entry_reference_page=None,
            entry=_build_new_entry(
                _find_seed_entry(known_work_entries, source_entry_reference_page),
                entry_corrections,
                store,
            ),
        )
        for (
            source_entry_reference_page,
            _,
        ), entry_corrections in new_entry_corrections.items()
    ]


def write_corrections(path: str | Path, corrections: Iterable[SeedCorrection]) -> None:
    """Write the corrections in place of the corrections file, one per line.

    Lines are sorted by entry and page, so the same corrections always give the
    same file and two runs can be compared line by line.
    """
    replace_file_lines(
        path,
        (
            correction.model_dump_json() + "\n"
            for correction in sorted(
                corrections,
                key=lambda correction: (
                    correction.entry_reference_page,
                    correction.page_key,
                ),
            )
        ),
    )


def _group_corrections(corrections: Iterable[SeedCorrection]) -> _GroupedCorrections:
    """Group the corrections by the entry each one changes or creates."""
    corrections_by_reference_page: defaultdict[str, list[SeedCorrection]] = defaultdict(
        list
    )
    moved_in_corrections_by_reference_page: defaultdict[str, list[SeedCorrection]] = (
        defaultdict(list)
    )
    new_entry_corrections: defaultdict[tuple[str, str], list[SeedCorrection]] = (
        defaultdict(list)
    )
    for correction in corrections:
        corrections_by_reference_page[correction.entry_reference_page].append(
            correction
        )
        if correction.target_entry_reference_page:
            moved_in_corrections_by_reference_page[
                correction.target_entry_reference_page
            ].append(correction)
        elif correction.placement_word == OTHER_WORK:
            new_entry_corrections[
                correction.entry_reference_page, correction.page_key
            ].append(correction)
        elif correction.placement_word != ENTRY_WORK:
            new_entry_corrections[
                correction.entry_reference_page, correction.placement_word
            ].append(correction)
    return _GroupedCorrections(
        corrections_by_reference_page,
        moved_in_corrections_by_reference_page,
        new_entry_corrections,
    )


def _build_entry_corrections(
    entry: WorkEntry,
    decisions: Sequence[DoubledProviderDecision],
    placements: Mapping[str, PagePlacement],
    reason_by_page_key: Mapping[str, str],
    pages: Iterable[ProviderPage],
) -> list[SeedCorrection]:
    """Turn one entry's page places into corrections, in the entry's page order.

    A doubled page listed under an old address takes the place of its current
    page, since the decisions name doubled pages by their current key. The old
    address is dropped only when the entry also lists the current one; otherwise
    dropping it would lose the page, so it keeps its address. Any other page is
    placed by the key the entry lists, as an answer names it. Pages without a
    place stay as they are and get no correction.
    """
    pages = list(pages)
    current_key_by_page_key = {
        page.key: page.renamed_to for page in pages if page.renamed_to
    }
    target_reference_page_by_page_key = _find_target_reference_pages(decisions, pages)
    corrections: list[SeedCorrection] = []
    for key, address in entry.addresses_by_page_key.items():
        placed_key_and_placement = _find_placement(
            key, current_key_by_page_key.get(key, key), placements
        )
        if placed_key_and_placement is None:
            continue
        current_key, placement = placed_key_and_placement
        corrections.append(
            SeedCorrection(
                entry_reference_page=entry.reference_page,
                page_key=key,
                address=address,
                placement_word=placement.placement_word,
                is_main_page=placement.is_main_page,
                renamed_to=(
                    current_key
                    if current_key != key and current_key in entry.addresses_by_page_key
                    else None
                ),
                target_entry_reference_page=(
                    target_reference_page_by_page_key.get(current_key)
                    if placement.placement_word == OTHER_WORK
                    else None
                ),
                reason=reason_by_page_key[current_key],
            )
        )
    return corrections


def _find_target_reference_pages(
    decisions: Sequence[DoubledProviderDecision], pages: Iterable[ProviderPage]
) -> dict[str, str]:
    """Return the reference page each unrelated page moves to, when the move is sure.

    The rules and review answers share one check (``find_move_target``), so a
    page an answer sends to another work never makes a doubled provider in
    the entry it joins; without a sure target it starts a new entry.
    """
    pages_by_current_key = index_pages_by_current_key(pages)
    return {
        key: target.reference_page
        for decision in decisions
        for key, unrelated_page in decision.unrelated_pages.items()
        if (
            target := find_move_target(
                key, unrelated_page, pages_by_current_key.get(key)
            )
        )
    }


def _find_placement(
    key: str, current_key: str, placements: Mapping[str, PagePlacement]
) -> tuple[str, PagePlacement] | None:
    """Return a page's place and the key it was found by, or None if it has none.

    Decisions name a doubled page by its current key, and answers name any
    other page by the key the entry lists, so both are tried.
    """
    for placed_key in (current_key, key):
        if placed_key in placements:
            return placed_key, placements[placed_key]
    return None


def _find_seed_entry(
    known_work_entries: KnownWorkEntries, reference_page: str
) -> WorkEntry:
    """Return the seed entry named by this reference page."""
    entry = known_work_entries.find_by_page_key(reference_page)
    if entry is None:
        raise UnknownSeedEntryError(reference_page)
    return entry


def _correct_seed_entry(
    entry: WorkEntry,
    corrections: Sequence[SeedCorrection],
    moved_in_corrections: Sequence[SeedCorrection],
) -> WorkEntry:
    """Return a seed entry as its corrections leave it.

    Args:
        entry: The seed entry.
        corrections: The corrections naming the seed entry: pages leaving it,
            staying in it, or added to it.
        moved_in_corrections: The corrections of pages moving into the entry.

    Returns:
        The entry without the pages that left it, with the pages moved in or
        added, and each provider's main page listed first.
    """
    leaving_page_keys = {
        correction.page_key
        for correction in corrections
        if correction.placement_word != ENTRY_WORK or correction.renamed_to
    }
    joining_corrections = [
        *moved_in_corrections,
        *(correction for correction in corrections if correction.is_added_page),
    ]
    addresses_by_page_key = {
        key: address
        for key, address in entry.addresses_by_page_key.items()
        if key not in leaving_page_keys
    } | {correction.page_key: correction.address for correction in joining_corrections}
    return entry.model_copy(
        update={
            "addresses_by_page_key": _order_main_pages_first(
                addresses_by_page_key, [*corrections, *moved_in_corrections]
            )
        }
    )


def _build_new_entry(
    source_seed_entry: WorkEntry,
    corrections: Sequence[SeedCorrection],
    store: ProviderPageStore,
) -> WorkEntry:
    """Build a new entry from the pages the corrections group into one work.

    Its title, type, year and season come from the most trusted page with
    fetched details; without one, the title of the entry the pages came from.
    """
    addresses_by_page_key = _order_main_pages_first(
        {correction.page_key: correction.address for correction in corrections},
        corrections,
    )
    trusted_page = next(
        (
            page
            for key in sorted(addresses_by_page_key, key=rank_by_provider_trust)
            if (page := store.get_page(key)) and page.titles and not page.is_dead
        ),
        None,
    )
    if trusted_page is None:
        logger.warning(
            f"New entry has no fetched page details, so it takes the title of "
            f"{source_seed_entry.reference_page} and an unknown type: "
            f"{list(addresses_by_page_key)}"
        )
        return WorkEntry(
            title=source_seed_entry.title,
            type="UNKNOWN",
            addresses_by_page_key=addresses_by_page_key,
        )
    # Validated, so the page's season goes through the entry's season check.
    return WorkEntry.model_validate(
        {
            "title": trusted_page.titles[0],
            "type": get_seed_type(trusted_page) or "UNKNOWN",
            "year": trusted_page.year,
            "season": trusted_page.season,
            "addresses_by_page_key": addresses_by_page_key,
        }
    )


def _order_main_pages_first(
    addresses_by_page_key: Mapping[str, str], corrections: Iterable[SeedCorrection]
) -> dict[str, str]:
    """Order the page addresses with main pages first, the rest in their given order.

    Enrichment fetches the first address of each provider, so a main page must
    come before the other pages of its provider.
    """
    main_page_keys = {
        correction.page_key for correction in corrections if correction.is_main_page
    }
    return {
        key: addresses_by_page_key[key]
        for key in addresses_by_page_key
        if key in main_page_keys
    } | dict(addresses_by_page_key)
