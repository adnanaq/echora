"""Corrected entries: the work entries the seed corrections change or create.

The cleaned seed is the downloaded release plus the seed corrections. This
module rebuilds each entry the corrections touch:

- Moves: a page placed in ``other work`` joins the seed entry its sure move
  names (``UnrelatedPage.move_target_entry``); otherwise it starts a new entry.
  Pages a review answer puts in the same ``work N`` form one new entry.
- Renamed pages: an old address is dropped, because the entry also lists the
  page under its current address.
- Added pages: a page the corrections add joins the entry they add it to,
  unless its correction also moves it; then it joins only the entry it moves
  to, as a page the seed lists would.
- Partial sources: a page a correction makes a partial source of other
  entries joins their partial sources, besides its own place. A partial source
  never stands among an entry's own pages.
- Dropped pages: a dropped page leaves its entry and joins no other, like a
  dead page.
- New entries take their title, type, year and season from their most trusted
  fetched page. Pages with no fetched details only have the entries they came
  from to go by, so the new entry takes the title of the most trusted of those
  entries, and an unknown type. Choosing by trust, not by correction order,
  gives the same title whatever order the corrections come in.
"""

import logging
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple

from enrichment.pipeline.link_rules import find_anime_id_page_key
from seed_cleaning.seed_corrections import SeedCorrection
from work_matching import (
    KnownWorkEntries,
    ProviderPageStore,
    WorkEntry,
    rank_by_provider_trust,
    resolve_seed_type,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CorrectedEntry:
    """A work entry after corrections, and the original seed entry it stands in for."""

    # The reference page of the original seed entry this corrected entry takes
    # the place of in the cleaned seed; None for a new entry, which takes the
    # place of nothing.
    seed_entry_reference_page: str | None
    # None when every page of the seed entry left it: the entry is removed from
    # the cleaned seed, since an entry without pages is no work.
    entry: WorkEntry | None
    # The key the corrections give a new entry's pages; None for a seed entry.
    new_entry_key: str | None = None


class UnknownSeedEntryError(LookupError):
    """A correction names a seed entry the known works do not hold.

    Corrections are made from the same seed they are applied to, so this means
    they were made from another seed release.
    """

    def __init__(self, reference_page: str) -> None:
        """Name the reference page no seed entry has."""
        super().__init__(f"No seed entry has the reference page {reference_page}")


class _GroupedCorrections(NamedTuple):
    """The corrections grouped by the entry each one changes or creates."""

    # Each seed entry's own corrections, by its reference page.
    corrections_by_reference_page: dict[str, list[SeedCorrection]]
    # The corrections of pages moving into each seed entry, by its reference page.
    moved_in_corrections_by_reference_page: dict[str, list[SeedCorrection]]
    # The corrections of each new entry, by the new entry key they share.
    new_entry_corrections: dict[str, list[SeedCorrection]]
    # The corrections of pages joining each seed entry as a partial source, by
    # its reference page.
    partial_source_corrections_by_reference_page: dict[str, list[SeedCorrection]]
    # The corrections of pages joining each new entry as a partial source, by
    # its new entry key.
    partial_source_corrections_by_new_entry_key: dict[str, list[SeedCorrection]]


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
        entries. A seed entry every page left comes back with no entry.
    """
    grouped_corrections = _group_corrections(corrections)
    return _build_corrected_seed_entries(
        grouped_corrections, known_work_entries
    ) + _build_new_entries(grouped_corrections, known_work_entries, store)


def _build_corrected_seed_entries(
    grouped_corrections: _GroupedCorrections, known_work_entries: KnownWorkEntries
) -> list[CorrectedEntry]:
    """Build each seed entry the corrections change, with the pages moved in.

    Args:
        grouped_corrections: The corrections, grouped by the entry each changes.
        known_work_entries: Every seed entry, to find the entries the corrections name.

    Returns:
        The corrected seed entries, in the order the corrections first name
        them. A seed entry every page left comes back with no entry.
    """
    changed_reference_pages = dict.fromkeys(
        [
            *grouped_corrections.corrections_by_reference_page,
            *grouped_corrections.moved_in_corrections_by_reference_page,
            *grouped_corrections.partial_source_corrections_by_reference_page,
        ]
    )
    return [
        CorrectedEntry(
            seed_entry_reference_page=reference_page,
            entry=_correct_seed_entry(
                _get_seed_entry(known_work_entries, reference_page),
                grouped_corrections.corrections_by_reference_page.get(
                    reference_page, []
                ),
                grouped_corrections.moved_in_corrections_by_reference_page.get(
                    reference_page, []
                ),
                grouped_corrections.partial_source_corrections_by_reference_page.get(
                    reference_page, []
                ),
            ),
        )
        for reference_page in changed_reference_pages
    ]


def _build_new_entries(
    grouped_corrections: _GroupedCorrections,
    known_work_entries: KnownWorkEntries,
    store: ProviderPageStore,
) -> list[CorrectedEntry]:
    """Build each new entry the corrections create.

    Args:
        grouped_corrections: The corrections, grouped by the entry each changes.
        known_work_entries: Every seed entry, to find the entries the pages leave.
        store: The fetched pages, for the details of new entries.

    Returns:
        One entry per new entry key, in the order the corrections first name it.
    """
    return [
        CorrectedEntry(
            seed_entry_reference_page=None,
            new_entry_key=new_entry_key,
            entry=_build_new_entry(
                _get_seed_entry(
                    known_work_entries,
                    _find_most_trusted_reference_page(entry_corrections),
                ),
                entry_corrections,
                grouped_corrections.partial_source_corrections_by_new_entry_key.get(
                    new_entry_key, []
                ),
                store,
            ),
        )
        for new_entry_key, entry_corrections in (
            grouped_corrections.new_entry_corrections.items()
        )
    ]


def _find_most_trusted_reference_page(corrections: Iterable[SeedCorrection]) -> str:
    """Find the most trusted seed entry the pages of a new entry leave.

    The new entry takes its details from that seed entry, as from the entry
    most likely to describe the work.

    Args:
        corrections: The corrections moving pages into one new entry.

    Returns:
        The reference page of the most trusted seed entry, by provider trust,
        then by the reference page itself for a stable order.
    """
    return min(
        (correction.entry_reference_page for correction in corrections),
        key=lambda reference_page: (
            rank_by_provider_trust(reference_page),
            reference_page,
        ),
    )


def mark_anime_id_pages(
    corrections: Iterable[SeedCorrection],
    corrected_entries: Iterable[CorrectedEntry],
) -> list[SeedCorrection]:
    """Mark each added page the anime id of its seed entry now comes from.

    Enrichment derives a work's anime id from the work's pages, MAL first
    (``find_anime_id_page_key``). When the id comes from an added page, the
    search entry stored under the seed entry's old id must be replaced.

    - The rule reads the entries as both passes' corrections leave them. The
      second pass can move an added page, or move another entry's MAL page into
      the added page's entry, so the first pass's entries can name the wrong page.
    - A page in a new entry is never marked. A new entry takes the place of no
      seed entry, so no search entry is stored under an old id.

    Args:
        corrections: Both passes' corrections, merged.
        corrected_entries: The entries those corrections change or create.

    Returns:
        The corrections in the order given, each added page's
        ``is_anime_id_page`` set by the rules above.
    """
    anime_id_page_key_by_seed_reference_page = {
        corrected.seed_entry_reference_page: find_anime_id_page_key(
            corrected.entry.addresses_by_page_key
        )
        for corrected in corrected_entries
        if corrected.seed_entry_reference_page is not None
        and corrected.entry is not None
    }
    return [
        correction.model_copy(
            update={
                "is_anime_id_page": correction.new_entry_key is None
                and anime_id_page_key_by_seed_reference_page.get(
                    correction.target_entry_reference_page
                    or correction.entry_reference_page
                )
                == correction.page_key
            }
        )
        if correction.is_added_page
        else correction
        for correction in corrections
    ]


def _group_corrections(corrections: Iterable[SeedCorrection]) -> _GroupedCorrections:
    """Group the corrections by the entry each one changes or creates."""
    corrections_by_reference_page: defaultdict[str, list[SeedCorrection]] = defaultdict(
        list
    )
    moved_in_corrections_by_reference_page: defaultdict[str, list[SeedCorrection]] = (
        defaultdict(list)
    )
    new_entry_corrections: defaultdict[str, list[SeedCorrection]] = defaultdict(list)
    partial_source_corrections_by_reference_page: defaultdict[
        str, list[SeedCorrection]
    ] = defaultdict(list)
    partial_source_corrections_by_new_entry_key: defaultdict[
        str, list[SeedCorrection]
    ] = defaultdict(list)
    for correction in corrections:
        for reference_page in correction.partial_entry_reference_pages:
            partial_source_corrections_by_reference_page[reference_page].append(
                correction
            )
        for new_entry_key in correction.partial_new_entry_keys:
            partial_source_corrections_by_new_entry_key[new_entry_key].append(
                correction
            )
        corrections_by_reference_page[correction.entry_reference_page].append(
            correction
        )
        if correction.target_entry_reference_page:
            moved_in_corrections_by_reference_page[
                correction.target_entry_reference_page
            ].append(correction)
        elif correction.new_entry_key:
            new_entry_corrections[correction.new_entry_key].append(correction)
    return _GroupedCorrections(
        corrections_by_reference_page,
        moved_in_corrections_by_reference_page,
        new_entry_corrections,
        partial_source_corrections_by_reference_page,
        partial_source_corrections_by_new_entry_key,
    )


def _get_seed_entry(
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
    partial_source_corrections: Sequence[SeedCorrection],
) -> WorkEntry | None:
    """Return a seed entry as its corrections leave it.

    A page leaves the entry when its correction moves it to a seed entry or a
    new entry, when the entry also lists it under its current address, or when
    its provider says the page does not exist or an answer drops it.

    Args:
        entry: The seed entry.
        corrections: The corrections naming the seed entry: pages leaving it,
            staying in it, or added to it.
        moved_in_corrections: The corrections of pages moving into the entry.
        partial_source_corrections: The corrections of pages joining the
            entry as a partial source.

    Returns:
        The entry without the pages that left it, with the pages moved in or
        added, and with its partial sources; None when no page is left.
    """
    leaving_page_keys = {
        correction.page_key
        for correction in corrections
        if correction.target_entry_reference_page
        or correction.new_entry_key
        or correction.renamed_to
        or correction.is_dead_page
        or correction.is_dropped_page
    }
    # An added page that also moves joins only the entry it moves to, and a page
    # added only as a partial source joins no entry's own pages.
    joining_corrections = [
        *moved_in_corrections,
        *(
            correction
            for correction in corrections
            if correction.is_added_page
            and correction.placement_word is not None
            and not correction.target_entry_reference_page
            and not correction.new_entry_key
        ),
    ]
    addresses_by_page_key = {
        key: address
        for key, address in entry.addresses_by_page_key.items()
        if key not in leaving_page_keys
    } | {correction.page_key: correction.address for correction in joining_corrections}
    if not addresses_by_page_key:
        return None
    return entry.model_copy(
        update={
            "addresses_by_page_key": addresses_by_page_key,
            "partial_addresses_by_page_key": _build_partial_addresses(
                entry.partial_addresses_by_page_key,
                partial_source_corrections,
                addresses_by_page_key,
            ),
        }
    )


def _build_new_entry(
    source_seed_entry: WorkEntry,
    corrections: Sequence[SeedCorrection],
    partial_source_corrections: Sequence[SeedCorrection],
    store: ProviderPageStore,
) -> WorkEntry:
    """Build a new entry from the pages the corrections group into one work.

    Its title, type, year and season come from the most trusted page with
    fetched details; without one, the title of the entry the pages came from.
    """
    addresses_by_page_key = {
        correction.page_key: correction.address for correction in corrections
    }
    partial_addresses_by_page_key = _build_partial_addresses(
        {}, partial_source_corrections, addresses_by_page_key
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
            f"{source_seed_entry.reference_page_key} and an unknown type: "
            f"{list(addresses_by_page_key)}"
        )
        return WorkEntry(
            title=source_seed_entry.title,
            type="UNKNOWN",
            addresses_by_page_key=addresses_by_page_key,
            partial_addresses_by_page_key=partial_addresses_by_page_key,
        )
    # Validated, so the page's season goes through the entry's season check.
    return WorkEntry.model_validate(
        {
            "title": trusted_page.titles[0],
            "type": resolve_seed_type(trusted_page) or "UNKNOWN",
            "year": trusted_page.year,
            "season": trusted_page.season,
            "addresses_by_page_key": addresses_by_page_key,
            "partial_addresses_by_page_key": partial_addresses_by_page_key,
        }
    )


def _build_partial_addresses(
    partial_addresses_by_page_key: Mapping[str, str],
    partial_source_corrections: Iterable[SeedCorrection],
    addresses_by_page_key: Mapping[str, str],
) -> dict[str, str]:
    """Return an entry's partial sources after corrections add pages to them.

    Args:
        partial_addresses_by_page_key: The entry's partial sources before.
        partial_source_corrections: The corrections of pages joining the entry
            as a partial source.
        addresses_by_page_key: The entry's own pages after corrections, which
            are never also partial sources.

    Returns:
        The partial sources by page key, the earlier ones first.
    """
    return {
        page_key: address
        for page_key, address in (
            *partial_addresses_by_page_key.items(),
            *(
                (correction.page_key, correction.address)
                for correction in partial_source_corrections
            ),
        )
        if page_key not in addresses_by_page_key
    }
