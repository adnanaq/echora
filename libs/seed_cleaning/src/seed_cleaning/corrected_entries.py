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
- Main page: a work's main page is listed before its other pages, because
  enrichment fetches the first address of each provider.
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
    get_seed_type,
    rank_by_provider_trust,
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
                _get_seed_entry(known_work_entries, reference_page),
                corrections_by_reference_page.get(reference_page, []),
                moved_in_corrections_by_reference_page.get(reference_page, []),
            ),
        )
        for reference_page in changed_reference_pages
    ] + [
        CorrectedEntry(
            seed_entry_reference_page=None,
            new_entry_key=new_entry_key,
            entry=_build_new_entry(
                _get_seed_entry(
                    known_work_entries,
                    min(
                        (
                            correction.entry_reference_page
                            for correction in entry_corrections
                        ),
                        key=lambda reference_page: (
                            rank_by_provider_trust(reference_page),
                            reference_page,
                        ),
                    ),
                ),
                entry_corrections,
                store,
            ),
        )
        for new_entry_key, entry_corrections in new_entry_corrections.items()
    ]


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
    for correction in corrections:
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
) -> WorkEntry | None:
    """Return a seed entry as its corrections leave it.

    A page leaves the entry when its correction moves it to a seed entry or a
    new entry, or when the entry also lists it under its current address.

    Args:
        entry: The seed entry.
        corrections: The corrections naming the seed entry: pages leaving it,
            staying in it, or added to it.
        moved_in_corrections: The corrections of pages moving into the entry.

    Returns:
        The entry without the pages that left it, with the pages moved in or
        added, and each provider's main page listed first; None when no page
        is left.
    """
    leaving_page_keys = {
        correction.page_key
        for correction in corrections
        if correction.target_entry_reference_page
        or correction.new_entry_key
        or correction.renamed_to
    }
    # An added page that also moves joins only the entry it moves to.
    joining_corrections = [
        *moved_in_corrections,
        *(
            correction
            for correction in corrections
            if correction.is_added_page
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
