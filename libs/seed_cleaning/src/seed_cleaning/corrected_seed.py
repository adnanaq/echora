"""The corrected seed: every work entry as the first pass's corrections leave it.

Split works are looked for in a second pass, over the entries the first pass
leaves (doubled providers resolved, linked pages added), so a page moved or
added in the first pass is grouped where it now is. Corrections still name the
downloaded seed, though, since the cleaned seed is built from the release plus
corrections. So the corrected seed keeps two things for the second pass:

Each item reads ``<kept>: <why>``.

- Each entry's identity, the seed entry it stands in for or the new entry key
  its pages share: a second-pass correction moving a page into the entry names
  it this way.
- Each page's origin, the seed entry listing it (or the entry it was added
  to), its address, and whether it was added: a second-pass correction names
  the page this way, and an added page stays marked added wherever it goes.

Entries keep the seed's order; corrected entries take their seed entry's place,
entries left without pages are dropped, and new entries come last.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import NamedTuple

from seed_cleaning.corrected_entries import build_corrected_entries
from seed_cleaning.seed_corrections import SeedCorrection
from work_matching import KnownWorkEntriesFromSeed, ProviderPageStore, WorkEntry


class EntryIdentity(NamedTuple):
    """How seed corrections name an entry: by its seed entry or its new entry key."""

    # The reference page of the seed entry the entry stands in for; None for a
    # new entry.
    seed_entry_reference_page: str | None
    # The key a new entry's pages share in their corrections; None for a seed
    # entry.
    new_entry_key: str | None


class PageOrigin(NamedTuple):
    """How seed corrections name a page: the entry listing it and its address."""

    # The seed entry listing the page, or the seed entry it was added to.
    entry_reference_page: str
    address: str
    is_added_page: bool = False


@dataclass(frozen=True)
class CorrectedSeed:
    """Every work entry after the first pass, and how corrections name each one."""

    known_work_entries: KnownWorkEntriesFromSeed
    # Each entry's identity, by the entry's reference page.
    identity_by_reference_page: dict[str, EntryIdentity]
    # Each page's origin, by the entry's reference page and the page key.
    origin_by_entry_page: dict[tuple[str, str], PageOrigin]

    def find_page_origin(self, entry_reference_page: str, page_key: str) -> PageOrigin:
        """Return the origin of one page of one entry of the corrected seed."""
        return self.origin_by_entry_page[entry_reference_page, page_key]


def build_corrected_seed(
    corrections: Iterable[SeedCorrection],
    known_work_entries: KnownWorkEntriesFromSeed,
    store: ProviderPageStore,
) -> CorrectedSeed:
    """Build every work entry as the corrections leave it, with names for corrections.

    Args:
        corrections: The first pass's seed corrections.
        known_work_entries: Every seed entry.
        store: The fetched pages, for the details of new entries.

    Returns:
        The corrected seed, in the order the module docstring gives.
    """
    corrections = list(corrections)
    corrected_entries = build_corrected_entries(corrections, known_work_entries, store)
    corrected_entry_by_seed_reference_page = {
        corrected.seed_entry_reference_page: corrected.entry
        for corrected in corrected_entries
        if corrected.seed_entry_reference_page is not None
    }
    identified_entries: list[tuple[WorkEntry, EntryIdentity]] = []
    for seed_entry in known_work_entries:
        reference_page = seed_entry.reference_page
        entry = corrected_entry_by_seed_reference_page.get(reference_page, seed_entry)
        if entry is not None:
            identified_entries.append((entry, EntryIdentity(reference_page, None)))
    identified_entries.extend(
        (corrected.entry, EntryIdentity(None, corrected.new_entry_key))
        for corrected in corrected_entries
        if corrected.seed_entry_reference_page is None and corrected.entry is not None
    )
    joining_corrections = _index_joining_corrections(corrections)
    return CorrectedSeed(
        KnownWorkEntriesFromSeed.from_work_entries(
            entry for entry, _ in identified_entries
        ),
        {entry.reference_page: identity for entry, identity in identified_entries},
        {
            (entry.reference_page, page_key): _find_origin(
                identity, page_key, address, joining_corrections
            )
            for entry, identity in identified_entries
            for page_key, address in entry.addresses_by_page_key.items()
        },
    )


def _index_joining_corrections(
    corrections: Iterable[SeedCorrection],
) -> dict[tuple[EntryIdentity, str], SeedCorrection]:
    """Index each correction bringing a page into an entry by that entry and page.

    A page joins the entry its correction moves it to, or, when added without
    a move, the entry it is added to.
    """
    joining_corrections: dict[tuple[EntryIdentity, str], SeedCorrection] = {}
    for correction in corrections:
        if correction.target_entry_reference_page:
            destination = EntryIdentity(correction.target_entry_reference_page, None)
        elif correction.new_entry_key:
            destination = EntryIdentity(None, correction.new_entry_key)
        elif correction.is_added_page:
            destination = EntryIdentity(correction.entry_reference_page, None)
        else:
            continue
        joining_corrections[destination, correction.page_key] = correction
    return joining_corrections


def _find_origin(
    identity: EntryIdentity,
    page_key: str,
    address: str,
    joining_corrections: dict[tuple[EntryIdentity, str], SeedCorrection],
) -> PageOrigin:
    """Return a page's origin: the correction that brought it in, else its own entry."""
    joining_correction = joining_corrections.get((identity, page_key))
    if joining_correction is not None:
        return PageOrigin(
            joining_correction.entry_reference_page,
            joining_correction.address,
            joining_correction.is_added_page,
        )
    if identity.seed_entry_reference_page is None:
        raise UnknownPageOriginError(page_key, identity)
    return PageOrigin(identity.seed_entry_reference_page, address)


class UnknownPageOriginError(LookupError):
    """A page of a new entry has no correction bringing it in."""

    def __init__(self, page_key: str, identity: EntryIdentity) -> None:
        """Name the page and the new entry holding it."""
        super().__init__(
            f"Page {page_key} of new entry {identity.new_entry_key} has no "
            "correction bringing it in"
        )
