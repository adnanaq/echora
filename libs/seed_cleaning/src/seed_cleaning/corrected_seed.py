"""Build the corrected seed holding every work entry as the first pass leaves it.

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

from seed_cleaning.corrected_entries import build_corrected_entries
from seed_cleaning.models import (
    CorrectedSeed,
    EntryIdentity,
    PageOrigin,
    SeedCorrection,
)
from work_matching import KnownWorkEntriesFromSeed, ProviderPageStore, WorkEntry


class UnknownPageOriginError(LookupError):
    """A page of a new entry has no correction bringing it in."""

    def __init__(self, page_key: str, identity: EntryIdentity) -> None:
        """Name the page and the new entry holding it.

        Args:
            page_key: The page key with no joining correction.
            identity: The identity of the new entry holding the page.
        """
        super().__init__(
            f"Page {page_key} of new entry {identity.new_entry_key} has no "
            "correction bringing it in"
        )


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
        corrected_entry.seed_entry_reference_page: corrected_entry.entry
        for corrected_entry in corrected_entries
        if corrected_entry.seed_entry_reference_page is not None
    }
    identified_entries: list[tuple[WorkEntry, EntryIdentity]] = []
    for seed_entry in known_work_entries:
        reference_page = seed_entry.reference_page_key
        entry = corrected_entry_by_seed_reference_page.get(reference_page, seed_entry)
        if entry is not None:
            identified_entries.append((entry, EntryIdentity(reference_page, None)))
    identified_entries.extend(
        (corrected_entry.entry, EntryIdentity(None, corrected_entry.new_entry_key))
        for corrected_entry in corrected_entries
        if corrected_entry.seed_entry_reference_page is None
        and corrected_entry.entry is not None
    )
    joining_corrections = _index_joining_corrections(corrections)
    return CorrectedSeed(
        KnownWorkEntriesFromSeed.index_work_entries(
            entry for entry, _ in identified_entries
        ),
        {entry.reference_page_key: identity for entry, identity in identified_entries},
        {
            (entry.reference_page_key, page_key): _find_origin(
                identity, page_key, address, joining_corrections
            )
            for entry, identity in identified_entries
            for page_key, address in entry.addresses_by_page_key.items()
        },
    )


def find_move_destination(correction: SeedCorrection) -> EntryIdentity | None:
    """Find the entry a correction moves its page to, as corrections name entries.

    Args:
        correction: A seed correction.

    Returns:
        The seed entry or the new entry the page moves to; None when the page
        stays in the entry listing it.
    """
    if correction.target_entry_reference_page:
        return EntryIdentity(correction.target_entry_reference_page, None)
    if correction.new_entry_key:
        return EntryIdentity(None, correction.new_entry_key)
    return None


def _index_joining_corrections(
    corrections: Iterable[SeedCorrection],
) -> dict[tuple[EntryIdentity, str], SeedCorrection]:
    """Index each correction bringing a page into an entry by that entry and page.

    A page joins the entry its correction moves it to, or, when added without
    a move, the entry it is added to.

    Args:
        corrections: The first pass's seed corrections.

    Returns:
        Each joining correction, by the entry the page joins and the page key.
    """
    joining_corrections: dict[tuple[EntryIdentity, str], SeedCorrection] = {}
    for correction in corrections:
        destination = find_move_destination(correction)
        if destination is None and correction.is_added_page:
            destination = EntryIdentity(correction.entry_reference_page, None)
        if destination is not None:
            joining_corrections[destination, correction.page_key] = correction
    return joining_corrections


def _find_origin(
    identity: EntryIdentity,
    page_key: str,
    address: str,
    joining_corrections: dict[tuple[EntryIdentity, str], SeedCorrection],
) -> PageOrigin:
    """Return a page's origin: the correction that brought it in, else its own entry.

    Args:
        identity: The identity of the entry holding the page.
        page_key: The page key within the entry.
        address: The page's address on the entry.
        joining_corrections: Corrections indexed by destination entry identity
            and page key.

    Returns:
        The page's origin in the downloaded seed.

    Raises:
        UnknownPageOriginError: A page of a new entry has no joining correction.
    """
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


def translate_corrected_seed_corrections(
    corrections: Iterable[SeedCorrection],
    corrected_seed: CorrectedSeed,
) -> tuple[SeedCorrection, ...]:
    """Translate corrections written against a corrected seed to the downloaded seed.

    Args:
        corrections: Corrections produced by placing groups of ``corrected_seed``.
        corrected_seed: The corrected seed holding each page's origin and each
            entry's identity in the downloaded seed.

    Returns:
        The corrections rewritten to name pages by their origin and entries by
        their downloaded-seed identity.
    """
    return tuple(
        _translate_correction(correction, corrected_seed) for correction in corrections
    )


def _translate_correction(
    correction: SeedCorrection,
    corrected_seed: CorrectedSeed,
) -> SeedCorrection:
    """Translate one correction from corrected-seed names to downloaded-seed names.

    Args:
        correction: A correction written against an entry of ``corrected_seed``.
        corrected_seed: The corrected seed holding each page's origin and each
            entry's identity in the downloaded seed.

    Returns:
        The correction rewritten with downloaded-seed origin and destination names.
    """
    origin = corrected_seed.find_page_origin(
        correction.entry_reference_page, correction.page_key
    )
    target_entry_reference_page, new_entry_key = _translate_primary_destination(
        correction, origin, corrected_seed
    )
    partial_entry_reference_pages, partial_new_entry_keys = (
        _translate_partial_destinations(correction, corrected_seed)
    )
    return correction.model_copy(
        update={
            "entry_reference_page": origin.entry_reference_page,
            "address": origin.address,
            "is_added_page": origin.is_added_page,
            "target_entry_reference_page": target_entry_reference_page,
            "new_entry_key": new_entry_key,
            "partial_entry_reference_pages": partial_entry_reference_pages,
            "partial_new_entry_keys": partial_new_entry_keys,
        }
    )


def _translate_primary_destination(
    correction: SeedCorrection,
    origin: PageOrigin,
    corrected_seed: CorrectedSeed,
) -> tuple[str | None, str | None]:
    """Translate a correction's primary destination to downloaded-seed names.

    Args:
        correction: The second-pass correction to translate.
        origin: The page's downloaded-seed origin.
        corrected_seed: The corrected seed holding each entry's identity.

    Returns:
        A ``(target_entry_reference_page, new_entry_key)`` pair, both None when
        dropped or staying in ``origin.entry_reference_page``.
    """
    if correction.is_dropped_page:
        return None, None
    if correction.target_entry_reference_page is not None:
        identity = corrected_seed.identity_by_reference_page[
            correction.target_entry_reference_page
        ]
        destination = (identity.seed_entry_reference_page, identity.new_entry_key)
    elif correction.new_entry_key is not None:
        destination = (None, correction.new_entry_key)
    else:
        identity = corrected_seed.identity_by_reference_page[
            correction.entry_reference_page
        ]
        destination = (identity.seed_entry_reference_page, identity.new_entry_key)
    if destination == (origin.entry_reference_page, None):
        return None, None
    return destination


def _translate_partial_destinations(
    correction: SeedCorrection,
    corrected_seed: CorrectedSeed,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Translate a correction's partial destinations to downloaded-seed names.

    Args:
        correction: The second-pass correction whose partial destinations are
            translated.
        corrected_seed: The corrected seed holding each entry's identity.

    Returns:
        The translated ``(partial_entry_reference_pages, partial_new_entry_keys)``
        tuples.
    """
    partial_entry_reference_pages: list[str] = []
    partial_new_entry_keys = list(correction.partial_new_entry_keys)
    for reference_page in correction.partial_entry_reference_pages:
        identity = corrected_seed.identity_by_reference_page[reference_page]
        if identity.seed_entry_reference_page is not None:
            partial_entry_reference_pages.append(identity.seed_entry_reference_page)
        if identity.new_entry_key is not None:
            partial_new_entry_keys.append(identity.new_entry_key)
    return tuple(partial_entry_reference_pages), tuple(partial_new_entry_keys)
