"""Seed corrections: where each page of an entry with doubled providers ends up.

For every entry with a doubled provider, the work matching rules either place
its pages themselves or leave the entry to a person as one review case. A
correction records one page's final place, from the rules or from an accepted
answer. The cleaned seed is built from the downloaded release plus these
corrections, so the release itself is never edited. The rules:

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

from pydantic import BaseModel, ConfigDict
from work_matching import (
    ENTRY_WORK,
    OTHER_WORK,
    DoubledProviderDecision,
    KnownWorks,
    PagePlacement,
    ProviderPage,
    ProviderPageStore,
    ReviewCase,
    ReviewFile,
    WorkEntry,
    decide_doubled_providers,
    get_seed_type,
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
    work: str
    # Whether enrichment fetches this page for its provider in its work.
    is_main_page: bool = False
    # The current page key, when the seed lists the page under an old address.
    renamed_to: str | None = None
    # The reference page of the seed entry an ``other work`` page joins; None
    # when the page starts a new entry.
    target_entry_reference_page: str | None = None
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


def correct_doubled_provider_entries(
    entries: Iterable[WorkEntry],
    store: ProviderPageStore,
    known_works: KnownWorks,
    review_file: ReviewFile,
) -> DoubledProviderCorrections:
    """Place the pages of each entry with doubled providers, or ask a person.

    Args:
        entries: The seed entries with a doubled provider.
        store: The fetched pages, including pages found through the entries'.
        known_works: Every seed entry, to find the entries pages move to.
        review_file: The review file, for the answers kept in it.

    Returns:
        The corrections from settled entries and accepted answers, and every
        review case of this run.
    """
    corrections: list[SeedCorrection] = []
    review_cases: list[ReviewCase] = []
    for entry in entries:
        pages = _load_entry_pages(entry, store)
        decisions = decide_doubled_providers(entry, pages, known_works)
        placements = place_pages_without_review(entry, decisions, pages)
        reason_by_page = {
            key: decision.reason.name
            for decision in decisions
            for key in decision.pages
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
            reason_by_page = dict.fromkeys(accepted_placements, REVIEW_ANSWER_REASON)
        corrections.extend(
            _build_entry_corrections(
                entry, decisions, placements, reason_by_page, pages
            )
        )
    return DoubledProviderCorrections(tuple(corrections), tuple(review_cases))


def build_corrected_entries(
    corrections: Iterable[SeedCorrection],
    known_works: KnownWorks,
    store: ProviderPageStore,
) -> list[CorrectedEntry]:
    """Build every work entry the corrections change or create.

    Args:
        corrections: The seed corrections.
        known_works: Every seed entry, to find the entries the corrections name.
        store: The fetched pages, for the details of new entries.

    Returns:
        The corrected seed entries, in the order the corrections first name
        them (pages moved in come after the entries they leave), then the new
        entries.
    """
    corrections_by_entry: defaultdict[str, list[SeedCorrection]] = defaultdict(list)
    moved_in_corrections_by_entry: defaultdict[str, list[SeedCorrection]] = defaultdict(
        list
    )
    # The corrections of each new entry, keyed by the seed entry the pages leave
    # and by what makes them one new entry: an ``other work`` page alone forms
    # one, keyed by its page key; the pages of one ``work N`` form one together,
    # keyed by ``work N``.
    new_entry_corrections: defaultdict[tuple[str, str], list[SeedCorrection]] = (
        defaultdict(list)
    )
    for correction in corrections:
        corrections_by_entry[correction.entry_reference_page].append(correction)
        if correction.target_entry_reference_page:
            moved_in_corrections_by_entry[
                correction.target_entry_reference_page
            ].append(correction)
        elif correction.work == OTHER_WORK:
            new_entry_corrections[
                correction.entry_reference_page, correction.page_key
            ].append(correction)
        elif correction.work != ENTRY_WORK:
            new_entry_corrections[
                correction.entry_reference_page, correction.work
            ].append(correction)
    changed_reference_pages = dict.fromkeys(
        [*corrections_by_entry, *moved_in_corrections_by_entry]
    )
    return [
        CorrectedEntry(
            seed_entry_reference_page=reference_page,
            entry=_correct_seed_entry(
                _find_seed_entry(known_works, reference_page),
                corrections_by_entry[reference_page],
                moved_in_corrections_by_entry[reference_page],
            ),
        )
        for reference_page in changed_reference_pages
    ] + [
        CorrectedEntry(
            seed_entry_reference_page=None,
            entry=_build_new_entry(
                _find_seed_entry(known_works, source_entry_reference_page),
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


def _load_entry_pages(entry: WorkEntry, store: ProviderPageStore) -> list[ProviderPage]:
    """Return the stored details of an entry's pages and of pages found through them."""
    return [
        *(page for key in entry.pages if (page := store.get_page(key))),
        *(
            found_page
            for key in entry.pages
            for found_page in store.get_pages_found_by(key)
        ),
    ]


def _build_entry_corrections(
    entry: WorkEntry,
    decisions: Sequence[DoubledProviderDecision],
    placements: Mapping[str, PagePlacement],
    reason_by_page: Mapping[str, str],
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
    current_key_by_page = {
        page.key: page.renamed_to for page in pages if page.renamed_to
    }
    target_entry_by_page = {
        key: unrelated_page.other_works[0].reference_page
        for decision in decisions
        for key, unrelated_page in decision.unrelated_pages.items()
        if len(unrelated_page.other_works) == 1
    }
    corrections: list[SeedCorrection] = []
    for key, address in entry.pages.items():
        current_key = current_key_by_page.get(key, key)
        placement = placements.get(current_key)
        if placement is None:
            current_key = key
            placement = placements.get(key)
        if placement is None:
            continue
        corrections.append(
            SeedCorrection(
                entry_reference_page=entry.reference_page,
                page_key=key,
                address=address,
                work=placement.work,
                is_main_page=placement.is_main_page,
                renamed_to=(
                    current_key
                    if current_key != key and current_key in entry.pages
                    else None
                ),
                target_entry_reference_page=(
                    target_entry_by_page.get(current_key)
                    if placement.work == OTHER_WORK
                    else None
                ),
                reason=reason_by_page[current_key],
            )
        )
    return corrections


def _find_seed_entry(known_works: KnownWorks, reference_page: str) -> WorkEntry:
    """Return the seed entry named by this reference page."""
    entry = known_works.find_by_page_key(reference_page)
    if entry is None:
        raise UnknownSeedEntryError(reference_page)
    return entry


def _correct_seed_entry(
    entry: WorkEntry,
    corrections: Sequence[SeedCorrection],
    moved_in_corrections: Sequence[SeedCorrection],
) -> WorkEntry:
    """Return a seed entry without the pages that left it, with those moved in."""
    leaving_pages = {
        correction.page_key
        for correction in corrections
        if correction.work != ENTRY_WORK or correction.renamed_to
    }
    pages = {
        key: address for key, address in entry.pages.items() if key not in leaving_pages
    } | {correction.page_key: correction.address for correction in moved_in_corrections}
    return entry.model_copy(
        update={
            "pages": _order_main_pages_first(
                pages, [*corrections, *moved_in_corrections]
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
    pages = _order_main_pages_first(
        {correction.page_key: correction.address for correction in corrections},
        corrections,
    )
    trusted_page = next(
        (
            page
            for key in sorted(pages, key=rank_by_provider_trust)
            if (page := store.get_page(key)) and page.titles and not page.is_dead
        ),
        None,
    )
    if trusted_page is None:
        logger.warning(
            f"New entry has no fetched page details, so it takes the title of "
            f"{source_seed_entry.reference_page} and an unknown type: {list(pages)}"
        )
        return WorkEntry(title=source_seed_entry.title, type="UNKNOWN", pages=pages)
    # Validated, so the page's season goes through the entry's season check.
    return WorkEntry.model_validate(
        {
            "title": trusted_page.titles[0],
            "type": get_seed_type(trusted_page) or "UNKNOWN",
            "year": trusted_page.year,
            "season": trusted_page.season,
            "pages": pages,
        }
    )


def _order_main_pages_first(
    pages: Mapping[str, str], corrections: Iterable[SeedCorrection]
) -> dict[str, str]:
    """Order the pages with main pages first, the rest in their given order.

    Enrichment fetches the first address of each provider, so a main page must
    come before the other pages of its provider.
    """
    main_pages = {
        correction.page_key for correction in corrections if correction.is_main_page
    }
    return {key: pages[key] for key in pages if key in main_pages} | dict(pages)
