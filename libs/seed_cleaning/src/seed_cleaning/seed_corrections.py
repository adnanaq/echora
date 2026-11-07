"""Seed corrections: where each checked page ends up, written to one file.

A correction records one page's final place, from the rules or from an accepted
answer (``group_placement``); a correction can also add a page the seed lacks
(``linked_page_additions``). The cleaned seed is built from the downloaded
release plus these corrections, so the release itself is never edited. The
rules:

- Final places, not steps: each correction says where its page ends up, so the
  order corrections are applied in never matters, even when a page moves into
  an entry that is itself being corrected.
- Rebuilt every run: corrections come from this run's decisions and the kept
  answers, so the file always matches them and needs no clean-up.
- Part pages: a page an answer marks ``, part`` is recorded as a part page, so
  enrichment combines it with the work's other part pages.
- Renamed pages: a page an entry lists under an old address records its
  current key, so the old address can be dropped (the rules send it to review
  otherwise).
- Dead pages: a page the provider says does not exist is dropped from its
  entry, and its correction records ``is_dead_page``. The run drops dead pages
  before looking for doubled providers, so a dead page never makes a doubled
  provider or reaches review (``dead_pages``).
- Two passes: a second-pass correction stands in for the first pass's
  correction of the same page of the same seed entry (``merge_corrections``).
  The second pass places pages where the first pass left them, so its
  correction already holds the page's final place.
- Named pages: a page an answer sends to ``other work <page key>`` belongs
  with the named page, so the page ends up wherever the named page ends up
  (``follow_named_entry_pages``). When this run moves the named page out of
  its seed entry, the page follows it to the other seed entry or to the new
  entry. Following runs once both passes are merged, because only then is
  every page's final place known.

How corrected entries are rebuilt from the corrections is in
``corrected_entries``.

Real lines of the corrections file are shown in the README, under "Lines of
the corrections file".
"""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict
from work_matching import ReviewCase, replace_file_lines

# The reason recorded for a page placed by a review answer.
REVIEW_ANSWER_REASON = "review answer"
# The reason recorded for a page the run drops because its provider says the
# page does not exist.
DEAD_PAGE_REASON = "dead page"


class SeedCorrection(BaseModel):
    """Where one checked page ends up, and why."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # The reference page of the seed entry that lists the page.
    entry_reference_page: str
    # The page key and address as the seed entry lists them.
    page_key: str
    address: str
    # The page's work in answer words: ``work 1``, ``work 2`` and so on or
    # ``other work``. None for a dead page, which the run drops from every work.
    placement_word: str | None
    # Whether enrichment fetches this page for its provider in its work.
    is_main_page: bool = False
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
    # Whether the page covers only part of its work, such as one season of a
    # run MyAnimeList lists whole, so enrichment combines it with the work's
    # other part pages rather than taking it for the whole work.
    is_part_page: bool = False
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
    """The corrections placing the pages of the entries checked, and their review cases."""

    corrections: tuple[SeedCorrection, ...]
    # Every case of this run, with its kept answer, for the review file.
    review_cases: tuple[ReviewCase, ...]


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


def merge_corrections(
    first_pass_corrections: Iterable[SeedCorrection],
    second_pass_corrections: Iterable[SeedCorrection],
) -> list[SeedCorrection]:
    """Merge two passes' corrections, the second standing in for the first's.

    Args:
        first_pass_corrections: The first pass's corrections.
        second_pass_corrections: The second pass's corrections, made on the
            entries as the first pass left them.

    Returns:
        Each first-pass correction no second-pass correction names by the same
        seed entry and page, in order, then every second-pass correction.
    """
    second_pass_corrections = list(second_pass_corrections)
    replaced_entry_pages = {
        (correction.entry_reference_page, correction.page_key)
        for correction in second_pass_corrections
    }
    return [
        correction
        for correction in first_pass_corrections
        if (correction.entry_reference_page, correction.page_key)
        not in replaced_entry_pages
    ] + second_pass_corrections


def follow_named_entry_pages(
    corrections: Iterable[SeedCorrection],
) -> list[SeedCorrection]:
    """Move each page an answer sent to a named page into the entry where that named page ends up.

    A page sent to ``other work <page key>`` first joins the seed entry
    holding the named page. When another correction moves the named page out
    of that entry, the page follows it, and a named page that was itself sent
    to a named page is followed in turn.

    Args:
        corrections: Every correction of the run, both passes merged.

    Returns:
        The corrections in the same order, each page that follows a moved
        named page pointing at the named page's final entry.
    """
    corrections = list(corrections)
    correction_by_entry_page = {
        (correction.entry_reference_page, correction.page_key): correction
        for correction in corrections
    }
    return [
        _follow_named_entry_page(correction, correction_by_entry_page)
        for correction in corrections
    ]


def _follow_named_entry_page(
    correction: SeedCorrection,
    correction_by_entry_page: dict[tuple[str, str], SeedCorrection],
) -> SeedCorrection:
    """Point one correction at the final entry of the page its answer named.

    Args:
        correction: The correction to follow.
        correction_by_entry_page: Every correction, by its seed entry's
            reference page and its page key.

    Returns:
        The correction pointing at the named page's final entry, or the
        correction unchanged when no answer named a page or the named page
        stays in its seed entry.
    """
    destination = (correction.target_entry_reference_page, correction.new_entry_key)
    followed_correction = correction
    seen_entry_pages: set[tuple[str, str]] = set()
    while (
        followed_correction.named_entry_page_key
        and followed_correction.target_entry_reference_page
    ):
        named_entry_page = (
            followed_correction.target_entry_reference_page,
            followed_correction.named_entry_page_key,
        )
        named_correction = correction_by_entry_page.get(named_entry_page)
        if named_entry_page in seen_entry_pages or named_correction is None:
            break
        seen_entry_pages.add(named_entry_page)
        if not (
            named_correction.target_entry_reference_page
            or named_correction.new_entry_key
        ):
            break
        destination = (
            named_correction.target_entry_reference_page,
            named_correction.new_entry_key,
        )
        followed_correction = named_correction
    if destination == (
        correction.target_entry_reference_page,
        correction.new_entry_key,
    ):
        return correction
    return correction.model_copy(
        update={
            "target_entry_reference_page": destination[0],
            "new_entry_key": destination[1],
        }
    )
