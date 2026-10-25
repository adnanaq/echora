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
- Two passes: a second-pass correction stands in for the first pass's
  correction of the same page of the same seed entry (``merge_corrections``).
  The second pass places pages where the first pass left them, so its
  correction already holds the page's final place.

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


class SeedCorrection(BaseModel):
    """Where one checked page ends up, and why."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # The reference page of the seed entry that lists the page.
    entry_reference_page: str
    # The page key and address as the seed entry lists them.
    page_key: str
    address: str
    # The page's work in answer words: ``work 1``, ``work 2`` and so on or ``other work``.
    placement_word: str
    # Whether enrichment fetches this page for its provider in its work.
    is_main_page: bool = False
    # The current page key, when the seed lists the page under an old address.
    renamed_to: str | None = None
    # The reference page of the seed entry the page joins; None when the page
    # stays or starts a new entry.
    target_entry_reference_page: str | None = None
    # The new entry the page joins: pages with the same key form one new entry.
    # None when the page stays or joins a seed entry.
    new_entry_key: str | None = None
    # Whether the page is not in the seed and is added to the entry: a page the
    # entry's pages link that no seed entry holds.
    is_added_page: bool = False
    # Whether the page covers only part of its work, such as one season of a
    # run MyAnimeList lists whole, so enrichment combines it with the work's
    # other part pages rather than taking it for the whole work.
    is_part_page: bool = False
    # Whether the entry's anime id now comes from this added page (MAL first), so
    # the search entry stored under the entry's old id must be replaced.
    is_anime_id_page: bool = False
    # Why the page is placed here: the doubled-provider decision's outcome name,
    # the linked-page decision's reason name, or ``review answer``.
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
