"""Run both seed-cleaning passes over the stored pages, and keep what the run did.

``scripts/clean_seed.py`` fetches the pages and then calls ``run_seed_cleaning``;
the review page starts the same script with ``--no-fetch``. The run:

Each item gives the rule, then the reason for it.

- Holds the review file's lock (``work_matching.review_file_lock``) from reading
  the answers to rewriting the file. A save in between would be read by
  neither the run nor the file, and so lost.
- Runs the first pass (doubled providers and linked pages), then the second
  (split works) on the entries as the first leaves them, and writes the review,
  corrections and entries-to-check files, as the script did.
- Writes ``review_run.json`` beside the review file: what the run counted, the
  answers it refused, the answers whose case no longer comes up, its case
  names, the cases new since the previous run, and every answer as written.
  The review page reads it: the answers as written are each case's original
  value until the next run, and the counts fill its results panel.
- A first run has no previous run to compare with, so it marks no case new.
"""

import json
import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, JsonValue
from seed_cleaning.cleaning_report import (
    build_entries_to_check_document,
    count_corrected_entries,
    count_corrections,
    describe_linked_page_outcomes,
)
from seed_cleaning.corrected_entries import build_corrected_entries, mark_anime_id_pages
from seed_cleaning.doubled_provider_entries import find_doubled_provider_entries
from seed_cleaning.entry_groups import EntryGroup
from seed_cleaning.group_placement import correct_entry_groups
from seed_cleaning.linked_page_additions import (
    LinkedPageCorrections,
    correct_linked_pages,
    find_entries_without_mal_page,
    find_linking_seed_entries,
    list_entries_with_placed_pages,
)
from seed_cleaning.seed_corrections import (
    PlacementCorrections,
    follow_named_entry_pages,
    merge_corrections,
    write_corrections,
)
from seed_cleaning.split_work_candidates import SplitWorkCandidates
from seed_cleaning.split_work_pass import SplitWorkPass, prepare_split_work_pass
from work_matching import (
    KnownWorkEntriesFromSeed,
    ProviderPageStore,
    ReviewCase,
    ReviewFile,
    WorkEntry,
    hold_review_file_lock,
    replace_file_lines,
)

logger = logging.getLogger(__name__)

RUN_RESULT_FILE_NAME = "review_run.json"
REVIEW_FILE_NAME = "review.jsonl"
CORRECTIONS_FILE_NAME = "corrections.jsonl"
ENTRIES_TO_CHECK_FILE_NAME = "entries_to_check.json"
# How long a run waits for a review-page save to finish; saves take milliseconds.
_SECONDS_TO_WAIT_FOR_SAVE = 30.0


class SeedCleaningRunResult(BaseModel):
    """What one seed-cleaning run did, as the review page reads it."""

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


def run_seed_cleaning(
    seed_document: Mapping[str, Any],
    seed_file: Path,
    known_work_entries: KnownWorkEntriesFromSeed,
    store: ProviderPageStore,
    data_dir: Path,
    *,
    seconds_to_wait_for_lock: float = _SECONDS_TO_WAIT_FOR_SAVE,
) -> SeedCleaningRunResult:
    """Run both passes over the stored pages, write their files and the run's result.

    Args:
        seed_document: The downloaded seed release, as read from its file.
        seed_file: The seed release file, kept in the result.
        known_work_entries: Every seed entry.
        store: The provider page store.
        data_dir: The folder the review, corrections, entries and result
            files go to.
        seconds_to_wait_for_lock: How long to wait for a save holding the
            review file's lock.

    Returns:
        The run's result, also written to ``review_run.json``.

    Raises:
        ReviewFileLockedError: Another writer still held the review file's lock.
    """
    started_at = datetime.now(UTC)
    previous_result = read_seed_cleaning_run_result(data_dir)
    review_file_path = data_dir / REVIEW_FILE_NAME
    with hold_review_file_lock(
        review_file_path, seconds_to_wait=seconds_to_wait_for_lock
    ):
        review_file = ReviewFile(review_file_path)
        group_corrections, linked_page_corrections = run_first_pass(
            known_work_entries, store, review_file
        )
        first_pass_corrections = [
            *group_corrections.corrections,
            *linked_page_corrections.corrections,
        ]
        first_pass_cases = [
            *group_corrections.review_cases,
            *linked_page_corrections.review_cases,
        ]
        split_work_pass = prepare_split_work_pass(
            first_pass_corrections, first_pass_cases, known_work_entries, store
        )
        split_work_corrections = run_second_pass(split_work_pass, store, review_file)
        write_entries_to_check(
            seed_document,
            find_doubled_provider_entries(known_work_entries),
            split_work_pass.candidates,
            data_dir,
        )
        merged_corrections = follow_named_entry_pages(
            merge_corrections(
                first_pass_corrections, split_work_corrections.corrections
            )
        )
        corrected_entries = build_corrected_entries(
            merged_corrections, known_work_entries, store
        )
        corrections = mark_anime_id_pages(merged_corrections, corrected_entries)
        review_cases = [*first_pass_cases, *split_work_corrections.review_cases]
        unused_answer_case_names = review_file.write_cases(review_cases)
        write_corrections(data_dir / CORRECTIONS_FILE_NAME, corrections)
    case_names = tuple(sorted(case.case_name for case in review_cases))
    result = SeedCleaningRunResult(
        seed_file=str(seed_file),
        started_at=started_at,
        finished_at=datetime.now(UTC),
        correction_counts=count_corrections(corrections, review_cases),
        corrected_entry_counts=count_corrected_entries(corrected_entries),
        linked_page_counts=_count_linked_pages_not_added(linked_page_corrections),
        rejected_answer_case_names=_list_rejected_answer_case_names(review_cases),
        conflicting_linked_page_keys=linked_page_corrections.conflicting_page_keys,
        unused_answer_case_names=unused_answer_case_names,
        case_names=case_names,
        new_case_names=()
        if previous_result is None
        else tuple(sorted(set(case_names) - set(previous_result.case_names))),
        original_answers=review_file.get_answers(),
    )
    replace_file_lines(
        data_dir / RUN_RESULT_FILE_NAME, [result.model_dump_json(indent=1) + "\n"]
    )
    _log_result(result, linked_page_corrections)
    return result


def read_seed_cleaning_run_result(data_dir: Path) -> SeedCleaningRunResult | None:
    """Read the last run's result from the data folder.

    Args:
        data_dir: The folder ``review_run.json`` is written to.

    Returns:
        The result, or None when no run has written one yet.
    """
    result_path = data_dir / RUN_RESULT_FILE_NAME
    if not result_path.exists():
        return None
    return SeedCleaningRunResult.model_validate_json(
        result_path.read_text(encoding="utf-8")
    )


def run_first_pass(
    known_work_entries: KnownWorkEntriesFromSeed,
    store: ProviderPageStore,
    review_file: ReviewFile,
) -> tuple[PlacementCorrections, LinkedPageCorrections]:
    """Place the entries with a doubled provider and add the pages entries link.

    Links are read from the doubled-provider entries and the entries without a
    MAL page.

    Args:
        known_work_entries: Every seed entry.
        store: The provider page store, holding every page the pass reads.
        review_file: The review file, for the answers kept in it.

    Returns:
        The doubled-provider corrections and cases, and the linked-page ones.
    """
    doubled_provider_entries = find_doubled_provider_entries(known_work_entries)
    entries_without_mal_page = find_entries_without_mal_page(known_work_entries)
    logger.info(f"Entries without a MAL page: {len(entries_without_mal_page)}")
    group_corrections = correct_entry_groups(
        [EntryGroup((entry,)) for entry in doubled_provider_entries],
        store,
        known_work_entries,
        review_file,
    )
    linking_seed_entries_by_page_key = find_linking_seed_entries(
        list_entries_with_placed_pages(
            [*doubled_provider_entries, *entries_without_mal_page],
            group_corrections,
            known_work_entries,
            store,
        ),
        store,
        known_work_entries,
    )
    linked_page_corrections = correct_linked_pages(
        linking_seed_entries_by_page_key, store, known_work_entries, review_file
    )
    return group_corrections, linked_page_corrections


def run_second_pass(
    split_work_pass: SplitWorkPass, store: ProviderPageStore, review_file: ReviewFile
) -> PlacementCorrections:
    """Place each group of split-work candidates.

    Args:
        split_work_pass: The second pass's entries and candidate groups.
        store: The provider page store, holding every page the pass reads.
        review_file: The review file, for the answers kept in it.

    Returns:
        The split-work corrections and cases.
    """
    return correct_entry_groups(
        [EntryGroup.from_candidates(group) for group in split_work_pass.candidates],
        store,
        split_work_pass.known_work_entries,
        review_file,
        split_work_pass.corrected_seed,
    )


def write_entries_to_check(
    seed_document: Mapping[str, Any],
    doubled_provider_entries: list[WorkEntry],
    candidate_groups: list[SplitWorkCandidates],
    data_dir: Path,
) -> None:
    """Write ``entries_to_check.json`` and log how many entries each pass checks.

    Args:
        seed_document: The downloaded seed release, as read from its file.
        doubled_provider_entries: The first pass's entries.
        candidate_groups: The second pass's groups of split-work candidates.
        data_dir: The folder ``entries_to_check.json`` is written to.
    """
    entries_to_check = build_entries_to_check_document(
        seed_document.get("$schema"), doubled_provider_entries, candidate_groups
    )
    replace_file_lines(
        data_dir / ENTRIES_TO_CHECK_FILE_NAME,
        [json.dumps(entries_to_check, ensure_ascii=False, indent=1) + "\n"],
    )
    marked_group_count = sum(
        group.shared_provider is not None for group in candidate_groups
    )
    logger.info(f"Entries with a doubled provider: {len(doubled_provider_entries)}")
    logger.info(
        f"Split-work candidate groups: {len(candidate_groups)} ({marked_group_count} marked)"
    )


def _list_rejected_answer_case_names(
    review_cases: list[ReviewCase],
) -> tuple[str, ...]:
    """Return the cases answered with an answer the rules refuse, sorted."""
    return tuple(
        sorted(
            case.case_name
            for case in review_cases
            if case.answer is not None and case.accepted_answer is None
        )
    )


def _count_linked_pages_not_added(
    linked_page_corrections: LinkedPageCorrections,
) -> dict[str, int]:
    """Count the linked pages waiting for a fetch, not found, and held elsewhere."""
    return {
        "linked pages waiting for a fetch": len(
            linked_page_corrections.waiting_page_keys
        ),
        "linked pages not found": len(linked_page_corrections.not_found_page_keys),
        "linked pages held under another address": len(
            linked_page_corrections.held_page_keys
        ),
    }


def _log_result(
    result: SeedCleaningRunResult, linked_page_corrections: LinkedPageCorrections
) -> None:
    """Log the run's counts, in the order and wording the run log has always used."""
    for count_name, correction_count in result.correction_counts.items():
        logger.info(f"{count_name.capitalize()}: {correction_count}")
    for report_line in describe_linked_page_outcomes(linked_page_corrections):
        logger.info(report_line)
    for case_name in result.unused_answer_case_names:
        logger.info(f"Answer kept, its case no longer comes up: {case_name}")
    for count_name, entry_count in result.corrected_entry_counts.items():
        logger.info(f"{count_name.capitalize()}: {entry_count}")
