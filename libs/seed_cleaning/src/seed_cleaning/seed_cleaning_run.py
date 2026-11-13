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
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from seed_cleaning.answer_placements import attach_remaining_previous_answers
from seed_cleaning.cleaning_report import (
    build_entries_to_check_document,
    count_corrected_entries,
    count_corrections,
    describe_linked_page_outcomes,
)
from seed_cleaning.corrected_entries import build_corrected_entries, mark_anime_id_pages
from seed_cleaning.corrected_seed import build_corrected_seed
from seed_cleaning.group_placement import correct_entry_groups
from seed_cleaning.linked_page_additions import (
    correct_linked_pages,
    find_entries_without_mal_page,
    find_linking_seed_entries,
    list_entries_with_placed_pages,
)
from seed_cleaning.models import (
    CorrectedSeed,
    EntryGroup,
    LinkedPageCorrections,
    PlacementCorrections,
    SeedCleaningRunResult,
    SplitWorkCandidates,
)
from seed_cleaning.seed_corrections import (
    follow_named_entry_pages,
    list_dead_page_corrections,
    merge_corrections,
    write_corrections,
)
from seed_cleaning.split_work_pass import SplitWorkPass, prepare_split_work_pass
from work_matching import (
    KnownWorkEntriesFromSeed,
    ProviderPageStore,
    ReviewCase,
    ReviewFile,
    WorkEntry,
    find_doubled_provider_page_keys,
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
    previous_run_result = read_seed_cleaning_run_result(data_dir)
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
        review_cases = attach_remaining_previous_answers(
            [*first_pass_cases, *split_work_corrections.review_cases], review_file
        )
        unused_answer_case_names = review_file.write_cases(review_cases)
        write_corrections(data_dir / CORRECTIONS_FILE_NAME, corrections)
    case_names = tuple(sorted(case.case_name for case in review_cases))
    run_result = SeedCleaningRunResult(
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
        if previous_run_result is None
        else tuple(sorted(set(case_names) - set(previous_run_result.case_names))),
        original_answers=review_file.get_answers(),
    )
    replace_file_lines(
        data_dir / RUN_RESULT_FILE_NAME, [run_result.model_dump_json(indent=1) + "\n"]
    )
    _log_run_result(run_result, linked_page_corrections)
    return run_result


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
    """Drop dead pages, place the entries with a doubled provider, and add linked pages.

    Each item is one step, in order.

    - Every dead page the seed lists is dropped (``dead_pages``), and the
      entries are checked as that leaves them, so a dead page never makes a
      doubled provider or reaches review. Corrections still name the
      downloaded seed's entries.
    - The entries with a doubled provider are placed.
    - Links are read from the doubled-provider entries and the entries without
      a MAL page, and the pages they link are added.

    Args:
        known_work_entries: Every seed entry.
        store: The provider page store, holding every page the pass reads.
        review_file: The review file, for the answers kept in it.

    Returns:
        The dead-page and doubled-provider corrections with the doubled-provider
        cases, and the linked-page corrections and cases.
    """
    dead_page_corrections = list_dead_page_corrections(known_work_entries, store)
    logger.info(f"Dead pages dropped: {len(dead_page_corrections)}")
    live_seed = build_corrected_seed(dead_page_corrections, known_work_entries, store)
    doubled_provider_entries = find_doubled_provider_entries(
        live_seed.known_work_entries
    )
    entries_without_mal_page = find_entries_without_mal_page(
        live_seed.known_work_entries
    )
    logger.info(f"Entries without a MAL page: {len(entries_without_mal_page)}")
    entry_group_corrections = correct_entry_groups(
        [EntryGroup((entry,)) for entry in doubled_provider_entries],
        store,
        live_seed.known_work_entries,
        review_file,
        live_seed,
    )
    linking_seed_entries_by_page_key = find_linking_seed_entries(
        list_entries_with_placed_pages(
            _find_seed_entries(
                [*doubled_provider_entries, *entries_without_mal_page],
                live_seed,
                known_work_entries,
            ),
            entry_group_corrections,
            known_work_entries,
            store,
        ),
        store,
        known_work_entries,
    )
    linked_page_corrections = correct_linked_pages(
        linking_seed_entries_by_page_key, store, known_work_entries, review_file
    )
    group_corrections = PlacementCorrections(
        (*dead_page_corrections, *entry_group_corrections.corrections),
        entry_group_corrections.review_cases,
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
        [
            EntryGroup.from_candidates(candidate_group)
            for candidate_group in split_work_pass.candidates
        ],
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
        candidate_group.shared_provider is not None
        for candidate_group in candidate_groups
    )
    logger.info(f"Entries with a doubled provider: {len(doubled_provider_entries)}")
    logger.info(
        f"Split-work candidate groups: {len(candidate_groups)} ({marked_group_count} marked)"
    )


def _find_seed_entries(
    live_entries: list[WorkEntry],
    live_seed: CorrectedSeed,
    known_work_entries: KnownWorkEntriesFromSeed,
) -> list[WorkEntry]:
    """Find the downloaded seed's entry each entry without dead pages stands for.

    The linked-page pass names entries by the downloaded seed's reference
    pages, so it reads the seed's own entries. A dead page links nothing, so
    reading those entries with their dead pages changes no link.

    Args:
        live_entries: Entries of the seed without its dead pages.
        live_seed: The seed without its dead pages, with each entry's identity.
        known_work_entries: Every entry of the downloaded seed.

    Returns:
        The downloaded seed's entries, in the order given.
    """
    seed_entries = []
    for live_entry in live_entries:
        seed_reference_page = live_seed.identity_by_reference_page[
            live_entry.reference_page_key
        ].seed_entry_reference_page
        if seed_reference_page and (
            seed_entry := known_work_entries.find_by_page_key(seed_reference_page)
        ):
            seed_entries.append(seed_entry)
    return seed_entries


def _list_rejected_answer_case_names(
    review_cases: Iterable[ReviewCase],
) -> tuple[str, ...]:
    """Return the cases answered with an answer the rules refuse, sorted.

    Args:
        review_cases: Every review case of the run, with its kept answer.

    Returns:
        Sorted case names whose answer was rejected by the answer rules.
    """
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
    """Count the linked pages waiting for a fetch, not found, and held elsewhere.

    Args:
        linked_page_corrections: The linked-page step's outcomes.

    Returns:
        Counts keyed by their printed description.
    """
    return {
        "linked pages waiting for a fetch": len(
            linked_page_corrections.waiting_page_keys
        ),
        "linked pages not found": len(linked_page_corrections.not_found_page_keys),
        "linked pages held under another address": len(
            linked_page_corrections.held_page_keys
        ),
    }


def _log_run_result(
    run_result: SeedCleaningRunResult, linked_page_corrections: LinkedPageCorrections
) -> None:
    """Log the run's counts, in the order and wording the run log has always used.

    Args:
        run_result: The completed seed-cleaning run result.
        linked_page_corrections: The linked-page step's outcomes.
    """
    for count_name, correction_count in run_result.correction_counts.items():
        logger.info(f"{count_name.capitalize()}: {correction_count}")
    for report_line in describe_linked_page_outcomes(linked_page_corrections):
        logger.info(report_line)
    for case_name in run_result.unused_answer_case_names:
        logger.info(f"Unused answer, kept but not applied: {case_name}")
    for count_name, entry_count in run_result.corrected_entry_counts.items():
        logger.info(f"{count_name.capitalize()}: {entry_count}")


def find_doubled_provider_entries(entries: Iterable[WorkEntry]) -> list[WorkEntry]:
    """Return the entries holding a doubled provider, in the order given.

    The rule is the doubled-provider rules' own
    (``find_doubled_provider_page_keys``), so seed cleaning checks exactly the
    pages ``decide_doubled_providers`` decides:

    - two pages of one provider count only when their ids differ;
    - Simkl and AnimeCountdown count as one provider, because they number a work
      the same, so a Simkl page and an AnimeCountdown page with the same number
      are one page.

    Args:
        entries: The entries to check, such as every entry of a seed file.

    Returns:
        The entries holding two or more pages of one provider, in order.
    """
    return [
        entry
        for entry in entries
        if find_doubled_provider_page_keys(entry.addresses_by_page_key)
    ]
