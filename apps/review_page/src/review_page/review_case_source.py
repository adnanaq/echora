"""Define the abstract interface for reading review cases, saving answers, and running seed cleaning.

Decouples the review web application from underlying storage so that local file
storage can be swapped for a remote gRPC service without modifying route handlers
or presentation views.

Each design rule reads `<rule>: <reason>`:
- Protobuf message boundary: Every method exchanges protobuf messages defined in
  ``shared_proto/v1/review_cases.proto`` so a gRPC service implementation requires
  no type adaptations.
- Swappable implementations: ``FileReviewCaseSource`` reads and writes local
  seed-cleaning files, while future discovery workflows can provide a gRPC-backed
  source implementing the same contract.
- Pre-save validation: Implementations validate every submitted answer against
  domain placement rules before persisting changes so invalid placements are
  never stored.
- Optimistic concurrency control: Answer submissions may include the baseline
  saved answer observed when editing began, rejecting concurrent modifications
  with ``ANSWER_SAVED_ELSEWHERE_RULE_NAME`` so edits from another session are
  never overwritten silently.
- Shared loaders: ``load_cases`` and ``load_last_run_summary`` convert the
  source's messages into the models the routes use, so each route class reads
  them the same way.
- One conversion per message: converting every case takes about a third of a
  second, and the toolbar, the case list and a save each need every case.
  ``load_cases`` keeps the model it converted from each message, and reuses it
  while the source hands out the very same message object, as a file-backed
  source does for every case its file still holds unchanged. Keeping the
  converted messages keeps them alive, so a new message can never be taken for
  one of them, and only the messages of the last call are kept.
"""

import threading
from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime
from typing import NamedTuple

from review_page.models import InvalidNamedPageKeyReason, RunSummary
from review_page.review_case_messages import (
    convert_message_to_case,
    convert_message_to_fields,
    join_ordered_mappings,
)
from shared_proto.v1 import review_cases_pb2
from work_matching import ReviewCase

# Rule identifier returned when a case's saved answer changed after editing began.
ANSWER_SAVED_ELSEWHERE_RULE_NAME = "ANSWER_SAVED_ELSEWHERE"


class SeedFileUnavailableError(RuntimeError):
    """Report that the seed file an answer's named entries are checked against cannot be read."""

    def __init__(self) -> None:
        """Explain how to make a seed file available."""
        super().__init__(
            "The seed file that named entries are checked against is missing or "
            "unknown, so nothing was saved. Set REVIEW_PAGE_SEED_FILE, or run "
            "scripts/clean_seed.py once so its result names the file."
        )


class ReviewCaseSource(ABC):
    """Provide storage and execution operations for review cases, answers, and runs."""

    @abstractmethod
    def list_cases(self) -> list[review_cases_pb2.ReviewCase]:
        """Retrieve all review cases with their persisted answers in case name order.

        Returns:
            All review case messages stored in the source.
        """

    @abstractmethod
    def get_case(self, case_name: str) -> review_cases_pb2.ReviewCase | None:
        """Retrieve a single review case with its persisted answer by case name.

        Args:
            case_name: The unique identifier of the review case to read.

        Returns:
            The matching review case message, or None when absent from the source.
        """

    @abstractmethod
    def get_last_run_result(self) -> review_cases_pb2.SeedCleaningRunResult | None:
        """Retrieve the ``SeedCleaningRunResult`` message from the most recent run.

        Returns:
            The last run result message, or None when no run has completed yet.
        """

    @abstractmethod
    def save_answers(
        self, case_answers: Sequence[review_cases_pb2.CaseAnswer]
    ) -> list[review_cases_pb2.RuleCheckResult]:
        """Validate submitted answers against placement rules and persist valid entries.

        Args:
            case_answers: The submitted case answers to validate and persist.
                An unset answer field clears the stored answer for that case.
                A populated ``seen_saved_answer`` specifies the baseline answer
                present when editing started for optimistic concurrency checks.

        Returns:
            One validation check outcome per submitted answer in request order,
            including broken rules and cross-case provider conflict warnings.

        Raises:
            ReviewFileLockedError: A pipeline run holds the review file lock,
                preventing any answers from being saved.
            UnknownCaseError: A concurrent run removed a target case between
                validation and persistence, leaving storage unchanged.
            SavedAnswerChangedError: Another session modified a case's stored
                answer between validation and persistence, leaving storage unchanged.
            SeedFileUnavailableError: An answer names a seed entry and the seed
                file to look it up in is missing or unknown, leaving storage
                unchanged.
        """

    @abstractmethod
    def start_run(self) -> review_cases_pb2.RunState:
        """Start a seed-cleaning pipeline run to apply saved answers if none is active.

        Returns:
            The active pipeline run state after starting or checking the run.
        """

    @abstractmethod
    def get_run_state(self) -> review_cases_pb2.RunState:
        """Retrieve whether a pipeline run is active and the exit code of the last run.

        Returns:
            The current run state message indicating active execution and the
            exit code of the most recent page-initiated run.
        """

    @abstractmethod
    def find_invalid_named_page_keys(
        self, page_keys: Sequence[str]
    ) -> dict[str, InvalidNamedPageKeyReason]:
        """Find the page keys an answer cannot name after ``other work``, and why.

        A key cannot be used when its provider no longer has the page, or when
        no seed entry holds it. Saving refuses such an answer, so the case
        panel marks the row before the reviewer saves.

        Args:
            page_keys: The page keys an answer names after ``other work``.

        Returns:
            Each invalid key mapped to the reason it is invalid. Valid keys are left out.
        """

    @abstractmethod
    def find_named_entry_suggestions(
        self, page_keys: Sequence[str]
    ) -> dict[str, review_cases_pb2.TargetEntrySuggestion]:
        """Find the seed entry holding each page key an answer can name after ``other work``.

        The picker shows the entry a typed key or address names, so the
        reviewer sees which work it is before choosing it.

        Args:
            page_keys: The page keys to look up.

        Returns:
            Each key a seed entry holds, mapped to that entry's reference page,
            title and providers. Keys no seed entry holds are left out, and so
            is every key when the seed file cannot be read.
        """

    @abstractmethod
    def list_corrections(self) -> list[review_cases_pb2.SeedCorrection]:
        """Retrieve all seed entry corrections produced by the most recent run.

        Returns:
            The seed correction messages in source order, or an empty list
            before the first run completes.
        """

    @abstractmethod
    def list_checked_entries(self) -> list[review_cases_pb2.CheckedEntry]:
        """Retrieve the seed entries inspected during the most recent pipeline run.

        Returns:
            One checked entry message per reference page key for resolving titles,
            or an empty list before the first run completes.
        """


class _ConvertedCase(NamedTuple):
    """Hold one case message and the case model converted from it."""

    case_message: review_cases_pb2.ReviewCase
    case: ReviewCase


class ConvertedReviewCases:
    """Hold the case model converted from each case message of the last call."""

    def __init__(self) -> None:
        """Start with nothing converted."""
        self._conversion_lock = threading.Lock()
        # Each converted message and its model, by the message's identity.
        self._converted_case_by_message_id: dict[int, _ConvertedCase] = {}

    def convert_cases(
        self, case_messages: Sequence[review_cases_pb2.ReviewCase]
    ) -> list[ReviewCase]:
        """Convert case messages into case models, reusing the model of each message seen last time.

        Args:
            case_messages: The cases as the source hands them out.

        Returns:
            The case models in message order. Callers must not change the
            models, which later requests share.
        """
        with self._conversion_lock:
            converted_case_by_message_id: dict[int, _ConvertedCase] = {}
            for case_message in case_messages:
                converted_case = self._converted_case_by_message_id.get(
                    id(case_message)
                )
                if converted_case is None or (
                    converted_case.case_message is not case_message
                ):
                    converted_case = _ConvertedCase(
                        case_message, convert_message_to_case(case_message)
                    )
                converted_case_by_message_id[id(case_message)] = converted_case
            self._converted_case_by_message_id = converted_case_by_message_id
            return [
                converted_case_by_message_id[id(case_message)].case
                for case_message in case_messages
            ]


_CONVERTED_REVIEW_CASES = ConvertedReviewCases()


def load_cases(source: ReviewCaseSource) -> list[ReviewCase]:
    """Load every review case from a source as case models.

    Args:
        source: The review case source to read.

    Returns:
        All review cases, in the source's order.
    """
    return _CONVERTED_REVIEW_CASES.convert_cases(source.list_cases())


def load_case(source: ReviewCaseSource, case_name: str | None) -> ReviewCase | None:
    """Load a single review case by name from a source as a case model.

    Args:
        source: The review case source to read.
        case_name: The case identifier to look up, or None.

    Returns:
        The converted ``ReviewCase`` model, or None when ``case_name`` is empty
        or absent from the source.
    """
    case_message = source.get_case(case_name) if case_name else None
    return None if case_message is None else convert_message_to_case(case_message)


def load_last_run_summary(source: ReviewCaseSource) -> RunSummary | None:
    """Load the summary of the last run from a source.

    Args:
        source: The review case source to read.

    Returns:
        The last run's summary, or None before the first run.
    """
    last_run_result = source.get_last_run_result()
    return None if last_run_result is None else build_run_summary(last_run_result)


def build_run_summary(message: review_cases_pb2.SeedCleaningRunResult) -> RunSummary:
    """Construct a ``RunSummary`` model from a pipeline run summary protobuf message.

    Args:
        message: The last run summary message returned by the review case source.

    Returns:
        The parsed ``RunSummary`` with ISO timestamps converted to ``datetime``
        instances and count mappings preserved in report order.
    """
    fields = join_ordered_mappings(convert_message_to_fields(message))
    return RunSummary(
        seed_file=fields["seed_file"],
        started_at=datetime.fromisoformat(fields["started_at"]),
        finished_at=datetime.fromisoformat(fields["finished_at"]),
        correction_counts=fields["correction_counts"],
        corrected_entry_counts=fields["corrected_entry_counts"],
        linked_page_counts=fields["linked_page_counts"],
        rejected_answer_case_names=tuple(fields["rejected_answer_case_names"]),
        conflicting_linked_page_keys=tuple(fields["conflicting_linked_page_keys"]),
        unused_answer_case_names=tuple(fields["unused_answer_case_names"]),
        case_names=tuple(fields["case_names"]),
        new_case_names=tuple(fields["new_case_names"]),
        original_answers=fields["original_answers"],
    )
