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
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from shared_proto.v1 import review_cases_pb2

# Rule identifier returned when a case's saved answer changed after editing began.
ANSWER_SAVED_ELSEWHERE_RULE_NAME = "ANSWER_SAVED_ELSEWHERE"


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
