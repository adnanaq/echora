"""Where the review page reads cases, saves answers and starts a Run: one swappable source.

The page talks to nothing else, so where cases are kept can change without
changing the page. Each item gives the rule, then the reason for it.

- Every method takes and returns protobuf messages
  (``shared_proto/v1/review_cases.proto``), the same messages a gRPC service
  will send, so a gRPC source needs no new types.
- The first source, ``FileReviewCaseSource``, reads and writes seed cleaning's
  local review file. Discovery adds a source backed by the enrichment
  service's gRPC API; it implements these same methods and is tested against
  the same checks.
- A source checks every answer before saving it, with the same rules a run
  uses, so a page using any source never saves an answer the run refuses.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from shared_proto.v1 import review_cases_pb2


class ReviewCaseSource(ABC):
    """Read review cases, save their answers, and apply the answers with a Run."""

    @abstractmethod
    def list_cases(self) -> list[review_cases_pb2.ReviewCase]:
        """Return every review case with its saved answer, in case name order."""

    @abstractmethod
    def get_case(self, case_name: str) -> review_cases_pb2.ReviewCase | None:
        """Return one review case with its saved answer, or None when it is unknown."""

    @abstractmethod
    def get_last_run_result(self) -> review_cases_pb2.SeedCleaningRunResult | None:
        """Return what the last run did, or None when no run has finished yet."""

    @abstractmethod
    def save_answers(
        self, case_answers: Sequence[review_cases_pb2.CaseAnswer]
    ) -> list[review_cases_pb2.RuleCheckResult]:
        """Check each answer and save the ones that follow the rules.

        Args:
            case_answers: Each case's new answer; an unset answer clears it.

        Returns:
            One result per answer, in the given order: saved, or the rules it
            breaks; with any warning about answers that conflict.

        Raises:
            ReviewFileLockedError: A run is writing the cases; nothing is saved.
            UnknownCaseError: A run removed a named case after the answers
                were checked; nothing is saved.
        """

    @abstractmethod
    def start_run(self) -> review_cases_pb2.RunState:
        """Start applying the saved answers, unless a run is already going.

        Returns:
            The run state after starting.
        """

    @abstractmethod
    def get_run_state(self) -> review_cases_pb2.RunState:
        """Return whether a run is going, and how the last one started here ended."""

    @abstractmethod
    def list_corrections(self) -> list[review_cases_pb2.SeedCorrection]:
        """Return every correction the last run made, each with its change and reason.

        Returns:
            The corrections in the source's order; empty before the first run.
        """

    @abstractmethod
    def list_checked_entries(self) -> list[review_cases_pb2.CheckedEntry]:
        """Return the entries the last run checked, so corrections show their titles.

        Returns:
            One entry per reference page; empty before the first run.
        """
