"""Read the last run's summary from its message, for the page's run panel and states.

The page reads the run only through the review case source's message
(``SeedCleaningRunResult`` in ``shared_proto/v1/review_cases.proto``), never
through the seed-cleaning library, so a source for Discovery replaces the file
source without changing the page. ``RunSummary`` keeps the message's field
names, so the templates read the same names whichever source wrote it.
"""

from dataclasses import dataclass
from datetime import datetime

from pydantic import JsonValue
from review_page.review_case_messages import (
    convert_message_to_fields,
    join_ordered_mappings,
)
from shared_proto.v1 import review_cases_pb2


@dataclass(frozen=True, slots=True)
class RunSummary:
    """What the last run did, as the run panel and the case states read it."""

    seed_file: str
    started_at: datetime
    finished_at: datetime
    # Counts by name, in the order the run reports them.
    correction_counts: dict[str, int]
    corrected_entry_counts: dict[str, int]
    linked_page_counts: dict[str, int]
    rejected_answer_case_names: tuple[str, ...]
    conflicting_linked_page_keys: tuple[str, ...]
    unused_answer_case_names: tuple[str, ...]
    case_names: tuple[str, ...]
    new_case_names: tuple[str, ...]
    # Every answer as the run wrote it: each case's original value.
    original_answers: dict[str, JsonValue]


def build_run_summary(message: review_cases_pb2.SeedCleaningRunResult) -> RunSummary:
    """Read a run's summary from its message.

    Args:
        message: The last run's result, as the review case source gives it.

    Returns:
        The summary, with times as datetimes and counts in the run's order.
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
