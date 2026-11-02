"""Read the last run's summary from its message, for the page's run panel and states.

The page reads the run only through the review case source's message
(``SeedCleaningRunResult`` in ``shared_proto/v1/review_cases.proto``), never
through the seed-cleaning library. A source for Discovery then replaces the
file source without changing the page. ``RunSummary`` keeps the message's field
names, so the templates read the same names whichever source wrote it.
"""

from datetime import datetime

from review_page.models import RunSummary
from review_page.review_case_messages import (
    convert_message_to_fields,
    join_ordered_mappings,
)
from shared_proto.v1 import review_cases_pb2


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
