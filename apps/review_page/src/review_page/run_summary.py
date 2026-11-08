"""Convert pipeline run summary protobuf messages into domain summary models.

Decouples the review page from the ``seed_cleaning`` library by reading run
outcomes exclusively from ``SeedCleaningRunResult`` protobuf messages defined in
``shared_proto/v1/review_cases.proto``. ``RunSummary`` preserves the message
field names so templates render summaries identically across storage backends.
"""

from datetime import datetime

from review_page.models import RunSummary
from review_page.review_case_messages import (
    convert_message_to_fields,
    join_ordered_mappings,
)
from shared_proto.v1 import review_cases_pb2


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
