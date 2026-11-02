"""Convert review cases between the library model and protobuf messages.

The review case source hands the page protobuf messages
(``shared_proto/v1/review_cases.proto``), so a gRPC source can replace the file
source later. The rules and choices still run on the library models
(``work_matching.ReviewCase``), so the page converts between them. A source
converts its own records to messages with the same mapping helpers
(``list_ordered_mappings``, ``join_ordered_mappings``). Each item gives the rule, then the reason for
it.

- Each message has the model's field names, so Google's
  ``google.protobuf.json_format`` converts through the models' JSON form; no
  field is copied by hand, and a field added to both is converted at once.
- The one difference is a case's decisions: the model keeps one list of either
  kind, and a message cannot repeat a union, so the list moves to the field of
  the case's kind (``doubled_provider_decisions`` or ``linked_page_decisions``).
- The models' mappings are lists of pairs in the messages, because a proto map
  keeps no order and the order matters: an entry's addresses (enrichment
  fetches each provider's first address), a case's entries (named A, B and so
  on), suggestions in answer page order, and counts in the order a run reports
  them. ``_ORDERED_MAPPING_FIELDS`` names each field and its pair; a mapping
  becomes its list and back in the same order. Only a lookup with no order,
  the run's original answers by case name, stays a map.
- Fields without a set value are written out, so a model's required fields
  that are empty, such as a case with no reasons, survive the round trip.

When the enrichment service serves cases over gRPC, this module moves to a
library both import.
"""

from typing import Any

from google.protobuf import json_format
from shared_proto.v1 import review_cases_pb2
from work_matching import ReviewCase, ReviewCaseKind

_DOUBLED_PROVIDER_DECISIONS_FIELD = "doubled_provider_decisions"
# Each model mapping kept as a list of pairs in the messages: the field name,
# then the names of the pair's key and value fields.
_ORDERED_MAPPING_FIELDS: dict[str, tuple[str, str]] = {
    "addresses_by_page_key": ("page_key", "address"),
    "entries": ("reference_page", "entry"),
    "suggested_answer": ("page_key", "place"),
    "belonging_page_keys": ("page_key", "linked_page_keys"),
    "unrelated_pages": ("page_key", "unrelated_page"),
    "correction_counts": ("name", "count"),
    "corrected_entry_counts": ("name", "count"),
    "linked_page_counts": ("name", "count"),
}
_LINKED_PAGE_DECISIONS_FIELD = "linked_page_decisions"


def convert_case_to_message(case: ReviewCase) -> review_cases_pb2.ReviewCase:
    """Convert a review case to its message.

    Args:
        case: The review case.

    Returns:
        The message, with the decisions in the field of the case's kind.
    """
    case_fields = case.model_dump(mode="json")
    decisions = case_fields.pop("decisions")
    decisions_field = (
        _LINKED_PAGE_DECISIONS_FIELD
        if case.kind is ReviewCaseKind.LINKED_PAGE
        else _DOUBLED_PROVIDER_DECISIONS_FIELD
    )
    case_fields[decisions_field] = decisions
    case_fields = list_ordered_mappings(case_fields)
    if case_fields["answer"] is None:
        # ParseDict would set a JSON null Value; an open case leaves it unset.
        del case_fields["answer"]
    return json_format.ParseDict(case_fields, review_cases_pb2.ReviewCase())


def convert_message_to_case(message: review_cases_pb2.ReviewCase) -> ReviewCase:
    """Convert a review case message back to the review case.

    Args:
        message: The review case message.

    Returns:
        The review case, with both decision fields joined into one list.
    """
    case_fields = join_ordered_mappings(convert_message_to_fields(message))
    case_fields["decisions"] = [
        *case_fields.pop(_DOUBLED_PROVIDER_DECISIONS_FIELD),
        *case_fields.pop(_LINKED_PAGE_DECISIONS_FIELD),
    ]
    return ReviewCase.model_validate(case_fields)


def build_case_answer(case_name: str, answer: Any) -> review_cases_pb2.CaseAnswer:
    """Build the message saving one case's answer; None clears the answer.

    Args:
        case_name: The case.
        answer: The answer as it would be written in the file, or None.

    Returns:
        The message, with the answer as a JSON value, or unset for None.
    """
    case_answer = review_cases_pb2.CaseAnswer(case_name=case_name)
    if answer not in (None, {}):
        json_format.ParseDict(answer, case_answer.answer)
    return case_answer


def convert_message_to_fields(message: Any) -> dict[str, Any]:
    """Return a message's fields by their proto names, unset fields included."""
    return json_format.MessageToDict(
        message,
        preserving_proto_field_name=True,
        always_print_fields_with_no_presence=True,
    )


def list_ordered_mappings(fields: Any) -> Any:
    """Turn every ordered mapping into its list of pairs, in the mapping's order."""
    if isinstance(fields, list):
        return [list_ordered_mappings(item) for item in fields]
    if not isinstance(fields, dict):
        return fields
    listed_fields = {key: list_ordered_mappings(value) for key, value in fields.items()}
    for field_name, (key_name, value_name) in _ORDERED_MAPPING_FIELDS.items():
        if isinstance(listed_fields.get(field_name), dict):
            listed_fields[field_name] = [
                {key_name: key, value_name: value}
                for key, value in listed_fields[field_name].items()
            ]
    return listed_fields


def join_ordered_mappings(fields: Any) -> Any:
    """Turn every list of pairs back into its mapping, in the list's order."""
    if isinstance(fields, list):
        return [join_ordered_mappings(item) for item in fields]
    if not isinstance(fields, dict):
        return fields
    joined_fields = {key: join_ordered_mappings(value) for key, value in fields.items()}
    for field_name, (key_name, value_name) in _ORDERED_MAPPING_FIELDS.items():
        if isinstance(joined_fields.get(field_name), list):
            joined_fields[field_name] = {
                pair[key_name]: pair.get(value_name)
                for pair in joined_fields[field_name]
            }
    return joined_fields
