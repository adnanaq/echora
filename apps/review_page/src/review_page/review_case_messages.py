"""Convert review cases and answers between domain models and protobuf messages.

Translates between ``work_matching.ReviewCase`` domain models and protobuf
messages defined in ``shared_proto/v1/review_cases.proto`` so the review page
can evaluate placement rules in memory while exchanging protobuf messages with
any ``ReviewCaseSource``.

Each conversion rule reads `<rule>: <reason>`:
- Shared field names: Protobuf messages mirror the Pydantic model field names so
  ``google.protobuf.json_format`` converts payloads directly without manual
  field-by-field copying.
- Kind-specific decision lists: Protobuf repeated fields cannot represent a
  polymorphic union, so ``ReviewCase.decisions`` is stored in either
  ``doubled_provider_decisions`` or ``linked_page_decisions`` according to the
  case kind.
- Ordered key-value pairs: Protobuf maps do not preserve insertion order, so
  ordered mappings (entry page addresses, case entries, suggested placements,
  and run summary counts) are serialized as repeated key-value pair messages
  listed in ``_ORDERED_MAPPING_FIELDS``.
- Explicit empty collections: ``always_print_fields_with_no_presence=True``
  emits empty repeated and map fields so required Pydantic collection fields
  survive round-trip conversion.
"""

from collections.abc import Mapping
from typing import Any

from google.protobuf import json_format
from shared_proto.v1 import review_cases_pb2
from work_matching import ReviewCase, ReviewCaseKind

_DOUBLED_PROVIDER_DECISIONS_FIELD = "doubled_provider_decisions"
# Model mapping fields stored as repeated key-value pair messages in protobuf,
# mapped to their (key_field_name, value_field_name) pair.
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
    """Convert a ``ReviewCase`` domain model into its protobuf message representation.

    Args:
        case: The domain review case model to serialize.

    Returns:
        The populated ``ReviewCase`` protobuf message with decisions routed to
        the field matching ``case.kind``.
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
    """Convert a ``ReviewCase`` protobuf message into a ``ReviewCase`` domain model.

    Args:
        message: The protobuf review case message to deserialize.

    Returns:
        The validated ``ReviewCase`` domain model with decision lists merged.
    """
    case_fields = join_ordered_mappings(convert_message_to_fields(message))
    case_fields["decisions"] = [
        *case_fields.pop(_DOUBLED_PROVIDER_DECISIONS_FIELD),
        *case_fields.pop(_LINKED_PAGE_DECISIONS_FIELD),
    ]
    return ReviewCase.model_validate(case_fields)


def build_case_answer(
    case_name: str,
    answer: Any,
    *,
    seen_saved_answers: Mapping[str, Any] | None = None,
) -> review_cases_pb2.CaseAnswer:
    """Build a ``CaseAnswer`` protobuf message for saving or clearing a case's answer.

    Args:
        case_name: The unique identifier of the review case being answered.
        answer: The answer payload to persist, or None to clear the saved answer.
        seen_saved_answers: Optional mapping of case names to baseline saved
            answers observed when editing started for concurrency checking.

    Returns:
        The constructed ``CaseAnswer`` protobuf message.
    """
    case_answer = review_cases_pb2.CaseAnswer(case_name=case_name)
    if answer not in (None, {}):
        json_format.ParseDict(answer, case_answer.answer)
    if seen_saved_answers and case_name in seen_saved_answers:
        json_format.ParseDict(
            seen_saved_answers[case_name], case_answer.seen_saved_answer
        )
    return case_answer


def convert_message_to_fields(message: Any) -> dict[str, Any]:
    """Convert a protobuf message to a dictionary keyed by proto field names.

    Includes fields without explicit presence so empty collections are preserved.

    Args:
        message: The protobuf message to convert.

    Returns:
        A dictionary of JSON-compatible field values keyed by proto field names.
    """
    return json_format.MessageToDict(
        message,
        preserving_proto_field_name=True,
        always_print_fields_with_no_presence=True,
    )


def list_ordered_mappings(fields: Any) -> Any:
    """Recursively convert ordered mapping fields into lists of key-value pair dictionaries.

    Args:
        fields: A JSON-serialized model dictionary or nested value.

    Returns:
        The transformed structure with every field in ``_ORDERED_MAPPING_FIELDS``
        converted from a mapping to a list of key-value dictionaries.
    """
    if isinstance(fields, list):
        return [list_ordered_mappings(field_value) for field_value in fields]
    if not isinstance(fields, dict):
        return fields
    listed_fields = {
        field_name: list_ordered_mappings(field_value)
        for field_name, field_value in fields.items()
    }
    for field_name, (key_name, value_name) in _ORDERED_MAPPING_FIELDS.items():
        if isinstance(listed_fields.get(field_name), dict):
            listed_fields[field_name] = [
                {key_name: entry_key, value_name: entry_value}
                for entry_key, entry_value in listed_fields[field_name].items()
            ]
    return listed_fields


def join_ordered_mappings(fields: Any) -> Any:
    """Recursively convert lists of key-value pair dictionaries back into ordered mappings.

    Args:
        fields: A JSON-serialized protobuf dictionary or nested value.

    Returns:
        The transformed structure with every field in ``_ORDERED_MAPPING_FIELDS``
        reconstructed as an ordered dictionary.
    """
    if isinstance(fields, list):
        return [join_ordered_mappings(field_value) for field_value in fields]
    if not isinstance(fields, dict):
        return fields
    joined_fields = {
        field_name: join_ordered_mappings(field_value)
        for field_name, field_value in fields.items()
    }
    for field_name, (key_name, value_name) in _ORDERED_MAPPING_FIELDS.items():
        if isinstance(joined_fields.get(field_name), list):
            joined_fields[field_name] = {
                pair[key_name]: pair.get(value_name)
                for pair in joined_fields[field_name]
            }
    return joined_fields
