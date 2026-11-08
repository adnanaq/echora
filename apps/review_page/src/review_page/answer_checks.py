"""Validate review answers before saving and detect conflicts between concurrent edits.

Validates user decisions against placement constraints and ensures optimistic
concurrency so edits from one browser session do not silently overwrite changes
made in another session.

Refusal conditions and warnings include:
- Unknown case: Rejects an answer if the review case no longer exists in
  storage, such as when a background pipeline run resolved the ambiguity.
- Concurrency conflict: Rejects an answer if the stored answer was modified
  after the current edit began, preventing stale overwrites. Resubmitting
  without the base revision explicitly forces the update.
- Rule violations: Rejects an answer if placement decisions violate domain rules
  (such as assigning two MyAnimeList pages to one work), checked via
  ``ReviewCase.list_broken_answer_rules``. Clearing an answer always succeeds.
- Provider conflicts: Generates a warning when multiple linked-page answers
  attempt to attach pages from the same provider to one entry, which the
  cleaning pipeline rejects.
"""

from collections.abc import Mapping, Sequence

from google.protobuf import json_format
from pydantic import JsonValue
from review_page.review_case_source import ANSWER_SAVED_ELSEWHERE_RULE_NAME
from shared_proto.v1 import review_cases_pb2
from work_matching import ReviewCase, ReviewCaseKind, find_linked_page_conflicts

_UNKNOWN_CASE_RULE_NAME = "UNKNOWN_CASE"
_UNKNOWN_CASE_SENTENCE = (
    "the case is not in the review file; a run has dropped it since the page was loaded"
)
_ANSWER_SAVED_ELSEWHERE_SENTENCE = (
    "another tab saved a different answer after this edit began; "
    "save again to replace it with this one"
)


def read_seen_saved_answers(
    case_answers: Sequence[review_cases_pb2.CaseAnswer],
) -> dict[str, JsonValue]:
    """Extract the base saved answers from submitted answer messages.

    Args:
        case_answers: The sequence of case answer messages submitted for saving.

    Returns:
        A dictionary mapping each case name to the saved answer present when
        the edit started. Cases omitting the base revision field are excluded.
    """
    return {
        case_answer.case_name: json_format.MessageToDict(case_answer.seen_saved_answer)
        for case_answer in case_answers
        if case_answer.HasField("seen_saved_answer")
    }


def read_case_answer(case_answer: review_cases_pb2.CaseAnswer) -> JsonValue:
    """Extract the JSON payload of an answer from its protobuf message.

    Args:
        case_answer: The answer message to read.

    Returns:
        The answer dictionary or decision string, or None when the field is unset.
    """
    if not case_answer.HasField("answer"):
        return None
    return json_format.MessageToDict(case_answer.answer)


def check_case_answer(
    case: ReviewCase | None,
    case_name: str,
    answer: JsonValue,
    seen_saved_answers: Mapping[str, JsonValue],
) -> review_cases_pb2.RuleCheckResult:
    """Validate a candidate answer before persisting it to the review file.

    Checks whether the case exists, detects optimistic concurrency conflicts,
    and verifies that page placements satisfy placement rules.

    Args:
        case: The review case currently stored on disk, or None when absent.
        case_name: The identifier of the review case being answered.
        answer: The candidate answer payload; None indicates clearing the answer.
        seen_saved_answers: The baseline saved answer for each case at the time
            editing began, used to detect stale writes. Cases omitted from this
            mapping bypass concurrency validation.

    Returns:
        The validation result indicating whether the answer is accepted or
        listing the broken rule codes and human-readable explanation messages.
    """
    if case is None:
        return _refuse_answer(
            case_name, _UNKNOWN_CASE_RULE_NAME, _UNKNOWN_CASE_SENTENCE
        )
    if case_name in seen_saved_answers and case.answer != seen_saved_answers[case_name]:
        return _refuse_answer(
            case_name,
            ANSWER_SAVED_ELSEWHERE_RULE_NAME,
            _ANSWER_SAVED_ELSEWHERE_SENTENCE,
        )
    broken_rules = () if answer is None else case.list_broken_answer_rules(answer)
    return review_cases_pb2.RuleCheckResult(
        case_name=case_name,
        is_saved=not broken_rules,
        broken_rule_names=[rule.name for rule in broken_rules],
        broken_rule_sentences=[rule.value for rule in broken_rules],
    )


def _refuse_answer(
    case_name: str, rule_name: str, rule_sentence: str
) -> review_cases_pb2.RuleCheckResult:
    """Construct a rejected validation result containing a single violation reason.

    Args:
        case_name: The identifier of the review case whose answer is rejected.
        rule_name: The programmatic identifier of the broken constraint.
        rule_sentence: The user-facing explanation sentence describing the refusal.

    Returns:
        A RuleCheckResult message marked as not saved with the given rule.
    """
    return review_cases_pb2.RuleCheckResult(
        case_name=case_name,
        is_saved=False,
        broken_rule_names=[rule_name],
        broken_rule_sentences=[rule_sentence],
    )


def list_conflict_warnings(
    cases_by_name: Mapping[str, ReviewCase], saved_answers: Mapping[str, JsonValue]
) -> dict[str, list[str]]:
    """Detect conflicting linked-page answers that attempt to add duplicate providers.

    Args:
        cases_by_name: The full mapping of stored review cases by case name.
        saved_answers: The newly saved answers mapped by case name.

    Returns:
        A dictionary mapping affected case names to warning messages describing
        conflicting cases that target the same work entry with the same provider.
    """
    linked_page_cases = [
        case.model_copy(
            update={"answer": saved_answers.get(case.case_name, case.answer)}
        )
        for case in cases_by_name.values()
        if case.kind is ReviewCaseKind.LINKED_PAGE
    ]
    warnings_by_case_name: dict[str, list[str]] = {}
    for conflict in find_linked_page_conflicts(linked_page_cases):
        for case_name in conflict.case_names:
            if case_name in saved_answers:
                other_case_names = ", ".join(
                    name for name in conflict.case_names if name != case_name
                )
                warnings_by_case_name.setdefault(case_name, []).append(
                    f"another answer also adds a whole {conflict.provider} page to "
                    f"{conflict.entry_reference_page}, so the run adds neither: "
                    f"{other_case_names}"
                )
    return warnings_by_case_name
