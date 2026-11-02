"""Check answers before the file source saves them, and warn on answers that conflict.

Each item gives the rule, then the reason for it.

- An answer is refused when its case is not in the review file: a run
  dropped it after the page loaded it.
- An answer is refused when its case's saved answer is not the one its edit
  started from: another tab saved since, and that answer must not be replaced
  unseen. Saving again without the seen answer replaces it on purpose.
- An answer is refused when it breaks the run's rules, checked with
  ``ReviewCase.list_broken_answer_rules``; clearing an answer always passes.
- Two saved linked-page answers adding whole pages of one provider to one
  entry are saved with a warning, because the run adds neither page.
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
    """Return the saved answer each edit started from, for answers that carry one.

    Args:
        case_answers: The answers to save.

    Returns:
        Each case's seen saved answer (None for none), by case name; answers
        without one are left out and not checked.
    """
    return {
        case_answer.case_name: json_format.MessageToDict(case_answer.seen_saved_answer)
        for case_answer in case_answers
        if case_answer.HasField("seen_saved_answer")
    }


def read_case_answer(case_answer: review_cases_pb2.CaseAnswer) -> JsonValue:
    """Return a case answer's JSON value, or None when it is unset."""
    if not case_answer.HasField("answer"):
        return None
    return json_format.MessageToDict(case_answer.answer)


def check_case_answer(
    case: ReviewCase | None,
    case_name: str,
    answer: JsonValue,
    seen_saved_answers: Mapping[str, JsonValue],
) -> review_cases_pb2.RuleCheckResult:
    """Check one answer before saving it, and say why it is refused.

    Each item gives the refusal, then the reason for it.

    - The case is not in the review file: a run dropped it after the page
      loaded it.
    - The case's saved answer is not the one the edit started from: another
      tab saved since, and that answer must not be replaced unseen.
    - The answer breaks the run's rules; clearing an answer always passes.

    Args:
        case: The case as the review file holds it now, or None when absent.
        case_name: The case the answer is for.
        answer: The new answer; None clears it.
        seen_saved_answers: The saved answer each edit started from, by case
            name; cases left out are not checked.

    Returns:
        The result: saved, or the refusal's rule names and sentences.
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
    """Return a refused answer's result with its one rule."""
    return review_cases_pb2.RuleCheckResult(
        case_name=case_name,
        is_saved=False,
        broken_rule_names=[rule_name],
        broken_rule_sentences=[rule_sentence],
    )


def list_conflict_warnings(
    cases_by_name: Mapping[str, ReviewCase], saved_answers: Mapping[str, JsonValue]
) -> dict[str, list[str]]:
    """Warn on saved linked-page answers that add a whole page beside another of its provider.

    Args:
        cases_by_name: Every case in the file, before the save.
        saved_answers: The answers just saved, by case name.

    Returns:
        Each just-saved case in a conflict, mapped to its warnings.
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
