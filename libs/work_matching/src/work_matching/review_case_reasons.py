"""Turn a review case's reason names into the sentences a reviewer reads.

A case keeps its reasons as names (``PAGES_OF_OTHER_WORKS``), so they stay
when the wording changes. A reviewer needs the sentence each name stands for.
One name can mean different things in different rules (``DETAILS_DIFFER`` is
both a doubled-provider review reason and a MAL comparison), so the name alone
is not enough. Each item gives the rule, then the reason for it.

- An entry group's case writes its reasons in a fixed order
  (``_list_entry_group_reason_names``): each decision's outcome and review
  reasons, then the split-work reasons, then the MAL comparisons. The
  decisions are kept in the case with their typed outcome and reasons, so
  their sentences come from them, and the names after them can only be
  split-work reasons or MAL comparisons, whose names never repeat.
- A linked-page case has one reason, its decision's.
- A name no rule knows, as from a newer release of these rules, is shown as
  written, so a case is never hidden by its reasons.
"""

from work_matching.doubled_provider_decision import DoubledProviderDecision
from work_matching.linked_page_check import LinkedPageDecision
from work_matching.page_details_comparison import MalPageComparison
from work_matching.review_cases import ReviewCase
from work_matching.split_work_check import SplitWorkReviewReason

# Every name that can follow an entry group's decisions in its reasons.
_SENTENCE_BY_LATER_REASON_NAME: dict[str, str] = {
    **{reason.name: reason.value for reason in SplitWorkReviewReason},
    **{comparison.name: comparison.value for comparison in MalPageComparison},
}


def describe_case_reasons(case: ReviewCase) -> tuple[str, ...]:
    """Return the sentence of each of a case's reasons, in the case's order.

    Args:
        case: The review case.

    Returns:
        One sentence per name in ``case.reasons``.
    """
    decision_sentences: list[str] = []
    for decision in case.decisions:
        if isinstance(decision, LinkedPageDecision):
            decision_sentences.append(decision.reason.value)
        else:
            decision_sentences.extend(_describe_decision(decision))
    later_reason_names = case.reasons[len(decision_sentences) :]
    return (
        *decision_sentences,
        *(
            _SENTENCE_BY_LATER_REASON_NAME.get(name, name)
            for name in later_reason_names
        ),
    )


def _describe_decision(decision: DoubledProviderDecision) -> list[str]:
    """Return a doubled-provider decision's outcome and review reasons as sentences.

    Each sentence starts with the decision's provider. A case can hold one
    decision per doubled provider, often with the same reasons, and the
    provider tells the reviewer which pages each sentence is about.
    """
    return [
        f"{decision.provider}: {sentence}"
        for sentence in (
            decision.outcome.value,
            *(reason.value for reason in decision.review_reasons),
        )
    ]
