"""Decide how two provider pages relate from Jev's answers about them.

Jev gives judgments and this decision table turns them into a relation. A Noul
reads as yes at 0.7 and above and as no at 0.3 and below, and the band between
leaves the question unsettled. Each item gives the rule, then the reason for it.

- The same work with neither work clearly part of the other is ``SAME_WORK``,
  but only when both pages give a year. Without one, two works sharing a title
  cannot be told apart, by a person or by Jev.
- The same work while one work may be part of the other stays ``UNSETTLED``,
  because an unsure answer there is where parts were merged into their whole.
- One work part of the other, while the work is not clearly the same, makes the
  pages ``PARTIAL_IN`` each other's work, as the review answer writes it.
- A clearly different work is ``DIFFERENT_WORKS``.
- Anything else stays ``UNSETTLED`` for a reviewer.
"""

from enum import StrEnum
from typing import Final

from match_suggestions.model_pair_judgment import PairJudgment
from match_suggestions.pair_questions import resolve_page_year
from work_matching import ProviderPage

HIGH_THRESHOLD: Final = 0.7
LOW_THRESHOLD: Final = 0.3


class PairRelation(StrEnum):
    """Name how two provider pages relate."""

    SAME_WORK = "same work"
    PARTIAL_IN = "partial in"
    DIFFERENT_WORKS = "different works"
    UNSETTLED = "unsettled"


def decide_pair_relation(
    pair_judgment: PairJudgment, first_page: ProviderPage, second_page: ProviderPage
) -> PairRelation:
    """Decide the relation of a pair from Jev's answers, by the rules above.

    Args:
        pair_judgment: Jev's answers about the pair.
        first_page: One page of the pair.
        second_page: The other page of the pair.

    Returns:
        The relation the answers decide.
    """
    is_same_work = pair_judgment.is_same_work
    part_probability = read_part_probability(pair_judgment)
    if is_same_work >= HIGH_THRESHOLD and part_probability <= LOW_THRESHOLD:
        has_years = None not in (
            resolve_page_year(first_page),
            resolve_page_year(second_page),
        )
        return PairRelation.SAME_WORK if has_years else PairRelation.UNSETTLED
    if is_same_work >= HIGH_THRESHOLD:
        return PairRelation.UNSETTLED
    if part_probability >= HIGH_THRESHOLD:
        return PairRelation.PARTIAL_IN
    if is_same_work <= LOW_THRESHOLD:
        return PairRelation.DIFFERENT_WORKS
    return PairRelation.UNSETTLED


def read_part_probability(pair_judgment: PairJudgment) -> float:
    """Read how likely one work of the pair is part of the other, either way.

    Args:
        pair_judgment: Jev's answers about the pair.

    Returns:
        The higher of the two part probabilities.
    """
    return max(
        pair_judgment.is_first_work_part_of_second_work,
        pair_judgment.is_second_work_part_of_first_work,
    )
