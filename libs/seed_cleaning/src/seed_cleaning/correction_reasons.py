"""Turn a seed correction's reason name into the sentence a reviewer reads.

A correction keeps its reason as a name, so the reason stays when the wording
changes. The name comes from one of four places, and the correction itself
says which:

- ``review answer``: a reviewer's answer placed the page.
- ``dead page``: the provider says the page does not exist, so the run drops
  the page.
- An added page (``is_added_page``): the linked-page decision's reason, a
  ``LinkedPageReason`` name.
- Any other page: the doubled-provider decision's outcome, a
  ``DoubledProviderOutcome`` name.

A name no rule knows, as from a newer release of the rules, is shown as written,
so a correction is never hidden by its reason.
"""

from typing import Final

from seed_cleaning.seed_corrections import (
    DEAD_PAGE_REASON,
    REVIEW_ANSWER_REASON,
    SeedCorrection,
)
from work_matching import DoubledProviderOutcome, LinkedPageReason

ANSWER_REASON_SENTENCE: Final = "a reviewer's answer placed the page"
DEAD_PAGE_REASON_SENTENCE: Final = "the provider says the page does not exist"


def describe_correction_reason(correction: SeedCorrection) -> str:
    """Return the sentence for why a correction places its page where it does.

    Args:
        correction: One seed correction.

    Returns:
        The reason's sentence, or the reason name when no rule knows it.
    """
    if correction.reason == REVIEW_ANSWER_REASON:
        return ANSWER_REASON_SENTENCE
    if correction.reason == DEAD_PAGE_REASON:
        return DEAD_PAGE_REASON_SENTENCE
    reason_enum = (
        LinkedPageReason if correction.is_added_page else DoubledProviderOutcome
    )
    if correction.reason in reason_enum.__members__:
        return reason_enum[correction.reason].value
    return correction.reason
