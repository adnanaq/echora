"""Hold what Jev answered about one pair of provider pages."""

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

Probability = Annotated[float, Field(ge=0.0, le=1.0)]


class PairJudgment(BaseModel):
    """Hold Jev's four probabilities for one pair of pages and what they answer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # The pair's page keys, the lower key first, so one pair has one name.
    first_page_key: str
    second_page_key: str
    # A hash of the question wording and the Jev model, so changing either
    # leaves every stored answer unused.
    question_version: str
    # A hash of everything sent about the two pages, so a pair whose details
    # changed is asked again.
    request_fingerprint: str
    is_same_franchise: Probability
    is_same_work: Probability
    is_first_work_part_of_second_work: Probability
    is_second_work_part_of_first_work: Probability
    judged_at: datetime

    @property
    def page_key_pair(self) -> tuple[str, str]:
        """Return the pair's page keys, the lower key first."""
        return (self.first_page_key, self.second_page_key)


def build_page_key_pair(page_key: str, other_page_key: str) -> tuple[str, str]:
    """Name a pair of pages by its two page keys, the lower key first.

    Args:
        page_key: One page of the pair.
        other_page_key: The other page of the pair.

    Returns:
        The two page keys in sorted order.

    Examples:
        >>> build_page_key_pair("mal:anime:2", "anilist:anime:9")
        ('anilist:anime:9', 'mal:anime:2')
    """
    first_page_key, second_page_key = sorted((page_key, other_page_key))
    return (first_page_key, second_page_key)
