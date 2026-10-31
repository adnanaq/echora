"""Report what a page fetch came to: a count per outcome, and each failure."""

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum


class FetchOutcome(StrEnum):
    """What fetching one page, or one AnimeSchedule search, came to."""

    FETCHED = "fetched"
    # The provider said the page does not exist.
    DEAD = "dead"
    # AnimeSchedule has no page for the page searched by: nothing to store.
    NOT_LISTED = "not listed"
    FAILED = "failed"
    ALREADY_STORED = "already stored"
    # An AnimeSchedule search that found nothing earlier in this seed release.
    NOT_LISTED_EARLIER = "not listed earlier in this release"


@dataclass(frozen=True)
class FetchSummary:
    """How a fetch run went: a count per outcome, and why each failure failed."""

    outcomes: Mapping[FetchOutcome, int]
    # Each failed request's error, by its label: the page key, or
    # "animeschedule via <page key searched by>".
    errors_by_request_label: Mapping[str, str]


def combine_fetch_summaries(*summaries: FetchSummary) -> FetchSummary:
    """Add up the outcomes and failures of several fetches into one summary.

    Args:
        *summaries: The summaries of the fetches.

    Returns:
        One summary with every count added and every failure kept.
    """
    outcomes: Counter[FetchOutcome] = Counter()
    errors_by_request_label: dict[str, str] = {}
    for summary in summaries:
        outcomes.update(summary.outcomes)
        errors_by_request_label |= summary.errors_by_request_label
    return FetchSummary(
        outcomes=dict(outcomes), errors_by_request_label=errors_by_request_label
    )


def describe_fetch_summary(
    pages_name: str, fetch_summary: FetchSummary, listed_failure_count: int
) -> list[str]:
    """Describe a fetch in report lines: a count per outcome, then the first failures.

    Only the first failures are listed, because a fetch over a whole seed can
    fail thousands of pages; each failure is logged when it happens.

    Args:
        pages_name: The name of the pages fetched, starting each line.
        fetch_summary: The fetch's summary.
        listed_failure_count: How many failed pages to list.

    Returns:
        One line per outcome, in outcome name order, the failure count, then
        one line per listed failure with its error cut to 160 characters.
    """
    failed_pages = list(fetch_summary.errors_by_request_label.items())
    return [
        *(
            f"{pages_name} {outcome}: {page_count}"
            for outcome, page_count in sorted(fetch_summary.outcomes.items())
        ),
        f"{pages_name} failed, retried next run: {len(failed_pages)}",
        *(
            f"  {request_label}: {fetch_error[:160]}"
            for request_label, fetch_error in failed_pages[:listed_failure_count]
        ),
    ]
