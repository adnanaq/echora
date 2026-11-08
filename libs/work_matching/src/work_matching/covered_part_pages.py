"""Find the part pages of a provider that splits a work which another provider's page covers.

A provider that splits a work lists each part as its own page, while another
provider may list one page for one part or for several parts in a row. The
episode counts and air dates of the pages show which parts a page covers. The
episode counts and dates agree within the limits the MAL comparison uses
(``page_details_comparison``).
"""

from collections.abc import Sequence
from datetime import datetime, timedelta

from work_matching.page_details_comparison import (
    is_air_date_matching,
    is_episode_count_matching,
    resolve_last_air_date,
)
from work_matching.provider_page import ProviderPage

# Providers record a release day in different time zones, so dates within this
# difference are the same day.
_SAME_AIR_DAY_DIFFERENCE = timedelta(days=1)


def find_covered_part_pages(
    page: ProviderPage, part_pages: Sequence[ProviderPage]
) -> tuple[ProviderPage, ...]:
    """Find the part pages a page covers, by episode count and air dates.

    A provider that splits a work lists each part as its own page, while
    another provider may list one page for one part or for several parts in a
    row. Types are not compared, because providers call the same special a
    special, an ONA or an OVA. The rules:

    - A page whose episode count or start date is unknown covers nothing,
      since nothing can be compared.
    - A candidate is one part, or a run of parts in air order, whose episodes
      added together agree with the page's, whose first start agrees with the
      page's start, and whose latest end agrees with the page's end.
    - A part starting within one day of the page's start must be the
      candidate's first part, and a part ending within one day of the page's
      end must be in the candidate. Short parts aired a few weeks apart all
      fall within the tolerance, so the parts lining up with the page win.
    - Candidates are narrowed in tiers, and the first tier holding any
      candidate decides. The tiers are equal episodes with dates within one
      day, then equal episodes, then every candidate. Providers record dates
      in different time zones, so one day apart is the same day.
    - The deciding tier must hold exactly one candidate. Several candidates
      cannot be told apart, so the page covers nothing.

    Episode counts and dates otherwise agree within the limits the MAL
    comparison uses.

    Args:
        page: The page compared with the parts.
        part_pages: The part pages of the provider that splits the work.

    Returns:
        The covered part pages in air order, or an empty tuple when the page
        covers no part or the parts it covers cannot be told.
    """
    if page.episode_count is None or page.aired_from is None:
        return ()
    dated_parts = sorted(
        (
            part_page
            for part_page in part_pages
            if part_page.aired_from is not None and part_page.episode_count is not None
        ),
        key=lambda part_page: part_page.aired_from or datetime.min,
    )
    return _choose_closest_run(page, _list_covering_runs(page, dated_parts))


def _list_covering_runs(
    page: ProviderPage, dated_parts: Sequence[ProviderPage]
) -> list[tuple[ProviderPage, ...]]:
    """List the runs of parts whose episodes and air dates agree with a page.

    A part starting within one day of the page's start must be the run's first
    part, and a part ending within one day of the page's end must be in the
    run, so the parts lining up with the page win.

    Args:
        page: The page compared with the parts, with a start date.
        dated_parts: The part pages with an episode count and a start date,
            in air order.

    Returns:
        Each covering run of parts, in air order of its first part.
    """
    page_end_date = resolve_last_air_date(page)
    first_anchor_parts = [
        part_page
        for part_page in dated_parts
        if _is_same_air_day(page.aired_from, part_page.aired_from)
    ]
    last_anchor_parts = [
        part_page
        for part_page in dated_parts
        if page_end_date is not None
        and (part_end_date := resolve_last_air_date(part_page)) is not None
        and _is_same_air_day(page_end_date, part_end_date)
    ]
    return [
        candidate
        for first_index in range(len(dated_parts))
        for last_index in range(first_index, len(dated_parts))
        if _is_covering_run(
            page, candidate := tuple(dated_parts[first_index : last_index + 1])
        )
        and (not first_anchor_parts or candidate[0] in first_anchor_parts)
        and (
            not last_anchor_parts
            or any(part_page in last_anchor_parts for part_page in candidate)
        )
    ]


def _choose_closest_run(
    page: ProviderPage, candidates: Sequence[tuple[ProviderPage, ...]]
) -> tuple[ProviderPage, ...]:
    """Choose the one run of parts closest to a page, by tiers.

    The tiers are equal episodes with dates within one day, then equal
    episodes, then every candidate. The first tier holding any candidate
    decides, and that tier must hold exactly one.

    Args:
        page: The page compared with the parts.
        candidates: The covering runs of parts.

    Returns:
        The chosen run, or an empty tuple when no tier holds exactly one run.
    """
    exact_candidates = [
        candidate
        for candidate in candidates
        if _count_run_episodes(candidate) == page.episode_count
    ]
    closest_candidates = [
        candidate
        for candidate in exact_candidates
        if _is_same_air_day(page.aired_from, candidate[0].aired_from)
        and _is_same_air_day(
            resolve_last_air_date(page), _resolve_run_end_date(candidate)
        )
    ]
    for tier_candidates in (closest_candidates, exact_candidates, candidates):
        if tier_candidates:
            return tier_candidates[0] if len(tier_candidates) == 1 else ()
    return ()


def _is_covering_run(page: ProviderPage, run_parts: Sequence[ProviderPage]) -> bool:
    """Report whether a page's episodes and air dates agree with a run of parts.

    Args:
        page: The page compared with the run.
        run_parts: One or more parts in air order, each with an episode count
            and a start date.

    Returns:
        True when the episode counts added together, the first part's start
        and the latest end among the parts all agree with the page's. The
        latest end is used because a short part, such as one OVA, can air
        within a longer part's run.
    """
    return (
        page.episode_count is not None
        and is_episode_count_matching(
            page.episode_count, _count_run_episodes(run_parts)
        )
        and is_air_date_matching(page.aired_from, run_parts[0].aired_from)
        and is_air_date_matching(
            resolve_last_air_date(page), _resolve_run_end_date(run_parts)
        )
    )


def _count_run_episodes(run_parts: Sequence[ProviderPage]) -> int:
    """Add up the episodes of a run of parts.

    Args:
        run_parts: Parts whose episode counts are known.

    Returns:
        The episodes of every part added together.
    """
    return sum(part_page.episode_count or 0 for part_page in run_parts)


def _resolve_run_end_date(run_parts: Sequence[ProviderPage]) -> datetime | None:
    """Return the latest last air date among a run of parts, or None when none is known.

    Args:
        run_parts: Parts in air order.

    Returns:
        The latest known last air date of the parts.
    """
    return max(
        (
            end_date
            for part_page in run_parts
            if (end_date := resolve_last_air_date(part_page)) is not None
        ),
        default=None,
    )


def _is_same_air_day(first_date: datetime | None, second_date: datetime | None) -> bool:
    """Report whether two air dates are at most one day apart, or either is unknown.

    Args:
        first_date: The first date to compare, or None.
        second_date: The second date to compare, or None.

    Returns:
        True when either date is None or the dates are within one day.
    """
    return (
        first_date is None
        or second_date is None
        or abs(first_date - second_date) <= _SAME_AIR_DAY_DIFFERENCE
    )
