"""Compare a provider page's details with a work entry's or with a MAL page's.

Providers name types more finely than the seed, and fill years in at different
times, so a page is compared with an entry through the seed's coarser type, and
a missing year or unknown type agrees with any.

A page is also compared with the MAL page of the work it may join. A work is
what MAL lists, so the episode counts and air dates of the two pages show
whether the page is that whole work, one part of it, or several works.
"""

from datetime import datetime, timedelta
from enum import StrEnum

from work_matching.provider_page import ProviderPage
from work_matching.work_entry import (
    WorkEntry,
    WorkType,
    is_type_matching,
    is_year_matching,
)

# The seed type each provider type falls under. A provider type missing here,
# such as UNKNOWN, agrees with every seed type.
_SEED_TYPE_BY_PROVIDER_TYPE: dict[str, WorkType] = {
    "TV": "TV",
    "TV SHORT": "TV",
    "MOVIE": "MOVIE",
    "OVA": "OVA",
    "ONA": "ONA",
    "SPECIAL": "SPECIAL",
    "TV SPECIAL": "SPECIAL",
    "MUSIC": "SPECIAL",
    "CM": "SPECIAL",
    "PV": "SPECIAL",
    "OTHER": "SPECIAL",
}


def resolve_seed_type(page: ProviderPage) -> WorkType | None:
    """Return the seed type a provider page's type falls under, or None when unknown.

    Providers name types more finely than the seed (``TV SHORT``, ``MUSIC``), so
    comparing a page with an entry, or building an entry from a page, needs the
    seed's coarser type.

    Args:
        page: The provider page whose type is mapped to a seed type.

    Returns:
        The corresponding coarse ``WorkType``, or None when unknown.
    """
    return _SEED_TYPE_BY_PROVIDER_TYPE.get(page.type or "")


def is_page_details_matching(page: ProviderPage, entry: WorkEntry) -> bool:
    """Report whether a page's year and type agree with an entry's.

    A page joining an entry, by a move or as an added linked page, must agree.
    A disagreement shows the page's link may be a wrong mapping. The rules:

    - A missing year agrees with any year, because providers fill years in at
      different times.
    - An unknown type agrees with any type, for the same reason.
    - The page's type is compared as the seed's coarser type (``resolve_seed_type``).

    Args:
        page: The provider page's fetched details.
        entry: The work entry the page would join.

    Returns:
        True when both the year and the type agree.
    """
    return is_year_matching(page.year, entry.year) and is_type_matching(
        resolve_seed_type(page), entry.type
    )


# Providers give a release day differently by up to a few weeks (time zones,
# first broadcast against first stream), so air dates within this difference agree.
_MAX_AIR_DATE_DIFFERENCE = timedelta(days=31)
# Providers count a recap or special as an episode differently, so episode
# counts within this difference agree when the ratio is also within the limit.
_MAX_EPISODE_COUNT_DIFFERENCE = 2
_MAX_EPISODE_COUNT_DIFFERENCE_RATIO = 0.1


class MalPageComparison(StrEnum):
    """Compare a provider page against the MAL page of the work it may join."""

    SAME_WORK = "the page lists the same episodes as the MAL page"
    PART_OF_WORK = "the page lists fewer episodes, aired within the MAL page's run"
    SEVERAL_WORKS = (
        "the page lists more episodes, aired beyond the MAL page's run, "
        "so it covers several works"
    )
    DETAILS_DIFFER = (
        "the page's type, episode count or air dates disagree with the MAL page"
    )
    DETAILS_UNKNOWN = (
        "an episode count or air date needed to compare the page with the MAL "
        "page is unknown"
    )
    # Given by the entry-group check, not by ``compare_page_with_mal_page``.
    NO_MAL_PAGE = (
        "the suggested work has no MAL page to compare episode counts and air "
        "dates against"
    )


def compare_page_with_mal_page(
    page: ProviderPage, mal_page: ProviderPage
) -> MalPageComparison:
    """Compare a page with the MAL page of the work it may join.

    Each item reads ``<details>: <comparison>``. Checks run in this order.

    - Seed types known and different: ``DETAILS_DIFFER``.
    - An episode count unknown: ``DETAILS_UNKNOWN``.
    - Air spans more than 31 days apart: ``DETAILS_DIFFER``.
    - Counts equal, or within 2 and 10%: ``SAME_WORK`` when the start dates
      agree or one is unknown, otherwise ``DETAILS_DIFFER``.
    - Fewer episodes: ``PART_OF_WORK`` when the page aired within the MAL
      page's run and more than 31 days shorter at one end.
    - More episodes: ``SEVERAL_WORKS`` when the page aired more than 31 days
      beyond the MAL page's run at one end.
    - A count difference the dates do not explain: ``DETAILS_DIFFER``.
    - A date the last two checks need is unknown: ``DETAILS_UNKNOWN``.

    Args:
        page: The page that may join the work.
        mal_page: The MAL page of that work.

    Returns:
        The comparison.
    """
    page_seed_type, mal_seed_type = resolve_seed_type(page), resolve_seed_type(mal_page)
    if page_seed_type and mal_seed_type and page_seed_type != mal_seed_type:
        return MalPageComparison.DETAILS_DIFFER
    if page.episode_count is None or mal_page.episode_count is None:
        return MalPageComparison.DETAILS_UNKNOWN
    if not _has_overlapping_air_dates(page, mal_page):
        return MalPageComparison.DETAILS_DIFFER
    if _is_episode_count_matching(page.episode_count, mal_page.episode_count):
        if not _is_air_date_matching(page.aired_from, mal_page.aired_from):
            return MalPageComparison.DETAILS_DIFFER
        return MalPageComparison.SAME_WORK
    if page.episode_count < mal_page.episode_count:
        return _compare_fewer_episodes(page, mal_page)
    return _compare_more_episodes(page, mal_page)


def _compare_fewer_episodes(
    page: ProviderPage, mal_page: ProviderPage
) -> MalPageComparison:
    """Distinguish a part of the MAL page's work from an unexplained episode count.

    Args:
        page: The page listing fewer episodes than ``mal_page``.
        mal_page: The MAL page of the work.

    Returns:
        The ``MalPageComparison`` classification for the shorter page.
    """
    page_end_date = _resolve_last_air_date(page)
    mal_end_date = _resolve_last_air_date(mal_page)
    if page.aired_from is None or mal_page.aired_from is None or mal_end_date is None:
        return MalPageComparison.DETAILS_UNKNOWN
    is_within_run = (
        page.aired_from >= mal_page.aired_from - _MAX_AIR_DATE_DIFFERENCE
        and (page_end_date or page.aired_from)
        <= mal_end_date + _MAX_AIR_DATE_DIFFERENCE
    )
    is_shorter_run = (
        page.aired_from > mal_page.aired_from + _MAX_AIR_DATE_DIFFERENCE
        or (
            page_end_date is not None
            and page_end_date < mal_end_date - _MAX_AIR_DATE_DIFFERENCE
        )
    )
    if is_within_run and is_shorter_run:
        return MalPageComparison.PART_OF_WORK
    if is_within_run and page_end_date is None:
        # A page that started with the MAL page's run and gives no end date may
        # still end well before it.
        return MalPageComparison.DETAILS_UNKNOWN
    return MalPageComparison.DETAILS_DIFFER


def _compare_more_episodes(
    page: ProviderPage, mal_page: ProviderPage
) -> MalPageComparison:
    """Distinguish a page covering several works from an unexplained episode count.

    Args:
        page: The page listing more episodes than ``mal_page``.
        mal_page: The MAL page of the work.

    Returns:
        The ``MalPageComparison`` classification for the longer page.
    """
    page_end_date = _resolve_last_air_date(page)
    mal_end_date = _resolve_last_air_date(mal_page)
    is_starting_earlier = _is_after_max_air_date_difference(
        mal_page.aired_from, page.aired_from
    )
    is_ending_later = _is_after_max_air_date_difference(page_end_date, mal_end_date)
    if is_starting_earlier or is_ending_later:
        return MalPageComparison.SEVERAL_WORKS
    if None in (page.aired_from, mal_page.aired_from, page_end_date, mal_end_date):
        return MalPageComparison.DETAILS_UNKNOWN
    return MalPageComparison.DETAILS_DIFFER


def _is_episode_count_matching(page_count: int, mal_count: int) -> bool:
    """Report whether two episode counts differ by no more than recaps and specials.

    Args:
        page_count: Episode count reported by the provider page.
        mal_count: Episode count reported by the MAL page.

    Returns:
        True when the counts are equal or within both difference limits.
    """
    difference = abs(page_count - mal_count)
    return difference == 0 or (
        difference <= _MAX_EPISODE_COUNT_DIFFERENCE
        and difference
        <= _MAX_EPISODE_COUNT_DIFFERENCE_RATIO * max(page_count, mal_count)
    )


def _has_overlapping_air_dates(page: ProviderPage, mal_page: ProviderPage) -> bool:
    """Report whether two pages' air date spans overlap within the 31-day difference.

    Args:
        page: The provider page being compared.
        mal_page: The MAL page being compared.

    Returns:
        True when neither page's run ends more than 31 days before the other's starts.
    """
    return not (
        _is_after_max_air_date_difference(
            page.aired_from, _resolve_last_air_date(mal_page)
        )
        or _is_after_max_air_date_difference(
            mal_page.aired_from, _resolve_last_air_date(page)
        )
    )


def _is_air_date_matching(
    first_date: datetime | None, second_date: datetime | None
) -> bool:
    """Report whether two air dates are within 31 days or either is unknown.

    Args:
        first_date: The first date to compare, or None.
        second_date: The second date to compare, or None.

    Returns:
        True when either date is None or the dates differ by at most 31 days.
    """
    return not (
        _is_after_max_air_date_difference(first_date, second_date)
        or _is_after_max_air_date_difference(second_date, first_date)
    )


def _is_after_max_air_date_difference(
    later_date: datetime | None, earlier_date: datetime | None
) -> bool:
    """Report whether a known date falls more than 31 days after another known date.

    Args:
        later_date: The candidate later date, or None.
        earlier_date: The candidate earlier date, or None.

    Returns:
        True when both dates are present and ``later_date`` exceeds
        ``earlier_date`` by more than ``_MAX_AIR_DATE_DIFFERENCE``.
    """
    return (
        later_date is not None
        and earlier_date is not None
        and later_date > earlier_date + _MAX_AIR_DATE_DIFFERENCE
    )


def _resolve_last_air_date(page: ProviderPage) -> datetime | None:
    """Return a page's last air date, using the first air date for a single episode.

    A provider gives no end date for a single episode, which airs on one day.

    Args:
        page: The provider page whose last air date is resolved.

    Returns:
        The last air date of the page, or None when unknown.
    """
    if page.aired_to is not None:
        return page.aired_to
    return page.aired_from if page.episode_count == 1 else None
