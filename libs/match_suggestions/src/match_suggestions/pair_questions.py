"""Hold every word Jev is sent about a pair of provider pages, and the pair's state.

Jev answers one judgment per question, and code decides what the answers mean
(``pair_relation``). Each item gives the rule, then the reason for it.

- Code compares types, episode counts, years, first air dates and air spans,
  and states each outcome as a sentence. Jev reads numbers and dates as text
  and compares them unreliably.
- The air span sentence uses the episode count and 31-day air date rules of
  ``work_matching.page_details_comparison``, so it agrees with the MAL check.
  The sentence names which fact differs, such as first air dates more than
  a month apart, because Jev reads a vague "details disagree" as agreement.
- What Jev reads is worded exactly as measured. The pages are ``page_a`` and
  ``page_b``, sorted by page key, and the questions ask about a "release",
  because "work" reads as the same story to Jev and merges a remake into its
  original. The question names follow the code's names and are never sent.
- Each page is named by its site's name, or by its address's host for a
  provider without one, so a provider fetched later needs no entry.
- The question version is a hash of the wording and the Jev model, so changing
  either asks every pair again without a version to bump by hand.
"""

import hashlib
import json
from collections.abc import Mapping
from datetime import datetime
from typing import Any, Final
from urllib.parse import urlsplit

from typesafe_sdk import Noul
from work_matching import ProviderPage, read_page_provider
from work_matching.page_details_comparison import (
    is_air_date_matching,
    is_episode_count_matching,
    resolve_last_air_date,
)

# A fixed model, so a newer Jev behind ``jev-latest`` never changes stored
# answers unnoticed.
JEV_MODEL: Final = "jev-1.14.0"
# TypeSafe's price for the model's input tokens; output tokens are free.
JEV_DOLLARS_PER_INPUT_TOKEN: Final = 0.042 / 1_000_000
_SYNOPSIS_CHARACTERS: Final = 1500
# The names Jev was measured with. A provider missing here is named by the
# host of its address, so a provider fetched later needs no entry.
_SITE_NAMES: Final = {
    "mal": "MyAnimeList",
    "anilist": "AniList",
    "kitsu": "Kitsu",
    "anidb": "AniDB",
    "animeplanet": "Anime-Planet",
    "anisearch": "aniSearch",
    "animeschedule": "AnimeSchedule",
}

PAIR_QUESTIONS: Final[dict[str, Noul]] = {
    "is_same_franchise": Noul(
        instructions="Are `page_a` and `page_b` about the same story or franchise?",
        criteria={
            "true": "Both pages are about one story, such as two seasons of one series.",
            "false": "The pages are about unrelated stories that only share words in their titles.",
        },
    ),
    "is_same_work": Noul(
        instructions="Do `page_a` and `page_b` list the same release?",
        criteria={
            "true": "Both pages list one release, such as one TV season, one film or one OVA, on two different sites.",
            "false": "The pages list different releases, such as season 1 and season 2, a TV series and its film, or a series and its recap.",
        },
    ),
    "is_first_work_part_of_second_work": Noul(
        instructions="Does `page_a` list only one part of the release that `page_b` lists?",
        criteria={
            "true": "The release on `page_b` contains `page_a`, such as `page_a` listing one cour of a two-cour season on `page_b`.",
            "false": "The release on `page_b` does not contain `page_a`, such as both pages listing the whole season.",
        },
    ),
    "is_second_work_part_of_first_work": Noul(
        instructions="Does `page_b` list only one part of the release that `page_a` lists?",
        criteria={
            "true": "The release on `page_a` contains `page_b`, such as `page_b` listing one cour of a two-cour season on `page_a`.",
            "false": "The release on `page_a` does not contain `page_b`, such as both pages listing the whole season.",
        },
    ),
}

QUESTION_VERSION: Final = hashlib.sha256(
    json.dumps(
        {
            "model": JEV_MODEL,
            "questions": {
                name: question.model_dump(mode="json")
                for name, question in PAIR_QUESTIONS.items()
            },
        },
        sort_keys=True,
    ).encode()
).hexdigest()[:16]


def build_pair_state(
    first_page: ProviderPage, second_page: ProviderPage, synopses: Mapping[str, str]
) -> dict[str, Any]:
    """Build the state Jev reads about one pair of pages.

    Args:
        first_page: The page whose key sorts first.
        second_page: The other page.
        synopses: The English synopsis of each page that has one, by page key.

    Returns:
        Both pages under ``page_a`` and ``page_b``, and the comparisons code
        made of their details under ``comparison``.
    """
    return {
        "page_a": _build_page_state(first_page, synopses.get(first_page.key)),
        "page_b": _build_page_state(second_page, synopses.get(second_page.key)),
        "comparison": {
            "type": _compare_types(first_page, second_page),
            "episodes": _compare_episode_counts(first_page, second_page),
            "years": _compare_years(first_page, second_page),
            "start_dates": _compare_start_dates(first_page, second_page),
            "air_spans": _compare_air_spans(first_page, second_page),
        },
    }


def build_request_fingerprint(pair_state: Mapping[str, Any]) -> str:
    """Hash everything sent about a pair, with the question version.

    Args:
        pair_state: The state ``build_pair_state`` built.

    Returns:
        A hex digest that changes when the state or the questions change.
    """
    request_text = json.dumps(
        {"question_version": QUESTION_VERSION, "state": pair_state}, sort_keys=True
    )
    return hashlib.sha256(request_text.encode()).hexdigest()


def resolve_page_year(page: ProviderPage) -> int | None:
    """Read the year a page gives, or the year of its first air date.

    Some providers fill only one of the two.

    Args:
        page: The provider page.

    Returns:
        The year, or None when the page gives neither.
    """
    if page.year is not None:
        return page.year
    return page.aired_from.year if page.aired_from else None


def _build_page_state(page: ProviderPage, synopsis: str | None) -> dict[str, Any]:
    """Build the state of one page, leaving out details the page does not give.

    Args:
        page: The provider page.
        synopsis: The page's English synopsis, or None.

    Returns:
        The site, titles, type, season, episode count, air days and synopsis.
    """
    page_state: dict[str, Any] = {
        "site": _SITE_NAMES.get(read_page_provider(page.key) or "")
        or (urlsplit(page.address).hostname or "").removeprefix("www."),
        "titles": list(page.titles),
        "type": page.type,
        "season": " ".join(str(part) for part in (page.season, page.year) if part),
        "episodes": page.episode_count,
        "first_aired": _read_day(page.aired_from),
        "last_aired": _read_day(page.aired_to),
        "synopsis": (synopsis or "")[:_SYNOPSIS_CHARACTERS],
    }
    return {name: value for name, value in page_state.items() if value}


def _compare_types(first_page: ProviderPage, second_page: ProviderPage) -> str:
    """Say whether both pages give the same type.

    Args:
        first_page: The first page.
        second_page: The second page.

    Returns:
        A sentence about the two types.
    """
    if not first_page.type or not second_page.type:
        return "One page gives no type."
    if first_page.type == second_page.type:
        return f"Both pages give the type {first_page.type}."
    return f"The types differ, {first_page.type} on `page_a` and {second_page.type} on `page_b`."


def _compare_episode_counts(first_page: ProviderPage, second_page: ProviderPage) -> str:
    """Say whether both pages give the same episode count.

    Args:
        first_page: The first page.
        second_page: The second page.

    Returns:
        A sentence about the two episode counts.
    """
    first_count, second_count = first_page.episode_count, second_page.episode_count
    if first_count is None or second_count is None:
        return "One page gives no episode count."
    if first_count == second_count:
        return f"Both pages list {first_count} episodes."
    return f"The episode counts differ, {first_count} on `page_a` and {second_count} on `page_b`."


def _compare_years(first_page: ProviderPage, second_page: ProviderPage) -> str:
    """Say whether both pages give the same year and season.

    Args:
        first_page: The first page.
        second_page: The second page.

    Returns:
        A sentence about the two years, naming the seasons when both differ.
    """
    first_year, second_year = (
        resolve_page_year(first_page),
        resolve_page_year(second_page),
    )
    if first_year is None or second_year is None:
        return "One page gives no year."
    if first_year != second_year:
        return (
            f"The years differ by {abs(first_year - second_year)}, "
            f"{first_year} on `page_a` and {second_year} on `page_b`."
        )
    if (
        first_page.season
        and second_page.season
        and first_page.season != second_page.season
    ):
        return (
            f"Both pages give the year {first_year}, but the seasons differ, "
            f"{first_page.season} on `page_a` and {second_page.season} on `page_b`."
        )
    return f"Both pages give the year {first_year}."


def _compare_start_dates(first_page: ProviderPage, second_page: ProviderPage) -> str:
    """Say how far apart the two first air dates are.

    Args:
        first_page: The first page.
        second_page: The second page.

    Returns:
        A sentence about the gap between the first air dates.
    """
    if first_page.aired_from is None or second_page.aired_from is None:
        return "One page gives no first air date."
    gap_days = abs((first_page.aired_from.date() - second_page.aired_from.date()).days)
    if gap_days == 0:
        return "Both pages give the same first air date."
    if gap_days <= 7:
        return "The first air dates are less than a week apart."
    if gap_days <= 62:
        return "The first air dates are a few weeks apart."
    if gap_days < 365:
        return f"The first air dates are about {round(gap_days / 30)} months apart."
    return f"The first air dates are about {round(gap_days / 365)} years apart."


def _compare_air_spans(first_page: ProviderPage, second_page: ProviderPage) -> str:
    """Say whether the page with fewer episodes airs within the other page's run.

    Args:
        first_page: The first page.
        second_page: The second page.

    Returns:
        A sentence about the two pages' air spans.
    """
    if None in (first_page.episode_count, second_page.episode_count) or None in (
        first_page.aired_from,
        second_page.aired_from,
    ):
        return "One page gives no episode count or first air date, so their air spans cannot be compared."
    if is_episode_count_matching(
        first_page.episode_count or 0, second_page.episode_count or 0
    ):
        if is_air_date_matching(first_page.aired_from, second_page.aired_from):
            return "Both pages cover about the same span: similar episode counts, first aired within a month of each other."
        return "The episode counts are similar, but the first air dates are more than a month apart."
    (shorter_name, shorter_page), (longer_name, longer_page) = sorted(
        (("page_a", first_page), ("page_b", second_page)),
        key=lambda named_page: named_page[1].episode_count or 0,
    )
    shorter_start, longer_start = shorter_page.aired_from, longer_page.aired_from
    shorter_end = resolve_last_air_date(shorter_page) or shorter_start
    longer_end = resolve_last_air_date(longer_page)
    if None in (shorter_start, longer_start, shorter_end, longer_end):
        return (
            f"`{shorter_name}` lists fewer episodes than `{longer_name}`, but a missing end date "
            f"means it is unknown whether `{shorter_name}` airs within `{longer_name}`'s run."
        )
    is_within_run = _is_on_or_after(shorter_start, longer_start) and _is_on_or_after(
        longer_end, shorter_end
    )
    if is_within_run:
        return f"`{shorter_name}` lists fewer episodes than `{longer_name}`, and all of `{shorter_name}`'s air dates fall within `{longer_name}`'s run."
    return f"`{shorter_name}` lists fewer episodes than `{longer_name}`, and `{shorter_name}` airs outside `{longer_name}`'s run."


def _is_on_or_after(later_date: datetime | None, earlier_date: datetime | None) -> bool:
    """Report whether a date falls on or after another, or within the 31 days air dates may differ.

    Args:
        later_date: The date expected to be later.
        earlier_date: The date expected to be earlier.

    Returns:
        True when ``later_date`` is not more than 31 days before ``earlier_date``.
    """
    return (
        later_date is not None
        and earlier_date is not None
        and (
            later_date >= earlier_date or is_air_date_matching(later_date, earlier_date)
        )
    )


def _read_day(aired_date: datetime | None) -> str | None:
    """Write an air date as its calendar day.

    Args:
        aired_date: The air date, or None.

    Returns:
        The day as ``YYYY-MM-DD``, or None.
    """
    return aired_date.date().isoformat() if aired_date else None
