"""Fetch every page of a seed release, then find and correct the entries that need checking.

Seed cleaning reads a downloaded seed release, never changes it, and runs in
three steps:

- Fetch: every seed entry's pages, then every page they link that no seed
  entry holds, go into the page store. A rerun fetches only what is not stored
  yet, so a new release fetches only its new pages.
- First pass: entries with a doubled provider are placed, and pages entries
  link that no seed entry holds are added.
- Second pass: split-work candidates are found on the entries as the first
  pass leaves them, leaving out entries with a first-pass case still open, and
  placed. Its corrections stand in for the first pass's for the same pages.
- After both passes, each added page the anime id of its seed entry now comes
  from is marked, on the entries as the merged corrections leave them.

It writes to the data folder:

- ``provider_pages.jsonl``: every fetched page's details and links;
- ``entries_to_check.json``: the entries with a doubled provider, and the
  groups of split-work candidates with the titles and links that made them;
- ``review.jsonl``: one case per entry group the rules cannot place, and per
  linked page they cannot add on their own, where a person writes the answer;
  answers are kept across runs;
- ``corrections.jsonl``: where each page of an entry group ends up, and each
  linked page added. Rewritten on each run;
- ``not_listed_searches.jsonl``: the AnimeSchedule searches that found nothing
  in this release, which are not repeated until a new release;
- ``logs/clean_seed_<release>.log``: everything the run reports, one file per
  seed release, appended on each run, so each release's fetch failures stay
  readable after later releases.

The fetch keeps to each provider's limits:

- a gap between one provider's network requests; an answer from the cache
  never waits;
- after failures in a row, a pause probing every 30 minutes, given up after
  6 hours so the run still ends; the pages left wait for the next run;
- AniDB's stand-down after a ban equal to the probe interval, so the probe
  reaches AniDB.

Cache entries are kept 30 days during development, so enrichment still finds
the records of a fetch that took days; the default lifetime is 7 days.
Environment variables already set win over these defaults.

Usage:
    uv run python scripts/clean_seed.py [SEED_FILE] [--data-dir DIR]
        [--store PATH] [--no-fetch | --refetch-stored]
        (pacing options: see --help)
"""

import argparse
import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path

from common.config import get_settings
from enrichment.pipeline.circuit_breaker_api_fetcher import CircuitBreakerApiFetcher
from enrichment.pipeline.fetch_summary import describe_fetch_summary
from enrichment.pipeline.provider_page_fetcher import ProviderPageFetcher
from enrichment.sources.animeschedule.animeschedule_helper import require_app_token
from enrichment.sources.mal import mal_anime_crawler
from http_cache import close_cache, configure_cache
from seed_cleaning import (
    NotListedSearchFile,
    fetch_seed_and_linked_pages,
    read_seed_release_name,
    run_seed_cleaning,
)
from work_matching import (
    KnownWorkEntriesFromSeed,
    ProviderPageStore,
)

logger = logging.getLogger(__name__)

# Thirty days, in seconds, for every provider's cache entries: a fetch over the
# whole seed takes days, and enrichment should still find its records. A
# development setting; the cache's default of 7 days applies again later.
_CACHE_TTL_SECONDS = str(30 * 24 * 60 * 60)
_CACHE_TTL_VARIABLES = (
    "TTL_MAL",
    "TTL_ANILIST",
    "TTL_KITSU",
    "TTL_ANIDB",
    "TTL_ANIME_PLANET",
    "TTL_ANISEARCH",
    "TTL_ANIMESCHEDULE",
)
# Thirty minutes, in seconds: the wait between probes of a paused provider.
_SECONDS_BETWEEN_PROBES = 30 * 60
# Failed pages listed after a fetch; every failure is also in the log.
_FAILED_PAGES_LISTED = 20


def add_pacing_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the options pacing the fetch to each provider's limits.

    Args:
        parser: The script's argument parser.
    """
    parser.add_argument(
        "--seconds-between-requests",
        default=2.0,
        type=float,
        help="Wait between one provider's network requests; cache hits never wait (default: %(default)s)",
    )
    parser.add_argument(
        "--failures-before-pausing",
        default=10,
        type=int,
        help="Failed requests in a row that pause a provider (default: %(default)s)",
    )
    parser.add_argument(
        "--seconds-between-probes",
        default=_SECONDS_BETWEEN_PROBES,
        type=float,
        help="Wait between requests of a paused provider (default: %(default)s)",
    )
    parser.add_argument(
        "--hours-before-giving-up",
        default=6.0,
        type=float,
        help="Time paused before a provider is given up on; 0 never (default: %(default)s)",
    )
    parser.add_argument(
        "--seconds-before-mal-pictures",
        default=1.5,
        type=float,
        help="Wait between a MAL anime page and its /pics page (default: %(default)s)",
    )


def parse_arguments() -> argparse.Namespace:
    """Read the seed file, the data folder and how to fetch from the command line.

    Returns:
        The arguments, one attribute per option, named as in the usage above.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "seed_file",
        nargs="?",
        default="assets/seed_data/anime-offline-database-minified.json",
        type=Path,
        help="The downloaded seed release (default: %(default)s)",
    )
    parser.add_argument(
        "--data-dir",
        default="data/seed_cleaning",
        type=Path,
        help="Where the pages, cases, corrections and logs are written (default: %(default)s)",
    )
    parser.add_argument(
        "--store",
        type=Path,
        help="The provider page store (default: provider_pages.jsonl in the data folder)",
    )
    fetch_choice = parser.add_mutually_exclusive_group()
    fetch_choice.add_argument(
        "--no-fetch",
        action="store_true",
        help="Fetch no pages; correct the entries from the pages already stored",
    )
    fetch_choice.add_argument(
        "--refetch-stored",
        action="store_true",
        help="Fetch stored pages again, so their details gain fields added since",
    )
    add_pacing_arguments(parser)
    return parser.parse_args()


def configure_run_log(log_path: Path) -> None:
    """Send the run's report to the terminal and append it to the release's log file.

    Seed cleaning reports its plan, progress and counts at INFO; other
    libraries report only warnings, so the log holds what the run did and
    what went wrong.

    Args:
        log_path: The log file of the seed release being cleaned.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, "a", "utf-8")],
    )
    for logger_name in (
        __name__,
        "seed_cleaning",
        "enrichment.pipeline.circuit_breaker_api_fetcher",
        "enrichment.pipeline.provider_page_fetcher",
    ):
        logging.getLogger(logger_name).setLevel(logging.INFO)


def set_fetch_defaults(arguments: argparse.Namespace) -> None:
    """Set the cache lifetimes, AniDB's stand-down and MAL's picture wait for the fetch.

    The settings read these variables when ``get_settings`` first builds them,
    so this runs before that. Variables already set are kept.

    Args:
        arguments: The command-line arguments, for the probe interval and
            MAL's picture wait.
    """
    for variable in _CACHE_TTL_VARIABLES:
        os.environ.setdefault(variable, _CACHE_TTL_SECONDS)
    os.environ.setdefault(
        "ANIDB_BAN_COOLDOWN", f"{arguments.seconds_between_probes:.0f}"
    )
    # The crawler reads its pause on each call, so this run's value applies.
    mal_anime_crawler._INTER_REQUEST_DELAY = arguments.seconds_before_mal_pictures


async def fetch_pages(
    known_work_entries: KnownWorkEntriesFromSeed,
    store: ProviderPageStore,
    release_name: str,
    arguments: argparse.Namespace,
) -> None:
    """Fetch every seed entry's pages and their unheld linked pages, paced per provider.

    Args:
        known_work_entries: Every seed entry.
        store: The provider page store the fetch adds pages to.
        release_name: The seed release, so an AnimeSchedule search that found
            nothing earlier in it is not repeated.
        arguments: The command-line arguments, for the data folder, the
            pacing and ``refetch_stored``.
    """
    set_fetch_defaults(arguments)
    settings = get_settings()
    # Every AnimeSchedule search would fail without the token; stop before fetching.
    require_app_token(settings.enrichment.animeschedule_token)
    cache = configure_cache(settings.cache, settings.redis)
    # The gap applies only to requests that reach a provider; cache hits never wait.
    cache.network_request_pacer.set_seconds_between_requests(
        arguments.seconds_between_requests
    )
    try:
        seed_fetch_summary = await fetch_seed_and_linked_pages(
            known_work_entries,
            store,
            ProviderPageFetcher(
                lambda: CircuitBreakerApiFetcher(
                    settings.enrichment,
                    cache,
                    failures_before_pausing=arguments.failures_before_pausing,
                    seconds_between_probes=arguments.seconds_between_probes,
                    seconds_before_giving_up=arguments.hours_before_giving_up * 3600
                    or None,
                ),
                store,
                is_stored_page_fetched_again=arguments.refetch_stored,
                not_listed_searches=NotListedSearchFile(
                    arguments.data_dir / "not_listed_searches.jsonl", release_name
                ),
            ),
        )
    finally:
        await close_cache()
    for pages_name, fetch_summary in [
        ("Seed pages", seed_fetch_summary.seed_page_summary),
        ("Linked pages", seed_fetch_summary.linked_page_summary),
        ("AnimeSchedule searches", seed_fetch_summary.animeschedule_search_summary),
    ]:
        for report_line in describe_fetch_summary(
            pages_name, fetch_summary, _FAILED_PAGES_LISTED
        ):
            logger.info(report_line)
    logger.info(f"Pages in the store: {len(store)}")


async def main() -> None:
    """Fetch the seed release's pages, then run both passes and write their results."""
    arguments = parse_arguments()
    seed_document = json.loads(arguments.seed_file.read_text(encoding="utf-8"))
    release_name = read_seed_release_name(seed_document)
    configure_run_log(arguments.data_dir / "logs" / f"clean_seed_{release_name}.log")
    logger.info(
        f"=== Run started {datetime.now():%Y-%m-%d %H:%M:%S}: release {release_name} "
        f"({arguments.seed_file}) ==="
    )
    known_work_entries = KnownWorkEntriesFromSeed(seed_document["data"])
    store = ProviderPageStore(
        arguments.store or arguments.data_dir / "provider_pages.jsonl"
    )
    if not arguments.no_fetch:
        await fetch_pages(known_work_entries, store, release_name, arguments)
    run_seed_cleaning(
        seed_document,
        arguments.seed_file,
        known_work_entries,
        store,
        arguments.data_dir,
    )


if __name__ == "__main__":
    asyncio.run(main())
