"""Fetch every page of a seed release, then find and correct the entries that need checking.

Seed cleaning reads a downloaded seed release without modifying the source file.
Each item reads ``<step>: <action>``:

- Fetch: Every seed entry's pages, followed by every linked page that no seed
  entry holds, are fetched into the page store. A rerun fetches only pages not
  stored yet, so a new release fetches only its added pages.
- First pass: Entries with a doubled provider are placed, and linked pages that
  no seed entry holds are added.
- Second pass: Split-work candidates are found on the entries as the first pass
  leaves them, excluding entries with an open first-pass review case, and
  placed. Second-pass corrections replace first-pass corrections for the same
  pages.
- Anime ID marking: After both passes, each added page that now supplies the
  anime ID of its seed entry is marked on the merged entries.

The script writes the following files to the data directory.
Each item reads ``<file>: <purpose>``:

- ``provider_pages.jsonl``: Every fetched page's details and links.
- ``entries_to_check.json``: The entries with a doubled provider and the groups
  of split-work candidates with the titles and links that grouped them.
- ``review.jsonl``: One review case per entry group or linked page that the
  rules cannot resolve automatically, preserving saved answers across runs.
- ``corrections.jsonl``: The final placement of each checked entry group's pages
  and each added linked page, rewritten on every run.
- ``not_listed_searches.jsonl``: AnimeSchedule searches that returned no match
  in this release, skipped until a new release is used.
- ``logs/clean_seed_<release>.log``: Run progress and fetch failure reports,
  appended per seed release so earlier release logs remain available.

Network fetching respects each provider's rate limits.
Each item reads ``<mechanism>: <behavior>``:

- Request spacing: Enforces a delay between network requests to the same
  provider while serving cache hits immediately.
- Circuit breaker: Pauses a provider after consecutive failures and probes
  every 30 minutes, giving up after 6 hours so the run still terminates.
- AniDB cooldown: Matches AniDB's ban stand-down to the probe interval so the
  next probe reaches AniDB after the cooldown expires.

Cache entries are retained for 30 days during development so enrichment can
reuse records from multi-day fetches, overriding the default 7-day TTL unless
already set in the environment.

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
_CACHE_TTL_ENV_VARS = (
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
# Maximum number of failed pages listed in the summary; all failures are logged.
_MAX_LISTED_FAILED_PAGES = 20


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


def parse_arguments() -> argparse.Namespace:
    """Parse the seed file path, data directory, and fetch options from the command line.

    Returns:
        The parsed command-line arguments namespace.
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


def add_pacing_arguments(parser: argparse.ArgumentParser) -> None:
    """Add command-line options that pace fetching to each provider's limits.

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


def configure_run_log(log_path: Path) -> None:
    """Configure logging to stream to the terminal and append to the release log file.

    Seed cleaning reports its plan, progress, and counts at INFO level, while
    other libraries log only warnings so the log records run actions and errors.

    Args:
        log_path: The log file path for the seed release being cleaned.
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


async def fetch_pages(
    known_work_entries: KnownWorkEntriesFromSeed,
    store: ProviderPageStore,
    release_name: str,
    arguments: argparse.Namespace,
) -> None:
    """Fetch every seed entry's pages and their unheld linked pages, paced per provider.

    Args:
        known_work_entries: Every seed entry in the release.
        store: The provider page store that receives fetched pages.
        release_name: The seed release identifier used to skip AnimeSchedule
            searches that already returned no match in this release.
        arguments: The parsed command-line arguments controlling data paths,
            pacing intervals, and ``refetch_stored``.
    """
    apply_fetch_defaults(arguments)
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
    for summary_label, fetch_summary in [
        ("Seed pages", seed_fetch_summary.seed_page_summary),
        ("Linked pages", seed_fetch_summary.linked_page_summary),
        ("AnimeSchedule searches", seed_fetch_summary.animeschedule_search_summary),
    ]:
        for report_line in describe_fetch_summary(
            summary_label, fetch_summary, _MAX_LISTED_FAILED_PAGES
        ):
            logger.info(report_line)
    logger.info(f"Pages in the store: {len(store)}")


def apply_fetch_defaults(arguments: argparse.Namespace) -> None:
    """Apply default cache lifetimes, AniDB cooldown, and MAL picture delay for fetching.

    Settings read these environment variables when ``get_settings`` first
    initializes the configuration, while any variables already set in the
    environment are preserved.

    Args:
        arguments: The parsed command-line arguments providing the probe
            interval and MAL picture delay.
    """
    for env_var_name in _CACHE_TTL_ENV_VARS:
        os.environ.setdefault(env_var_name, _CACHE_TTL_SECONDS)
    os.environ.setdefault(
        "ANIDB_BAN_COOLDOWN", f"{arguments.seconds_between_probes:.0f}"
    )
    # The crawler reads its pause on each call, so this run's value applies.
    mal_anime_crawler._INTER_REQUEST_DELAY = arguments.seconds_before_mal_pictures


if __name__ == "__main__":
    asyncio.run(main())
