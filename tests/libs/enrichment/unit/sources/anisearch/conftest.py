"""Fixtures for AniSearch crawler unit tests.

Saved pages sit in the folder named after the test module that reads them
(``test_anisearch_anime_crawler/`` and so on), the layout pytest-datadir reads,
next to the values ``data_regression`` saved from them (``<page>.yml``). After an
intended change to what a crawler extracts, regenerate those files with plain
pytest (``--force-regen``), not Pants, which runs tests in a copy of the tree,
and review the diff.

Where each saved page came from:

Each item reads ``<file>: <address>, captured <date>``.

- one_piece_main.html: https://www.anisearch.com/anime/2227,one-piece, captured
  on 2026-06-09.
- one_piece_relations.html:
  https://www.anisearch.com/anime/2227,one-piece/relations?show=overall,
  captured on 2026-06-09.
- balala_the_fairies_2_main.html:
  https://www.anisearch.com/anime/11340,ba-la-la-xiao-mo-xian-zhi-qiji-wubu,
  captured by 2026-10-17.
- mo_dao_zu_shi_main.html:
  https://www.anisearch.com/anime/13751,mo-dao-zu-shi-qianchen-pian, captured by
  2026-10-17.
- twelve_sai_main.html: https://www.anisearch.com/anime/9589, captured on
  2026-10-24.
- luffy_char.html: https://www.anisearch.com/character/4852,monkey-d-luffy,
  captured on 2026-06-09.
- luffy_anime_ography.html:
  https://www.anisearch.com/character/4852,monkey-d-luffy/anime, captured on
  2026-06-09.
- luffy_manga_ography.html:
  https://www.anisearch.com/character/4852,monkey-d-luffy/manga, captured on
  2026-06-09.
- one_piece_episodes.html:
  https://www.anisearch.com/anime/2227,one-piece/episodes, captured on
  2026-04-28.
- one_piece_refs.json: https://www.anisearch.com/anime/2227,one-piece/characters
  (extracted links only), captured on 2026-04-23.
"""

from collections.abc import Callable, Iterator
from functools import cache
from pathlib import Path
from unittest.mock import patch

import pytest
from common.config import CacheConfig, RedisConfig
from enrichment.sources.anisearch import anisearch_http
from http_cache import configure_cache


@pytest.fixture(scope="module")
def read_page(original_datadir: Path) -> Callable[[str], str]:
    return cache(lambda page_name: (original_datadir / f"{page_name}.html").read_text())


@pytest.fixture(autouse=True)
def no_real_anisearch_requests() -> Iterator[None]:
    with patch.object(
        anisearch_http.ANISEARCH_CLIENT,
        "fetch",
        autospec=True,
        side_effect=AssertionError("a unit test tried to reach AniSearch"),
    ):
        yield


@pytest.fixture
def cache_off() -> None:
    configure_cache(CacheConfig(cache_enabled=False), RedisConfig())
