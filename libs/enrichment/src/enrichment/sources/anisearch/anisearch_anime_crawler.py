"""Crawls anime information from anisearch.com over plain HTTP + lxml XPath.

Extracts metadata and relations using lxml XPath on raw page HTML.
Results are cached in Redis. Two page fetches per anime (main +
/relations?show=overall), spaced out like every AniSearch request; see
``anisearch_http``.
"""

import html
import logging
import re
from typing import Any, cast

from enrichment.sources.anisearch.anisearch_anime_models import (
    AniSearchAnime,
)
from enrichment.sources.anisearch.anisearch_http import fetch_anisearch_page
from enrichment.sources.anisearch.anisearch_mapper import anime_from_anisearch
from enrichment.sources.base.exceptions import ServiceParseError
from enrichment.sources.base.framework import (
    BaseCrawler,
    FileRepository,
    NullRepository,
)
from http_cache.result_cache import cached_result

logger = logging.getLogger(__name__)


BASE_ANIME_URL = "https://www.anisearch.com/anime/"
_ANISEARCH_BASE_URL = "https://www.anisearch.com"

_VOTES_RE = re.compile(r"(\d[\d,]*)\s+members? have rated it")
_IMG_SRC_RE = re.compile(r'<img src="([^"]+)"')
# The tooltip AniSearch puts before an episode count it has only estimated, such
# as the whole run of an anime still airing. Sometimes "Runtime: not confirmed"
# follows in the same tooltip.
_ESTIMATED_EPISODE_COUNT_TOOLTIP = "Number of episodes: preliminary estimate"

# ---------------------------------------------------------------------------
# XPath selectors
# ---------------------------------------------------------------------------

_XPATHS: dict[str, str] = {
    # Main page
    "cover_image": "//section[@id='information']//img[@id='details-cover']/@src",
    # One block per language, the work's original language first
    "title_blocks": "//section[@id='information']//div[contains(@class,'title') and @lang]",
    "title_block_main": ".//strong[contains(@class,'f16')]",
    "title_block_native": ".//div[contains(@class,'grey')]",
    "type": "//section[@id='information']//div[contains(@class,'type')]",
    "type_tooltips": "//section[@id='information']//div[contains(@class,'type')]//@data-tooltip",
    "status": "//section[@id='information']//div[contains(@class,'status')]",
    "published": "//section[@id='information']//div[contains(@class,'released')]",
    "studio": "//section[@id='information']//div[contains(@class,'company')]//a[contains(@href,'company')]",
    "studio_url": "//section[@id='information']//div[contains(@class,'company')]//a[contains(@href,'company')]/@href",
    "broadcast_raw": "//section[@id='information']//div[contains(@class,'broadcast')]",
    "source_material": "//section[@id='information']//div[contains(@class,'adapted')]",
    "synonyms": "//section[@id='information']//div[contains(@class,'synonyms')]",
    "description": "//section[@id='description']//div[contains(@class,'textblock') and contains(@class,'details-text')]",
    "genres": "//section[@id='genres-tags']//ul[contains(@class,'cloud')]//a[contains(@href,'/genre/main/') or contains(@href,'/genre/subsidiary/')]",
    "tags": "//section[@id='genres-tags']//ul[contains(@class,'cloud')]//a[contains(@href,'/genre/tag/')]",
    "rating_score": "//*[@id='ratingstats']//tr[2]//td[1]//b",
    "rank_toplist": "//*[@id='ratingstats']//tr[2]//td[2]//b",
    "rank_trending": "//*[@id='ratingstats']//tr[3]//td[2]//b",
    "rating_votes": "//ul[@id='rating-stats']//a/@aria-label",
    "websites": "//section[@id='information']//div[contains(@class,'websites')]//a",
    # Relations sub-page
    "anime_relation_rows": "//section[@id='relations_anime']//tbody//tr",
    "manga_relation_rows": "//section[@id='relations_manga']//tbody//tr",
}

# ---------------------------------------------------------------------------
# lxml extraction helpers
# ---------------------------------------------------------------------------


def _read_synonym_names(element: Any) -> list[str]:
    """Read the names on AniSearch's synonyms line, each name whole.

    AniSearch writes every synonym on one line, separated by commas, and sets
    every second name in italics. A name can hold commas itself ("Age 12: Kiss,
    Hate, Love"), so names are told apart by their elements, never by commas:

    | Part of the line | Read as |
    | -- | -- |
    | A run of plain text | One name, without the commas around it. |
    | An italic element | One name. |
    | The hidden span that "read more" opens | Its own names, read the same way. |
    | The "Synonyms:" label and the "read more" button | Nothing. |

    Args:
        element: The synonyms line, or the hidden span inside it.

    Returns:
        The names in page order.
    """
    names: list[str] = []
    for node in element.xpath("node()"):
        if isinstance(node, str):
            name = node.strip().strip(",").strip()
        elif node.tag == "i":
            name = "".join(node.itertext()).strip()
        elif node.tag == "span" and "header" not in (node.get("class") or ""):
            names.extend(_read_synonym_names(node))
            continue
        else:
            continue
        if name:
            names.append(name)
    return names


def _extract_anime_from_html(html_text: str) -> dict[str, Any] | None:
    """Parse an AniSearch anime main page into the raw field dict.

    Values are the page text as found, labels included ("Type: TV-Series");
    ``AniSearchAnime`` turns them into field values. Two values are read from
    the page's elements instead, because the text alone loses them:

    - ``synonyms``: one name per plain run or italic element, since a name can
      hold commas (``_read_synonym_names``).
    - ``is_episode_count_estimated``: whether the type line's tooltip says the
      episode count is only AniSearch's estimate.

    Returns None if the HTML cannot be parsed.
    """
    from lxml import etree

    try:
        parser = etree.HTMLParser()
        tree = etree.fromstring(html_text.encode(), parser)
        if tree is None:  # pragma: no cover
            return None  # pragma: no cover
    except Exception:  # pragma: no cover
        return None  # pragma: no cover

    def _element_text(element: Any) -> str:
        return re.sub(r" {2,}", " ", "".join(element.itertext())).strip()

    def _text(key: str) -> str | None:
        els = cast(list[Any], tree.xpath(_XPATHS[key]))
        return (_element_text(els[0]) or None) if els else None

    def _title_blocks() -> list[dict[str, Any]]:
        """Read each language's title block as it stands on the page."""
        return [
            {
                "language": block.get("lang"),
                "title": next(
                    map(_element_text, block.xpath(_XPATHS["title_block_main"])),
                    None,
                ),
                "native_titles": [
                    _element_text(line)
                    for line in block.xpath(_XPATHS["title_block_native"])
                ],
            }
            for block in cast(list[Any], tree.xpath(_XPATHS["title_blocks"]))
        ]

    def _attr(key: str) -> str | None:
        vals = cast(list[str], tree.xpath(_XPATHS[key]))
        return vals[0].strip() if vals else None

    def _vote_total() -> int | None:
        """Sum the per-star vote counts into one total."""
        labels = cast(list[str], tree.xpath(_XPATHS["rating_votes"]))
        counts = [
            int(m.group(1).replace(",", ""))
            for label in labels
            if (m := _VOTES_RE.match(label.strip()))
        ]
        return sum(counts) if counts else None

    def _names(key: str) -> list[str]:
        elements = cast(list[Any], tree.xpath(_XPATHS[key]))
        names = ("".join(el.itertext()).strip() for el in elements)
        return [name for name in names if name]

    website_els = cast(list[Any], tree.xpath(_XPATHS["websites"]))
    websites = [
        {
            "name": "".join(el.itertext()).strip(),
            "url": el.get("href") or "",
        }
        for el in website_els
        if el.get("href")
    ]

    return {
        "cover_image": _attr("cover_image"),
        "title_blocks": _title_blocks(),
        "type": _text("type"),
        "is_episode_count_estimated": any(
            tooltip.startswith(_ESTIMATED_EPISODE_COUNT_TOOLTIP)
            for tooltip in cast(list[str], tree.xpath(_XPATHS["type_tooltips"]))
        ),
        "status": _text("status"),
        "published": _text("published"),
        "studio": _text("studio"),
        "studio_url": _attr("studio_url"),
        "broadcast_raw": _text("broadcast_raw"),
        "source_material": _text("source_material"),
        "synonyms": next(
            map(_read_synonym_names, cast(list[Any], tree.xpath(_XPATHS["synonyms"]))),
            [],
        ),
        "description": _text("description"),
        "genres": _names("genres"),
        "tags": _names("tags"),
        "rating_score": _text("rating_score"),
        "rank_toplist": _text("rank_toplist"),
        "rank_trending": _text("rank_trending"),
        "rating_votes": _vote_total(),
        "websites": websites,
    }


def _extract_relations_from_html(html_text: str) -> dict[str, Any] | None:
    """Parse an AniSearch /relations?show=overall page into anime/manga relation lists.

    Returns a dict with keys 'anime_relations' and 'manga_relations', each a list
    of {relation_type, title, url, details, rating, image} dicts.
    Returns None if the HTML cannot be parsed.
    """
    from lxml import etree

    try:
        parser = etree.HTMLParser()
        tree = etree.fromstring(html_text.encode(), parser)
        if tree is None:  # pragma: no cover
            return None  # pragma: no cover
    except Exception:  # pragma: no cover
        return None  # pragma: no cover

    def _parse_rows(xpath_key: str) -> list[dict[str, Any]]:
        rows = cast(list[Any], tree.xpath(_XPATHS[xpath_key]))
        result = []
        for row in rows:
            span = row.xpath(".//th//span")
            relation_type = "".join(span[0].itertext()).strip() if span else None

            a = row.xpath(".//th//a")
            title = "".join(a[0].itertext()).strip() if a else None
            url = a[0].get("href") or None if a else None

            details_els = row.xpath(".//td[@data-title='Type / Episodes / Year']")
            details = (
                "".join(details_els[0].itertext()).strip() if details_els else None
            )

            rating_els = row.xpath(
                ".//td[contains(@class,'rating')]//div[contains(@class,'star0')]"
            )
            rating = rating_els[0].get("title") if rating_els else None

            # Plain strings: lxml's default ones keep the whole parsed page alive.
            image_attr = row.xpath(
                ".//th[@scope='row']/@data-tooltip", smart_strings=False
            )
            image = image_attr[0] if image_attr else None

            result.append(
                {
                    "relation_type": relation_type,
                    "title": title,
                    "url": url,
                    "details": details,
                    "rating": rating,
                    "image": image,
                }
            )
        return result

    return {
        "anime_relations": _parse_rows("anime_relation_rows"),
        "manga_relations": _parse_rows("manga_relation_rows"),
    }


# ---------------------------------------------------------------------------
# URL and tooltip helpers
# ---------------------------------------------------------------------------


def _extract_path_from_url(url: str) -> str:
    """Extract the anime path from a canonical AniSearch anime URL.

    Raises:
        ValueError: If the URL doesn't start with BASE_ANIME_URL or has no path.
    """
    if not url.startswith(BASE_ANIME_URL):
        raise ValueError(f"URL must start with {BASE_ANIME_URL!r}: {url!r}")
    path = url[len(BASE_ANIME_URL) :].strip("/")
    if not path:
        raise ValueError(f"URL does not contain anime path: {url!r}")
    return path


def _process_relation_tooltips(relations: list[dict[str, Any]]) -> None:
    """Extract image URL from HTML-escaped data-tooltip attribute (mutates in-place)."""
    for rel in relations:
        image = rel.get("image")
        if image:
            m = _IMG_SRC_RE.search(html.unescape(image))
            if m:
                rel["image"] = m.group(1)


# ---------------------------------------------------------------------------
# Relations
# ---------------------------------------------------------------------------


def _parse_relations(
    raw: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not raw:
        return [], []
    anime = list(raw.get("anime_relations") or [])
    manga = list(raw.get("manga_relations") or [])
    _process_relation_tooltips(anime)
    _process_relation_tooltips(manga)
    return anime, manga


# ---------------------------------------------------------------------------
# Cached fetch
# ---------------------------------------------------------------------------


@cached_result(
    service="anisearch",
    key_prefix="anisearch_anime",
)
async def _fetch_anisearch_anime_data(canonical_path: str) -> dict[str, Any] | None:
    """Fetch and extract raw anime data for a given AniSearch anime path.

    Two page fetches (main, relations) over plain HTTP, spaced out by
    ``anisearch_http``. Cached by canonical path;
    cache is automatically invalidated when any extraction function changes.

    Returns the page text as a JSON-serializable dict for _build_anime_from_raw,
    or None when AniSearch answers 404: the anime does not exist. A missing
    relations page leaves the relations empty.

    Raises:
        ServiceNetworkError: If either page fails to load.
        ServiceParseError: If the main page loads but its data cannot be read.
    """
    base_url = f"{BASE_ANIME_URL}{canonical_path}"
    main_page = await fetch_anisearch_page(base_url)
    if main_page is None:
        logger.warning(f"AniSearch anime page not found: {base_url}")
        return None

    main_raw = _extract_anime_from_html(main_page.html)
    if main_raw is None:
        raise ServiceParseError(
            service="anisearch", cause=f"no anime data in {base_url}"
        )

    final_url = main_page.url
    rels_page = await fetch_anisearch_page(
        f"{final_url.rstrip('/')}/relations?show=overall"
    )
    rels_raw = _extract_relations_from_html(rels_page.html) if rels_page else None

    data = dict(main_raw)
    data["anime_relations"], data["manga_relations"] = _parse_relations(rels_raw)
    if final_url != base_url:
        data["_canonical_url"] = final_url
    return data


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------


def _build_anime_from_raw(raw: dict[str, Any], url: str) -> AniSearchAnime:
    """Construct the AniSearchAnime source model from the cached page text."""
    return AniSearchAnime.model_validate(
        {
            "title_blocks": raw.get("title_blocks", []),
            "synonyms": raw.get("synonyms"),
            "type": raw.get("type"),
            "is_episode_count_estimated": raw.get("is_episode_count_estimated", False),
            "source_material": raw.get("source_material"),
            "published": raw.get("published"),
            "status": raw.get("status"),
            "synopsis": raw.get("description"),
            "genres": raw.get("genres", []),
            "tags": raw.get("tags", []),
            "broadcast": raw.get("broadcast_raw"),
            "studio": raw.get("studio"),
            "studio_url": raw.get("studio_url"),
            "websites": raw.get("websites", []),
            "statistics": {
                "score": raw.get("rating_score"),
                "scored_by": raw.get("rating_votes"),
                "rank": raw.get("rank_toplist"),
                "trending": raw.get("rank_trending"),
            },
            "cover_image": raw.get("cover_image"),
            "anime_relations": raw.get("anime_relations", []),
            "manga_relations": raw.get("manga_relations", []),
            "url": url,
        }
    )


# ---------------------------------------------------------------------------
# Crawler class
# ---------------------------------------------------------------------------


class AniSearchAnimeCrawler(BaseCrawler[AniSearchAnime, dict[str, Any]]):
    """Crawler for AniSearch anime detail pages over plain HTTP + lxml XPath."""

    def get_extraction_schema(self) -> dict[str, Any]:
        return {"xpaths": _XPATHS}

    def normalize_identifier(self, identifier: str) -> str:
        normalized = identifier.replace(
            "https://anisearch.com/", "https://www.anisearch.com/", 1
        )
        if not normalized.startswith(BASE_ANIME_URL):
            raise ValueError(f"Not an AniSearch anime URL: {identifier!r}")
        return normalized

    async def fetch_raw_data(self, url: str) -> dict[str, Any] | None:
        return await _fetch_anisearch_anime_data(_extract_path_from_url(url))

    def build_source_model(
        self, processed_raw: dict[str, Any], url: str
    ) -> AniSearchAnime:
        return _build_anime_from_raw(
            processed_raw, processed_raw.get("_canonical_url", url)
        )

    def map_to_canonical(self, source_model: AniSearchAnime) -> dict[str, Any]:
        return anime_from_anisearch(source_model)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def fetch_anisearch_anime(
    url: str, output_path: str | None = None
) -> dict[str, Any] | None:
    """Fetch canonical anime dict for an AniSearch anime URL.

    Args:
        url: Full AniSearch anime URL (e.g. "https://www.anisearch.com/anime/18878,dan-da-dan").
        output_path: Optional path to write JSON result.

    Returns:
        Canonical anime dict, or None if fetch or mapping fails.
    """
    repo = FileRepository(output_path) if output_path else NullRepository()
    return await AniSearchAnimeCrawler(repo).crawl(url)
