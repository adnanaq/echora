"""Source-faithful Pydantic models for anime-planet.com scraped data.

Field names mirror what anime-planet.com actually sends — no renaming to
canonical names here.  The mapper (animeplanet_mapper.py) handles all
translation to the canonical Anime model.
"""

from pydantic import BaseModel


class AnimePlanetRelatedEntry(BaseModel):
    """Related anime entry from the relations tab."""

    title: str
    url: str  # relative href, e.g. "/anime/one-piece-film-red"
    slug: str  # extracted from url
    relation_subtype: str | None = None  # "Same Franchise", "Sequel", "Spin Off", etc.
    type: str | None = None  # "TV", "Movie", "OVA" — parsed from fa-tv span
    episode_count: int | None = None  # parsed from "OVA: 1 ep", None when not shown
    image: str | None = None


class AnimePlanetMangaEntry(BaseModel):
    """Related manga entry from the relations tab."""

    title: str
    url: str  # relative href, e.g. "/manga/one-piece"
    slug: str  # extracted from url
    relation_subtype: str | None = None
    type: str | None = None  # "One Shot" when AP shows it; None otherwise
    volumes: int | None = None  # parsed from "Vol: X - Ch: Y" or "Vol: X"
    chapters: int | None = (
        None  # parsed from "Vol: X - Ch: Y" or "Ch: Y"; 1 for One Shot
    )
    image: str | None = None


class AnimePlanetAggregateRating(BaseModel):
    """Aggregate user rating from JSON-LD aggregateRating block."""

    rating_value: float | None = None  # 4.315 (anime-planet uses 0–5 scale)
    rating_count: int | None = None  # 64676


class AnimePlanetAnime(BaseModel):
    """Scraped anime data from an anime-planet.com anime page.

    Populated from two sources:
    - JSON-LD <script type="application/ld+json"> block
    - XPath extraction via JsonXPathExtractionStrategy (entryBar, tags, relations)
    """

    # ── From JSON-LD ──────────────────────────────────────────────────────
    name: str
    schema_type: str | None = None  # "@type" value: "TVSeries", "Movie", etc.
    description: str | None = None  # synopsis
    url: str | None = None  # canonical AP URL
    start_date: str | None = None  # "1999-10-20"
    end_date: str | None = None
    number_of_episodes: int | None = None
    genres: list[str] = []
    aggregate_rating: AnimePlanetAggregateRating | None = None

    # ── From XPath ────────────────────────────────────────────────────────
    type_raw: str | None = None  # "TV\n  (1156+ eps)" raw entryBar span text
    season_url: str | None = None  # full href e.g. "/anime/seasons/fall-1999"
    rank_text: str | None = None  # "Rank #157"
    studios: list[str] = []
    aka: str | None = None  # raw h2.aka text: "Alt title: ワンピース"
    tags: list[str] = []  # ["Action", "Adventure", ...]
    cover: str | None = None  # from img[itemprop="image"] — actual poster
    slug: str  # canonical slug, set by caller

    # ── Relations ─────────────────────────────────────────────────────────
    related_anime: list[AnimePlanetRelatedEntry] = []
    related_anime_other: list[AnimePlanetRelatedEntry] = []
    related_manga: list[AnimePlanetMangaEntry] = []
