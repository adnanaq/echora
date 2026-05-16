"""Source-faithful Pydantic models for anisearch.com scraped data.

Field names mirror what AniSearch actually provides — no renaming to canonical
names here.  The mapper (anisearch_mapper.py) handles all translation.
"""

from pydantic import BaseModel, ConfigDict


class AniSearchStatistics(BaseModel):
    score: float | None = None
    rank: int | None = None
    trending: int | None = None


class AniSearchRelatedEntry(BaseModel):
    relation_type: str | None = None  # "Sequel", "Prequel", "Side Story", etc.
    title: str | None = None
    url: str | None = None
    details: str | None = None  # "TV-Series, 12 (2025)"
    rating: str | None = None
    image: str | None = None


class AniSearchAnime(BaseModel):
    """Scraped anime data from anisearch.com."""

    # ── Titles ────────────────────────────────────────────────────────────
    title: str | None = None  # primary display title (strong.f16)
    title_japanese: str | None = None  # native script (div.grey)
    synonyms: list[str] = []

    # ── Classification ────────────────────────────────────────────────────
    type: str | None = None  # raw: "TV-Series", "Movie", "OVA", etc.
    source_material: str | None = None  # raw: "Manga", "Light Novel", etc.

    # ── Dates (DD.MM.YYYY — datetime_utils handles this format natively) ──
    start_date: str | None = None
    end_date: str | None = None

    # ── Content ───────────────────────────────────────────────────────────
    synopsis: str | None = None
    genres: list[str] = []
    tags: list[str] = []

    # ── Broadcast schedule ────────────────────────────────────────────────
    broadcast_day: str | None = None
    broadcast_time: str | None = None
    broadcast_timezone: str | None = None

    # ── Studio (primary) ──────────────────────────────────────────────────
    studio: str | None = None
    studio_url: str | None = None

    # ── External links ────────────────────────────────────────────────────
    websites: list[dict[str, str]] = []

    # ── Statistics ────────────────────────────────────────────────────────
    statistics: AniSearchStatistics | None = None

    # ── Media ─────────────────────────────────────────────────────────────
    cover_image: str | None = None

    # ── Relations ─────────────────────────────────────────────────────────
    anime_relations: list[AniSearchRelatedEntry] = []
    manga_relations: list[AniSearchRelatedEntry] = []

    # ── Source URL (injected by build_source_model) ───────────────────────
    url: str | None = None


# =============================================================================
# EPISODE MODELS
# =============================================================================


class AniSearchEpisode(BaseModel):
    """Scraped from a single row of an anisearch.com /episodes page.

    All fields are pre-parsed by the crawler before model construction:
    runtime "24 min" → duration seconds, "20. Oct 1999" → aired ISO date,
    "Romaji (Kanji)" → separate title_romaji and title_japanese fields.
    """

    episode_number: int
    is_filler: bool = False
    is_recap: bool = False
    duration: int | None = None  # seconds, e.g. 1440
    aired: str | None = None  # ISO date string, e.g. "1999-10-20"
    title: str | None = None  # English title
    title_romaji: str | None = None  # Romanized Japanese title
    title_japanese: str | None = None  # Native script Japanese title
    titles: dict[str, str] = {}  # Additional titles keyed by BCP 47 code (de, fr, it)
    source: str | None = None  # /episodes page URL


class AniSearchEpisodesPage(BaseModel):
    """All episodes extracted from a single anisearch.com /episodes page."""

    episodes: list[AniSearchEpisode] = []
    source: str | None = None


# =============================================================================
# CHARACTER MODELS
# =============================================================================


class AniSearchVoiceActorRef(BaseModel):
    name: str
    language: str
    url: str | None = None


class AniSearchCharacterAnimeRole(BaseModel):
    title: str
    url: str | None = None
    role: str | None = None


class AniSearchCharacter(BaseModel):
    """Scraped from https://www.anisearch.com/character/{id},{slug}."""

    model_config = ConfigDict(extra="forbid")

    source: str
    name: str | None = None
    name_native: str | None = None
    description: str | None = None
    image: str | None = None
    favorites: int | None = None
    role: str | None = None
    tags: list[str] = []
    screenshot_images: list[str] = []
    picture_images: list[str] = []
    voice_actors: list[AniSearchVoiceActorRef] = []
    anime_roles: list[AniSearchCharacterAnimeRole] = []
    anime_ography: list[AniSearchCharacterAnimeRole] = []
    manga_ography: list[AniSearchCharacterAnimeRole] = []
    attributes: dict[str, str] = {}
