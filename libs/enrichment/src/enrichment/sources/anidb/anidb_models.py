"""Pydantic source models for AniDB HTTP XML API responses.

Field names mirror canonical model fields wherever possible so the mapper
(anidb_mapper.py) only performs value normalization, not field renaming.
XML tag names are used where no canonical equivalent exists.
"""

from pydantic import BaseModel, ConfigDict


# =============================================================================
# SUPPORTING MODELS
# =============================================================================


class AniDBSeiyuu(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int | None = None
    name: str | None = None
    picture: str | None = None  # raw filename (e.g. "299023.jpg") — mapper adds CDN prefix


class AniDBCharacter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int | None = None
    type: str | None = None            # raw attr: "main character in", "secondary cast in", "appears in"
    name: str | None = None            # from <name> element
    gender: str | None = None
    character_type: str | None = None  # from <charactertype> text
    character_type_id: int | None = None
    description: str | None = None
    picture: str | None = None         # raw filename — mapper adds CDN prefix
    rating: float | None = None
    rating_votes: int = 0
    seiyuu: list[AniDBSeiyuu] = []     # findall("seiyuu") — XML allows multiple per character


class AniDBEpisode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int | None = None
    episode_number: int | str | None = None  # int for type=1 regular, str for specials (S1, C1, etc.)
    episode_type: int | None = None          # 1=Regular 2=Special 3=Credit 4=Trailer 5=Parody
    length: int | None = None                # minutes — mapper converts to seconds
    airdate: str | None = None               # YYYY-MM-DD (matches XML tag name)
    rating: float | None = None
    rating_votes: int = 0
    summary: str | None = None               # matches XML tag name
    titles: dict[str, str] = {}              # {lang_code: title} — "x-jat" normalized to "romaji"
    streaming: dict[str, str] = {}           # {platform: url}


class AniDBCreator(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int | None = None
    name: str | None = None
    role: str | None = None  # from type attr: "Direction", "Original Work", "Music", etc.


class AniDBRelatedAnime(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    relation_type: str         # raw attr value: "Sequel", "Prequel", "Same Setting", etc.
    title: str | None = None


class AniDBCategory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str | None = None
    name: str
    weight: int = 0
    hentai: bool = False


class AniDBRatings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    permanent: float | None = None
    permanent_count: int = 0
    temporary: float | None = None
    temporary_count: int = 0
    review: float | None = None
    review_count: int = 0


class AniDBExternalResource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str            # numeric string: "1", "2", "4", "6", etc.
    urls: list[str] = []
    identifiers: list[str] = []


# =============================================================================
# TOP-LEVEL ANIME MODEL
# =============================================================================


class AniDBAnime(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Root element attributes
    id: int
    restricted: bool = False

    # Scalar elements (names match XML tags or canonical fields)
    type: str | None = None          # <type> text — mapper normalizes to AnimeType
    episode_count: int = 0           # <episodecount> — avoids collision with episodes list
    start_date: str | None = None    # <startdate> YYYY-MM-DD
    end_date: str | None = None      # <enddate> YYYY-MM-DD
    description: str | None = None   # <description> — mapper renames to synopsis
    picture: str | None = None       # <picture> filename only — mapper adds CDN prefix
    url: str | None = None           # <url> official website

    # Titles (parsed from <titles>)
    title: str | None = None                # type="main"
    title_english: str | None = None        # type="official" lang="en"
    title_japanese: str | None = None       # type="official" lang="ja"
    synonyms: list[str] = []                # type="synonym" + type="short" (informal only)
    title_others: dict[str, str] = {}       # type="official" other langs → canonical Anime.titles

    # Array elements
    categories: list[AniDBCategory] = []
    characters: list[AniDBCharacter] = []
    creators: list[AniDBCreator] = []
    episodes: list[AniDBEpisode] = []
    related_anime: list[AniDBRelatedAnime] = []
    resources: list[AniDBExternalResource] = []
    tags: list[str] = []

    # Object fields
    ratings: AniDBRatings | None = None


# =============================================================================
# CHARACTER PAGE MODEL (from anidb_character_crawler)
# =============================================================================


class AniDBCharacterPage(BaseModel):
    """Enrichment data from AniDB character web page — supplements XML character data."""

    model_config = ConfigDict(extra="forbid")

    name_main: str | None = None
    name_kanji: str | None = None
    description: str | None = None
    gender: str | None = None
    nicknames: list[str] = []
    official_names: list[str] = []
    abilities: list[str] = []
    looks: list[str] = []
    personality: list[str] = []
    role: list[str] = []
    supernatural_abilities: list[str] = []
    animeography: list[dict[str, str]] = []  # [{"title": ..., "role": ..., "url": ...}]
