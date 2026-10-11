"""Pydantic source models for AniDB HTTP XML API responses.

Field names mirror canonical model fields wherever possible so the mapper
(anidb_mapper.py) only performs value normalization, not field renaming.
XML tag names are used where no canonical equivalent exists.
"""

from typing import Annotated, Self

from enrichment.sources.base.field_types import strip_text_or_none
from pydantic import BaseModel, BeforeValidator, ConfigDict, model_validator

# AniDB's placeholder for a date it does not know.
_UNKNOWN_DATE = "1970-01-01"


def _whole_number_or_none(value: object) -> object:
    """Read AniDB text as a whole number; text that is not one becomes None.

    AniDB documents are thousands of lines; one malformed value must not
    abort the parse and discard the anime, its episodes and its characters.

    >>> _whole_number_or_none(" 12 ")
    12
    >>> _whole_number_or_none("12a") is None
    True
    """
    value = strip_text_or_none(value)
    if isinstance(value, str):
        return int(value) if value.isdigit() else None
    return value


def _count_or_zero(value: object) -> object:
    """Read an AniDB count; a missing or malformed count is 0.

    >>> _count_or_zero("2500000")
    2500000
    >>> _count_or_zero(None)
    0
    """
    number = _whole_number_or_none(value)
    return 0 if number is None else number


def _decimal_or_none(value: object) -> object:
    """Read AniDB text as a decimal number; text that is not one becomes None.

    >>> _decimal_or_none("8.43")
    8.43
    >>> _decimal_or_none("n/a") is None
    True
    """
    value = strip_text_or_none(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return value


def _known_date(value: object) -> object:
    """Turn AniDB's unknown-date placeholder ("1970-01-01") into None.

    >>> _known_date("1970-01-01") is None
    True
    >>> _known_date("1999-10-20")
    '1999-10-20'
    """
    value = strip_text_or_none(value)
    return None if value == _UNKNOWN_DATE else value


WholeNumber = Annotated[int | None, BeforeValidator(_whole_number_or_none)]
Count = Annotated[int, BeforeValidator(_count_or_zero)]
Decimal = Annotated[float | None, BeforeValidator(_decimal_or_none)]
KnownDate = Annotated[str | None, BeforeValidator(_known_date)]

# =============================================================================
# SUPPORTING MODELS
# =============================================================================


class AniDBSeiyuu(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: WholeNumber = None
    name: str | None = None
    picture: str | None = (
        None  # raw filename (e.g. "299023.jpg") — mapper adds CDN prefix
    )


class AniDBCharacter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: WholeNumber = None
    type: str | None = (
        None  # raw attr: "main character in", "secondary cast in", "appears in"
    )
    name: str | None = None  # from <name> element
    gender: str | None = None
    character_type: str | None = None  # from <charactertype> text
    character_type_id: WholeNumber = None
    description: str | None = None
    picture: str | None = None  # raw filename — mapper adds CDN prefix
    rating: Decimal = None
    rating_votes: Count = 0
    seiyuu: list[
        AniDBSeiyuu
    ] = []  # findall("seiyuu") — XML allows multiple per character


class AniDBEpisode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: WholeNumber = None
    episode_number: int | str | None = (
        None  # int for type=1 regular, str for specials (S1, C1, etc.)
    )
    episode_type: WholeNumber = None  # 1=Regular 2=Special 3=Credit 4=Trailer 5=Parody
    length: WholeNumber = None  # minutes — mapper converts to seconds
    airdate: str | None = None  # YYYY-MM-DD (matches XML tag name)
    rating: Decimal = None
    rating_votes: Count = 0
    summary: str | None = None  # matches XML tag name
    titles: dict[str, str] = {}  # {lang_code: title} — "x-jat" normalized to "romaji"
    streaming: dict[str, str] = {}  # {platform: url}

    @model_validator(mode="after")
    def _regular_episode_number_as_int(self) -> Self:
        """Store a regular episode's number (type 1) as an int.

        Other episode types keep their text (``S1``, ``C3``), as does a regular
        episode whose number is not a whole number.
        """
        if (
            self.episode_type == 1
            and isinstance(self.episode_number, str)
            and self.episode_number.isdigit()
        ):
            self.episode_number = int(self.episode_number)
        return self


class AniDBCreator(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: WholeNumber = None
    name: str | None = None
    role: str | None = (
        None  # from type attr: "Direction", "Original Work", "Music", etc.
    )


class AniDBRelatedAnime(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    relation_type: str  # raw attr value: "Sequel", "Prequel", "Same Setting", etc.
    title: str | None = None


class AniDBCategory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str | None = None
    name: str
    weight: Count = 0
    hentai: bool = False


class AniDBRatings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    permanent: Decimal = None
    permanent_count: Count = 0
    temporary: Decimal = None
    temporary_count: Count = 0
    review: Decimal = None
    review_count: Count = 0


class AniDBExternalResource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str  # numeric string: "1", "2", "4", "6", etc.
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
    type: str | None = None  # <type> text — mapper normalizes to AnimeType
    episode_count: Count = 0  # <episodecount> — avoids collision with episodes list
    start_date: KnownDate = None  # <startdate> YYYY-MM-DD
    end_date: KnownDate = None  # <enddate> YYYY-MM-DD
    description: str | None = None  # <description> — mapper renames to synopsis
    picture: str | None = None  # <picture> filename only — mapper adds CDN prefix
    url: str | None = None  # <url> official website

    # Titles (parsed from <titles>)
    title: str | None = None  # type="main"
    title_english: str | None = None  # type="official" lang="en"
    title_japanese: str | None = None  # type="official" lang="ja"
    synonyms: list[str] = []  # type="synonym" + type="short" (informal only)
    title_others: dict[
        str, str
    ] = {}  # type="official" other langs → canonical Anime.titles

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
