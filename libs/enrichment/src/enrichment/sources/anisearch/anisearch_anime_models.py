"""Source-faithful Pydantic models for anisearch.com scraped data.

Field names mirror what AniSearch actually provides — no renaming to canonical
names here. The models turn the page text into field values (labels removed,
dates, ranks and scores read); the mapper (anisearch_mapper.py) translates them
to the canonical model.
"""

import re
from typing import Annotated, Any, Self

from enrichment.sources.base.field_types import (
    NumberFromText,
    StrippedText,
    number_from_display_text,
    strip_text_or_none,
)
from enrichment.sources.base.utils import (
    month_name,
    parse_broadcast_string,
    parse_iso_date,
    parse_partial_date,
    split_date_range,
)
from enrichment.utils.text_utils import fold_for_comparison
from pydantic import BaseModel, BeforeValidator, ConfigDict, model_validator

_ANISEARCH_BASE_URL = "https://www.anisearch.com"
# The label AniSearch prints before a value, e.g. "Type: " or "Published: ".
_LABEL = re.compile(r"^\s*[^:]+:\s*")
_DECIMAL = re.compile(r"(\d+\.\d+)")
# The type at the start of a relation's details, e.g. "TV-Series" in "TV-Series, 12 (2025)".
_DETAILS_TYPE = re.compile(r"^([^,]+)")
# AniSearch's language codes that differ from the BCP 47 codes AniDB uses.
_LANGUAGE_CODES = {"kr": "ko", "hk": "zh-HK"}
# AniDB's code for Japanese written in Latin letters.
_JAPANESE_ROMAJI = "x-jat"


def _without_label(value: object) -> object:
    """Remove the label AniSearch prints before a value ("Status: Ongoing").

    >>> _without_label("Status: Ongoing")
    'Ongoing'
    """
    value = strip_text_or_none(value)
    if isinstance(value, str):
        return _LABEL.sub("", value, count=1).strip() or None
    return value


def _type_name(value: object) -> object:
    """Read the type from "Type: TV-Series, 25 (~24 min)".

    >>> _type_name("Type: TV-Series, 25 (~24 min)")
    'TV-Series'
    """
    value = _without_label(value)
    if isinstance(value, str):
        return value.split(",")[0].strip() or None
    return value


def _episode_count_from_type_line(type_line: object) -> int | None:
    """Read the episode count from "Type: TV-Series, 25 (~24 min)".

    - AniSearch prints the count after the type and before the episode length
      in brackets.
    - AniSearch prints "?" while the count is not known, which reads as None.
    - A count AniSearch marks as its estimate is not read here: the mark is a
      tooltip the line's text does not hold, so ``AniSearchAnime`` checks it.

    Args:
        type_line: The type line as the crawler extracted it, label included.

    Returns:
        The episode count, or None when the line gives none.

    >>> _episode_count_from_type_line("Type: TV-Series, 1200 (~24 min, Total: ~20 d)")
    1200
    >>> _episode_count_from_type_line("Type: TV-Series, ? (~24 min)") is None
    True
    >>> _episode_count_from_type_line("Type: Movie") is None
    True
    """
    type_text = _without_label(type_line)
    if not isinstance(type_text, str) or "," not in type_text:
        return None
    count_text = type_text.split(",", 1)[1].split("(", 1)[0]
    episode_count = number_from_display_text(count_text)
    return episode_count if isinstance(episode_count, int) else None


def _strip_synonym_names(value: object) -> object:
    """Strip the synonym names the crawler read, dropping blank ones.

    - No synonyms gives an empty list.
    - Names are never split: the crawler tells them apart by the page's
      elements, because a name can hold commas.
    - Text joined into one string fails validation instead of being split.

    >>> _strip_synonym_names([" Age 12: Kiss, Hate, Love ", "", "OP"])
    ['Age 12: Kiss, Hate, Love', 'OP']
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [
            synonym_name.strip()
            for synonym_name in value
            if isinstance(synonym_name, str) and synonym_name.strip()
        ]
    return value


def _full_studio_url(value: object) -> object:
    """Turn the studio path AniSearch links to into a full address.

    >>> _full_studio_url("company/412,toei")
    'https://www.anisearch.com/company/412,toei'
    """
    value = strip_text_or_none(value)
    if isinstance(value, str) and not value.startswith("http"):
        return f"{_ANISEARCH_BASE_URL}/{value.lstrip('/')}"
    return value


def _rated_score(value: object) -> object:
    """Read the score from "Calculated Value4.18 = 84%"; 0 means not rated.

    >>> _rated_score("Calculated Value4.18 = 84%")
    4.18
    >>> _rated_score("Calculated Value0.00 = 0%") is None
    True
    """
    if isinstance(value, str):
        match = _DECIMAL.search(value)
        value = float(match.group(1)) if match else None
    if isinstance(value, int | float) and value <= 0:
        return None
    return value


def _vote_count(value: object) -> object:
    """Keep a vote count above 0; 0 votes means not rated."""
    if isinstance(value, int) and value <= 0:
        return None
    return value


def _published_dates(published: object) -> dict[str, Any]:
    """Read AniSearch's published value at whatever precision it states.

    AniSearch writes a full date ("20.10.1999"), a month and year ("11.2008"),
    a year ("2027") or "?", optionally as a range ("2027 ‑ ?"). Only a full
    date becomes ``start_date``/``end_date``; a partial start keeps its year
    and month so nothing is made up.

    Examples:
        A full date, with an open or a closed end:

        >>> _published_dates("Published: 20.10.1999 \u2011 ?")
        {'start_date': '1999-10-20', 'end_date': None, 'start_year': 1999, 'start_month': None}
        >>> _published_dates("Published: 20.10.1999 \u2011 05.11.2000")
        {'start_date': '1999-10-20', 'end_date': '2000-11-05', 'start_year': 1999, 'start_month': None}

        A month and year, or a year alone, fills only what it states:

        >>> _published_dates("Published: 11.2008")
        {'start_date': None, 'end_date': None, 'start_year': 2008, 'start_month': 'November'}
        >>> _published_dates("Published: 2027 \u2011 ?")
        {'start_date': None, 'end_date': None, 'start_year': 2027, 'start_month': None}
    """
    text = _without_label(published)
    start_raw, end_raw = split_date_range(text if isinstance(text, str) else "")
    start_date = parse_iso_date(start_raw)
    start_year, start_month = parse_partial_date(start_raw)
    return {
        "start_date": start_date,
        "end_date": parse_iso_date(end_raw),
        "start_year": start_year,
        "start_month": None if start_date else month_name(start_month),
    }


def _broadcast_parts(broadcast: object) -> dict[str, str | None]:
    """Read day, time and timezone from AniSearch's broadcast value.

    >>> _broadcast_parts("Broadcast: Sunday 23:15 (JST)")
    {'broadcast_day': 'Sunday', 'broadcast_time': '23:15', 'broadcast_timezone': 'JST'}
    >>> _broadcast_parts("Broadcast: ?")
    {'broadcast_day': None, 'broadcast_time': None, 'broadcast_timezone': None}
    """
    text = _without_label(broadcast)
    day, time, timezone = parse_broadcast_string(text if isinstance(text, str) else "")
    return {
        "broadcast_day": day,
        "broadcast_time": time,
        "broadcast_timezone": timezone,
    }


LabelledText = Annotated[str | None, BeforeValidator(_without_label)]


def _standard_language_code(code: object) -> object:
    """Replace AniSearch's own language codes with the standard ones.

    >>> _standard_language_code("kr")
    'ko'
    >>> _standard_language_code("de")
    'de'
    """
    return _LANGUAGE_CODES.get(code, code) if isinstance(code, str) else code


class AniSearchTitleBlock(BaseModel):
    """One language's titles from the page's title section."""

    language: Annotated[str, BeforeValidator(_standard_language_code)]
    title: StrippedText = None
    native_titles: list[str] = []

    @property
    def native_title(self) -> str | None:
        """The last native line: a Japanese block gives the romaji before the kanji."""
        return self.native_titles[-1] if self.native_titles else None


def _title_fields(raw_blocks: list[Any]) -> dict[str, Any]:
    """Turn the page's title blocks into the title fields.

    The first block is the work's own title, in its original language; its
    native script goes into ``titles`` unless the language has a field of its
    own. English and Japanese have their own fields, and a Japanese block that
    is not the first is a Japanese release, whose romaji goes under ``x-jat``.
    A title repeating one already kept is left out.
    """
    blocks = [AniSearchTitleBlock.model_validate(block) for block in raw_blocks]
    if not blocks:
        return {}
    fields: dict[str, Any] = {"title": blocks[0].title}
    candidates: list[tuple[str, str | None]] = []
    for position, block in enumerate(blocks):
        if block.language == "ja":
            fields["title_japanese"] = block.native_title
            if position:
                candidates.append((_JAPANESE_ROMAJI, block.title))
        elif block.language == "en":
            fields["title_english"] = block.title
        else:
            title = block.title if position else block.native_title
            candidates.append((block.language, title))
    kept = {
        fold_for_comparison(title)
        for title in (fields["title"], fields.get("title_english"))
        if title
    }
    titles: dict[str, str] = {}
    for language, title in candidates:
        key = fold_for_comparison(title) if title else ""
        if title and key and key not in kept and language not in titles:
            titles[language] = title
            kept.add(key)
    fields["titles"] = titles
    return fields


class AniSearchStatistics(BaseModel):
    score: Annotated[float | None, BeforeValidator(_rated_score)] = None
    scored_by: Annotated[int | None, BeforeValidator(_vote_count)] = None
    rank: NumberFromText = None
    trending: NumberFromText = None

    @property
    def is_empty(self) -> bool:
        """Tell whether AniSearch reported none of the statistics."""
        return all(value is None for value in self.model_dump().values())


class AniSearchRelatedEntry(BaseModel):
    relation_type: str | None = None  # "Sequel", "Prequel", "Side Story", etc.
    title: str | None = None
    url: str | None = None
    details: str | None = None  # "TV-Series, 12 (2025)"
    rating: str | None = None
    image: str | None = None

    @property
    def media_type(self) -> str | None:
        """The type at the start of ``details``, e.g. "TV-Series" or "Manga".

        >>> AniSearchRelatedEntry(details="TV-Series, 12 (2025)").media_type
        'TV-Series'
        """
        match = _DETAILS_TYPE.match((self.details or "").strip())
        return (match.group(1).strip() or None) if match else None


class AniSearchAnime(BaseModel):
    """Scraped anime data from anisearch.com."""

    # ── Titles ────────────────────────────────────────────────────────────
    title: StrippedText = None  # the original-language title block's main line
    title_english: StrippedText = None
    title_japanese: StrippedText = None  # the Japanese block's native script
    titles: dict[str, str] = {}  # other languages, keyed by BCP 47 code
    synonyms: Annotated[list[str], BeforeValidator(_strip_synonym_names)] = []

    # ── Classification ────────────────────────────────────────────────────
    type: Annotated[str | None, BeforeValidator(_type_name)] = (
        None  # "TV-Series", "Movie", "OVA", etc.
    )
    # Read from the same "Type:" line as the type; None while AniSearch gives "?"
    # or marks the count as its estimate.
    episode_count: int | None = None
    source_material: LabelledText = None  # "Manga", "Light Novel", etc.

    # ── Dates (ISO YYYY-MM-DD, only when AniSearch states the day) ─────────
    start_date: str | None = None  # only a full date
    end_date: str | None = None  # only a full date
    start_year: int | None = None  # stated even when the start has no day
    start_month: str | None = None  # month name, only when the start has no day
    status: LabelledText = None  # "Upcoming", "On Hold", "Completed", etc.

    # ── Content ───────────────────────────────────────────────────────────
    synopsis: StrippedText = None
    genres: list[str] = []
    tags: list[str] = []

    # ── Broadcast schedule ────────────────────────────────────────────────
    broadcast_day: str | None = None
    broadcast_time: str | None = None
    broadcast_timezone: str | None = None

    # ── Studio (primary) ──────────────────────────────────────────────────
    studio: StrippedText = None
    studio_url: Annotated[str | None, BeforeValidator(_full_studio_url)] = None

    # ── External links ────────────────────────────────────────────────────
    websites: list[dict[str, str]] = []

    # ── Statistics ────────────────────────────────────────────────────────
    statistics: AniSearchStatistics | None = None

    # ── Media ─────────────────────────────────────────────────────────────
    cover_image: StrippedText = None

    # ── Relations ─────────────────────────────────────────────────────────
    anime_relations: list[AniSearchRelatedEntry] = []
    manga_relations: list[AniSearchRelatedEntry] = []

    # ── Source URL (injected by build_source_model) ───────────────────────
    url: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _read_page_sections(cls, data: Any) -> Any:
        """Split the page's type, published, broadcast and title sections into their fields.

        The "Type:" line holds the episode count after the type. The count is
        read before the type field keeps only the type:

        - A count AniSearch marks as its estimate (``is_episode_count_estimated``)
          reads as unknown, like AniSearch's "?": it is a guess, such as the
          whole run of an anime still airing, and work matching compares counts.
        - Any other count is kept.

        Args:
            data: The raw values the crawler extracted, by field name.

        Returns:
            The values with each section split into its fields; other input
            unchanged.
        """
        if not isinstance(data, dict):
            return data
        data = dict(data)
        is_episode_count_estimated = data.pop("is_episode_count_estimated", False)
        if "type" in data and "episode_count" not in data:
            data["episode_count"] = (
                None
                if is_episode_count_estimated
                else _episode_count_from_type_line(data["type"])
            )
        if "published" in data:
            data.update(_published_dates(data.pop("published")))
        if "broadcast" in data:
            data.update(_broadcast_parts(data.pop("broadcast")))
        if "title_blocks" in data:
            data.update(_title_fields(data.pop("title_blocks")))
        return data

    @model_validator(mode="after")
    def _drop_empty_statistics(self) -> Self:
        """Store no statistics when AniSearch reported none."""
        if self.statistics is not None and self.statistics.is_empty:
            self.statistics = None
        return self


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
    # AniSearch does not publish a role on a character's own filmography page,
    # so this stays None there; it is set only for the anime being enriched.
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
    # Anime whose characters page supplied `role`, so the mapper can attach it
    # to the matching animeography entry.
    anime_url: str | None = None
    tags: list[str] = []
    screenshot_images: list[str] = []
    picture_images: list[str] = []
    voice_actors: list[AniSearchVoiceActorRef] = []
    anime_roles: list[AniSearchCharacterAnimeRole] = []
    anime_ography: list[AniSearchCharacterAnimeRole] = []
    manga_ography: list[AniSearchCharacterAnimeRole] = []
    attributes: dict[str, str] = {}
