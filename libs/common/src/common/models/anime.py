"""Pydantic models for anime, character, and episode data used across Echora services."""
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

# =============================================================================
# ENUMS
# =============================================================================

class AnimeStatus(str, Enum):
    """Anime airing status classification."""

    CANCELLED = "CANCELLED"
    FINISHED = "FINISHED"
    ONGOING = "ONGOING"
    UNKNOWN = "UNKNOWN"
    UPCOMING = "UPCOMING"

    @classmethod
    def _missing_(cls, value: object) -> "AnimeStatus":
        """Normalize source-specific strings into standard Enum members.

        Handles values from MAL, AniList, Kitsu, AniDB, AniSearch, etc.
        """
        if not isinstance(value, str):
            return cls.UNKNOWN

        v = value.lower()
        _map = {
            # MAL / Jikan
            "currently airing": cls.ONGOING,
            "finished airing": cls.FINISHED,
            "not yet aired": cls.UPCOMING,
            # AniList
            "releasing": cls.ONGOING,
            "not_yet_released": cls.UPCOMING,
            "cancelled": cls.CANCELLED,
            "hiatus": cls.ONGOING,
            # Kitsu
            "current": cls.ONGOING,
            "tba": cls.UPCOMING,
            "unreleased": cls.UPCOMING,
            "upcoming": cls.UPCOMING,
            # AnimSchedule / AniSearch
            "ongoing": cls.ONGOING,
            "completed": cls.FINISHED,
            "delayed": cls.ONGOING,
            "on hold": cls.ONGOING,
            # Catch-all
            "finished": cls.FINISHED,
            "airing": cls.ONGOING,
            "unknown": cls.UNKNOWN,
        }
        return _map.get(v, cls.UNKNOWN)


class EntityType(str, Enum):
    """Primary entity type classification for vector search."""

    ANIME = "anime"
    CHARACTER = "character"
    EPISODE = "episode"


class AnimeType(str, Enum):
    """Anime type/format classification."""

    CM = "CM"
    MOVIE = "MOVIE"
    MUSIC = "MUSIC"
    ONA = "ONA"
    OVA = "OVA"
    PV = "PV"
    SPECIAL = "SPECIAL"
    TV = "TV"
    TV_SHORT = "TV SHORT"
    TV_SPECIAL = "TV SPECIAL"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def _missing_(cls, value: object) -> "AnimeType":
        """Normalize format strings from multiple sources.

        Handles variants from MAL, AniList, AniDB, AnimePlanet, etc.
        """
        if not isinstance(value, str):
            return cls.UNKNOWN

        v = value.lower()
        _map = {
            # Common across MAL, AniList, Kitsu, AnimSchedule, AnimePlanet, AniSearch
            "tv": cls.TV,
            "movie": cls.MOVIE,
            "ova": cls.OVA,
            "special": cls.SPECIAL,
            # Common across MAL, AniList, Kitsu, AnimePlanet
            "ona": cls.ONA,
            "music": cls.MUSIC,
            # MAL
            "tv special": cls.TV_SPECIAL,
            "pv": cls.PV,
            "cm": cls.CM,
            # AniList
            "tv_short": cls.TV_SHORT,
            # AnimSchedule
            "tv short": cls.TV_SHORT,
            "ona (chinese)": cls.ONA,
            "movie (chinese)": cls.MOVIE,
            "tv (chinese)": cls.TV,
            "ova (chinese)": cls.OVA,
            "special (chinese)": cls.SPECIAL,
            "tv short (chinese)": cls.TV_SHORT,
            # AniDB / AnimePlanet / AniSearch
            "tv series": cls.TV,
            "web": cls.ONA,
            "music video": cls.MUSIC,
            "bonus": cls.SPECIAL,
            "dvd special": cls.SPECIAL,
            "oav": cls.OVA,
            "tv-series": cls.TV,
            "tv-special": cls.TV_SPECIAL,
            # Catch-all
            "unknown": cls.UNKNOWN,
        }
        return _map.get(v, cls.UNKNOWN)


class AnimeRating(str, Enum):
    """Anime content rating classification."""

    G = "G - All Ages"
    PG = "PG - Children"
    PG13 = "PG-13 - Teens 13 or older"
    R = "R - 17+ (violence & profanity)"
    RPLUS = "R+ - Mild Nudity"
    RX = "Rx - Hentai"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def _missing_(cls, value: object) -> "AnimeRating":
        if not isinstance(value, str):
            return cls.UNKNOWN
        _map = {
            "g":     cls.G,
            "pg":    cls.PG,
            "pg_13": cls.PG13,
            "r17+":  cls.R,
            "r+":    cls.RPLUS,
            "rx":    cls.RX,
        }
        return _map.get(value.lower(), cls.UNKNOWN)


class AnimeSeason(str, Enum):
    """Anime season classification."""

    SPRING = "SPRING"
    SUMMER = "SUMMER"
    FALL = "FALL"
    WINTER = "WINTER"

    @classmethod
    def _missing_(cls, value: object) -> "AnimeSeason | None":
        if not isinstance(value, str):
            return None  # type: ignore[return-value]
        _map = {
            "spring": cls.SPRING,
            "summer": cls.SUMMER,
            "fall":   cls.FALL,
            "winter": cls.WINTER,
        }
        return _map.get(value.lower())  # type: ignore[return-value]


class CharacterRole(str, Enum):
    """Character significance within a specific anime."""

    BACKGROUND = "BACKGROUND"
    MAIN = "MAIN"
    SUPPORTING = "SUPPORTING"
    UNSPECIFIED = "UNSPECIFIED"

    @classmethod
    def _missing_(cls, value: object) -> "CharacterRole":
        """Normalize source-specific strings into standard Enum members.

        Handles values from MAL, AniList, AnimePlanet, AniSearch, and AniDB.
        """
        if not isinstance(value, str):
            return cls.UNSPECIFIED

        v = value.lower()
        _map = {
            # MAL / AniList / AnimePlanet / AniSearch
            "main": cls.MAIN,
            "supporting": cls.SUPPORTING,
            "background": cls.BACKGROUND,
            # AniDB
            "main character in": cls.MAIN,
            "secondary cast in": cls.SUPPORTING,
            "appears in": cls.BACKGROUND,
            "cameo appearance in": cls.BACKGROUND,
            # AnimePlanet / AniSearch
            "secondary": cls.SUPPORTING,
            "minor": cls.BACKGROUND,
            "extra": cls.BACKGROUND,
            "other": cls.BACKGROUND,
            # common / catch-all
            "sub": cls.SUPPORTING,
            "unknown": cls.BACKGROUND,
        }
        return _map.get(v, cls.UNSPECIFIED)


class SourceMaterialType(str, Enum):
    """Source material type — used on both Anime and RelatedSourceMaterial models."""

    BOOK = "BOOK"
    CARD_GAME = "CARD GAME"
    GAME = "GAME"
    LIGHT_NOVEL = "LIGHT NOVEL"
    MANGA = "MANGA"
    MIXED_MEDIA = "MIXED MEDIA"
    MUSIC = "MUSIC"
    NOVEL = "NOVEL"
    ORIGINAL = "ORIGINAL"
    OTHER = "OTHER"
    PICTURE_BOOK = "PICTURE BOOK"
    RADIO = "RADIO"
    UNKNOWN = "UNKNOWN"
    VISUAL_NOVEL = "VISUAL NOVEL"
    WEB_MANGA = "WEB MANGA"
    WEB_NOVEL = "WEB NOVEL"


class SourceMaterialRelationType(str, Enum):
    """Relation type between an anime and its original source work."""

    ADAPTATION = "ADAPTATION"
    SOURCE = "SOURCE"
    ALTERNATIVE = "ALTERNATIVE"
    SPIN_OFF = "SPIN_OFF"
    OTHER = "OTHER"


class AnimeRelationType(str, Enum):
    """Relation type between two anime entries (cross-source normalized).

    Maps from platform-specific strings:
    - MAL: "Sequel", "Prequel", "Alternative version", "Alternative setting",
           "Side story", "Full story", "Parent story", "Spin-off", "Summary",
           "Adaptation", "Character", "Crossover", "Other"
    - AniList: "SEQUEL", "PREQUEL", "ALTERNATIVE" (→ ALTERNATIVE_VERSION),
               "SIDE_STORY", "CHARACTER", "SUMMARY", "PARENT", "SPIN_OFF",
               "ADAPTATION", "OTHER"

    ALTERNATIVE_VERSION = same story retold (e.g. TV vs. movie cut).
    ALTERNATIVE_SETTING = same characters, different universe / AU.
    """

    ADAPTATION = "ADAPTATION"
    ALTERNATIVE_VERSION = "ALTERNATIVE VERSION"   # Same story, different version (e.g. TV vs movie cut)
    ALTERNATIVE_SETTING = "ALTERNATIVE SETTING"   # Same characters, different universe/AU
    CHARACTER = "CHARACTER"
    CROSSOVER = "CROSSOVER"
    FULL_STORY = "FULL_STORY"
    OTHER = "OTHER"
    PARENT_STORY = "PARENT_STORY"
    PREQUEL = "PREQUEL"
    SEQUEL = "SEQUEL"
    SIDE_STORY = "SIDE_STORY"
    SPIN_OFF = "SPIN_OFF"
    SUMMARY = "SUMMARY"


# =============================================================================
# SUPPORTING MODELS
# =============================================================================


class Ography(BaseModel):
    """A single anime or manga appearance entry on a character's page."""

    title: str = Field(..., description="Title of the anime or manga")
    role: CharacterRole = Field(default=CharacterRole.UNSPECIFIED, description="Character's role in this title")
    sources: list[str] = Field(
        default_factory=list,
        description="Source URLs for this entry (e.g., MAL anime/manga page URL)",
    )


class TrailerEntry(BaseModel):
    """Trailer information from external APIs"""

    source: str | None = Field(None, description="Trailer video URL")
    title: str | None = Field(None, description="Trailer title")
    thumbnail: str | None = Field(None, description="Trailer thumbnail URL")


class AiredDates(BaseModel):
    """Detailed airing dates."""

    aired_from: datetime | None = Field(None, description="Start date with timezone")
    aired_to: datetime | None = Field(None, description="End date with timezone")


class Broadcast(BaseModel):
    """Recurring broadcast schedule and premiere dates.

    Merges the weekly broadcast slot (from MAL/Jikan) with per-version
    airtimes and premiere dates (from AnimSchedule).
    """

    # Weekly recurring slot (from MAL/Jikan)
    day: str | None = Field(None, description="Broadcast day (e.g., 'Sundays')")
    time: str | None = Field(None, description="Broadcast time in JP timezone (e.g., '23:15')")
    timezone: str | None = Field(None, description="Broadcast timezone (e.g., 'JST')")

    # Per-version airtimes (from AnimSchedule)
    jp_time: str | None = Field(None, description="Japanese broadcast time with timezone")
    sub_time: str | None = Field(None, description="Subtitle broadcast time with timezone")
    dub_time: str | None = Field(None, description="Dub broadcast time with timezone")
    sub_delay_days: int | None = Field(None, description="Days after JP that sub drops")
    dub_delay_days: int | None = Field(None, description="Days after JP that dub drops")

    # Per-version premiere dates (from AnimSchedule)
    premiere_jp: datetime | None = Field(None, description="Original JP premiere date")
    premiere_sub: datetime | None = Field(None, description="Subtitle premiere date")
    premiere_dub: datetime | None = Field(None, description="Dub premiere date")


class AnimeHiatus(BaseModel):
    """Current hiatus snapshot from AnimSchedule."""

    reason: str | None = Field(None, description="Reason for hiatus or delay")
    hiatus_from: str | None = Field(None, description="Hiatus start date")
    hiatus_until: str | None = Field(None, description="Hiatus end date (None = still on hiatus)")


class RelatedSourceMaterial(BaseModel):
    """Source work (manga, light novel, visual novel, game, etc.) with multi-source data."""

    title: str = Field(..., description="Original work title")
    type: SourceMaterialType = Field(..., description="Source material type")
    sources: list[str] = Field(
        default_factory=list,
        description="Source URLs for this work from various platforms",
    )


class RelatedAnime(BaseModel):
    """Related anime entry with consolidated multi-source data."""

    title: str = Field(..., description="Related anime title")
    type: AnimeType = Field(
        ..., description="Type of the related anime (TV, MOVIE, OVA, etc.)"
    )
    sources: list[str] = Field(
        default_factory=list,
        description="Source URLs for this anime from various platforms",
    )


class StreamingEntry(BaseModel):
    """Streaming platform entry"""

    platform: str = Field(..., description="Streaming platform name")
    source: str = Field(..., description="Streaming URL")
    region: str | None = Field(None, description="Available regions")
    free: bool | None = Field(None, description="Free to watch")
    premium_required: bool | None = Field(None, description="Premium subscription required")
    dub_available: bool | None = Field(None, description="Dub available")
    subtitle_languages: list[str] = Field(
        default_factory=list, description="Available subtitle languages"
    )


class AnimeImages(BaseModel):
    """Anime image URLs organized by type, aggregated across all sources."""

    covers: list[str] = Field(default_factory=list, description="Cover/key visual image URLs")
    posters: list[str] = Field(default_factory=list, description="Poster image URLs")
    banners: list[str] = Field(default_factory=list, description="Banner/wide image URLs")


class ThemeEntry(BaseModel):
    """Thematic element with description"""

    name: str = Field(..., description="Theme name")
    description: str | None = Field(None, description="Theme description")


class EpisodeRange(BaseModel):
    """Represents a range of episodes (e.g., 1-12) or a single episode (start=end)."""

    start: int = Field(..., description="Starting episode number")
    end: int | None = Field(
        None, description="Ending episode number (None means ongoing)"
    )


class ThemeSong(BaseModel):
    """Opening or ending theme song entry with structured episode coverage."""

    title: str = Field(..., description="Theme song title")
    artist: str | None = Field(None, description="Artist name")
    episodes: list[EpisodeRange] = Field(
        default_factory=list, description="List of episode ranges covered by this song"
    )


class EpisodeCharacter(BaseModel):
    """Character appearance in a specific episode (community-contributed on MAL)."""

    # ── Scalar fields (alphabetical) ──────────────────────────────────────
    name: str = Field(..., description="Character name")
    role: CharacterRole = Field(..., description="Role in episode (Main or Supporting)")

    # ── Array fields (alphabetical) ───────────────────────────────────────
    sources: list[str] = Field(
        default_factory=list,
        description="Source URLs for this character (e.g., MAL character page URL)",
    )
    voice_actors: list["VoiceActor"] = Field(
        default_factory=list,
        description="Voice actors for this character in this episode",
    )


class EpisodeStaff(BaseModel):
    """Staff credit for a specific episode (community-contributed on MAL)."""

    # ── Scalar fields (alphabetical) ──────────────────────────────────────
    name: str = Field(..., description="Staff member name")
    role: str = Field(..., description="Role (Script, Animation Director, etc.)")

    # ── Array fields (alphabetical) ───────────────────────────────────────
    sources: list[str] = Field(
        default_factory=list,
        description="Source URLs for this staff member (e.g., MAL people page URL)",
    )


class StaffMember(BaseModel):
    """Individual staff member with multi-source integration"""

    staff_ids: dict[str, int] = Field(
        default_factory=dict, description="Staff IDs across platforms (anidb, anilist)"
    )
    name: str = Field(..., description="Staff member name")
    native_name: str | None = Field(None, description="Native language name")
    role: str = Field(..., description="Primary role")
    image: str | None = Field(None, description="Staff member image URL")
    biography: str | None = Field(None, description="Staff member biography")
    birth_date: str | None = Field(None, description="Birth date")
    hometown: str | None = Field(None, description="Hometown")
    primary_occupations: list[str] = Field(default_factory=list, description="Primary occupations")
    years_active: list[int] = Field(default_factory=list, description="Years active")
    gender: str | None = Field(None, description="Gender")
    blood_type: str | None = Field(None, description="Blood type")
    community_favorites: int | None = Field(None, description="Community favorites count")
    enhancement_status: str | None = Field(None, description="Enhancement status from AniList matching")


class VoiceActor(BaseModel):
    """Voice actor — unified model covering all detail levels, from episode reference to full entity.

    Sparse at fetch time (name + language only); fully populated after consolidation.
    Fields absent from a given source are omitted via model_dump(exclude_none=True).
    """

    # ── Scalar fields (alphabetical) ──────────────────────────────────────
    biography: str | None = Field(None, description="Voice actor biography")
    birth_date: str | None = Field(None, description="Birth date")
    blood_type: str | None = Field(None, description="Blood type")
    id: str | None = Field(None, description="Unique UUID for the voice actor")
    image: str | None = Field(None, description="Voice actor image URL")
    language: str | None = Field(None, description="Voice acting language (Japanese, English, etc.)")
    name: str = Field(..., description="Voice actor name")
    native_name: str | None = Field(None, description="Native language name")

    # ── Array / dict fields (alphabetical) ────────────────────────────────
    character_assignments: list[str] = Field(default_factory=list, description="Characters voiced")
    sources: list[str] = Field(default_factory=list, description="Voice actor profile URLs")


class CompanyEntry(BaseModel):
    """Studio/Producer/Licensor company entry"""

    name: str = Field(..., description="Company name")
    description: str | None = Field(None, description="Company bio/description")
    sources: list[str] = Field(default_factory=list, description="Canonical source URLs")


class ProductionStaff(BaseModel):
    """Production staff organized by role — supports unlimited roles dynamically"""

    model_config = ConfigDict(extra="allow")

    def get_all_roles(self) -> dict[str, list[StaffMember]]:
        """Return all non-empty extra role fields as a dictionary of live StaffMember instances."""
        extra = self.__pydantic_extra__ or {}
        return {k: v for k, v in extra.items() if isinstance(v, list) and v}


class StaffData(BaseModel):
    """Comprehensive staff data structure"""

    production_staff: ProductionStaff = Field(
        default_factory=ProductionStaff, description="Production staff by role"
    )


class ContextualRank(BaseModel):
    """Contextual ranking information from platforms like AniList"""

    rank: int = Field(..., description="Rank position")
    type: str = Field(..., description="Ranking type (POPULAR, RATED, etc.)")
    format: str | None = Field(None, description="Format context (TV, Movie, etc.)")
    year: int | None = Field(None, description="Year context")
    season: str | None = Field(None, description="Season context (SPRING, SUMMER, FALL, WINTER)")
    all_time: bool | None = Field(None, description="Whether this is an all-time ranking")


class Statistics(BaseModel):
    """Standardized statistics entry — AI maps all platforms to these uniform properties"""

    score: float | None = Field(None, description="Rating score (normalized to 0-10 scale)")
    scored_by: int | None = Field(None, description="Number of users who rated")
    rank: int | None = Field(None, description="Overall ranking position")
    popularity: int | None = Field(None, description="Popularity ranking position")
    members: int | None = Field(None, description="Total members/users tracking")
    favorites: int | None = Field(None, description="Number of users who favorited")
    contextual_ranks: list[ContextualRank] | None = Field(
        None, description="Contextual ranking achievements (e.g., 'Best of 2021')"
    )


class ScoreCalculations(BaseModel):
    """Aggregated score calculations across platforms"""

    arithmetic_geometric_mean: float | None = Field(
        None, description="Arithmetic-geometric mean of scores"
    )
    arithmetic_mean: float | None = Field(None, description="Arithmetic mean of scores")
    median: float | None = Field(None, description="Median of scores")


# =============================================================================
# ANIME ENTITY MODEL (becomes an Anime Point in the vector database)
# =============================================================================


class Anime(BaseModel):
    """Core anime data that becomes an Anime Point in the vector database."""

    # =====================================================================
    # SCALAR FIELDS (alphabetical)
    # =====================================================================
    background: str | None = Field(None, description="Background information from MAL")
    duration: int | None = Field(None, description="Episode duration in seconds")
    episode_count: int = Field(default=0, description="Number of episodes")
    id: str | None = Field(None, description="Unique identifier for the anime entry")
    entity_type: EntityType = Field(
        default=EntityType.ANIME,
        description="Entity type for vector search filtering",
    )
    month: str | None = Field(None, description="Premiere month from AnimSchedule")
    nsfw: bool | None = Field(None, description="Not Safe For Work flag from Kitsu")
    rating: AnimeRating = Field(default=AnimeRating.UNKNOWN, description="Content rating (PG-13, R, etc.)")
    season: AnimeSeason | None = Field(None, description="Anime season (SPRING, SUMMER, FALL, WINTER)")
    similarity_score: float | None = Field(
        None,
        description="Vector similarity score from Qdrant search (populated at query time, not persisted)",
        exclude=True,
    )
    source_material: SourceMaterialType | None = Field(
        None, description="Source material type (manga, light novel, etc.)"
    )
    status: AnimeStatus = Field(..., description="Airing status")
    synopsis: str | None = Field(None, description="Detailed anime synopsis from external sources")
    title: str = Field(..., description="Primary anime title")
    title_english: str | None = Field(None, description="English title")
    title_japanese: str | None = Field(None, description="Japanese title")
    type: AnimeType = Field(..., description="TV, Movie, OVA, etc.")
    year: int | None = Field(None, description="Release year")

    # =====================================================================
    # ARRAY FIELDS (alphabetical)
    # =====================================================================
    content_warnings: list[str] = Field(default_factory=list, description="Content warnings")
    demographics: list[str] = Field(
        default_factory=list, description="Target demographics (Shounen, Seinen, etc.)"
    )
    ending_themes: list[ThemeSong] = Field(default_factory=list, description="Ending theme songs")
    genres: list[str] = Field(default_factory=list, description="Anime genres")
    licensors: list[CompanyEntry] = Field(default_factory=list, description="Licensors")
    opening_themes: list[ThemeSong] = Field(default_factory=list, description="Opening theme songs")
    producers: list[CompanyEntry] = Field(default_factory=list, description="Producers")
    related_anime: dict[AnimeRelationType, list[RelatedAnime]] = Field(
        default_factory=dict,
        description="Related anime grouped by relationship type (SEQUEL, PREQUEL, etc.)",
    )
    related_source_material: dict[SourceMaterialRelationType, list[RelatedSourceMaterial]] = Field(
        default_factory=dict,
        description="Original source work relations (manga, light novel, visual novel, game, etc.)",
    )
    sources: list[str] = Field(..., description="Source URLs from various providers")
    streaming_sources: list[StreamingEntry] = Field(
        default_factory=list, description="Streaming platform information"
    )
    studios: list[CompanyEntry] = Field(default_factory=list, description="Animation studios")
    synonyms: list[str] = Field(default_factory=list, description="Alternative titles")
    tags: list[str] = Field(default_factory=list, description="Original tags from offline database")
    themes: list[ThemeEntry] = Field(default_factory=list, description="Thematic elements with descriptions")
    trailers: list[TrailerEntry] = Field(default_factory=list, description="Trailer information")

    # =====================================================================
    # OBJECT/DICT FIELDS (alphabetical)
    # =====================================================================
    aired_dates: AiredDates | None = Field(None, description="Detailed airing dates")
    broadcast: Broadcast | None = Field(
        None, description="Recurring broadcast schedule and premiere dates"
    )
    external_sources: dict[str, str] = Field(
        default_factory=dict, description="External links (official site, social media)"
    )
    hiatus: AnimeHiatus | None = Field(None, description="Current hiatus snapshot from AnimSchedule")
    images: AnimeImages = Field(
        default_factory=AnimeImages,
        description="Images organized by type (covers, posters, banners)",
    )
    score: ScoreCalculations | None = Field(
        None, description="Aggregated score calculations from all platforms"
    )
    staff_data: StaffData | None = Field(
        None, description="Comprehensive staff data with multi-source integration"
    )
    statistics: dict[str, Statistics] = Field(
        default_factory=dict,
        description="Standardized statistics from different platforms (mal, anilist, kitsu, animeschedule)",
    )


# =============================================================================
# CHARACTER ENTITY MODEL (becomes an Character Point in the vector database)
# =============================================================================




class Character(BaseModel):
    """Character data that becomes a Character Point in the vector database."""

    # =====================================================================
    # SCALAR FIELDS (alphabetical)
    # =====================================================================
    description: str | None = Field(None, description="Character description/biography")
    favorites: int | None = Field(None, description="Number of users who favorited this character")
    id: str | None = Field(None, description="Unique UUID for the character")
    entity_type: EntityType = Field(
        default=EntityType.CHARACTER,
        description="Entity type for vector search filtering",
    )
    name: str = Field(..., description="Character name")
    name_native: str | None = Field(None, description="Native language name (Japanese/Kanji)")
    roles: list[CharacterRole] = Field(
        default_factory=list,
        description="All roles this character has played across anime (Main, Supporting, etc.)",
    )

    # =====================================================================
    # ARRAY FIELDS (alphabetical)
    # =====================================================================
    anime_ids: list[str] = Field(
        default_factory=list,
        description="UUIDs of anime this character appears in",
    )
    animeography: list[Ography] = Field(
        default_factory=list,
        description="Anime appearances with role context (from MAL character page)",
    )
    traits: list[str] = Field(
        default_factory=list,
        description="Character traits/tags (e.g., 'Ninja', 'Pirates', 'Superpowers')",
    )
    images: list[str] = Field(default_factory=list, description="Character image URLs")
    mangaography: list[Ography] = Field(
        default_factory=list,
        description="Manga appearances with role context (from MAL character page)",
    )
    name_variations: list[str] = Field(
        default_factory=list, description="All name spellings and variations"
    )
    nicknames: list[str] = Field(default_factory=list, description="Character nicknames")
    sources: list[str] = Field(
        default_factory=list,
        description="Character profile URLs from different platforms",
    )
    voice_actors: list[VoiceActor] = Field(
        default_factory=list, description="Voice actor information"
    )

    # =====================================================================
    # OBJECT/DICT FIELDS (alphabetical)
    # =====================================================================
    attributes: dict[str, str] = Field(
        default_factory=dict,
        description="Biographical attributes scraped from character page (keys vary by series)",
    )
    spoilers: dict[str, str] = Field(
        default_factory=dict,
        description="Spoiler values keyed by field name (same keys as attributes); 'description' key holds the prose description spoiler",
    )


# =============================================================================
# EPISODE ENTITY MODEL (becomes an Episode Point in the vector database)
# =============================================================================


class Episode(BaseModel):
    """Episode data that becomes an Episode Point in the vector database."""

    # =====================================================================
    # SCALAR FIELDS (alphabetical)
    # =====================================================================
    aired: datetime | None = Field(None, description="Episode air date with timezone")
    anime_id: str | None = Field(None, description="UUID of the parent anime")
    description: str | None = Field(None, description="Episode description")
    duration: int | None = Field(None, description="Episode duration in seconds")
    episode_number: int = Field(..., description="Episode number")
    filler: bool = Field(default=False, description="Whether episode is filler")
    id: str | None = Field(None, description="Deterministic ID (hash of anime_id + episode_number)")
    entity_type: EntityType = Field(
        default=EntityType.EPISODE,
        description="Entity type for vector search filtering",
    )
    recap: bool = Field(default=False, description="Whether episode is recap")
    score: float | None = Field(None, description="Episode rating score")
    season_number: int | None = Field(None, description="Season number")
    synopsis: str | None = Field(None, description="Episode synopsis/description")
    title: str = Field(..., description="Primary episode title")
    title_japanese: str | None = Field(None, description="Japanese episode title")
    title_romaji: str | None = Field(None, description="Romanized episode title")

    # =====================================================================
    # ARRAY FIELDS (alphabetical)
    # =====================================================================
    characters: list["EpisodeCharacter"] = Field(
        default_factory=list,
        description="Characters appearing in this episode (community-contributed via MAL)",
    )
    images: list[str] = Field(default_factory=list, description="Episode image URLs")
    sources: list[str] = Field(
        default_factory=list,
        description="Episode page URLs from different platforms",
    )
    staff: list["EpisodeStaff"] = Field(
        default_factory=list,
        description="Staff credits for this episode (community-contributed via MAL)",
    )

    # =====================================================================
    # OBJECT/DICT FIELDS (alphabetical)
    # =====================================================================
    streaming: dict[str, str] = Field(
        default_factory=dict, description="Streaming platforms and URLs {platform: url}"
    )
# =============================================================================
# AGGREGATE MODEL (container for data ingestion)
# =============================================================================


class AnimeRecord(BaseModel):
    """Aggregate container for data ingestion that groups anime with its characters and episodes."""

    anime: Anime = Field(..., description="Core anime data")
    characters: list[Character] = Field(
        default_factory=list,
        description="Character information with multi-source support",
    )
    episodes: list[Episode] = Field(
        default_factory=list,
        description="Detailed episode information with multi-source integration",
    )
