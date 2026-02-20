"""Pydantic models for anime, character, and episode data used across Echora services."""
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AnimeStatus(str, Enum):
    """Anime airing status classification."""

    FINISHED = "FINISHED"
    UPCOMING = "UPCOMING"
    ONGOING = "ONGOING"
    UNKNOWN = "UNKNOWN"


class EntityType(str, Enum):
    """Primary entity type classification for vector search."""

    ANIME = "anime"
    CHARACTER = "character"
    EPISODE = "episode"


class AnimeType(str, Enum):
    """Anime type classification."""

    MOVIE = "MOVIE"
    ONA = "ONA"
    OVA = "OVA"
    SPECIAL = "SPECIAL"
    TV = "TV"
    UNKNOWN = "UNKNOWN"
    MUSIC = "MUSIC"
    PV = "PV"


class AnimeRating(str, Enum):
    """Anime content rating classification."""

    G = "G - All Ages"
    PG = "PG - Children"
    PG13 = "PG-13 - Teens 13 or older"
    R = "R - 17+ (violence & profanity)"
    RPLUS = "R+ - Mild Nudity"
    RX = "Rx - Hentai"


class AnimeSeason(str, Enum):
    """Anime season classification."""

    SPRING = "SPRING"
    SUMMER = "SUMMER"
    FALL = "FALL"
    WINTER = "WINTER"


class CharacterRole(str, Enum):
    """Character role in an anime — stored on anime_character junction table."""

    MAIN = "MAIN"
    SUPPORTING = "SUPPORTING"
    BACKGROUND = "BACKGROUND"


class SourceMaterialType(str, Enum):
    """Source material type — used for both anime.source_material and original_work.type."""

    MANGA = "MANGA"
    LIGHT_NOVEL = "LIGHT_NOVEL"
    NOVEL = "NOVEL"
    VISUAL_NOVEL = "VISUAL_NOVEL"
    GAME = "GAME"
    WEB_MANGA = "WEB_MANGA"
    WEB_NOVEL = "WEB_NOVEL"
    OTHER = "OTHER"
    ORIGINAL = "ORIGINAL"
    MIXED_MEDIA = "MIXED_MEDIA"
    UNKNOWN = "UNKNOWN"


class SourceMaterialRelationType(str, Enum):
    """Relation type between an anime and its original source work — stored on anime_original_work_relation.relation_type."""

    ADAPTATION = "ADAPTATION"
    SOURCE = "SOURCE"
    ALTERNATIVE = "ALTERNATIVE"
    SPIN_OFF = "SPIN_OFF"
    OTHER = "OTHER"


class AnimeRelationType(str, Enum):
    """Relation type between two anime — stored on anime_relation.relation_type."""

    SEQUEL = "SEQUEL"
    PREQUEL = "PREQUEL"
    SIDE_STORY = "SIDE_STORY"
    ALTERNATIVE = "ALTERNATIVE"
    SUMMARY = "SUMMARY"
    FULL_STORY = "FULL_STORY"
    CHARACTER = "CHARACTER"
    SPIN_OFF = "SPIN_OFF"
    ADAPTATION = "ADAPTATION"
    OTHER = "OTHER"


# =============================================================================
# ENTITY MODELS (each becomes a vector point)
# =============================================================================


class Character(BaseModel):
    """Character data that becomes a Character Point in the vector database."""

    # =====================================================================
    # SCALAR FIELDS (alphabetical)
    # =====================================================================
    age: str | None = Field(None, description="Character age")
    description: str | None = Field(None, description="Character description/biography")
    eye_color: str | None = Field(None, description="Eye color")
    favorites: int | None = Field(
        None,
        description="Number of users who favorited this character (from Jikan/MAL + AniList)",
    )
    gender: str | None = Field(None, description="Character gender")
    hair_color: str | None = Field(None, description="Hair color")
    id: str | None = Field(None, description="Unique UUID for the character")
    entity_type: EntityType = Field(
        default=EntityType.CHARACTER,
        description="Entity type for vector search filtering",
    )
    name: str = Field(..., description="Character name")
    name_native: str | None = Field(
        None, description="Native language name (Japanese/Kanji)"
    )
    role: str = Field(..., description="Character role (Main, Supporting, etc.)")

    # =====================================================================
    # ARRAY FIELDS (alphabetical)
    # =====================================================================
    anime_ids: list[str] = Field(
        default_factory=list,
        description="List of UUIDs of anime this character appears in",
    )
    character_traits: list[str] = Field(
        default_factory=list,
        description="Character traits/tags from AnimePlanet (e.g., 'Ninja', 'Pirates', 'Superpowers')",
    )
    images: list[str] = Field(default_factory=list, description="Character image URLs")
    name_variations: list[str] = Field(
        default_factory=list, description="All name spellings and variations"
    )
    nicknames: list[str] = Field(
        default_factory=list, description="Character nicknames from Jikan API"
    )
    voice_actors: list["SimpleVoiceActor"] = Field(
        default_factory=list, description="Voice actor information"
    )
    sources: list[str] = Field(
        default_factory=list,
        description="Character profile URLs from different platforms",
    )


class Episode(BaseModel):
    """Episode data that becomes an Episode Point in the vector database."""

    # =====================================================================
    # SCALAR FIELDS (alphabetical)
    # =====================================================================
    aired: datetime | None = Field(None, description="Episode air date with timezone")
    anime_id: str | None = Field(None, description="UUID of the parent anime")
    description: str | None = Field(None, description="Episode description from Kitsu")
    duration: int | None = Field(None, description="Episode duration in seconds")
    episode_number: int = Field(..., description="Episode number")
    filler: bool = Field(default=False, description="Whether episode is filler")
    id: str | None = Field(
        None,
        description="Deterministic ID (hash of anime_id + episode_number)",
    )
    entity_type: EntityType = Field(
        default=EntityType.EPISODE,
        description="Entity type for vector search filtering",
    )
    recap: bool = Field(default=False, description="Whether episode is recap")
    score: float | None = Field(None, description="Episode rating score")
    season_number: int | None = Field(None, description="Season number from Kitsu")
    synopsis: str | None = Field(None, description="Episode synopsis/description")
    title: str = Field(..., description="Primary episode title")
    title_japanese: str | None = Field(None, description="Japanese episode title")
    title_romaji: str | None = Field(None, description="Romanized episode title")

    # =====================================================================
    # ARRAY FIELDS (alphabetical)
    # =====================================================================
    images: list[str] = Field(
        default_factory=list, description="Episode image URLs"
    )
    sources: list[str] = Field(
        default_factory=list,
        description="Episode page URLs from different platforms",
    )

    # =====================================================================
    # OBJECT/DICT FIELDS (alphabetical)
    # =====================================================================
    streaming: dict[str, str] = Field(
        default_factory=dict, description="Streaming platforms and URLs {platform: url}"
    )


# =============================================================================
# SUPPORTING MODELS
# =============================================================================


class TrailerEntry(BaseModel):
    """Trailer information from external APIs"""

    url: str | None = Field(None, description="Trailer video URL")
    title: str | None = Field(None, description="Trailer title")
    thumbnail_url: str | None = Field(None, description="Trailer thumbnail URL")




class AiredDates(BaseModel):
    """Detailed airing dates"""

    from_date: datetime | None = Field(
        None, alias="from", description="Start date with timezone"
    )
    to: datetime | None = Field(None, description="End date with timezone")
    string: str | None = Field(None, description="Human readable date range")


class Broadcast(BaseModel):
    """Recurring broadcast schedule and premiere dates — maps to anime_broadcast table"""

    # Weekly recurring slot (from Jikan)
    day: str | None = Field(None, description="Broadcast day (e.g. 'Sunday')")
    time: str | None = Field(None, description="Broadcast time in JP timezone")
    timezone: str | None = Field(None, description="Broadcast timezone (e.g. 'Asia/Tokyo')")

    # Per-version airtimes (from AnimSchedule)
    jp_time: str | None = Field(None, description="Japanese broadcast time with timezone")
    sub_time: str | None = Field(None, description="Subtitle broadcast time with timezone")
    dub_time: str | None = Field(None, description="Dub broadcast time with timezone")
    sub_delay_days: int | None = Field(None, description="Days after JP that sub drops")
    dub_delay_days: int | None = Field(None, description="Days after JP that dub drops")

    # Per-version premiere dates (from AnimSchedule)
    premiere_jp: str | None = Field(None, description="Original JP premiere date")
    premiere_sub: str | None = Field(None, description="Subtitle premiere date")
    premiere_dub: str | None = Field(None, description="Dub premiere date")


class AnimeHiatus(BaseModel):
    """Current hiatus snapshot from AnimSchedule — maps to anime_hiatus table"""

    reason: str | None = Field(None, description="Reason for hiatus or delay")
    hiatus_from: str | None = Field(None, description="Hiatus start date")
    hiatus_until: str | None = Field(None, description="Hiatus end date (null = still on hiatus)")


class EnrichmentMetadata(BaseModel):
    """Metadata about data enrichment process"""

    source: str = Field(
        ..., description="Source of enrichment (mal, anilist, multi-source, etc.)"
    )
    enriched_at: datetime = Field(..., description="When enrichment was performed")
    success: bool = Field(default=True, description="Whether enrichment was successful")
    error_message: str | None = Field(
        None, description="Error message if enrichment failed"
    )


class SourceMaterial(BaseModel):
    """Source work (manga, light novel, visual novel, game, etc.) entry with platform URLs"""

    title: str = Field(..., description="Original work title")
    relation_type: SourceMaterialRelationType = Field(..., description="Relation type (ADAPTATION, SOURCE, etc.)")
    url: str = Field(..., description="Original work URL")
    type: SourceMaterialType | None = Field(None, description="Source material type")


class RelatedAnimeEntry(BaseModel):
    """Related anime entry from URL processing"""

    title: str = Field(..., description="Related anime title extracted from URL")
    relation_type: AnimeRelationType = Field(..., description="Relation type (SEQUEL, PREQUEL, etc.)")
    url: str = Field(..., description="Original URL")
    type: AnimeType | None = Field(None, description="Type of the related anime (TV, MOVIE, OVA, etc.)")


class StreamingEntry(BaseModel):
    """Streaming platform entry"""

    platform: str = Field(..., description="Streaming platform name")
    url: str = Field(..., description="Streaming URL")
    region: str | None = Field(None, description="Available regions")
    free: bool | None = Field(None, description="Free to watch")
    premium_required: bool | None = Field(
        None, description="Premium subscription required"
    )
    dub_available: bool | None = Field(None, description="Dub available")
    subtitle_languages: list[str] = Field(
        default_factory=list, description="Available subtitle languages"
    )


class ThemeEntry(BaseModel):
    """Theme entry with description"""

    name: str = Field(..., description="Theme name")
    description: str | None = Field(None, description="Theme description")


class ThemeSong(BaseModel):
    """Opening or ending theme song entry"""

    title: str = Field(..., description="Theme song title")
    artist: str | None = Field(None, description="Artist name")
    episodes: str | None = Field(None, description="Episode range (e.g., '1-26')")


class SimpleVoiceActor(BaseModel):
    """Simple voice actor reference for character entries"""

    name: str = Field(..., description="Voice actor name")
    language: str = Field(
        ..., description="Voice acting language (Japanese, English, etc.)"
    )


class VoiceActorEntry(BaseModel):
    """Voice actor entry for characters"""

    name: str = Field(..., description="Voice actor name")
    native_name: str | None = Field(None, description="Native language name")
    language: str = Field(
        ..., description="Voice acting language (Japanese, English, etc.)"
    )
    image: str | None = Field(None, description="Voice actor image URL")


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
    primary_occupations: list[str] = Field(
        default_factory=list, description="Primary occupations"
    )
    years_active: list[int] = Field(default_factory=list, description="Years active")
    gender: str | None = Field(None, description="Gender")
    blood_type: str | None = Field(None, description="Blood type")
    community_favorites: int | None = Field(
        None, description="Community favorites count"
    )
    enhancement_status: str | None = Field(
        None, description="Enhancement status from AniList matching"
    )


class VoiceActor(BaseModel):
    """Voice actor with character assignments"""

    staff_ids: dict[str, int] = Field(
        default_factory=dict, description="Staff IDs across platforms"
    )
    name: str = Field(..., description="Voice actor name")
    native_name: str | None = Field(None, description="Native language name")
    character_assignments: list[str] = Field(
        default_factory=list, description="Characters voiced"
    )
    image: str | None = Field(None, description="Voice actor image URL")
    biography: str | None = Field(None, description="Voice actor biography")
    birth_date: str | None = Field(None, description="Birth date")
    blood_type: str | None = Field(None, description="Blood type")


class CompanyEntry(BaseModel):
    """Studio/Producer/Licensor company entry"""

    name: str = Field(..., description="Company name")
    description: str | None = Field(default=None, description="Company bio/description")
    sources: list[str] = Field(default_factory=list, description="Canonical source URLs")


class ProductionStaff(BaseModel):
    """Production staff organized by role - supports unlimited roles dynamically"""

    class Config:
        extra = "allow"  # Accept any role field name dynamically

    def get_all_roles(self) -> dict[str, list[StaffMember]]:
        """Return all non-empty role fields as a dictionary.

        Returns:
            Mapping of role name to list of ``StaffMember`` objects, including
            any dynamically added role fields. Empty lists are excluded.
        """
        roles = {}
        # Use model_dump() to access all fields including dynamic ones
        all_data = self.model_dump()
        for field_name, value in all_data.items():
            if isinstance(value, list) and value:
                # Validate it contains staff member objects
                if all(isinstance(item, dict | StaffMember) for item in value):
                    roles[field_name] = value
        return roles


class VoiceActors(BaseModel):
    """Voice actors organized by language"""

    japanese: list[VoiceActor] = Field(
        default_factory=list, description="Japanese voice actors"
    )


class StaffData(BaseModel):
    """Comprehensive staff data structure"""

    production_staff: ProductionStaff = Field(
        default_factory=ProductionStaff, description="Production staff by role"
    )
    licensors: list[CompanyEntry] = Field(default_factory=list, description="Licensors")
    voice_actors: VoiceActors = Field(
        default_factory=VoiceActors, description="Voice actors by language"
    )


class ContextualRank(BaseModel):
    """Contextual ranking information from platforms like AniList"""

    rank: int = Field(..., description="Rank position")
    type: str = Field(..., description="Ranking type (POPULAR, RATED, etc.)")
    format: str | None = Field(None, description="Format context (TV, Movie, etc.)")
    year: int | None = Field(None, description="Year context")
    season: str | None = Field(
        None, description="Season context (SPRING, SUMMER, FALL, WINTER)"
    )
    all_time: bool | None = Field(
        None, description="Whether this is an all-time ranking"
    )


class Statistics(BaseModel):
    """Standardized statistics entry - AI maps all platforms to these uniform properties"""

    score: float | None = Field(
        None, description="Rating score (normalized to 0-10 scale)"
    )
    scored_by: int | None = Field(None, description="Number of users who rated")
    rank: int | None = Field(None, description="Overall ranking position")
    popularity_rank: int | None = Field(None, description="Popularity ranking position")
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
    duration: int | None = Field(
        None,
        description="Episode duration in seconds",
    )
    episode_count: int = Field(default=0, description="Number of episodes")
    id: str = Field(..., description="Unique identifier for the anime entry")
    entity_type: EntityType = Field(
        default=EntityType.ANIME,
        description="Entity type for vector search filtering",
    )
    month: str | None = Field(None, description="Premiere month from AnimSchedule")
    nsfw: bool | None = Field(None, description="Not Safe For Work flag from Kitsu")
    rating: AnimeRating | None = Field(
        None, description="Content rating (PG-13, R, etc.)"
    )
    season: AnimeSeason | None = Field(
        None, description="Anime season (SPRING, SUMMER, FALL, WINTER)"
    )
    similarity_score: float | None = Field(
        None,
        description="Vector similarity score from Qdrant search (populated at query time, not persisted)",
        exclude=True,
    )
    source_material: SourceMaterialType | None = Field(
        None, description="Source material (manga, light novel, etc.)"
    )
    status: AnimeStatus = Field(..., description="Airing status")
    synopsis: str | None = Field(
        None, description="Detailed anime synopsis from external sources"
    )
    title: str = Field(..., description="Primary anime title")
    title_english: str | None = Field(None, description="English title")
    title_japanese: str | None = Field(None, description="Japanese title")
    type: AnimeType = Field(..., description="TV, Movie, OVA, etc.")
    year: int | None = Field(None, description="Release year")

    # =====================================================================
    # ARRAY FIELDS (alphabetical)
    # =====================================================================
    content_warnings: list[str] = Field(
        default_factory=list, description="Content warnings"
    )
    demographics: list[str] = Field(
        default_factory=list, description="Target demographics (Shounen, Seinen, etc.)"
    )
    ending_themes: list[ThemeSong] = Field(
        default_factory=list, description="Ending theme songs"
    )
    genres: list[str] = Field(
        default_factory=list, description="Anime genres from AniList/other sources"
    )
    opening_themes: list[ThemeSong] = Field(
        default_factory=list, description="Opening theme songs"
    )
    producers: list[CompanyEntry] = Field(default_factory=list, description="Producers")
    related_anime: list[RelatedAnimeEntry] = Field(
        default_factory=list, description="Related anime entries from URL processing"
    )
    related_source_material: list[SourceMaterial] = Field(
        default_factory=list, description="Source work relations (manga, light novel, visual novel, game, etc.)"
    )
    sources: list[str] = Field(..., description="Source URLs from various providers")
    streaming_info: list[StreamingEntry] = Field(
        default_factory=list, description="Streaming platform information"
    )
    streaming_licenses: list[str] = Field(
        default_factory=list, description="Streaming licenses"
    )
    studios: list[CompanyEntry] = Field(
        default_factory=list, description="Animation studios"
    )
    synonyms: list[str] = Field(default_factory=list, description="Alternative titles")
    tags: list[str] = Field(
        default_factory=list, description="Original tags from offline database"
    )
    themes: list[ThemeEntry] = Field(
        default_factory=list, description="Thematic elements with descriptions"
    )
    trailers: list[TrailerEntry] = Field(
        default_factory=list, description="Trailer information from external APIs"
    )

    # =====================================================================
    # OBJECT/DICT FIELDS (alphabetical)
    # =====================================================================
    aired_dates: AiredDates | None = Field(None, description="Detailed airing dates")
    broadcast: Broadcast | None = Field(
        None, description="Recurring broadcast schedule and premiere dates"
    )
    enrichment_metadata: EnrichmentMetadata | None = Field(
        None, description="Metadata about enrichment process"
    )
    external_links: dict[str, str] = Field(
        default_factory=dict, description="External links (official site, social media)"
    )
    hiatus: AnimeHiatus | None = Field(
        None, description="Current hiatus snapshot from AnimSchedule"
    )
    images: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Images organized by type (covers, posters, banners) with URLs only",
    )
    popularity_trends: dict[str, Any] | None = Field(
        None, description="Popularity trend data"
    )
    score: ScoreCalculations | None = Field(
        None,
        description="Aggregated score calculations from all platforms",
    )
    staff_data: StaffData | None = Field(
        None, description="Comprehensive staff data with multi-source integration"
    )
    statistics: dict[str, Statistics] = Field(
        default_factory=dict,
        description="Standardized statistics from different platforms (mal, anilist, kitsu, animeschedule)",
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
