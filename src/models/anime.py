# src/models/anime.py - Pydantic Models for Anime Data
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AnimeStatus(str, Enum):
    """Anime airing status classification."""

    FINISHED = "FINISHED"
    UPCOMING = "UPCOMING"
    ONGOING = "ONGOING"
    UNKNOWN = "UNKNOWN"


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


class AnimeSourceMaterial(str, Enum):
    """Anime source material classification."""

    ORIGINAL = "ORIGINAL"
    LIGHT_NOVEL = "LIGHT NOVEL"
    MANGA = "MANGA"
    OTHER = "OTHER"
    MIXED_MEDIA = "MIXED MEDIA"
    UNKNOWN = "UNKNOWN"
    GAME = "GAME"


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


class CharacterEntry(BaseModel):
    """Character information from external APIs with multi-source support"""

    # =====================================================================
    # SCALAR FIELDS (alphabetical)
    # =====================================================================
    age: Optional[str] = Field(None, description="Character age")
    description: Optional[str] = Field(
        None, description="Character description/biography"
    )
    eye_color: Optional[str] = Field(None, description="Eye color")
    favorites: Optional[int] = Field(
        None,
        description="Number of users who favorited this character (from Jikan/MAL + AniList)",
    )
    gender: Optional[str] = Field(None, description="Character gender")
    hair_color: Optional[str] = Field(None, description="Hair color")
    name: str = Field(..., description="Character name")
    name_native: Optional[str] = Field(
        None, description="Native language name (Japanese/Kanji)"
    )
    role: str = Field(..., description="Character role (Main, Supporting, etc.)")

    # =====================================================================
    # ARRAY FIELDS (alphabetical)
    # =====================================================================
    character_traits: List[str] = Field(
        default_factory=list,
        description="Character traits/tags from AnimePlanet (e.g., 'Ninja', 'Pirates', 'Superpowers')",
    )
    images: List[str] = Field(default_factory=list, description="Character image URLs")
    name_variations: List[str] = Field(
        default_factory=list, description="All name spellings and variations"
    )
    nicknames: List[str] = Field(
        default_factory=list, description="Character nicknames from Jikan API"
    )
    voice_actors: List["SimpleVoiceActor"] = Field(
        default_factory=list, description="Voice actor information"
    )

    # =====================================================================
    # OBJECT/DICT FIELDS (alphabetical)
    # =====================================================================
    character_pages: Dict[str, str] = Field(
        default_factory=dict,
        description="Character profile page URLs from different platforms (platform: url)",
    )


class EpisodeDetailEntry(BaseModel):
    """Comprehensive episode details with multi-source integration"""

    # =====================================================================
    # SCALAR FIELDS (alphabetical)
    # =====================================================================
    aired: Optional[datetime] = Field(
        None, description="Episode air date with timezone"
    )
    description: Optional[str] = Field(
        None, description="Episode description from Kitsu"
    )
    duration: Optional[int] = Field(None, description="Episode duration in seconds")
    episode_number: int = Field(..., description="Episode number")
    filler: bool = Field(default=False, description="Whether episode is filler")
    recap: bool = Field(default=False, description="Whether episode is recap")
    score: Optional[float] = Field(None, description="Episode rating score")
    season_number: Optional[int] = Field(None, description="Season number from Kitsu")
    synopsis: Optional[str] = Field(None, description="Episode synopsis/description")
    title: str = Field(..., description="Primary episode title")
    title_japanese: Optional[str] = Field(None, description="Japanese episode title")
    title_romaji: Optional[str] = Field(None, description="Romanized episode title")

    # =====================================================================
    # ARRAY FIELDS (alphabetical)
    # =====================================================================
    thumbnails: List[str] = Field(
        default_factory=list, description="Episode thumbnail URLs"
    )

    # =====================================================================
    # OBJECT/DICT FIELDS (alphabetical)
    # =====================================================================
    episode_pages: Dict[str, str] = Field(
        default_factory=dict,
        description="Episode page URLs from different platforms (mal, anilist, etc.)",
    )
    streaming: Dict[str, str] = Field(
        default_factory=dict, description="Streaming platforms and URLs {platform: url}"
    )


class TrailerEntry(BaseModel):
    """Trailer information from external APIs"""

    url: Optional[str] = Field(None, description="Trailer video URL")
    title: Optional[str] = Field(None, description="Trailer title")
    thumbnail_url: Optional[str] = Field(None, description="Trailer thumbnail URL")


class BroadcastSchedule(BaseModel):
    """Broadcast timing for different versions"""

    jpn_time: Optional[str] = Field(
        None, description="Japanese broadcast time with timezone"
    )
    sub_time: Optional[str] = Field(
        None, description="Subtitle broadcast time with timezone"
    )
    dub_time: Optional[str] = Field(
        None, description="Dub broadcast time with timezone"
    )


class DelayInformation(BaseModel):
    """Current delay status and reasons"""

    delayed_timetable: bool = Field(
        default=False, description="Whether timetable is delayed"
    )
    delayed_from: Optional[str] = Field(None, description="Delay start date")
    delayed_until: Optional[str] = Field(None, description="Delay end date")
    delay_reason: Optional[str] = Field(None, description="Reason for delay")


class PremiereDates(BaseModel):
    """Premiere dates for different versions"""

    original: Optional[str] = Field(None, description="Original premiere date")
    sub: Optional[str] = Field(None, description="Subtitle premiere date")
    dub: Optional[str] = Field(None, description="Dub premiere date")


class AiredDates(BaseModel):
    """Detailed airing dates"""

    from_date: Optional[datetime] = Field(
        None, alias="from", description="Start date with timezone"
    )
    to: Optional[datetime] = Field(None, description="End date with timezone")
    string: Optional[str] = Field(None, description="Human readable date range")


class Broadcast(BaseModel):
    """Broadcast schedule information"""

    day: Optional[str] = Field(None, description="Broadcast day")
    time: Optional[str] = Field(None, description="Broadcast time")
    timezone: Optional[str] = Field(None, description="Broadcast timezone")


class EnrichmentMetadata(BaseModel):
    """Metadata about data enrichment process"""

    source: str = Field(
        ..., description="Source of enrichment (mal, anilist, multi-source, etc.)"
    )
    enriched_at: datetime = Field(..., description="When enrichment was performed")
    success: bool = Field(default=True, description="Whether enrichment was successful")
    error_message: Optional[str] = Field(
        None, description="Error message if enrichment failed"
    )


class RelationEntry(BaseModel):
    """Related anime entry with multi-platform URLs"""

    title: str = Field(..., description="Related anime title")
    relation_type: str = Field(..., description="Relation type (sequel, prequel, etc.)")
    url: str = Field(..., description="Related anime URL")


class RelatedAnimeEntry(BaseModel):
    """Related anime entry from URL processing"""

    title: str = Field(..., description="Related anime title extracted from URL")
    relation_type: str = Field(
        ..., description="Relation type (Sequel, Prequel, Other, etc.)"
    )
    url: str = Field(..., description="Original URL")


class StreamingEntry(BaseModel):
    """Streaming platform entry"""

    platform: str = Field(..., description="Streaming platform name")
    url: str = Field(..., description="Streaming URL")
    region: Optional[str] = Field(None, description="Available regions")
    free: Optional[bool] = Field(None, description="Free to watch")
    premium_required: Optional[bool] = Field(
        None, description="Premium subscription required"
    )
    dub_available: Optional[bool] = Field(None, description="Dub available")
    subtitle_languages: List[str] = Field(
        default_factory=list, description="Available subtitle languages"
    )


class ThemeEntry(BaseModel):
    """Theme entry with description"""

    name: str = Field(..., description="Theme name")
    description: Optional[str] = Field(None, description="Theme description")


class ThemeSong(BaseModel):
    """Opening or ending theme song entry"""

    title: str = Field(..., description="Theme song title")
    artist: Optional[str] = Field(None, description="Artist name")
    episodes: Optional[str] = Field(None, description="Episode range (e.g., '1-26')")


class SimpleVoiceActor(BaseModel):
    """Simple voice actor reference for character entries"""

    name: str = Field(..., description="Voice actor name")
    language: str = Field(
        ..., description="Voice acting language (Japanese, English, etc.)"
    )


class VoiceActorEntry(BaseModel):
    """Voice actor entry for characters"""

    name: str = Field(..., description="Voice actor name")
    native_name: Optional[str] = Field(None, description="Native language name")
    language: str = Field(
        ..., description="Voice acting language (Japanese, English, etc.)"
    )
    image: Optional[str] = Field(None, description="Voice actor image URL")


class StaffMember(BaseModel):
    """Individual staff member with multi-source integration"""

    staff_ids: Dict[str, int] = Field(
        default_factory=dict, description="Staff IDs across platforms (anidb, anilist)"
    )
    name: str = Field(..., description="Staff member name")
    native_name: Optional[str] = Field(None, description="Native language name")
    role: str = Field(..., description="Primary role")
    image: Optional[str] = Field(None, description="Staff member image URL")
    biography: Optional[str] = Field(None, description="Staff member biography")
    birth_date: Optional[str] = Field(None, description="Birth date")
    hometown: Optional[str] = Field(None, description="Hometown")
    primary_occupations: List[str] = Field(
        default_factory=list, description="Primary occupations"
    )
    years_active: List[int] = Field(default_factory=list, description="Years active")
    gender: Optional[str] = Field(None, description="Gender")
    blood_type: Optional[str] = Field(None, description="Blood type")
    community_favorites: Optional[int] = Field(
        None, description="Community favorites count"
    )
    enhancement_status: Optional[str] = Field(
        None, description="Enhancement status from AniList matching"
    )


class VoiceActor(BaseModel):
    """Voice actor with character assignments"""

    staff_ids: Dict[str, int] = Field(
        default_factory=dict, description="Staff IDs across platforms"
    )
    name: str = Field(..., description="Voice actor name")
    native_name: Optional[str] = Field(None, description="Native language name")
    character_assignments: List[str] = Field(
        default_factory=list, description="Characters voiced"
    )
    image: Optional[str] = Field(None, description="Voice actor image URL")
    biography: Optional[str] = Field(None, description="Voice actor biography")
    birth_date: Optional[str] = Field(None, description="Birth date")
    blood_type: Optional[str] = Field(None, description="Blood type")


class CompanyEntry(BaseModel):
    """Studio/Producer/Licensor company entry"""

    name: str = Field(..., description="Company name")
    type: str = Field(
        ..., description="Company type (animation_studio, producer, licensor)"
    )
    url: Optional[str] = Field(None, description="Company URL")


class ProductionStaff(BaseModel):
    """Production staff organized by role - supports unlimited roles dynamically"""

    class Config:
        extra = "allow"  # Accept any role field name dynamically

    def get_all_roles(self) -> Dict[str, List[StaffMember]]:
        """Get all role fields as dictionary for dynamic access"""
        roles = {}
        # Use model_dump() to access all fields including dynamic ones
        all_data = self.model_dump()
        for field_name, value in all_data.items():
            if isinstance(value, list) and value:
                # Validate it contains staff member objects
                if all(isinstance(item, (dict, StaffMember)) for item in value):
                    roles[field_name] = value
        return roles


class VoiceActors(BaseModel):
    """Voice actors organized by language"""

    japanese: List[VoiceActor] = Field(
        default_factory=list, description="Japanese voice actors"
    )


class StaffData(BaseModel):
    """Comprehensive staff data structure"""

    production_staff: ProductionStaff = Field(
        default_factory=ProductionStaff, description="Production staff by role"
    )
    studios: List[CompanyEntry] = Field(
        default_factory=list, description="Animation studios"
    )
    producers: List[CompanyEntry] = Field(default_factory=list, description="Producers")
    licensors: List[CompanyEntry] = Field(default_factory=list, description="Licensors")
    voice_actors: VoiceActors = Field(
        default_factory=VoiceActors, description="Voice actors by language"
    )


class ContextualRank(BaseModel):
    """Contextual ranking information from platforms like AniList"""

    rank: int = Field(..., description="Rank position")
    type: str = Field(..., description="Ranking type (POPULAR, RATED, etc.)")
    format: Optional[str] = Field(None, description="Format context (TV, Movie, etc.)")
    year: Optional[int] = Field(None, description="Year context")
    season: Optional[str] = Field(
        None, description="Season context (SPRING, SUMMER, FALL, WINTER)"
    )
    all_time: Optional[bool] = Field(
        None, description="Whether this is an all-time ranking"
    )


class StatisticsEntry(BaseModel):
    """Standardized statistics entry - AI maps all platforms to these uniform properties"""

    score: Optional[float] = Field(
        None, description="Rating score (normalized to 0-10 scale)"
    )
    scored_by: Optional[int] = Field(None, description="Number of users who rated")
    rank: Optional[int] = Field(None, description="Overall ranking position")
    popularity_rank: Optional[int] = Field(
        None, description="Popularity ranking position"
    )
    members: Optional[int] = Field(None, description="Total members/users tracking")
    favorites: Optional[int] = Field(None, description="Number of users who favorited")
    contextual_ranks: Optional[List[ContextualRank]] = Field(
        None, description="Contextual ranking achievements (e.g., 'Best of 2021')"
    )


class ScoreCalculations(BaseModel):
    """Aggregated score calculations across platforms"""

    arithmeticGeometricMean: Optional[float] = Field(
        None, description="Arithmetic-geometric mean of scores"
    )
    arithmeticMean: Optional[float] = Field(
        None, description="Arithmetic mean of scores"
    )
    median: Optional[float] = Field(None, description="Median of scores")


class AnimeEntry(BaseModel):
    """Anime entry from anime-offline-database with comprehensive enhancement support"""

    # =====================================================================
    # SCALAR FIELDS (alphabetical)
    # =====================================================================
    background: Optional[str] = Field(
        None, description="Background information from MAL"
    )
    episodes: int = Field(default=0, description="Number of episodes")
    id: str = Field(..., description="Unique identifier for the anime entry")
    month: Optional[str] = Field(None, description="Premiere month from AnimSchedule")
    nsfw: Optional[bool] = Field(None, description="Not Safe For Work flag from Kitsu")
    similarity_score: Optional[float] = Field(
        None, description="Vector similarity score from Qdrant search (populated at query time, not persisted)"
    )
    rating: Optional[AnimeRating] = Field(
        None, description="Content rating (PG-13, R, etc.)"
    )
    season: Optional[AnimeSeason] = Field(
        None, description="Anime season (SPRING, SUMMER, FALL, WINTER)"
    )
    source_material: Optional[AnimeSourceMaterial] = Field(
        None, description="Source material (manga, light novel, etc.)"
    )
    status: AnimeStatus = Field(..., description="Airing status")
    synopsis: Optional[str] = Field(
        None, description="Detailed anime synopsis from external sources"
    )
    title: str = Field(..., description="Primary anime title")
    title_english: Optional[str] = Field(None, description="English title")
    title_japanese: Optional[str] = Field(None, description="Japanese title")
    type: AnimeType = Field(..., description="TV, Movie, OVA, etc.")
    year: Optional[int] = Field(None, description="Release year")

    # =====================================================================
    # ARRAY FIELDS (alphabetical)
    # =====================================================================
    characters: List[CharacterEntry] = Field(
        default_factory=list,
        description="Character information with multi-source support",
    )
    content_warnings: List[str] = Field(
        default_factory=list, description="Content warnings"
    )
    demographics: List[str] = Field(
        default_factory=list, description="Target demographics (Shounen, Seinen, etc.)"
    )
    ending_themes: List["ThemeSong"] = Field(
        default_factory=list, description="Ending theme songs"
    )
    episode_details: List[EpisodeDetailEntry] = Field(
        default_factory=list,
        description="Detailed episode information with multi-source integration",
    )
    genres: List[str] = Field(
        default_factory=list, description="Anime genres from AniList/other sources"
    )
    opening_themes: List["ThemeSong"] = Field(
        default_factory=list, description="Opening theme songs"
    )
    related_anime: List[RelatedAnimeEntry] = Field(
        default_factory=list, description="Related anime entries from URL processing"
    )
    relations: List[RelationEntry] = Field(
        default_factory=list, description="Related anime with platform URLs"
    )
    sources: List[str] = Field(..., description="Source URLs from various providers")
    streaming_info: List[StreamingEntry] = Field(
        default_factory=list, description="Streaming platform information"
    )
    streaming_licenses: List[str] = Field(
        default_factory=list, description="Streaming licenses"
    )
    synonyms: List[str] = Field(default_factory=list, description="Alternative titles")
    tags: List[str] = Field(
        default_factory=list, description="Original tags from offline database"
    )
    themes: List[ThemeEntry] = Field(
        default_factory=list, description="Thematic elements with descriptions"
    )
    trailers: List[TrailerEntry] = Field(
        default_factory=list, description="Trailer information from external APIs"
    )

    # =====================================================================
    # OBJECT/DICT FIELDS (alphabetical)
    # =====================================================================
    aired_dates: Optional["AiredDates"] = Field(
        None, description="Detailed airing dates"
    )
    broadcast: Optional["Broadcast"] = Field(
        None, description="Broadcast schedule information"
    )
    broadcast_schedule: Optional["BroadcastSchedule"] = Field(
        None,
        description="Broadcast timing for different versions (jpn_time, sub_time, dub_time)",
    )
    delay_information: Optional["DelayInformation"] = Field(
        None, description="Current delay status and reasons"
    )
    duration: Optional[int] = Field(
        None,
        description="Episode duration in seconds",
    )
    enrichment_metadata: Optional[EnrichmentMetadata] = Field(
        None, description="Metadata about enrichment process"
    )
    external_links: Dict[str, str] = Field(
        default_factory=dict, description="External links (official site, social media)"
    )
    images: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Images organized by type (covers, posters, banners) with URLs only",
    )
    popularity_trends: Optional[Dict[str, Any]] = Field(
        None, description="Popularity trend data"
    )
    premiere_dates: Optional["PremiereDates"] = Field(
        None, description="Premiere dates for different versions (original, sub, dub)"
    )
    score: Optional[ScoreCalculations] = Field(
        None,
        description="Aggregated score calculations from all platforms",
    )
    staff_data: Optional[StaffData] = Field(
        None, description="Comprehensive staff data with multi-source integration"
    )
    statistics: Dict[str, StatisticsEntry] = Field(
        default_factory=dict,
        description="Standardized statistics from different platforms (mal, anilist, kitsu, animeschedule)",
    )
