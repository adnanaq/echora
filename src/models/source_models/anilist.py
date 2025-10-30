from typing import Any, Dict, List, Optional
from pydantic import BaseModel

# Based on the structure from anilist.json
# Fields are sorted by type (scalar, array, object) and then alphabetically.

class AnilistTitle(BaseModel):
    # SCALARS
    english: Optional[str] = None
    native: Optional[str] = None
    romaji: Optional[str] = None
    userPreferred: Optional[str] = None

class AnilistCoverImage(BaseModel):
    # SCALARS
    color: Optional[str] = None
    extraLarge: Optional[str] = None
    large: Optional[str] = None
    medium: Optional[str] = None

class AnilistTrailer(BaseModel):
    # SCALARS
    id: Optional[str] = None
    site: Optional[str] = None
    thumbnail: Optional[str] = None

class AnilistTag(BaseModel):
    # SCALARS
    category: Optional[str] = None
    description: Optional[str] = None
    id: int
    isAdult: bool
    isGeneralSpoiler: bool
    isMediaSpoiler: bool
    name: str
    rank: Optional[int] = None

class AnilistRelationNode(BaseModel):
    # SCALARS
    format: Optional[str] = None
    id: int
    status: Optional[str] = None
    # OBJECTS
    title: AnilistTitle

class AnilistRelationEdge(BaseModel):
    # SCALARS
    relationType: str
    # OBJECTS
    node: AnilistRelationNode

class AnilistRelations(BaseModel):
    # ARRAYS
    edges: List[AnilistRelationEdge]

class AnilistStudioNode(BaseModel):
    # SCALARS
    id: int
    isAnimationStudio: bool
    name: str

class AnilistStudioEdge(BaseModel):
    # SCALARS
    isMain: bool
    # OBJECTS
    node: AnilistStudioNode

class AnilistStudios(BaseModel):
    # ARRAYS
    edges: List[AnilistStudioEdge]

class AnilistExternalLink(BaseModel):
    # SCALARS
    color: Optional[str] = None
    icon: Optional[str] = None
    id: int
    language: Optional[str] = None
    site: str
    type: Optional[str] = None
    url: str

class AnilistStreamingEpisode(BaseModel):
    # SCALARS
    site: Optional[str] = None
    thumbnail: Optional[str] = None
    title: Optional[str] = None
    url: Optional[str] = None

class AnilistAiringEpisode(BaseModel):
    # SCALARS
    airingAt: int
    episode: int
    timeUntilAiring: int

class AnilistRanking(BaseModel):
    # SCALARS
    allTime: Optional[bool] = None
    context: str
    format: str
    id: int
    rank: int
    season: Optional[str] = None
    type: str
    year: Optional[int] = None

class AnilistScoreDistribution(BaseModel):
    # SCALARS
    amount: int
    score: int

class AnilistStatusDistribution(BaseModel):
    # SCALARS
    amount: int
    status: str

class AnilistStats(BaseModel):
    # ARRAYS
    scoreDistribution: List[AnilistScoreDistribution] = []
    statusDistribution: List[AnilistStatusDistribution] = []

# New models for Staff and AiringSchedule
class AnilistStaffName(BaseModel):
    # SCALARS
    full: Optional[str] = None
    native: Optional[str] = None

class AnilistStaffNode(BaseModel):
    # SCALARS
    id: int
    # OBJECTS
    name: AnilistStaffName

class AnilistStaffEdge(BaseModel):
    # OBJECTS
    node: AnilistStaffNode
    # SCALARS
    role: Optional[str] = None

class AnilistStaff(BaseModel):
    # ARRAYS
    edges: List[AnilistStaffEdge]

class AnilistAiringScheduleNode(BaseModel):
    # SCALARS
    id: int
    episode: int
    airingAt: int

class AnilistAiringScheduleEdge(BaseModel):
    # OBJECTS
    node: AnilistAiringScheduleNode

class AnilistAiringSchedule(BaseModel):
    # ARRAYS
    edges: List[AnilistAiringScheduleEdge]

class AnilistAnimeData(BaseModel):
    # SCALARS
    averageScore: Optional[int] = None
    bannerImage: Optional[str] = None
    countryOfOrigin: Optional[str] = None
    description: Optional[str] = None
    duration: Optional[int] = None
    episodes: Optional[int] = None
    favourites: Optional[int] = None
    format: Optional[str] = None
    hashtag: Optional[str] = None
    id: int
    idMal: Optional[int] = None
    isAdult: bool
    meanScore: Optional[int] = None
    popularity: Optional[int] = None
    updatedAt: int # Added this field

    # ARRAYS
    externalLinks: List[AnilistExternalLink] = []
    genres: List[str] = []
    rankings: List[AnilistRanking] = []
    streamingEpisodes: List[AnilistStreamingEpisode] = []
    synonyms: List[str] = []
    tags: List[AnilistTag] = []

    # OBJECTS
    airingSchedule: AnilistAiringSchedule # Added this field
    coverImage: AnilistCoverImage
    nextAiringEpisode: Optional[AnilistAiringEpisode] = None
    relations: AnilistRelations
    staff: AnilistStaff # Added this field
    stats: Optional[AnilistStats] = None
    studios: AnilistStudios
    title: AnilistTitle
    trailer: Optional[AnilistTrailer] = None

# The root of the anilist.json is the data object itself
class AnilistApiResponse(AnilistAnimeData):
    pass