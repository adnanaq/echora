
from typing import Any, Dict, List, Optional
from pydantic import BaseModel

# Based on the structure from animeschedule.json
# Fields are sorted by type (scalar, array, object) and then alphabetically.

class AnimeScheduleSeason(BaseModel):
    # SCALARS
    route: str
    season: str
    title: str
    year: str

class AnimeScheduleEpisodeOverride(BaseModel):
    # SCALARS
    episodesAired: int
    overrideDate: str
    overrideEpisode: int

class AnimeScheduleGenre(BaseModel):
    # SCALARS
    name: str
    route: str

class AnimeScheduleStudio(BaseModel):
    # SCALARS
    name: str
    route: str

class AnimeScheduleSource(BaseModel):
    # SCALARS
    name: str
    route: str

class AnimeScheduleMediaType(BaseModel):
    # SCALARS
    name: str
    route: str

class AnimeScheduleStats(BaseModel):
    # SCALARS
    averageScore: float
    colorDarkMode: str
    colorLightMode: str
    ratingCount: int
    trackedCount: int
    trackedRating: int

class AnimeScheduleNames(BaseModel):
    # SCALARS
    abbreviation: str
    native: str
    # ARRAYS
    synonyms: List[str]

class AnimeScheduleRelations(BaseModel):
    # ARRAYS
    alternatives: List[str] = []
    other: List[str] = []
    prequels: List[str] = []
    sideStories: List[str] = []

class AnimeScheduleWebsite(BaseModel):
    # SCALARS
    name: str
    platform: str
    url: str

class AnimeScheduleWebsites(BaseModel):
    # SCALARS
    official: Optional[str] = None
    mal: Optional[str] = None
    aniList: Optional[str] = None
    kitsu: Optional[str] = None
    animePlanet: Optional[str] = None
    anidb: Optional[str] = None
    # ARRAYS
    streams: List[AnimeScheduleWebsite] = []

class AnimeScheduleApiResponse(BaseModel):
    # SCALARS
    createdAt: str
    delayedFrom: str
    delayedTimetable: str
    delayedUntil: str
    description: str
    dubDelayedFrom: str
    dubDelayedUntil: str
    dubPremier: str
    dubTime: str
    id: str
    imageVersionRoute: str
    jpnTime: str
    lengthMin: int
    month: str
    premier: str
    route: str
    status: str
    subDelayedFrom: str
    subDelayedUntil: str
    subPremier: str
    subTime: str
    title: str
    updatedAt: str
    year: int

    # ARRAYS
    genres: List[AnimeScheduleGenre]
    mediaTypes: List[AnimeScheduleMediaType]
    sources: List[AnimeScheduleSource]
    studios: List[AnimeScheduleStudio]

    # OBJECTS
    dubEpisodeOverride: AnimeScheduleEpisodeOverride
    episodeOverride: AnimeScheduleEpisodeOverride
    names: AnimeScheduleNames
    relations: AnimeScheduleRelations
    season: AnimeScheduleSeason
    stats: AnimeScheduleStats
    websites: AnimeScheduleWebsites
