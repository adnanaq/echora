
from typing import Any, Dict, List, Optional
from pydantic import BaseModel

# Based on the structure from kitsu.json
# Fields are sorted by type (scalar, array, object) and then alphabetically.

class KitsuTitles(BaseModel):
    # SCALARS
    en: Optional[str] = None
    en_jp: Optional[str] = None
    ja_jp: Optional[str] = None

class KitsuImageMeta(BaseModel):
    # SCALARS
    height: Optional[int] = None
    width: Optional[int] = None

class KitsuImageDimensions(BaseModel):
    # OBJECTS
    large: Optional[KitsuImageMeta] = None
    medium: Optional[KitsuImageMeta] = None
    small: Optional[KitsuImageMeta] = None
    tiny: Optional[KitsuImageMeta] = None

class KitsuImage(BaseModel):
    # SCALARS
    large: Optional[str] = None
    medium: Optional[str] = None
    original: Optional[str] = None
    small: Optional[str] = None
    tiny: Optional[str] = None
    # OBJECTS
    meta: Optional[KitsuImageDimensions] = None

class KitsuAnimeAttributes(BaseModel):
    # SCALARS
    ageRating: Optional[str] = None
    ageRatingGuide: Optional[str] = None
    averageRating: Optional[str] = None
    canonicalTitle: Optional[str] = None
    coverImageTopOffset: int
    createdAt: str
    description: Optional[str] = None
    endDate: Optional[str] = None
    episodeCount: Optional[int] = None
    episodeLength: Optional[int] = None
    favoritesCount: int
    nextRelease: Optional[str] = None
    nsfw: bool
    popularityRank: Optional[int] = None
    ratingRank: Optional[int] = None
    showType: Optional[str] = None
    slug: str
    startDate: Optional[str] = None
    status: Optional[str] = None
    subtype: Optional[str] = None
    synopsis: Optional[str] = None
    tba: Optional[str] = None
    totalLength: Optional[int] = None
    updatedAt: str
    userCount: int
    youtubeVideoId: Optional[str] = None

    # ARRAYS
    abbreviatedTitles: List[str] = []

    # OBJECTS
    coverImage: Optional[KitsuImage] = None
    posterImage: KitsuImage
    ratingFrequencies: Dict[str, str]
    titles: KitsuTitles

class KitsuGenericData(BaseModel):
    # SCALARS
    id: str
    type: str
    # OBJECTS
    attributes: Dict[str, Any] # Can be KitsuAnimeAttributes or KitsuEpisodeAttributes

class KitsuApiResponse(BaseModel):
    # OBJECTS
    anime: KitsuGenericData
    # ARRAYS
    episodes: List[KitsuGenericData] = []
