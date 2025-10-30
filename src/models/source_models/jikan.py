from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

# Based on the structure from https://api.jikan.moe/v4/anime/21
# Fields are sorted by type (scalar, array, object) and then alphabetically.

class JikanImage(BaseModel):
    # SCALARS
    image_url: Optional[str] = None
    large_image_url: Optional[str] = None
    small_image_url: Optional[str] = None

class JikanImages(BaseModel):
    # OBJECTS
    jpg: Optional[JikanImage] = None
    webp: Optional[JikanImage] = None

class JikanTrailerImages(BaseModel):
    # SCALARS
    image_url: Optional[str] = None
    large_image_url: Optional[str] = None
    medium_image_url: Optional[str] = None
    maximum_image_url: Optional[str] = None
    small_image_url: Optional[str] = None

class JikanTrailer(BaseModel):
    # SCALARS
    embed_url: Optional[str] = None
    url: Optional[str] = None
    youtube_id: Optional[str] = None
    # OBJECTS
    images: Optional[JikanTrailerImages] = None

class JikanTitle(BaseModel):
    # SCALARS
    title: str
    type: str

class JikanAiredPropDate(BaseModel):
    # SCALARS
    day: Optional[int] = None
    month: Optional[int] = None
    year: Optional[int] = None

class JikanAiredProp(BaseModel):
    # OBJECTS
    from_prop: Optional[JikanAiredPropDate] = Field(None, alias='from')
    to: Optional[JikanAiredPropDate] = None

class JikanAired(BaseModel):
    # SCALARS
    from_date: Optional[str] = Field(None, alias='from')
    string: Optional[str] = None
    to: Optional[str] = None
    # OBJECTS
    prop: Optional[JikanAiredProp] = None

class JikanBroadcast(BaseModel):
    # SCALARS
    day: Optional[str] = None
    string: Optional[str] = None
    time: Optional[str] = None
    timezone: Optional[str] = None

class JikanMalUrl(BaseModel):
    # SCALARS
    mal_id: int
    name: str
    type: str
    url: str

class JikanRelation(BaseModel):
    # SCALARS
    relation: str
    # ARRAYS
    entry: List[JikanMalUrl]

class JikanTheme(BaseModel):
    # ARRAYS
    endings: List[str] = []
    openings: List[str] = []

class JikanExternalLink(BaseModel):
    # SCALARS
    name: str
    url: str

class JikanAnimeData(BaseModel):
    # SCALARS
    airing: bool
    approved: bool
    background: Optional[str] = None
    duration: Optional[str] = None
    episodes: Optional[int] = None
    favorites: Optional[int] = None
    mal_id: int
    members: Optional[int] = None
    popularity: Optional[int] = None
    rank: Optional[int] = None
    rating: Optional[str] = None
    score: Optional[float] = None
    scored_by: Optional[int] = None
    season: Optional[str] = None
    source: Optional[str] = None
    status: Optional[str] = None
    synopsis: Optional[str] = None
    title: str
    title_english: Optional[str] = None
    title_japanese: Optional[str] = None
    type: Optional[str] = None
    url: str
    year: Optional[int] = None

    # ARRAYS
    demographics: List[JikanMalUrl] = []
    explicit_genres: List[Any] = []
    external: List[JikanExternalLink] = []
    genres: List[JikanMalUrl] = []
    licensors: List[JikanMalUrl] = []
    producers: List[JikanMalUrl] = []
    relations: List[JikanRelation] = []
    streaming: List[JikanExternalLink] = []
    studios: List[JikanMalUrl] = []
    themes: List[JikanMalUrl] = []
    title_synonyms: List[str] = []
    titles: List[JikanTitle]

    # OBJECTS
    aired: JikanAired
    broadcast: Optional[JikanBroadcast] = None
    images: JikanImages
    theme: JikanTheme = Field(default_factory=JikanTheme)
    trailer: JikanTrailer

class JikanApiResponse(BaseModel):
    # OBJECTS
    data: JikanAnimeData