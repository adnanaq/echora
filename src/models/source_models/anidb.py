from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel

# Based on the structure from anidb.json
# Fields are sorted by type (scalar, array, object) and then alphabetically.

class AnidbTitle(BaseModel):
    # SCALARS
    main: Optional[str] = None
    # ARRAYS
    short: List[str] = []
    synonyms: List[str] = []
    # OBJECTS
    official: Dict[str, str] = {}

class AnidbTag(BaseModel):
    # SCALARS
    count: int
    description: Optional[str] = None
    id: str
    name: str
    weight: int

class AnidbRatingValue(BaseModel):
    # SCALARS
    count: int
    value: float

class AnidbRatings(BaseModel):
    # OBJECTS
    permanent: AnidbRatingValue
    review: AnidbRatingValue
    temporary: AnidbRatingValue

class AnidbCreator(BaseModel):
    # SCALARS
    id: str
    name: str
    type: Optional[str] = None

class AnidbSeiyuu(BaseModel):
    # SCALARS
    id: str
    name: str
    picture: Optional[str] = None

class AnidbCharacter(BaseModel):
    # SCALARS
    character_type: str
    character_type_id: str
    description: Optional[str] = None
    gender: str
    id: str
    name: str
    picture: Optional[str] = None
    rating: Optional[float] = None
    rating_votes: Optional[int] = None
    type: str
    update: str
    # OBJECTS
    seiyuu: Optional[AnidbSeiyuu] = None

class AnidbRelatedAnime(BaseModel):
    # SCALARS
    id: str
    title: str
    type: str

class AnidbEpisode(BaseModel):
    # SCALARS
    air_date: Optional[str] = None
    episode_number: str
    episode_type: str
    id: str
    length: Optional[int] = None
    rating: Optional[float] = None
    rating_votes: Optional[int] = None
    summary: Optional[str] = None
    update: str
    # ARRAYS
    resources: List[Dict[str, Any]] = []
    # OBJECTS
    titles: Dict[str, str]

class AnidbApiResponse(BaseModel):
    # SCALARS
    anidb_id: str
    description: Optional[str] = None
    end_date: Optional[str] = None
    episode_count: str
    picture: Optional[str] = None
    start_date: Optional[str] = None
    type: str
    url: Optional[str] = None

    # ARRAYS
    categories: List[Any] = []
    characters: List[AnidbCharacter]
    creators: List[AnidbCreator]
    episodes: List[AnidbEpisode]
    related_anime: List[AnidbRelatedAnime]
    tags: List[AnidbTag]

    # OBJECTS
    ratings: AnidbRatings
    titles: AnidbTitle