
from typing import Any, Dict, List, Optional
from pydantic import BaseModel

# Based on the structure from anisearch.json
# Fields are sorted by type (scalar, array, object) and then alphabetically.

class AniSearchStaff(BaseModel):
    # SCALARS
    name: str
    url: str

class AniSearchWebsite(BaseModel):
    # SCALARS
    name: str
    url: str

class AniSearchPublisher(BaseModel):
    # SCALARS
    name: str
    url: str

class AniSearchRelation(BaseModel):
    # SCALARS
    details: Optional[str] = None
    genres: Optional[str] = None
    image: Optional[str] = None
    rating: Optional[str] = None
    title: str
    type: str
    url: str

class AniSearchEpisode(BaseModel):
    # SCALARS
    episodeNumber: int
    releaseDate: Optional[str] = None
    runtime: Optional[str] = None
    title: Optional[str] = None

class AniSearchCharacter(BaseModel):
    # SCALARS
    favorites: Optional[int] = None
    image: Optional[str] = None
    name: str
    role: str
    url: str

class AniSearchApiResponse(BaseModel):
    # SCALARS
    cover_image: Optional[str] = None
    description: Optional[str] = None
    end_date: Optional[str] = None
    japanese_title: Optional[str] = None
    japanese_title_alt: Optional[str] = None
    source_material: Optional[str] = None
    start_date: Optional[str] = None
    status: Optional[str] = None
    studio: Optional[str] = None
    synonyms: Optional[str] = None
    type: Optional[str] = None

    # ARRAYS
    anime_relations: List[AniSearchRelation] = []
    characters: List[AniSearchCharacter] = []
    episodes: List[AniSearchEpisode] = []
    genres: List[str] = []
    manga_relations: List[AniSearchRelation] = []
    publishers: List[AniSearchPublisher] = []
    screenshots: List[str] = []
    staff: List[AniSearchStaff] = []
    tags: List[str] = []
    websites: List[AniSearchWebsite] = []
