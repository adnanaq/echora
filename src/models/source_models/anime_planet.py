
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

# Based on the structure from anime_planet.json
# Fields are sorted by type (scalar, array, object) and then alphabetically.

class APJsonLdPerson(BaseModel):
    # SCALARS
    name: str
    type: str = Field(..., alias='@type')
    url: str

class APJsonLd(BaseModel):
    # SCALARS
    context: str = Field(..., alias='@context')
    description: str
    image: str
    name: str
    numberOfEpisodes: int
    startDate: str
    type: str = Field(..., alias='@type')
    url: str
    # ARRAYS
    actor: List[APJsonLdPerson]
    creator: List[APJsonLdPerson]
    director: List[APJsonLdPerson]
    genre: List[str]

class APAggregateRating(BaseModel):
    # SCALARS
    bestRating: str
    ratingCount: str
    ratingValue: str
    type: str = Field(..., alias='@type')
    worstRating: str

class APReviewAuthor(BaseModel):
    # SCALARS
    name: str
    type: str = Field(..., alias='@type')

class APReviewRating(BaseModel):
    # SCALARS
    ratingValue: str
    type: str = Field(..., alias='@type')

class APReview(BaseModel):
    # SCALARS
    datePublished: str
    reviewBody: str
    type: str = Field(..., alias='@type')
    # OBJECTS
    author: APReviewAuthor
    reviewRating: APReviewRating

class APCharacter(BaseModel):
    # SCALARS
    name: str
    role: str
    # ARRAYS
    tags: List[str]

class APStaff(BaseModel):
    # SCALARS
    name: str
    role: str

class APRelatedAnime(BaseModel):
    # SCALARS
    relation_subtype: Optional[str] = None
    relation_type: str
    title: str
    type: str
    url: str

class AnimePlanetApiResponse(BaseModel):
    # SCALARS
    domain: str
    meta_description: str
    page_title: str
    rank: int
    synopsis: str
    type: str
    year: int

    # ARRAYS
    alt_titles: List[str]
    characters: List[APCharacter]
    related_anime: List[APRelatedAnime]
    staff: List[APStaff]
    tags: List[str]
    voice_actors: List[Dict[str, Any]]

    # OBJECTS
    aggregate_rating: APAggregateRating
    json_ld: APJsonLd
    review: APReview
