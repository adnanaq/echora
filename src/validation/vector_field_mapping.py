#!/usr/bin/env python3
"""Vector field mapping definitions based on 11-vector architecture."""

from typing import Dict, List

# 11-Vector Architecture Field Mappings (from tasks/tasks_plan.md)
VECTOR_FIELD_MAPPINGS: Dict[str, Dict[str, List[str]]] = {
    # Text Vectors (BGE-M3, 1024-dim each)
    "title_vector": {
        "fields": [
            "title",
            "title_english",
            "title_japanese",
            "synopsis",
            "background",
            "synonyms",
        ],
        "description": "Title information and basic descriptions",
    },
    "character_vector": {
        "fields": ["characters"],
        "description": "Character names, descriptions, relationships, multi-source data",
    },
    "genre_vector": {
        "fields": ["genres", "tags", "themes", "demographics", "content_warnings"],
        "description": "Comprehensive classification and content categorization",
    },
    "staff_vector": {
        "fields": ["staff_data"],
        "description": "Directors, composers, studios, voice actors, multi-source integration",
    },
    "temporal_vector": {
        "fields": [
            "aired_dates",
            "broadcast",
            "broadcast_schedule",
            "delay_information",
            "premiere_dates",
        ],
        "description": "Semantic temporal data and scheduling information",
    },
    "streaming_vector": {
        "fields": ["streaming_info", "streaming_licenses"],
        "description": "Platform availability and licensing information",
    },
    "related_vector": {
        "fields": ["related_anime", "relations"],
        "description": "Franchise connections with URLs",
    },
    "franchise_vector": {
        "fields": ["trailers", "opening_themes", "ending_themes"],
        "description": "Multimedia content and franchise materials",
    },
    "episode_vector": {
        "fields": ["episode_details"],
        "description": "Detailed episode information, filler/recap status",
    },
    # Visual Vectors (JinaCLIP v2, 1024-dim each)
    "image_vector": {
        "fields": ["images"],
        "description": "General anime visual content (covers, posters, banners, trailer thumbnails)",
    },
    "character_image_vector": {
        "fields": ["characters.images"],
        "description": "Character visual content from characters.images (Dict[str, str] per character)",
    },
}

# Fields that appear in both vectors and payload (dual-indexed)
DUAL_INDEXED_FIELDS = ["title", "genres", "tags", "demographics", "type", "status"]

# Payload-only fields (precise filtering, no semantic search)
PAYLOAD_ONLY_FIELDS = [
    "id",
    "episodes",
    "rating",
    "nsfw",
    "anime_season",
    "duration",
    "sources",
    "statistics",
    "score",
]

# Non-indexed payload (storage only)
NON_INDEXED_PAYLOAD = ["enrichment_metadata", "images"]


def get_vector_fields(vector_name: str) -> List[str]:
    """Get the list of fields indexed in a specific vector."""
    return VECTOR_FIELD_MAPPINGS.get(vector_name, {}).get("fields", [])


def get_vector_description(vector_name: str) -> str:
    """Get the description of what a specific vector contains."""
    return VECTOR_FIELD_MAPPINGS.get(vector_name, {}).get(
        "description", "Unknown vector"
    )


def get_text_vectors() -> List[str]:
    """Get list of all text vector names."""
    return [
        name
        for name in VECTOR_FIELD_MAPPINGS.keys()
        if name not in ["image_vector", "character_image_vector"]
    ]


def get_image_vectors() -> List[str]:
    """Get list of all image vector names."""
    return ["image_vector", "character_image_vector"]


def is_vector_populated(vector_name: str) -> bool:
    """Check if a vector is typically populated based on field availability."""
    # Known empty vectors in current dataset
    typically_empty = {"review_vector", "streaming_vector"}
    return vector_name not in typically_empty


def get_searchable_vectors() -> List[str]:
    """Get list of vectors that should contain meaningful data for search."""
    return [name for name in VECTOR_FIELD_MAPPINGS.keys() if is_vector_populated(name)]
