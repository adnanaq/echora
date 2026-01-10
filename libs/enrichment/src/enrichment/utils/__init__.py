"""Enrichment utility functions.

This package provides utility functions for the enrichment pipeline including:
- Advanced deduplication with semantic similarity support
- Japanese text normalization
"""

from enrichment.utils.deduplication import (
    SEMANTIC_SIMILARITY_THRESHOLD,
    deduplicate_semantic_array_field,
    deduplicate_simple_array_field,
    deduplicate_synonyms_language_aware,
    normalize_string_for_comparison,
)
from enrichment.utils.text_utils import normalize_japanese_text

__all__ = [
    "SEMANTIC_SIMILARITY_THRESHOLD",
    "deduplicate_semantic_array_field",
    "deduplicate_simple_array_field",
    "deduplicate_synonyms_language_aware",
    "normalize_japanese_text",
    "normalize_string_for_comparison",
]
