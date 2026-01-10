"""Advanced deduplication utilities with semantic similarity support.

This module provides deduplication functions for array fields in the enrichment pipeline.
It supports simple string-based deduplication as well as semantic similarity-based
deduplication using embedding models with dependency injection.
"""

import logging
from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

try:
    from langdetect import LangDetectException, detect
except ImportError:
    detect = None
    LangDetectException = Exception

logger = logging.getLogger(__name__)

# Default semantic similarity threshold for deduplication
SEMANTIC_SIMILARITY_THRESHOLD = 0.85

__all__ = [
    "SEMANTIC_SIMILARITY_THRESHOLD",
    "deduplicate_semantic_array_field",
    "deduplicate_simple_array_field",
    "deduplicate_synonyms_language_aware",
    "normalize_string_for_comparison",
]


def normalize_string_for_comparison(text: str) -> str:
    """Normalize string for case-insensitive comparison.

    Args:
        text: Input string to normalize

    Returns:
        Normalized lowercase string with stripped whitespace
    """
    if not text:
        return ""
    return text.lower().strip()


def _is_semantically_duplicate(
    new_embedding: list[float],
    existing_embeddings: list[list[float]],
    threshold: float = SEMANTIC_SIMILARITY_THRESHOLD,
) -> bool:
    """Check if new embedding is semantically similar to existing ones.

    Args:
        new_embedding: Embedding of the new value
        existing_embeddings: List of embeddings for already accepted values
        threshold: Similarity threshold (default: SEMANTIC_SIMILARITY_THRESHOLD)

    Returns:
        True if similar to any existing embedding above threshold
    """
    if not existing_embeddings:
        return False
    similarities = cosine_similarity([new_embedding], existing_embeddings)[0]
    return float(np.max(similarities)) > threshold


def deduplicate_simple_array_field(
    offline_values: list[str], external_values: list[str]
) -> list[str]:
    """Deduplicate array field values with offline database as foundation using simple string comparison.

    Args:
        offline_values: Values from offline database (given priority)
        external_values: Values from external sources

    Returns:
        Deduplicated list with offline values first, then unique external values
    """
    result = []
    seen = set()

    # Add offline values first (they have priority)
    for value in offline_values:
        if value and value.strip():
            normalized = normalize_string_for_comparison(value)
            if normalized not in seen:
                result.append(value.strip())
                seen.add(normalized)

    # Add unique external values
    for value in external_values:
        if value and value.strip():
            normalized = normalize_string_for_comparison(value)
            if normalized not in seen:
                result.append(value.strip())
                seen.add(normalized)

    return result


def deduplicate_semantic_array_field(
    values: list[str], embedding_model: Any | None = None
) -> list[str]:
    """Deduplicate array field values using semantic similarity for cross-language support.

    Uses embedding-based semantic similarity to detect duplicates beyond simple string matching.
    If no embedding model is provided, falls back to simple string deduplication.

    Args:
        values: List of string values to deduplicate
        embedding_model: Optional TextEmbeddingModel instance for generating embeddings.
                        If None, falls back to simple string deduplication.

    Returns:
        Deduplicated list of string values
    """
    if embedding_model is None:
        # Fallback to simple deduplication if no model provided
        logger.debug("No embedding model provided, using simple deduplication")
        return deduplicate_simple_array_field(values, [])

    result_values = []
    result_embeddings = []

    for value in values:
        if not value or not value.strip():
            continue

        try:
            # Generate embedding using the injected model
            embeddings = embedding_model.encode([value.strip()])
            if not embeddings or embeddings[0] is None:
                # Fallback to simple string deduplication if embedding fails
                if value.strip() not in result_values:
                    result_values.append(value.strip())
                continue

            embedding = embeddings[0]

            if not _is_semantically_duplicate(embedding, result_embeddings):
                result_values.append(value.strip())
                result_embeddings.append(embedding)

        except Exception as e:
            logger.warning(f"Embedding generation failed for '{value[:50]}': {e}")
            # Fallback: add if not already present by exact match
            if value.strip() not in result_values:
                result_values.append(value.strip())

    return result_values


def deduplicate_synonyms_language_aware(
    values: list[str],
    embedding_model: Any | None = None,
    similarity_threshold: float = 0.85,
) -> list[str]:
    """Deduplicate synonyms preserving all language variants while deduplicating within each language.

    This function groups synonyms by detected language and applies semantic similarity
    deduplication only within each language group. This ensures that cross-language
    variants (e.g., "ONE PIECE" in English, "tek parça" in Turkish, "원피스" in Korean)
    are all preserved, while removing duplicates like case/punctuation variants within
    the same language.

    Args:
        values: List of synonym strings to deduplicate
        embedding_model: Optional TextEmbeddingModel instance for generating embeddings.
                        If None, falls back to simple string deduplication.
        similarity_threshold: Semantic similarity threshold for within-language deduplication
                             (default: 0.85)

    Returns:
        Deduplicated list preserving cross-language variants

    Example:
        >>> synonyms = ["ONE PIECE", "tek parça", "All'arrembaggio!", "all'arrembaggio!"]
        >>> deduplicate_synonyms_language_aware(synonyms, model)
        ["ONE PIECE", "tek parça", "All'arrembaggio!"]  # Italian duplicate removed
    """
    if not values:
        return []

    # Check if langdetect is available
    if detect is None:
        logger.warning("langdetect not available, falling back to simple deduplication")
        return deduplicate_simple_array_field(values, [])

    # If no embedding model provided, fall back to simple deduplication
    if embedding_model is None:
        logger.debug(
            "No embedding model provided for language-aware deduplication, using simple deduplication"
        )
        return deduplicate_simple_array_field(values, [])

    # Step 1: Simple whitespace normalization
    normalized_values = [v.strip() for v in values if v and v.strip()]

    if not normalized_values:
        return []

    # Step 2: Group by language
    language_groups: dict[str, list[str]] = defaultdict(list)
    unknown_language: list[str] = []

    for value in normalized_values:
        try:
            lang = detect(value)
            language_groups[lang].append(value)
        except (LangDetectException, Exception) as e:
            # If language detection fails, keep the value in unknown group
            logger.debug(f"Language detection failed for '{value}': {e}")
            unknown_language.append(value)

    # Step 3: Apply semantic deduplication within each language group
    result_values: list[str] = []

    for lang, lang_values in language_groups.items():
        if len(lang_values) == 1:
            # Single value in this language, no deduplication needed
            result_values.extend(lang_values)
            continue

        # Apply semantic deduplication within this language group
        deduped_lang_values: list[str] = []
        deduped_embeddings: list[list[float]] = []

        for value in lang_values:
            try:
                # Generate embedding using the injected model
                embeddings = embedding_model.encode([value])
                if not embeddings or embeddings[0] is None:
                    # Fallback: if embedding fails, check for exact match
                    if value not in deduped_lang_values:
                        deduped_lang_values.append(value)
                    continue

                embedding = embeddings[0]

                # Check semantic similarity against existing embeddings in this language
                if not _is_semantically_duplicate(
                    embedding, deduped_embeddings, similarity_threshold
                ):
                    deduped_lang_values.append(value)
                    deduped_embeddings.append(embedding)
                else:
                    logger.debug(
                        f"Removing duplicate within {lang}: '{value}'"
                    )

            except Exception as e:
                logger.warning(
                    f"Error processing value '{value[:50]}' in language '{lang}': {e}"
                )
                # Fallback: add if not already present
                if value not in deduped_lang_values:
                    deduped_lang_values.append(value)

        result_values.extend(deduped_lang_values)

    # Step 4: Add unknown language values (keep as-is, no deduplication)
    # These are typically special characters, numbers, or mixed-language strings
    result_values.extend(unknown_language)

    return result_values
