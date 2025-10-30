import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List
from collections import defaultdict
import logging

try:
    from langdetect import detect, LangDetectException
except ImportError:
    detect = None
    LangDetectException = Exception

from src.vector.embedding_models.model_singleton import get_text_embedding, normalize_japanese_text

SEMANTIC_SIMILARITY_THRESHOLD = 0.85 # This threshold needs tuning

logger = logging.getLogger(__name__)

__all__ = [
    "normalize_string_for_comparison",
    "deduplicate_simple_array_field",
    "deduplicate_semantic_array_field",
    "deduplicate_synonyms_language_aware",
    "SEMANTIC_SIMILARITY_THRESHOLD",
]

def normalize_string_for_comparison(text: str) -> str:
    """Normalize string for case-insensitive comparison."""
    if not text:
        return ""
    return text.lower().strip()

def deduplicate_simple_array_field(
    offline_values: List[str], external_values: List[str]
) -> List[str]:
    """
    Deduplicate array field values with offline database as foundation using simple string comparison.

    Args:
        offline_values: Values from offline database
        external_values: Values from external sources

    Returns:
        Deduplicated list with offline values first, then unique external values
    """
    result = []
    seen = set()

    # Add offline values first
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
    values: List[str]
) -> List[str]:
    """
    Deduplicate array field values using semantic similarity for cross-language support.

    Args:
        values: List of string values to deduplicate.

    Returns:
        Deduplicated list of string values.
    """
    result_values = []
    result_embeddings = []

    # Helper to check for semantic duplication
    def is_semantically_duplicate(new_embedding: List[float], existing_embeddings: List[List[float]]) -> bool:
        if not existing_embeddings:
            return False
        similarities = cosine_similarity([new_embedding], existing_embeddings)[0]
        return np.max(similarities) > SEMANTIC_SIMILARITY_THRESHOLD

    for value in values:
        if not value or not value.strip():
            continue
        # Normalize and get embedding
        normalized_value = normalize_japanese_text(value)
        embedding = get_text_embedding(normalized_value)

        if embedding is None:
            # Fallback to simple string deduplication if embedding fails
            # This fallback needs to be robust. For now, we'll just add if not already present by exact match
            if value.strip() not in result_values:
                result_values.append(value.strip())
            continue

        if not is_semantically_duplicate(embedding, result_embeddings):
            result_values.append(value.strip())
            result_embeddings.append(embedding)

    return result_values


def deduplicate_synonyms_language_aware(
    values: List[str],
    similarity_threshold: float = 0.85
) -> List[str]:
    """
    Deduplicate synonyms preserving all language variants while deduplicating within each language.

    This function groups synonyms by detected language and applies semantic similarity
    deduplication only within each language group. This ensures that cross-language
    variants (e.g., "ONE PIECE" in English, "tek parça" in Turkish, "원피스" in Korean)
    are all preserved, while removing duplicates like case/punctuation variants within
    the same language.

    Args:
        values: List of synonym strings to deduplicate
        similarity_threshold: Semantic similarity threshold for within-language deduplication

    Returns:
        Deduplicated list preserving cross-language variants

    Example:
        >>> synonyms = ["ONE PIECE", "tek parça", "All'arrembaggio!", "all'arrembaggio!"]
        >>> deduplicate_synonyms_language_aware(synonyms)
        ["ONE PIECE", "tek parça", "All'arrembaggio!"]  # Italian duplicate removed
    """
    if not values:
        return []

    # Check if langdetect is available
    if detect is None:
        logger.warning("langdetect not available, falling back to simple deduplication")
        return deduplicate_simple_array_field(values, [])

    # Step 1: Simple whitespace normalization
    normalized_values = [v.strip() for v in values if v and v.strip()]

    if not normalized_values:
        return []

    # Step 2: Group by language
    language_groups: Dict[str, List[str]] = defaultdict(list)
    unknown_language: List[str] = []

    for value in normalized_values:
        try:
            lang = detect(value)
            language_groups[lang].append(value)
        except (LangDetectException, Exception) as e:
            # If language detection fails, keep the value in unknown group
            logger.debug(f"Language detection failed for '{value}': {e}")
            unknown_language.append(value)

    # Step 3: Apply semantic deduplication within each language group
    result_values: List[str] = []

    for lang, lang_values in language_groups.items():
        if len(lang_values) == 1:
            # Single value in this language, no deduplication needed
            result_values.extend(lang_values)
            continue

        # Apply semantic deduplication within this language group
        deduped_lang_values: List[str] = []
        deduped_embeddings: List[List[float]] = []

        for value in lang_values:
            # Normalize and get embedding
            normalized_value = normalize_japanese_text(value)
            embedding = get_text_embedding(normalized_value)

            if embedding is None:
                # Fallback: if embedding fails, check for exact match
                if value not in deduped_lang_values:
                    deduped_lang_values.append(value)
                continue

            # Check semantic similarity against existing embeddings in this language
            is_duplicate = False
            if deduped_embeddings:
                similarities = cosine_similarity([embedding], deduped_embeddings)[0]
                max_sim = np.max(similarities)
                if max_sim > similarity_threshold:
                    is_duplicate = True
                    logger.debug(f"Removing duplicate within {lang}: '{value}' (similarity: {max_sim:.3f})")

            if not is_duplicate:
                deduped_lang_values.append(value)
                deduped_embeddings.append(embedding)

        result_values.extend(deduped_lang_values)

    # Step 4: Add unknown language values (keep as-is, no deduplication)
    # These are typically special characters, numbers, or mixed-language strings
    result_values.extend(unknown_language)

    return result_values
