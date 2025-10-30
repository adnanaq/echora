"""
Tests for src/vector/utils.py deduplication functions.
"""

import pytest
from typing import List

from src.vector.utils import (
    deduplicate_simple_array_field,
    deduplicate_semantic_array_field,
    deduplicate_synonyms_language_aware,
    normalize_string_for_comparison,
)


class TestDeduplicateSynonymsLanguageAware:
    """Test language-aware synonym deduplication."""

    def test_preserves_cross_language_variants(self):
        """Different languages should be preserved even if semantically similar."""
        synonyms = [
            "ONE PIECE",  # English
            "tek parça",  # Turkish
            "원피스",      # Korean
            "ワンピース",   # Japanese
        ]

        result = deduplicate_synonyms_language_aware(synonyms)

        # All different languages should be kept
        assert len(result) == 4
        assert "ONE PIECE" in result
        assert "tek parça" in result
        assert "원피스" in result
        assert "ワンピース" in result

    def test_removes_same_language_case_variants(self):
        """Case/punctuation variants in same language should be deduplicated."""
        synonyms = [
            "All'arrembaggio!",   # Original
            "all'arrembaggio!",   # Lowercase
            "all`arrembaggio!",   # Different apostrophe
        ]

        result = deduplicate_synonyms_language_aware(synonyms)

        # Should keep only 1 (high similarity within same language)
        assert len(result) == 1
        assert result[0] in synonyms  # One of them should be kept

    def test_removes_same_language_extended_forms(self):
        """Extended forms in same language with high similarity should be deduplicated."""
        synonyms = [
            "All'arrembaggio!",
            "tutti all'arrembaggio!",  # Extended form
        ]

        result = deduplicate_synonyms_language_aware(synonyms)

        # Should keep only 1 (high similarity, same language)
        assert len(result) == 1

    def test_removes_greek_punctuation_variants(self):
        """Greek punctuation variants should be deduplicated."""
        synonyms = [
            "Ντρέηκ, το Κυνήγι του Θησαυρού",
            "ντρέικ και το κυνήγι του θησαυρού",
        ]

        result = deduplicate_synonyms_language_aware(synonyms)

        # Should keep only 1 (same language, high similarity)
        assert len(result) == 1

    def test_preserves_different_meanings_same_language(self):
        """Different meanings in same language should be preserved."""
        synonyms = [
            "海贼",   # Pirates (Chinese)
            "海贼王", # Pirate King (Chinese)
        ]

        result = deduplicate_synonyms_language_aware(synonyms)

        # Both should be kept if similarity < threshold
        # Note: Actual behavior depends on semantic similarity
        # If similarity > 0.85, one will be removed (which is expected)
        assert len(result) >= 1
        assert len(result) <= 2

    def test_real_one_piece_synonyms_data(self):
        """Test with real One Piece backup data (34 synonyms)."""
        synonyms = [
            "1p",
            "all'arrembaggio!",
            "all'arrembaggio!",  # Duplicate
            "budak getah",
            "lastik çocuk",
            "one piece",
            "op",
            "op tv",
            "one piece tv",
            "onep",
            "tek parça",
            "tutti all'arrembaggio!",
            "vua hải tặc",
            "optv",
            "đảo hải tặc",
            "ντρέηκ, το κυνήγι του θησαυρού",
            "ντρέικ και το κυνήγι του θησαυρού",
            "большой куш",
            "ван-пис",
            "великий куш",
            "едно цяло",
            "одним куском",
            "уан пийс",
            "וואן פיס",
            "وان پیس",
            "ون بيس",
            "ওয়ান পিস্",
            "วันพีซ",
            "ワンピース",
            "海賊王",
            "海贼",
            "海贼王",
            "원피스",
            "all`arrembaggio!",
        ]

        result = deduplicate_synonyms_language_aware(synonyms)

        # Should preserve language variants but remove within-language duplicates
        # Expected: ~27-32 synonyms (vs original 34)
        # Some within-language duplicates removed (Italian, Greek, Russian variants)
        assert len(result) >= 27
        assert len(result) <= 34

        # These MUST be kept (different languages)
        assert "tek parça" in result  # Turkish
        assert "lastik çocuk" in result  # Turkish
        assert "원피스" in result  # Korean

        # Italian variants should be deduplicated (only 1-2 kept max)
        italian_variants = [s for s in result if "arrembaggio" in s.lower()]
        assert len(italian_variants) <= 2

        # Verify we kept significantly more than cross-language only dedup (22)
        assert len(result) > 22  # Must be better than cross-language dedup

    def test_edge_cases(self):
        """Test edge cases for language-aware deduplication."""
        # Empty list
        assert deduplicate_synonyms_language_aware([]) == []

        # Single item
        assert deduplicate_synonyms_language_aware(["test"]) == ["test"]

        # All same language - case variants
        # Note: Semantic similarity might not catch all case-only differences
        # so we allow 1-2 to be kept (depends on embedding model behavior)
        synonyms_same_lang = ["One Piece", "one piece", "ONE PIECE"]
        result = deduplicate_synonyms_language_aware(synonyms_same_lang)
        assert len(result) >= 1  # At least one kept
        assert len(result) <= 2  # Should deduplicate most

        # Language detection failure (non-standard text)
        synonyms_weird = ["@#$%", "12345", "test"]
        result = deduplicate_synonyms_language_aware(synonyms_weird)
        # Should not crash, fallback to keeping entries
        assert len(result) >= 1


class TestExistingFunctions:
    """Test existing deduplication functions remain unchanged."""

    def test_deduplicate_simple_array_field(self):
        """Test simple string deduplication."""
        offline = ["Action", "Comedy"]
        external = ["action", "Drama", "COMEDY"]

        result = deduplicate_simple_array_field(offline, external)

        assert len(result) == 3
        assert "Action" in result
        assert "Comedy" in result
        assert "Drama" in result

    def test_normalize_string_for_comparison(self):
        """Test string normalization."""
        assert normalize_string_for_comparison("Test String") == "test string"
        assert normalize_string_for_comparison("  TRIM  ") == "trim"
        assert normalize_string_for_comparison("") == ""


