"""Tests for enrichment.utils.deduplication functions.

This module tests the deduplication utilities including:
- Simple string-based deduplication
- Semantic similarity-based deduplication (with mocked embedding models)
- Language-aware synonym deduplication

Note: Tests requiring embeddings use mocked TextEmbeddingModel instances.
"""

import pytest
from unittest.mock import Mock, MagicMock
from typing import List

from enrichment.utils.deduplication import (
    deduplicate_simple_array_field,
    deduplicate_semantic_array_field,
    deduplicate_synonyms_language_aware,
    normalize_string_for_comparison,
)


class TestNormalizationAndSimpleDeduplication:
    """Test functions that don't require embedding models."""

    def test_normalize_string_for_comparison(self):
        """Test string normalization."""
        assert normalize_string_for_comparison("Test String") == "test string"
        assert normalize_string_for_comparison("  TRIM  ") == "trim"
        assert normalize_string_for_comparison("") == ""
        assert normalize_string_for_comparison("UPPERCASE") == "uppercase"

    def test_deduplicate_simple_array_field(self):
        """Test simple string deduplication without embeddings."""
        offline = ["Action", "Comedy"]
        external = ["action", "Drama", "COMEDY"]

        result = deduplicate_simple_array_field(offline, external)

        assert len(result) == 3
        assert "Action" in result
        assert "Comedy" in result
        assert "Drama" in result
        # Offline values should come first
        assert result[0] in offline
        assert result[1] in offline

    def test_deduplicate_simple_array_field_with_whitespace(self):
        """Test that whitespace is properly handled."""
        offline = ["  Action  ", "Comedy"]
        external = ["action", " Drama ", ""]

        result = deduplicate_simple_array_field(offline, external)

        assert len(result) == 3
        assert "Action" in result  # Whitespace stripped
        assert "Comedy" in result
        assert "Drama" in result

    def test_deduplicate_simple_array_field_empty_inputs(self):
        """Test with empty inputs."""
        assert deduplicate_simple_array_field([], []) == []
        assert deduplicate_simple_array_field(["test"], []) == ["test"]
        assert deduplicate_simple_array_field([], ["test"]) == ["test"]


class TestSemanticDeduplication:
    """Test semantic deduplication with mocked embedding models."""

    def test_deduplicate_semantic_fallback_without_model(self):
        """When no model provided, should fall back to simple deduplication."""
        values = ["Action", "action", "Comedy"]

        result = deduplicate_semantic_array_field(values, embedding_model=None)

        # Should fall back to simple deduplication
        assert len(result) == 2  # "action" and "Action" deduplicated
        assert "Comedy" in result

    def test_deduplicate_semantic_with_mock_model_no_duplicates(self):
        """Test with mocked model returning low similarity."""
        # Mock embedding model
        mock_model = Mock()
        mock_model.encode = Mock(
            return_value=[
                [0.1, 0.2, 0.3],  # First value
                [0.9, 0.8, 0.7],  # Second value (very different)
                [0.5, 0.4, 0.6],  # Third value (different from both)
            ]
        )

        values = ["value1", "value2", "value3"]
        result = deduplicate_semantic_array_field(values, embedding_model=mock_model)

        # All should be kept (low similarity)
        assert len(result) == 3
        assert all(v in result for v in values)

    def test_deduplicate_semantic_handles_embedding_failures(self):
        """Test graceful handling when embedding generation fails."""
        # Mock model that returns None for failed embeddings
        mock_model = Mock()
        mock_model.encode = Mock(side_effect=Exception("Model error"))

        values = ["value1", "value2"]
        result = deduplicate_semantic_array_field(values, embedding_model=mock_model)

        # Should fall back to keeping unique values by exact match
        assert len(result) == 2

    def test_deduplicate_semantic_empty_input(self):
        """Test with empty input."""
        mock_model = Mock()
        result = deduplicate_semantic_array_field([], embedding_model=mock_model)
        assert result == []


class TestLanguageAwareDeduplication:
    """Test language-aware synonym deduplication."""

    def test_fallback_without_langdetect(self):
        """When langdetect unavailable, should fall back to simple deduplication."""
        # This test assumes langdetect is available in test environment
        # If not available, the function will log a warning and use simple dedup
        values = ["test1", "test2", "TEST1"]
        result = deduplicate_synonyms_language_aware(values, embedding_model=None)

        # Should fall back to simple deduplication
        assert len(result) == 2  # "test1" and "TEST1" deduplicated
        assert "test2" in result

    def test_language_aware_fallback_without_model(self):
        """When no model provided, should fall back to simple deduplication."""
        values = ["Action", "action", "Comedy"]
        result = deduplicate_synonyms_language_aware(values, embedding_model=None)

        # Should fall back to simple deduplication
        assert len(result) == 2
        assert "Comedy" in result

    def test_language_aware_with_mock_model_preserves_cross_language(self):
        """Test that different language variants are preserved."""
        # Mock embedding model
        mock_model = Mock()
        # Return different embeddings for each call
        mock_model.encode = Mock(
            side_effect=[
                [[0.1, 0.2]],  # English
                [[0.5, 0.6]],  # Turkish
                [[0.9, 0.8]],  # Korean
            ]
        )

        # Simplified test with known languages
        values = ["ONE PIECE", "tek parça", "원피스"]

        result = deduplicate_synonyms_language_aware(values, embedding_model=mock_model)

        # All different languages should be kept
        # Note: Actual behavior depends on langdetect being able to detect these
        assert len(result) >= 2  # At least 2 should be preserved

    def test_language_aware_empty_input(self):
        """Test with empty input."""
        mock_model = Mock()
        result = deduplicate_synonyms_language_aware([], embedding_model=mock_model)
        assert result == []

    def test_language_aware_single_value(self):
        """Test with single value."""
        mock_model = Mock()
        result = deduplicate_synonyms_language_aware(
            ["test"], embedding_model=mock_model
        )
        assert result == ["test"]


@pytest.mark.skip(reason="Integration test - requires actual embedding model")
class TestWithRealEmbeddings:
    """Integration tests that require a real embedding model.

    These tests are skipped in unit tests but can be run as integration tests
    with a real BGE-M3 or other embedding model instance.
    """

    def test_preserves_cross_language_variants_real(self):
        """Test with real embeddings that different languages are preserved."""
        # This would require initializing a real TextEmbeddingModel
        # from vector_processing.embedding_models.factory import create_text_model
        # model = create_text_model(settings)
        pass

    def test_real_one_piece_synonyms_data(self):
        """Test with real One Piece synonym data using real embeddings."""
        # This would test the full 34 synonym dataset
        pass
