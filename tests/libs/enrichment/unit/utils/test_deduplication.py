"""Tests for enrichment.utils.deduplication functions.

This module tests the deduplication utilities including:
- Simple string-based deduplication
- Semantic similarity-based deduplication (with mocked embedding models)
- Language-aware synonym deduplication

Note: Tests requiring embeddings use mocked TextEmbeddingModel instances.
"""

from unittest.mock import Mock

import pytest
from enrichment.utils.deduplication import (
    deduplicate_semantic_array_field,
    deduplicate_simple_array_field,
    deduplicate_synonyms_language_aware,
    normalize_string_for_comparison,
)

try:
    from langdetect import DetectorFactory

    # Set seed for deterministic language detection in tests
    DetectorFactory.seed = 0
except ImportError:
    DetectorFactory = None


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
        # Side effect to return different embeddings for each call
        # Using orthogonal vectors to ensure zero similarity
        mock_model.encode = Mock(
            side_effect=[
                [[1.0, 0.0, 0.0]],  # First value
                [[0.0, 1.0, 0.0]],  # Second value
                [[0.0, 0.0, 1.0]],  # Third value
            ]
        )

        values = ["value1", "value2", "value3"]
        result = deduplicate_semantic_array_field(values, embedding_model=mock_model)

        # All should be kept (low similarity)
        assert len(result) == 3
        assert all(v in result for v in values)

    def test_deduplicate_semantic_handles_embedding_failures(self):
        """Test graceful handling when embedding generation fails."""
        # Mock model that raises RuntimeError (common ML library exception)
        mock_model = Mock()
        mock_model.encode = Mock(side_effect=RuntimeError("CUDA out of memory"))

        values = ["value1", "value2"]
        result = deduplicate_semantic_array_field(values, embedding_model=mock_model)

        # Should fall back to keeping unique values by normalized comparison
        assert len(result) == 2

    def test_deduplicate_semantic_fallback_uses_normalized_comparison(self):
        """Test that embedding failure fallback uses case-insensitive deduplication."""
        # Mock model that raises RuntimeError (common ML library exception)
        mock_model = Mock()
        mock_model.encode = Mock(side_effect=RuntimeError("Model execution failed"))

        # Test case variations that should be deduplicated
        values = ["Action", "action", "Comedy", "COMEDY"]
        result = deduplicate_semantic_array_field(values, embedding_model=mock_model)

        # Should deduplicate using normalized comparison (case-insensitive)
        assert len(result) == 2
        # First occurrence of each unique value (normalized) should be kept
        assert "Action" in result
        assert "Comedy" in result
        # Case variations should be removed
        assert "action" not in result
        assert "COMEDY" not in result

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
        """Test that different language variants are preserved.

        Uses real langdetect with seed=0 for deterministic results.
        DetectorFactory.seed is set at module import time.
        """
        # Mock embedding model - return orthogonal embeddings for each language
        mock_model = Mock()
        # Each language gets a distinct embedding (orthogonal vectors)
        mock_model.encode = Mock(
            side_effect=[
                [[1.0, 0.0, 0.0]],  # English: "ONE PIECE"
                [[0.0, 1.0, 0.0]],  # Turkish: "tek parça"
                [[0.0, 0.0, 1.0]],  # Korean: "원피스"
            ]
        )

        # Test with known language variants (seed=0 ensures deterministic detection)
        values = ["ONE PIECE", "tek parça", "원피스"]

        result = deduplicate_synonyms_language_aware(values, embedding_model=mock_model)

        # All 3 different languages should be preserved (no cross-language deduplication)
        assert len(result) == 3
        assert "ONE PIECE" in result
        assert "tek parça" in result
        assert "원피스" in result

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

    def test_language_aware_uses_batch_encoding(self):
        """Test that batch encoding is used for efficiency within each language group.

        The function should encode all values in a language group at once
        rather than making N separate encode() calls for N values.
        """
        from unittest.mock import patch

        # Mock embedding model that returns unique embeddings
        mock_model = Mock()
        # Return a batch of orthogonal embeddings (zero cosine similarity)
        mock_model.encode = Mock(
            return_value=[
                [1.0, 0.0, 0.0],  # First English value
                [0.0, 1.0, 0.0],  # Second English value (orthogonal to first)
                [0.0, 0.0, 1.0],  # Third English value (orthogonal to both)
            ]
        )

        # All English words that should be batch-encoded together
        values = ["action movie", "adventure film", "thriller series"]

        # Mock langdetect to ensure all values are detected as same language
        with patch("enrichment.utils.deduplication.detect", return_value="en"):
            result = deduplicate_synonyms_language_aware(
                values, embedding_model=mock_model
            )

        # Verify encode was called once with the full batch, not 3 times with single values
        assert mock_model.encode.call_count == 1
        # Verify it was called with all values at once
        call_args = mock_model.encode.call_args[0][0]
        assert len(call_args) == 3
        assert all(v in call_args for v in values)

        # All values should be preserved (orthogonal embeddings)
        assert len(result) == 3

    def test_language_aware_deduplicates_unknown_language(self):
        """Test that unknown_language values are deduplicated by normalized string.

        Values that fail language detection should still be deduplicated
        using simple string normalization to avoid duplicate "unknown" entries.
        """
        # Mock embedding model (won't be used for unknown language values)
        mock_model = Mock()
        mock_model.encode = Mock(return_value=[[1.0, 0.0, 0.0]])

        # Use values that will fail language detection (special chars, numbers)
        # and include exact duplicates and case variations
        values = ["123", "123", "???", "???", "##$", "##$", "456"]

        result = deduplicate_synonyms_language_aware(values, embedding_model=mock_model)

        # Should deduplicate the unknown language values
        # "123" appears twice -> keep one
        # "???" appears twice -> keep one
        # "##$" appears twice -> keep one
        # "456" appears once -> keep it
        assert len(result) == 4
        assert "123" in result
        assert "???" in result
        assert "##$" in result
        assert "456" in result

        # Count occurrences - each unique value should appear exactly once
        assert result.count("123") == 1
        assert result.count("???") == 1
        assert result.count("##$") == 1


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
