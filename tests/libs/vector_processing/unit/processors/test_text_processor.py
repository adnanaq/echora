"""Unit tests for TextProcessor.

Tests cover all code paths including initialization, encoding,
batch processing, embedding cache integration, and edge cases.
"""

import hashlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from vector_processing.cache import EmbeddingCache
from vector_processing.processors.text_processor import TextProcessor

# Fixtures mock_text_model and mock_settings are provided by conftest.py


class TestTextProcessorInit:
    """Tests for TextProcessor initialization."""

    def test_init_with_settings(self, mock_text_model, mock_settings):
        """Test initialization with provided settings."""
        processor = TextProcessor(model=mock_text_model, config=mock_settings)

        assert processor.model == mock_text_model
        assert processor.config == mock_settings

    def test_init_without_settings_uses_defaults(self, mock_text_model):
        """Test initialization without config uses default EmbeddingConfig."""
        with patch(
            "vector_processing.processors.text_processor.EmbeddingConfig"
        ) as mock_config_class:
            mock_default_config = MagicMock()
            mock_default_config.embed_max_concurrency = 2
            mock_config_class.return_value = mock_default_config

            processor = TextProcessor(model=mock_text_model)

            mock_config_class.assert_called_once()
            assert processor.config == mock_default_config

    def test_init_logs_model_name(self, mock_text_model, mock_settings, caplog):
        """Test that initialization logs the model name."""
        with caplog.at_level("INFO"):
            TextProcessor(model=mock_text_model, config=mock_settings)

        assert "Initialized TextProcessor with model: test-text-model" in caplog.text


class TestEncodeText:
    """Tests for encode_text method."""

    @pytest.mark.asyncio
    async def test_encode_text_success(self, mock_text_model, mock_settings):
        """Test successful text encoding."""
        processor = TextProcessor(model=mock_text_model, config=mock_settings)

        result = await processor.encode_text("Hello world")

        assert result == [0.1] * 1024
        mock_text_model.encode.assert_called_once_with(["Hello world"])

    @pytest.mark.asyncio
    async def test_encode_text_empty_string_returns_zero_embedding(
        self, mock_text_model, mock_settings
    ):
        """Test empty string returns zero embedding."""
        processor = TextProcessor(model=mock_text_model, config=mock_settings)

        result = await processor.encode_text("")

        assert result == [0.0] * 1024
        mock_text_model.encode.assert_not_called()

    @pytest.mark.asyncio
    async def test_encode_text_whitespace_only_returns_zero_embedding(
        self, mock_text_model, mock_settings
    ):
        """Test whitespace-only string returns zero embedding."""
        processor = TextProcessor(model=mock_text_model, config=mock_settings)

        result = await processor.encode_text("   \t\n  ")

        assert result == [0.0] * 1024
        mock_text_model.encode.assert_not_called()

    @pytest.mark.asyncio
    async def test_encode_text_model_returns_empty_list(
        self, mock_text_model, mock_settings
    ):
        """Test when model returns empty list."""
        mock_text_model.encode.return_value = []
        processor = TextProcessor(model=mock_text_model, config=mock_settings)

        result = await processor.encode_text("Hello")

        assert result is None

    @pytest.mark.asyncio
    async def test_encode_text_model_raises_exception(
        self, mock_text_model, mock_settings, caplog
    ):
        """Test when model raises exception."""
        mock_text_model.encode.side_effect = RuntimeError("Model error")
        processor = TextProcessor(model=mock_text_model, config=mock_settings)

        with caplog.at_level("ERROR"):
            result = await processor.encode_text("Hello")

        assert result is None
        assert "Text encoding failed" in caplog.text


class TestEncodeTextsBatch:
    """Tests for encode_texts_batch method."""

    @pytest.mark.asyncio
    async def test_encode_texts_batch_success(self, mock_text_model, mock_settings):
        """Test successful batch encoding."""
        mock_text_model.encode.return_value = [
            [0.1] * 1024,
            [0.2] * 1024,
            [0.3] * 1024,
        ]
        processor = TextProcessor(model=mock_text_model, config=mock_settings)

        result = await processor.encode_texts_batch(["text1", "text2", "text3"])

        assert len(result) == 3
        assert result[0] == [0.1] * 1024
        assert result[1] == [0.2] * 1024
        assert result[2] == [0.3] * 1024
        mock_text_model.encode.assert_called_once_with(["text1", "text2", "text3"])

    @pytest.mark.asyncio
    async def test_encode_texts_batch_empty_list(self, mock_text_model, mock_settings):
        """Test batch encoding with empty list."""
        mock_text_model.encode.return_value = []
        processor = TextProcessor(model=mock_text_model, config=mock_settings)

        result = await processor.encode_texts_batch([])

        assert result == []

    @pytest.mark.asyncio
    async def test_encode_texts_batch_model_raises_exception(
        self, mock_text_model, mock_settings, caplog
    ):
        """Test when model raises exception during batch encoding."""
        mock_text_model.encode.side_effect = RuntimeError("Batch error")
        processor = TextProcessor(model=mock_text_model, config=mock_settings)

        with caplog.at_level("ERROR"):
            result = await processor.encode_texts_batch(["text1", "text2"])

        assert result == [None, None]
        assert "Batch text encoding failed" in caplog.text

    @pytest.mark.asyncio
    async def test_encode_texts_batch_with_empty_strings(
        self, mock_text_model, mock_settings
    ):
        """Test batch encoding filters out empty strings and returns zero vectors."""
        # Model should only receive non-empty texts
        mock_text_model.encode.return_value = [
            [0.1] * 1024,  # For "valid text"
        ]
        processor = TextProcessor(model=mock_text_model, config=mock_settings)

        result = await processor.encode_texts_batch(["", "valid text", ""])

        assert len(result) == 3
        assert result[0] == [0.0] * 1024  # Empty string -> zero vector
        assert result[1] == [0.1] * 1024  # Valid text -> encoded
        assert result[2] == [0.0] * 1024  # Empty string -> zero vector
        # Model should only be called with valid text
        mock_text_model.encode.assert_called_once_with(["valid text"])

    @pytest.mark.asyncio
    async def test_encode_texts_batch_with_whitespace_strings(
        self, mock_text_model, mock_settings
    ):
        """Test batch encoding filters out whitespace-only strings."""
        mock_text_model.encode.return_value = [
            [0.2] * 1024,  # For "real content"
        ]
        processor = TextProcessor(model=mock_text_model, config=mock_settings)

        result = await processor.encode_texts_batch(["   \t\n  ", "real content", "  "])

        assert len(result) == 3
        assert result[0] == [0.0] * 1024  # Whitespace -> zero vector
        assert result[1] == [0.2] * 1024  # Valid text -> encoded
        assert result[2] == [0.0] * 1024  # Whitespace -> zero vector
        mock_text_model.encode.assert_called_once_with(["real content"])

    @pytest.mark.asyncio
    async def test_encode_texts_batch_all_empty(self, mock_text_model, mock_settings):
        """Test batch encoding when all inputs are empty/whitespace."""
        processor = TextProcessor(model=mock_text_model, config=mock_settings)

        result = await processor.encode_texts_batch(["", "   ", "\t\n"])

        assert len(result) == 3
        assert all(vec == [0.0] * 1024 for vec in result)
        # Model should not be called at all
        mock_text_model.encode.assert_not_called()

    @pytest.mark.asyncio
    async def test_encode_texts_batch_mixed_content(
        self, mock_text_model, mock_settings
    ):
        """Test batch encoding with realistic mixed content."""
        mock_text_model.encode.return_value = [
            [0.1] * 1024,  # "Action anime"
            [0.2] * 1024,  # "Character development"
            [0.3] * 1024,  # "Epic finale"
        ]
        processor = TextProcessor(model=mock_text_model, config=mock_settings)

        result = await processor.encode_texts_batch(
            [
                "Action anime",
                "",
                "Character development",
                "  \t  ",
                "Epic finale",
            ]
        )

        assert len(result) == 5
        assert result[0] == [0.1] * 1024
        assert result[1] == [0.0] * 1024
        assert result[2] == [0.2] * 1024
        assert result[3] == [0.0] * 1024
        assert result[4] == [0.3] * 1024
        mock_text_model.encode.assert_called_once_with(
            [
                "Action anime",
                "Character development",
                "Epic finale",
            ]
        )

    @pytest.mark.asyncio
    async def test_zero_vectors_independent(
        self, mock_text_model, mock_settings
    ):
        """Test that zero vectors are independent copies, not the same object.

        Regression test for issue where all empty inputs shared the same
        zero vector instance, causing mutations to affect all empty embeddings.
        """
        mock_text_model.encode.return_value = [[0.1] * 1024]
        processor = TextProcessor(model=mock_text_model, config=mock_settings)

        result = await processor.encode_texts_batch(["", "valid", "", ""])

        # All empty positions should have zero vectors
        assert result[0] == [0.0] * 1024
        assert result[2] == [0.0] * 1024
        assert result[3] == [0.0] * 1024

        # Critical: They must be DIFFERENT objects
        assert id(result[0]) != id(result[2])
        assert id(result[0]) != id(result[3])
        assert id(result[2]) != id(result[3])

        # Verify mutation isolation (narrow types — these are known zero vectors)
        assert result[0] is not None
        assert result[2] is not None
        assert result[3] is not None
        result[0][0] = 999.0
        assert result[2][0] == 0.0  # Should NOT be affected
        assert result[3][0] == 0.0  # Should NOT be affected

    @pytest.mark.asyncio
    async def test_all_empty_batch_produces_independent_vectors(
        self, mock_text_model, mock_settings
    ):
        """Test all-empty case produces independent zero vectors.

        Regression test for the same issue in the all-empty code path.
        """
        processor = TextProcessor(model=mock_text_model, config=mock_settings)

        result = await processor.encode_texts_batch(["", "", "", ""])

        # All should be zero vectors
        assert all(vec == [0.0] * 1024 for vec in result)

        # Critical: Each must be a different object
        assert id(result[0]) != id(result[1])
        assert id(result[0]) != id(result[2])
        assert id(result[0]) != id(result[3])

        # Verify mutation isolation (narrow types — these are known zero vectors)
        assert result[0] is not None
        assert result[1] is not None
        assert result[2] is not None
        assert result[3] is not None
        result[0][0] = 999.0
        assert result[1][0] == 0.0
        assert result[2][0] == 0.0
        assert result[3][0] == 0.0

    @pytest.mark.asyncio
    async def test_encode_texts_batch_large_batch_correctness(
        self, mock_text_model, mock_settings
    ):
        """Test correctness with large batch (regression test for O(n^2) complexity).

        While we can't easily test performance in a unit test, we verify that
        the set-based lookup produces correct results with large batches.
        """
        # Create large batch with alternating pattern
        size = 1000
        texts = ["valid" if i % 2 == 0 else "" for i in range(size)]

        # Mock should return embeddings for 500 valid texts
        mock_text_model.encode.return_value = [
            [float(i)] * 1024 for i in range(size // 2)
        ]
        processor = TextProcessor(model=mock_text_model, config=mock_settings)

        result = await processor.encode_texts_batch(texts)

        # Verify length
        assert len(result) == size

        # Verify alternating pattern: even indices have embeddings, odd have zeros
        for i in range(size):
            vec = result[i]
            assert vec is not None
            if i % 2 == 0:  # Valid text
                assert vec[0] == float(i // 2)
            else:  # Empty string
                assert vec == [0.0] * 1024


class TestGetZeroEmbedding:
    """Tests for get_zero_embedding method."""

    def test_get_zero_embedding_correct_size(self, mock_text_model, mock_settings):
        """Test zero embedding has correct dimensions."""
        mock_text_model.embedding_size = 512
        processor = TextProcessor(model=mock_text_model, config=mock_settings)

        result = processor.get_zero_embedding()

        assert len(result) == 512
        assert all(x == 0.0 for x in result)

    def test_get_zero_embedding_different_sizes(self, mock_text_model, mock_settings):
        """Test zero embedding works for different model sizes."""
        for size in [256, 768, 1024, 1536]:
            mock_text_model.embedding_size = size
            processor = TextProcessor(model=mock_text_model, config=mock_settings)

            result = processor.get_zero_embedding()

            assert len(result) == size


class TestGetModelInfo:
    """Tests for get_model_info method."""

    def test_get_model_info_returns_model_info(self, mock_text_model, mock_settings):
        """Test get_model_info delegates to model."""
        expected_info = {
            "model_name": "test-model",
            "embedding_size": 1024,
            "provider": "test",
        }
        mock_text_model.get_model_info.return_value = expected_info
        processor = TextProcessor(model=mock_text_model, config=mock_settings)

        result = processor.get_model_info()

        assert result == expected_info
        mock_text_model.get_model_info.assert_called_once()


# --- Embedding Cache Integration Tests ---


@pytest.fixture
def mock_embedding_cache():
    """Create a mock EmbeddingCache for unit tests."""
    cache = AsyncMock(spec=EmbeddingCache)
    cache.get.return_value = None
    cache.set.return_value = None
    cache.get_batch.return_value = []
    cache.set_batch.return_value = None
    return cache


class TestEncodeTextWithCache:
    """Tests for encode_text with embedding cache."""

    @pytest.mark.asyncio
    async def test_cache_hit_skips_model_inference(
        self, mock_text_model, mock_settings, mock_embedding_cache
    ):
        """Test that a cache hit returns the cached embedding without calling the model."""
        cached_embedding = [0.5] * 1024
        mock_embedding_cache.get.return_value = cached_embedding

        processor = TextProcessor(
            model=mock_text_model,
            config=mock_settings,
            embedding_cache=mock_embedding_cache,
        )

        result = await processor.encode_text("Hello world")

        assert result == cached_embedding
        mock_text_model.encode.assert_not_called()
        mock_embedding_cache.get.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_cache_miss_runs_model_and_stores(
        self, mock_text_model, mock_settings, mock_embedding_cache
    ):
        """Test that a cache miss runs inference and writes the result back."""
        mock_embedding_cache.get.return_value = None

        processor = TextProcessor(
            model=mock_text_model,
            config=mock_settings,
            embedding_cache=mock_embedding_cache,
        )

        result = await processor.encode_text("Hello world")

        assert result == [0.1] * 1024
        mock_text_model.encode.assert_called_once_with(["Hello world"])
        # Should write back to cache
        text_hash = hashlib.sha256(b"Hello world").hexdigest()
        mock_embedding_cache.set.assert_awaited_once_with(
            "test-text-model", text_hash, [0.1] * 1024
        )

    @pytest.mark.asyncio
    async def test_empty_text_bypasses_cache(
        self, mock_text_model, mock_settings, mock_embedding_cache
    ):
        """Test that empty text returns zero vector without checking cache."""
        processor = TextProcessor(
            model=mock_text_model,
            config=mock_settings,
            embedding_cache=mock_embedding_cache,
        )

        result = await processor.encode_text("")

        assert result == [0.0] * 1024
        mock_embedding_cache.get.assert_not_awaited()
        mock_embedding_cache.set.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_cache_works_as_before(self, mock_text_model, mock_settings):
        """Test that cache=None path works identically to the original code."""
        processor = TextProcessor(model=mock_text_model, config=mock_settings)

        result = await processor.encode_text("Hello world")

        assert result == [0.1] * 1024
        mock_text_model.encode.assert_called_once()


class TestEncodeTextsBatchWithCache:
    """Tests for encode_texts_batch with embedding cache."""

    @pytest.mark.asyncio
    async def test_all_cache_hits_skips_model(
        self, mock_text_model, mock_settings, mock_embedding_cache
    ):
        """Test that all cache hits means no model inference at all."""
        cached = [[0.5] * 1024, [0.6] * 1024]
        mock_embedding_cache.get_batch.return_value = cached

        processor = TextProcessor(
            model=mock_text_model,
            config=mock_settings,
            embedding_cache=mock_embedding_cache,
        )

        result = await processor.encode_texts_batch(["text1", "text2"])

        assert result == cached
        mock_text_model.encode.assert_not_called()
        mock_embedding_cache.set_batch.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_partial_cache_hits_encodes_only_misses(
        self, mock_text_model, mock_settings, mock_embedding_cache
    ):
        """Test that only uncached texts are sent to the model."""
        cached_emb = [0.5] * 1024
        mock_embedding_cache.get_batch.return_value = [cached_emb, None, None]
        mock_text_model.encode.return_value = [[0.2] * 1024, [0.3] * 1024]

        processor = TextProcessor(
            model=mock_text_model,
            config=mock_settings,
            embedding_cache=mock_embedding_cache,
        )

        result = await processor.encode_texts_batch(["cached", "miss1", "miss2"])

        assert len(result) == 3
        assert result[0] == cached_emb
        assert result[1] == [0.2] * 1024
        assert result[2] == [0.3] * 1024
        # Model should only encode the 2 uncached texts
        mock_text_model.encode.assert_called_once_with(["miss1", "miss2"])
        # Should write back the 2 new embeddings
        mock_embedding_cache.set_batch.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_all_cache_misses_encodes_all(
        self, mock_text_model, mock_settings, mock_embedding_cache
    ):
        """Test that all misses sends everything to the model."""
        mock_embedding_cache.get_batch.return_value = [None, None]
        mock_text_model.encode.return_value = [[0.1] * 1024, [0.2] * 1024]

        processor = TextProcessor(
            model=mock_text_model,
            config=mock_settings,
            embedding_cache=mock_embedding_cache,
        )

        result = await processor.encode_texts_batch(["text1", "text2"])

        assert len(result) == 2
        mock_text_model.encode.assert_called_once_with(["text1", "text2"])
        mock_embedding_cache.set_batch.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_mixed_empty_and_cached(
        self, mock_text_model, mock_settings, mock_embedding_cache
    ):
        """Test batch with empty strings and cache hits — no model call needed."""
        cached_emb = [0.5] * 1024
        mock_embedding_cache.get_batch.return_value = [cached_emb]

        processor = TextProcessor(
            model=mock_text_model,
            config=mock_settings,
            embedding_cache=mock_embedding_cache,
        )

        result = await processor.encode_texts_batch(["", "cached_text", "  "])

        assert len(result) == 3
        assert result[0] == [0.0] * 1024  # empty → zero vector
        assert result[1] == cached_emb  # cache hit
        assert result[2] == [0.0] * 1024  # whitespace → zero vector
        mock_text_model.encode.assert_not_called()
