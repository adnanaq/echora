"""Unit tests for TextProcessor.

Tests cover all code paths including initialization, encoding,
batch processing, embedding cache integration, and edge cases.
"""

import hashlib
from unittest.mock import create_autospec

import pytest
from common.config import EmbeddingConfig
from opentelemetry.sdk.metrics.export import HistogramDataPoint, InMemoryMetricReader
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from vector_processing.cache import EmbeddingCache
from vector_processing.processors.text_processor import TextProcessor

# Fixtures text_model and embedding_config are provided by conftest.py


class TestTextProcessorInit:
    """Tests for TextProcessor initialization."""

    def test_init_with_config_stores_model_and_config(
        self, text_model, embedding_config
    ):
        """Test initialization with provided settings."""
        processor = TextProcessor(model=text_model, config=embedding_config)

        assert processor.model == text_model
        assert processor.config == embedding_config

    def test_init_without_config_uses_default_config(self, text_model):
        """Test initialization without config uses default EmbeddingConfig."""
        processor = TextProcessor(model=text_model)

        assert processor.config == EmbeddingConfig()

    def test_init_logs_model_name(self, text_model, embedding_config, caplog):
        """Test that initialization logs the model name."""
        with caplog.at_level("INFO"):
            TextProcessor(model=text_model, config=embedding_config)

        assert "Initialized TextProcessor with model: test-text-model" in caplog.text


class TestEncodeText:
    """Tests for encode_text method."""

    @pytest.mark.asyncio
    async def test_encode_text_valid_text_returns_embedding(
        self, text_model, embedding_config
    ):
        """Test successful text encoding."""
        processor = TextProcessor(model=text_model, config=embedding_config)

        result = await processor.encode_text("Hello world")

        assert result == [0.1] * 1024
        text_model.encode.assert_called_once_with(["Hello world"])

    @pytest.mark.asyncio
    async def test_encode_text_empty_string_returns_zero_embedding(
        self, text_model, embedding_config
    ):
        """Test empty string returns zero embedding."""
        processor = TextProcessor(model=text_model, config=embedding_config)

        result = await processor.encode_text("")

        assert result == [0.0] * 1024
        text_model.encode.assert_not_called()

    @pytest.mark.asyncio
    async def test_encode_text_whitespace_only_returns_zero_embedding(
        self, text_model, embedding_config
    ):
        """Test whitespace-only string returns zero embedding."""
        processor = TextProcessor(model=text_model, config=embedding_config)

        result = await processor.encode_text("   \t\n  ")

        assert result == [0.0] * 1024
        text_model.encode.assert_not_called()

    @pytest.mark.asyncio
    async def test_encode_text_model_returns_empty_list_returns_none(
        self, text_model, embedding_config
    ):
        """Test when model returns empty list."""
        text_model.encode.return_value = []
        processor = TextProcessor(model=text_model, config=embedding_config)

        result = await processor.encode_text("Hello")

        assert result is None

    @pytest.mark.asyncio
    async def test_encode_text_model_raises_returns_none_and_logs_error(
        self, text_model, embedding_config, caplog
    ):
        """Test when model raises exception."""
        text_model.encode.side_effect = RuntimeError("Model error")
        processor = TextProcessor(model=text_model, config=embedding_config)

        with caplog.at_level("ERROR"):
            result = await processor.encode_text("Hello")

        assert result is None
        assert "Text encoding failed" in caplog.text


class TestEncodeTextsBatch:
    """Tests for encode_texts_batch method."""

    @pytest.mark.asyncio
    async def test_encode_texts_batch_valid_texts_returns_embeddings(
        self, text_model, embedding_config
    ):
        """Test successful batch encoding."""
        text_model.encode.return_value = [
            [0.1] * 1024,
            [0.2] * 1024,
            [0.3] * 1024,
        ]
        processor = TextProcessor(model=text_model, config=embedding_config)

        result = await processor.encode_texts_batch(["text1", "text2", "text3"])

        assert len(result) == 3
        assert result[0] == [0.1] * 1024
        assert result[1] == [0.2] * 1024
        assert result[2] == [0.3] * 1024
        text_model.encode.assert_called_once_with(["text1", "text2", "text3"])

    @pytest.mark.asyncio
    async def test_encode_texts_batch_empty_list_returns_empty_list(
        self, text_model, embedding_config
    ):
        """Test batch encoding with empty list."""
        text_model.encode.return_value = []
        processor = TextProcessor(model=text_model, config=embedding_config)

        result = await processor.encode_texts_batch([])

        assert result == []

    @pytest.mark.asyncio
    async def test_encode_texts_batch_model_raises_returns_none_per_text(
        self, text_model, embedding_config, caplog
    ):
        """Test when model raises exception during batch encoding."""
        text_model.encode.side_effect = RuntimeError("Batch error")
        processor = TextProcessor(model=text_model, config=embedding_config)

        with caplog.at_level("ERROR"):
            result = await processor.encode_texts_batch(["text1", "text2"])

        assert result == [None, None]
        assert "Batch text encoding failed" in caplog.text

    @pytest.mark.asyncio
    async def test_encode_texts_batch_empty_strings_returns_zero_vectors_for_them(
        self, text_model, embedding_config
    ):
        """Test batch encoding filters out empty strings and returns zero vectors."""
        # Model should only receive non-empty texts
        text_model.encode.return_value = [
            [0.1] * 1024,  # For "valid text"
        ]
        processor = TextProcessor(model=text_model, config=embedding_config)

        result = await processor.encode_texts_batch(["", "valid text", ""])

        assert len(result) == 3
        assert result[0] == [0.0] * 1024  # Empty string -> zero vector
        assert result[1] == [0.1] * 1024  # Valid text -> encoded
        assert result[2] == [0.0] * 1024  # Empty string -> zero vector
        # Model should only be called with valid text
        text_model.encode.assert_called_once_with(["valid text"])

    @pytest.mark.asyncio
    async def test_encode_texts_batch_whitespace_strings_returns_zero_vectors_for_them(
        self, text_model, embedding_config
    ):
        """Test batch encoding filters out whitespace-only strings."""
        text_model.encode.return_value = [
            [0.2] * 1024,  # For "real content"
        ]
        processor = TextProcessor(model=text_model, config=embedding_config)

        result = await processor.encode_texts_batch(["   \t\n  ", "real content", "  "])

        assert len(result) == 3
        assert result[0] == [0.0] * 1024  # Whitespace -> zero vector
        assert result[1] == [0.2] * 1024  # Valid text -> encoded
        assert result[2] == [0.0] * 1024  # Whitespace -> zero vector
        text_model.encode.assert_called_once_with(["real content"])

    @pytest.mark.asyncio
    async def test_encode_texts_batch_all_empty_returns_zero_vectors(
        self, text_model, embedding_config
    ):
        """Test batch encoding when all inputs are empty/whitespace."""
        processor = TextProcessor(model=text_model, config=embedding_config)

        result = await processor.encode_texts_batch(["", "   ", "\t\n"])

        assert len(result) == 3
        assert all(vec == [0.0] * 1024 for vec in result)
        # Model should not be called at all
        text_model.encode.assert_not_called()

    @pytest.mark.asyncio
    async def test_encode_texts_batch_mixed_content_keeps_input_order(
        self, text_model, embedding_config
    ):
        """Test batch encoding with realistic mixed content."""
        text_model.encode.return_value = [
            [0.1] * 1024,  # "Action anime"
            [0.2] * 1024,  # "Character development"
            [0.3] * 1024,  # "Epic finale"
        ]
        processor = TextProcessor(model=text_model, config=embedding_config)

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
        text_model.encode.assert_called_once_with(
            [
                "Action anime",
                "Character development",
                "Epic finale",
            ]
        )

    @pytest.mark.asyncio
    async def test_encode_texts_batch_some_empty_texts_returns_independent_zero_vectors(
        self, text_model, embedding_config
    ):
        """Test that zero vectors are independent copies, not the same object.

        Regression test for issue where all empty inputs shared the same
        zero vector instance, causing mutations to affect all empty embeddings.
        """
        text_model.encode.return_value = [[0.1] * 1024]
        processor = TextProcessor(model=text_model, config=embedding_config)

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
    async def test_encode_texts_batch_all_empty_returns_independent_zero_vectors(
        self, text_model, embedding_config
    ):
        """Test all-empty case produces independent zero vectors.

        Regression test for the same issue in the all-empty code path.
        """
        processor = TextProcessor(model=text_model, config=embedding_config)

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
    async def test_encode_texts_batch_large_batch_returns_embedding_per_text(
        self, text_model, embedding_config
    ):
        """Test correctness with large batch (regression test for O(n^2) complexity).

        While we can't easily test performance in a unit test, we verify that
        the set-based lookup produces correct results with large batches.
        """
        # Create large batch with alternating pattern
        size = 1000
        texts = ["valid" if i % 2 == 0 else "" for i in range(size)]

        # Mock should return embeddings for 500 valid texts
        text_model.encode.return_value = [[float(i)] * 1024 for i in range(size // 2)]
        processor = TextProcessor(model=text_model, config=embedding_config)

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


def _text_inference_duration_count(reader: InMemoryMetricReader) -> int:
    data = reader.get_metrics_data()
    if data is None:
        return 0
    return sum(
        point.count
        for resource_metrics in data.resource_metrics
        for scope_metrics in resource_metrics.scope_metrics
        for metric in scope_metrics.metrics
        if metric.name == "echora_embedding_duration_seconds"
        for point in metric.data.data_points
        if isinstance(point, HistogramDataPoint)
        and point.attributes.get("modality") == "text"
    )


class TestEmbeddingDurationMetric:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("method", "argument"),
        [
            ("encode_text", "Hello world"),
            ("encode_texts_batch", ["Hello", "world"]),
            ("encode_text_with_sparse", "Hello world"),
            ("encode_texts_batch_with_sparse", ["Hello", "world"]),
        ],
    )
    async def test_text_processor_each_encode_method_records_one_text_inference_duration(
        self,
        text_model,
        embedding_config,
        method,
        argument,
        metric_reader: InMemoryMetricReader,
    ):
        text_model.encode.return_value = [[0.1] * 1024, [0.2] * 1024]
        text_model.encode_with_sparse.return_value = (
            [[0.1] * 1024, [0.2] * 1024],
            [None, None],
        )
        processor = TextProcessor(model=text_model, config=embedding_config)
        recorded_before = _text_inference_duration_count(metric_reader)

        await getattr(processor, method)(argument)

        assert _text_inference_duration_count(metric_reader) == recorded_before + 1


class TestEncodingSpans:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("method", "argument", "span_name"),
        [
            ("encode_text", "Hello world", "vector_processing.text.encode"),
            ("encode_text_with_sparse", "Hello world", "vector_processing.text.encode"),
            (
                "encode_texts_batch",
                ["Hello", "world"],
                "vector_processing.text.encode_batch",
            ),
            (
                "encode_texts_batch_with_sparse",
                ["Hello", "world"],
                "vector_processing.text.encode_batch",
            ),
        ],
    )
    async def test_text_processor_each_encode_method_records_one_encoding_span(
        self,
        text_model,
        embedding_config,
        method,
        argument,
        span_name,
        span_exporter: InMemorySpanExporter,
    ):
        text_model.encode.return_value = [[0.1] * 1024, [0.2] * 1024]
        text_model.encode_with_sparse.return_value = (
            [[0.1] * 1024, [0.2] * 1024],
            [None, None],
        )
        processor = TextProcessor(model=text_model, config=embedding_config)
        span_exporter.clear()

        await getattr(processor, method)(argument)

        (span,) = span_exporter.get_finished_spans()
        assert span.name == span_name
        assert span.attributes["embedding.model"] == "test-text-model"
        assert span.attributes["embedding.sparse"] == method.endswith("_with_sparse")


class TestGetZeroEmbedding:
    """Tests for get_zero_embedding method."""

    def test_get_zero_embedding_model_size_returns_zeros_of_that_size(
        self, text_model, embedding_config
    ):
        """Test zero embedding has correct dimensions."""
        text_model.embedding_size = 512
        processor = TextProcessor(model=text_model, config=embedding_config)

        result = processor.get_zero_embedding()

        assert len(result) == 512
        assert all(x == 0.0 for x in result)

    def test_get_zero_embedding_any_model_size_returns_matching_length(
        self, text_model, embedding_config
    ):
        """Test zero embedding works for different model sizes."""
        for size in [256, 768, 1024, 1536]:
            text_model.embedding_size = size
            processor = TextProcessor(model=text_model, config=embedding_config)

            result = processor.get_zero_embedding()

            assert len(result) == size


class TestGetModelInfo:
    """Tests for get_model_info method."""

    def test_get_model_info_returns_model_info(self, text_model, embedding_config):
        """Test get_model_info delegates to model."""
        expected_info = {
            "model_name": "test-model",
            "embedding_size": 1024,
            "provider": "test",
        }
        text_model.get_model_info.return_value = expected_info
        processor = TextProcessor(model=text_model, config=embedding_config)

        result = processor.get_model_info()

        assert result == expected_info
        text_model.get_model_info.assert_called_once()


# --- Embedding Cache Integration Tests ---


@pytest.fixture
def embedding_cache():
    """Create a mock EmbeddingCache for unit tests."""
    cache = create_autospec(EmbeddingCache, instance=True)
    cache.get.return_value = None
    cache.set.return_value = None
    cache.get_batch.return_value = []
    cache.set_batch.return_value = None
    return cache


class TestEncodeTextWithCache:
    """Tests for encode_text with embedding cache."""

    @pytest.mark.asyncio
    async def test_encode_text_cache_hit_skips_model(
        self, text_model, embedding_config, embedding_cache
    ):
        """Test that a cache hit returns the cached embedding without calling the model."""
        cached_embedding = [0.5] * 1024
        embedding_cache.get.return_value = cached_embedding

        processor = TextProcessor(
            model=text_model,
            config=embedding_config,
            embedding_cache=embedding_cache,
        )

        result = await processor.encode_text("Hello world")

        assert result == cached_embedding
        text_model.encode.assert_not_called()
        embedding_cache.get.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_encode_text_cache_miss_runs_model_and_stores_result(
        self, text_model, embedding_config, embedding_cache
    ):
        """Test that a cache miss runs inference and writes the result back."""
        embedding_cache.get.return_value = None

        processor = TextProcessor(
            model=text_model,
            config=embedding_config,
            embedding_cache=embedding_cache,
        )

        result = await processor.encode_text("Hello world")

        assert result == [0.1] * 1024
        text_model.encode.assert_called_once_with(["Hello world"])
        # Should write back to cache
        text_hash = hashlib.sha256(b"Hello world").hexdigest()
        embedding_cache.set.assert_awaited_once_with(
            "test-text-model", text_hash, [0.1] * 1024
        )

    @pytest.mark.asyncio
    async def test_encode_text_empty_text_bypasses_cache(
        self, text_model, embedding_config, embedding_cache
    ):
        """Test that empty text returns zero vector without checking cache."""
        processor = TextProcessor(
            model=text_model,
            config=embedding_config,
            embedding_cache=embedding_cache,
        )

        result = await processor.encode_text("")

        assert result == [0.0] * 1024
        embedding_cache.get.assert_not_awaited()
        embedding_cache.set.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_encode_text_without_cache_runs_model(
        self, text_model, embedding_config
    ):
        """Test that cache=None path works identically to the original code."""
        processor = TextProcessor(model=text_model, config=embedding_config)

        result = await processor.encode_text("Hello world")

        assert result == [0.1] * 1024
        text_model.encode.assert_called_once()


class TestEncodeTextsBatchWithCache:
    """Tests for encode_texts_batch with embedding cache."""

    @pytest.mark.asyncio
    async def test_encode_texts_batch_all_cache_hits_skips_model(
        self, text_model, embedding_config, embedding_cache
    ):
        """Test that all cache hits means no model inference at all."""
        cached = [[0.5] * 1024, [0.6] * 1024]
        embedding_cache.get_batch.return_value = cached

        processor = TextProcessor(
            model=text_model,
            config=embedding_config,
            embedding_cache=embedding_cache,
        )

        result = await processor.encode_texts_batch(["text1", "text2"])

        assert result == cached
        text_model.encode.assert_not_called()
        embedding_cache.set_batch.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_encode_texts_batch_partial_cache_hits_encodes_only_misses(
        self, text_model, embedding_config, embedding_cache
    ):
        """Test that only uncached texts are sent to the model."""
        cached_emb = [0.5] * 1024
        embedding_cache.get_batch.return_value = [cached_emb, None, None]
        text_model.encode.return_value = [[0.2] * 1024, [0.3] * 1024]

        processor = TextProcessor(
            model=text_model,
            config=embedding_config,
            embedding_cache=embedding_cache,
        )

        result = await processor.encode_texts_batch(["cached", "miss1", "miss2"])

        assert len(result) == 3
        assert result[0] == cached_emb
        assert result[1] == [0.2] * 1024
        assert result[2] == [0.3] * 1024
        # Model should only encode the 2 uncached texts
        text_model.encode.assert_called_once_with(["miss1", "miss2"])
        # Should write back the 2 new embeddings
        embedding_cache.set_batch.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_encode_texts_batch_all_cache_misses_encodes_every_text(
        self, text_model, embedding_config, embedding_cache
    ):
        """Test that all misses sends everything to the model."""
        embedding_cache.get_batch.return_value = [None, None]
        text_model.encode.return_value = [[0.1] * 1024, [0.2] * 1024]

        processor = TextProcessor(
            model=text_model,
            config=embedding_config,
            embedding_cache=embedding_cache,
        )

        result = await processor.encode_texts_batch(["text1", "text2"])

        assert len(result) == 2
        text_model.encode.assert_called_once_with(["text1", "text2"])
        embedding_cache.set_batch.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_encode_texts_batch_empty_and_cached_texts_returns_zero_and_cached_vectors(
        self, text_model, embedding_config, embedding_cache
    ):
        """Test batch with empty strings and cache hits — no model call needed."""
        cached_emb = [0.5] * 1024
        embedding_cache.get_batch.return_value = [cached_emb]

        processor = TextProcessor(
            model=text_model,
            config=embedding_config,
            embedding_cache=embedding_cache,
        )

        result = await processor.encode_texts_batch(["", "cached_text", "  "])

        assert len(result) == 3
        assert result[0] == [0.0] * 1024  # empty → zero vector
        assert result[1] == cached_emb  # cache hit
        assert result[2] == [0.0] * 1024  # whitespace → zero vector
        text_model.encode.assert_not_called()
