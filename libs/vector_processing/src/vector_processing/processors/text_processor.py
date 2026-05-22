"""Text embedding processor for converting text to vector embeddings.

This module provides the TextProcessor class which serves as a pure compute
engine for text embedding. It is strictly responsible for converting strings
to embedding vectors using configured ML models, with no domain-specific logic.
"""

import asyncio
import hashlib
import logging
import time
from typing import Any, cast

from common.config import EmbeddingConfig
from opentelemetry import metrics as otel_metrics
from opentelemetry import trace as otel_trace
from vector_db_interface import SparseVectorData

from ..cache import EmbeddingCache
from ..embedding_models.text.base import TextEmbeddingModel

logger = logging.getLogger(__name__)

_tracer = otel_trace.get_tracer("echora.vector_processing")
_meter = otel_metrics.get_meter("echora.vector_processing")
_embedding_duration = _meter.create_histogram(
    "echora_embedding_duration_seconds",
    unit="s",
    description="Embedding model inference duration in seconds",
)
_embedding_cache_lookups = _meter.create_counter(
    "echora_embedding_cache_total",
    description="Embedding cache lookups by result (hit/miss) and modality",
)


class TextProcessor:
    """Pure text embedding processor with no domain-specific logic.

    This class serves as the "Text Compute Engine" in the vector processing
    pipeline. It knows HOW to turn text into numbers but has no knowledge
    of anime, characters, or any domain concepts.

    Responsibilities:
        - Interface with TextEmbeddingModels (FastEmbed, BGE, etc.).
        - Handle batching and list processing for text.
        - Provide zero-vectors for empty content.
    """

    def __init__(
        self,
        model: TextEmbeddingModel,
        config: EmbeddingConfig | None = None,
        embedding_cache: EmbeddingCache | None = None,
    ):
        """Initialize the text processor with an embedding model.

        Args:
            model: An initialized TextEmbeddingModel instance.
            config: Embedding configuration instance. Uses defaults if None.
            embedding_cache: Optional Redis-backed embedding cache.
                When provided, embeddings are cached by content hash to avoid
                redundant model inference. When None, every call runs inference.
        """
        if config is None:
            config = EmbeddingConfig()

        self.config = config
        self.model = model
        self._semaphore = asyncio.Semaphore(config.embed_max_concurrency)
        self._cache = embedding_cache

        logger.info(f"Initialized TextProcessor with model: {model.model_name}")

    async def encode_text(self, text: str) -> list[float] | None:
        """Encode a single string to an embedding vector.

        Runs the model inference in a thread to avoid blocking the event loop,
        with semaphore-based concurrency control. When an embedding cache is
        configured, results are looked up by content hash before inference
        and stored after a successful encode.

        Args:
            text: The text string to encode.

        Returns:
            A list of floats representing the embedding vector, a zero vector
            if the input is empty/whitespace, or None if encoding fails.
        """
        if not text or not text.strip():
            return self.get_zero_embedding()

        text_hash = hashlib.sha256(text.encode()).hexdigest()

        # Fast path: check cache BEFORE acquiring the inference semaphore
        if self._cache is not None:
            cached = await self._cache.get(self.model.model_name, text_hash)
            if cached is not None:
                _embedding_cache_lookups.add(
                    1, {"result": "hit", "modality": "text"}
                )
                return cached
            _embedding_cache_lookups.add(
                1, {"result": "miss", "modality": "text"}
            )

        with _tracer.start_as_current_span(
            "vector_processing.text.encode",
            attributes={
                "embedding.model": self.model.model_name,
                "embedding.input_length": len(text),
            },
        ):
            try:
                async with self._semaphore:
                    _start = time.perf_counter()
                    embeddings = await asyncio.to_thread(self.model.encode, [text])
                    _embedding_duration.record(
                        time.perf_counter() - _start, {"modality": "text"}
                    )
            except Exception:
                logger.exception("Text encoding failed")
                return None
            else:
                if not embeddings:
                    return None
                result = embeddings[0]
                if self._cache is not None:
                    await self._cache.set(self.model.model_name, text_hash, result)
                return result

    async def encode_texts_batch(self, texts: list[str]) -> list[list[float] | None]:
        """Encode multiple texts in a single batch call.

        More efficient than calling encode_text repeatedly as it leverages
        the model's batch processing capabilities. Pre-processing runs on
        the event loop; the model call runs in a thread with semaphore control.

        When an embedding cache is configured, cached results are fetched via
        MGET first — only uncached texts are sent to the model. Results are
        written back via batch set.

        Args:
            texts: List of text strings to encode.

        Returns:
            A list of embedding vectors (or None for failed encodings),
            in the same order as the input texts. Empty or whitespace-only
            strings return zero vectors to maintain consistency with encode_text.
        """
        # Pre-process texts to identify empty/whitespace entries
        zero_embedding = self.get_zero_embedding()
        valid_indices: list[int] = []
        valid_texts: list[str] = []

        for i, text in enumerate(texts):
            if text and text.strip():
                valid_indices.append(i)
                valid_texts.append(text)

        # If all texts are empty, return independent zero vectors for all
        if not valid_texts:
            return [zero_embedding.copy() for _ in texts]

        # Compute hashes for valid texts and check cache
        valid_hashes = [hashlib.sha256(t.encode()).hexdigest() for t in valid_texts]
        cached_embeddings: list[list[float] | None] = [None] * len(valid_texts)
        uncached_positions: list[int] = []  # indices into valid_texts

        if self._cache is not None:
            cached_embeddings = await self._cache.get_batch(
                self.model.model_name, valid_hashes
            )
            uncached_positions = [
                i for i, emb in enumerate(cached_embeddings) if emb is None
            ]
            cache_hits = len(valid_texts) - len(uncached_positions)
            if cache_hits:
                _embedding_cache_lookups.add(
                    cache_hits, {"result": "hit", "modality": "text"}
                )
            if uncached_positions:
                _embedding_cache_lookups.add(
                    len(uncached_positions), {"result": "miss", "modality": "text"}
                )
        else:
            uncached_positions = list(range(len(valid_texts)))

        # Encode only uncached texts
        if uncached_positions:
            uncached_texts = [valid_texts[i] for i in uncached_positions]

            with _tracer.start_as_current_span(
                "vector_processing.text.encode_batch",
                attributes={
                    "embedding.model": self.model.model_name,
                    "embedding.batch_size": len(uncached_texts),
                    "embedding.cache_hits": len(valid_texts) - len(uncached_positions),
                },
            ):
                try:
                    async with self._semaphore:
                        _start = time.perf_counter()
                        encoded_new = await asyncio.to_thread(
                            self.model.encode, uncached_texts
                        )
                        _embedding_duration.record(
                            time.perf_counter() - _start, {"modality": "text"}
                        )
                except Exception:
                    logger.exception("Batch text encoding failed")
                    return [None] * len(texts)

                # Merge newly encoded into cached_embeddings and prepare cache writes
                to_cache: dict[str, list[float]] = {}
                for j, pos in enumerate(uncached_positions):
                    embedding = cast(list[float] | None, encoded_new[j])
                    cached_embeddings[pos] = embedding
                    if embedding is not None:
                        to_cache[valid_hashes[pos]] = embedding

                if self._cache is not None and to_cache:
                    await self._cache.set_batch(self.model.model_name, to_cache)

        # Reconstruct result list with zero vectors for empty inputs
        result: list[list[float] | None] = []
        valid_idx = 0
        valid_index_set = set(valid_indices)

        for i in range(len(texts)):
            if i in valid_index_set:
                result.append(cached_embeddings[valid_idx])
                valid_idx += 1
            else:
                result.append(zero_embedding.copy())

        return result

    async def encode_text_with_sparse(
        self, text: str
    ) -> tuple[list[float] | None, SparseVectorData | None]:
        """Encode a single text, returning dense and sparse vectors.

        Uses one forward pass when the underlying model supports sparse output
        (``model.supports_sparse``). Falls back to dense-only for other models.

        Args:
            text: Input text to encode.

        Returns:
            Tuple of ``(dense, sparse)`` where ``sparse`` is ``None`` when the
            model does not produce sparse output.
        """
        if not text or not text.strip():
            return self.get_zero_embedding(), None

        try:
            async with self._semaphore:
                dense_list, sparse_list = await asyncio.to_thread(
                    self.model.encode_with_sparse, [text]
                )
        except Exception:
            logger.exception("Text encoding with sparse failed")
            return None, None
        else:
            dense = dense_list[0] if dense_list else None
            sparse = sparse_list[0] if sparse_list else None
            return dense, sparse

    async def encode_texts_batch_with_sparse(
        self,
        texts: list[str],
    ) -> tuple[list[list[float] | None], list[SparseVectorData | None]]:
        """Encode a batch of texts, returning aligned dense and sparse vectors.

        Uses one forward pass per batch when the model supports sparse output.

        Args:
            texts: Input texts.

        Returns:
            Tuple of ``(dense_list, sparse_list)`` both aligned to input order.
            Entries for empty texts are zero-vector / ``None``.
        """
        zero_embedding = self.get_zero_embedding()
        valid_indices: list[int] = []
        valid_texts: list[str] = []

        for i, text in enumerate(texts):
            if text and text.strip():
                valid_indices.append(i)
                valid_texts.append(text)

        if not valid_texts:
            return (
                [zero_embedding.copy() for _ in texts],
                [None] * len(texts),
            )

        try:
            async with self._semaphore:
                encoded_dense, encoded_sparse = await asyncio.to_thread(
                    self.model.encode_with_sparse, valid_texts
                )
        except Exception:
            logger.exception("Batch text encoding with sparse failed")
            return [None] * len(texts), [None] * len(texts)

        dense_result: list[list[float] | None] = []
        sparse_result: list[SparseVectorData | None] = []
        valid_idx = 0
        valid_index_set = set(valid_indices)

        for i in range(len(texts)):
            if i in valid_index_set:
                dense_result.append(cast(list[float] | None, encoded_dense[valid_idx]))
                sparse_result.append(encoded_sparse[valid_idx])
                valid_idx += 1
            else:
                dense_result.append(zero_embedding.copy())
                sparse_result.append(None)

        return dense_result, sparse_result

    def get_zero_embedding(self) -> list[float]:
        """Get a zero embedding vector matching model dimensions.

        Returns:
            A list of zeros with length equal to the model's embedding size.
        """
        return [0.0] * self.model.embedding_size

    def get_model_info(self) -> dict[str, Any]:
        """Get metadata about the underlying embedding model.

        Returns:
            A dictionary containing model information such as name,
            embedding size, and other model-specific details.
        """
        return self.model.get_model_info()
