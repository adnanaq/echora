"""Text embedding processor for converting text to vector embeddings.

This module provides the TextProcessor class, which serves as a pure compute
engine for text embedding. The processor converts strings into embedding
vectors using configured ML models with no domain-specific logic.
"""

import asyncio
import hashlib
import logging
import time
from typing import Any, cast

from common.config import EmbeddingConfig
from common.utils.request_batcher import RequestBatcher
from vector_db_interface import SparseVectorData

from ..cache import EmbeddingCache
from ..embedding_models.text.base import TextEmbeddingModel
from .inference_pipeline import (
    _embedding_cache_lookups,
    _embedding_duration,
    _tracer,
    fetch_cached_embeddings,
    run_cached_batch_inference,
    run_cached_single_inference,
)

logger = logging.getLogger(__name__)


class TextProcessor:
    """Pure text embedding processor with no domain-specific logic.

    This class serves as the text compute engine in the vector processing
    pipeline. The processor turns text into numbers with no knowledge of anime,
    characters, or any domain concepts.

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
        self._batcher: (
            RequestBatcher[str, tuple[list[float] | None, SparseVectorData | None]]
            | None
        ) = None
        if config.embed_batch_max_size > 1:
            self._batcher = RequestBatcher(
                self._encode_texts_as_pairs,
                max_batch_size=config.embed_batch_max_size,
                max_wait_seconds=config.embed_batch_max_wait_ms / 1000,
                concurrency=config.embed_max_concurrency,
                name="text_embedding",
            )

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
                _embedding_cache_lookups.add(1, {"result": "hit", "modality": "text"})
                return cached
            _embedding_cache_lookups.add(1, {"result": "miss", "modality": "text"})

        return await run_cached_single_inference(
            single_input=text,
            input_hash=text_hash,
            encode_fn=self.model.encode,
            cache=self._cache,
            model_name=self.model.model_name,
            semaphore=self._semaphore,
            span_name="vector_processing.text.encode",
            span_attributes={
                "embedding.model": self.model.model_name,
                "embedding.input_length": len(text),
                "embedding.sparse": False,
            },
            modality="text",
            failure_message="Text encoding failed",
        )

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
        zero_embedding = self.get_zero_embedding()
        valid_indices, valid_texts = self._filter_non_empty_texts(texts)

        if not valid_texts:
            return [zero_embedding.copy() for _ in texts]

        valid_hashes = [hashlib.sha256(t.encode()).hexdigest() for t in valid_texts]
        cached_embeddings, uncached_positions = await fetch_cached_embeddings(
            self._cache, self.model.model_name, valid_hashes, modality="text"
        )

        if uncached_positions:
            succeeded = await run_cached_batch_inference(
                inputs=valid_texts,
                input_hashes=valid_hashes,
                cached_embeddings=cached_embeddings,
                uncached_positions=uncached_positions,
                encode_fn=self.model.encode,
                cache=self._cache,
                model_name=self.model.model_name,
                semaphore=self._semaphore,
                span_name="vector_processing.text.encode_batch",
                span_attributes={
                    "embedding.model": self.model.model_name,
                    "embedding.batch_size": len(uncached_positions),
                    "embedding.cache_hits": len(valid_texts) - len(uncached_positions),
                    "embedding.sparse": False,
                },
                modality="text",
                failure_message="Batch text encoding failed",
            )
            if not succeeded:
                return [None] * len(texts)

        return self._align_embeddings_with_inputs(
            len(texts), valid_indices, cached_embeddings, zero_embedding
        )

    async def encode_text_with_sparse(
        self, text: str
    ) -> tuple[list[float] | None, SparseVectorData | None]:
        """Encode a single text, returning dense and sparse vectors.

        Uses one forward pass when the underlying model supports sparse output
        (``model.supports_sparse``). Falls back to dense-only for other models.
        With ``embed_batch_max_size`` above 1, concurrent calls are combined
        into shared model calls.

        Args:
            text: Input text to encode.

        Returns:
            Tuple of ``(dense, sparse)`` where ``sparse`` is ``None`` when the
            model does not produce sparse output.
        """
        if not text or not text.strip():
            return self.get_zero_embedding(), None

        try:
            with _tracer.start_as_current_span(
                "vector_processing.text.encode",
                attributes={
                    "embedding.model": self.model.model_name,
                    "embedding.input_length": len(text),
                    "embedding.sparse": True,
                    "embedding.batched": self._batcher is not None,
                },
            ):
                if self._batcher is not None:
                    return await self._batcher.submit(text)
                async with self._semaphore:
                    _start = time.perf_counter()
                    dense_list, sparse_list = await asyncio.to_thread(
                        self.model.encode_with_sparse, [text]
                    )
                    _embedding_duration.record(
                        time.perf_counter() - _start, {"modality": "text"}
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
            with _tracer.start_as_current_span(
                "vector_processing.text.encode_batch",
                attributes={
                    "embedding.model": self.model.model_name,
                    "embedding.batch_size": len(valid_texts),
                    "embedding.sparse": True,
                },
            ):
                async with self._semaphore:
                    _start = time.perf_counter()
                    encoded_dense, encoded_sparse = await asyncio.to_thread(
                        self.model.encode_with_sparse, valid_texts
                    )
                    _embedding_duration.record(
                        time.perf_counter() - _start, {"modality": "text"}
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

    async def close(self) -> None:
        """Stop the query batcher, if batching is on."""
        if self._batcher is not None:
            await self._batcher.close()

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

    @staticmethod
    def _filter_non_empty_texts(texts: list[str]) -> tuple[list[int], list[str]]:
        """Collect indices and values of non-empty, non-whitespace texts.

        Args:
            texts: Raw input strings.

        Returns:
            Tuple of ``(valid_indices, valid_texts)``.
        """
        valid_indices: list[int] = []
        valid_texts: list[str] = []
        for idx, text in enumerate(texts):
            if text and text.strip():
                valid_indices.append(idx)
                valid_texts.append(text)
        return valid_indices, valid_texts

    @staticmethod
    def _align_embeddings_with_inputs(
        total_count: int,
        valid_indices: list[int],
        valid_embeddings: list[list[float] | None],
        zero_embedding: list[float],
    ) -> list[list[float] | None]:
        """Reconstruct a batch result list, inserting zero vectors for blank inputs.

        Args:
            total_count: Total number of original input texts.
            valid_indices: Original indices corresponding to ``valid_embeddings``.
            valid_embeddings: Encoded embeddings for non-empty inputs.
            zero_embedding: Zero vector copied for blank inputs.

        Returns:
            List of embeddings aligned with the original input sequence.
        """
        aligned: list[list[float] | None] = []
        valid_cursor = 0
        valid_index_set = set(valid_indices)
        for idx in range(total_count):
            if idx in valid_index_set:
                aligned.append(valid_embeddings[valid_cursor])
                valid_cursor += 1
            else:
                aligned.append(zero_embedding.copy())
        return aligned

    async def _encode_texts_as_pairs(
        self, texts: list[str]
    ) -> list[tuple[list[float] | None, SparseVectorData | None]]:
        """Encode a batch and pair each text's dense and sparse vectors.

        Args:
            texts: Input text strings to encode.

        Returns:
            List of ``(dense_vector, sparse_vector)`` tuples aligned with ``texts``.
        """
        dense, sparse = await self.encode_texts_batch_with_sparse(texts)
        return list(zip(dense, sparse, strict=True))
