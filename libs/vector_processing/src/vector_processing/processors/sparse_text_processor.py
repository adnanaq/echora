"""Sparse text embedding processor for BM25-style lexical vectors.

This module provides a sparse embedding processor that produces explicit
``indices``/``values`` vectors suitable for Qdrant sparse vector fields.
"""

import asyncio
import logging
from typing import Any

from common.config import EmbeddingConfig
from vector_db_interface import SparseVectorData

logger = logging.getLogger(__name__)


class SparseTextProcessor:
    """Processor that generates sparse text embeddings for lexical retrieval."""

    def __init__(self, config: EmbeddingConfig | None = None):
        """Initialize sparse text processor.

        Args:
            config: Embedding configuration. Uses defaults when omitted.

        Raises:
            ValueError: If sparse provider configuration is unsupported.
            ImportError: If required sparse embedding dependencies are missing.
        """
        if config is None:
            config = EmbeddingConfig()
        self.config = config
        self._semaphore = asyncio.Semaphore(config.embed_max_concurrency)
        self._model = self._init_model()
        logger.info(
            "Initialized SparseTextProcessor with provider=%s model=%s",
            self.config.sparse_embedding_provider,
            self.config.sparse_embedding_model,
        )

    def _init_model(self) -> Any:
        """Initialize sparse embedding model backend.

        Returns:
            Initialized sparse embedding model instance.

        Raises:
            ValueError: If sparse embedding provider is unsupported.
            ImportError: If FastEmbed is not installed.
        """
        if self.config.sparse_embedding_provider != "fastembed":
            raise ValueError(
                "Only fastembed sparse provider is currently supported"
            )

        try:
            from fastembed import SparseTextEmbedding
        except ImportError as error:
            logger.exception(
                "FastEmbed sparse dependencies missing. Install with: pip install fastembed"
            )
            raise ImportError("FastEmbed sparse dependencies missing") from error

        init_kwargs: dict[str, Any] = {
            "model_name": self.config.sparse_embedding_model,
        }
        if self.config.model_cache_dir:
            init_kwargs["cache_dir"] = self.config.model_cache_dir

        return SparseTextEmbedding(**init_kwargs)

    def _normalize_sparse_embedding(self, embedding: Any) -> SparseVectorData:
        """Convert raw model sparse embedding object into typed payload.

        Args:
            embedding: Raw sparse embedding object from model backend.

        Returns:
            Sparse vector payload containing ``indices`` and ``values``.

        Raises:
            ValueError: If embedding does not expose sparse index/value vectors.
        """
        indices_raw = getattr(embedding, "indices", None)
        values_raw = getattr(embedding, "values", None)
        if indices_raw is None or values_raw is None:
            raise ValueError("Sparse embedding must expose indices and values")

        indices = indices_raw.tolist() if hasattr(indices_raw, "tolist") else indices_raw
        values = values_raw.tolist() if hasattr(values_raw, "tolist") else values_raw
        return {
            "indices": [int(index) for index in indices],
            "values": [float(value) for value in values],
        }

    async def encode_text(self, text: str) -> SparseVectorData | None:
        """Encode single text into sparse vector payload.

        Args:
            text: Input text to embed.

        Returns:
            Sparse embedding payload, or ``None`` on empty input or failures.
        """
        if not text or not text.strip():
            return None

        try:
            async with self._semaphore:
                embeddings = await asyncio.to_thread(lambda: list(self._model.embed([text])))
            if not embeddings:
                return None
            return self._normalize_sparse_embedding(embeddings[0])
        except Exception:
            logger.exception("Sparse text encoding failed")
            return None

    async def encode_texts_batch(
        self,
        texts: list[str],
    ) -> list[SparseVectorData | None]:
        """Encode a batch of texts into sparse vector payloads.

        Args:
            texts: Input texts to embed.

        Returns:
            Sparse embeddings aligned to input order. Entries are ``None`` for
            empty inputs or failed embeddings.
        """
        valid_indices: list[int] = []
        valid_texts: list[str] = []
        for idx, text in enumerate(texts):
            if text and text.strip():
                valid_indices.append(idx)
                valid_texts.append(text)

        if not valid_texts:
            return [None for _ in texts]

        try:
            async with self._semaphore:
                raw_embeddings = await asyncio.to_thread(
                    lambda: list(self._model.embed(valid_texts))
                )
        except Exception:
            logger.exception("Sparse batch text encoding failed")
            return [None for _ in texts]

        result: list[SparseVectorData | None] = [None for _ in texts]
        for idx, embedding in zip(valid_indices, raw_embeddings, strict=False):
            try:
                result[idx] = self._normalize_sparse_embedding(embedding)
            except Exception:
                logger.exception("Failed to normalize sparse embedding at index %s", idx)

        return result
