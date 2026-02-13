"""Pure reranking compute engine with async support."""

import asyncio
import logging
from typing import TypeVar

from common.config import EmbeddingConfig
from vector_processing.reranking.base import RerankerModel

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RerankerProcessor:
    """Pure reranking compute engine.

    Handles concurrent reranking operations with semaphore-based
    concurrency control. Follows the same async pattern as
    TextProcessor and VisionProcessor.
    """

    def __init__(self, model: RerankerModel, config: EmbeddingConfig):
        """Initialize reranker processor.

        Args:
            model: Reranker model instance.
            config: Embedding configuration (for concurrency settings).
        """
        self.model = model
        self.config = config
        self._semaphore = asyncio.Semaphore(config.embed_max_concurrency)
        logger.info(
            "Initialized reranker processor: %s (max_concurrency=%d)",
            model.model_name,
            config.embed_max_concurrency,
        )

    async def rerank(
        self,
        query: str,
        documents: list[tuple[T, str]],
        top_k: int | None = None,
    ) -> list[tuple[T, float]]:
        """Rerank documents by relevance to query.

        Args:
            query: Search query text.
            documents: List of (item, text) tuples where item is the
                original object and text is the content to score.
            top_k: Optional limit on returned results (returns all if None).

        Returns:
            List of (item, score) tuples sorted by relevance (descending).
            Scores are from the reranker model (typically 0-1 for BGE).
        """
        if not documents:
            return []

        # Extract items and texts
        items, texts = zip(*documents)

        # Build query-document pairs
        pairs = [[query, text] for text in texts]

        # Score pairs (offload to thread pool for sync model)
        async with self._semaphore:
            scores = await asyncio.to_thread(self.model.predict, pairs)

        # Zip items with scores and sort
        ranked = sorted(
            zip(items, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        # Apply top_k filter if specified
        if top_k is not None and top_k > 0:
            ranked = ranked[:top_k]

        logger.debug(
            "Reranked %d documents, returning top %d",
            len(documents),
            len(ranked),
        )

        return ranked
