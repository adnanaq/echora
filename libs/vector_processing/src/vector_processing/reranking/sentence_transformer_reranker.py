"""Sentence Transformers CrossEncoder implementation for reranking."""

import logging

from sentence_transformers import CrossEncoder

from .base import RerankerModel

logger = logging.getLogger(__name__)


class SentenceTransformerReranker(RerankerModel):
    """Sentence Transformers CrossEncoder implementation.

    Uses pre-trained cross-encoder models from HuggingFace to score
    query-document relevance. Supports models like:
    - BAAI/bge-reranker-v2-m3 (multilingual, recommended)
    - BAAI/bge-reranker-base
    - cross-encoder/ms-marco-MiniLM-L-6-v2
    """

    def __init__(
        self,
        model_name: str,
        cache_dir: str | None = None,
        max_length: int = 512,
    ):
        """Initialize sentence transformer reranker.

        Args:
            model_name: HuggingFace model identifier.
            cache_dir: Optional cache directory for model files.
            max_length: Maximum sequence length (default: 512).
        """
        logger.info("Loading reranker model: %s", model_name)
        self.model = CrossEncoder(
            model_name,
            max_length=max_length,
            cache_folder=cache_dir,
        )
        self._model_name = model_name
        self._max_length = max_length
        logger.info("Reranker model loaded successfully")

    def predict(self, pairs: list[list[str]]) -> list[float]:
        """Score query-document pairs using cross-encoder.

        Args:
            pairs: List of [query, document] pairs to score.

        Returns:
            List of relevance scores. BGE reranker returns normalized
            scores in [0, 1] range where higher = more relevant.
        """
        if not pairs:
            return []

        scores = self.model.predict(pairs, show_progress_bar=False)
        return scores.tolist()

    @property
    def model_name(self) -> str:
        """Get model identifier."""
        return self._model_name

    @property
    def max_length(self) -> int:
        """Get maximum sequence length."""
        return self._max_length
