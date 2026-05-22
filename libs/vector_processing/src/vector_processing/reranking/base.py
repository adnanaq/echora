"""Abstract base class for reranking models."""

from abc import ABC, abstractmethod


class RerankerModel(ABC):
    """Abstract base for reranking models.

    Rerankers score query-document pairs to determine relevance,
    enabling more accurate ranking than vector similarity alone.
    """

    @abstractmethod
    def predict(self, pairs: list[list[str]]) -> list[float]:
        """Score query-document pairs.

        Args:
            pairs: List of [query, document] pairs to score.

        Returns:
            List of relevance scores (higher = more relevant).
            Scores should be comparable across calls.
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get model identifier."""
        pass

    @property
    def max_length(self) -> int:
        """Maximum sequence length for input pairs.

        Returns:
            Maximum tokens accepted (default: 512).
        """
        return 512
