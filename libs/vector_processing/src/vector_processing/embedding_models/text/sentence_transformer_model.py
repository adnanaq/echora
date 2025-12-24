import logging
from typing import Any, Dict, List, Optional, cast

from .base import TextEmbeddingModel

logger = logging.getLogger(__name__)


class SentenceTransformerModel(TextEmbeddingModel):
    """Sentence Transformers implementation of TextEmbeddingModel."""

    def __init__(self, model_name: str, cache_dir: Optional[str] = None):
        """Initialize Sentence Transformers model.

        Args:
            model_name: Sentence Transformers model name
            cache_dir: Optional directory to cache model files
        """
        try:
            from sentence_transformers import SentenceTransformer

            self._model_name = model_name
            
            # Load model
            self.model = SentenceTransformer(model_name, cache_folder=cache_dir)

            # Get model info
            self._embedding_size = cast(int, self.model.get_sentence_embedding_dimension())
            self._max_length = self.model.max_seq_length
            
            logger.info(f"Initialized Sentence Transformers model: {model_name}")

        except ImportError as e:
            logger.error(
                "Sentence Transformers not installed. Install with: pip install sentence-transformers"
            )
            raise ImportError("Sentence Transformers dependencies missing") from e

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode a list of texts into embeddings.

        Args:
            texts: List of text strings to encode

        Returns:
            List of embedding vectors
        """
        try:
            # Generate embeddings
            # sentence-transformers returns numpy array by default
            embeddings = self.model.encode(texts)
            
            # Convert to list of lists
            return cast(List[List[float]], embeddings.tolist())

        except Exception as e:
            logger.error(f"Sentence Transformers encoding failed: {e}")
            raise

    @property
    def embedding_size(self) -> int:
        return self._embedding_size

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def supports_multilingual(self) -> bool:
        multilingual_indicators = [
            "multilingual",
            "m3",
            "xlm",
            "xlm-roberta",
            "mbert",
            "distilbert-base-multilingual",
            "jina-embeddings-v2-base-multilingual",
        ]
        model_lower = self._model_name.lower()
        return any(indicator in model_lower for indicator in multilingual_indicators)
