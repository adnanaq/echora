"""FastEmbed text embedding model implementation."""

import logging
import re
from typing import Any, cast

from .base import TextEmbeddingModel

logger = logging.getLogger(__name__)

_DEFAULT_EMBEDDING_SIZE = 384
_DEFAULT_MAX_LENGTH = 512
_TOKEN_LIMIT_PATTERN = re.compile(r"(\d+)\s+input tokens", re.IGNORECASE)
_FALLBACK_MODEL_DIMENSIONS: dict[str, int] = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "BAAI/bge-m3": 1024,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
    "intfloat/e5-small-v2": 384,
    "intfloat/e5-base-v2": 768,
    "intfloat/e5-large-v2": 1024,
}


class FastEmbedModel(TextEmbeddingModel):
    """FastEmbed implementation of TextEmbeddingModel."""

    def __init__(self, model_name: str, cache_dir: str | None = None, **kwargs):
        """Initialize FastEmbed model.

        Args:
            model_name: FastEmbed model name.
            cache_dir: Optional directory to cache model files.
            **kwargs: Additional arguments passed to TextEmbedding.
        """
        try:
            from fastembed import TextEmbedding

            self._model_name = model_name

            init_kwargs: dict[str, Any] = {"model_name": model_name}
            if cache_dir:
                init_kwargs["cache_dir"] = cache_dir
            init_kwargs.update(kwargs)

            self.model = TextEmbedding(**init_kwargs)
            model_metadata = self._find_model_metadata(TextEmbedding, model_name)
            self._embedding_size = self._resolve_embedding_size(
                model_name, model_metadata
            )
            self._max_length = self._resolve_max_length(model_metadata)
            self._supports_multilingual = self._resolve_multilingual_support(
                model_name, model_metadata
            )

            logger.info(f"Initialized FastEmbed model: {model_name}")

        except ImportError as e:
            logger.exception(
                "FastEmbed not installed. Install with: pip install fastembed"
            )
            raise ImportError("FastEmbed dependencies missing") from e

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
        return self._supports_multilingual

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode a list of texts into embeddings.

        Args:
            texts: List of text strings to encode

        Returns:
            List of embedding vectors
        """
        try:
            # FastEmbed returns a generator, convert to list
            embeddings = list(self.model.embed(texts))
            return [cast(list[float], e.tolist()) for e in embeddings]

        except Exception:
            logger.exception("FastEmbed encoding failed")
            raise

    @staticmethod
    def _find_model_metadata(
        text_embedding_cls: Any, model_name: str
    ) -> dict[str, Any] | None:
        """Find FastEmbed registry metadata for the requested model name.

        Args:
            text_embedding_cls: FastEmbed ``TextEmbedding`` class.
            model_name: FastEmbed model identifier.

        Returns:
            Matching model metadata dictionary, or None when unavailable.
        """
        list_models = getattr(text_embedding_cls, "list_supported_models", None)
        if not callable(list_models):
            return None
        supported_models = list_models()
        if not isinstance(supported_models, list):
            return None
        for entry in supported_models:
            if isinstance(entry, dict) and entry.get("model") == model_name:
                return entry
        return None

    def _resolve_embedding_size(
        self, model_name: str, model_metadata: dict[str, Any] | None
    ) -> int:
        """Resolve the embedding dimension from the loaded model or FastEmbed metadata.

        Args:
            model_name: FastEmbed model identifier.
            model_metadata: Optional metadata dictionary from FastEmbed's registry.

        Returns:
            Embedding vector dimensionality for the model.
        """
        loaded_size = getattr(self.model, "embedding_size", None)
        if (
            isinstance(loaded_size, int)
            and not isinstance(loaded_size, bool)
            and loaded_size > 0
        ):
            return loaded_size
        if model_metadata is not None:
            registry_dim = model_metadata.get("dim")
            if isinstance(registry_dim, int) and registry_dim > 0:
                return registry_dim
        return _FALLBACK_MODEL_DIMENSIONS.get(model_name, _DEFAULT_EMBEDDING_SIZE)

    def _resolve_max_length(self, model_metadata: dict[str, Any] | None) -> int:
        """Resolve the effective input token limit from the tokenizer or registry metadata.

        Args:
            model_metadata: Optional metadata dictionary from FastEmbed's registry.

        Returns:
            Maximum input token count supported by the model.
        """
        backend_model = getattr(self.model, "model", None)
        tokenizer = getattr(backend_model, "tokenizer", None)
        tokenizer_limit = getattr(tokenizer, "model_max_length", None)
        if (
            isinstance(tokenizer_limit, int)
            and not isinstance(tokenizer_limit, bool)
            and 0 < tokenizer_limit <= 100_000
        ):
            return tokenizer_limit
        if model_metadata is not None:
            description = str(model_metadata.get("description", ""))
            if match := _TOKEN_LIMIT_PATTERN.search(description):
                return int(match.group(1))
        return _DEFAULT_MAX_LENGTH

    @staticmethod
    def _resolve_multilingual_support(
        model_name: str, model_metadata: dict[str, Any] | None
    ) -> bool:
        """Resolve whether the model supports multilingual text.

        Args:
            model_name: FastEmbed model identifier.
            model_metadata: Optional metadata dictionary from FastEmbed's registry.

        Returns:
            True when registry metadata or model identifier indicates multilingual support.
        """
        if model_metadata is not None:
            description = str(model_metadata.get("description", "")).lower()
            if "multilingual" in description or "multi-lingual" in description:
                return True
            if description:
                return False
        lower_name = model_name.lower()
        return "multilingual" in lower_name or "m3" in lower_name
