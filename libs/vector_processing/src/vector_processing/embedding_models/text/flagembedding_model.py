"""FlagEmbedding BGE-M3 backend producing dense and sparse vectors in one pass."""

import logging
from typing import Any, cast

import numpy as np

from vector_db_interface import SparseVectorData

from .base import TextEmbeddingModel

logger = logging.getLogger(__name__)

_BGE_M3_DENSE_DIM = 1024


class FlagEmbeddingModel(TextEmbeddingModel):
    """BGE-M3 backend via FlagEmbedding.

    Produces both dense (1024-dim) and sparse (lexical) vectors in a single
    forward pass. Dense uses CLS-token pooling (correct for BGE-M3);
    sparse weights are the model's learned lexical projection.
    """

    def __init__(
        self,
        model_name: str,
        cache_dir: str | None = None,
        max_length: int = 8192,
    ) -> None:
        """Initialize BGE-M3 via FlagEmbedding.

        Args:
            model_name: HuggingFace model identifier (e.g. ``BAAI/bge-m3``).
            cache_dir: Optional directory for downloaded model files.
            max_length: Maximum token sequence length for passage encoding.

        Raises:
            ImportError: If FlagEmbedding is not installed.
        """
        try:
            import torch
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:
            raise ImportError(
                "FlagEmbedding not installed. Install with: pip install FlagEmbedding"
            ) from exc

        self._model_name = model_name
        self._max_length = max_length
        use_fp16 = torch.cuda.is_available()

        self._model: BGEM3FlagModel = BGEM3FlagModel(
            model_name_or_path=model_name,
            normalize_embeddings=True,
            use_fp16=use_fp16,
            cache_dir=cache_dir,
            passage_max_length=max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

        logger.info(
            "Initialized FlagEmbeddingModel: %s (fp16=%s, max_length=%d)",
            model_name,
            use_fp16,
            max_length,
        )

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode texts to dense vectors (sparse output discarded).

        Args:
            texts: Input texts.

        Returns:
            Dense embedding vectors, one per input text.
        """
        output = self._model.encode(texts, return_dense=True, return_sparse=False)
        return cast(np.ndarray, output["dense_vecs"]).tolist()

    def encode_with_sparse(
        self, texts: list[str]
    ) -> tuple[list[list[float]], list[SparseVectorData | None]]:
        """Encode texts to dense and sparse vectors in one forward pass.

        Args:
            texts: Input texts.

        Returns:
            Tuple of ``(dense_list, sparse_list)`` aligned to input order.
            ``dense_list[i]`` is a 1024-dim float list.
            ``sparse_list[i]`` is ``{"indices": [...], "values": [...]}``
            where indices are BGE-M3 vocabulary token IDs.
        """
        output = self._model.encode(texts, return_dense=True, return_sparse=True)
        dense: list[list[float]] = cast(np.ndarray, output["dense_vecs"]).tolist()
        sparse: list[SparseVectorData | None] = [
            {
                "indices": [int(k) for k in lw],
                "values": [float(v) for v in cast(dict[str, float], lw).values()],
            }
            for lw in output["lexical_weights"]
        ]
        return dense, sparse

    @property
    def embedding_size(self) -> int:
        return _BGE_M3_DENSE_DIM

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def supports_sparse(self) -> bool:
        return True

    @property
    def supports_multilingual(self) -> bool:
        return True

    def get_model_info(self) -> dict[str, Any]:
        """Return model metadata including sparse support flag.

        Returns:
            Dictionary with model name, dimensions, and capability flags.
        """
        return {
            **super().get_model_info(),
            "supports_sparse": True,
        }
