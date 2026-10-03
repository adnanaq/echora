"""FlagEmbedding BGE-M3 backend producing dense and sparse vectors in one pass."""

import logging
from typing import Any

import numpy as np
from vector_db_interface import SparseVectorData

from .base import TextEmbeddingModel

logger = logging.getLogger(__name__)

_BGE_M3_DENSE_DIM = 1024
_SPECIAL_TOKENS = ("cls_token", "eos_token", "pad_token", "unk_token")


class ChunkSizeError(ValueError):
    """Raised when the model chunk size is below 1."""

    def __init__(self, chunk_size: int) -> None:
        super().__init__(f"chunk_size must be at least 1, got {chunk_size}")


class TokenBudgetError(ValueError):
    """Raised when the token budget per model pass is negative."""

    def __init__(self, max_tokens_per_pass: int) -> None:
        super().__init__(
            f"max_tokens_per_pass must be 0 (off) or more, got {max_tokens_per_pass}"
        )


class FlagEmbeddingModel(TextEmbeddingModel):
    """BGE-M3 backend via FlagEmbedding.

    Produces both dense (1024-dim) and sparse (lexical) vectors in a single
    forward pass. Dense uses CLS-token pooling (correct for BGE-M3);
    sparse weights are the model's learned lexical projection.

    Encoding runs FlagEmbedding's tokenizer and model directly rather than
    ``BGEM3FlagModel.encode``, which runs the model twice per call: a trial
    pass to pick a batch size, then the real one (FlagEmbedding issue #1308).
    The output is identical; running out of GPU memory halves the chunk and
    retries, as the trial pass did.

    A batch is sorted by length, longest first, and run in chunks. A chunk
    ends at ``chunk_size`` texts, or, with ``max_tokens_per_pass`` set, when
    one more text would make its padded size (texts times its longest text's
    tokens) exceed that budget. Each chunk is padded only to its own longest
    text, so long texts share a pass with long ones and short with short.
    """

    def __init__(
        self,
        model_name: str,
        cache_dir: str | None = None,
        max_length: int = 8192,
        chunk_size: int = 256,
        max_tokens_per_pass: int = 0,
    ) -> None:
        """Initialize BGE-M3 via FlagEmbedding.

        Args:
            model_name: HuggingFace model identifier (e.g. ``BAAI/bge-m3``).
            cache_dir: Optional directory for downloaded model files.
            max_length: Maximum token sequence length for passage encoding.
            chunk_size: Most texts in one model pass.
            max_tokens_per_pass: Most padded tokens in one model pass; 0 turns
                the budget off. A text longer than the budget gets a pass alone.

        Raises:
            ImportError: If FlagEmbedding is not installed.
            ChunkSizeError: If ``chunk_size`` is below 1.
            TokenBudgetError: If ``max_tokens_per_pass`` is negative.
        """
        if chunk_size < 1:
            raise ChunkSizeError(chunk_size)
        if max_tokens_per_pass < 0:
            raise TokenBudgetError(max_tokens_per_pass)
        try:
            import torch
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:
            raise ImportError(
                "FlagEmbedding not installed. Install with: pip install FlagEmbedding"
            ) from exc

        self._model_name = model_name
        self._max_length = max_length
        self._chunk_size = chunk_size
        self._max_tokens_per_pass = max_tokens_per_pass
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
        encoder, tokenizer = self._model.model, self._model.tokenizer
        if encoder is None or tokenizer is None:
            raise RuntimeError(f"FlagEmbedding loaded no model for {model_name}")
        self._encoder = encoder
        self._tokenizer = tokenizer
        device = self._model.target_devices[0]
        self._device = device
        self._move_model_to_device()
        self._special_token_ids = {
            tokenizer.convert_tokens_to_ids(tokenizer.special_tokens_map[name])
            for name in _SPECIAL_TOKENS
            if name in tokenizer.special_tokens_map
        }

        logger.info(
            f"Initialized FlagEmbeddingModel: {model_name} on {device} (fp16={use_fp16}, max_length={max_length}, chunk_size={chunk_size}, max_tokens_per_pass={max_tokens_per_pass})"
        )

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode texts to dense vectors (sparse output discarded).

        Args:
            texts: Input texts.

        Returns:
            Dense embedding vectors, one per input text.
        """
        dense, _ = self._encode_single_pass(texts, return_sparse=False)
        return dense.tolist()

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
        dense, token_weights = self._encode_single_pass(texts, return_sparse=True)
        sparse: list[SparseVectorData | None] = [
            {"indices": list(weights), "values": list(weights.values())}
            for weights in token_weights
        ]
        return dense.tolist(), sparse

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

    def _move_model_to_device(self) -> None:
        """Move the model to its device once, at load.

        FlagEmbedding loads the model on the CPU and moves it at the start of
        every encode call. Two threads doing that first move together crash
        the process, so the move happens here instead and later calls find the
        model already in place.
        """
        if self._device == "cpu":
            self._encoder.float()
        self._encoder.to(self._device)
        self._encoder.eval()

    def _encode_single_pass(
        self, texts: list[str], *, return_sparse: bool
    ) -> tuple[np.ndarray, list[dict[int, float]]]:
        """Tokenize, sort by length, and run each chunk through the model once.

        Sorting by length keeps padding small, as FlagEmbedding does. Results
        are returned in input order.
        """
        import torch

        tokenized = self._tokenizer(texts, truncation=True, max_length=self._max_length)
        examples = [
            {key: tokenized[key][index] for key in tokenized}
            for index in range(len(texts))
        ]
        order = np.argsort([-len(example["input_ids"]) for example in examples])
        dense_chunks: list[np.ndarray] = []
        weights_in_order: list[dict[int, float]] = []
        chunk_size = self._chunk_size
        position = 0
        while position < len(texts):
            chunk_end = self._chunk_end(examples, order, position, chunk_size)
            chunk = [examples[index] for index in order[position:chunk_end]]
            try:
                dense, weights = self._run_model(chunk, return_sparse=return_sparse)
            except torch.OutOfMemoryError:
                if chunk_size == 1:
                    raise
                chunk_size = max(1, chunk_size // 2)
                logger.warning(
                    f"Out of GPU memory; retrying with chunks of {chunk_size}"
                )
                continue
            dense_chunks.append(dense)
            weights_in_order.extend(weights)
            position += len(chunk)
        restore = np.argsort(order)
        token_weights = (
            [weights_in_order[index] for index in restore] if return_sparse else []
        )
        return np.concatenate(dense_chunks)[restore], token_weights

    def _chunk_end(
        self,
        examples: list[dict[str, Any]],
        order: np.ndarray,
        start: int,
        chunk_size: int,
    ) -> int:
        """Where the chunk starting at ``start`` ends, in the length-sorted order.

        The first text is the chunk's longest, so the chunk's padded size is
        its text count times that text's length.
        """
        end = min(start + chunk_size, len(order))
        if not self._max_tokens_per_pass:
            return end
        longest = len(examples[order[start]]["input_ids"])
        fitting_texts = max(1, self._max_tokens_per_pass // longest)
        return min(end, start + fitting_texts)

    def _run_model(
        self, chunk: list[dict[str, Any]], *, return_sparse: bool
    ) -> tuple[np.ndarray, list[dict[int, float]]]:
        """Run one padded chunk through the model; return dense rows and token weights."""
        import torch
        from FlagEmbedding.utils.tokenizer_compat import pad_with_compat

        inputs = pad_with_compat(
            self._tokenizer, chunk, padding=True, return_tensors="pt"
        ).to(self._device)
        with torch.no_grad():
            outputs = self._encoder(
                inputs,
                return_dense=True,
                return_sparse=return_sparse,
                return_colbert_vecs=False,
                truncate_dim=self._model.truncate_dim,
            )
        dense = self._model._convert_to_numpy(
            outputs["dense_vecs"], device=self._device
        )
        if not return_sparse:
            return dense, []
        weights = self._model._convert_to_numpy(
            outputs["sparse_vecs"].squeeze(-1), device=self._device
        )
        return dense, [
            self._token_weights(row, token_ids)
            for row, token_ids in zip(
                weights, inputs["input_ids"].tolist(), strict=True
            )
        ]

    def _token_weights(
        self, weights: np.ndarray, token_ids: list[int]
    ) -> dict[int, float]:
        """Keep each non-special token's highest positive weight, as FlagEmbedding does."""
        result: dict[int, float] = {}
        for weight, token_id in zip(weights, token_ids, strict=True):
            if token_id in self._special_token_ids or weight <= 0:
                continue
            if weight > result.get(token_id, 0.0):
                result[token_id] = float(weight)
        return result
