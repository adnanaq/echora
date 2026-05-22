import logging
from typing import cast

from .base import TextEmbeddingModel

logger = logging.getLogger(__name__)


class HuggingFaceModel(TextEmbeddingModel):
    """HuggingFace implementation of TextEmbeddingModel."""

    def __init__(self, model_name: str, cache_dir: str | None = None):
        """Initialize HuggingFace model.

        Args:
            model_name: HuggingFace model name
            cache_dir: Optional directory to cache model files
        """
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            self._model_name = model_name

            # Load model and tokenizer
            self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, cache_dir=cache_dir
            )

            # Set device
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            self.model.eval()

            # Get embedding size
            self._embedding_size = self.model.config.hidden_size

            # Get max length
            self._max_length = min(self.tokenizer.model_max_length, 512)

            logger.info(f"Initialized HuggingFace model: {model_name} on {self.device}")

        except ImportError as e:
            logger.exception(
                "HuggingFace dependencies not installed. Install with: pip install transformers torch"
            )
            raise ImportError("HuggingFace dependencies missing") from e

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode a list of texts into embeddings.

        Args:
            texts: List of text strings to encode

        Returns:
            List of embedding vectors
        """
        try:
            import torch

            # Tokenize text
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self._max_length,
            ).to(self.device)

            # Generate embedding
            with torch.no_grad():
                outputs = self.model(**inputs)

                # Attention-mask-weighted mean pooling.
                # Simple .mean(dim=1) is wrong for padded batches: it averages
                # real tokens together with zero-padded positions, biasing every
                # embedding.  We weight by the attention mask so only real tokens
                # contribute to the mean.
                token_embeddings = outputs.last_hidden_state
                attention_mask = inputs["attention_mask"]
                mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                summed = (token_embeddings * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-9)
                embeddings = summed / counts

                # Normalize for cosine similarity
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

                return cast(list[list[float]], embeddings.cpu().numpy().tolist())

        except Exception:
            logger.exception("HuggingFace encoding failed")
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
