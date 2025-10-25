"""Text embedding processor supporting multiple models for anime text search.

Supports FastEmbed, HuggingFace, and BGE models with dynamic model selection
for optimal performance.
"""

import logging
from typing import TYPE_CHECKING, Any, cast

from ...config import Settings
from ...models.anime import AnimeEntry

if TYPE_CHECKING:
    from .anime_field_mapper import AnimeFieldMapper

logger = logging.getLogger(__name__)


class TextProcessor:
    """Text embedding processor supporting multiple models."""

    def __init__(self, settings: Settings | None = None):
        """Initialize modern text processor with configuration.

        Args:
            settings: Configuration settings instance
        """
        if settings is None:
            from ...config import Settings

            settings = Settings()

        self.settings = settings
        self.provider = settings.text_embedding_provider
        self.model_name = settings.text_embedding_model
        self.cache_dir = settings.model_cache_dir

        # Model instance
        self.model: dict[str, Any] | None = None

        # Model metadata
        self.model_info: dict[str, Any] = {}

        # Initialize models
        self._init_models()

        # Initialize field mapper for multi-vector processing
        self._field_mapper: AnimeFieldMapper | None = None

    def _init_models(self) -> None:
        """Initialize text embedding model."""
        try:
            # Initialize model
            self.model = self._create_model(self.provider, self.model_name)

            # Warm up model if enabled
            if self.settings.model_warm_up:
                self._warm_up_model()

            logger.info(
                f"Initialized modern text processor with {self.provider} model: {self.model_name}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize modern text processor: {e}")
            raise

    def _create_model(self, provider: str, model_name: str) -> dict[str, Any]:
        """Create a model instance based on provider and model name.

        Args:
            provider: Model provider (fastembed, huggingface, sentence-transformers)
            model_name: Model name/path

        Returns:
            Dictionary containing model instance and metadata
        """
        if provider == "fastembed":
            return self._create_fastembed_model(model_name)
        elif provider == "huggingface":
            return self._create_huggingface_model(model_name)
        elif provider == "sentence-transformers":
            return self._create_sentence_transformers_model(model_name)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def _create_fastembed_model(self, model_name: str) -> dict[str, Any]:
        """Create FastEmbed model instance.

        Args:
            model_name: FastEmbed model name

        Returns:
            Dictionary with FastEmbed model and metadata
        """
        try:
            from fastembed import TextEmbedding

            # Initialize FastEmbed model
            init_kwargs: dict[str, Any] = {"model_name": model_name}
            if self.cache_dir:
                init_kwargs["cache_dir"] = self.cache_dir

            model = TextEmbedding(**init_kwargs)

            # Get model info
            embedding_size = self._get_fastembed_embedding_size(model_name)

            return {
                "model": model,
                "provider": "fastembed",
                "model_name": model_name,
                "embedding_size": embedding_size,
                "max_length": 512,  # FastEmbed default
                "batch_size": 256,
                "supports_multilingual": "multilingual" in model_name.lower()
                or "m3" in model_name.lower(),
            }

        except ImportError as e:
            logger.error("FastEmbed not installed. Install with: pip install fastembed")
            raise ImportError("FastEmbed dependencies missing") from e

    def _create_huggingface_model(self, model_name: str) -> dict[str, Any]:
        """Create HuggingFace model instance.

        Args:
            model_name: HuggingFace model name

        Returns:
            Dictionary with HuggingFace model and metadata
        """
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            # Load model and tokenizer
            model = AutoModel.from_pretrained(model_name, cache_dir=self.cache_dir)
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, cache_dir=self.cache_dir
            )  # type: ignore[no-untyped-call]

            # Set device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            model.eval()

            # Get embedding size
            embedding_size = model.config.hidden_size

            # Get max length
            max_length = min(tokenizer.model_max_length, 512)

            return {
                "model": model,
                "tokenizer": tokenizer,
                "device": device,
                "provider": "huggingface",
                "model_name": model_name,
                "embedding_size": embedding_size,
                "max_length": max_length,
                "batch_size": 32,
                "supports_multilingual": self._is_multilingual_model(model_name),
            }

        except ImportError as e:
            logger.error(
                "HuggingFace dependencies not installed. Install with: pip install transformers torch"
            )
            raise ImportError("HuggingFace dependencies missing") from e

    def _create_sentence_transformers_model(self, model_name: str) -> dict[str, Any]:
        """Create Sentence Transformers model instance.

        Args:
            model_name: Sentence Transformers model name

        Returns:
            Dictionary with Sentence Transformers model and metadata
        """
        try:
            from sentence_transformers import SentenceTransformer

            # Load model
            model = SentenceTransformer(model_name, cache_folder=self.cache_dir)

            # Get model info
            embedding_size = model.get_sentence_embedding_dimension()
            max_length = model.max_seq_length

            return {
                "model": model,
                "provider": "sentence-transformers",
                "model_name": model_name,
                "embedding_size": embedding_size,
                "max_length": max_length,
                "batch_size": 32,
                "supports_multilingual": self._is_multilingual_model(model_name),
            }

        except ImportError as e:
            logger.error(
                "Sentence Transformers not installed. Install with: pip install sentence-transformers"
            )
            raise ImportError("Sentence Transformers dependencies missing") from e

    def _get_fastembed_embedding_size(self, model_name: str) -> int:
        """Get embedding size for FastEmbed model.

        Args:
            model_name: FastEmbed model name

        Returns:
            Embedding dimension size
        """
        # Common FastEmbed model dimensions
        model_dimensions = {
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

        return model_dimensions.get(model_name, 384)  # Default to 384

    def _is_multilingual_model(self, model_name: str) -> bool:
        """Check if model supports multilingual text.

        Args:
            model_name: Model name

        Returns:
            True if model supports multiple languages
        """
        multilingual_indicators = [
            "multilingual",
            "m3",
            "xlm",
            "xlm-roberta",
            "mbert",
            "distilbert-base-multilingual",
            "jina-embeddings-v2-base-multilingual",
        ]

        model_lower = model_name.lower()
        return any(indicator in model_lower for indicator in multilingual_indicators)

    def _detect_model_provider(self, model_name: str) -> str:
        """Detect model provider from model name.

        Args:
            model_name: Model name or path

        Returns:
            Provider name (fastembed, huggingface, sentence-transformers)
        """
        model_lower = model_name.lower()

        # FastEmbed common models
        fastembed_models = [
            "baai/bge",
            "sentence-transformers/all-minilm",
            "intfloat/e5",
            "sentence-transformers/all-mpnet",
        ]

        if any(model in model_lower for model in fastembed_models):
            return "fastembed"
        elif "sentence-transformers" in model_lower:
            return "sentence-transformers"
        else:
            return "huggingface"

    def _warm_up_model(self) -> None:
        """Warm up model with dummy data."""
        try:
            if self.model is None:
                logger.warning("Cannot warm up model: model not initialized")
                return

            dummy_text = "This is a test sentence for model warm-up."
            self._encode_text_with_model(dummy_text, self.model)
            logger.info("Text model warmed up successfully")

        except Exception as e:
            logger.warning(f"Text model warm-up failed: {e}")

    def encode_text(self, text: str) -> list[float] | None:
        """Encode text to embedding vector.

        Args:
            text: Input text string

        Returns:
            Embedding vector or None if encoding fails
        """
        try:
            # Handle empty text
            if not text.strip():
                return self._create_zero_vector()

            # Encode with model
            if self.model is not None:
                embedding = self._encode_text_with_model(text, self.model)
                return embedding
            else:
                logger.error("Model not initialized")
                return self._create_zero_vector()

        except Exception as e:
            logger.error(f"Text encoding failed: {e}")
            return None

    def _encode_text_with_model(
        self, text: str, model_dict: dict[str, Any]
    ) -> list[float] | None:
        """Encode text with specific model.

        Args:
            text: Input text
            model_dict: Model dictionary with instance and metadata

        Returns:
            Embedding vector or None if encoding fails
        """
        try:
            provider = model_dict["provider"]

            if provider == "fastembed":
                return self._encode_with_fastembed(text, model_dict)
            elif provider == "huggingface":
                return self._encode_with_huggingface(text, model_dict)
            elif provider == "sentence-transformers":
                return self._encode_with_sentence_transformers(text, model_dict)
            else:
                logger.error(f"Unsupported provider: {provider}")
                return None

        except Exception as e:
            logger.error(f"Model encoding failed: {e}")
            return None

    def _encode_with_fastembed(
        self, text: str, model_dict: dict[str, Any]
    ) -> list[float] | None:
        """Encode text with FastEmbed model.

        Args:
            text: Input text
            model_dict: FastEmbed model dictionary

        Returns:
            Embedding vector or None if encoding fails
        """
        try:
            model = model_dict["model"]

            # Generate embedding
            embeddings = list(model.embed([text]))
            if embeddings:
                return cast(list[float], embeddings[0].tolist())
            else:
                return None

        except Exception as e:
            logger.error(f"FastEmbed encoding failed: {e}")
            return None

    def _encode_with_huggingface(
        self, text: str, model_dict: dict[str, Any]
    ) -> list[float] | None:
        """Encode text with HuggingFace model.

        Args:
            text: Input text
            model_dict: HuggingFace model dictionary

        Returns:
            Embedding vector or None if encoding fails
        """
        try:
            import torch

            model = model_dict["model"]
            tokenizer = model_dict["tokenizer"]
            device = model_dict["device"]
            max_length = model_dict["max_length"]

            # Tokenize text
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_length,
            ).to(device)

            # Generate embedding
            with torch.no_grad():
                outputs = model(**inputs)
                # Use CLS token or mean pooling
                if hasattr(outputs, "last_hidden_state"):
                    embeddings = outputs.last_hidden_state.mean(dim=1)
                else:
                    embeddings = outputs.pooler_output

                # Normalize
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                embedding = embeddings.cpu().numpy().flatten().tolist()

            return cast(list[float], embedding)

        except Exception as e:
            logger.error(f"HuggingFace encoding failed: {e}")
            return None

    def _encode_with_sentence_transformers(
        self, text: str, model_dict: dict[str, Any]
    ) -> list[float] | None:
        """Encode text with Sentence Transformers model.

        Args:
            text: Input text
            model_dict: Sentence Transformers model dictionary

        Returns:
            Embedding vector or None if encoding fails
        """
        try:
            model = model_dict["model"]

            # Generate embedding
            embedding = model.encode(text)
            return cast(list[float], embedding.tolist())

        except Exception as e:
            logger.error(f"Sentence Transformers encoding failed: {e}")
            return None

    def _create_zero_vector(self) -> list[float]:
        """Create zero vector for empty text.

        Returns:
            Zero vector with appropriate dimensions
        """
        if self.model:
            size = self.model["embedding_size"]
            return [0.0] * size
        else:
            return [0.0] * 384  # Default size

    def encode_texts_batch(
        self, texts: list[str], batch_size: int | None = None
    ) -> list[list[float] | None]:
        """Encode multiple texts in batches.

        Args:
            texts: List of text strings
            batch_size: Batch size (uses model-specific default if None)

        Returns:
            List of embedding vectors
        """
        try:
            if batch_size is None:
                batch_size = self.model.get("batch_size", 32) if self.model else 32

            embeddings = []
            total_texts = len(texts)

            logger.info(f"Processing {total_texts} texts in batches of {batch_size}")

            for i in range(0, total_texts, batch_size):
                batch = texts[i : i + batch_size]
                batch_embeddings = []

                for text in batch:
                    embedding = self.encode_text(text)
                    batch_embeddings.append(embedding)

                embeddings.extend(batch_embeddings)

                # Log progress
                processed = min(i + batch_size, total_texts)
                logger.info(f"Processed {processed}/{total_texts} texts")

            return embeddings

        except Exception as e:
            logger.error(f"Batch text encoding failed: {e}")
            return [None] * len(texts)

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the current model.

        Returns:
            Dictionary with model information
        """
        if self.model:
            return {
                "provider": self.model["provider"],
                "model_name": self.model["model_name"],
                "embedding_size": self.model["embedding_size"],
                "max_length": self.model["max_length"],
                "batch_size": self.model["batch_size"],
                "supports_multilingual": self.model["supports_multilingual"],
            }
        else:
            return {"error": "No model loaded"}

    def switch_model(self, provider: str, model_name: str) -> bool:
        """Switch to a different model.

        Args:
            provider: New model provider
            model_name: New model name

        Returns:
            True if switch successful, False otherwise
        """
        try:
            # Create new model
            new_model = self._create_model(provider, model_name)

            # Switch to new model
            self.model = new_model
            self.provider = provider
            self.model_name = model_name

            logger.info(f"Switched to {provider} model: {model_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to switch model: {e}")
            return False

    def get_bge_model_name(self) -> str:
        """Get the appropriate BGE model name based on settings.

        Returns:
            BGE model name
        """
        version = self.settings.bge_model_version
        size = self.settings.bge_model_size

        if version == "m3":
            return "BAAI/bge-m3"  # Always multilingual
        elif version == "reranker":
            return f"BAAI/bge-reranker-{size}"
        else:  # v1.5 series
            return f"BAAI/bge-{size}-en-{version}"

    def validate_text(self, text: str) -> bool:
        """Validate if text can be processed.

        Args:
            text: Input text string

        Returns:
            True if text is valid, False otherwise
        """
        try:
            if not isinstance(text, str):
                return False

            # Check length
            if self.model:
                max_length = self.model.get("max_length", 512)
                if len(text) > max_length * 4:  # Rough token estimate
                    return False

            return True

        except Exception:
            return False

    # ============================================================================
    # MULTI-VECTOR PROCESSING METHODS
    # ============================================================================

    def _get_field_mapper(self) -> "AnimeFieldMapper":
        """Lazy initialization of field mapper."""
        if self._field_mapper is None:
            from .anime_field_mapper import AnimeFieldMapper

            self._field_mapper = AnimeFieldMapper()
        return self._field_mapper

    def process_anime_vectors(self, anime: AnimeEntry) -> dict[str, list[float]]:
        """
        Process anime data into multiple semantic text embeddings.

        Args:
            anime: AnimeEntry with comprehensive anime data

        Returns:
            Dict mapping vector names to their embeddings
        """
        try:
            field_mapper = self._get_field_mapper()

            # Extract field content for all vectors
            vector_data = field_mapper.map_anime_to_vectors(anime)

            # Generate embeddings for text vectors only
            text_embeddings = {}
            text_vectors = [
                name
                for name, vec_type in field_mapper.get_vector_types().items()
                if vec_type == "text"
            ]

            for vector_name in text_vectors:
                if vector_name in vector_data:
                    text_content = vector_data[vector_name]

                    # Convert to string if it's a list
                    if isinstance(text_content, list):
                        content_str = " ".join(str(item) for item in text_content)
                    else:
                        content_str = str(text_content)

                    # Apply field-specific preprocessing
                    processed_text = self._preprocess_field_content(
                        content_str, vector_name
                    )

                    # Generate embedding with hierarchical averaging for episode chunks
                    if processed_text.strip():
                        if (
                            vector_name == "episode_vector"
                            and "|| CHUNK_SEPARATOR ||" in processed_text
                        ):
                            # Handle hierarchical averaging for episode chunks
                            embedding = self._encode_with_hierarchical_averaging(
                                processed_text
                            )
                        else:
                            # Standard single embedding
                            embedding = self.encode_text(processed_text)

                        if embedding:
                            text_embeddings[vector_name] = embedding
                        else:
                            # Use zero vector for failed embedding
                            text_embeddings[vector_name] = self._get_zero_embedding()
                    else:
                        # Use zero vector for empty content
                        text_embeddings[vector_name] = self._get_zero_embedding()

            logger.debug(
                f"Generated embeddings for {len(text_embeddings)} text vectors"
            )
            return text_embeddings

        except Exception as e:
            logger.error(f"Failed to process anime vectors: {e}")
            raise

    def _encode_with_hierarchical_averaging(
        self, chunked_text: str
    ) -> list[float] | None:
        """
        Encode text with hierarchical averaging for episode chunks.

        Args:
            chunked_text: Text with "|| CHUNK_SEPARATOR ||" delimiters

        Returns:
            Single averaged embedding vector or None if encoding fails
        """
        try:
            # Split text into chunks
            chunks = [
                chunk.strip() for chunk in chunked_text.split("|| CHUNK_SEPARATOR ||")
            ]
            chunks = [chunk for chunk in chunks if chunk]  # Remove empty chunks

            if not chunks:
                return self._get_zero_embedding()

            # For single chunk, encode directly (no averaging needed)
            if len(chunks) == 1:
                return self.encode_text(chunks[0])

            logger.debug(
                f"Processing {len(chunks)} episode chunks with hierarchical averaging"
            )

            # Encode each chunk individually
            chunk_embeddings = []
            for i, chunk in enumerate(chunks):
                chunk_embedding = self.encode_text(chunk)
                if chunk_embedding:
                    chunk_embeddings.append(chunk_embedding)
                else:
                    logger.warning(
                        f"Failed to encode episode chunk {i+1}/{len(chunks)}"
                    )

            if not chunk_embeddings:
                logger.error("All episode chunks failed to encode")
                return self._get_zero_embedding()

            # Hierarchical averaging: equal weight for all chunks
            # This preserves semantic coherence better than weighted averaging for BGE-M3
            from typing import cast

            import numpy as np

            # Convert to numpy for efficient averaging
            chunk_matrix = np.array(chunk_embeddings)
            averaged_embedding = np.mean(chunk_matrix, axis=0)

            # Convert back to list with proper typing
            result_embedding: list[float] = cast(
                list[float], averaged_embedding.tolist()
            )

            logger.debug(
                f"Successfully averaged {len(chunk_embeddings)} episode chunks"
            )
            return result_embedding

        except Exception as e:
            logger.error(f"Hierarchical averaging failed: {e}")
            return self._get_zero_embedding()

    def _preprocess_field_content(self, content: str, vector_name: str) -> str:
        """
        Apply field-specific preprocessing to improve embedding quality.

        Args:
            content: Raw text content from field mapper
            vector_name: Name of the vector being processed

        Returns:
            Preprocessed text optimized for embedding
        """
        if not content:
            return ""

        # Apply general preprocessing
        processed = content.strip()

        # Field-specific preprocessing rules
        if vector_name == "title_vector":
            processed = processed.replace("Synopsis:", "Story:")
            processed = processed.replace("Background:", "Production:")

        elif vector_name == "character_vector":
            processed = processed.replace("Role:", "Character Role:")
            processed = processed.replace("Description:", "Background:")

        elif vector_name == "genre_vector":
            processed = processed.replace("Shounen", "Shonen (young male)")
            processed = processed.replace("Shoujo", "Shojo (young female)")
            processed = processed.replace("Seinen", "Seinen (adult male)")
            processed = processed.replace("Josei", "Josei (adult female)")

        elif vector_name == "sources_vector":
            processed = processed.replace("Source:", "Platform:")
            processed = processed.replace("External:", "External Platform:")

        return processed

    def _get_zero_embedding(self) -> list[float]:
        """Get zero embedding vector for empty/failed content."""
        if self.model:
            embedding_size = self.model["embedding_size"]
            return [0.0] * embedding_size
        else:
            return [0.0] * 384  # Default size

    def get_text_vector_names(self) -> list[str]:
        """Get list of text vector names supported by this processor."""
        field_mapper = self._get_field_mapper()
        return [
            name
            for name, vec_type in field_mapper.get_vector_types().items()
            if vec_type == "text"
        ]
