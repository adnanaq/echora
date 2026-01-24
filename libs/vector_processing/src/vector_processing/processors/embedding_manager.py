"""Multi-Vector Embedding Manager for coordinated embedding generation.

This module coordinates the generation of hierarchical vector points (Anime, Character, Episode)
for the comprehensive anime search system.
"""

import asyncio
import logging
from typing import Any

from common.config import Settings
from common.models.anime import AnimeRecord
from common.utils.id_generation import (
    generate_deterministic_id,
)
from vector_db_interface.base import VectorDocument

from .anime_field_mapper import AnimeFieldMapper
from .text_processor import TextProcessor
from .vision_processor import VisionProcessor

logger = logging.getLogger(__name__)


class MultiVectorEmbeddingManager:
    """Manager for coordinated generation of hierarchical vector points."""

    def __init__(
        self,
        text_processor: TextProcessor,
        vision_processor: VisionProcessor,
        field_mapper: AnimeFieldMapper,
        settings: Settings | None = None,
    ):
        """Initialize the embedding manager with injected processors.

        Args:
            text_processor: An initialized TextProcessor instance.
            vision_processor: An initialized VisionProcessor instance.
            field_mapper: An initialized AnimeFieldMapper instance.
            settings: Configuration settings instance.
        """
        if settings is None:
            settings = Settings()

        self.settings = settings
        self.text_processor = text_processor
        self.vision_processor = vision_processor
        self.field_mapper = field_mapper

        logger.info("Initialized Hierarchical EmbeddingManager")

    async def process_anime_vectors(self, record: AnimeRecord) -> list[VectorDocument]:
        """Process a single anime record into multiple vector documents.

        Creates:
        - 1 Anime Point (text_vector + image_vector as multivector)
        - N Character Points (text_vector + image_vector as multivector each)
        - M Episode Points (text_vector only)

        Images are embedded as multivector matrices within parent entities,
        eliminating the need for separate Image points.

        Args:
            record: The AnimeRecord containing anime, characters, and episodes.

        Returns:
            A list of VectorDocuments for the anime, its characters, and episodes.
        """
        documents: list[VectorDocument] = []

        # 1. Create Anime Point (text + images as multivector)
        anime_doc = await self._create_anime_point(record)
        documents.append(anime_doc)

        # 2. Create Character Points (text + images as multivector)
        character_docs = await self._create_character_points(record)
        documents.extend(character_docs)

        # 3. Create Episode Points (text only)
        episode_docs = await self._create_episode_points(record)
        documents.extend(episode_docs)

        logger.info(
            f"Processed anime '{record.anime.title}': "
            f"1 anime, {len(character_docs)} characters, {len(episode_docs)} episodes"
        )

        return documents

    async def process_anime_batch(
        self, records: list[AnimeRecord]
    ) -> list[VectorDocument]:
        """Process multiple anime records concurrently.

        Processes all records in parallel using asyncio.gather, flattening
        the results into a single list of VectorDocuments.

        Args:
            records: List of AnimeRecord instances to process.

        Returns:
            A flattened list of all VectorDocuments from all records.
            Returns empty list if processing fails entirely.
        """
        try:
            logger.info(f"Processing batch of {len(records)} anime")

            # Process all anime concurrently
            tasks = [self.process_anime_vectors(record) for record in records]
            results_list = await asyncio.gather(*tasks, return_exceptions=True)

            all_documents: list[VectorDocument] = []

            for i, result in enumerate(results_list):
                if isinstance(result, BaseException):
                    logger.error(f"Batch processing error for anime {i}: {result}")
                    continue

                all_documents.extend(result)

            logger.info(
                f"Batch processing complete. Generated {len(all_documents)} total points."
            )
        except Exception:
            logger.exception("Batch processing failed")
        else:
            return all_documents
        return []

    # ============================================================================
    # POINT GENERATION HELPER METHODS
    # ============================================================================

    async def _create_anime_point(self, record: AnimeRecord) -> VectorDocument:
        """Create the parent Anime VectorDocument with text and image embeddings.

        Extracts text content via the field_mapper and encodes it using the
        text_processor. Also extracts all image URLs and encodes them into a
        multivector matrix using the vision_processor.

        Args:
            record: The AnimeRecord containing the anime to process.

        Returns:
            A VectorDocument with text_vector and optionally image_vector (multivector).
        """
        anime = record.anime

        # 1. Extract text using field_mapper
        full_text = self.field_mapper.extract_anime_text(anime)

        # 2. Generate Text Vector
        embeddings: dict[str, list[float] | list[list[float]]] = {}

        text_vec = self.text_processor.encode_text(full_text)
        if text_vec:
            embeddings["text_vector"] = text_vec
        else:
            embeddings["text_vector"] = self.text_processor.get_zero_embedding()

        # 3. Generate Image Vector (multivector matrix)
        image_urls = self.field_mapper.extract_image_urls(anime)
        if image_urls:
            image_matrix = await self.vision_processor.encode_images_batch(image_urls)
            if image_matrix:
                embeddings["image_vector"] = image_matrix

        # 4. Construct Payload
        payload = anime.model_dump(exclude_none=True)
        payload["entity_type"] = "anime"

        return VectorDocument(id=anime.id, vectors=embeddings, payload=payload)

    async def _create_character_points(
        self, record: AnimeRecord
    ) -> list[VectorDocument]:
        """Create Character VectorDocuments linked to this anime.

        Extracts text for each character via field_mapper and batch-encodes
        them using text_processor. Also extracts character image URLs and
        encodes them into a multivector matrix. Characters without IDs get
        deterministic IDs generated from anime_id and character name.

        Args:
            record: The AnimeRecord containing characters to process.

        Returns:
            A list of VectorDocuments, one per character with text_vector
            and optionally image_vector (multivector).
        """
        if not record.characters:
            return []

        anime_id = record.anime.id
        documents: list[VectorDocument] = []

        texts_to_embed: list[str] = []
        valid_characters = []

        for char in record.characters:
            # Use field_mapper directly
            full_text = self.field_mapper.extract_character_text(char)
            texts_to_embed.append(full_text)
            valid_characters.append(char)

        # Batch Embed text
        embeddings_batch = self.text_processor.encode_texts_batch(texts_to_embed)

        for i, char in enumerate(valid_characters):
            text_embedding = embeddings_batch[i]
            if not text_embedding:
                continue

            char_id = char.id
            if not char_id:
                seed = f"{anime_id}_{char.name}"
                char_id = generate_deterministic_id(seed)

            vectors: dict[str, list[float] | list[list[float]]] = {
                "text_vector": text_embedding
            }

            # Generate Image Vector (multivector matrix) for character
            image_urls = self.field_mapper.extract_character_image_urls(char)
            if image_urls:
                image_matrix = await self.vision_processor.encode_images_batch(
                    image_urls
                )
                if image_matrix:
                    vectors["image_vector"] = image_matrix

            payload = char.model_dump(exclude_none=True)
            payload["entity_type"] = "character"
            # Merge anime_ids: preserve existing anime relationships
            existing_anime_ids = payload.get("anime_ids") or []
            if anime_id not in existing_anime_ids:
                existing_anime_ids.append(anime_id)
            payload["anime_ids"] = existing_anime_ids

            documents.append(
                VectorDocument(id=char_id, vectors=vectors, payload=payload)
            )

        return documents

    async def _create_episode_points(self, record: AnimeRecord) -> list[VectorDocument]:
        """Create Episode VectorDocuments linked to this anime.

        Extracts text for each episode via field_mapper and batch-encodes
        them using text_processor. Episodes without IDs get generated IDs
        based on anime_id and episode number.

        Args:
            record: The AnimeRecord containing episodes to process.

        Returns:
            A list of VectorDocuments, one per episode with text_vector.
        """
        if not record.episodes:
            return []

        anime_id = record.anime.id
        documents: list[VectorDocument] = []

        texts_to_embed: list[str] = []
        valid_episodes = []

        for ep in record.episodes:
            # Use field_mapper directly
            full_text = self.field_mapper.extract_episode_text(ep)
            texts_to_embed.append(full_text)
            valid_episodes.append(ep)

        # Batch Embed
        embeddings_batch = self.text_processor.encode_texts_batch(texts_to_embed)

        for i, ep in enumerate(valid_episodes):
            embedding = embeddings_batch[i]
            if not embedding:
                continue

            ep_id = ep.id
            if not ep_id:
                ep_id = generate_deterministic_id(f"{anime_id}_{ep.episode_number}")

            payload = ep.model_dump(exclude_none=True)
            payload["entity_type"] = "episode"
            payload["anime_id"] = anime_id

            documents.append(
                VectorDocument(
                    id=ep_id, vectors={"text_vector": embedding}, payload=payload
                )
            )

        return documents

    def get_embedding_stats(self) -> dict[str, Any]:
        """Get statistics and model info from all processors.

        Returns:
            A dictionary containing model info from text_processor and
            vision_processor, or an error message if retrieval fails.
        """
        try:
            return {
                "text_processor": self.text_processor.get_model_info(),
                "vision_processor": self.vision_processor.get_model_info(),
            }
        except Exception as e:
            return {"error": str(e)}
