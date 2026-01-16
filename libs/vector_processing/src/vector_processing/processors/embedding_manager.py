"""Multi-Vector Embedding Manager for coordinated embedding generation.

This module coordinates the generation of hierarchical vector points (Anime, Character, Episode)
for the comprehensive anime search system.
"""

import asyncio
import logging
from typing import Any

from common.config import Settings
from common.models.anime import Anime, AnimeRecord, Character, Image
from common.utils.id_generation import (
    generate_deterministic_id,
    generate_episode_id,
    generate_ulid,
)
from vector_db_interface.base import VectorDocument

from .text_processor import TextProcessor
from .vision_processor import VisionProcessor

logger = logging.getLogger(__name__)


class MultiVectorEmbeddingManager:
    """Manager for coordinated generation of hierarchical vector points."""

    def __init__(
        self,
        text_processor: TextProcessor,
        vision_processor: VisionProcessor,
        # field_mapper legacy dependency removed
        settings: Settings | None = None,
    ):
        """Initialize the embedding manager with injected processors.

        Args:
            text_processor: An initialized TextProcessor instance.
            vision_processor: An initialized VisionProcessor instance.
            settings: Configuration settings instance.
        """
        if settings is None:
            settings = Settings()

        self.settings = settings
        self.text_processor = text_processor
        self.vision_processor = vision_processor

        logger.info("Initialized Hierarchical EmbeddingManager")

    async def process_anime_vectors(self, record: AnimeRecord) -> list[VectorDocument]:
        """Process a single anime record into multiple vector documents.

        Creates:
        - 1 Anime Point (text_vector)
        - N Character Points (text_vector each)
        - M Episode Points (text_vector each)
        - P Image Points (image_vector each) - for anime and character images
        """
        documents: list[VectorDocument] = []

        # Create semaphore for parallel image processing (5 concurrent)
        semaphore = asyncio.Semaphore(5)

        # 1. Create Anime Point (text only)
        anime_doc = await self._create_anime_point(record)
        documents.append(anime_doc)

        # 2. Create Character Points (text only)
        character_docs = await self._create_character_points(record)
        documents.extend(character_docs)

        # 3. Create Episode Points (text only)
        episode_docs = await self._create_episode_points(record)
        documents.extend(episode_docs)

        # 4. Create Image Points (image_vector) - parallel processing
        anime_image_docs = await self._create_anime_image_points(record.anime, semaphore)
        documents.extend(anime_image_docs)

        character_image_docs = await self._create_character_image_points(
            record.characters, semaphore
        )
        documents.extend(character_image_docs)

        logger.info(
            f"Processed anime '{record.anime.title}': "
            f"1 anime, {len(character_docs)} characters, {len(episode_docs)} episodes, "
            f"{len(anime_image_docs)} anime images, {len(character_image_docs)} character images"
        )

        return documents

    async def process_anime_batch(
        self, records: list[AnimeRecord]
    ) -> list[VectorDocument]:
        """Process multiple anime records in batch.

        Args:
            records: List of AnimeRecord instances

        Returns:
            Flattened list of all VectorDocuments from all anime in batch.
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

                # result is list[VectorDocument]
                all_documents.extend(result)

            logger.info(
                f"Batch processing complete. Generated {len(all_documents)} total points."
            )
            return all_documents

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            return []

    # ============================================================================
    # POINT GENERATION HELPER METHODS
    # ============================================================================

    async def _create_anime_point(self, record: AnimeRecord) -> VectorDocument:
        """Create the parent Anime Point."""
        anime = record.anime

        # 1. Construct Text
        # Combine visible metadata into a rich text representation
        text_parts = [
            f"Title: {anime.title}",
        ]
        if anime.title_english:
            text_parts.append(f"English Title: {anime.title_english}")

        text_parts.append(f"Type: {anime.type}")

        if anime.genres:
            text_parts.append(f"Genres: {', '.join(anime.genres)}")

        if anime.synopsis:
            text_parts.append(f"Synopsis: {anime.synopsis}")

        full_text = "\n".join(text_parts)

        # 2. Generate Vectors
        embeddings: dict[str, list[float]] = {}

        # Text Vector (only - image vectors are separate Image Points now)
        text_vec = self.text_processor.encode_text(full_text)
        if text_vec:
            embeddings["text_vector"] = text_vec
        else:
            # Fallback zero vector if text is empty/error
            embeddings["text_vector"] = self.text_processor._get_zero_embedding()

        # 3. Construct Payload
        payload = anime.model_dump(exclude_none=True)
        payload["type"] = "anime"
        payload["title_text"] = anime.title  # Full-text searchable

        return VectorDocument(id=anime.id, vectors=embeddings, payload=payload)

    async def _create_anime_image_points(
        self, anime: Anime, semaphore: asyncio.Semaphore
    ) -> list[VectorDocument]:
        """Create Image Points for all anime images (covers, posters, banners)."""
        if not anime.images:
            return []

        documents: list[VectorDocument] = []
        image_urls: list[str] = []

        # Flatten all image URLs from dict structure
        for image_type, urls in anime.images.items():
            image_urls.extend(urls)

        # Remove duplicates while preserving order
        image_urls = list(dict.fromkeys(image_urls))

        if not image_urls:
            return []

        async def process_single_image(url: str) -> VectorDocument | None:
            async with semaphore:
                try:
                    image_path = (
                        await self.vision_processor.downloader.download_and_cache_image(
                            url
                        )
                    )
                    if not image_path:
                        return None

                    embedding = self.vision_processor.encode_image(image_path)
                    if not embedding:
                        return None

                    image = Image(
                        id=generate_ulid("image"),
                        image_url=url,
                        anime_id=anime.id,
                    )

                    payload = image.model_dump(exclude_none=True)
                    payload["type"] = "image"

                    return VectorDocument(
                        id=image.id,
                        vectors={"image_vector": embedding},
                        payload=payload,
                    )
                except Exception as e:
                    logger.warning(f"Failed to process anime image {url}: {e}")
                    return None

        # Process all images in parallel with semaphore
        tasks = [process_single_image(url) for url in image_urls]
        results = await asyncio.gather(*tasks)

        # Filter out None results
        documents = [doc for doc in results if doc is not None]

        logger.info(f"Created {len(documents)} image points for anime {anime.title}")
        return documents

    async def _create_character_image_points(
        self, characters: list[Character], semaphore: asyncio.Semaphore
    ) -> list[VectorDocument]:
        """Create Image Points for all character images (portraits)."""
        # Collect all (character_id, image_url) pairs
        image_tasks: list[tuple[str, str]] = []
        for char in characters:
            if not char.images:
                continue
            char_id = char.id
            if not char_id:
                continue
            for url in char.images:
                image_tasks.append((char_id, url))

        if not image_tasks:
            return []

        async def process_single_image(char_id: str, url: str) -> VectorDocument | None:
            async with semaphore:
                try:
                    image_path = (
                        await self.vision_processor.downloader.download_and_cache_image(
                            url
                        )
                    )
                    if not image_path:
                        return None

                    embedding = self.vision_processor.encode_image(image_path)
                    if not embedding:
                        return None

                    image = Image(
                        id=generate_ulid("image"),
                        image_url=url,
                        character_id=char_id,
                    )

                    payload = image.model_dump(exclude_none=True)
                    payload["type"] = "image"

                    return VectorDocument(
                        id=image.id,
                        vectors={"image_vector": embedding},
                        payload=payload,
                    )
                except Exception as e:
                    logger.warning(f"Failed to process character image {url}: {e}")
                    return None

        # Process all images in parallel with semaphore
        tasks = [process_single_image(char_id, url) for char_id, url in image_tasks]
        results = await asyncio.gather(*tasks)

        # Filter out None results
        documents = [doc for doc in results if doc is not None]

        logger.info(
            f"Created {len(documents)} image points for {len(characters)} characters"
        )
        return documents

    async def _create_character_points(
        self, record: AnimeRecord
    ) -> list[VectorDocument]:
        """Create Character Points linked to this anime."""
        if not record.characters:
            return []

        anime_id = record.anime.id
        documents: list[VectorDocument] = []

        # Prepare batch texts for efficient embedding
        texts_to_embed: list[str] = []
        valid_characters = []  # Keep track of which chars map to which text index

        for char in record.characters:
            # Construct rich text for character
            parts = [f"Character: {char.name}"]
            if char.role:
                parts.append(f"Role: {char.role}")
            if char.description:
                parts.append(f"Description: {char.description}")
            if char.character_traits:
                parts.append(f"Traits: {', '.join(char.character_traits)}")

            full_text = "\n".join(parts)
            texts_to_embed.append(full_text)
            valid_characters.append(char)

        # Batch Embed text
        embeddings_batch = self.text_processor.encode_texts_batch(texts_to_embed)

        # Create Documents with text vectors only (image vectors are separate Image Points)
        for i, char in enumerate(valid_characters):
            text_embedding = embeddings_batch[i]
            if not text_embedding:
                continue  # Skip if text embedding failed entirely

            # Determine ID
            char_id = char.id
            if not char_id:
                # Deterministic fallback: anime_id + char_name
                seed = f"{anime_id}_{char.name}"
                char_id = generate_deterministic_id(seed, "character")

            # Build vectors dict (text only)
            vectors: dict[str, list[float]] = {"text_vector": text_embedding}

            # Payload
            payload = char.model_dump(exclude_none=True)
            payload["type"] = "character"
            payload["anime_ids"] = [anime_id]  # Link to parent

            documents.append(
                VectorDocument(id=char_id, vectors=vectors, payload=payload)
            )

        return documents

    async def _create_episode_points(self, record: AnimeRecord) -> list[VectorDocument]:
        """Create Episode Points linked to this anime."""
        if not record.episodes:
            return []

        anime_id = record.anime.id
        documents: list[VectorDocument] = []

        # Prepare batch texts
        texts_to_embed: list[str] = []
        valid_episodes = []

        for ep in record.episodes:
            # Construct rich text
            parts = [f"Episode {ep.episode_number}: {ep.title}"]
            if ep.title_japanese:
                parts.append(f"Japanese Title: {ep.title_japanese}")
            if ep.synopsis:
                parts.append(f"Synopsis: {ep.synopsis}")
            if ep.description:
                parts.append(f"Description: {ep.description}")

            full_text = "\n".join(parts)
            texts_to_embed.append(full_text)
            valid_episodes.append(ep)

        # Batch Embed
        embeddings_batch = self.text_processor.encode_texts_batch(texts_to_embed)

        # Create Documents
        for i, ep in enumerate(valid_episodes):
            embedding = embeddings_batch[i]
            if not embedding:
                continue

            # Determine ID
            ep_id = ep.id
            if not ep_id:
                ep_id = generate_episode_id(anime_id, ep.episode_number)

            # Payload
            payload = ep.model_dump(exclude_none=True)
            payload["type"] = "episode"
            payload["anime_id"] = anime_id  # Parent link

            documents.append(
                VectorDocument(
                    id=ep_id, vectors={"text_vector": embedding}, payload=payload
                )
            )

        return documents

    def validate_vectors(self, vectors: dict[str, list[float]]) -> dict[str, Any]:
        """Refactored validation stub - mostly for legacy compatibility if needed."""
        # This method assumes dictionary input, but we now work with Documents.
        # We can implement a document validator or remove this.
        # For now, leaving a placeholder to avoid breaking callers explicitly importing it?
        # But this method was instance method.
        # Let's keep a simplified version matching new schema
        return {"valid": True, "note": "Validation defered to schema check"}

    def get_embedding_stats(self) -> dict[str, Any]:
        """Get embedding manager statistics."""
        try:
            return {
                "text_processor": self.text_processor.get_model_info(),
                "vision_processor": self.vision_processor.get_model_info(),
            }
        except Exception as e:
            return {"error": str(e)}
