"""Multi-Vector Embedding Manager for coordinated embedding generation.

This module coordinates the generation of all 11 vectors (9 text + 2 visual)
for the comprehensive anime search system with error handling and validation.
"""

import asyncio
import logging
from typing import Any

from common.config import Settings
from common.models.anime import AnimeEntry

from .anime_field_mapper import AnimeFieldMapper
from .text_processor import TextProcessor
from .vision_processor import VisionProcessor

logger = logging.getLogger(__name__)


class MultiVectorEmbeddingManager:
    """Manager for coordinated generation of all 11 embedding vectors."""

    def __init__(
        self,
        text_processor: TextProcessor,
        vision_processor: VisionProcessor,
        field_mapper: AnimeFieldMapper,
        settings: Settings | None = None,
    ):
        """Initialize the multi-vector embedding manager with injected processors.

        Args:
            text_processor: An initialized TextProcessor instance.
            vision_processor: An initialized VisionProcessor instance.
            field_mapper: Shared AnimeFieldMapper instance.
            settings: Configuration settings instance.
        """
        if settings is None:
            settings = Settings()

        self.settings = settings

        # Use injected processors
        self.text_processor = text_processor
        self.vision_processor = vision_processor

        # Initialize field mapper
        self.field_mapper = field_mapper

        # Get vector configuration
        self.vector_names = list(settings.vector_names.keys())
        self.text_vector_names = [
            v
            for v in self.vector_names
            if v not in ["image_vector", "character_image_vector"]
        ]
        self.image_vector_names = ["image_vector", "character_image_vector"]

        logger.info(
            f"Initialized MultiVectorEmbeddingManager with {len(self.vector_names)} vectors"
        )

    async def process_anime_vectors(self, anime: AnimeEntry) -> dict[str, Any]:
        """Process anime data to generate all embedding vectors.

        Args:
            anime: AnimeEntry instance with anime data

        Returns:
            Dictionary containing:
            - vectors: Dict with vector name -> embedding list mappings
            - payload: Dict with additional data for Qdrant storage
            - metadata: Dict with processing metadata
        """
        try:
            # Generate all text vectors
            text_vectors = await self._generate_text_vectors(anime)

            # Generate image vectors
            image_vectors = await self._generate_image_vectors(anime)

            # Combine all vectors
            all_vectors = {**text_vectors, **image_vectors}

            # Generate payload data
            payload = self._generate_payload(anime)

            # Generate processing metadata
            metadata = self._generate_metadata(all_vectors, anime)

            return {"vectors": all_vectors, "payload": payload, "metadata": metadata}

        except Exception as e:
            logger.error(f"Failed to process anime vectors for {anime.title}: {e}")
            return {
                "vectors": {},
                "payload": {},
                "metadata": {"error": str(e), "processing_failed": True},
            }

    async def _generate_text_vectors(self, anime: AnimeEntry) -> dict[str, list[float]]:
        """Generate all text-based embedding vectors.

        Args:
            anime: AnimeEntry instance

        Returns:
            Dictionary mapping text vector names to embeddings
        """
        try:
            # Use text processor's multi-vector method
            text_vectors = self.text_processor.process_anime_vectors(anime)

            # Filter out None vectors and log status
            valid_vectors = {}
            failed_vectors = []

            for vector_name in self.text_vector_names:
                if (
                    vector_name in text_vectors
                    and text_vectors[vector_name] is not None
                ):
                    valid_vectors[vector_name] = text_vectors[vector_name]
                else:
                    failed_vectors.append(vector_name)

            if failed_vectors:
                logger.warning(f"Failed to generate text vectors: {failed_vectors}")

            logger.debug(
                f"Generated {len(valid_vectors)}/{len(self.text_vector_names)} text vectors"
            )
            return valid_vectors

        except Exception as e:
            logger.error(f"Text vector generation failed: {e}")
            return {}

    async def _generate_image_vectors(
        self, anime: AnimeEntry
    ) -> dict[str, list[float]]:
        """Generate both image embedding vectors.

        Args:
            anime: AnimeEntry instance

        Returns:
            Dictionary mapping image vector names to embeddings
        """
        try:
            image_vectors = {}

            # Generate general image vector
            general_image_vector = (
                await self.vision_processor.process_anime_image_vector(anime)
            )
            if general_image_vector is not None:
                image_vectors["image_vector"] = general_image_vector
                logger.debug("Successfully generated general image vector")
            else:
                logger.debug(
                    "General image vector generation failed - will store URLs in payload"
                )

            # Generate character image vector
            character_image_vector = (
                await self.vision_processor.process_anime_character_image_vector(anime)
            )
            if character_image_vector is not None:
                image_vectors["character_image_vector"] = character_image_vector
                logger.debug("Successfully generated character image vector")
            else:
                logger.debug(
                    "Character image vector generation failed - will store URLs in payload"
                )

            logger.debug(f"Generated {len(image_vectors)}/2 image vectors")
            return image_vectors

        except Exception as e:
            logger.error(f"Image vector generation failed: {e}")
            return {}

    def _generate_payload(self, anime: AnimeEntry) -> dict[str, Any]:
        """Generate payload data for Qdrant storage.

        Stores ALL fields from AnimeEntry to preserve complete data.

        Args:
            anime: AnimeEntry instance

        Returns:
            Payload dictionary for Qdrant with ALL source fields
        """
        try:
            # Convert the entire AnimeEntry to dict to preserve all fields
            payload = anime.model_dump(exclude_none=True)

            # Ensure ID is always present
            if "id" not in payload or not payload["id"]:
                payload["id"] = anime.id

            # Convert complex nested objects to serializable format
            # Note: year and season are now direct fields, no conversion needed

            if anime.score:
                payload["score"] = {
                    "arithmetic_geometric_mean": anime.score.arithmetic_geometric_mean,
                    "arithmetic_mean": anime.score.arithmetic_mean,
                    "median": anime.score.median,
                }

            if anime.statistics:
                payload["statistics"] = {}
                for platform, stats in anime.statistics.items():
                    stats_dict: dict[str, Any] = {
                        "score": getattr(stats, "score", None),
                        "scored_by": getattr(stats, "scored_by", None),
                        "popularity_rank": getattr(stats, "popularity_rank", None),
                        "members": getattr(stats, "members", None),
                        "favorites": getattr(stats, "favorites", None),
                        "rank": getattr(stats, "rank", None),
                    }
                    if hasattr(stats, "contextual_ranks") and stats.contextual_ranks:
                        stats_dict["contextual_ranks"] = [
                            rank.model_dump() for rank in stats.contextual_ranks
                        ]
                    payload["statistics"][platform] = stats_dict

            # Convert complex array fields to serializable format
            if anime.characters:
                payload["characters"] = [char.model_dump() for char in anime.characters]

            if anime.episode_details:
                payload["episode_details"] = [
                    ep.model_dump() for ep in anime.episode_details
                ]

            if anime.themes:
                payload["themes"] = [theme.model_dump() for theme in anime.themes]

            if anime.trailers:
                payload["trailers"] = [
                    trailer.model_dump() for trailer in anime.trailers
                ]

            if anime.streaming_info:
                payload["streaming_info"] = [
                    stream.model_dump() for stream in anime.streaming_info
                ]

            if anime.related_anime:
                payload["related_anime"] = [
                    rel.model_dump() for rel in anime.related_anime
                ]

            if anime.relations:
                payload["relations"] = [rel.model_dump() for rel in anime.relations]

            if anime.opening_themes:
                payload["opening_themes"] = [
                    theme.model_dump() for theme in anime.opening_themes
                ]

            if anime.ending_themes:
                payload["ending_themes"] = [
                    theme.model_dump() for theme in anime.ending_themes
                ]

            # Convert object fields to serializable format
            if anime.aired_dates:
                payload["aired_dates"] = anime.aired_dates.model_dump()

            if anime.broadcast:
                payload["broadcast"] = anime.broadcast.model_dump()

            if anime.broadcast_schedule:
                payload["broadcast_schedule"] = anime.broadcast_schedule.model_dump()

            if anime.delay_information:
                payload["delay_information"] = anime.delay_information.model_dump()

            if anime.premiere_dates:
                payload["premiere_dates"] = anime.premiere_dates.model_dump()

            if anime.staff_data:
                payload["staff_data"] = anime.staff_data.model_dump()

            if anime.enrichment_metadata:
                payload["enrichment_metadata"] = anime.enrichment_metadata.model_dump()

            # Add flattened fields for optimized indexing
            self._add_flattened_fields(payload, anime)

            return payload

        except Exception as e:
            logger.error(f"Payload generation failed: {e}")
            return {"error": str(e)}

    def _add_flattened_fields(self, payload: dict[str, Any], anime: AnimeEntry) -> None:
        """Add flattened fields for optimized Qdrant indexing.

        Creates flattened fields that enable efficient filtering:
        - year/season for temporal filtering (already flattened in model)
        - score.median for numerical score filtering
        - title_text for full-text title search

        Args:
            payload: Existing payload dictionary to modify
            anime: AnimeEntry with source data
        """
        try:
            # Flatten score for numerical filtering
            if anime.score and anime.score.median is not None:
                payload["score.median"] = anime.score.median

            # Add title_text field for full-text search (duplicate of title)
            if anime.title:
                payload["title_text"] = anime.title

            logger.debug("Added flattened fields for optimized indexing")

        except Exception as e:
            logger.warning(f"Failed to add flattened fields: {e}")
            # Don't fail payload generation if flattening fails

    def _generate_metadata(
        self, vectors: dict[str, list[float]], anime: AnimeEntry
    ) -> dict[str, Any]:
        """Generate processing metadata.

        Args:
            vectors: Generated vectors dictionary
            anime: AnimeEntry instance

        Returns:
            Processing metadata dictionary
        """
        try:
            # Vector generation statistics
            total_expected = len(self.vector_names)
            total_generated = len(vectors)
            text_vectors_generated = len(
                [v for v in vectors.keys() if v not in self.image_vector_names]
            )
            image_vectors_generated = len(
                [v for v in vectors.keys() if v in self.image_vector_names]
            )

            # Missing vectors
            missing_vectors = [v for v in self.vector_names if v not in vectors]

            metadata: dict[str, Any] = {
                "processing_timestamp": None,  # Will be set by caller
                "total_vectors_expected": total_expected,
                "total_vectors_generated": total_generated,
                "text_vectors_generated": text_vectors_generated,
                "image_vectors_generated": image_vectors_generated,
                "missing_vectors": missing_vectors,
                "success_rate": (
                    total_generated / total_expected if total_expected > 0 else 0.0
                ),
                "anime_id": anime.id,
                "anime_title": anime.title,
                "processing_complete": len(missing_vectors) == 0,
            }

            # Add vector dimensions for validation
            vector_dimensions: dict[str, int] = {}
            for vector_name, vector_data in vectors.items():
                if vector_data:
                    vector_dimensions[vector_name] = len(vector_data)

            metadata["vector_dimensions"] = vector_dimensions

            return metadata

        except Exception as e:
            logger.error(f"Metadata generation failed: {e}")
            return {"error": str(e)}

    async def process_anime_batch(
        self, anime_list: list[AnimeEntry]
    ) -> list[dict[str, Any]]:
        """Process multiple anime entries in batch.

        Args:
            anime_list: List of AnimeEntry instances

        Returns:
            List of processing results for each anime
        """
        try:
            logger.info(f"Processing batch of {len(anime_list)} anime")

            # Process all anime concurrently
            tasks = [self.process_anime_vectors(anime) for anime in anime_list]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle any exceptions in results
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Batch processing error for anime {i}: {result}")
                    error_result: dict[str, Any] = {
                        "vectors": {},
                        "payload": {},
                        "metadata": {
                            "error": str(result),
                            "processing_failed": True,
                        },
                    }
                    processed_results.append(error_result)
                else:
                    # result is not an Exception, it's a Dict[str, Any]
                    processed_results.append(result)

            # Log batch statistics
            successful = sum(
                1
                for r in processed_results
                if not r["metadata"].get("processing_failed", False)
            )
            logger.info(
                f"Batch processing complete: {successful}/{len(anime_list)} successful"
            )

            return processed_results

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            return []

    def validate_vectors(self, vectors: dict[str, list[float]]) -> dict[str, Any]:
        """Validate generated vectors.

        Args:
            vectors: Dictionary of vector name -> embedding mappings

        Returns:
            Validation report dictionary
        """
        try:
            validation_report: dict[str, Any] = {
                "valid": True,
                "errors": [],
                "warnings": [],
                "vector_stats": {},
            }

            expected_dimensions = self.settings.vector_names

            for vector_name, vector_data in vectors.items():
                if vector_name not in expected_dimensions:
                    validation_report["errors"].append(
                        f"Unexpected vector: {vector_name}"
                    )
                    validation_report["valid"] = False
                    continue

                expected_dim = expected_dimensions[vector_name]
                actual_dim = len(vector_data) if vector_data else 0

                if actual_dim != expected_dim:
                    validation_report["errors"].append(
                        f"{vector_name}: dimension mismatch (expected {expected_dim}, got {actual_dim})"
                    )
                    validation_report["valid"] = False

                # Check for invalid values
                if vector_data and any(
                    not isinstance(x, int | float) for x in vector_data
                ):
                    validation_report["errors"].append(
                        f"{vector_name}: contains non-numeric values"
                    )
                    validation_report["valid"] = False

                # Statistics
                validation_report["vector_stats"][vector_name] = {
                    "dimension": actual_dim,
                    "non_zero_values": (
                        sum(1 for x in vector_data if x != 0) if vector_data else 0
                    ),
                    "magnitude": (
                        sum(x * x for x in vector_data) ** 0.5 if vector_data else 0.0
                    ),
                }

            # Check for missing critical vectors
            missing_critical = []
            critical_vectors = self.settings.vector_priorities.get("high", [])
            for critical_vector in critical_vectors:
                if critical_vector not in vectors:
                    missing_critical.append(critical_vector)

            if missing_critical:
                validation_report["warnings"].append(
                    f"Missing critical vectors: {missing_critical}"
                )

            return validation_report

        except Exception as e:
            logger.error(f"Vector validation failed: {e}")
            return {"valid": False, "errors": [str(e)]}

    def get_embedding_stats(self) -> dict[str, Any]:
        """Get embedding manager statistics.

        Returns:
            Statistics dictionary
        """
        try:
            text_model_info = self.text_processor.get_model_info()
            vision_model_info = self.vision_processor.get_model_info()

            return {
                "text_processor": text_model_info,
                "vision_processor": vision_model_info,
                "vector_configuration": {
                    "total_vectors": len(self.vector_names),
                    "text_vectors": len(self.text_vector_names),
                    "image_vectors": len(self.image_vector_names),
                    "vector_names": self.vector_names,
                    "vector_dimensions": dict(self.settings.vector_names),
                    "vector_priorities": dict(self.settings.vector_priorities),
                },
                "cache_stats": {"image_cache": self.vision_processor.get_cache_stats()},
            }

        except Exception as e:
            logger.error(f"Failed to get embedding stats: {e}")
            return {"error": str(e)}
