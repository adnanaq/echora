#!/usr/bin/env python3
"""
Generic vector update script for selective updates to Qdrant collection.

Supports updating any named vector(s) without touching other vectors.
Can be used both as a CLI tool and as an importable module.

Usage Examples:

    # CLI: Update single vector for all anime
    python scripts/update_vectors.py --vectors text_vector

    # CLI: Update multiple vectors
    python scripts/update_vectors.py --vectors text_vector image_vector

    # CLI: Update specific anime by index
    python scripts/update_vectors.py --vectors text_vector --index 0

    # CLI: Update specific anime by title
    python scripts/update_vectors.py --vectors text_vector --title "Dandadan"

    # CLI: Batch size control
    python scripts/update_vectors.py --vectors text_vector --batch-size 50

    # Programmatic: Update vectors for all anime
    from scripts.update_vectors import update_vectors
    await update_vectors(["text_vector"])

    # Programmatic: Update vectors for specific anime by index
    from scripts.update_vectors import update_vectors
    await update_vectors(["text_vector"], anime_index=0)

    # Programmatic: Update vectors for specific anime by title
    from scripts.update_vectors import update_vectors
    await update_vectors(["text_vector"], anime_title="Dandadan")
"""

import argparse
import asyncio
import json
import logging
import sys
import uuid
from pathlib import Path
from typing import Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.config import get_settings
from common.models.anime import AnimeRecord
from pydantic import ValidationError
from qdrant_client import AsyncQdrantClient
from qdrant_db.client import QdrantClient
from vector_processing.embedding_models.factory import EmbeddingModelFactory
from vector_processing.processors.anime_field_mapper import AnimeFieldMapper
from vector_processing.processors.embedding_manager import MultiVectorEmbeddingManager
from vector_processing.processors.text_processor import TextProcessor
from vector_processing.processors.vision_processor import VisionProcessor
from vector_processing.utils.image_downloader import ImageDownloader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class VectorUpdateError(Exception):
    """Base exception for vector update operations."""

    pass


class InvalidVectorNameError(VectorUpdateError):
    """Raised when an invalid vector name is provided."""

    pass


class AnimeNotFoundError(VectorUpdateError):
    """Raised when specified anime cannot be found."""

    pass


class VectorGenerationError(VectorUpdateError):
    """Raised when vector data generation fails."""

    pass


async def update_vectors(
    client: QdrantClient,
    embedding_manager: MultiVectorEmbeddingManager,
    vector_names: list[str],
    anime_index: int | None = None,
    anime_title: str | None = None,
    batch_size: int = 100,
    data_file: str = "./assets/seed_data/anime_database.json",
) -> dict[str, Any]:
    """Regenerate and upsert vectors for anime entries.

    NOTE: In the current hierarchical architecture, this performs a FULL upsert
    of all vectors (text + image) for the specified records. Selective single-vector
    updates are not supported because the embedding manager produces complete
    VectorDocuments containing all vector types.

    The vector_names parameter is validated against the configuration but currently
    serves as a sanity check that the requested vectors exist - the actual operation
    regenerates all vectors for the targeted anime entries.

    Process:
    1. Loads and filters anime data from JSON file
    2. Validates vector_names against configuration
    3. Generates ALL vectors using MultiVectorEmbeddingManager
    4. Upserts complete VectorDocuments to Qdrant
    5. Formats and logs results for CLI display

    Args:
        client: An initialized QdrantClient instance.
        embedding_manager: An initialized MultiVectorEmbeddingManager instance.
        vector_names: List of vector names to validate (full upsert performed regardless)
        anime_index: Optional index of specific anime to update
        anime_title: Optional title to search for specific anime
        batch_size: Number of anime to process per batch
        data_file: Path to enriched anime database

    Returns:
        Dictionary with update statistics

    Raises:
        InvalidVectorNameError: If any vector name is invalid
        AnimeNotFoundError: If specified anime cannot be found
        FileNotFoundError: If data file does not exist
        VectorGenerationError: If vector generation fails
    """
    logger.info(f"Starting vector update for: {', '.join(vector_names)}")

    # Load settings (still needed for vector_names validation)
    settings = get_settings()

    # Validate vector names against settings.vector_names
    valid_vectors = list(settings.vector_names.keys())
    invalid_vectors = [v for v in vector_names if v not in valid_vectors]
    if invalid_vectors:
        raise InvalidVectorNameError(
            f"Invalid vector names: {', '.join(invalid_vectors)}. "
            f"Valid vectors: {', '.join(valid_vectors)}"
        )

    # Load anime data
    data_path = Path(data_file)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    logger.info(f"Loading anime data from {data_file}")
    with open(data_file) as f:
        enrichment_data = json.load(f)

    anime_data = enrichment_data["data"]
    logger.info(f"Loaded {len(anime_data)} anime entries")

    # Filter to specific anime if requested
    target_anime = []

    if anime_index is not None:
        if not (0 <= anime_index < len(anime_data)):
            raise AnimeNotFoundError(
                f"Invalid index {anime_index} (valid range: 0-{len(anime_data) - 1})"
            )
        target_anime = [anime_data[anime_index]]
        logger.info(
            f"Targeting anime at index {anime_index}: {target_anime[0].get('title', 'Unknown')}"
        )

    elif anime_title is not None:
        search_title = anime_title.lower()
        for anime_dict in anime_data:
            if search_title in anime_dict.get("title", "").lower():
                target_anime.append(anime_dict)

        if not target_anime:
            raise AnimeNotFoundError(f"No anime found matching title: '{anime_title}'")

        if len(target_anime) == 1:
            logger.info(f"Found anime: {target_anime[0]['title']}")
        else:
            logger.info(f"Found {len(target_anime)} matching anime")

    else:
        target_anime = anime_data
        logger.info(f"Processing all {len(target_anime)} anime")

    # Convert to AnimeRecord objects, adding UUIDs to anime.id if missing
    records: list[AnimeRecord] = []
    for i, anime_dict in enumerate(target_anime):
        try:
            if "id" not in anime_dict["anime"] or not anime_dict["anime"]["id"]:
                anime_dict["anime"]["id"] = str(uuid.uuid4())
            records.append(AnimeRecord(**anime_dict))
        except (KeyError, ValidationError) as e:
            logger.warning(f"Skipping malformed record {i}: {e}")
            continue

    total_anime = len(records)
    num_batches = (total_anime + batch_size - 1) // batch_size  # Ceiling division
    logger.info(
        f"Updating {len(vector_names)} vector(s) across {total_anime} anime entries "
        f"in {num_batches} batch(es) of {batch_size}"
    )

    # Process anime in batches to control memory usage
    all_batch_results: list[dict[str, Any]] = []
    all_generation_failures: list[dict[str, Any]] = []

    for batch_num in range(num_batches):
        batch_start = batch_num * batch_size
        batch_end = min(batch_start + batch_size, total_anime)
        batch_anime = records[batch_start:batch_end]

        logger.info(
            f"Processing batch {batch_num + 1}/{num_batches}: "
            f"anime {batch_start + 1}-{batch_end} of {total_anime}"
        )

        # Generate vectors for this batch
        # Note: In the new architecture, this regenerates ALL vectors (text+image)
        # because the Manager produces complete VectorDocuments.
        # This is a full upsert of the records.
        gen_results = await embedding_manager.process_anime_batch(batch_anime)

        # Count anime-level success by checking which anime actually generated documents
        anime_doc_ids = {
            doc.id for doc in gen_results if doc.payload.get("type") == "anime"
        }
        successful_batch = len(anime_doc_ids)
        failed_batch = len(batch_anime) - successful_batch

        # Track which specific anime failed to generate
        if failed_batch:
            all_generation_failures.extend(
                {"anime_id": rec.anime.id, "title": rec.anime.title}
                for rec in batch_anime
                if rec.anime.id not in anime_doc_ids
            )

        # Update Qdrant with this batch (Using Upsert/add_documents instead of partial update)
        if gen_results:
            # gen_results is list[VectorDocument]
            result = await client.add_documents(gen_results, batch_size=batch_size)

            # Adapt result format to match expected loop output
            # add_documents returns {"success": bool, "points_count": int, ...}
            if result["success"]:
                all_batch_results.append(
                    {
                        "success": successful_batch,
                        "failed": failed_batch,
                        "results": [],  # We lose granular per-vector results in add_documents
                    }
                )
                logger.info(
                    f"Batch {batch_num + 1} complete: Upserted {len(gen_results)} points "
                    f"({successful_batch} anime, {failed_batch} failed)."
                )
            else:
                error_msg = result.get("error", "Unknown error")
                logger.error(f"Batch {batch_num + 1} failed to upsert: {error_msg}")
                all_batch_results.append({"success": 0, "failed": len(batch_anime)})
        else:
            logger.warning(f"Batch {batch_num + 1} generated no documents.")
            all_batch_results.append({"success": 0, "failed": len(batch_anime)})

    # Calculate batch success from all_batch_results
    successful_count = sum(r.get("success", 0) for r in all_batch_results)
    failed_count = sum(r.get("failed", 0) for r in all_batch_results)

    # Log summary
    logger.info("=" * 80)
    logger.info("Update Summary")
    logger.info("=" * 80)
    logger.info(
        f"Records Processed: {successful_count} successful, {failed_count} failed"
    )
    logger.info(f"Vectors Updated: {vector_names}")

    if len(all_generation_failures) > 0:
        logger.warning(
            f"Generation Failures: {len(all_generation_failures)} anime failed to generate"
        )
        for failure in all_generation_failures[:10]:  # Show first 10
            logger.warning(f"  - {failure['title']} (ID: {failure['anime_id']})")
        if len(all_generation_failures) > 10:
            logger.warning(f"  ... and {len(all_generation_failures) - 10} more")

    return {
        "total_anime": total_anime,
        "successful_count": successful_count,
        "failed_count": failed_count,
        "generation_failures": len(all_generation_failures),
    }


async def async_main(args, settings):
    """Async main function that initializes clients and runs updates."""
    # Initialize AsyncQdrantClient
    if settings.qdrant_api_key:
        async_qdrant_client = AsyncQdrantClient(
            url=settings.qdrant_url, api_key=settings.qdrant_api_key
        )
    else:
        async_qdrant_client = AsyncQdrantClient(url=settings.qdrant_url)

    try:
        # Initialize embedding manager and processors using factory pattern
        # 1. Initialize Mapper
        field_mapper = AnimeFieldMapper()

        # 2. Initialize Processors
        text_model = EmbeddingModelFactory.create_text_model(settings)
        text_processor = TextProcessor(model=text_model, settings=settings)

        vision_model = EmbeddingModelFactory.create_vision_model(settings)
        image_downloader = ImageDownloader(
            cache_dir=settings.model_cache_dir or "cache"
        )
        vision_processor = VisionProcessor(
            model=vision_model,
            downloader=image_downloader,
            # field_mapper removed from vision processor
            settings=settings,
        )

        # 3. Initialize Manager
        embedding_manager = MultiVectorEmbeddingManager(
            text_processor=text_processor,
            vision_processor=vision_processor,
            field_mapper=field_mapper,
            settings=settings,
        )

        # Initialize Qdrant client using async factory
        qdrant_client = await QdrantClient.create(
            settings=settings,
            async_qdrant_client=async_qdrant_client,
            url=settings.qdrant_url,
            collection_name=settings.qdrant_collection_name,
        )

        # Run update and return result
        return await update_vectors(
            client=qdrant_client,
            embedding_manager=embedding_manager,
            vector_names=args.vectors,
            anime_index=args.index,
            anime_title=args.title,
            batch_size=args.batch_size,
            data_file=args.file,
        )
    finally:
        await async_qdrant_client.close()


def main():
    """CLI entry point."""
    # Load settings to get valid vector names for help text and client initialization
    settings = get_settings()
    valid_vectors = list(settings.vector_names.keys())

    parser = argparse.ArgumentParser(
        description="Generic vector update script for selective Qdrant updates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Update single vector for all anime
  python scripts/update_vectors.py --vectors text_vector

  # Update multiple vectors
  python scripts/update_vectors.py --vectors text_vector image_vector

  # Update specific anime by index
  python scripts/update_vectors.py --vectors text_vector --index 0

  # Update specific anime by title
  python scripts/update_vectors.py --vectors text_vector --title "Dandadan"

  # Control batch size
  python scripts/update_vectors.py --vectors text_vector --batch-size 50
        """,
    )

    parser.add_argument(
        "--vectors",
        nargs="+",
        required=True,
        help=f"Vector name(s) to update. Valid: {', '.join(valid_vectors)}",
    )
    parser.add_argument(
        "--index", type=int, help="Process anime at specific index (0-based)"
    )
    parser.add_argument(
        "--title", type=str, help="Search for anime by title (partial match)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of anime to process per batch (default: 100)",
    )
    parser.add_argument(
        "--file",
        type=str,
        default="./assets/seed_data/anime_database.json",
        help="Path to enriched anime database JSON file",
    )

    args = parser.parse_args()

    # Run update with proper error handling
    try:
        result = asyncio.run(async_main(args, settings))

        # Exit with appropriate code
        if result.get("failed_count", 0) > 0:
            logger.warning("Update completed with failures")
            sys.exit(2)  # Partial failure
        else:
            logger.info("Update completed successfully")
            sys.exit(0)  # Complete success

    except (InvalidVectorNameError, AnimeNotFoundError, FileNotFoundError) as e:
        logger.error(f"Validation error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Update interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
