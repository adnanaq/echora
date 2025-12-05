#!/usr/bin/env python3
"""
Generic vector update script for selective updates to Qdrant collection.

Supports updating any named vector(s) without touching other vectors.
Can be used both as a CLI tool and as an importable module.

Usage Examples:

    # CLI: Update single vector for all anime
    python scripts/update_vectors.py --vectors title_vector

    # CLI: Update multiple vectors
    python scripts/update_vectors.py --vectors title_vector episode_vector

    # CLI: Update specific anime by index
    python scripts/update_vectors.py --vectors title_vector --index 0

    # CLI: Update specific anime by title
    python scripts/update_vectors.py --vectors character_vector --title "Dandadan"

    # CLI: Batch size control
    python scripts/update_vectors.py --vectors title_vector --batch-size 50

    # Programmatic: Update vectors for all anime
    from scripts.update_vectors import update_vectors
    await update_vectors(["title_vector", "episode_vector"])

    # Programmatic: Update vectors for specific anime by index
    from scripts.update_vectors import update_vectors
    await update_vectors(["character_vector"], anime_index=0)

    # Programmatic: Update vectors for specific anime by title
    from scripts.update_vectors import update_vectors
    await update_vectors(["character_vector"], anime_title="Dandadan")
"""

import argparse
import asyncio
import json
import logging
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_settings
from src.models.anime import AnimeEntry
from src.vector.client.qdrant_client import QdrantClient
from src.vector.processors.embedding_manager import MultiVectorEmbeddingManager
from src.vector.processors.text_processor import TextProcessor
from src.vector.processors.vision_processor import VisionProcessor
from qdrant_client import AsyncQdrantClient

# Configure logging
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
    vector_names: List[str],
    anime_index: Optional[int] = None,
    anime_title: Optional[str] = None,
    batch_size: int = 100,
    data_file: str = "./data/qdrant_storage/enriched_anime_database.json",
) -> Dict[str, Any]:
    """Update specific vectors for anime entries.

    This function now handles vector generation and calls the low-level
    QdrantClient.update_batch_vectors() method.

    1. Loads and filters anime data from JSON file
    2. Validates input parameters
    3. Generates vectors using MultiVectorEmbeddingManager
    4. Prepares updates for Qdrant
    5. Calls QdrantClient.update_batch_vectors()
    6. Formats and logs results for CLI display

    Args:
        client: An initialized QdrantClient instance.
        embedding_manager: An initialized MultiVectorEmbeddingManager instance.
        vector_names: List of vector names to update
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
    with open(data_file, "r") as f:
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

    # Convert to AnimeEntry objects, adding UUIDs if missing
    anime_entries: List[AnimeEntry] = []
    for anime_dict in target_anime:
        if "id" not in anime_dict or not anime_dict["id"]:
            anime_dict["id"] = str(uuid.uuid4())
        anime_entries.append(AnimeEntry(**anime_dict))

    total_anime = len(anime_entries)
    num_batches = (total_anime + batch_size - 1) // batch_size  # Ceiling division
    logger.info(
        f"Updating {len(vector_names)} vector(s) across {total_anime} anime entries "
        f"in {num_batches} batch(es) of {batch_size}"
    )

    # Process anime in batches to control memory usage
    all_batch_results: List[Dict[str, Any]] = []
    all_generation_failures: List[Dict[str, Any]] = []

    for batch_num in range(num_batches):
        batch_start = batch_num * batch_size
        batch_end = min(batch_start + batch_size, total_anime)
        batch_anime = anime_entries[batch_start:batch_end]

        logger.info(
            f"Processing batch {batch_num + 1}/{num_batches}: "
            f"anime {batch_start + 1}-{batch_end} of {total_anime}"
        )

        # Generate vectors for this batch
        gen_results = await embedding_manager.process_anime_batch(batch_anime)

        # Validate generation results match batch length
        if not gen_results or len(gen_results) != len(batch_anime):
            raise VectorGenerationError(
                f"Vector generation failed for batch {batch_num + 1}: "
                f"expected {len(batch_anime)} results, got {len(gen_results) if gen_results else 0}"
            )

        # Prepare updates for this batch
        batch_updates: List[Dict[str, Any]] = []
        batch_generation_failures: List[Dict[str, Any]] = []

        for i, anime_entry in enumerate(batch_anime):
            gen_result = gen_results[i]
            vectors = gen_result.get("vectors", {})

            for vector_name in vector_names:
                vector_data = vectors.get(vector_name)
                if vector_data and len(vector_data) > 0:
                    batch_updates.append(
                        {
                            "anime_id": anime_entry.id,
                            "vector_name": vector_name,
                            "vector_data": vector_data,
                        }
                    )
                else:
                    batch_generation_failures.append(
                        {
                            "anime_id": anime_entry.id,
                            "vector_name": vector_name,
                            "error": "Vector generation failed or returned None",
                        }
                    )

        all_generation_failures.extend(batch_generation_failures)

        # Update Qdrant with this batch
        if batch_updates:
            batch_result = await client.update_batch_vectors(
                updates=batch_updates,
                dedup_policy="last-wins",
            )
            all_batch_results.append(batch_result)
            logger.info(
                f"Batch {batch_num + 1} complete: "
                f"{batch_result['success']} successful, {batch_result['failed']} failed"
            )
        else:
            logger.warning(f"Batch {batch_num + 1} had no valid updates to process")

    # Check if we got any results
    if not all_batch_results:
        raise VectorGenerationError("No vectors were successfully generated for update.")

    # Aggregate results from all batches
    total_successful = sum(r["success"] for r in all_batch_results)
    total_failed = sum(r["failed"] for r in all_batch_results)
    combined_results: List[Dict[str, Any]] = []
    for batch_result in all_batch_results:
        combined_results.extend(batch_result.get("results", []))

    # Create aggregated result
    result = {
        "success": total_successful,
        "failed": total_failed,
        "results": combined_results,
        "total_requested_updates": total_anime * len(vector_names),
        "generation_failures": len(all_generation_failures),
        "generation_failures_detail": all_generation_failures,
    }

    # Calculate per-vector statistics from results
    vector_stats = {v: {"success": 0, "failed": 0} for v in vector_names}
    for update_result in result["results"]:
        vector_name = update_result["vector_name"]
        if vector_name in vector_stats:
            if update_result["success"]:
                vector_stats[vector_name]["success"] += 1
            else:
                vector_stats[vector_name]["failed"] += 1

    # Calculate per-anime success (all vectors must succeed)
    anime_success_map: Dict[str, Dict[str, int]] = {}
    for update_result in result["results"]:
        anime_id = update_result["anime_id"]
        if anime_id not in anime_success_map:
            anime_success_map[anime_id] = {"total": 0, "success": 0}
        anime_success_map[anime_id]["total"] += 1
        if update_result["success"]:
            anime_success_map[anime_id]["success"] += 1

    successful_anime = sum(
        1
        for stats in anime_success_map.values()
        if stats["success"] == stats["total"]
    )
    failed_anime = total_anime - successful_anime

    # Log summary
    logger.info("=" * 80)
    logger.info("Update Summary")
    logger.info("=" * 80)
    logger.info(
        f"Anime Processing: {successful_anime}/{total_anime} successful "
        f"({successful_anime/total_anime*100:.1f}%), "
        f"{failed_anime} failed ({failed_anime/total_anime*100:.1f}%)"
    )

    logger.info(f"Vector Updates (Total: {result['total_requested_updates']}):")
    for vector_name in vector_names:
        stats = vector_stats[vector_name]
        total = stats["success"] + stats["failed"]
        success_rate = stats["success"] / total * 100 if total > 0 else 0
        logger.info(
            f"  {vector_name}: {stats['success']}/{total} ({success_rate:.1f}%)"
        )

    if result["generation_failures"] > 0:
        logger.warning(
            f"Generation Failures: {result['generation_failures']} vectors failed to generate"
        )

    return {
        "total_anime": total_anime,
        "successful_anime": successful_anime,
        "failed_anime": failed_anime,
        "total_updates": result["total_requested_updates"],
        "successful_updates": result["success"],
        "failed_updates": result["failed"],
        "generation_failures": result["generation_failures"],
        "vector_stats": vector_stats,
    }

async def async_main(args, settings):
    """Async main function that initializes clients and runs updates."""
    # Initialize AsyncQdrantClient
    if settings.qdrant_api_key:
        async_qdrant_client = AsyncQdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    else:
        async_qdrant_client = AsyncQdrantClient(url=settings.qdrant_url)

    try:
        # Initialize embedding manager and processors
        text_processor = TextProcessor(settings)
        vision_processor = VisionProcessor(settings)
        embedding_manager = MultiVectorEmbeddingManager(
            text_processor=text_processor,
            vision_processor=vision_processor,
            settings=settings
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
  python scripts/update_vectors.py --vectors genre_vector

  # Update multiple vectors
  python scripts/update_vectors.py --vectors genre_vector episode_vector

  # Update specific anime by index
  python scripts/update_vectors.py --vectors title_vector --index 0

  # Update specific anime by title
  python scripts/update_vectors.py --vectors character_vector --title "Dandadan"

  # Control batch size
  python scripts/update_vectors.py --vectors staff_vector --batch-size 50
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
        default="./data/qdrant_storage/enriched_anime_database.json",
        help="Path to enriched anime database JSON file",
    )

    args = parser.parse_args()

    # Run update with proper error handling
    try:
        result = asyncio.run(async_main(args, settings))

        # Exit with appropriate code
        if result.get("failed_anime", 0) > 0:
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
