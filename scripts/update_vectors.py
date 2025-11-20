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
from qdrant_client import QdrantClient as QdrantSDK

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
    logger.info(
        f"Updating {len(vector_names)} vector(s) across {total_anime} anime entries "
        f"(batch size: {batch_size})"
    )

    # Generate vectors for the target anime
    logger.info(f"Generating vectors for {len(anime_entries)} anime entries...")
    gen_results = await embedding_manager.process_anime_batch(anime_entries)

    # Prepare updates for Qdrant
    updates: List[Dict[str, Any]] = []
    generation_failures: List[Dict[str, Any]] = []

    for i, anime_entry in enumerate(anime_entries):
        gen_result = gen_results[i]
        vectors = gen_result.get("vectors", {})

        for vector_name in vector_names:
            vector_data = vectors.get(vector_name)
            if vector_data and len(vector_data) > 0:
                updates.append(
                    {
                        "anime_id": anime_entry.id,
                        "vector_name": vector_name,
                        "vector_data": vector_data,
                    }
                )
            else:
                generation_failures.append(
                    {
                        "anime_id": anime_entry.id,
                        "vector_name": vector_name,
                        "error": "Vector generation failed or returned None",
                    }
                )
    
    if not updates:
        raise VectorGenerationError("No vectors were successfully generated for update.")

    logger.info(f"Prepared {len(updates)} vector updates.")

    # Call the low-level update method
    result = await client.update_batch_vectors(
        updates=updates,
        dedup_policy="last-wins", # Default policy
    )

    # Add generation failures to the result for comprehensive reporting
    result["generation_failures"] = len(generation_failures)
    result["generation_failures_detail"] = generation_failures
    result["total_requested_updates"] = total_anime * len(vector_names)

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
        "successful_updates": result["successful_updates"],
        "failed_updates": result["failed_updates"],
        "generation_failures": result["generation_failures"],
        "vector_stats": vector_stats,
    }

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

    # Initialize Qdrant SDK client
    if settings.qdrant_api_key:
        qdrant_sdk_client = QdrantSDK(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    else:
        qdrant_sdk_client = QdrantSDK(url=settings.qdrant_url)

    # Initialize embedding manager and processors
    text_processor = TextProcessor(settings)
    vision_processor = VisionProcessor(settings)
    embedding_manager = MultiVectorEmbeddingManager(
        text_processor=text_processor,
        vision_processor=vision_processor,
        settings=settings
    )

    # Initialize Qdrant client
    qdrant_client = QdrantClient(
        settings=settings,
        qdrant_sdk_client=qdrant_sdk_client,
        url=settings.qdrant_url,
        collection_name=settings.qdrant_collection_name,
    )

    # Run update with proper error handling
    try:
        result = asyncio.run(
            update_vectors(
                client=qdrant_client, # Pass the injected client
                embedding_manager=embedding_manager, # Pass the embedding manager
                vector_names=args.vectors,
                anime_index=args.index,
                anime_title=args.title,
                batch_size=args.batch_size,
                data_file=args.file,
            )
        )

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
