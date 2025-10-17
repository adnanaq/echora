#!/usr/bin/env python3
"""
Generic vector update script for selective updates to Qdrant collection.

Supports updating any named vector(s) without touching other vectors.
Can be used both as a CLI tool and as an importable module.

Usage Examples:

    # CLI: Update single vector for all anime
    python scripts/update_vectors.py --vectors review_vector

    # CLI: Update multiple vectors
    python scripts/update_vectors.py --vectors review_vector episode_vector

    # CLI: Update specific anime by index
    python scripts/update_vectors.py --vectors title_vector --index 0

    # CLI: Update specific anime by title
    python scripts/update_vectors.py --vectors character_vector --title "Dandadan"

    # CLI: Batch size control
    python scripts/update_vectors.py --vectors review_vector --batch-size 50

    # Programmatic: Update specific vectors
    from scripts.update_vectors import update_vectors
    await update_vectors(["review_vector", "episode_vector"])

    # Programmatic: Update single anime
    from scripts.update_vectors import update_single_anime_vectors
    await update_single_anime_vectors(anime_entry, ["character_vector"])
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_settings
from src.models.anime import AnimeEntry
from src.vector.client.qdrant_client import QdrantClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
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


# Valid vector names in the system
VALID_VECTORS = [
    "title_vector",
    "character_vector",
    "genre_vector",
    "staff_vector",
    "temporal_vector",
    "streaming_vector",
    "related_vector",
    "franchise_vector",
    "episode_vector",
    "image_vector",
    "character_image_vector",
    # Future: "review_vector" when implemented
]


async def update_single_anime_vectors(
    anime_entry: AnimeEntry,
    vector_names: List[str],
    client: QdrantClient,
) -> Dict[str, bool]:
    """Update specific vectors for a single anime entry.

    Args:
        anime_entry: AnimeEntry object to process
        vector_names: List of vector names to update
        client: QdrantClient instance

    Returns:
        Dictionary mapping vector names to success status

    Raises:
        InvalidVectorNameError: If any vector name is not valid
    """
    # Validate vector names first
    invalid_vectors = [v for v in vector_names if v not in VALID_VECTORS]
    if invalid_vectors:
        raise InvalidVectorNameError(
            f"Invalid vector names: {', '.join(invalid_vectors)}. "
            f"Valid vectors: {', '.join(VALID_VECTORS)}"
        )

    results = {}

    for vector_name in vector_names:
        try:
            vector_data = await _generate_vector_data(anime_entry, vector_name, client)

            if not vector_data:
                logger.warning(
                    f"No vector data generated for {vector_name} (anime: {anime_entry.title})"
                )
                results[vector_name] = False
                continue

            success = await client.update_single_vector(
                anime_id=anime_entry.id,
                vector_name=vector_name,
                vector_data=vector_data,
            )
            results[vector_name] = success

            if success:
                logger.debug(f"Updated {vector_name} for {anime_entry.title}")
            else:
                logger.error(f"Failed to update {vector_name} for {anime_entry.title}")

        except Exception as e:
            logger.error(
                f"Error updating {vector_name} for {anime_entry.title}: {e}",
                exc_info=True
            )
            results[vector_name] = False

    return results


async def _generate_vector_data(
    anime_entry: AnimeEntry, vector_name: str, client: QdrantClient
) -> Optional[List[float]]:
    """Generate vector data for a specific vector type.

    Args:
        anime_entry: AnimeEntry object
        vector_name: Name of the vector to generate
        client: QdrantClient instance

    Returns:
        Vector data as list of floats, or None if generation failed
    """
    # Text vectors - use field mapper extractors
    if vector_name == "title_vector":
        content = client.embedding_manager.field_mapper._extract_title_content(
            anime_entry
        )
        return client.embedding_manager.text_processor.encode_text(content)

    elif vector_name == "character_vector":
        content = client.embedding_manager.field_mapper._extract_character_content(
            anime_entry
        )
        return client.embedding_manager.text_processor.encode_text(content)

    elif vector_name == "genre_vector":
        content = client.embedding_manager.field_mapper._extract_genre_content(
            anime_entry
        )
        return client.embedding_manager.text_processor.encode_text(content)

    elif vector_name == "staff_vector":
        content = client.embedding_manager.field_mapper._extract_staff_content(
            anime_entry
        )
        return client.embedding_manager.text_processor.encode_text(content)

    elif vector_name == "temporal_vector":
        content = client.embedding_manager.field_mapper._extract_temporal_content(
            anime_entry
        )
        return client.embedding_manager.text_processor.encode_text(content)

    elif vector_name == "streaming_vector":
        content = client.embedding_manager.field_mapper._extract_streaming_content(
            anime_entry
        )
        return client.embedding_manager.text_processor.encode_text(content)

    elif vector_name == "related_vector":
        content = client.embedding_manager.field_mapper._extract_related_content(
            anime_entry
        )
        return client.embedding_manager.text_processor.encode_text(content)

    elif vector_name == "franchise_vector":
        content = client.embedding_manager.field_mapper._extract_franchise_content(
            anime_entry
        )
        return client.embedding_manager.text_processor.encode_text(content)

    elif vector_name == "episode_vector":
        content = client.embedding_manager.field_mapper._extract_episode_content(
            anime_entry
        )
        return client.embedding_manager.text_processor.encode_text(content)

    # Image vectors
    elif vector_name == "image_vector":
        # Extract image URLs from anime entry
        image_urls = []
        if hasattr(anime_entry, "images") and anime_entry.images:
            if hasattr(anime_entry.images, "cover_image"):
                image_urls.append(anime_entry.images.cover_image)
            if hasattr(anime_entry.images, "poster_image"):
                image_urls.append(anime_entry.images.poster_image)
            if hasattr(anime_entry.images, "banner_image"):
                image_urls.append(anime_entry.images.banner_image)

        if image_urls:
            return await client.embedding_manager.vision_processor.encode_images(
                image_urls
            )
        return None

    elif vector_name == "character_image_vector":
        # Extract character image URLs
        image_urls = []
        if hasattr(anime_entry, "characters") and anime_entry.characters:
            for character in anime_entry.characters:
                if hasattr(character, "image_url") and character.image_url:
                    image_urls.append(character.image_url)

        if image_urls:
            return await client.embedding_manager.vision_processor.encode_images(
                image_urls
            )
        return None

    else:
        raise InvalidVectorNameError(f"Unknown vector type: {vector_name}")


async def update_vectors(
    vector_names: List[str],
    anime_index: Optional[int] = None,
    anime_title: Optional[str] = None,
    batch_size: int = 100,
    data_file: str = "./data/qdrant_storage/enriched_anime_database.json",
) -> Dict[str, Any]:
    """Update specific vectors for anime entries.

    Args:
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
    """
    logger.info(f"Starting vector update for: {', '.join(vector_names)}")

    # Validate vector names
    invalid_vectors = [v for v in vector_names if v not in VALID_VECTORS]
    if invalid_vectors:
        raise InvalidVectorNameError(
            f"Invalid vector names: {', '.join(invalid_vectors)}. "
            f"Valid vectors: {', '.join(VALID_VECTORS)}"
        )

    # Load settings and client
    settings = get_settings()
    client = QdrantClient(settings=settings)

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
        logger.info(f"Targeting anime at index {anime_index}: {target_anime[0].get('title', 'Unknown')}")

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

    # Statistics tracking
    total_anime = len(target_anime)
    total_updates = total_anime * len(vector_names)
    successful_anime = 0
    failed_anime = 0
    vector_stats = {v: {"success": 0, "failed": 0} for v in vector_names}

    # Process anime entries
    logger.info(f"Updating {len(vector_names)} vector(s) across {total_anime} anime entries")

    for i, anime_dict in enumerate(target_anime, 1):
        try:
            anime_entry = AnimeEntry(**anime_dict)
            logger.info(f"[{i}/{total_anime}] Processing: {anime_entry.title}")

            results = await update_single_anime_vectors(
                anime_entry, vector_names, client
            )

            # Track results
            anime_success = all(results.values())
            if anime_success:
                successful_anime += 1
            else:
                failed_anime += 1
                failed_vectors = [v for v, s in results.items() if not s]
                logger.warning(
                    f"Partial update failure for {anime_entry.title}: "
                    f"failed vectors: {', '.join(failed_vectors)}"
                )

            # Update per-vector statistics
            for vector_name, success in results.items():
                if success:
                    vector_stats[vector_name]["success"] += 1
                else:
                    vector_stats[vector_name]["failed"] += 1

        except Exception as e:
            failed_anime += 1
            anime_title = anime_dict.get('title', 'Unknown')
            logger.error(f"Failed to process {anime_title}: {e}", exc_info=True)
            # Mark all vectors as failed for this anime
            for vector_name in vector_names:
                vector_stats[vector_name]["failed"] += 1

    # Log summary
    logger.info("=" * 80)
    logger.info("Update Summary")
    logger.info("=" * 80)
    logger.info(f"Anime Processing: {successful_anime}/{total_anime} successful "
                f"({successful_anime/total_anime*100:.1f}%), "
                f"{failed_anime} failed ({failed_anime/total_anime*100:.1f}%)")

    logger.info(f"Vector Updates (Total: {total_updates}):")
    for vector_name in vector_names:
        stats = vector_stats[vector_name]
        total = stats["success"] + stats["failed"]
        success_rate = stats["success"] / total * 100 if total > 0 else 0
        logger.info(f"  {vector_name}: {stats['success']}/{total} ({success_rate:.1f}%)")

    return {
        "total_anime": total_anime,
        "successful_anime": successful_anime,
        "failed_anime": failed_anime,
        "total_updates": total_updates,
        "vector_stats": vector_stats,
    }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generic vector update script for selective Qdrant updates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Update single vector for all anime
  python scripts/update_vectors.py --vectors review_vector

  # Update multiple vectors
  python scripts/update_vectors.py --vectors review_vector episode_vector

  # Update specific anime by index
  python scripts/update_vectors.py --vectors title_vector --index 0

  # Update specific anime by title
  python scripts/update_vectors.py --vectors character_vector --title "Dandadan"

  # Control batch size
  python scripts/update_vectors.py --vectors review_vector --batch-size 50
        """,
    )

    parser.add_argument(
        "--vectors",
        nargs="+",
        required=True,
        help=f"Vector name(s) to update. Valid: {', '.join(VALID_VECTORS)}",
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
        result = asyncio.run(
            update_vectors(
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
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
