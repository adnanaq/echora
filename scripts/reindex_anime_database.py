#!/usr/bin/env python3
"""
Simple reindexing script for anime database with proper vector generation.

Uses existing infrastructure:
- QdrantClient.add_documents()
- MultiVectorEmbeddingManager
- AnimeFieldMapper
- TextProcessor and VisionProcessor
"""

import argparse
import asyncio
import json
import os
import uuid

os.environ["CUDA_VISIBLE_DEVICES"] = ""
from common.config import get_settings
from common.models.anime import AnimeRecord
from qdrant_client import AsyncQdrantClient
from qdrant_db.client import QdrantClient
from vector_db_interface import VectorDocument
from vector_processing.embedding_models.factory import EmbeddingModelFactory
from vector_processing.processors.anime_field_mapper import AnimeFieldMapper
from vector_processing.processors.embedding_manager import MultiVectorEmbeddingManager
from vector_processing.processors.text_processor import TextProcessor
from vector_processing.processors.vision_processor import VisionProcessor
from vector_processing.utils.image_downloader import ImageDownloader


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Reindex anime database with vector embeddings"
    )
    parser.add_argument(
        "--data-file",
        default="./data/qdrant_storage/enriched_anime_database.json",
        help="Path to enriched anime database JSON file (default: ./data/qdrant_storage/enriched_anime_database.json)",
    )
    return parser.parse_args()


async def main() -> None:
    """Main reindexing function."""
    args = parse_args()
    print(" Starting anime database reindexing...")

    # Load settings
    settings = get_settings()
    print(f" Configuration loaded: {len(settings.vector_names)} vectors configured")

    # Initialize AsyncQdrantClient
    if settings.qdrant_api_key:
        async_qdrant_client = AsyncQdrantClient(
            url=settings.qdrant_url, api_key=settings.qdrant_api_key
        )
    else:
        async_qdrant_client = AsyncQdrantClient(url=settings.qdrant_url)

    try:
        # Initialize embedding manager and processors using factory pattern
        # 1. Initialize Mapper (Content Strategist)
        field_mapper = AnimeFieldMapper()

        # 2. Initialize Processors (Compute Engines)
        text_model = EmbeddingModelFactory.create_text_model(settings)
        text_processor = TextProcessor(model=text_model, settings=settings)

        vision_model = EmbeddingModelFactory.create_vision_model(settings)
        image_downloader = ImageDownloader(
            cache_dir=settings.model_cache_dir or "cache"
        )
        vision_processor = VisionProcessor(
            model=vision_model,
            downloader=image_downloader,
            # field_mapper removed from vision processor (SRP)
            settings=settings,
        )

        # 3. Initialize Manager (Orchestrator) with all dependencies
        embedding_manager = MultiVectorEmbeddingManager(
            text_processor=text_processor,
            vision_processor=vision_processor,
            field_mapper=field_mapper,  # Injected directly
            settings=settings,
        )

        # Initialize Qdrant client
        client = await QdrantClient.create(
            settings=settings,
            async_qdrant_client=async_qdrant_client,
            url=settings.qdrant_url,
            collection_name=settings.qdrant_collection_name,
        )

        # Delete existing collection if it exists
        try:
            await client.delete_collection()
            print(f"  Deleted existing collection: {settings.qdrant_collection_name}")
        except Exception as e:  # noqa: BLE001 - Best effort: continue if collection doesn't exist
            print(f"  Could not delete existing collection (may not exist): {e}")

        # Create fresh collection
        await client.create_collection()
        print(" Created fresh collection with multi-vector configuration")

        # Load anime data
        print(f" Loading anime data from {args.data_file}...")
        with open(args.data_file, encoding="utf-8") as f:
            enrichment_data = json.load(f)

        anime_data = enrichment_data["data"]
        print(f" Loaded {len(anime_data)} anime entries")

        # Convert to AnimeRecord objects
        print(" Converting to AnimeRecord objects...")
        records: list[AnimeRecord] = []

        for i, anime_dict in enumerate(anime_data):
            try:
                # Add UUID to anime.id if missing
                if "id" not in anime_dict["anime"] or not anime_dict["anime"]["id"]:
                    anime_dict["anime"]["id"] = str(uuid.uuid4())

                # Convert to AnimeRecord
                record = AnimeRecord(**anime_dict)
                records.append(record)
                print(f"   {i + 1}/{len(anime_data)}: {record.anime.title}")

            except Exception as e:  # noqa: BLE001 - Skip invalid entries, continue processing rest
                print(f"   Failed to convert entry {i + 1}: {e}")
                continue

        print(f" Successfully converted {len(records)} entries")

        if not records:
            print(" No valid anime entries to index")
            return

        # Start indexing with existing infrastructure
        print("\n Starting vector indexing using hierarchical infrastructure...")
        print(" This will generate:")
        print("   - Text Vectors (BGE-M3) for Anime, Characters, Episodes")
        print(
            "   - Image Vectors (OpenCLIP) for Anime Covers/Posters and Character Portraits"
        )
        print("   - Comprehensive payload indexing")

        try:
            # Process batch to get vectors and payloads
            print(
                f"\n Processing {len(records)} anime entries to generate hierarchical vectors..."
            )
            # embedding_manager.process_anime_batch now returns list[VectorDocument] directly
            points: list[VectorDocument] = await embedding_manager.process_anime_batch(
                records
            )

            print(
                f"Successfully generated {len(points)} vector points (Anime + Characters + Episodes)."
            )
            print(
                "   Note: Images are embedded as multivectors within Anime and Character points."
            )

            if not points:
                print("No points to index after embedding; skipping Qdrant upsert.")
                return

            # Add documents in batches
            result = await client.add_documents(
                points,
                batch_size=64,  # Use a reasonable batch size for efficiency
            )
            if result["success"]:
                print(f"\nSuccessfully indexed {len(points)} documents.")

                # Save updated anime data with generated IDs
                print(
                    f"\nSaving updated anime data with generated IDs to {args.data_file}..."
                )
                with open(args.data_file, "w", encoding="utf-8") as f:
                    json.dump(enrichment_data, f, indent=2, ensure_ascii=False)
                print("Updated data saved successfully")

                # Verify results using wrapper methods
                info = await client.get_collection_info()
                print("\n Final collection status:")
                print(f"   Points: {info.points_count}")
                print(f"   Expected: {len(points)} points")

            else:
                print(" All indexing failed")

        except Exception as e:  # noqa: BLE001 - Log all indexing errors with traceback for debugging
            print(f" Indexing error: {e}")
            import traceback

            traceback.print_exc()

    finally:
        await async_qdrant_client.close()


if __name__ == "__main__":
    asyncio.run(main())
