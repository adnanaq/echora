#!/usr/bin/env python3
"""
Simple reindexing script for anime database with proper vector generation.

Uses existing infrastructure:
- QdrantClient.add_documents()
- MultiVectorEmbeddingManager
- AnimeFieldMapper
- TextProcessor and VisionProcessor
"""

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path
from typing import List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ["CUDA_VISIBLE_DEVICES"] = ""
from src.config import get_settings
from src.models.anime import AnimeEntry
from src.vector.client.qdrant_client import QdrantClient


async def main():
    """Main reindexing function."""
    print(" Starting anime database reindexing...")

    # Load settings
    settings = get_settings()
    print(f" Configuration loaded: {len(settings.vector_names)} vectors configured")

    # Initialize client and clean existing collection
    client = QdrantClient(settings=settings)

    # Delete existing collection if it exists
    try:
        client.client.delete_collection(settings.qdrant_collection_name)
        print(f"  Deleted existing collection: {settings.qdrant_collection_name}")
    except Exception as e:
        print(f"  No existing collection to delete: {e}")

    # Create fresh collection
    await client.create_collection()
    print(" Created fresh collection with 11-vector configuration")

    # Load anime data
    print(" Loading anime data...")
    with open("./data/qdrant_storage/enriched_anime_database.json", "r") as f:
        enrichment_data = json.load(f)

    anime_data = enrichment_data["data"]
    print(f" Loaded {len(anime_data)} anime entries")

    # Convert to AnimeEntry objects
    print(" Converting to AnimeEntry objects...")
    anime_entries: List[AnimeEntry] = []

    for i, anime_dict in enumerate(anime_data):
        try:
            # Add UUID if missing
            if "id" not in anime_dict:
                anime_dict["id"] = str(uuid.uuid4())

            # Convert to AnimeEntry
            anime_entry = AnimeEntry(**anime_dict)
            anime_entries.append(anime_entry)
            print(f"   {i+1}/{len(anime_data)}: {anime_entry.title}")

        except Exception as e:
            print(f"   Failed to convert entry {i+1}: {e}")
            continue

    print(f" Successfully converted {len(anime_entries)} entries")

    if not anime_entries:
        print(" No valid anime entries to index")
        return

    # Start indexing with existing infrastructure
    print(f"\n Starting vector indexing using existing infrastructure...")
    print(" This will generate:")
    print("   - 9 text vectors (BGE-M3 1024D)")
    print("   - 2 image vectors (OpenCLIP 768D)")
    print("   - Total: 11 named vectors per entry")
    print("   - Comprehensive payload indexing")

    try:
        # Process entries individually for better logging
        print(f"\n Processing {len(anime_entries)} anime entries individually...")

        successful_entries = 0
        failed_entries = 0

        for i, anime_entry in enumerate(anime_entries):
            try:
                print(
                    f"\n Processing {i+1}/{len(anime_entries)}: {anime_entry.title}"
                )

                # Process single entry
                success = await client.add_documents([anime_entry], batch_size=1)

                if success:
                    print(f"    Successfully indexed: {anime_entry.title}")
                    successful_entries += 1
                else:
                    print(f"    Failed to index: {anime_entry.title}")
                    failed_entries += 1

            except Exception as e:
                print(f"    Error processing {anime_entry.title}: {e}")
                failed_entries += 1
                continue

        print(f"\n Processing Summary:")
        print(f"   Successful: {successful_entries}/{len(anime_entries)}")
        print(f"   Failed: {failed_entries}/{len(anime_entries)}")

        if successful_entries > 0:
            print("\n Indexing completed with some success!")

            # Verify results
            info = client.client.get_collection(settings.qdrant_collection_name)
            print(f"\n Final collection status:")
            print(f"   Points: {info.points_count}")
            print(f"   Expected: {len(anime_entries)} points")

            # Check vector completeness across sample points
            try:
                result, _ = client.client.scroll(
                    collection_name=settings.qdrant_collection_name,
                    limit=5,
                    with_vectors=True,
                )

                points_with_11_vectors = 0
                points_with_character_images = 0

                for point in result:
                    vector_count = len(point.vector) if point.vector else 0
                    has_character_images = "character_image_vector" in (
                        point.vector or {}
                    )

                    if vector_count == 11:
                        points_with_11_vectors += 1
                    if has_character_images:
                        points_with_character_images += 1

                print(
                    f"   Points with 11 vectors: {points_with_11_vectors}/{len(result)}"
                )
                print(
                    f"   Points with character images: {points_with_character_images}/{len(result)}"
                )

                if points_with_11_vectors > 0:
                    print(" 11-vector architecture working successfully!")
                    print(" Character image vectors being generated!")
                else:
                    print("  Warning: Not all vectors being generated")

            except Exception as e:
                print(f"  Could not verify vector completeness: {e}")

        else:
            print(" All indexing failed")

    except Exception as e:
        print(f" Indexing error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
