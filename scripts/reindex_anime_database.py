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
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, cast

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ["CUDA_VISIBLE_DEVICES"] = ""
from src.config import get_settings
from src.models.anime import AnimeEntry
from src.vector.client.qdrant_client import QdrantClient
from src.vector.processors.embedding_manager import MultiVectorEmbeddingManager
from src.vector.processors.text_processor import TextProcessor
from src.vector.processors.vision_processor import VisionProcessor
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct, Record


async def main() -> None:
    """Main reindexing function."""
    print(" Starting anime database reindexing...")

    # Load settings
    settings = get_settings()
    print(f" Configuration loaded: {len(settings.vector_names)} vectors configured")

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
        print("\n Starting vector indexing using existing infrastructure...")
        print(" This will generate:")
        print("   - 9 text vectors (BGE-M3 1024D)")
        print("   - 2 image vectors (OpenCLIP 768D)")
        print("   - Total: 11 named vectors per entry")
        print("   - Comprehensive payload indexing")

        try:
            # Process batch to get vectors and payloads
            print(f"\n Processing {len(anime_entries)} anime entries to generate vectors...")
            processed_batch = await embedding_manager.process_anime_batch(
                anime_entries
            )

            points = []
            for doc_data in processed_batch:
                if doc_data["metadata"].get("processing_failed"):
                    print(
                        f"Skipping failed document: {doc_data['metadata'].get('anime_title')}"
                    )
                    continue

                # Generate point ID from anime ID (same logic as QdrantClient._generate_point_id)
                point_id = hashlib.md5(doc_data["payload"]["id"].encode()).hexdigest()

                point = PointStruct(
                    id=point_id,
                    vector=doc_data["vectors"],
                    payload=doc_data["payload"],
                )
                points.append(point)

            print(f"Successfully generated vectors for {len(points)} entries.")

            # Add documents in batches
            success = await client.add_documents(
                points,
                batch_size=64, # Use a reasonable batch size for efficiency
            )
            if success:
                print(f"\nSuccessfully indexed {len(points)} documents.")
        
                # Save updated anime data with generated IDs
                print("\n💾 Saving updated anime data with generated IDs...")
                with open("./data/qdrant_storage/enriched_anime_database.json", "w", encoding="utf-8") as f:
                    json.dump(enrichment_data, f, indent=2, ensure_ascii=False)
                print("✅ Updated data saved successfully")

                # Verify results
                # Note: Using internal client.client for verification operations not exposed in wrapper
                # TODO: Consider adding wrapper methods for get_collection() and scroll() if used frequently
                info = await client.client.get_collection(settings.qdrant_collection_name)
                print("\n Final collection status:")
                print(f"   Points: {info.points_count}")
                print(f"   Expected: {len(points)} points")

                # Check vector completeness across sample points
                try:
                    # scroll() returns tuple[list[Record], Union[int, str, PointId, None]]
                    scroll_result: Tuple[List[Record], Any] = await client.client.scroll(
                        collection_name=settings.qdrant_collection_name,
                        limit=5,
                        with_vectors=True,
                    )
                    records, _ = scroll_result

                    points_with_11_vectors = 0
                    points_with_character_images = 0

                    for sample_point in records:
                        # sample_point.vector can be various types, handle as dict for named vectors
                        vectors_dict = cast(Dict[str, Any], sample_point.vector if isinstance(sample_point.vector, dict) else {})
                        vector_count = len(vectors_dict) if vectors_dict else 0
                        has_character_images = "character_image_vector" in vectors_dict

                        if vector_count == 11:
                            points_with_11_vectors += 1
                        if has_character_images:
                            points_with_character_images += 1

                    print(
                        f"   Points with 11 vectors: {points_with_11_vectors}/{len(records)}"
                    )
                    print(
                        f"   Points with character images: {points_with_character_images}/{len(records)}"
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


    finally:
        await async_qdrant_client.close()

if __name__ == "__main__":
    asyncio.run(main())
