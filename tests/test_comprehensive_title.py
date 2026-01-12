#!/usr/bin/env python3
"""
Comprehensive title vector validation test with random entry selection, field combinations, and multimodal testing.
"""

import asyncio
import base64
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path

import pytest

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration

import requests
from common.config import get_settings
from qdrant_client import AsyncQdrantClient
from qdrant_db import QdrantClient
from vector_processing import AnimeFieldMapper, TextProcessor, VisionProcessor
from vector_processing.embedding_models.factory import EmbeddingModelFactory
from vector_processing.utils.image_downloader import ImageDownloader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_anime_database() -> dict:
    """Load full anime database from enrichment file."""
    with open("./data/qdrant_storage/enriched_anime_database.json") as f:
        return json.load(f)


def download_anime_image(image_url: str) -> str | None:
    """Download anime image to temporary file and return path."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(image_url, headers=headers, timeout=10)
        response.raise_for_status()

        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(response.content)
            return temp_file.name

    except Exception as e:
        print(f"   ⚠️  Failed to download image {image_url}: {e}")
        return None


def get_title_related_fields():
    """Get all title-related fields that map to title_vector."""
    return [
        "title",
        "title_english",
        "title_japanese",
        "synonyms",
        "synopsis",
        "background",
    ]


def generate_all_field_combinations(fields):
    """Generate all possible combinations of title fields (1 to N fields)."""
    from itertools import combinations

    all_combinations = []

    for r in range(1, len(fields) + 1):
        for combo in combinations(fields, r):
            all_combinations.append(list(combo))

    return all_combinations


def create_field_combination_query(anime: dict, field_combination: list[str]) -> str:
    """Create a query using specific field combination with actual anime data."""
    query_parts = []

    for field in field_combination:
        field_value = anime.get(field)
        if field_value:
            if field == "synonyms":
                # Handle synonyms as list
                if isinstance(field_value, list) and field_value:
                    # Randomly select 1-3 synonyms to avoid overly long queries
                    selected_synonyms = random.sample(
                        field_value, min(3, len(field_value))
                    )
                    query_parts.append(" ".join(selected_synonyms))
            elif field in ["synopsis", "background"]:
                # Truncate long text fields
                text_value = str(field_value)
                if len(text_value) > 100:
                    query_parts.append(text_value[:100])
                else:
                    query_parts.append(text_value)
            else:
                query_parts.append(str(field_value))

    return " ".join(query_parts).strip()


def extract_random_images(anime: dict) -> list[str]:
    """Extract random images from anime (covers, posters, banners)."""
    all_images = []

    if anime.get("images"):
        images = anime["images"]
        # Collect all available image types
        for image_type in ["covers", "posters", "banners", "screenshots"]:
            if image_type in images and images[image_type]:
                all_images.extend(images[image_type])

    return all_images


def test_title_vector_comprehensive():
    """Test title_vector with true field combination testing using random entries."""
    from utils.test_formatter import formatter

    formatter.print_header(
        "🎯 Comprehensive Title Vector Validation",
        "Testing all field combinations with production search_single_vector() method",
    )

    settings = get_settings()
    field_mapper = AnimeFieldMapper()
    text_model = EmbeddingModelFactory.create_text_model(settings)
    text_processor = TextProcessor(
        model=text_model, field_mapper=field_mapper, settings=settings
    )

    # Load anime database
    anime_database = load_anime_database()
    anime_data = anime_database.get("data", [])

    if not anime_data:
        print("   ❌ No anime data found for testing")
        return

    # True randomization with timestamp seed
    random.seed(int(time.time() * 1000) % 2**32)
    random_seed = int(time.time() * 1000) % 2**32

    # Get all title-related fields
    title_fields = get_title_related_fields()
    all_field_combinations = generate_all_field_combinations(title_fields)

    # Randomly select at least 5 anime for testing
    test_count = max(5, len(anime_data) // 5)
    test_anime = random.sample(anime_data, min(test_count, len(anime_data)))

    # Print configuration summary
    formatter.print_test_summary(
        "Title Vector Test",
        len(anime_data),
        len(test_anime),
        len(all_field_combinations),
        random_seed,
    )

    passed_tests = 0
    total_tests = 0
    field_combination_stats = {}

    # Initialize for stacked panels instead of table
    formatter.create_anime_test_panels()

    for i, anime in enumerate(test_anime):
        anime_title = anime.get("title", "Unknown")

        # Filter combinations to only those with available fields
        valid_combinations = []
        for combination in all_field_combinations:
            # Check if all fields in this combination have values
            if all(anime.get(field) for field in combination):
                valid_combinations.append(combination)

        if not valid_combinations:
            continue

        # Randomly select a field combination for this anime
        selected_combination = random.choice(valid_combinations)
        combination_key = "+".join(selected_combination)

        # Create query using the selected field combination
        text_query = create_field_combination_query(anime, selected_combination)

        if not text_query:
            continue

        # Generate embedding
        embedding = text_processor.encode_text(text_query)

        if not embedding:
            continue

        # Search title_vector using production method with raw similarity scores
        try:
            # Initialize Qdrant client with async factory pattern
            async_qdrant_client = AsyncQdrantClient(
                url=settings.qdrant_url, api_key=settings.qdrant_api_key
            )
            qdrant_client = asyncio.run(
                QdrantClient.create(settings, async_qdrant_client)
            )

            # Use search_single_vector to get real similarity scores
            results = asyncio.run(
                qdrant_client.search_single_vector(
                    vector_name="title_vector", vector_data=embedding, limit=5
                )
            )

            if results:
                top_result = results[0]
                top_title = top_result.get("title", "Unknown")
                top_score = top_result.get("score", 0.0)

                # Enhanced validation
                test_passed = False
                synonym_match = False
                if top_title == anime_title:
                    test_passed = True
                    passed_tests += 1
                elif any(r.get("title") == anime_title for r in results[:3]):
                    test_passed = True
                    passed_tests += 1
                else:
                    # Check if any synonyms match
                    anime_synonyms = anime.get("synonyms", [])
                    for synonym in anime_synonyms:
                        if any(
                            synonym.lower() in r.get("title", "").lower()
                            for r in results[:3]
                        ):
                            test_passed = True
                            synonym_match = True
                            passed_tests += 1
                            break

                # Print detailed test result using stacked panels
                formatter.print_detailed_test_result(
                    i + 1,
                    anime,
                    selected_combination,
                    text_query,
                    results,
                    test_passed,
                    synonym_match,
                )

                # Track field combination effectiveness
                if combination_key not in field_combination_stats:
                    field_combination_stats[combination_key] = {
                        "tests": 0,
                        "passes": 0,
                        "avg_score": 0.0,
                    }

                field_combination_stats[combination_key]["tests"] += 1
                if test_passed:
                    field_combination_stats[combination_key]["passes"] += 1
                field_combination_stats[combination_key]["avg_score"] += top_score

                total_tests += 1
            else:
                # Print detailed result for no results case
                formatter.print_detailed_test_result(
                    i + 1, anime, selected_combination, text_query, [], False
                )
                total_tests += 1

        except Exception as e:
            # Print error case
            error_results = [{"title": f"Error: {str(e)}", "score": 0.0}]
            formatter.print_detailed_test_result(
                i + 1, anime, selected_combination, text_query, error_results, False
            )
            total_tests += 1

    # Calculate field combination effectiveness
    for combination_key in field_combination_stats:
        if field_combination_stats[combination_key]["tests"] > 0:
            field_combination_stats[combination_key]["avg_score"] /= (
                field_combination_stats[combination_key]["tests"]
            )

    # Print field combination analysis
    if field_combination_stats:
        formatter.print_field_combination_analysis(field_combination_stats)

    # Print final results
    insights = [
        "True field combination testing validates all possible query patterns",
        "Random field selection ensures comprehensive coverage",
        "Each anime tested with available field combinations",
        "Uses production search_single_vector() method with real similarity scores",
    ]

    formatter.print_final_results(
        "Title Vector Test", passed_tests, total_tests, random_seed, insights
    )


def test_image_vector_comprehensive():
    """Test image_vector with random image downloads from titles."""
    from utils.test_formatter import formatter

    formatter.print_header(
        "📸 Comprehensive Image Vector Validation",
        "Testing image_vector with random image downloads using search_single_vector() method",
    )

    settings = get_settings()
    field_mapper = AnimeFieldMapper()
    vision_model = EmbeddingModelFactory.create_vision_model(settings)
    image_downloader = ImageDownloader(cache_dir=settings.model_cache_dir)
    vision_processor = VisionProcessor(
        model=vision_model,
        downloader=image_downloader,
        field_mapper=field_mapper,
        settings=settings,
    )

    # Load anime database
    anime_database = load_anime_database()
    anime_data = anime_database.get("data", [])

    # Find anime with images
    anime_with_images = []
    for anime in anime_data:
        images = extract_random_images(anime)
        if images:
            anime_with_images.append((anime, images))

    if not anime_with_images:
        print("   ❌ No anime with images found for testing")
        return

    # True randomization
    random.seed(int(time.time() * 1000) % 2**32)
    random_seed = int(time.time() * 1000) % 2**32

    # Randomly select 5 anime for testing
    test_anime = random.sample(anime_with_images, min(5, len(anime_with_images)))

    # Print configuration summary
    formatter.print_test_summary(
        "Image Vector Test",
        len(anime_data),
        len(test_anime),
        len(anime_with_images),  # Total anime with images
        random_seed,
    )

    passed_tests = 0
    total_tests = 0
    temp_files = []

    try:
        for i, (anime, available_images) in enumerate(test_anime):
            anime_title = anime.get("title", "Unknown")

            # Randomly select an image
            selected_image_url = random.choice(available_images)

            # Download image
            temp_image_path = download_anime_image(selected_image_url)
            if not temp_image_path:
                total_tests += 1
                continue

            temp_files.append(temp_image_path)

            try:
                # Convert to base64
                with open(temp_image_path, "rb") as f:
                    image_data = base64.b64encode(f.read()).decode("utf-8")
                    image_b64 = f"data:image/jpeg;base64,{image_data}"

                # Generate embedding
                embedding = vision_processor.encode_image(image_b64)
                if not embedding:
                    total_tests += 1
                    continue

                # Search image_vector using production method with raw similarity scores
                # Initialize Qdrant client with async factory pattern
                async_qdrant_client = AsyncQdrantClient(
                    url=settings.qdrant_url, api_key=settings.qdrant_api_key
                )
                qdrant_client = asyncio.run(
                    QdrantClient.create(settings, async_qdrant_client)
                )

                # Use search_single_vector to get real similarity scores
                results = asyncio.run(
                    qdrant_client.search_single_vector(
                        vector_name="image_vector", vector_data=embedding, limit=5
                    )
                )

                if results:
                    # Validation logic
                    top_result = results[0]
                    top_title = top_result.get("title", "Unknown")

                    if top_title == anime_title:
                        test_passed = True
                        passed_tests += 1
                    elif any(r.get("title") == anime_title for r in results[:3]):
                        test_passed = True
                        passed_tests += 1
                    else:
                        test_passed = False

                    # Use detailed formatter for results
                    formatter.print_detailed_image_test_result(
                        test_num=i + 1,
                        anime_data=anime,
                        image_url=selected_image_url,
                        available_images=len(available_images),
                        embedding_dim=len(embedding),
                        results=results,
                        test_passed=test_passed,
                    )

                    total_tests += 1
                else:
                    total_tests += 1

            except Exception:
                total_tests += 1
                continue

    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except Exception:
                pass

    # Results Summary using Rich formatter
    if total_tests > 0:
        success_rate = (passed_tests / total_tests) * 100

        # Determine status message and color
        if success_rate >= 80:
            status_msg = "🎉 Image vector working excellently!"
            status_color = "bright_green"
        elif success_rate >= 60:
            status_msg = "✅ Image vector working adequately!"
            status_color = "green"
        else:
            status_msg = "⚠️ Image vector needs improvement"
            status_color = "yellow"

        formatter.print_final_summary(
            "📊 Image Vector Test Results",
            passed_tests,
            total_tests,
            success_rate,
            status_msg,
            status_color,
            random_seed,
        )
    else:
        formatter.print_error_summary("❌ No image vector tests completed")


def test_multimodal_title_search():
    """Test multimodal title search with true field combination testing."""
    from utils.test_formatter import formatter

    formatter.print_header(
        "🔄 Comprehensive Multimodal Search Validation",
        "Testing title_vector + image_vector fusion with RRF method",
    )

    settings = get_settings()
    async_qdrant_client = AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key
        if hasattr(settings, "qdrant_api_key")
        else None,
    )
    qdrant_client = QdrantClient(
        settings=settings, async_qdrant_client=async_qdrant_client
    )

    # Initialize processors once before the loop
    field_mapper = AnimeFieldMapper()
    text_model = EmbeddingModelFactory.create_text_model(settings)
    vision_model = EmbeddingModelFactory.create_vision_model(settings)
    downloader = ImageDownloader(settings.model_cache_dir)
    text_processor = TextProcessor(
        model=text_model, field_mapper=field_mapper, settings=settings
    )
    vision_processor = VisionProcessor(
        model=vision_model,
        downloader=downloader,
        field_mapper=field_mapper,
        settings=settings,
    )

    # Load anime database
    anime_database = load_anime_database()
    anime_data = anime_database.get("data", [])

    # Find anime with images
    anime_with_images = []
    for anime in anime_data:
        images = extract_random_images(anime)
        if images:
            anime_with_images.append((anime, images))

    if not anime_with_images:
        formatter.print_error_summary(
            "❌ No anime with images found for multimodal testing"
        )
        return

    # True randomization
    random.seed(int(time.time() * 1000) % 2**32)
    random_seed = int(time.time() * 1000) % 2**32

    # Get all title-related fields
    title_fields = get_title_related_fields()
    all_field_combinations = generate_all_field_combinations(title_fields)

    # Select 5 anime for multimodal testing
    test_anime = random.sample(anime_with_images, min(5, len(anime_with_images)))

    # Configuration summary
    config_data = {
        "Strategy": "Multimodal fusion with field combinations",
        "Vectors": "title_vector (BGE-M3) + image_vector (OpenCLIP)",
        "Fusion Method": "RRF (Reciprocal Rank Fusion)",
        "Field Combinations": f"{len(all_field_combinations)} total combinations",
        "Test Count": f"{len(test_anime)} anime selected",
    }
    formatter.print_config_summary(config_data)

    passed_tests = 0
    total_tests = 0
    temp_files = []
    field_combination_stats = {}

    try:
        for i, (anime, available_images) in enumerate(test_anime):
            anime_title = anime.get("title", "Unknown")

            # Filter combinations to only those with available fields
            valid_combinations = []
            for combination in all_field_combinations:
                if all(anime.get(field) for field in combination):
                    valid_combinations.append(combination)

            if not valid_combinations:
                continue

            # Randomly select a field combination for this anime
            selected_combination = random.choice(valid_combinations)

            # Create query using the selected field combination
            text_query = create_field_combination_query(anime, selected_combination)
            if not text_query:
                continue

            # Random image
            selected_image_url = random.choice(available_images)

            temp_image_path = download_anime_image(selected_image_url)
            if not temp_image_path:
                total_tests += 1
                continue

            temp_files.append(temp_image_path)

            try:
                # Convert to base64
                with open(temp_image_path, "rb") as f:
                    image_data = base64.b64encode(f.read()).decode("utf-8")
                    image_b64 = f"data:image/jpeg;base64,{image_data}"

                # Generate embeddings (processors initialized before loop)
                text_embedding = text_processor.encode_text(text_query)
                image_embedding = vision_processor.encode_image(image_b64)

                if not text_embedding or not image_embedding:
                    total_tests += 1
                    continue

                # Perform multimodal search using search_multi_vector
                vector_queries = [
                    {"vector_name": "title_vector", "vector_data": text_embedding},
                    {"vector_name": "image_vector", "vector_data": image_embedding},
                ]

                multimodal_results = asyncio.run(
                    qdrant_client.search_multi_vector(
                        vector_queries=vector_queries, limit=5, fusion_method="rrf"
                    )
                )

                # Individual vector searches for comparison
                text_only_results = asyncio.run(
                    qdrant_client.search_multi_vector(
                        vector_queries=[
                            {
                                "vector_name": "title_vector",
                                "vector_data": text_embedding,
                            }
                        ],
                        limit=5,
                    )
                )

                image_only_results = asyncio.run(
                    qdrant_client.search_multi_vector(
                        vector_queries=[
                            {
                                "vector_name": "image_vector",
                                "vector_data": image_embedding,
                            }
                        ],
                        limit=5,
                    )
                )

                if multimodal_results:
                    top_result = multimodal_results[0]
                    top_title = top_result.get("title", "Unknown")
                    top_score = top_result.get("score", 0.0)

                    text_score = (
                        text_only_results[0].get("score", 0.0)
                        if text_only_results
                        else 0.0
                    )
                    image_score = (
                        image_only_results[0].get("score", 0.0)
                        if image_only_results
                        else 0.0
                    )

                    # Validation
                    test_passed = False
                    if top_title == anime_title:
                        test_passed = True
                        passed_tests += 1
                    elif any(
                        r.get("title") == anime_title for r in multimodal_results[:3]
                    ):
                        test_passed = True
                        passed_tests += 1
                    else:
                        # Check if any synonyms match
                        anime_synonyms = anime.get("synonyms", [])
                        for synonym in anime_synonyms:
                            if any(
                                synonym.lower() in r.get("title", "").lower()
                                for r in multimodal_results[:3]
                            ):
                                test_passed = True
                                passed_tests += 1
                                break

                    # Fusion effectiveness
                    fusion_boost = False
                    if top_score >= max(text_score, image_score) * 1.1:
                        fusion_boost = True

                    # Use detailed formatter for results
                    combination_key = "+".join(selected_combination)
                    formatter.print_detailed_multimodal_test_result(
                        test_num=i + 1,
                        anime_data=anime,
                        selected_combination=selected_combination,
                        text_query=text_query,
                        image_url=selected_image_url,
                        available_images=len(available_images),
                        text_score=text_score,
                        image_score=image_score,
                        multimodal_score=top_score,
                        multimodal_results=multimodal_results,
                        test_passed=test_passed,
                        fusion_boost=fusion_boost,
                    )

                    # Track field combination effectiveness
                    if combination_key not in field_combination_stats:
                        field_combination_stats[combination_key] = {
                            "tests": 0,
                            "passes": 0,
                            "fusion_boost": 0,
                            "avg_score": 0.0,
                        }

                    field_combination_stats[combination_key]["tests"] += 1
                    if test_passed:
                        field_combination_stats[combination_key]["passes"] += 1
                    if fusion_boost:
                        field_combination_stats[combination_key]["fusion_boost"] += 1
                    field_combination_stats[combination_key]["avg_score"] += top_score

                    total_tests += 1
                else:
                    total_tests += 1

            except Exception:
                total_tests += 1
                continue

    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except Exception:
                pass

    # Calculate field combination effectiveness
    for combination_key in field_combination_stats:
        if field_combination_stats[combination_key]["tests"] > 0:
            field_combination_stats[combination_key]["avg_score"] /= (
                field_combination_stats[combination_key]["tests"]
            )

    # Results Summary using Rich formatter
    if total_tests > 0:
        success_rate = (passed_tests / total_tests) * 100

        # Determine status message and color
        if success_rate >= 80:
            status_msg = "🎉 Multimodal fusion working excellently!"
            status_color = "bright_green"
        elif success_rate >= 60:
            status_msg = "✅ Multimodal fusion working adequately!"
            status_color = "green"
        else:
            status_msg = "⚠️ Multimodal fusion needs improvement"
            status_color = "yellow"

        formatter.print_final_summary(
            "📊 Multimodal Search Test Results",
            passed_tests,
            total_tests,
            success_rate,
            status_msg,
            status_color,
            random_seed,
        )

        # Field combination analysis
        if field_combination_stats:
            formatter.print_field_combination_analysis(field_combination_stats)
    else:
        formatter.print_error_summary("❌ No multimodal tests completed")


if __name__ == "__main__":
    # Run comprehensive title vector tests
    test_title_vector_comprehensive()

    # Run image vector tests
    test_image_vector_comprehensive()

    # Run multimodal title search tests
    test_multimodal_title_search()
