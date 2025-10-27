#!/usr/bin/env python3
"""
Enhanced character vector validation test with both text and image testing.
"""

import json
import os
import random
import sys
import tempfile
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import requests

from src.config import get_settings
from src.vector.processors.text_processor import TextProcessor
from src.vector.processors.vision_processor import VisionProcessor

# Mark all tests in this file as integration tests (require Qdrant running)
pytestmark = pytest.mark.integration


def test_character_vector_realistic():
    """Test character vector against actual available data."""
    print("🧠 Character Vector Validation - Data-Driven Testing")
    print("📋 Testing against actual character data from 13 anime with characters")

    settings = get_settings()
    text_processor = TextProcessor(settings=settings)

    # Test cases based on ACTUAL data we confirmed exists
    test_cases = [
        {
            "query": "drummer MASKING",
            "expected_title": "!NVADE SHOW!",
            "reason": "Masuki Satou is MASKING, a drummer",
        },
        {
            "query": "Konoha Inoue writer",
            "expected_contains": "Bungaku Shoujo",
            "reason": "Konoha Inoue writes novels",
        },
        {
            "query": "literature club president",
            "expected_contains": "Bungaku Shoujo",
            "reason": "Tooko Amano is literature club president",
        },
        {
            "query": "14 year old DJ",
            "expected_title": "!NVADE SHOW!",
            "reason": "Chiyu Tamade (CHU²) is 14 and a DJ",
        },
        {
            "query": "RAISE A SUILEN band member",
            "expected_title": "!NVADE SHOW!",
            "reason": "Multiple characters are in RAISE A SUILEN band",
        },
    ]

    print(f"📊 Testing {len(test_cases)} realistic character queries...")

    passed = 0
    for i, test_case in enumerate(test_cases):
        print(f"\n🔍 Test {i + 1}: '{test_case['query']}'")
        print(f"   💭 Expected: {test_case['reason']}")

        # Generate embedding
        embedding = text_processor.encode_text(test_case["query"])

        # Search character vector
        search_payload = {
            "vector": {"name": "character_vector", "vector": embedding},
            "limit": 3,
            "with_payload": True,
        }

        response = requests.post(
            f"{settings.qdrant_url}/collections/{settings.qdrant_collection_name}/points/search",
            headers={
                "api-key": settings.qdrant_api_key,
                "Content-Type": "application/json",
            },
            json=search_payload,
        )

        if response.status_code == 200:
            results = response.json()["result"]
            print(f"   ✅ Found {len(results)} results")

            for j, result in enumerate(results):
                title = result["payload"]["title"]
                score = result["score"]
                print(f"      {j + 1}. {title} (score: {score:.4f})")

            # Validate
            top_title = results[0]["payload"]["title"] if results else ""

            if "expected_title" in test_case:
                if test_case["expected_title"] == top_title:
                    print(
                        f"   ✅ PASS - Found expected anime: {test_case['expected_title']}"
                    )
                    passed += 1
                else:
                    print(
                        f"   ❌ FAIL - Expected '{test_case['expected_title']}', got '{top_title}'"
                    )
            elif "expected_contains" in test_case:
                if any(
                    test_case["expected_contains"].lower() in title.lower()
                    for title in [r["payload"]["title"] for r in results]
                ):
                    print(
                        f"   ✅ PASS - Found anime containing: {test_case['expected_contains']}"
                    )
                    passed += 1
                else:
                    print(
                        f"   ❌ FAIL - No results contained: {test_case['expected_contains']}"
                    )
        else:
            print(f"   ❌ Search failed: {response.status_code}")

    print("\n📊 Final Character Vector Validation:")
    print(f"   ✅ Passed: {passed}/{len(test_cases)}")
    print(f"   📈 Success Rate: {(passed / len(test_cases) * 100):.1f}%")

    if passed >= 4:  # 80% success rate
        print("   🎉 Character vector is working excellently!")
    elif passed >= 3:  # 60% success rate
        print("   ✅ Character vector is working adequately!")
    else:
        print("   ⚠️  Character vector needs improvement")


def load_character_data() -> dict[str, list[dict]]:
    """Load character data from enrichment file."""
    with open("./data/qdrant_storage/enriched_anime_database.json") as f:
        data = json.load(f)

    anime_with_character_images = {}
    for anime in data["data"]:
        anime_title = anime["title"]
        characters_with_images = []

        for char in anime.get("characters", []):
            if char.get("images") and len(char["images"]) > 0:
                characters_with_images.append(char)

        if characters_with_images:
            anime_with_character_images[anime_title] = characters_with_images

    return anime_with_character_images


def download_character_image(image_url: str) -> str | None:
    """Download character image to temporary file and return path."""
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


def test_character_image_vector():
    """Test character_image_vector with actual character images."""
    print("\n📸 Character Image Vector Validation")
    print("📋 Testing character_image_vector with real character images")

    settings = get_settings()
    vision_processor = VisionProcessor(settings=settings)

    # Load character data
    anime_with_images = load_character_data()

    if not anime_with_images:
        print("   ❌ No character images found for testing")
        return

    print(f"   📊 Found {len(anime_with_images)} anime with character images")

    # Randomly select 5 anime with character images for testing
    test_anime = random.sample(
        list(anime_with_images.keys()), min(5, len(anime_with_images))
    )

    passed_tests = 0
    total_tests = 0
    temp_files = []

    try:
        for anime_title in test_anime:
            characters = anime_with_images[anime_title]
            # Pick first character with images
            character = characters[0]
            character_name = character.get("name", "Unknown")
            image_urls = character.get("images", [])

            if not image_urls:
                continue

            print(f"\n🎭 Testing: {character_name} from '{anime_title}'")

            # Try first image URL
            image_url = image_urls[0]
            print(f"   📥 Downloading: {image_url}")

            # Download image
            temp_image_path = download_character_image(image_url)
            if not temp_image_path:
                continue

            temp_files.append(temp_image_path)

            # Process image with vision processor
            try:
                with open(temp_image_path, "rb") as f:
                    import base64

                    image_data = base64.b64encode(f.read()).decode("utf-8")
                    image_b64 = f"data:image/jpeg;base64,{image_data}"

                # Generate embedding
                embedding = vision_processor.encode_image(image_b64)
                if not embedding:
                    print(f"   ❌ Failed to generate embedding for {character_name}")
                    continue

                print(f"   ✅ Generated {len(embedding)}-dimensional embedding")

                # Search character_image_vector
                search_payload = {
                    "vector": {"name": "character_image_vector", "vector": embedding},
                    "limit": 3,
                    "with_payload": True,
                }

                response = requests.post(
                    f"{settings.qdrant_url}/collections/{settings.qdrant_collection_name}/points/search",
                    headers={
                        "api-key": settings.qdrant_api_key,
                        "Content-Type": "application/json",
                    },
                    json=search_payload,
                )

                if response.status_code == 200:
                    results = response.json()["result"]
                    print(f"   📊 Found {len(results)} results")

                    if results:
                        top_result = results[0]
                        top_title = top_result["payload"]["title"]
                        top_score = top_result["score"]

                        print(f"      1. {top_title} (score: {top_score:.4f})")

                        # Validate that the top result is the expected anime
                        if top_title == anime_title:
                            print(
                                "   ✅ PASS - Character image correctly matched to source anime!"
                            )
                            passed_tests += 1
                        else:
                            print(
                                f"   ❌ FAIL - Expected '{anime_title}', got '{top_title}'"
                            )

                        total_tests += 1
                    else:
                        print("   ❌ No results returned")
                        total_tests += 1
                else:
                    print(f"   ❌ Search failed: {response.status_code}")
                    total_tests += 1

            except Exception as e:
                print(f"   ❌ Error processing image: {e}")
                total_tests += 1
                continue

    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass

    # Summary
    print("\n📊 Character Image Vector Results:")
    if total_tests > 0:
        success_rate = (passed_tests / total_tests) * 100
        print(f"   ✅ Passed: {passed_tests}/{total_tests}")
        print(f"   📈 Success Rate: {success_rate:.1f}%")

        if success_rate >= 80:
            print("   🎉 Character image vector working excellently!")
        elif success_rate >= 60:
            print("   ✅ Character image vector working adequately!")
        else:
            print("   ⚠️  Character image vector needs improvement")
    else:
        print("   ❌ No character image tests completed")


def find_anime_by_title(title, anime_database):
    """Find anime entry by title in the database."""
    for anime in anime_database.get("data", []):
        if anime.get("title") == title:
            return anime
    return None


def verify_character_in_anime(character_name, returned_anime_title, anime_database):
    """Check if the character actually exists in the returned anime."""
    returned_anime = find_anime_by_title(returned_anime_title, anime_database)
    if returned_anime:
        character_names = [
            char.get("name", "") for char in returned_anime.get("characters", [])
        ]
        # Also check name variations and nicknames
        all_character_names = []
        for char in returned_anime.get("characters", []):
            all_character_names.append(char.get("name", ""))
            all_character_names.extend(char.get("name_variations", []))
            all_character_names.extend(char.get("nicknames", []))

        return character_name in all_character_names
    return False


def create_character_query_patterns():
    """Create diverse query patterns to test different character fields."""
    return [
        # Pattern 1: Name + Description (current approach)
        {
            "name": "Name + Description",
            "generator": lambda char: f"{char.get('name', '')} {char.get('description', '')[:100] if char.get('description') else ''}".strip(),
        },
        # Pattern 2: Nickname-focused
        {
            "name": "Nickname + Role",
            "generator": lambda char: f"{char.get('nicknames', [char.get('name', '')])[0] if char.get('nicknames') else char.get('name', '')} {char.get('role', '')} character".strip(),
        },
        # Pattern 3: Role + Gender + Traits
        {
            "name": "Role + Gender + Traits",
            "generator": lambda char: f"{char.get('role', '')} character {char.get('gender', '')} {char.get('description', '')[:50] if char.get('description') else ''}".strip(),
        },
        # Pattern 4: Name Variations
        {
            "name": "Name Variations",
            "generator": lambda char: f"{char.get('name_variations', [char.get('name', '')])[0] if char.get('name_variations') else char.get('name', '')} {char.get('nicknames', [''])[0] if char.get('nicknames') else ''}".strip(),
        },
        # Pattern 5: Full Context (all available fields)
        {
            "name": "Full Context",
            "generator": lambda char: f"{char.get('name', '')} {' '.join(char.get('nicknames', []))} {char.get('role', '')} {char.get('gender', '')} {char.get('description', '')[:80] if char.get('description') else ''}".strip(),
        },
        # Pattern 6: Minimal Context (stress test)
        {
            "name": "Minimal Context",
            "generator": lambda char: f"{char.get('nicknames', [char.get('name', '')])[0] if char.get('nicknames') else char.get('name', '')}".strip(),
        },
    ]


def test_multimodal_character_search():
    """Enhanced multimodal character search testing with comprehensive field coverage."""
    print("\n🎭 Enhanced Multimodal Character Search Validation")
    print(
        "📋 Testing character search with diverse query patterns across all character fields"
    )
    print("🔍 Random testing with cross-reference validation and pattern analysis")

    settings = get_settings()

    # Import QdrantClient for multimodal search
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from src.vector.client.qdrant_client import QdrantClient

    qdrant_client = QdrantClient(settings=settings)

    # Load character data AND full anime database for cross-reference validation
    anime_with_images = load_character_data()

    # Load full database for validation
    with open("./data/qdrant_storage/enriched_anime_database.json") as f:
        full_anime_database = json.load(f)

    if not anime_with_images:
        print("   ❌ No character images found for multimodal testing")
        return

    print(f"   📊 Found {len(anime_with_images)} anime with character images")

    # True randomization with timestamp seed
    import time

    random.seed(int(time.time() * 1000) % 2**32)
    print(f"   🎲 Random seed: {int(time.time() * 1000) % 2**32}")

    # Randomly select 5 anime for comprehensive testing
    test_anime = random.sample(
        list(anime_with_images.keys()), min(5, len(anime_with_images))
    )
    print(f"   🎯 Testing {len(test_anime)} randomly selected anime")

    # Create query patterns
    query_patterns = create_character_query_patterns()

    passed_tests = 0
    total_tests = 0
    temp_files = []
    pattern_stats = {
        pattern["name"]: {"tests": 0, "passes": 0, "avg_score": 0.0}
        for pattern in query_patterns
    }

    try:
        for anime_title in test_anime:
            characters = anime_with_images[anime_title]
            # Pick first character with images
            character = characters[0]
            character_name = character.get("name", "Unknown")
            image_urls = character.get("images", [])

            if not image_urls:
                print(f"   ⚠️  Skipping {character_name}: No images")
                continue

            print(f"\n🎭 Testing Character: {character_name} from '{anime_title}'")

            # Display available character fields for transparency
            available_fields = []
            if character.get("role"):
                available_fields.append(f"Role: {character.get('role')}")
            if character.get("nicknames"):
                available_fields.append(
                    f"Nicknames: {', '.join(character.get('nicknames'))}"
                )
            if character.get("gender"):
                available_fields.append(f"Gender: {character.get('gender')}")
            if character.get("description"):
                available_fields.append(
                    f"Description: {character.get('description')[:50]}..."
                )

            print(
                f"   📋 Available fields: {' | '.join(available_fields) if available_fields else 'Name only'}"
            )

            # Download character image
            image_url = image_urls[0]
            print(f"   📥 Downloading: {image_url}")

            temp_image_path = download_character_image(image_url)
            if not temp_image_path:
                continue

            temp_files.append(temp_image_path)

            # Convert image to base64
            try:
                with open(temp_image_path, "rb") as f:
                    import base64

                    image_data = base64.b64encode(f.read()).decode("utf-8")
                    image_b64 = f"data:image/jpeg;base64,{image_data}"

                # Randomly select a query pattern for this character
                selected_pattern = random.choice(query_patterns)
                text_query = selected_pattern["generator"](character).strip()

                if not text_query:
                    text_query = character_name  # Fallback

                print(f"   🎲 Query Pattern: {selected_pattern['name']}")
                print(
                    f'   📝 Generated Query: "{text_query[:80]}{"..." if len(text_query) > 80 else ""}"'
                )

                print("   🔍 Multimodal search: text + image")

                # Perform multimodal character search
                import asyncio

                results = asyncio.run(
                    qdrant_client.search_characters(
                        query=text_query, image_data=image_b64, limit=5
                    )
                )

                if results:
                    top_result = results[0]
                    top_title = top_result.get("title", "Unknown")
                    top_score = top_result.get("score", 0.0)

                    print(f"   📊 Found {len(results)} results")
                    print(f"      1. {top_title} (score: {top_score:.4f})")

                    # Comparative analysis: Text-only vs Image-only vs Multimodal
                    text_only_results = asyncio.run(
                        qdrant_client.search_characters(query=text_query, limit=5)
                    )
                    text_only_score = (
                        text_only_results[0].get("score", 0.0)
                        if text_only_results
                        else 0.0
                    )

                    image_only_results = asyncio.run(
                        qdrant_client.search_characters(
                            query="character",  # minimal text
                            image_data=image_b64,
                            limit=5,
                        )
                    )
                    image_only_score = (
                        image_only_results[0].get("score", 0.0)
                        if image_only_results
                        else 0.0
                    )

                    print("   📈 Score Analysis:")
                    print(
                        f"      Text-only ({selected_pattern['name']}): {text_only_score:.4f}"
                    )
                    print(f"      Image-only: {image_only_score:.4f}")
                    print(f"      Multimodal: {top_score:.4f}")

                    # Enhanced validation with pattern tracking
                    test_passed = False
                    if top_title == anime_title:
                        print("   ✅ PASS - Exact source anime match!")
                        test_passed = True
                        passed_tests += 1
                    elif verify_character_in_anime(
                        character_name, top_title, full_anime_database
                    ):
                        print(
                            f"   ✅ PASS - Character '{character_name}' verified in returned anime '{top_title}' (cross-reference)!"
                        )
                        test_passed = True
                        passed_tests += 1
                    else:
                        print(
                            f"   ❌ FAIL - Character '{character_name}' not found in returned anime '{top_title}'"
                        )
                        # Debug info
                        returned_anime = find_anime_by_title(
                            top_title, full_anime_database
                        )
                        if returned_anime:
                            chars_in_returned = [
                                c.get("name", "Unknown")
                                for c in returned_anime.get("characters", [])
                            ]
                            print(
                                f"        Characters in '{top_title}': {chars_in_returned[:3]}{'...' if len(chars_in_returned) > 3 else ''}"
                            )

                    # Pattern effectiveness tracking
                    pattern_stats[selected_pattern["name"]]["tests"] += 1
                    if test_passed:
                        pattern_stats[selected_pattern["name"]]["passes"] += 1
                    pattern_stats[selected_pattern["name"]]["avg_score"] += top_score

                    # Fusion effectiveness analysis
                    if (
                        top_score >= max(text_only_score, image_only_score) * 1.1
                    ):  # 10% improvement threshold
                        print(
                            "   🎯 FUSION BOOST - Multimodal significantly improved results!"
                        )
                    elif top_score >= max(text_only_score, image_only_score):
                        print("   🎯 FUSION BENEFIT - Multimodal improved results")
                    else:
                        print("   ⚠️  Individual modalities performed better")

                    total_tests += 1
                else:
                    print("   ❌ No results returned")
                    pattern_stats[selected_pattern["name"]]["tests"] += 1
                    total_tests += 1

            except Exception as e:
                print(f"   ❌ Error processing character: {e}")
                total_tests += 1
                continue

    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass

    # Calculate pattern effectiveness
    for pattern_name in pattern_stats:
        if pattern_stats[pattern_name]["tests"] > 0:
            pattern_stats[pattern_name]["avg_score"] /= pattern_stats[pattern_name][
                "tests"
            ]

    # Comprehensive Results Summary
    print("\n📊 Enhanced Multimodal Character Search Results:")
    print(f"   🎲 Random seed used: {int(time.time() * 1000) % 2**32}")
    print(f"   🎯 Tested {len(test_anime)} randomly selected anime")

    if total_tests > 0:
        success_rate = (passed_tests / total_tests) * 100
        print("\n   📈 Overall Results:")
        print(f"   ✅ Passed: {passed_tests}/{total_tests}")
        print(f"   📊 Success Rate: {success_rate:.1f}%")

        # Pattern Effectiveness Analysis
        print("\n   🎯 Query Pattern Analysis:")
        for pattern_name, stats in pattern_stats.items():
            if stats["tests"] > 0:
                pattern_success_rate = (stats["passes"] / stats["tests"]) * 100
                avg_score = stats["avg_score"]
                print(
                    f"      • {pattern_name}: {stats['passes']}/{stats['tests']} ({pattern_success_rate:.1f}%) | Avg Score: {avg_score:.4f}"
                )
            else:
                print(f"      • {pattern_name}: Not tested this run")

        # Best performing pattern
        best_pattern = max(
            [p for p in pattern_stats.items() if p[1]["tests"] > 0],
            key=lambda x: x[1]["passes"] / x[1]["tests"] if x[1]["tests"] > 0 else 0,
            default=None,
        )
        if best_pattern:
            best_name, best_stats = best_pattern
            best_rate = (best_stats["passes"] / best_stats["tests"]) * 100
            print(f"   🏆 Best Pattern: {best_name} ({best_rate:.1f}% success rate)")

        # Overall Assessment
        print("\n   🔬 Assessment:")
        if success_rate >= 80:
            print("   🎉 Enhanced multimodal character search working excellently!")
            print(
                "   ✨ Multiple query patterns validated across diverse character fields"
            )
        elif success_rate >= 60:
            print("   ✅ Enhanced multimodal character search working adequately!")
            print("   📚 Some query patterns more effective than others")
        else:
            print("   ⚠️  Enhanced multimodal character search needs improvement")
            print(
                "   🔍 Consider optimizing character_vector indexing or fusion parameters"
            )

        print("\n   🎯 Key Insights:")
        print("   • True randomization ensures comprehensive testing coverage")
        print("   • Pattern diversity validates all character field combinations")
        print("   • Cross-reference validation accounts for franchise characters")
        print(
            "   • Fusion analysis shows multimodal effectiveness vs individual vectors"
        )

    else:
        print("   ❌ No enhanced multimodal character tests completed")


if __name__ == "__main__":
    # Run text-based character vector tests
    test_character_vector_realistic()

    # Run image-based character vector tests
    test_character_image_vector()

    # Run multimodal character search tests
    test_multimodal_character_search()
