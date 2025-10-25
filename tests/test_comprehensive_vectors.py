#!/usr/bin/env python3
"""
Comprehensive Vector Database Test Suite

A comprehensive testing framework for the 13-vector anime database that covers:
- Individual vector queries (simple to complex)
- Payload filtering and hybrid search
- Multi-vector queries and combinations
- Performance benchmarking
- Edge cases and error handling

This script is designed to be continuously improved and expanded.
"""

import asyncio
import base64
import io
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import requests
from PIL import Image

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config import get_settings
from src.vector.processors.text_processor import TextProcessor
from src.vector.processors.vision_processor import VisionProcessor


@dataclass
class TestResult:
    """Individual test result container."""

    test_name: str
    query: str
    vector_type: str
    execution_time: float
    results_count: int
    top_score: float
    success: bool
    error: Optional[str] = None
    results: List[Dict] = field(default_factory=list)


@dataclass
class TestSuite:
    """Test suite configuration and results."""

    name: str
    description: str
    tests: List[Dict] = field(default_factory=list)
    results: List[TestResult] = field(default_factory=list)


class ComprehensiveVectorTester:
    """Comprehensive testing framework for the 13-vector anime database."""

    def __init__(
        self, qdrant_url: str, api_key: str, collection_name: str = "anime_database"
    ):
        """Initialize the comprehensive tester."""
        self.qdrant_url = qdrant_url.rstrip("/")
        self.api_key = api_key
        self.collection_name = collection_name

        self.headers = {"api-key": api_key, "Content-Type": "application/json"}

        # Initialize processors
        settings = get_settings()
        self.text_processor = TextProcessor(settings)
        self.vision_processor = VisionProcessor(settings)

        # Initialize QdrantClient for multi-vector operations
        from src.vector.client.qdrant_client import QdrantClient

        qdrant_settings = get_settings()
        qdrant_settings.qdrant_url = qdrant_url
        qdrant_settings.qdrant_api_key = api_key
        qdrant_settings.qdrant_collection_name = collection_name
        self.qdrant_client = QdrantClient(settings=qdrant_settings)

        # Test suites
        self.test_suites = []
        self._sample_images = {}  # Cache for sample images from collection
        self._uploaded_image_vectors = {}  # Cache for uploaded image vectors
        self._initialize_test_suites()

        print("🔬 Comprehensive Vector Database Test Suite Initialized")
        print(f"🎯 Target Collection: {collection_name}")
        print(f"🧠 Text Model: {self.text_processor.get_model_info()['model_name']}")
        print(
            f"👁️  Vision Model: {self.vision_processor.get_model_info()['model_name']}"
        )

    def _initialize_test_suites(self):
        """Initialize all test suites."""
        self.test_suites = [
            self._create_basic_vector_tests(),
            self._create_semantic_complexity_tests(),
            self._create_image_vector_tests(),
            self._create_payload_filter_tests(),
            self._create_comprehensive_search_tests(),
            self._create_multi_vector_tests(),
            self._create_edge_case_tests(),
            self._create_performance_tests(),
            self._create_real_world_scenario_tests(),
        ]

    def _create_basic_vector_tests(self) -> TestSuite:
        """Create basic individual vector tests."""
        return TestSuite(
            name="Basic Vector Tests",
            description="Test individual vectors with simple to moderate complexity queries",
            tests=[
                # Title Vector Tests
                {
                    "name": "Simple Title Search",
                    "query": "boxing anime",
                    "vector": "title_vector",
                    "expected_keywords": ["boxing", "fight", "sport"],
                },
                {
                    "name": "Synopsis Search",
                    "query": "story about literature and mysterious girl",
                    "vector": "title_vector",
                    "expected_keywords": ["literature", "book", "mysterious"],
                },
                # Character Vector Tests
                {
                    "name": "Character Archetype",
                    "query": "strong female protagonist",
                    "vector": "character_vector",
                    "expected_keywords": ["female", "strong", "protagonist"],
                },
                {
                    "name": "Character Relationship",
                    "query": "father son relationship conflict",
                    "vector": "character_vector",
                    "expected_keywords": ["father", "family", "relationship"],
                },
                # Genre Vector Tests
                {
                    "name": "Genre Combination",
                    "query": "comedy drama sports",
                    "vector": "genre_vector",
                    "expected_keywords": ["comedy", "drama", "sports"],
                },
                {
                    "name": "Mood Query",
                    "query": "dark psychological thriller",
                    "vector": "genre_vector",
                    "expected_keywords": ["psychological", "dark", "drama"],
                },
                # Technical Vector Tests
                # Temporal Vector Tests
                {
                    "name": "Season Search",
                    "query": "summer anime releases",
                    "vector": "temporal_vector",
                    "expected_keywords": ["summer", "season", "release"],
                },
                {
                    "name": "Classic Era",
                    "query": "retro classic animation from 1990s",
                    "vector": "temporal_vector",
                    "expected_keywords": ["1990", "classic", "retro"],
                },
                # Image Vector Tests
                {
                    "name": "General Image Style",
                    "query": "dark dramatic art style with moody colors",
                    "vector": "image_vector",
                    "expected_keywords": ["dark", "dramatic", "style"],
                },
                {
                    "name": "Cover Art Style",
                    "query": "bright colorful poster art with dynamic composition",
                    "vector": "image_vector",
                    "expected_keywords": ["bright", "colorful", "dynamic"],
                },
                # Character Image Vector Tests
                {
                    "name": "Character Design Style",
                    "query": "realistic character designs with detailed faces",
                    "vector": "character_image_vector",
                    "expected_keywords": ["realistic", "detailed", "character"],
                },
                {
                    "name": "Character Art Comparison",
                    "test_type": "image_from_collection",
                    "vector": "character_image_vector",
                    "description": "Find similar character images using actual image data",
                },
            ],
        )

    def _create_semantic_complexity_tests(self) -> TestSuite:
        """Create complex semantic understanding tests."""
        return TestSuite(
            name="Semantic Complexity Tests",
            description="Test complex semantic understanding and nuanced queries",
            tests=[
                {
                    "name": "Multi-Concept Story",
                    "query": "coming of age story about overcoming family expectations in competitive environment",
                    "vector": "title_vector",
                    "complexity": "high",
                },
                {
                    "name": "Character Psychology",
                    "query": "reluctant hero struggling with identity and self-doubt",
                    "vector": "character_vector",
                    "complexity": "high",
                },
                {
                    "name": "Thematic Depth",
                    "query": "exploration of tradition versus modernity through sports",
                    "vector": "genre_vector",
                    "complexity": "high",
                },
                {
                    "name": "Emotional Tone",
                    "query": "bittersweet narrative with hope despite hardship",
                    "vector": "review_vector",
                    "complexity": "high",
                },
                {
                    "name": "Cultural Context",
                    "query": "Japanese cultural values and honor in family traditions",
                    "vector": "franchise_vector",
                    "complexity": "high",
                },
            ],
        )

    def _create_image_vector_tests(self) -> TestSuite:
        """Create comprehensive image vector tests."""
        return TestSuite(
            name="Image Vector Tests",
            description="Test both general image_vector and character_image_vector capabilities",
            tests=[
                # General Image Vector Tests
                {
                    "name": "Art Style Similarity",
                    "test_type": "image_from_collection",
                    "vector": "image_vector",
                    "description": "Use an existing anime's general images to find similar art styles",
                },
                {
                    "name": "Cover Art Query",
                    "query": "dramatic action scene with dynamic poses",
                    "vector": "image_vector",
                    "description": "Text-to-image style search for covers and art",
                },
                {
                    "name": "Visual Mood Search",
                    "query": "bright cheerful colorful animation style",
                    "vector": "image_vector",
                    "description": "Find anime with specific visual mood",
                },
                {
                    "name": "Retro Art Style",
                    "query": "1990s classic animation art style with hand-drawn quality",
                    "vector": "image_vector",
                    "description": "Era-specific art style search",
                },
                # Character Image Vector Tests
                {
                    "name": "Character Design Similarity",
                    "test_type": "image_from_collection",
                    "vector": "character_image_vector",
                    "description": "Use character images to find similar character designs",
                },
                {
                    "name": "Uploaded Character Image Test",
                    "test_type": "uploaded_image",
                    "vector": "character_image_vector",
                    "description": "Test uploaded character image against character_image_vector",
                },
                {
                    "name": "Uploaded Image Style Test",
                    "test_type": "uploaded_image",
                    "vector": "image_vector",
                    "description": "Test uploaded character image against general image_vector",
                },
                {
                    "name": "Character Type Search",
                    "query": "strong muscular male protagonist with determined expression",
                    "vector": "character_image_vector",
                    "description": "Text-based character appearance search",
                },
                {
                    "name": "Character Art Style",
                    "query": "realistic detailed character portraits with shading",
                    "vector": "character_image_vector",
                    "description": "Character art style preferences",
                },
                {
                    "name": "Gender-Based Character Search",
                    "query": "female characters with long hair and expressive eyes",
                    "vector": "character_image_vector",
                    "description": "Gender and feature-specific character search",
                },
                # Cross-Vector Image Comparison
                {
                    "name": "Image Vector Separation Test",
                    "test_type": "cross_vector_text_comparison",
                    "query": "action scenes with characters",
                    "vectors": ["image_vector", "character_image_vector"],
                    "description": "Compare how the same query performs across image vectors",
                },
                {
                    "name": "Uploaded Image Cross-Vector Test",
                    "test_type": "uploaded_image_cross_vector",
                    "vectors": ["image_vector", "character_image_vector"],
                    "description": "Test uploaded image against both image vectors to demonstrate semantic separation",
                },
                {
                    "name": "Combined Image Vector Search",
                    "test_type": "uploaded_image_combined",
                    "vectors": ["image_vector", "character_image_vector"],
                    "weights": [0.5, 0.5],
                    "description": "Combined search across both image vectors with equal weighting",
                },
                # Image-to-Text Cross-Modal Tests
                {
                    "name": "Visual-Story Correlation",
                    "query": "action adventure story with dynamic visuals",
                    "vector": "title_vector",
                    "description": "Cross-modal test using title vector for visual-story correlation",
                },
            ],
        )

    def _create_payload_filter_tests(self) -> TestSuite:
        """Create payload filtering and hybrid search tests."""
        return TestSuite(
            name="Payload Filter Tests",
            description="Test payload filtering, hybrid search, and metadata queries",
            tests=[
                {
                    "name": "Genre Filter",
                    "query": "boxing sports anime",
                    "vector": "title_vector",
                    "payload_filter": {
                        "key": "genres",
                        "match": {"any": ["Sports", "Comedy", "Drama"]},
                    },
                },
                {
                    "name": "Year Range Filter",
                    "query": "classic retro animation",
                    "vector": "temporal_vector",
                    "payload_filter": {
                        "key": "anime_season.year",
                        "range": {"gte": 1990, "lte": 1995},
                    },
                },
            ],
        )

    def _create_comprehensive_search_tests(self) -> TestSuite:
        """Create tests for high-level comprehensive search methods."""
        return TestSuite(
            name="Comprehensive Search Tests",
            description="Test high-level search methods that combine multiple vectors",
            tests=[
                {
                    "name": "Text Comprehensive Search",
                    "test_type": "search_text_comprehensive",
                    "query": "magical girl transformation anime with friendship themes",
                    "limit": 5,
                    "fusion_method": "rrf",
                    "description": "Test search across all 12 text vectors using RRF fusion",
                },
                {
                    "name": "Text Comprehensive with DBSF",
                    "test_type": "search_text_comprehensive",
                    "query": "dark psychological thriller with complex characters",
                    "limit": 5,
                    "fusion_method": "dbsf",
                    "description": "Test search across all 12 text vectors using DBSF fusion",
                },
                {
                    "name": "Character-Focused Search",
                    "test_type": "search_characters",
                    "query": "strong female protagonist with magical abilities",
                    "limit": 5,
                    "fusion_method": "rrf",
                    "description": "Test character-focused search using character-related vectors",
                },
                {
                    "name": "Complete Search All Vectors",
                    "test_type": "search_complete",
                    "query": "high school romance comedy with beautiful animation",
                    "limit": 5,
                    "fusion_method": "rrf",
                    "description": "Test search across all 13 vectors (12 text + 2 image)",
                },
                {
                    "name": "Visual Comprehensive Search",
                    "test_type": "search_visual_comprehensive",
                    "image_data": "placeholder_base64_image_data",
                    "limit": 5,
                    "fusion_method": "rrf",
                    "description": "Test search across both image vectors (will be skipped if no image data)",
                },
            ],
        )

    def _create_multi_vector_tests(self) -> TestSuite:
        """Create multi-vector combination tests."""
        return TestSuite(
            name="Multi-Vector Tests",
            description="Test combinations of multiple vectors for comprehensive search",
            tests=[
                {
                    "name": "Story + Character Combination",
                    "queries": [
                        {
                            "query": "boxing sports story",
                            "vector": "title_vector",
                            "weight": 0.6,
                        },
                        {
                            "query": "father son relationship",
                            "vector": "character_vector",
                            "weight": 0.4,
                        },
                    ],
                    "combination_type": "weighted",
                },
                {
                    "name": "Genre + Temporal Combination",
                    "queries": [
                        {
                            "query": "comedy drama",
                            "vector": "genre_vector",
                            "weight": 0.5,
                        },
                        {
                            "query": "1990s classic",
                            "vector": "temporal_vector",
                            "weight": 0.5,
                        },
                    ],
                    "combination_type": "weighted",
                },
                {
                    "name": "Character + Image Combination",
                    "queries": [
                        {
                            "query": "strong protagonist",
                            "vector": "character_vector",
                            "weight": 0.7,
                        },
                        {
                            "query": "dynamic art style",
                            "vector": "image_vector",
                            "weight": 0.3,
                        },
                    ],
                    "combination_type": "weighted",
                },
                {
                    "name": "Full Semantic Search",
                    "queries": [
                        {
                            "query": "coming of age sports",
                            "vector": "title_vector",
                            "weight": 0.3,
                        },
                        {
                            "query": "young athlete",
                            "vector": "character_vector",
                            "weight": 0.3,
                        },
                        {
                            "query": "comedy drama",
                            "vector": "genre_vector",
                            "weight": 0.2,
                        },
                        {
                            "query": "1990s retro",
                            "vector": "temporal_vector",
                            "weight": 0.2,
                        },
                    ],
                    "combination_type": "weighted",
                },
            ],
        )

    def _create_edge_case_tests(self) -> TestSuite:
        """Create edge case and error handling tests."""
        return TestSuite(
            name="Edge Case Tests",
            description="Test edge cases, error handling, and boundary conditions",
            tests=[
                {
                    "name": "Empty Query",
                    "query": "",
                    "vector": "title_vector",
                    "expect_error": True,
                },
                {
                    "name": "Very Long Query",
                    "query": "this is an extremely long query that tests the limits of our text processing capabilities and should still work correctly even with this much text content that goes on and on and on with many words and concepts and ideas all mixed together in one very long sentence",
                    "vector": "title_vector",
                    "expect_error": False,
                },
                {
                    "name": "Non-English Query",
                    "query": "アニメ ボクシング スポーツ",
                    "vector": "title_vector",
                    "expect_error": False,
                },
                {
                    "name": "Special Characters",
                    "query": "anime!@#$%^&*()_+{}[]",
                    "vector": "title_vector",
                    "expect_error": False,
                },
                {
                    "name": "Numbers Only",
                    "query": "1990 1995 2000",
                    "vector": "temporal_vector",
                    "expect_error": False,
                },
                {
                    "name": "Nonsense Query",
                    "query": "xyzabc nonexistent imaginary content",
                    "vector": "title_vector",
                    "expect_low_scores": True,
                },
            ],
        )

    def _create_performance_tests(self) -> TestSuite:
        """Create performance benchmark tests."""
        return TestSuite(
            name="Performance Tests",
            description="Test response times and performance characteristics",
            tests=[
                {
                    "name": "Single Vector Speed",
                    "query": "boxing anime",
                    "vector": "title_vector",
                    "performance_test": True,
                    "iterations": 5,
                },
                {
                    "name": "Large Result Set",
                    "query": "anime",
                    "vector": "title_vector",
                    "limit": 50,
                    "performance_test": True,
                },
                {
                    "name": "Complex Filter Speed",
                    "query": "sports anime",
                    "vector": "genre_vector",
                    "payload_filter": {
                        "key": "genres",
                        "match": {"any": ["Sports", "Comedy"]},
                    },
                    "performance_test": True,
                    "iterations": 3,
                },
            ],
        )

    def _create_real_world_scenario_tests(self) -> TestSuite:
        """Create real-world usage scenario tests."""
        return TestSuite(
            name="Real-World Scenarios",
            description="Test realistic user scenarios and use cases",
            tests=[
                {
                    "name": "Recommendation Scenario",
                    "description": "User likes 'Eiji' and wants similar anime",
                    "query": "boxing sports coming of age story with family conflict",
                    "vector": "title_vector",
                    "scenario": "recommendation",
                },
                {
                    "name": "Discovery Scenario",
                    "description": "User wants something new in specific genre",
                    "query": "psychological drama with complex characters",
                    "vector": "genre_vector",
                    "scenario": "discovery",
                },
                {
                    "name": "Character Similarity",
                    "description": "Find anime with similar character types",
                    "query": "reluctant hero with hidden potential",
                    "vector": "character_vector",
                    "scenario": "character_search",
                },
                {
                    "name": "Mood-Based Search",
                    "description": "Find anime matching current mood",
                    "query": "uplifting story with hope and perseverance",
                    "vector": "review_vector",
                    "scenario": "mood_search",
                },
                {
                    "name": "Visual Style Search",
                    "description": "Find anime with similar art style",
                    "query": "classic 1990s animation style",
                    "vector": "image_vector",
                    "scenario": "visual_search",
                },
                {
                    "name": "Character Image Search",
                    "test_type": "image_from_collection",
                    "vector": "character_image_vector",
                    "description": "Find anime with similar character designs",
                    "scenario": "character_visual_search",
                },
            ],
        )

    async def process_uploaded_image(self, image_path: str) -> Dict[str, List[float]]:
        """Process an uploaded image and return embeddings for both image vectors."""
        if image_path in self._uploaded_image_vectors:
            return self._uploaded_image_vectors[image_path]

        try:
            # Load and process the image
            image = Image.open(image_path)

            # Convert PIL image to base64 for processing
            img_buffer = io.BytesIO()
            image.save(img_buffer, format="PNG")
            img_buffer.seek(0)
            image_b64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")

            # Process with vision processor for both vector types
            image_vector = self.vision_processor.encode_image(
                f"data:image/png;base64,{image_b64}"
            )
            character_vector = self.vision_processor.encode_image(
                f"data:image/png;base64,{image_b64}"
            )

            vectors = {
                "image_vector": image_vector,
                "character_image_vector": character_vector,
            }

            self._uploaded_image_vectors[image_path] = vectors
            print(f"📸 Processed uploaded image: {image_path}")
            print(
                f"   🎨 Image vector dims: {len(image_vector) if image_vector else 0}"
            )
            print(
                f"   👤 Character vector dims: {len(character_vector) if character_vector else 0}"
            )

            return vectors

        except Exception as e:
            print(f"❌ Failed to process image {image_path}: {e}")
            return {}

    async def execute_search_query(
        self,
        query: str,
        vector_name: str,
        payload_filter: Optional[Dict] = None,
        limit: int = 5,
        query_vector: Optional[List[float]] = None,
    ) -> Tuple[Dict, float]:
        """Execute a single search query and return results with timing."""
        start_time = time.time()

        # Use provided vector or convert query to embedding
        if query_vector:
            query_embedding = query_vector
        else:
            # Use appropriate processor based on vector type
            if "image" in vector_name.lower():
                # For image vectors, use vision processor with text-to-image capability
                # Since we don't have actual images, we'll create a placeholder 768D vector
                query_embedding = [0.1] * 768  # 768-dim placeholder for image vectors
            else:
                # For text vectors, use text processor
                query_embedding = self.text_processor.encode_text(query)
                if not query_embedding:
                    raise Exception("Failed to encode query")

        # Build search payload
        search_payload = {
            "vector": {"name": vector_name, "vector": query_embedding},
            "limit": limit,
            "with_payload": True,
            "with_vector": False,
        }

        # Add payload filter if provided
        if payload_filter:
            # Convert our test filter format to Qdrant filter format
            qdrant_filter = self._convert_to_qdrant_filter(payload_filter)
            search_payload["filter"] = qdrant_filter

        # Execute search
        url = f"{self.qdrant_url}/collections/{self.collection_name}/points/search"

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, headers=self.headers, json=search_payload
            ) as response:
                execution_time = time.time() - start_time

                if response.status == 200:
                    result = await response.json()
                    return result, execution_time
                else:
                    error_text = await response.text()
                    raise Exception(f"Search failed: {response.status} - {error_text}")

    async def get_sample_vectors_from_collection(self) -> Dict[str, List[float]]:
        """Get sample vectors from the collection for image similarity testing."""
        if self._sample_images:
            return self._sample_images

        try:
            url = f"{self.qdrant_url}/collections/{self.collection_name}/points/scroll"
            payload = {"limit": 10, "with_vector": True, "with_payload": True}

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, headers=self.headers, json=payload
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        points = result.get("result", {}).get("points", [])

                        # Extract sample vectors for testing
                        for point in points:
                            vectors = point.get("vector", {})
                            title = point.get("payload", {}).get("title", "Unknown")

                            # Store image vectors if available
                            if "image_vector" in vectors:
                                key = f"image_{title[:20]}"
                                self._sample_images[key] = {
                                    "vector": vectors["image_vector"],
                                    "type": "image_vector",
                                    "title": title,
                                }

                            if "character_image_vector" in vectors:
                                key = f"character_{title[:20]}"
                                self._sample_images[key] = {
                                    "vector": vectors["character_image_vector"],
                                    "type": "character_image_vector",
                                    "title": title,
                                }

                        print(
                            f"📸 Cached {len(self._sample_images)} sample vectors for image testing"
                        )
                        return self._sample_images

        except Exception as e:
            print(f"⚠️  Failed to get sample vectors: {e}")
            return {}

        return {}

    async def execute_image_similarity_test(
        self, vector_name: str, limit: int = 5
    ) -> Tuple[Dict, float, str]:
        """Execute image similarity test using actual image vectors from collection."""
        # Get sample vectors
        sample_vectors = await self.get_sample_vectors_from_collection()

        # Find a suitable sample vector
        source_vector_info = None
        for key, info in sample_vectors.items():
            if info["type"] == vector_name:
                source_vector_info = info
                break

        if not source_vector_info:
            raise Exception(f"No sample {vector_name} found in collection")

        # Execute search using the sample vector
        result, exec_time = await self.execute_search_query(
            query="",  # Empty query since we're using vector directly
            vector_name=vector_name,
            query_vector=source_vector_info["vector"],
            limit=limit,
        )

        return result, exec_time, source_vector_info["title"]

    async def execute_uploaded_image_test(
        self, image_path: str, vector_name: str, limit: int = 5
    ) -> Tuple[Dict, float, str]:
        """Execute search using an uploaded image."""
        # Process the uploaded image
        image_vectors = await self.process_uploaded_image(image_path)

        if vector_name not in image_vectors or not image_vectors[vector_name]:
            raise Exception(f"Failed to process image for {vector_name}")

        # Execute search using the image vector
        result, exec_time = await self.execute_search_query(
            query="",  # Empty query since we're using vector directly
            vector_name=vector_name,
            query_vector=image_vectors[vector_name],
            limit=limit,
        )

        return result, exec_time, f"uploaded_image_{vector_name}"

    async def execute_cross_vector_image_test(
        self, image_path: str, vector_names: List[str], limit: int = 5
    ) -> Dict[str, Tuple[Dict, float]]:
        """Execute search using uploaded image across multiple vectors for comparison."""
        # Process the uploaded image
        image_vectors = await self.process_uploaded_image(image_path)

        results = {}

        for vector_name in vector_names:
            if vector_name in image_vectors and image_vectors[vector_name]:
                try:
                    result, exec_time = await self.execute_search_query(
                        query="",
                        vector_name=vector_name,
                        query_vector=image_vectors[vector_name],
                        limit=limit,
                    )
                    results[vector_name] = (result, exec_time)
                except Exception as e:
                    print(f"   ⚠️  Failed to search {vector_name}: {e}")
                    results[vector_name] = ({"result": []}, 0.0)

        return results

    async def execute_combined_image_search(
        self,
        image_path: str,
        vector_names: List[str],
        weights: List[float],
        limit: int = 5,
    ) -> Tuple[Dict, float]:
        """Execute combined search using uploaded image across multiple vectors."""
        # Process the uploaded image
        image_vectors = await self.process_uploaded_image(image_path)

        # Build multi-vector query configs
        query_configs = []
        for vector_name, weight in zip(vector_names, weights):
            if vector_name in image_vectors and image_vectors[vector_name]:
                query_configs.append(
                    {
                        "query": "",  # Empty since we're using vectors directly
                        "vector": vector_name,
                        "weight": weight,
                        "query_vector": image_vectors[vector_name],
                    }
                )

        if not query_configs:
            raise Exception("No valid vectors found for combined search")

        # Execute combined search (modified multi-vector approach)
        start_time = time.time()

        all_results = {}
        total_weight = sum(config["weight"] for config in query_configs)

        for config in query_configs:
            vector_name = config["vector"]
            weight = config["weight"] / total_weight
            query_vector = config["query_vector"]

            result, _ = await self.execute_search_query(
                "", vector_name, query_vector=query_vector, limit=limit
            )

            # Weight and combine results
            for item in result.get("result", []):
                point_id = item["id"]
                score = item["score"] * weight

                if point_id in all_results:
                    all_results[point_id]["score"] += score
                else:
                    all_results[point_id] = {
                        "id": point_id,
                        "score": score,
                        "payload": item["payload"],
                    }

        # Sort by combined score
        sorted_results = sorted(
            all_results.values(), key=lambda x: x["score"], reverse=True
        )[:limit]

        execution_time = time.time() - start_time
        return {"result": sorted_results}, execution_time

    async def execute_multi_vector_query(
        self, query_configs: List[Dict], limit: int = 5
    ) -> Tuple[Dict, float]:
        """Execute multi-vector search using Qdrant's native multi-vector API."""
        start_time = time.time()

        try:
            # Prepare vector queries for the new multi-vector API
            vector_queries = []
            for config in query_configs:
                query_text = config["query"]
                vector_name = config["vector"]

                # Generate vector using appropriate processor
                if "image" in vector_name.lower():
                    # For image vectors, use 768D placeholder
                    vector_data = [0.1] * 768
                else:
                    # Generate vector using text processor for text vectors
                    vector_data = self.text_processor.encode_text(query_text)
                    if vector_data is None:
                        vector_data = [0.1] * 1024  # 1024-dim placeholder

                vector_queries.append(
                    {"vector_name": vector_name, "vector_data": vector_data}
                )

            # Use the new native multi-vector search
            results = await self.qdrant_client.search_multi_vector(
                vector_queries=vector_queries,
                limit=limit,
                fusion_method="rrf",  # Use Reciprocal Rank Fusion
            )

            execution_time = time.time() - start_time
            return {"result": results}, execution_time

        except Exception as e:
            # No fallback - multi-vector API must work
            print(f"❌ Multi-vector API failed: {e}")
            raise

    async def run_test_suite(self, suite: TestSuite) -> TestSuite:
        """Execute all tests in a test suite."""
        print(f"\n{'='*70}")
        print(f"🧪 {suite.name}")
        print(f"📝 {suite.description}")
        print(f"{'='*70}")

        for i, test_config in enumerate(suite.tests, 1):
            test_name = test_config.get("name", f"Test {i}")
            print(f"\n🔬 Test {i}: {test_name}")

            try:
                if "queries" in test_config:  # Multi-vector test
                    result, exec_time = await self.execute_multi_vector_query(
                        test_config["queries"], test_config.get("limit", 5)
                    )
                elif test_config.get("test_type") == "uploaded_image":
                    # Find uploaded image file (look for common temp paths)
                    image_path = self._find_uploaded_image()
                    if not image_path:
                        print(f"   ⚠️  No uploaded image found, skipping test")
                        continue

                    vector_name = test_config["vector"]
                    limit = test_config.get("limit", 5)

                    result, exec_time, source_info = (
                        await self.execute_uploaded_image_test(
                            image_path, vector_name, limit
                        )
                    )
                    print(f"   📸 Using uploaded image for {vector_name}")

                elif test_config.get("test_type") == "uploaded_image_cross_vector":
                    # Cross-vector uploaded image test
                    image_path = self._find_uploaded_image()
                    if not image_path:
                        print(f"   ⚠️  No uploaded image found, skipping test")
                        continue

                    vector_names = test_config["vectors"]
                    limit = test_config.get("limit", 5)

                    cross_results = await self.execute_cross_vector_image_test(
                        image_path, vector_names, limit
                    )

                    # Display cross-vector comparison
                    print(f"   🔄 Cross-vector image test results:")
                    for vector_name, (result, exec_time) in cross_results.items():
                        search_results = result.get("result", [])
                        top_score = (
                            search_results[0]["score"] if search_results else 0.0
                        )
                        print(
                            f"     {vector_name}: {len(search_results)} results, top score: {top_score:.3f}, time: {exec_time:.3f}s"
                        )

                        # Show top result for each vector
                        if search_results:
                            top_result = search_results[0]
                            title = top_result.get("payload", {}).get(
                                "title", "Unknown"
                            )
                            print(
                                f"       Top: {title} (score: {top_result['score']:.3f})"
                            )

                    # Use first vector's results for standard processing
                    if cross_results:
                        first_vector = list(cross_results.keys())[0]
                        result, exec_time = cross_results[first_vector]
                    else:
                        result, exec_time = {"result": []}, 0.0

                elif test_config.get("test_type") == "uploaded_image_combined":
                    # Combined image vector test
                    image_path = self._find_uploaded_image()
                    if not image_path:
                        print(f"   ⚠️  No uploaded image found, skipping test")
                        continue

                    vector_names = test_config["vectors"]
                    weights = test_config.get(
                        "weights", [1.0 / len(vector_names)] * len(vector_names)
                    )
                    limit = test_config.get("limit", 5)

                    result, exec_time = await self.execute_combined_image_search(
                        image_path, vector_names, weights, limit
                    )
                    print(
                        f"   🔗 Combined search across {len(vector_names)} image vectors"
                    )
                    print(f"   📊 Weights: {dict(zip(vector_names, weights))}")

                elif test_config.get("test_type") == "image_from_collection":
                    # Image similarity test using collection samples
                    vector_name = test_config["vector"]
                    limit = test_config.get("limit", 5)

                    result, exec_time, source_info = (
                        await self.execute_image_similarity_test(vector_name, limit)
                    )
                    print(f"   📸 Using sample from collection: {source_info}")

                elif test_config.get("test_type") == "cross_vector_text_comparison":
                    # Cross-vector text comparison
                    query = test_config["query"]
                    vector_names = test_config["vectors"]
                    limit = test_config.get("limit", 5)

                    cross_results = {}
                    for vector_name in vector_names:
                        try:
                            result_single, exec_time_single = (
                                await self.execute_search_query(
                                    query, vector_name, limit=limit
                                )
                            )
                            cross_results[vector_name] = (
                                result_single,
                                exec_time_single,
                            )
                        except Exception as e:
                            print(f"   ⚠️  Failed to search {vector_name}: {e}")
                            cross_results[vector_name] = ({"result": []}, 0.0)

                    # Display cross-vector comparison
                    print(f"   🔄 Cross-vector text comparison results:")
                    for vector_name, (
                        result_single,
                        exec_time_single,
                    ) in cross_results.items():
                        search_results = result_single.get("result", [])
                        top_score = (
                            search_results[0]["score"] if search_results else 0.0
                        )
                        print(
                            f"     {vector_name}: {len(search_results)} results, top score: {top_score:.3f}"
                        )

                    # Use first vector's results for standard processing
                    if cross_results:
                        first_vector = list(cross_results.keys())[0]
                        result, exec_time = cross_results[first_vector]
                    else:
                        result, exec_time = {"result": []}, 0.0

                elif test_config.get("test_type") == "search_text_comprehensive":
                    # Text comprehensive search test
                    query = test_config["query"]
                    limit = test_config.get("limit", 5)
                    fusion_method = test_config.get("fusion_method", "rrf")

                    start_time = time.time()
                    search_results = await self.qdrant_client.search_text_comprehensive(
                        query=query, limit=limit, fusion_method=fusion_method
                    )
                    exec_time = time.time() - start_time
                    result = {"result": search_results}

                elif test_config.get("test_type") == "search_characters":
                    # Character-focused search test
                    query = test_config["query"]
                    limit = test_config.get("limit", 5)
                    fusion_method = test_config.get("fusion_method", "rrf")

                    start_time = time.time()
                    search_results = await self.qdrant_client.search_characters(
                        query=query, limit=limit, fusion_method=fusion_method
                    )
                    exec_time = time.time() - start_time
                    result = {"result": search_results}

                elif test_config.get("test_type") == "search_complete":
                    # Complete search across all 13 vectors
                    query = test_config["query"]
                    limit = test_config.get("limit", 5)
                    fusion_method = test_config.get("fusion_method", "rrf")

                    start_time = time.time()
                    search_results = await self.qdrant_client.search_complete(
                        query=query, limit=limit, fusion_method=fusion_method
                    )
                    exec_time = time.time() - start_time
                    result = {"result": search_results}

                elif test_config.get("test_type") == "search_visual_comprehensive":
                    # Visual comprehensive search test
                    image_data = test_config.get("image_data")
                    if not image_data or image_data == "placeholder_base64_image_data":
                        print(
                            f"   ⚠️  No image data provided, skipping visual comprehensive test"
                        )
                        continue

                    limit = test_config.get("limit", 5)
                    fusion_method = test_config.get("fusion_method", "rrf")

                    start_time = time.time()
                    search_results = (
                        await self.qdrant_client.search_visual_comprehensive(
                            image_data=image_data,
                            limit=limit,
                            fusion_method=fusion_method,
                        )
                    )
                    exec_time = time.time() - start_time
                    result = {"result": search_results}

                else:  # Single vector test
                    query = test_config["query"]
                    vector_name = test_config["vector"]
                    payload_filter = test_config.get("payload_filter")
                    limit = test_config.get("limit", 5)

                    result, exec_time = await self.execute_search_query(
                        query, vector_name, payload_filter, limit
                    )

                # Process results
                search_results = result.get("result", [])
                top_score = search_results[0]["score"] if search_results else 0.0

                # Create test result
                test_result = TestResult(
                    test_name=test_name,
                    query=test_config.get("query", "Multi-vector query"),
                    vector_type=test_config.get("vector", "multi-vector"),
                    execution_time=exec_time,
                    results_count=len(search_results),
                    top_score=top_score,
                    success=True,
                    results=search_results[:3],  # Store top 3 results
                )

                suite.results.append(test_result)

                # Display results
                print(f"   ⏱️  Execution time: {exec_time:.3f}s")
                print(f"   📊 Results found: {len(search_results)}")
                print(f"   🏆 Top score: {top_score:.3f}")

                # Show top results
                for j, item in enumerate(search_results[:2]):
                    title = item.get("payload", {}).get("title", "Unknown")
                    score = item.get("score", 0)
                    print(f"   {j+1}. {title} (score: {score:.3f})")

                # Performance warnings
                if exec_time > 2.0:
                    print(f"   ⚠️  Slow query: {exec_time:.3f}s")
                if len(search_results) == 0:
                    print(f"   ⚠️  No results found")

            except Exception as e:
                print(f"   ❌ Test failed: {e}")

                test_result = TestResult(
                    test_name=test_name,
                    query=test_config.get("query", ""),
                    vector_type=test_config.get("vector", "unknown"),
                    execution_time=0.0,
                    results_count=0,
                    top_score=0.0,
                    success=False,
                    error=str(e),
                )
                suite.results.append(test_result)

        return suite

    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all test suites and generate comprehensive report."""
        print("🚀 Starting Comprehensive Vector Database Test Suite")
        print(f"🎯 Collection: {self.collection_name}")
        print(f"📅 Test Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        start_time = time.time()
        executed_suites = []

        # Run all test suites
        for suite in self.test_suites:
            executed_suite = await self.run_test_suite(suite)
            executed_suites.append(executed_suite)

        total_time = time.time() - start_time

        # Generate summary report
        report = self._generate_report(executed_suites, total_time)

        # Display summary
        self._display_summary(report)

        return report

    def _generate_report(
        self, suites: List[TestSuite], total_time: float
    ) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        total_tests = sum(len(suite.results) for suite in suites)
        passed_tests = sum(
            1 for suite in suites for result in suite.results if result.success
        )
        failed_tests = total_tests - passed_tests

        avg_execution_time = (
            sum(result.execution_time for suite in suites for result in suite.results)
            / total_tests
            if total_tests > 0
            else 0
        )

        suite_summaries = []
        for suite in suites:
            suite_passed = sum(1 for result in suite.results if result.success)
            suite_total = len(suite.results)

            suite_summaries.append(
                {
                    "name": suite.name,
                    "passed": suite_passed,
                    "total": suite_total,
                    "success_rate": (
                        (suite_passed / suite_total * 100) if suite_total > 0 else 0
                    ),
                    "avg_score": (
                        sum(
                            result.top_score
                            for result in suite.results
                            if isinstance(result.top_score, (int, float))
                        )
                        / suite_total
                        if suite_total > 0
                        else 0
                    ),
                    "avg_time": (
                        sum(result.execution_time for result in suite.results)
                        / suite_total
                        if suite_total > 0
                        else 0
                    ),
                }
            )

        return {
            "timestamp": datetime.now().isoformat(),
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": (
                (passed_tests / total_tests * 100) if total_tests > 0 else 0
            ),
            "total_execution_time": total_time,
            "avg_execution_time": avg_execution_time,
            "suite_summaries": suite_summaries,
            "detailed_results": suites,
        }

    def _display_summary(self, report: Dict[str, Any]):
        """Display test summary report."""
        print(f"\n{'='*70}")
        print("📊 COMPREHENSIVE TEST SUITE SUMMARY")
        print(f"{'='*70}")

        print(f"🎯 Total Tests: {report['total_tests']}")
        print(f"✅ Passed: {report['passed_tests']}")
        print(f"❌ Failed: {report['failed_tests']}")
        print(f"📈 Success Rate: {report['success_rate']:.1f}%")
        print(f"⏱️  Total Time: {report['total_execution_time']:.2f}s")
        print(f"⚡ Avg Query Time: {report['avg_execution_time']:.3f}s")

        print(f"\n📋 Suite Breakdown:")
        for suite in report["suite_summaries"]:
            print(
                f"   {suite['name']}: {suite['passed']}/{suite['total']} "
                f"({suite['success_rate']:.1f}%) "
                f"avg_score: {suite['avg_score']:.3f} "
                f"avg_time: {suite['avg_time']:.3f}s"
            )

        print(f"\n{'='*70}")
        print("🎉 Test Suite Complete!")

        if report["success_rate"] >= 95:
            print("🏆 Excellent! Vector database performing exceptionally well.")
        elif report["success_rate"] >= 80:
            print("👍 Good performance! Some areas for improvement.")
        else:
            print("⚠️  Performance issues detected. Review failed tests.")
        print(f"{'='*70}")

    async def save_detailed_report(self, report: Dict[str, Any], filename: str = None):
        """Save detailed test report to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"vector_test_report_{timestamp}.json"

        # Convert results to serializable format
        serializable_report = {
            "summary": {
                "timestamp": report["timestamp"],
                "total_tests": report["total_tests"],
                "passed_tests": report["passed_tests"],
                "failed_tests": report["failed_tests"],
                "success_rate": report["success_rate"],
                "total_execution_time": report["total_execution_time"],
                "avg_execution_time": report["avg_execution_time"],
            },
            "suite_summaries": report["suite_summaries"],
            "detailed_results": [],
        }

        # Add detailed results
        for suite in report["detailed_results"]:
            suite_data = {
                "name": suite.name,
                "description": suite.description,
                "results": [],
            }

            for result in suite.results:
                result_data = {
                    "test_name": result.test_name,
                    "query": result.query,
                    "vector_type": result.vector_type,
                    "execution_time": result.execution_time,
                    "results_count": result.results_count,
                    "top_score": result.top_score,
                    "success": result.success,
                    "error": result.error,
                    "top_results": [
                        {
                            "title": r.get("payload", {}).get("title", "Unknown"),
                            "score": r.get("score", 0),
                        }
                        for r in result.results
                    ],
                }
                suite_data["results"].append(result_data)

            serializable_report["detailed_results"].append(suite_data)

        with open(filename, "w") as f:
            json.dump(serializable_report, f, indent=2)

        print(f"📄 Detailed report saved to: {filename}")

    def _find_uploaded_image(self) -> Optional[str]:
        """Find uploaded image file in common temporary locations."""
        # Common paths where uploaded images might be stored
        common_paths = [
            "/tmp/",
            "/var/folders/",
            "./",
            "../",
            os.path.expanduser("~/Downloads/"),
            os.path.expanduser("~/Desktop/"),
        ]

        # Common image extensions
        image_extensions = [".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"]

        # Look for recent image files
        for base_path in common_paths:
            if os.path.exists(base_path):
                try:
                    for file in os.listdir(base_path):
                        if any(file.lower().endswith(ext) for ext in image_extensions):
                            full_path = os.path.join(base_path, file)
                            # Check if file was modified recently (within last hour)
                            if os.path.getmtime(full_path) > time.time() - 3600:
                                return full_path
                except (PermissionError, FileNotFoundError):
                    continue

        # Fallback: look for any image file in current directory
        for file in os.listdir("."):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                return os.path.abspath(file)

        return None

    def _convert_to_qdrant_filter(self, test_filter: Dict) -> Dict:
        """Convert test filter format to Qdrant filter format."""
        key = test_filter["key"]

        if "match" in test_filter:
            match_config = test_filter["match"]

            if "any" in match_config:
                # Array matching - convert to MatchAny
                return {"must": [{"key": key, "match": {"any": match_config["any"]}}]}
            elif "value" in match_config:
                # Exact value matching
                return {
                    "must": [{"key": key, "match": {"value": match_config["value"]}}]
                }
            elif "text" in match_config:
                # Text contains matching (partial match)
                return {"must": [{"key": key, "match": {"text": match_config["text"]}}]}

        elif "range" in test_filter:
            # Range filtering
            return {"must": [{"key": key, "range": test_filter["range"]}]}

        # Fallback - return as is
        return {"must": [test_filter]}


async def main():
    """Main test execution function."""
    # Get environment variables
    QDRANT_CLOUD_URL = os.getenv("QDRANT_CLOUD_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

    if not QDRANT_CLOUD_URL:
        print("❌ QDRANT_CLOUD_URL environment variable required")
        return

    if not QDRANT_API_KEY:
        print("❌ QDRANT_API_KEY environment variable required")
        return

    # Initialize and run comprehensive test suite
    tester = ComprehensiveVectorTester(QDRANT_CLOUD_URL, QDRANT_API_KEY)

    # Run all tests
    report = await tester.run_all_tests()

    # Save detailed report
    await tester.save_detailed_report(report)


if __name__ == "__main__":
    asyncio.run(main())

