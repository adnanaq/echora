#!/usr/bin/env python3
"""Dataset Analyzer for Dynamic Validation Query Generation.

This module analyzes the actual dataset content to generate realistic
validation queries instead of using hardcoded assumptions.
"""

import asyncio
import logging
from collections import Counter
from typing import Any, Dict, List

from ..vector.client.qdrant_client import QdrantClient
from .vector_field_mapping import (
    get_searchable_vectors,
    get_vector_description,
    get_vector_fields,
    is_vector_populated,
)

logger = logging.getLogger(__name__)


class DatasetAnalyzer:
    """Analyzes dataset content to generate dynamic validation queries."""

    def __init__(self, qdrant_client: QdrantClient):
        """Initialize the dataset analyzer.

        Args:
            qdrant_client: The Qdrant client instance to analyze
        """
        self.client = qdrant_client
        self.dataset_profile: Dict[str, Any] = {}

    async def analyze_dataset(self) -> Dict[str, Any]:
        """Analyze the dataset to understand its content characteristics.

        Returns:
            Dataset profile with statistics and content analysis
        """
        try:
            logger.info("🔍 Analyzing dataset content for dynamic validation")

            # Get all points from the collection
            points_data = await self._sample_dataset_points()

            if not points_data:
                logger.warning("No data points found in dataset")
                return {"error": "No data points found"}

            # Analyze different aspects of the dataset
            content_analysis = self._analyze_content_characteristics(points_data)
            vector_analysis = await self._analyze_vector_populations(points_data)
            genre_analysis = self._analyze_genres(points_data)
            type_analysis = self._analyze_content_types(points_data)
            status_analysis = self._analyze_status_distribution(points_data)

            self.dataset_profile = {
                "total_points": len(points_data),
                "content_characteristics": content_analysis,
                "vector_populations": vector_analysis,
                "genre_distribution": genre_analysis,
                "type_distribution": type_analysis,
                "status_distribution": status_analysis,
                "sample_titles": self._extract_sample_titles(points_data),
                "timestamp": asyncio.get_event_loop().time(),
            }

            logger.info(
                f"✅ Dataset analysis complete: {len(points_data)} points analyzed"
            )
            return self.dataset_profile

        except Exception as e:
            logger.error(f"Failed to analyze dataset: {e}")
            return {"error": str(e)}

    async def generate_dynamic_queries(self) -> Dict[str, List[Dict[str, Any]]]:
        """Generate validation queries based on actual dataset content.

        Returns:
            Dynamic test queries for each vector type
        """
        try:
            if not self.dataset_profile:
                await self.analyze_dataset()

            if "error" in self.dataset_profile:
                return {"error": self.dataset_profile["error"]}

            dynamic_queries: Dict[str, List[Dict[str, Any]]] = {}

            # Generate queries for searchable vectors based on field mapping
            searchable_vectors = get_searchable_vectors()
            populated_vectors = self.dataset_profile.get("vector_populations", {})

            for vector_name in searchable_vectors:
                # Check if vector is actually populated in the dataset
                population_pct = populated_vectors.get(vector_name, {}).get(
                    "population_percentage", 0
                )

                if not is_vector_populated(vector_name):
                    logger.info(
                        f"Skipping {vector_name} - known to be empty in dataset type"
                    )
                    dynamic_queries[vector_name] = []
                    continue

                if population_pct < 10:
                    logger.info(
                        f"Skipping {vector_name} - only {population_pct:.1f}% populated"
                    )
                    dynamic_queries[vector_name] = []
                    continue

                # Generate queries based on vector type and indexed fields
                if vector_name == "title_vector":
                    dynamic_queries[vector_name] = self._generate_title_queries()
                elif vector_name == "genre_vector":
                    dynamic_queries[vector_name] = self._generate_genre_queries()
                elif vector_name == "temporal_vector":
                    dynamic_queries[vector_name] = self._generate_temporal_queries()
                elif vector_name == "episode_vector":
                    dynamic_queries[vector_name] = self._generate_episode_queries()
                elif vector_name == "character_vector":
                    dynamic_queries[vector_name] = self._generate_character_queries()
                elif vector_name == "franchise_vector":
                    dynamic_queries[vector_name] = self._generate_franchise_queries()
                elif vector_name == "staff_vector":
                    dynamic_queries[vector_name] = self._generate_staff_queries()
                elif vector_name == "related_vector":
                    dynamic_queries[vector_name] = self._generate_related_queries()
                else:
                    # Generic query generation for other vectors
                    dynamic_queries[vector_name] = self._generate_generic_queries(
                        vector_name
                    )

            logger.info(
                f"✅ Generated dynamic queries for {len(dynamic_queries)} vector types"
            )
            return dynamic_queries

        except Exception as e:
            logger.error(f"Failed to generate dynamic queries: {e}")
            return {"error": str(e)}

    async def _sample_dataset_points(
        self, sample_size: int = 100
    ) -> List[Dict[str, Any]]:
        """Sample points from the dataset for analysis."""
        try:
            # Use scroll to get all points (for small datasets) or sample
            points_result = self.client.client.scroll(
                collection_name=self.client.collection_name,
                limit=sample_size,
                with_payload=True,
                with_vectors=True,
            )

            points_data = []
            for point in points_result[0]:
                point_dict = {
                    "id": str(point.id),
                    "payload": dict(point.payload) if point.payload else {},
                    "vector": point.vector if point.vector else {},
                }
                points_data.append(point_dict)

            return points_data

        except Exception as e:
            logger.error(f"Failed to sample dataset points: {e}")
            return []

    def _analyze_content_characteristics(
        self, points_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze general content characteristics."""
        characteristics = {
            "has_synopsis": 0,
            "has_characters": 0,
            "has_staff": 0,
            "total_titles": 0,
            "unique_studios": set(),
            "content_themes": Counter(),
        }

        for point in points_data:
            payload = point.get("payload", {})

            characteristics["total_titles"] += 1

            if payload.get("synopsis"):
                characteristics["has_synopsis"] += 1

            # Check for studios - Fix: Read from correct location in staff_data
            staff_data = payload.get("staff_data", {})
            studios = []
            if isinstance(staff_data, dict) and "studios" in staff_data:
                studios = [
                    studio.get("name", "")
                    for studio in staff_data.get("studios", [])
                    if isinstance(studio, dict)
                ]

            if studios:
                characteristics["unique_studios"].update(studios)

            # Analyze title for themes
            title = payload.get("title", "").lower()
            if "music" in title or "song" in title:
                characteristics["content_themes"]["music"] += 1
            if "commercial" in title or "ad" in title:
                characteristics["content_themes"]["commercial"] += 1
            if "short" in title or "special" in title:
                characteristics["content_themes"]["short"] += 1

        characteristics["unique_studios"] = list(characteristics["unique_studios"])
        characteristics["content_themes"] = dict(characteristics["content_themes"])

        return characteristics

    async def _analyze_vector_populations(
        self, points_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze which vectors are populated with meaningful data."""
        vector_stats = {}

        vector_names = [
            "title_vector",
            "episode_vector",
            "character_vector",
            "franchise_vector",
            "genre_vector",
            "staff_vector",
            "temporal_vector",
            "related_vector",
            "streaming_vector",
            "review_vector",
            "image_vector",
            "character_image_vector",
        ]

        for vector_name in vector_names:
            populated_count = 0
            total_count = 0

            for point in points_data:
                vectors = point.get("vector", {})
                total_count += 1

                if vector_name in vectors:
                    vector = vectors[vector_name]
                    if isinstance(vector, list) and len(vector) > 0:
                        # Check if vector has meaningful data (not all zeros)
                        non_zero_count = sum(1 for v in vector if abs(v) > 0.001)
                        if non_zero_count > len(vector) * 0.1:  # At least 10% non-zero
                            populated_count += 1

            population_percentage = (
                (populated_count / total_count * 100) if total_count > 0 else 0
            )

            vector_stats[vector_name] = {
                "populated_count": populated_count,
                "total_count": total_count,
                "population_percentage": population_percentage,
                "is_meaningful": population_percentage > 10,
            }

        return vector_stats

    def _analyze_genres(self, points_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze genre distribution in the dataset."""
        genre_counter = Counter()
        demographics_counter = Counter()

        for point in points_data:
            payload = point.get("payload", {})

            genres = payload.get("genres", [])
            if isinstance(genres, list):
                genre_counter.update(genres)

            demographics = payload.get("demographics", [])
            if isinstance(demographics, list):
                demographics_counter.update(demographics)

        return {
            "top_genres": dict(genre_counter.most_common(10)),
            "top_demographics": dict(demographics_counter.most_common(5)),
            "total_unique_genres": len(genre_counter),
            "total_unique_demographics": len(demographics_counter),
        }

    def _analyze_content_types(
        self, points_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze content type distribution."""
        type_counter = Counter()

        for point in points_data:
            payload = point.get("payload", {})
            content_type = payload.get("type", "Unknown")
            type_counter[content_type] += 1

        return {
            "type_distribution": dict(type_counter),
            "most_common_type": (
                type_counter.most_common(1)[0] if type_counter else ("Unknown", 0)
            ),
        }

    def _analyze_status_distribution(
        self, points_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze status distribution."""
        status_counter = Counter()

        for point in points_data:
            payload = point.get("payload", {})
            status = payload.get("status", "Unknown")
            status_counter[status] += 1

        return {
            "status_distribution": dict(status_counter),
            "most_common_status": (
                status_counter.most_common(1)[0] if status_counter else ("Unknown", 0)
            ),
        }

    def _extract_sample_titles(
        self, points_data: List[Dict[str, Any]], limit: int = 10
    ) -> List[str]:
        """Extract sample titles for reference."""
        titles = []
        for point in points_data[:limit]:
            payload = point.get("payload", {})
            title = payload.get("title", "Unknown")
            titles.append(title)
        return titles

    def _generate_title_queries(self) -> List[Dict[str, Any]]:
        """Generate title-based validation queries."""
        queries = []
        sample_titles = self.dataset_profile.get("sample_titles", [])

        # Use actual titles from the dataset with better query generation
        for i, title in enumerate(sample_titles[:3]):
            if title and title != "Unknown":
                # Remove quotes and special characters for better matching
                clean_title = title.replace('"', "").replace("♪", "").replace("!", "")
                title_words = clean_title.split()

                if len(title_words) >= 2:
                    # Use meaningful words, skip common articles
                    meaningful_words = [
                        word
                        for word in title_words
                        if len(word) > 2
                        and word.lower()
                        not in {"the", "and", "in", "of", "to", "wo", "no", "ga", "wa"}
                    ]

                    if meaningful_words:
                        # Use 2-3 meaningful words for better semantic matching
                        if len(meaningful_words) >= 2:
                            query_term = " ".join(meaningful_words[:2])
                        else:
                            query_term = meaningful_words[0]

                        queries.append(
                            {
                                "query": query_term,
                                "expected_titles": [title],
                                "min_results": 1,
                            }
                        )

        return queries

    def _generate_genre_queries(self) -> List[Dict[str, Any]]:
        """Generate genre-based validation queries."""
        queries = []
        genre_dist = self.dataset_profile.get("genre_distribution", {})
        top_genres = genre_dist.get("top_genres", {})

        for genre, count in list(top_genres.items())[:3]:
            if count >= 2:  # Only test genres with at least 2 entries
                queries.append(
                    {
                        "query": genre.lower(),
                        "expected_genres": [genre],
                        "min_results": min(count, 2),
                    }
                )

        return queries

    def _generate_technical_queries(self) -> List[Dict[str, Any]]:
        """Generate technical-based validation queries."""
        queries = []
        type_dist = self.dataset_profile.get("type_distribution", {})
        type_distribution = type_dist.get("type_distribution", {})

        for content_type, count in type_distribution.items():
            if count >= 2 and content_type != "Unknown":
                queries.append(
                    {
                        "query": f"{content_type.lower()} format",
                        "expected_types": [content_type],
                        "min_results": min(count, 2),
                    }
                )

        return queries

    def _generate_temporal_queries(self) -> List[Dict[str, Any]]:
        """Generate temporal-based validation queries."""
        queries = []
        status_dist = self.dataset_profile.get("status_distribution", {})
        status_distribution = status_dist.get("status_distribution", {})

        for status, count in status_distribution.items():
            if count >= 3 and status != "Unknown":
                queries.append(
                    {
                        "query": f"{status.lower().replace('_', ' ')}",
                        "expected_status": [status],
                        "min_results": min(count - 1, 3),
                    }
                )

        return queries

    def _generate_episode_queries(self) -> List[Dict[str, Any]]:
        """Generate episode-based validation queries."""
        return [{"query": "single episode", "expected_episodes": [1], "min_results": 3}]

    def _generate_character_queries(self) -> List[Dict[str, Any]]:
        """Generate character-based validation queries."""
        return [
            {
                "query": "main character",
                "expected_content": ["character", "protagonist"],
                "min_results": 1,
            }
        ]

    def _generate_franchise_queries(self) -> List[Dict[str, Any]]:
        """Generate franchise-based validation queries."""
        sample_titles = self.dataset_profile.get("sample_titles", [])

        # Look for potential franchise titles
        for title in sample_titles:
            if any(keyword in title.lower() for keyword in ["bungaku", "mameshiba"]):
                return [
                    {
                        "query": f"{title.split()[0]} franchise",
                        "expected_franchise": [title.split()[0]],
                        "min_results": 1,
                    }
                ]

        return []

    def _generate_staff_queries(self) -> List[Dict[str, Any]]:
        """Generate staff-based validation queries."""
        return [
            {
                "query": "anime production staff",
                "expected_content": ["staff", "creator"],
                "min_results": 1,
            }
        ]

    def _generate_related_queries(self) -> List[Dict[str, Any]]:
        """Generate related content validation queries."""
        return [
            {
                "query": "related anime",
                "expected_content": ["related", "connection"],
                "min_results": 1,
            }
        ]

    def _generate_generic_queries(self, vector_name: str) -> List[Dict[str, Any]]:
        """Generate generic validation queries for any vector."""
        get_vector_description(vector_name)
        vector_fields = get_vector_fields(vector_name)

        # Create a basic query based on vector purpose
        if vector_fields:
            primary_field = vector_fields[0]
            return [
                {
                    "query": f"anime {primary_field}",
                    "expected_content": [primary_field],
                    "min_results": 1,
                }
            ]
        else:
            return [
                {
                    "query": "anime content",
                    "expected_content": ["anime"],
                    "min_results": 1,
                }
            ]
