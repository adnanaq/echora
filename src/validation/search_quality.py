"""Search Quality Validation Framework

This module provides comprehensive search quality validation including:
- Gold standard dataset creation and management
- Automated metrics (Precision@K, Recall@K, NDCG, MRR)
- Hard negative sampling for edge case testing
- Integration with existing search methods
"""

import logging
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class GoldStandardDataset:
    """Manages gold standard test datasets for search quality validation."""

    def __init__(self) -> None:
        """Initialize gold standard dataset manager."""
        self.anime_domain_queries: Dict[str, Dict[str, Any]] = {
            # Genre-based queries
            "shounen_action": {
                "query": "shounen action anime",
                "expected_results": [
                    "Attack on Titan",
                    "Demon Slayer",
                    "Jujutsu Kaisen",
                    "My Hero Academia",
                ],
                "expected_genres": ["Action", "Shounen"],
                "expected_demographics": ["Shounen"],
                "hard_negatives": ["Your Name", "Spirited Away", "Toradora"],
            },
            "shoujo_romance": {
                "query": "shoujo romance anime",
                "expected_results": ["Kaguya-sama", "Fruits Basket", "Ouran Host Club"],
                "expected_genres": ["Romance", "Shoujo"],
                "expected_demographics": ["Shoujo"],
                "hard_negatives": ["Attack on Titan", "Monster", "Ghost in the Shell"],
            },
            "seinen_psychological": {
                "query": "seinen psychological thriller",
                "expected_results": [
                    "Monster",
                    "Psycho-Pass",
                    "Serial Experiments Lain",
                ],
                "expected_genres": ["Psychological", "Thriller", "Seinen"],
                "expected_demographics": ["Seinen"],
                "hard_negatives": ["K-On!", "Lucky Star", "Azumanga Daioh"],
            },
            # Studio-based queries
            "studio_ghibli": {
                "query": "Studio Ghibli films",
                "expected_results": [
                    "Spirited Away",
                    "Princess Mononoke",
                    "My Neighbor Totoro",
                ],
                "expected_studios": ["Studio Ghibli"],
                "hard_negatives": ["Attack on Titan", "Demon Slayer", "One Piece"],
            },
            "mappa_studio": {
                "query": "MAPPA studio anime",
                "expected_results": [
                    "Attack on Titan Final Season",
                    "Jujutsu Kaisen",
                    "Chainsaw Man",
                ],
                "expected_studios": ["MAPPA"],
                "hard_negatives": [
                    "Spirited Away",
                    "Princess Mononoke",
                    "Kiki's Delivery Service",
                ],
            },
            # Character archetype queries
            "ninja_characters": {
                "query": "anime with ninja characters",
                "expected_results": ["Naruto", "Ninja Scroll", "Basilisk"],
                "expected_themes": ["ninja", "martial arts", "stealth"],
                "hard_negatives": ["Sailor Moon", "Cardcaptor Sakura", "Precure"],
            },
            "magical_girls": {
                "query": "magical girl anime",
                "expected_results": [
                    "Sailor Moon",
                    "Madoka Magica",
                    "Cardcaptor Sakura",
                ],
                "expected_themes": ["magic", "transformation", "girls"],
                "hard_negatives": ["Berserk", "Vinland Saga", "Kingdom"],
            },
            # Temporal queries
            "90s_classics": {
                "query": "90s classic anime",
                "expected_results": [
                    "Neon Genesis Evangelion",
                    "Cowboy Bebop",
                    "Ghost in the Shell",
                ],
                "expected_year_range": [1990, 1999],
                "hard_negatives": ["Demon Slayer", "Jujutsu Kaisen", "Attack on Titan"],
            },
            "modern_anime": {
                "query": "modern 2020s anime",
                "expected_results": ["Demon Slayer", "Jujutsu Kaisen", "Chainsaw Man"],
                "expected_year_range": [2020, 2025],
                "hard_negatives": ["Dragon Ball", "Sailor Moon", "Akira"],
            },
            # Complex multi-faceted queries
            "dark_psychological": {
                "query": "dark psychological anime with complex characters",
                "expected_results": [
                    "Monster",
                    "Psycho-Pass",
                    "Death Note",
                    "Serial Experiments Lain",
                ],
                "expected_genres": ["Psychological", "Thriller", "Drama"],
                "expected_themes": ["dark", "complex", "psychological"],
                "hard_negatives": ["K-On!", "Non Non Biyori", "Yuru Camp"],
            },
            "family_adventure": {
                "query": "family-friendly adventure anime",
                "expected_results": [
                    "Princess Mononoke",
                    "Castle in the Sky",
                    "Kiki's Delivery Service",
                ],
                "expected_genres": ["Adventure", "Family"],
                "expected_themes": ["family", "adventure", "wholesome"],
                "hard_negatives": ["Berserk", "Elfen Lied", "Gantz"],
            },
        }

        # Hard negative test cases for confusion detection
        self.hard_negative_tests: Dict[str, Dict[str, Any]] = {
            "genre_confusion": {
                "query": "romantic comedy anime like Toradora",
                "expected_genres": ["Romance", "Comedy"],
                "hard_negatives": ["Attack on Titan", "Monster", "Ghost in the Shell"],
                "confusion_threshold": 0.3,  # Should be very dissimilar
            },
            "demographic_confusion": {
                "query": "cute girls doing cute things",
                "expected_demographics": ["Slice of Life", "Moe"],
                "hard_negatives": ["Berserk", "Vinland Saga", "Monster"],
                "confusion_threshold": 0.2,
            },
            "temporal_confusion": {
                "query": "classic 80s mecha anime",
                "expected_year_range": [1980, 1989],
                "hard_negatives": ["Demon Slayer", "Jujutsu Kaisen", "Attack on Titan"],
                "confusion_threshold": 0.25,
            },
        }

    def get_test_queries(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get test queries for validation.

        Args:
            category: Optional category filter

        Returns:
            List of test query configurations
        """
        if category and category in self.anime_domain_queries:
            return [self.anime_domain_queries[category]]
        else:
            return list(self.anime_domain_queries.values())

    def get_hard_negative_tests(self) -> List[Dict[str, Any]]:
        """Get hard negative test cases.

        Returns:
            List of hard negative test configurations
        """
        return list(self.hard_negative_tests.values())


class SearchQualityValidator:
    """Comprehensive search quality validation with automated metrics."""

    def __init__(self) -> None:
        """Initialize search quality validator."""
        self.gold_standard = GoldStandardDataset()
        self.validation_history: List[Dict[str, Any]] = []

    async def validate_search_function(
        self,
        search_function: Callable[[str, int], Any],
        test_queries: Optional[List[Dict[str, Any]]] = None,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """Validate a search function against gold standard dataset.

        Args:
            search_function: Async search function to validate
            test_queries: Optional custom test queries
            limit: Number of results to evaluate

        Returns:
            Comprehensive validation results
        """
        try:
            logger.info("🔍 Starting search quality validation")

            if test_queries is None:
                test_queries = self.gold_standard.get_test_queries()

            validation_results: Dict[str, Any] = {
                "timestamp": time.time(),
                "total_queries": len(test_queries),
                "successful_queries": 0,
                "failed_queries": [],
                "metrics": {},
                "query_results": [],
            }

            query_results: List[Dict[str, Any]] = []
            successful_queries = 0
            failed_queries: List[Dict[str, Any]] = []

            # Track metrics across all queries
            precision_scores: List[float] = []
            recall_scores: List[float] = []
            ndcg_scores: List[float] = []
            mrr_scores: List[float] = []

            for query_config in test_queries:
                try:
                    query = query_config["query"]
                    logger.info(f"Testing query: {query}")

                    # Execute search
                    start_time = time.time()
                    results = await search_function(query, limit)
                    response_time = time.time() - start_time

                    # Extract result titles for evaluation
                    result_titles = []
                    for result in results:
                        if isinstance(result, dict):
                            title = result.get("title", "")
                            if title:
                                result_titles.append(title)

                    # Calculate metrics
                    metrics = self._calculate_query_metrics(query_config, result_titles)

                    query_result = {
                        "query": query,
                        "results_count": len(result_titles),
                        "response_time": response_time,
                        "metrics": metrics,
                        "passed_validation": metrics["precision_at_5"] > 0.6,
                    }

                    query_results.append(query_result)

                    if metrics["precision_at_5"] > 0.6:
                        successful_queries += 1

                    # Accumulate metrics
                    precision_scores.append(metrics["precision_at_5"])
                    recall_scores.append(metrics.get("recall_at_5", 0.0))
                    ndcg_scores.append(metrics.get("ndcg_at_5", 0.0))
                    mrr_scores.append(metrics.get("mrr", 0.0))

                except Exception as e:
                    logger.warning(f"Query validation failed: {query} - {e}")
                    failed_queries.append(
                        {
                            "query": query_config.get("query", ""),
                            "error": str(e),
                        }
                    )

            # Calculate aggregate metrics
            validation_results["successful_queries"] = successful_queries
            validation_results["failed_queries"] = failed_queries
            validation_results["query_results"] = query_results

            if precision_scores:
                validation_results["metrics"] = {
                    "average_precision_at_5": float(np.mean(precision_scores)),
                    "average_recall_at_5": float(np.mean(recall_scores)),
                    "average_ndcg_at_5": float(np.mean(ndcg_scores)),
                    "average_mrr": float(np.mean(mrr_scores)),
                    "success_rate": successful_queries / len(test_queries),
                }

            # Store in history
            self.validation_history.append(validation_results)

            logger.info(
                f"✅ Search validation completed. Success rate: {validation_results['metrics']['success_rate']:.2%}"
            )

            return validation_results

        except Exception as e:
            logger.error(f"Search quality validation failed: {e}")
            return {"error": str(e), "timestamp": time.time()}

    def _calculate_query_metrics(
        self, query_config: Dict[str, Any], result_titles: List[str]
    ) -> Dict[str, float]:
        """Calculate comprehensive metrics for a single query.

        Args:
            query_config: Query configuration with expected results
            result_titles: Actual search result titles

        Returns:
            Dictionary of calculated metrics
        """
        expected_results = query_config.get("expected_results", [])
        if not expected_results:
            return {
                "precision_at_5": 0.0,
                "recall_at_5": 0.0,
                "ndcg_at_5": 0.0,
                "mrr": 0.0,
            }

        # Convert to lowercase for matching
        expected_set = {title.lower() for title in expected_results}
        result_titles_lower = [title.lower() for title in result_titles]

        # Precision@K calculation
        def precision_at_k(k: int) -> float:
            if k == 0 or not result_titles_lower:
                return 0.0

            top_k = result_titles_lower[:k]
            relevant_in_top_k = sum(
                1 for title in top_k if any(exp in title for exp in expected_set)
            )
            return relevant_in_top_k / min(k, len(top_k))

        # Recall@K calculation
        def recall_at_k(k: int) -> float:
            if not expected_set or not result_titles_lower:
                return 0.0

            top_k = result_titles_lower[:k]
            relevant_in_top_k = sum(
                1 for title in top_k if any(exp in title for exp in expected_set)
            )
            return relevant_in_top_k / len(expected_set)

        # NDCG@K calculation (simplified)
        def ndcg_at_k(k: int) -> float:
            if k == 0 or not result_titles_lower or not expected_set:
                return 0.0

            # Calculate DCG
            dcg = 0.0
            for i, title in enumerate(result_titles_lower[:k]):
                relevance = 1.0 if any(exp in title for exp in expected_set) else 0.0
                if i == 0:
                    dcg += relevance
                else:
                    dcg += relevance / np.log2(i + 1)

            # Calculate IDCG (ideal DCG)
            ideal_relevances = [1.0] * min(len(expected_set), k) + [0.0] * max(
                0, k - len(expected_set)
            )
            idcg = 0.0
            for i, relevance in enumerate(ideal_relevances):
                if i == 0:
                    idcg += relevance
                else:
                    idcg += relevance / np.log2(i + 1)

            return dcg / idcg if idcg > 0 else 0.0

        # Mean Reciprocal Rank calculation
        def calculate_mrr() -> float:
            for i, title in enumerate(result_titles_lower):
                if any(exp in title for exp in expected_set):
                    return 1.0 / (i + 1)
            return 0.0

        return {
            "precision_at_5": precision_at_k(5),
            "recall_at_5": recall_at_k(5),
            "ndcg_at_5": ndcg_at_k(5),
            "mrr": calculate_mrr(),
        }

    async def validate_hard_negatives(
        self,
        search_function: Callable[[str, int], Any],
        similarity_threshold: float = 0.3,
    ) -> Dict[str, Any]:
        """Validate that hard negative samples are correctly rejected.

        Args:
            search_function: Search function to test
            similarity_threshold: Maximum allowed similarity for hard negatives

        Returns:
            Hard negative validation results
        """
        try:
            logger.info("🔥 Validating hard negative samples")

            hard_negative_tests = self.gold_standard.get_hard_negative_tests()
            validation_results: Dict[str, Any] = {
                "total_tests": len(hard_negative_tests),
                "passed_tests": 0,
                "failed_tests": [],
                "confusion_detected": False,
            }

            passed_tests = 0
            failed_tests: List[Dict[str, Any]] = []

            for test_config in hard_negative_tests:
                query = test_config["query"]
                hard_negatives = test_config["hard_negatives"]
                test_config.get("confusion_threshold", similarity_threshold)

                try:
                    # Get search results
                    results = await search_function(query, 10)
                    result_titles = [
                        r.get("title", "") for r in results if isinstance(r, dict)
                    ]

                    # Check if any hard negatives appear in top results
                    confusion_found = False
                    for hard_neg in hard_negatives:
                        for result_title in result_titles[:5]:  # Check top 5
                            if hard_neg.lower() in result_title.lower():
                                confusion_found = True
                                failed_tests.append(
                                    {
                                        "query": query,
                                        "confused_with": hard_neg,
                                        "position": result_titles.index(result_title)
                                        + 1,
                                    }
                                )
                                break

                    if not confusion_found:
                        passed_tests += 1

                except Exception as e:
                    logger.warning(f"Hard negative test failed: {query} - {e}")
                    failed_tests.append(
                        {
                            "query": query,
                            "error": str(e),
                        }
                    )

            validation_results["passed_tests"] = passed_tests
            validation_results["failed_tests"] = failed_tests
            validation_results["confusion_detected"] = len(failed_tests) > 0

            success_rate = (
                passed_tests / len(hard_negative_tests) if hard_negative_tests else 0.0
            )
            logger.info(
                f"🔥 Hard negative validation completed. Success rate: {success_rate:.2%}"
            )

            return validation_results

        except Exception as e:
            logger.error(f"Hard negative validation failed: {e}")
            return {"error": str(e)}

    def generate_quality_report(self) -> Dict[str, Any]:
        """Generate comprehensive search quality report.

        Returns:
            Quality report with trends and recommendations
        """
        try:
            if not self.validation_history:
                return {"error": "No validation history available"}

            latest_validation = self.validation_history[-1]

            report: Dict[str, Any] = {
                "timestamp": time.time(),
                "latest_metrics": latest_validation.get("metrics", {}),
                "trend_analysis": {},
                "quality_grade": "Unknown",
                "recommendations": [],
            }

            # Calculate quality grade based on latest metrics
            metrics = latest_validation.get("metrics", {})
            if metrics:
                avg_precision = metrics.get("average_precision_at_5", 0.0)
                success_rate = metrics.get("success_rate", 0.0)

                # Quality grading
                if avg_precision >= 0.8 and success_rate >= 0.8:
                    report["quality_grade"] = "Excellent"
                elif avg_precision >= 0.6 and success_rate >= 0.7:
                    report["quality_grade"] = "Good"
                elif avg_precision >= 0.4 and success_rate >= 0.5:
                    report["quality_grade"] = "Fair"
                else:
                    report["quality_grade"] = "Poor"

                # Generate recommendations
                recommendations: List[str] = []

                if avg_precision < 0.6:
                    recommendations.append(
                        f"Precision@5 ({avg_precision:.2%}) below target 60%. "
                        f"Review search relevance and ranking algorithms."
                    )

                if success_rate < 0.7:
                    recommendations.append(
                        f"Success rate ({success_rate:.2%}) below target 70%. "
                        f"Review query understanding and vector selection."
                    )

                if not recommendations:
                    recommendations.append(
                        "Search quality meets targets. Continue monitoring."
                    )

                report["recommendations"] = recommendations

            # Trend analysis if we have multiple validation runs
            if len(self.validation_history) >= 2:
                current_metrics = self.validation_history[-1].get("metrics", {})
                previous_metrics = self.validation_history[-2].get("metrics", {})

                trend_analysis: Dict[str, str] = {}

                for metric_name in ["average_precision_at_5", "success_rate"]:
                    current_val = current_metrics.get(metric_name, 0.0)
                    previous_val = previous_metrics.get(metric_name, 0.0)

                    if current_val > previous_val * 1.05:
                        trend_analysis[metric_name] = "Improving"
                    elif current_val < previous_val * 0.95:
                        trend_analysis[metric_name] = "Declining"
                    else:
                        trend_analysis[metric_name] = "Stable"

                report["trend_analysis"] = trend_analysis

            return report

        except Exception as e:
            logger.error(f"Failed to generate quality report: {e}")
            return {"error": str(e)}
