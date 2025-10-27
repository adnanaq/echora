"""A/B Testing Framework for Search Algorithm Comparison

This module provides statistical testing and user simulation models for
comparing different search algorithms and configurations.

Components:
- ABTestingFramework: Statistical significance testing
- CascadeClickModel: User behavior simulation for click-through rates
- DependentClickModel: Advanced click behavior modeling
"""

import logging
import time
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class CascadeClickModel:
    """User behavior simulation using cascade click model.

    Users scan results top-to-bottom, click first satisfying result.
    """

    def __init__(self, position_bias: list[float] | None = None) -> None:
        """Initialize cascade click model.

        Args:
            position_bias: Position-based examination probabilities
        """
        # Default position bias (decreasing with position)
        self.position_bias = position_bias or [
            0.95,
            0.85,
            0.70,
            0.50,
            0.30,
            0.15,
            0.10,
            0.05,
            0.02,
            0.01,
        ]

    def simulate_clicks(
        self,
        results: list[dict[str, Any]],
        relevance_scores: list[float],
        satisfaction_threshold: float = 0.7,
    ) -> dict[str, Any]:
        """Simulate user clicks on search results.

        Args:
            results: Search results to evaluate
            relevance_scores: Relevance scores for each result
            satisfaction_threshold: Minimum relevance for satisfaction

        Returns:
            Click simulation results
        """
        try:
            clicks: list[int] = []
            click_position = -1
            examined_positions = 0

            # Simulate user scanning behavior
            for i, (result, relevance) in enumerate(zip(results, relevance_scores)):
                if i >= len(self.position_bias):
                    break

                # Probability user examines this position
                examination_prob = self.position_bias[i]
                examined = np.random.random() < examination_prob

                if examined:
                    examined_positions = i + 1

                    # If result is satisfying, user clicks and stops
                    if relevance >= satisfaction_threshold:
                        clicks.append(i)
                        click_position = i
                        break

            return {
                "clicks": clicks,
                "click_position": click_position,
                "examined_positions": examined_positions,
                "clicked": len(clicks) > 0,
                "satisfaction": click_position >= 0,
            }

        except Exception as e:
            logger.error(f"Click simulation failed: {e}")
            return {"error": str(e)}

    def calculate_ctr_by_position(
        self, simulation_results: list[dict[str, Any]]
    ) -> dict[int, float]:
        """Calculate click-through rate by position.

        Args:
            simulation_results: Results from multiple simulations

        Returns:
            CTR by position mapping
        """
        position_clicks: dict[int, int] = {}
        position_examinations: dict[int, int] = {}

        for sim_result in simulation_results:
            if "error" in sim_result:
                continue

            examined = sim_result.get("examined_positions", 0)
            clicked_pos = sim_result.get("click_position", -1)

            # Count examinations
            for pos in range(examined):
                position_examinations[pos] = position_examinations.get(pos, 0) + 1

            # Count clicks
            if clicked_pos >= 0:
                position_clicks[clicked_pos] = position_clicks.get(clicked_pos, 0) + 1

        # Calculate CTR
        ctr_by_position: dict[int, float] = {}
        for pos in position_examinations:
            examinations = position_examinations[pos]
            clicks = position_clicks.get(pos, 0)
            ctr_by_position[pos] = clicks / examinations if examinations > 0 else 0.0

        return ctr_by_position


class DependentClickModel:
    """Advanced click model that separates examination and attractiveness.

    Models examination vs attractiveness separately with position bias.
    """

    def __init__(
        self,
        examination_probs: list[float] | None = None,
        attractiveness_weights: list[float] | None = None,
    ) -> None:
        """Initialize dependent click model.

        Args:
            examination_probs: Position-based examination probabilities
            attractiveness_weights: Relevance-based attractiveness weights
        """
        # Default examination probabilities (position-based)
        self.examination_probs = examination_probs or [
            0.95,
            0.85,
            0.70,
            0.50,
            0.30,
            0.15,
        ]

        # Default attractiveness weights (relevance-based)
        self.attractiveness_weights = attractiveness_weights or [
            0.1,
            0.3,
            0.5,
            0.7,
            0.9,
            1.0,
        ]

    def simulate_clicks(
        self, results: list[dict[str, Any]], relevance_scores: list[float]
    ) -> dict[str, Any]:
        """Simulate clicks using dependent click model.

        Args:
            results: Search results
            relevance_scores: Relevance scores (0-1 scale)

        Returns:
            Click simulation results
        """
        try:
            clicks: list[int] = []
            examined_positions: list[int] = []

            for i, (result, relevance) in enumerate(zip(results, relevance_scores)):
                if i >= len(self.examination_probs):
                    break

                # Examination probability (position-dependent)
                exam_prob = self.examination_probs[i]
                examined = np.random.random() < exam_prob

                if examined:
                    examined_positions.append(i)

                    # Attractiveness probability (relevance-dependent)
                    # Map relevance score to attractiveness weight
                    weight_idx = min(
                        int(relevance * len(self.attractiveness_weights)),
                        len(self.attractiveness_weights) - 1,
                    )
                    attractiveness = self.attractiveness_weights[weight_idx]

                    # Click probability = attractiveness given examination
                    clicked = np.random.random() < attractiveness

                    if clicked:
                        clicks.append(i)

            return {
                "clicks": clicks,
                "examined_positions": examined_positions,
                "num_clicks": len(clicks),
                "num_examinations": len(examined_positions),
                "overall_satisfaction": len(clicks) > 0,
            }

        except Exception as e:
            logger.error(f"Dependent click simulation failed: {e}")
            return {"error": str(e)}


class ABTestingFramework:
    """Statistical framework for comparing search algorithms."""

    def __init__(self, significance_level: float = 0.05) -> None:
        """Initialize A/B testing framework.

        Args:
            significance_level: Statistical significance threshold (alpha)
        """
        self.significance_level = significance_level
        self.cascade_model = CascadeClickModel()
        self.dependent_model = DependentClickModel()
        self.test_history: list[dict[str, Any]] = []

    async def compare_search_algorithms(
        self,
        algorithm_a: Callable[[str, int], Any],
        algorithm_b: Callable[[str, int], Any],
        test_queries: list[str],
        relevance_evaluator: Callable[[str, list[dict[str, Any]]], list[float]],
        algorithm_a_name: str = "Algorithm A",
        algorithm_b_name: str = "Algorithm B",
        num_simulations: int = 1000,
    ) -> dict[str, Any]:
        """Compare two search algorithms using statistical testing.

        Args:
            algorithm_a: First search algorithm function
            algorithm_b: Second search algorithm function
            test_queries: List of test queries
            relevance_evaluator: Function to evaluate result relevance
            algorithm_a_name: Name for algorithm A
            algorithm_b_name: Name for algorithm B
            num_simulations: Number of user behavior simulations

        Returns:
            Comprehensive comparison results
        """
        try:
            logger.info(f"🧪 Comparing {algorithm_a_name} vs {algorithm_b_name}")

            comparison_results: dict[str, Any] = {
                "timestamp": time.time(),
                "algorithm_a": algorithm_a_name,
                "algorithm_b": algorithm_b_name,
                "total_queries": len(test_queries),
                "num_simulations": num_simulations,
                "metrics_comparison": {},
                "statistical_tests": {},
                "recommendation": "",
            }

            # Collect metrics for both algorithms
            metrics_a: list[dict[str, float]] = []
            metrics_b: list[dict[str, float]] = []

            for query in test_queries:
                try:
                    # Get results from both algorithms
                    results_a = await algorithm_a(query, 10)
                    results_b = await algorithm_b(query, 10)

                    # Evaluate relevance
                    relevance_a = relevance_evaluator(query, results_a)
                    relevance_b = relevance_evaluator(query, results_b)

                    # Calculate metrics for algorithm A
                    metrics_a_query = self._calculate_query_metrics(
                        results_a, relevance_a, num_simulations
                    )
                    metrics_a.append(metrics_a_query)

                    # Calculate metrics for algorithm B
                    metrics_b_query = self._calculate_query_metrics(
                        results_b, relevance_b, num_simulations
                    )
                    metrics_b.append(metrics_b_query)

                except Exception as e:
                    logger.warning(f"Query comparison failed: {query} - {e}")

            if not metrics_a or not metrics_b:
                return {"error": "No successful query comparisons"}

            # Aggregate metrics across queries
            aggregated_a = self._aggregate_metrics(metrics_a)
            aggregated_b = self._aggregate_metrics(metrics_b)

            comparison_results["metrics_comparison"] = {
                algorithm_a_name: aggregated_a,
                algorithm_b_name: aggregated_b,
            }

            # Perform statistical tests
            statistical_tests = self._perform_statistical_tests(metrics_a, metrics_b)
            comparison_results["statistical_tests"] = statistical_tests

            # Generate recommendation
            recommendation = self._generate_recommendation(
                aggregated_a,
                aggregated_b,
                statistical_tests,
                algorithm_a_name,
                algorithm_b_name,
            )
            comparison_results["recommendation"] = recommendation

            # Store in history
            self.test_history.append(comparison_results)

            logger.info(f"🧪 A/B test completed. Recommendation: {recommendation}")

            return comparison_results

        except Exception as e:
            logger.error(f"A/B testing failed: {e}")
            return {"error": str(e)}

    def _calculate_query_metrics(
        self,
        results: list[dict[str, Any]],
        relevance_scores: list[float],
        num_simulations: int,
    ) -> dict[str, float]:
        """Calculate metrics for a single query using user simulation.

        Args:
            results: Search results
            relevance_scores: Relevance scores for results
            num_simulations: Number of simulations to run

        Returns:
            Query-level metrics
        """
        # Run cascade model simulations
        cascade_simulations = []
        for _ in range(num_simulations):
            sim_result = self.cascade_model.simulate_clicks(results, relevance_scores)
            if "error" not in sim_result:
                cascade_simulations.append(sim_result)

        # Run dependent model simulations
        dependent_simulations = []
        for _ in range(num_simulations):
            sim_result = self.dependent_model.simulate_clicks(results, relevance_scores)
            if "error" not in sim_result:
                dependent_simulations.append(sim_result)

        # Calculate metrics
        metrics: dict[str, float] = {}

        if cascade_simulations:
            # Cascade model metrics
            satisfaction_rate = sum(
                1 for sim in cascade_simulations if sim.get("satisfaction", False)
            ) / len(cascade_simulations)

            avg_click_position = (
                np.mean(
                    [
                        sim.get("click_position", 10)
                        for sim in cascade_simulations
                        if sim.get("clicked", False)
                    ]
                )
                if any(sim.get("clicked", False) for sim in cascade_simulations)
                else 10.0
            )

            metrics.update(
                {
                    "cascade_satisfaction_rate": satisfaction_rate,
                    "cascade_avg_click_position": float(avg_click_position),
                }
            )

        if dependent_simulations:
            # Dependent model metrics
            avg_clicks = np.mean(
                [sim.get("num_clicks", 0) for sim in dependent_simulations]
            )

            click_rate = sum(
                1
                for sim in dependent_simulations
                if sim.get("overall_satisfaction", False)
            ) / len(dependent_simulations)

            metrics.update(
                {
                    "dependent_avg_clicks": float(avg_clicks),
                    "dependent_click_rate": click_rate,
                }
            )

        # Basic relevance metrics
        if relevance_scores:
            metrics.update(
                {
                    "avg_relevance_top_5": float(np.mean(relevance_scores[:5])),
                    "max_relevance": float(np.max(relevance_scores)),
                    "relevance_at_1": (
                        float(relevance_scores[0]) if relevance_scores else 0.0
                    ),
                }
            )

        return metrics

    def _aggregate_metrics(
        self, query_metrics: list[dict[str, float]]
    ) -> dict[str, float]:
        """Aggregate metrics across all queries.

        Args:
            query_metrics: List of per-query metrics

        Returns:
            Aggregated metrics
        """
        if not query_metrics:
            return {}

        aggregated: dict[str, float] = {}

        # Get all metric names
        all_metric_names: set[str] = set()
        for metrics in query_metrics:
            all_metric_names.update(metrics.keys())

        # Calculate averages
        for metric_name in all_metric_names:
            values = [metrics.get(metric_name, 0.0) for metrics in query_metrics]
            aggregated[f"avg_{metric_name}"] = float(np.mean(values))
            aggregated[f"std_{metric_name}"] = float(np.std(values))

        return aggregated

    def _perform_statistical_tests(
        self, metrics_a: list[dict[str, float]], metrics_b: list[dict[str, float]]
    ) -> dict[str, Any]:
        """Perform statistical significance tests.

        Args:
            metrics_a: Metrics for algorithm A
            metrics_b: Metrics for algorithm B

        Returns:
            Statistical test results
        """
        statistical_tests: dict[str, Any] = {}

        try:
            # Get common metric names
            metric_names = set(metrics_a[0].keys()) & set(metrics_b[0].keys())

            for metric_name in metric_names:
                values_a = [metrics.get(metric_name, 0.0) for metrics in metrics_a]
                values_b = [metrics.get(metric_name, 0.0) for metrics in metrics_b]

                if len(values_a) >= 3 and len(values_b) >= 3:  # Minimum sample size
                    # Perform t-test
                    t_stat, p_value = stats.ttest_ind(values_a, values_b)

                    # Calculate effect size (Cohen's d)
                    pooled_std = np.sqrt(
                        (
                            (len(values_a) - 1) * np.var(values_a, ddof=1)
                            + (len(values_b) - 1) * np.var(values_b, ddof=1)
                        )
                        / (len(values_a) + len(values_b) - 2)
                    )

                    cohen_d = (
                        (np.mean(values_a) - np.mean(values_b)) / pooled_std
                        if pooled_std > 0
                        else 0.0
                    )

                    statistical_tests[metric_name] = {
                        "t_statistic": float(t_stat),
                        "p_value": float(p_value),
                        "significant": p_value < self.significance_level,
                        "effect_size": float(cohen_d),
                        "effect_magnitude": self._interpret_effect_size(cohen_d),
                    }

        except Exception as e:
            logger.warning(f"Statistical testing failed: {e}")
            statistical_tests["error"] = str(e)

        return statistical_tests

    def _interpret_effect_size(self, cohen_d: float) -> str:
        """Interpret Cohen's d effect size.

        Args:
            cohen_d: Cohen's d value

        Returns:
            Effect size interpretation
        """
        abs_d = abs(cohen_d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"

    def _generate_recommendation(
        self,
        metrics_a: dict[str, float],
        metrics_b: dict[str, float],
        statistical_tests: dict[str, Any],
        algorithm_a_name: str,
        algorithm_b_name: str,
    ) -> str:
        """Generate recommendation based on comparison results.

        Args:
            metrics_a: Aggregated metrics for algorithm A
            metrics_b: Aggregated metrics for algorithm B
            statistical_tests: Statistical test results
            algorithm_a_name: Name of algorithm A
            algorithm_b_name: Name of algorithm B

        Returns:
            Recommendation string
        """
        try:
            # Key metrics for decision making
            key_metrics = [
                "cascade_satisfaction_rate",
                "dependent_click_rate",
                "avg_relevance_top_5",
            ]

            significant_improvements = 0
            significant_degradations = 0
            better_metrics_a = 0
            better_metrics_b = 0

            for metric in key_metrics:
                avg_metric_a = metrics_a.get(f"avg_{metric}", 0.0)
                avg_metric_b = metrics_b.get(f"avg_{metric}", 0.0)

                # Count which algorithm is better
                if avg_metric_a > avg_metric_b:
                    better_metrics_a += 1
                elif avg_metric_b > avg_metric_a:
                    better_metrics_b += 1

                # Check statistical significance
                test_result = statistical_tests.get(metric, {})
                if test_result.get("significant", False):
                    t_stat = test_result.get("t_statistic", 0.0)
                    if t_stat > 0:  # A > B
                        significant_improvements += 1
                    else:  # B > A
                        significant_degradations += 1

            # Generate recommendation
            if significant_improvements > significant_degradations:
                return f"Recommend {algorithm_a_name}: Statistically significant improvements in {significant_improvements} key metrics."
            elif significant_degradations > significant_improvements:
                return f"Recommend {algorithm_b_name}: Statistically significant improvements in {significant_degradations} key metrics."
            elif better_metrics_a > better_metrics_b:
                return f"Slight preference for {algorithm_a_name}: Better performance in {better_metrics_a}/{len(key_metrics)} key metrics, but not statistically significant."
            elif better_metrics_b > better_metrics_a:
                return f"Slight preference for {algorithm_b_name}: Better performance in {better_metrics_b}/{len(key_metrics)} key metrics, but not statistically significant."
            else:
                return "No clear winner: Both algorithms perform similarly. Consider other factors (complexity, cost, etc.)."

        except Exception as e:
            logger.warning(f"Recommendation generation failed: {e}")
            return "Unable to generate recommendation due to insufficient data."

    def get_test_history(self) -> list[dict[str, Any]]:
        """Get history of A/B tests.

        Returns:
            List of previous test results
        """
        return self.test_history.copy()

    def calculate_statistical_power(
        self, effect_size: float, sample_size: int, alpha: float = 0.05
    ) -> float:
        """Calculate statistical power for given parameters.

        Args:
            effect_size: Expected effect size (Cohen's d)
            sample_size: Sample size per group
            alpha: Significance level

        Returns:
            Statistical power (1 - beta)
        """
        try:
            # Simplified power calculation for t-test
            # This is an approximation
            critical_value = stats.t.ppf(1 - alpha / 2, 2 * sample_size - 2)
            non_centrality = effect_size * np.sqrt(sample_size / 2)

            power = 1 - stats.t.cdf(
                critical_value - non_centrality, 2 * sample_size - 2
            )
            return float(power)

        except Exception as e:
            logger.warning(f"Power calculation failed: {e}")
            return 0.0
