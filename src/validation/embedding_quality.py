"""Embedding Quality Validation and Model Drift Detection

This module implements comprehensive embedding quality monitoring including:
- Distribution shift detection using Wasserstein distance
- Semantic coherence metrics (genre clustering, studio consistency)
- Cross-modal validation between text and image embeddings
- Temporal consistency tracking with rolling windows
"""

import logging
import time
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import stats

# Note: scipy.stats has wasserstein_distance, not scipy.spatial.distance
from scipy.stats import wasserstein_distance
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class EmbeddingQualityMonitor:
    """Monitor embedding quality with drift detection and semantic coherence metrics."""

    def __init__(self, history_days: int = 30):
        """Initialize embedding quality monitor.

        Args:
            history_days: Number of days to retain metrics history
        """
        self.history_days = history_days
        self.metrics_history: List[Dict[str, Any]] = []

        # Alert thresholds from technical.md specifications
        self.alert_bands = {
            "genre_clustering": {"excellent": 0.75, "warning": 0.65, "critical": 0.60},
            "studio_similarity": {"excellent": 0.70, "warning": 0.60, "critical": 0.55},
            "temporal_consistency": {
                "excellent": 0.80,
                "warning": 0.70,
                "critical": 0.65,
            },
            "cross_modal_consistency": {
                "excellent": 0.75,
                "warning": 0.65,
                "critical": 0.55,
            },
        }

    def compute_embedding_quality_metrics(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict[str, Any]],
        vector_type: str = "text",
    ) -> Dict[str, Any]:
        """Compute comprehensive embedding quality metrics.

        Args:
            embeddings: Array of embeddings (n_samples, embedding_dim)
            metadata: List of metadata dicts with genre, studio, etc.
            vector_type: Type of vector ("text", "image", "character_image")

        Returns:
            Dict of quality metrics with scores
        """
        try:
            metrics: Dict[str, Any] = {
                "timestamp": time.time(),
                "vector_type": vector_type,
                "n_samples": len(embeddings),
                "embedding_dim": embeddings.shape[1] if embeddings.ndim > 1 else 0,
            }

            # Genre clustering purity
            if vector_type in ["text", "character_image"]:
                genre_score = self._compute_genre_clustering_purity(
                    embeddings, metadata
                )
                metrics["genre_clustering"] = genre_score

            # Studio visual consistency (for image vectors)
            if vector_type in ["image", "character_image"]:
                studio_score = self._compute_studio_visual_consistency(
                    embeddings, metadata
                )
                metrics["studio_similarity"] = studio_score

            # Temporal consistency (for franchise/sequel relationships)
            temporal_score = self._compute_temporal_consistency(embeddings, metadata)
            metrics["temporal_consistency"] = temporal_score

            # Embedding space quality metrics
            metrics.update(self._compute_embedding_space_metrics(embeddings))

            # Store in history
            self._add_to_history(metrics)

            return metrics

        except Exception as e:
            logger.error(f"Failed to compute embedding quality metrics: {e}")
            return {"error": str(e), "timestamp": time.time()}

    def _compute_genre_clustering_purity(
        self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]
    ) -> float:
        """Compute genre clustering purity score.

        Measures how well embeddings cluster by genre using silhouette score.
        """
        try:
            # Extract genre labels
            genre_labels = []
            valid_indices = []

            for i, meta in enumerate(metadata):
                genres = meta.get("genres", [])
                if genres:
                    # Use primary genre (first in list)
                    genre_labels.append(genres[0])
                    valid_indices.append(i)

            if len(set(genre_labels)) < 2:
                return 0.0

            # Filter embeddings to valid samples
            valid_embeddings = embeddings[valid_indices]

            # Normalize embeddings
            scaler = StandardScaler()
            normalized_embeddings = scaler.fit_transform(valid_embeddings)

            # Compute silhouette score for genre clustering
            n_clusters = min(len(set(genre_labels)), 10)  # Limit clusters
            if len(valid_embeddings) < n_clusters * 2:
                return 0.0

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(normalized_embeddings)

            # Silhouette score measures clustering quality
            silhouette_avg = silhouette_score(normalized_embeddings, cluster_labels)

            # Convert to 0-1 scale (silhouette ranges from -1 to 1)
            normalized_score = (silhouette_avg + 1) / 2

            return float(normalized_score)

        except Exception as e:
            logger.warning(f"Failed to compute genre clustering purity: {e}")
            return 0.0

    def _compute_studio_visual_consistency(
        self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]
    ) -> float:
        """Compute studio visual consistency score for image embeddings."""
        try:
            # Extract studio labels
            studio_labels = []
            valid_indices = []

            for i, meta in enumerate(metadata):
                studios = meta.get("studios", [])
                if studios:
                    # Use primary studio
                    studio_labels.append(studios[0])
                    valid_indices.append(i)

            if len(set(studio_labels)) < 2:
                return 0.0

            # Filter to valid samples
            valid_embeddings = embeddings[valid_indices]

            # Compute intra-studio vs inter-studio similarity
            studio_similarities: Dict[str, List[float]] = {}

            unique_studios = list(set(studio_labels))
            intra_similarities = []
            inter_similarities = []

            for studio in unique_studios:
                studio_indices = [i for i, s in enumerate(studio_labels) if s == studio]
                if len(studio_indices) < 2:
                    continue

                studio_embeddings = valid_embeddings[studio_indices]

                # Intra-studio similarities
                for i in range(len(studio_embeddings)):
                    for j in range(i + 1, len(studio_embeddings)):
                        sim = np.dot(studio_embeddings[i], studio_embeddings[j]) / (
                            np.linalg.norm(studio_embeddings[i])
                            * np.linalg.norm(studio_embeddings[j])
                        )
                        intra_similarities.append(sim)

                # Inter-studio similarities (sample to avoid quadratic complexity)
                other_studios = [s for s in unique_studios if s != studio]
                for other_studio in other_studios[:3]:  # Limit comparisons
                    other_indices = [
                        i for i, s in enumerate(studio_labels) if s == other_studio
                    ]
                    if not other_indices:
                        continue

                    other_embeddings = valid_embeddings[other_indices[:5]]  # Sample

                    for studio_emb in studio_embeddings[:5]:  # Sample
                        for other_emb in other_embeddings:
                            sim = np.dot(studio_emb, other_emb) / (
                                np.linalg.norm(studio_emb) * np.linalg.norm(other_emb)
                            )
                            inter_similarities.append(sim)

            if not intra_similarities or not inter_similarities:
                return 0.0

            # Studio consistency = difference between intra and inter similarities
            intra_mean = np.mean(intra_similarities)
            inter_mean = np.mean(inter_similarities)

            # Normalize to 0-1 scale
            consistency_score = max(0, intra_mean - inter_mean)

            return float(consistency_score)

        except Exception as e:
            logger.warning(f"Failed to compute studio visual consistency: {e}")
            return 0.0

    def _compute_temporal_consistency(
        self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]
    ) -> float:
        """Compute temporal consistency for franchise/sequel relationships."""
        try:
            # Look for franchise relationships in metadata
            franchise_groups: Dict[str, List[int]] = {}

            for i, meta in enumerate(metadata):
                # Check for franchise indicators
                title = meta.get("title", "")
                meta.get("related_anime", [])

                # Simple franchise detection (same base title)
                base_title = self._extract_base_title(title)
                if base_title not in franchise_groups:
                    franchise_groups[base_title] = []
                franchise_groups[base_title].append(i)

            # Filter to franchises with multiple entries
            valid_franchises = {
                k: v for k, v in franchise_groups.items() if len(v) >= 2
            }

            if not valid_franchises:
                return 0.5  # Neutral score if no franchises found

            franchise_similarities = []

            for franchise, indices in valid_franchises.items():
                if len(indices) < 2:
                    continue

                franchise_embeddings = embeddings[indices]

                # Compute pairwise similarities within franchise
                for i in range(len(franchise_embeddings)):
                    for j in range(i + 1, len(franchise_embeddings)):
                        sim = np.dot(
                            franchise_embeddings[i], franchise_embeddings[j]
                        ) / (
                            np.linalg.norm(franchise_embeddings[i])
                            * np.linalg.norm(franchise_embeddings[j])
                        )
                        franchise_similarities.append(sim)

            if not franchise_similarities:
                return 0.5

            # Temporal consistency = mean franchise similarity
            mean_similarity = np.mean(franchise_similarities)

            return float(mean_similarity)

        except Exception as e:
            logger.warning(f"Failed to compute temporal consistency: {e}")
            return 0.5

    def _extract_base_title(self, title: str) -> str:
        """Extract base title for franchise detection."""
        # Remove common sequel indicators
        title = title.lower()
        for suffix in [
            " season ",
            " s2",
            " s3",
            " 2nd",
            " 3rd",
            " ii",
            " iii",
            " part ",
            ": ",
        ]:
            if suffix in title:
                title = title.split(suffix)[0]
                break
        return title.strip()

    def _compute_embedding_space_metrics(
        self, embeddings: np.ndarray
    ) -> Dict[str, float]:
        """Compute general embedding space quality metrics."""
        try:
            metrics = {}

            # Embedding space dimensionality and variance
            if embeddings.size > 0:
                metrics["mean_norm"] = float(
                    np.mean(np.linalg.norm(embeddings, axis=1))
                )
                metrics["std_norm"] = float(np.std(np.linalg.norm(embeddings, axis=1)))

                # Effective dimensionality (participation ratio)
                if embeddings.shape[1] > 1:
                    cov_matrix = np.cov(embeddings.T)
                    eigenvals = np.linalg.eigvals(cov_matrix)
                    eigenvals = eigenvals[eigenvals > 0]  # Remove negative eigenvalues

                    if len(eigenvals) > 0:
                        participation_ratio = (np.sum(eigenvals) ** 2) / np.sum(
                            eigenvals**2
                        )
                        metrics["effective_dimensionality"] = float(
                            participation_ratio / len(eigenvals)
                        )
                    else:
                        metrics["effective_dimensionality"] = 0.0
                else:
                    metrics["effective_dimensionality"] = 1.0

            return metrics

        except Exception as e:
            logger.warning(f"Failed to compute embedding space metrics: {e}")
            return {}

    def detect_distribution_shift(
        self,
        current_embeddings: np.ndarray,
        reference_embeddings: np.ndarray,
        threshold: float = 0.1,
    ) -> Dict[str, Any]:
        """Detect distribution shift using Wasserstein distance.

        Args:
            current_embeddings: Current embedding batch
            reference_embeddings: Reference (baseline) embeddings
            threshold: Drift detection threshold

        Returns:
            Dict with drift detection results
        """
        try:
            # Compute dimension-wise Wasserstein distances
            n_dims = min(current_embeddings.shape[1], reference_embeddings.shape[1])
            wasserstein_distances: List[float] = []

            for dim in range(n_dims):
                current_dim = current_embeddings[:, dim]
                reference_dim = reference_embeddings[:, dim]

                wd = wasserstein_distance(current_dim, reference_dim)
                wasserstein_distances.append(wd)

            wasserstein_array = np.array(wasserstein_distances)

            # Detection metrics
            mean_distance = np.mean(wasserstein_array)
            max_distance = np.max(wasserstein_array)
            drift_dimensions = np.sum(wasserstein_array > threshold)
            drift_percentage = drift_dimensions / n_dims

            # Drift detected if >10% dimensions exceed threshold (from technical.md)
            drift_detected = drift_percentage > 0.1

            return {
                "drift_detected": drift_detected,
                "mean_wasserstein_distance": float(mean_distance),
                "max_wasserstein_distance": float(max_distance),
                "drift_dimensions": int(drift_dimensions),
                "drift_percentage": float(drift_percentage),
                "threshold": threshold,
                "n_dimensions_analyzed": n_dims,
                "timestamp": time.time(),
            }

        except Exception as e:
            logger.error(f"Failed to detect distribution shift: {e}")
            return {"error": str(e), "drift_detected": False}

    def validate_cross_modal_consistency(
        self,
        text_embeddings: np.ndarray,
        image_embeddings: np.ndarray,
        same_anime_pairs: List[Tuple[int, int]],
        random_pairs: List[Tuple[int, int]],
    ) -> Dict[str, Any]:
        """Validate cross-modal consistency between text and image embeddings.

        Implementation of contrastive testing protocol from technical.md.
        """
        try:
            # Compute similarities for same anime pairs (positive examples)
            positive_similarities = []
            for text_idx, image_idx in same_anime_pairs:
                if text_idx < len(text_embeddings) and image_idx < len(
                    image_embeddings
                ):
                    sim = np.dot(
                        text_embeddings[text_idx], image_embeddings[image_idx]
                    ) / (
                        np.linalg.norm(text_embeddings[text_idx])
                        * np.linalg.norm(image_embeddings[image_idx])
                    )
                    positive_similarities.append(sim)

            # Compute similarities for random pairs (negative examples)
            negative_similarities = []
            for text_idx, image_idx in random_pairs:
                if text_idx < len(text_embeddings) and image_idx < len(
                    image_embeddings
                ):
                    sim = np.dot(
                        text_embeddings[text_idx], image_embeddings[image_idx]
                    ) / (
                        np.linalg.norm(text_embeddings[text_idx])
                        * np.linalg.norm(image_embeddings[image_idx])
                    )
                    negative_similarities.append(sim)

            if not positive_similarities or not negative_similarities:
                return {"error": "Insufficient pairs for cross-modal validation"}

            # Statistical validation with Mann-Whitney U test
            statistic, p_value = stats.mannwhitneyu(
                positive_similarities, negative_similarities, alternative="greater"
            )

            # Cross-modal consistency metrics
            pos_mean = np.mean(positive_similarities)
            neg_mean = np.mean(negative_similarities)
            separation = pos_mean - neg_mean

            # Effect size (Cohen's d)
            pooled_std = np.sqrt(
                (np.var(positive_similarities) + np.var(negative_similarities)) / 2
            )
            effect_size = separation / pooled_std if pooled_std > 0 else 0

            return {
                "cross_modal_consistency": float(separation),
                "positive_similarity_mean": float(pos_mean),
                "negative_similarity_mean": float(neg_mean),
                "mannwhitney_statistic": float(statistic),
                "mannwhitney_pvalue": float(p_value),
                "effect_size": float(effect_size),
                "statistically_significant": p_value < 0.001,  # From technical.md
                "n_positive_pairs": len(positive_similarities),
                "n_negative_pairs": len(negative_similarities),
                "timestamp": time.time(),
            }

        except Exception as e:
            logger.error(f"Failed to validate cross-modal consistency: {e}")
            return {"error": str(e)}

    def _add_to_history(self, metrics: Dict[str, Any]) -> None:
        """Add metrics to historical tracking."""
        self.metrics_history.append(metrics)

        # Clean old entries
        cutoff_time = time.time() - (self.history_days * 24 * 3600)
        self.metrics_history = [
            m for m in self.metrics_history if m.get("timestamp", 0) > cutoff_time
        ]

    def get_trend_analysis(self, metric_name: str, days: int = 7) -> Dict[str, Any]:
        """Analyze metric trends over specified period."""
        try:
            cutoff_time = time.time() - (days * 24 * 3600)
            recent_metrics = [
                m
                for m in self.metrics_history
                if m.get("timestamp", 0) > cutoff_time and metric_name in m
            ]

            if len(recent_metrics) < 2:
                return {"error": "Insufficient data for trend analysis"}

            values = [m[metric_name] for m in recent_metrics]
            timestamps = [m["timestamp"] for m in recent_metrics]

            # Linear regression for trend
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                timestamps, values
            )

            # Trend classification
            current_value = values[-1]
            alert_level = self._get_alert_level(metric_name, current_value)

            return {
                "metric_name": metric_name,
                "current_value": current_value,
                "trend_slope": slope,
                "trend_direction": "increasing" if slope > 0 else "decreasing",
                "trend_strength": abs(r_value),
                "trend_significance": p_value,
                "alert_level": alert_level,
                "n_data_points": len(recent_metrics),
                "time_range_days": days,
            }

        except Exception as e:
            logger.error(f"Failed to analyze trend for {metric_name}: {e}")
            return {"error": str(e)}

    def _get_alert_level(self, metric_name: str, value: float) -> str:
        """Get alert level based on configured thresholds."""
        if metric_name not in self.alert_bands:
            return "unknown"

        thresholds = self.alert_bands[metric_name]

        if value >= thresholds["excellent"]:
            return "excellent"
        elif value >= thresholds["warning"]:
            return "good"
        elif value >= thresholds["critical"]:
            return "warning"
        else:
            return "critical"

    def generate_quality_report(self) -> Dict[str, Any]:
        """Generate comprehensive embedding quality report."""
        try:
            if not self.metrics_history:
                return {"error": "No metrics history available"}

            latest_metrics = self.metrics_history[-1]

            report: Dict[str, Any] = {
                "timestamp": time.time(),
                "latest_metrics": latest_metrics,
                "alert_summary": {},
                "trend_analysis": {},
                "recommendations": [],
            }

            # Analyze each tracked metric
            for metric_name in [
                "genre_clustering",
                "studio_similarity",
                "temporal_consistency",
            ]:
                if metric_name in latest_metrics:
                    current_value = latest_metrics[metric_name]
                    alert_level = self._get_alert_level(metric_name, current_value)
                    report["alert_summary"][metric_name] = alert_level

                    # Add trend analysis
                    trend = self.get_trend_analysis(metric_name, days=7)
                    if "error" not in trend:
                        report["trend_analysis"][metric_name] = trend

                    # Generate recommendations
                    if alert_level in ["warning", "critical"]:
                        report["recommendations"].append(
                            f"{metric_name} is at {alert_level} level ({current_value:.3f}). "
                            f"Consider reviewing embedding model or data quality."
                        )

            return report

        except Exception as e:
            logger.error(f"Failed to generate quality report: {e}")
            return {"error": str(e)}
