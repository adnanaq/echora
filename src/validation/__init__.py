"""ML Validation Framework for Anime Vector Service

This module provides comprehensive validation capabilities for embedding quality,
search quality, and recommendation quality assessment.

Components:
- EmbeddingQualityMonitor: Model drift detection and embedding space analysis
- VectorSystemValidator: 13-vector system validation suite
- SearchQualityValidator: Ground truth validation and automated metrics
- GoldStandardDataset: Anime domain test dataset management
- ABTestingFramework: Statistical testing and user simulation models
- CascadeClickModel: User behavior simulation for click-through rates
- DependentClickModel: Advanced click behavior modeling
"""

from .ab_testing import ABTestingFramework, CascadeClickModel, DependentClickModel
from .dataset_analyzer import DatasetAnalyzer
from .embedding_quality import EmbeddingQualityMonitor
from .search_quality import GoldStandardDataset, SearchQualityValidator
from .vector_system_validator import VectorSystemValidator

__all__ = [
    "DatasetAnalyzer",
    "EmbeddingQualityMonitor",
    "VectorSystemValidator",
    "SearchQualityValidator",
    "GoldStandardDataset",
    "ABTestingFramework",
    "CascadeClickModel",
    "DependentClickModel",
]
