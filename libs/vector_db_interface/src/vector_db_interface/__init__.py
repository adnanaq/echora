"""Vector database interface - Abstract base class for all vector DB implementations."""

from vector_db_interface.base import VectorDBClient, VectorDocument
from vector_db_interface.telemetry import CounterMetric, HistogramMetric, TelemetryRegistry

__all__ = [
    "VectorDBClient",
    "VectorDocument",
    "CounterMetric",
    "HistogramMetric",
    "TelemetryRegistry"
]
