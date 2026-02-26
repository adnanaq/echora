"""Telemetry interface for database clients."""

from typing import Protocol, Any


class CounterMetric(Protocol):
    """Interface for counter metrics, matching OpenTelemetry Counter."""
    def add(self, amount: int | float, attributes: dict[str, Any] | None = None) -> None:
        """Add value to counter."""
        ...


class HistogramMetric(Protocol):
    """Interface for histogram metrics, matching OpenTelemetry Histogram."""
    def record(self, amount: int | float, attributes: dict[str, Any] | None = None) -> None:
        """Record value to histogram."""
        ...


class TelemetryRegistry(Protocol):
    """Registry providing access to platform-wide metrics."""
    DB_QUERY_DURATION: HistogramMetric
    DB_ERRORS: CounterMetric
