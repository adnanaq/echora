"""
Observability package - shared telemetry bootstrap and instrumentation.

Standalone package for logging, metrics, tracing, and context propagation.
"""

python_distribution(
    name="observability_dist",
    dependencies=[
        "//libs/observability/src/observability:observability",
    ],
    provides=python_artifact(
        name="echora-observability",
        version="0.1.0",
        description="Shared observability bootstrap and OpenTelemetry helpers for Echora",
    ),
    generate_setup=True,
)
