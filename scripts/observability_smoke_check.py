"""Minimal OTLP smoke emitter for CI checks.

This script emits a test trace and metric to an OTLP collector endpoint.
"""

from __future__ import annotations

import argparse
import time

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emit OTLP telemetry for smoke checks")
    parser.add_argument(
        "--endpoint",
        default="http://localhost:4317",
        help="OTLP gRPC endpoint",
    )
    parser.add_argument(
        "--service-name",
        default="echora-obs-smoke",
        help="service.name resource attribute",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    resource = Resource.create(
        {
            "service.name": args.service_name,
            "service.version": "ci-smoke",
            "deployment.environment": "ci",
        }
    )

    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=args.endpoint, insecure=True))
    )
    trace.set_tracer_provider(tracer_provider)

    metric_reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(endpoint=args.endpoint, insecure=True),
        export_interval_millis=500,
    )
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)

    meter = metrics.get_meter(__name__)
    counter = meter.create_counter("echora_smoke_events_total")

    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("observability-smoke-span") as span:
        span.set_attribute("smoke.check", True)
        counter.add(1, {"env": "ci"})

    # Give exporters a moment to flush one batch.
    time.sleep(2)
    tracer_provider.force_flush()
    metric_reader.force_flush()
    print("OTLP smoke emission complete")


if __name__ == "__main__":
    main()
