"""Observability configuration."""

from pydantic import BaseModel, Field


class ObservabilityConfig(BaseModel):
    """Configuration for telemetry signals and instrumentation toggles."""

    otel_enabled: bool = Field(
        default=True, description="Enable OpenTelemetry observability bootstrap"
    )
    otel_exporter_otlp_endpoint: str = Field(
        default="http://localhost:4317", description="OTLP collector endpoint"
    )
    otel_enable_logging: bool = Field(
        default=True, description="Enable structured logging setup"
    )
    otel_enable_tracing: bool = Field(
        default=True, description="Enable distributed tracing setup"
    )
    otel_enable_metrics: bool = Field(default=True, description="Enable metrics setup")
    otel_enable_grpc_server_instrumentation: bool = Field(
        default=True, description="Enable gRPC server auto-instrumentation"
    )
    otel_enable_grpc_client_instrumentation: bool = Field(
        default=True, description="Enable gRPC client auto-instrumentation"
    )
    otel_enable_aiohttp_client_instrumentation: bool = Field(
        default=False, description="Enable aiohttp client auto-instrumentation"
    )
