"""Enrichment service - pipeline sanitization and asset generation."""

target(
    name="lib",
    dependencies=[
        "//libs/common/src/common",
        "//libs/common/src/agent_proto:proto",
        "//libs/enrichment/src/enrichment",
        "//libs/http_cache/src/http_cache",
        "//libs/seed_cleaning/src/seed_cleaning",
        "//libs/work_matching/src/work_matching",
        "//:reqs0#grpcio",
        "//:reqs0#grpcio-health-checking",
    ],
)

pex_binary(
    name="enrichment_service",
    entry_point="enrichment_service.main",
    dependencies=[":lib"],
    layout="packed",
)

docker_image(
    name="enrichment-service-image",
    source="Dockerfile.prd",
)

docker_image(
    name="enrichment-service-dev",
    source="Dockerfile.dev",
)
