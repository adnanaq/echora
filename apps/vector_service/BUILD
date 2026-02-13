"""Main vector search service - gRPC application."""

target(
    name="lib",
    dependencies=[
        "//libs/common/src/common",
        "//libs/qdrant_db/src/qdrant_db",
        "//libs/vector_processing/src/vector_processing",
        "//:reqs0#grpcio",
        "//:reqs0#grpcio-health-checking",
    ],
)

pex_binary(
    name="vector_service",
    entry_point="vector_service.main",
    dependencies=[
        ":lib",
    ],
)

docker_image(
    name="vector-service-image",
    source="Dockerfile.prd",
    # The pex_binary dependency is auto-inferred from COPY in Dockerfile
)

docker_image(
    name="vector-service-dev",
    source="Dockerfile.dev",
    # The pex_binary dependency is auto-inferred from COPY in Dockerfile
)
