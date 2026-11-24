"""Agent service - gRPC endpoints for Jev and future reasoning models."""

target(
    name="lib",
    dependencies=[
        "//libs/common/src/common",
        "//libs/match_suggestions/src/match_suggestions",
        "//libs/work_matching/src/work_matching",
        "//:reqs0#grpcio",
        "//:reqs0#grpcio-health-checking",
    ],
)

pex_binary(
    name="agent_service",
    entry_point="agent_service.main",
    dependencies=[":lib"],
    layout="packed",
)

docker_image(
    name="agent-service-image",
    source="Dockerfile.prd",
)

docker_image(
    name="agent-service-dev",
    source="Dockerfile.dev",
)
