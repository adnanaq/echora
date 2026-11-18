"""PostgreSQL gRPC service Docker image target."""

files(
    name="sources",
    sources=["Cargo.toml", "src/**/*.rs"],
)

docker_image(
    name="postgres-service-image",
    source="Dockerfile.prd",
    context_root=".",
    dependencies=[
        "//:rust_workspace",
        "//crates/common:sources",
        ":sources",
    ],
)
