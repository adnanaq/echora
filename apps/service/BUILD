"""Main vector search service - FastAPI application."""

python_sources(
    name="lib",
    dependencies=[
        "//libs/common/src/common",
        "//libs/qdrant_db/src/qdrant_db",
        "//libs/vector_processing/src/vector_processing",
        "//libs/http_cache/src/http_cache",
    ],
)

pex_binary(
    name="service",
    entry_point="service.main:app",
    dependencies=[
        ":lib",
    ],
)
