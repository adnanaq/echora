python_sources(
    name="benchmarks",
    sources=["**/*.py", "!tests/**"],
    dependencies=[
        "//libs/common/src/common",
        "//libs/enrichment/src/enrichment",
        "//libs/qdrant_db/src/qdrant_db",
        "//libs/vector_processing/src/vector_processing",
        "//apps/vector_service/src/vector_service",
        "//apps/vector_service/src/vector_proto:proto",
    ],
    # The tools read and write paths relative to the repository root
    # (test data, query files, results), so they run outside the sandbox.
    run_goal_use_sandbox=False,
)

resources(name="environments", sources=["environments/*.toml", "sweeps/*.toml"])
