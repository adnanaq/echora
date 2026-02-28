python_sources(
    name="scripts",
    dependencies=[
        "//libs/common/src/common",
        "//libs/enrichment/src/enrichment",
        "//libs/qdrant_db/src/qdrant_db",
        "//libs/vector_processing/src/vector_processing",
        "//:reqs0#grpcio-tools",
    ],
    overrides={
        "generate-proto.py": {"run_goal_use_sandbox": False},
        "reindex_anime_database.py": {"run_goal_use_sandbox": False},
    },
)

pex_binary(
    name="populate_vector_observability",
    entry_point="populate_vector_observability",
    dependencies=[
        ":scripts",
        "//apps/vector_service/src/vector_proto:proto",
        "//libs/common/src/shared_proto:proto",
    ],
)
