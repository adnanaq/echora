python_sources(
    name="scripts",
    dependencies=[
        "//libs/common/src/common",
        "//libs/enrichment/src/enrichment",
        "//libs/qdrant_db/src/qdrant_db",
        "//libs/vector_processing/src/vector_processing",
    ],
    overrides={
        "reindex_anime_database.py": {"run_goal_use_sandbox": False},
    },
)
