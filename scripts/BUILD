python_sources(
    name="scripts",
    overrides={
        "reindex_anime_database.py": {"run_goal_use_sandbox": False},
    },
)

# Separate target for network test
python_source(
    name="network_test",
    source="test_network_minimal.py",
    run_goal_use_sandbox=False,
)
