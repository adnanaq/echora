python_tests(
    name="integration",
    dependencies=[
        "//:reqs0#pytest",
        "//:reqs0#pytest-asyncio",
        "//apps/enrichment_service/src/enrichment_service:enrichment_service",
    ],
    overrides={
        "test_merge_episode_integration.py": {
            "dependencies": [
                "//tests/libs/enrichment/unit/sources/anidb:fixtures",
                "//tests/libs/enrichment/unit/sources/anisearch:saved_pages",
                "//tests/libs/enrichment/unit/sources/mal:fixtures",
            ],
        },
    },
)
