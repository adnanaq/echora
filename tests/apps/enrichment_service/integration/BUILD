python_tests(
    name="integration",
    # These files share the running dev services and database, so they run in one
    # pytest process, one after another, instead of racing for the same pages.
    batch_compatibility_tag="enrichment-service-dev-stack",
    dependencies=[
        "//:reqs0#pytest",
        "//:reqs0#pytest-asyncio",
        "//apps/enrichment_service/src/enrichment_service:enrichment_service",
    ],
    overrides={
        "test_merge_episode_integration.py": {
            "dependencies": [
                "//tests/libs/enrichment/unit/sources/anidb:fixtures",
                "//tests/libs/enrichment/unit/sources/anisearch:fixtures",
                "//tests/libs/enrichment/unit/sources/mal:fixtures",
            ],
        },
    },
)
