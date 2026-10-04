"""Base enrichment abstractions unit tests."""

python_tests(
    dependencies=[
        "//tests/libs/enrichment/unit:conftest",
        "//libs/enrichment/src/enrichment",
        "//libs/http_cache/src/http_cache",
        "//:reqs0#pytest-asyncio",
        "//:reqs0#pytest-mock",
    ],
    overrides={
        # Reads every crawler's source to check how it opens browsers.
        "test_browser.py": {
            "dependencies": [
                "//libs/enrichment/src/enrichment/sources/anidb",
                "//libs/enrichment/src/enrichment/sources/anime_planet",
                "//libs/enrichment/src/enrichment/sources/anisearch",
                "//libs/enrichment/src/enrichment/sources/mal",
            ],
        },
    },
)
