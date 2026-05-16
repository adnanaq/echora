"""Enrichment pipeline unit tests."""

python_sources(
    name="conftest",
    sources=["conftest.py"],
)

python_tests(
    dependencies=[
        ":conftest",
        "//tests/libs/enrichment/unit:conftest",
        "//libs/enrichment/src/enrichment",
        "//libs/http_cache/src/http_cache",
        "//:reqs0#pytest-asyncio",
        "//:reqs0#pytest-mock",
    ],
)
