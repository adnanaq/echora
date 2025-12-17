"""Enrichment API helpers and crawlers unit tests."""

python_sources(
    name="conftest",
    sources=["conftest.py"],
    dependencies=[
        "//libs/http_cache/src/http_cache",
        "//:reqs0#pytest-asyncio",
    ],
)

python_tests(
    dependencies=[
        ":conftest",
        "//libs/enrichment/src/enrichment",
        "//libs/http_cache/src/http_cache",
        "//:reqs0#pytest-asyncio",
        "//:reqs0#pytest-mock",
    ],
)
