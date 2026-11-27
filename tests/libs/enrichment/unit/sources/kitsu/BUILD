"""Kitsu source unit tests."""

resources(
    name="saved_pages",
    sources=["test_*/**"],
)

python_test_utils(
    name="conftest",
    sources=["conftest.py"],
)

python_tests(
    dependencies=[
        ":conftest",
        ":saved_pages",
        "//:reqs0#pytest-regressions",
        "//tests/libs/enrichment/unit:conftest",
        "//libs/enrichment/src/enrichment",
        "//libs/http_cache/src/http_cache",
        "//libs/common/src/common",
        "//:reqs0#pytest-asyncio",
        "//:reqs0#pytest-mock",
    ],
)
