"""MAL source unit tests."""

resources(
    name="fixtures",
    sources=["fixtures/**"],
)

python_test_utils(
    name="conftest",
    sources=["conftest.py"],
    dependencies=[
        "//libs/enrichment/src/enrichment",
        "//libs/common/src/common",
        ":fixtures",
    ],
)

python_tests(
    dependencies=[
        "//tests/libs/enrichment/unit:conftest",
        "//tests/libs/enrichment/unit/sources/mal:conftest",
        "//libs/enrichment/src/enrichment",
        "//libs/common/src/common",
        "//libs/http_cache/src/http_cache",
        ":fixtures",
        "//:reqs0#pytest-asyncio",
        "//:reqs0#pytest-mock",
    ],
)
