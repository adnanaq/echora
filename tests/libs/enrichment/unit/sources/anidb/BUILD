"""AniDB source unit tests."""

resources(
    name="fixtures",
    sources=["fixtures/**"],
)

resources(
    name="regression_outputs",
    sources=["test_*/**"],
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
        "//tests/libs/enrichment/unit/sources/anidb:conftest",
        "//libs/enrichment/src/enrichment",
        "//libs/common/src/common",
        "//libs/http_cache/src/http_cache",
        ":fixtures",
        ":regression_outputs",
        "//:reqs0#pytest-asyncio",
        "//:reqs0#pytest-mock",
        "//:reqs0#pytest-regressions",
    ],
)
