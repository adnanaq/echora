"""HTTP Cache unit tests."""

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
        "//libs/http_cache/src/http_cache",
        "//:reqs0#pytest-asyncio",
    ],
)
