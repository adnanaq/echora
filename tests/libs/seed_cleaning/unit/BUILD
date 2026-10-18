"""Seed Cleaning unit tests."""

resources(
    name="fixtures",
    sources=["fixtures/**"],
)

python_test_utils(
    name="conftest",
    sources=["conftest.py"],
    dependencies=[
        "//libs/seed_cleaning/src/seed_cleaning",
        ":fixtures",
    ],
)

python_tests(
    dependencies=[
        ":conftest",
        ":fixtures",
        "//libs/seed_cleaning/src/seed_cleaning",
        "//:reqs0#pytest-asyncio",
    ],
)
