"""Work Matching unit tests."""

resources(
    name="fixtures",
    sources=["fixtures/**"],
)

python_test_utils(
    name="conftest",
    sources=["conftest.py"],
    dependencies=[
        "//libs/work_matching/src/work_matching",
        ":fixtures",
    ],
)

python_tests(
    dependencies=[
        ":conftest",
        ":fixtures",
        "//libs/work_matching/src/work_matching",
    ],
)
