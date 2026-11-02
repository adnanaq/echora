resources(
    name="fixtures",
    sources=["fixtures/**"],
)

python_test_utils(
    name="conftest",
    sources=["conftest.py"],
    dependencies=[
        ":fixtures",
        "//apps/review_page/src/review_page:review_page",
        "//tests/libs/work_matching/unit:fixtures",
    ],
)

python_tests(
    name="unit",
    sources=["**/test_*.py"],
    dependencies=[
        "//:reqs0#pytest",
        "//:reqs0#httpx2",
        "//apps/review_page/src/review_page:review_page",
        "//tests/libs/work_matching/unit:fixtures",
    ],
)
