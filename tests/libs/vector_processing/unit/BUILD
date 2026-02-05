python_test_utils(
    name="test_utils",
    sources=["conftest.py"],
)

python_tests(
    name="unit",
    sources=["**/*_test.py", "**/test_*.py"],
    dependencies=[
        ":test_utils",
        "//:reqs0#pytest",
        "//:reqs0#pytest-asyncio",
        "//:reqs0#pytest-mock",
        "libs/common/src/common:common",
        "libs/vector_processing/src/vector_processing:vector_processing",
        "libs/vector_db_interface/src/vector_db_interface:vector_db_interface",
    ],
)
