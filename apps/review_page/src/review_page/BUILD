"""Review page package sources, with its templates and static files."""

resources(
    name="page_files",
    sources=["templates/**", "static/**"],
)

python_sources(
    name="review_page",
    dependencies=[
        ":page_files",
        "//libs/common/src/shared_proto:proto",
        "//libs/common/src/common",
        "//libs/seed_cleaning/src/seed_cleaning",
        "//libs/work_matching/src/work_matching",
        "//:reqs0#fastapi",
        "//:reqs0#uvicorn",
        "//:reqs0#jinja2",
        "//:reqs0#python-multipart",
        "//:reqs0#filelock",
    ],
)
