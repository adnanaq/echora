"""Review page: answer seed-cleaning review cases in a local web page."""

pex_binary(
    name="review_page",
    entry_point="review_page.main",
    dependencies=["//apps/review_page/src/review_page:review_page"],
    layout="packed",
)

docker_image(
    name="review-page-dev",
    source="Dockerfile.dev",
)

docker_image(
    name="review-page-image",
    source="Dockerfile.prd",
)
