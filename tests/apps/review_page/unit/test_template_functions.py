from review_page.template_functions import build_element_id, build_static_url


def test_build_element_id_same_parts_give_same_id() -> None:
    assert build_element_id("choice", "work", "mal:anime:1") == build_element_id(
        "choice", "work", "mal:anime:1"
    )


def test_build_element_id_keeps_only_id_safe_characters() -> None:
    assert build_element_id("case", "split work: mal:anime:1 anidb:anime:2") == (
        "case-split-work-mal-anime-1-anidb-anime-2"
    )


def test_build_element_id_different_parts_give_different_ids() -> None:
    assert build_element_id("chip", "kinds", "split work") != build_element_id(
        "chip", "kinds", "linked page"
    )


def test_build_static_url_carries_content_hash() -> None:
    assert build_static_url("review_page.css").startswith("/static/review_page.css?v=")
