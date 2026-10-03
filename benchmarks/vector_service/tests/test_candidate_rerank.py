from benchmarks.vector_service.toolkit.candidate_rerank import (
    hit_rates,
    rerank_by_best_image,
)

ENTITY_IMAGES = {"luffy": ["l1", "l2"], "zoro": ["z1"], "nami": ["n1", "n2"]}


def test_rerank_by_best_image_scored_candidates_ordered_by_closest_image():
    differences = {"l1": 0.40, "l2": 0.10, "z1": 0.30, "n1": 0.50, "n2": 0.20}

    ranked = rerank_by_best_image(["zoro", "nami", "luffy"], ENTITY_IMAGES, differences)

    assert ranked == ["luffy", "nami", "zoro"]


def test_rerank_by_best_image_unscored_candidates_keep_order_at_end():
    differences = {"z1": 0.30}

    ranked = rerank_by_best_image(["nami", "zoro", "luffy"], ENTITY_IMAGES, differences)

    assert ranked == ["zoro", "nami", "luffy"]


def test_hit_rates_counts_first_place_and_top_places():
    results = [["a", "b", "c"], ["b", "a", "c"], ["c", "b", "a"]]

    assert hit_rates(results, ["a", "a", "x"], top=2) == (1 / 3, 2 / 3)
