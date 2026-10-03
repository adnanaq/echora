import random

import pytest

from benchmarks.vector_service.toolkit.query_mix import (
    parse_query_weights,
    sample_query_mix,
)

QUERIES = {
    "short": ["pirate crew", "mecha war"],
    "title": ["One Piece", "Gundam"],
    "long": ["a long story about pirates looking for treasure"],
}


def test_sample_query_mix_returns_texts_in_weight_proportions():
    texts = sample_query_mix(QUERIES, {"short": 3, "long": 1}, 400, random.Random(1))

    short = sum(text in QUERIES["short"] for text in texts)
    long = sum(text in QUERIES["long"] for text in texts)
    assert len(texts) == 400
    assert short + long == 400
    assert 260 <= short <= 340


def test_sample_query_mix_same_seed_returns_same_mix():
    weights = {"short": 40, "title": 40, "long": 20}

    first = sample_query_mix(QUERIES, weights, 50, random.Random(54))
    second = sample_query_mix(QUERIES, weights, 50, random.Random(54))

    assert first == second


def test_parse_query_weights_returns_weights_per_query_kind():
    assert parse_query_weights("40,40,20") == {"short": 40, "title": 40, "long": 20}


@pytest.mark.parametrize("text", ["40,40", "40,40,x", "0,0,0"])
def test_parse_query_weights_bad_weights_raise_value_error(text):
    with pytest.raises(ValueError, match="weights"):
        parse_query_weights(text)
