import numpy as np
import pytest
from vector_processing.embedding_models.text.flagembedding_model import (
    FlagEmbeddingModel,
)

pytestmark = pytest.mark.integration

TEXTS = [
    "pirate adventure at sea",
    "school romance comedy",
    "mecha",
    "Cowboy Bebop",
    "Shingeki no Kyojin: The Final Season",
    "Sousou no Frieren",
    "A bounty hunter crew travels the solar system in 2071, chasing criminals "
    "for money while each member is haunted by a past they cannot outrun.",
    "Nach dem Tod des Helden reist die Elfenmagierin weiter und lernt, was "
    "Menschen ihr bedeutet haben.",
    "葬送のフリーレン",
    "a" * 3000,
]


@pytest.fixture(scope="module")
def model() -> FlagEmbeddingModel:
    return FlagEmbeddingModel("BAAI/bge-m3")


def test_single_pass_matches_flagembedding_encode(model: FlagEmbeddingModel) -> None:
    reference = model._model.encode(TEXTS, return_dense=True, return_sparse=True)
    dense, sparse = model.encode_with_sparse(TEXTS)

    np.testing.assert_allclose(
        np.asarray(dense, dtype=np.float32),
        np.asarray(reference["dense_vecs"], dtype=np.float32),
        atol=1e-3,
    )
    for expected, actual in zip(reference["lexical_weights"], sparse, strict=True):
        assert actual is not None
        expected_weights = {
            int(token): float(weight) for token, weight in expected.items()
        }
        actual_weights = dict(zip(actual["indices"], actual["values"], strict=True))
        assert actual_weights.keys() == expected_weights.keys()
        for token, weight in expected_weights.items():
            assert actual_weights[token] == pytest.approx(weight, abs=1e-3)


def test_dense_only_encode_matches_flagembedding_encode(
    model: FlagEmbeddingModel,
) -> None:
    reference = model._model.encode(TEXTS, return_dense=True, return_sparse=False)

    np.testing.assert_allclose(
        np.asarray(model.encode(TEXTS), dtype=np.float32),
        np.asarray(reference["dense_vecs"], dtype=np.float32),
        atol=1e-3,
    )
