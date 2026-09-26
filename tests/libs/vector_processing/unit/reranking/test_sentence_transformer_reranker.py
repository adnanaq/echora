"""Unit tests for SentenceTransformerReranker."""

import pytest
from vector_processing.reranking import SentenceTransformerReranker

MODEL_NAME = "BAAI/bge-reranker-v2-m3"


@pytest.fixture(scope="module")
def reranker() -> SentenceTransformerReranker:
    """Load the reranker once per module, pinned to CPU.

    Pants runs test files as concurrent processes; two copies of this model on
    one GPU exhausts its memory, so the unit tests stay off the GPU.
    """
    return SentenceTransformerReranker(model_name=MODEL_NAME, device="cpu")


def test_reranker_initialization(reranker):
    """Test reranker model loads successfully."""
    assert reranker.model_name == MODEL_NAME
    assert reranker.max_length == 512


def test_reranker_predict_scores(reranker):
    """Test reranker returns valid scores."""
    pairs = [
        ["What is Python?", "Python is a programming language"],
        ["What is Python?", "A python is a type of snake"],
    ]

    scores = reranker.predict(pairs)

    assert len(scores) == 2
    assert all(isinstance(s, float) for s in scores)
    # First pair should score higher (more relevant)
    assert scores[0] > scores[1]


def test_reranker_empty_pairs(reranker):
    """Test reranker handles empty input."""
    scores = reranker.predict([])
    assert scores == []


def test_reranker_model_name_property(reranker):
    """Test model_name property returns correct value."""
    assert reranker.model_name == MODEL_NAME


def test_reranker_max_length_property():
    """Test max_length property returns correct value."""
    reranker = SentenceTransformerReranker(
        model_name=MODEL_NAME, max_length=256, device="cpu"
    )
    assert reranker.max_length == 256
