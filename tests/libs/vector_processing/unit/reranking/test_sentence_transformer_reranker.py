"""Unit tests for SentenceTransformerReranker."""

import pytest

from vector_processing.reranking import SentenceTransformerReranker


def test_reranker_initialization():
    """Test reranker model loads successfully."""
    reranker = SentenceTransformerReranker(model_name="BAAI/bge-reranker-v2-m3")
    assert reranker.model_name == "BAAI/bge-reranker-v2-m3"
    assert reranker.max_length == 512


def test_reranker_predict_scores():
    """Test reranker returns valid scores."""
    reranker = SentenceTransformerReranker(model_name="BAAI/bge-reranker-v2-m3")

    pairs = [
        ["What is Python?", "Python is a programming language"],
        ["What is Python?", "A python is a type of snake"],
    ]

    scores = reranker.predict(pairs)

    assert len(scores) == 2
    assert all(isinstance(s, float) for s in scores)
    # First pair should score higher (more relevant)
    assert scores[0] > scores[1]


def test_reranker_empty_pairs():
    """Test reranker handles empty input."""
    reranker = SentenceTransformerReranker(model_name="BAAI/bge-reranker-v2-m3")

    scores = reranker.predict([])
    assert scores == []


def test_reranker_model_name_property():
    """Test model_name property returns correct value."""
    model_name = "BAAI/bge-reranker-v2-m3"
    reranker = SentenceTransformerReranker(model_name=model_name)
    assert reranker.model_name == model_name


def test_reranker_max_length_property():
    """Test max_length property returns correct value."""
    reranker = SentenceTransformerReranker(
        model_name="BAAI/bge-reranker-v2-m3", max_length=256
    )
    assert reranker.max_length == 256
