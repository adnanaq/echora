"""Unit tests for RerankerProcessor."""

import pytest

from common.config import EmbeddingConfig
from vector_processing.processors import RerankerProcessor
from vector_processing.reranking import SentenceTransformerReranker


@pytest.fixture
def reranker_processor():
    """Create a reranker processor for testing."""
    config = EmbeddingConfig(
        reranking_enabled=True,
        reranking_model="BAAI/bge-reranker-v2-m3",
    )
    model = SentenceTransformerReranker(model_name=config.reranking_model)
    return RerankerProcessor(model, config)


@pytest.mark.asyncio
async def test_rerank_empty_documents(reranker_processor):
    """Test reranking with empty document list."""
    query = "test query"
    documents = []

    result = await reranker_processor.rerank(query, documents)

    assert result == []


@pytest.mark.asyncio
async def test_rerank_sorts_by_relevance(reranker_processor):
    """Test reranking sorts documents by relevance score."""
    query = "anime about time travel"
    documents = [
        ("One Piece", "One Piece is about pirates searching for treasure"),
        ("Steins;Gate", "Steins;Gate is about time travel and changing the past"),
        ("Naruto", "Naruto is about ninjas and friendship"),
    ]

    reranked = await reranker_processor.rerank(query, documents)

    # Check that results are returned
    assert len(reranked) == 3

    # Extract titles and scores
    titles = [item[0] for item, score in reranked]
    scores = [score for item, score in reranked]

    # Steins;Gate should rank highest for time travel query
    assert titles[0] == "Steins;Gate"

    # Scores should be in descending order
    assert scores == sorted(scores, reverse=True)


@pytest.mark.asyncio
async def test_rerank_with_top_k(reranker_processor):
    """Test reranking with top_k limit."""
    query = "action anime"
    documents = [
        ("Anime 1", "Description 1"),
        ("Anime 2", "Description 2"),
        ("Anime 3", "Description 3"),
        ("Anime 4", "Description 4"),
    ]

    reranked = await reranker_processor.rerank(query, documents, top_k=2)

    # Should only return top 2 results
    assert len(reranked) == 2


@pytest.mark.asyncio
async def test_rerank_preserves_item_data(reranker_processor):
    """Test that reranking preserves original item data."""
    query = "test"

    # Use complex objects as items
    class AnimeItem:
        def __init__(self, id, title):
            self.id = id
            self.title = title

    documents = [
        (AnimeItem(1, "Anime 1"), "Description for anime 1"),
        (AnimeItem(2, "Anime 2"), "Description for anime 2"),
    ]

    reranked = await reranker_processor.rerank(query, documents)

    # Check that items are preserved
    assert len(reranked) == 2
    assert all(isinstance(item, AnimeItem) for item, score in reranked)
    assert all(isinstance(score, float) for item, score in reranked)
