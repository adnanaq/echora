"""
Tests for src/agent_service.py - AgentService gRPC implementation.
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
import grpc

from protos import agent_pb2
from src.agent_service import AgentService


@pytest.fixture
def agent_service():
    """Create AgentService instance for testing."""
    return AgentService()


@pytest.fixture
def mock_context():
    """Create mock gRPC context."""
    context = Mock()
    context.set_code = Mock()
    context.set_details = Mock()
    return context


@pytest.fixture
def mock_formatted_results():
    """Create mock formatted results from AnimeQueryAgent."""
    mock_result1 = Mock()
    mock_result1.anime_id = "anime-123"
    mock_result1.title = "Test Anime 1"

    mock_result2 = Mock()
    mock_result2.anime_id = "anime-456"
    mock_result2.title = "Test Anime 2"

    mock_results = Mock()
    mock_results.results = [mock_result1, mock_result2]
    mock_results.summary = "Found 2 text search results"

    return mock_results


@pytest.mark.asyncio
async def test_search_success(agent_service, mock_context, mock_formatted_results):
    """Test Search returns results successfully when agent available."""
    mock_agent = AsyncMock()
    mock_agent.parse_and_search = AsyncMock(return_value=mock_formatted_results)

    with patch("src.globals.query_parser_agent", mock_agent):
        request = agent_pb2.SearchRequest(query="anime about pirates")
        response = await agent_service.Search(request, mock_context)

        assert len(response.anime_ids) == 2
        assert response.anime_ids[0] == "anime-123"
        assert response.anime_ids[1] == "anime-456"
        assert response.reasoning == "Found 2 text search results"


@pytest.mark.asyncio
async def test_search_agent_unavailable(agent_service, mock_context):
    """Test Search returns error when query parser agent unavailable."""
    with patch("src.globals.query_parser_agent", None):
        request = agent_pb2.SearchRequest(query="test query")
        response = await agent_service.Search(request, mock_context)

        # Verify error response
        mock_context.set_code.assert_called_once_with(grpc.StatusCode.UNAVAILABLE)
        mock_context.set_details.assert_called_once_with(
            "Query parser agent not available."
        )

        # Should return empty response
        assert len(response.anime_ids) == 0


@pytest.mark.asyncio
async def test_search_with_image_data(agent_service, mock_context, mock_formatted_results):
    """Test Search handles optional image_data parameter."""
    mock_agent = AsyncMock()
    mock_agent.parse_and_search = AsyncMock(return_value=mock_formatted_results)

    with patch("src.globals.query_parser_agent", mock_agent):
        request = agent_pb2.SearchRequest(
            query="similar anime", image_data="base64encodedimage"
        )
        response = await agent_service.Search(request, mock_context)

        # Verify parse_and_search was called with image_data
        mock_agent.parse_and_search.assert_called_once()
        call_kwargs = mock_agent.parse_and_search.call_args[1]
        assert call_kwargs["user_query"] == "similar anime"
        assert call_kwargs["image_data"] == "base64encodedimage"


@pytest.mark.asyncio
async def test_search_without_image_data(agent_service, mock_context, mock_formatted_results):
    """Test Search works without image_data."""
    mock_agent = AsyncMock()
    mock_agent.parse_and_search = AsyncMock(return_value=mock_formatted_results)

    with patch("src.globals.query_parser_agent", mock_agent):
        request = agent_pb2.SearchRequest(query="action anime")
        response = await agent_service.Search(request, mock_context)

        # Verify parse_and_search was called with None image_data
        mock_agent.parse_and_search.assert_called_once()
        call_kwargs = mock_agent.parse_and_search.call_args[1]
        assert call_kwargs["image_data"] is None


@pytest.mark.asyncio
async def test_search_exception_handling(agent_service, mock_context):
    """Test Search handles unexpected exceptions."""
    mock_agent = AsyncMock()
    mock_agent.parse_and_search = AsyncMock(
        side_effect=Exception("LLM inference failed")
    )

    with patch("src.globals.query_parser_agent", mock_agent):
        request = agent_pb2.SearchRequest(query="test query")
        response = await agent_service.Search(request, mock_context)

        # Verify error handling
        mock_context.set_code.assert_called_once_with(grpc.StatusCode.INTERNAL)
        assert "internal error" in mock_context.set_details.call_args[0][0].lower()


@pytest.mark.asyncio
async def test_search_empty_results(agent_service, mock_context):
    """Test Search handles empty results gracefully."""
    mock_results = Mock()
    mock_results.results = []
    mock_results.summary = "No results found"

    mock_agent = AsyncMock()
    mock_agent.parse_and_search = AsyncMock(return_value=mock_results)

    with patch("src.globals.query_parser_agent", mock_agent):
        request = agent_pb2.SearchRequest(query="nonexistent anime")
        response = await agent_service.Search(request, mock_context)

        assert len(response.anime_ids) == 0
        assert response.reasoning == "No results found"


def test_extract_anime_ids_with_results(agent_service):
    """Test _extract_anime_ids extracts IDs from results correctly."""
    mock_result1 = Mock()
    mock_result1.anime_id = "id1"

    mock_result2 = Mock()
    mock_result2.anime_id = "id2"

    mock_results = Mock()
    mock_results.results = [mock_result1, mock_result2]

    anime_ids = agent_service._extract_anime_ids(mock_results)

    assert anime_ids == ["id1", "id2"]


def test_extract_anime_ids_with_dict_results(agent_service):
    """Test _extract_anime_ids handles dict-based results."""
    mock_results = Mock()
    mock_results.results = [
        {"anime_id": "id1"},
        {"anime_id": "id2"},
    ]

    anime_ids = agent_service._extract_anime_ids(mock_results)

    assert anime_ids == ["id1", "id2"]


def test_extract_anime_ids_with_empty_results(agent_service):
    """Test _extract_anime_ids returns empty list for empty results."""
    mock_results = Mock()
    mock_results.results = []

    anime_ids = agent_service._extract_anime_ids(mock_results)

    assert anime_ids == []


def test_extract_anime_ids_with_invalid_format(agent_service):
    """Test _extract_anime_ids handles invalid result format."""
    mock_results = Mock()
    # No results attribute
    del mock_results.results

    anime_ids = agent_service._extract_anime_ids(mock_results)

    assert anime_ids == []


def test_agent_service_inherits_servicer():
    """Test AgentService inherits from correct base class."""
    from protos import agent_pb2_grpc

    assert issubclass(AgentService, agent_pb2_grpc.AgentServiceServicer)


def test_agent_service_has_search_method():
    """Test AgentService implements Search method."""
    service = AgentService()
    assert hasattr(service, "Search")
    assert callable(service.Search)
