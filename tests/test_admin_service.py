"""
Tests for src/admin_service.py - AdminService gRPC implementation.
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
import grpc
from google.protobuf.struct_pb2 import Struct

from protos import admin_pb2
from src.admin_service import AdminService


@pytest.fixture
def admin_service():
    """Create AdminService instance for testing."""
    return AdminService()


@pytest.fixture
def mock_context():
    """Create mock gRPC context."""
    context = Mock()
    context.set_code = Mock()
    context.set_details = Mock()
    return context


@pytest.mark.asyncio
async def test_get_stats_success(admin_service, mock_context):
    """Test GetStats returns stats successfully when client available."""
    mock_client = AsyncMock()
    mock_client.get_stats = AsyncMock(
        return_value={
            "collection_name": "test_collection",
            "total_documents": 100,
            "vector_size": 1024,
            "distance_metric": "cosine",
            "status": "green",
            "optimizer_status": "ok",
            "indexed_vectors_count": 100,
            "points_count": 100,
        }
    )

    with patch("src.globals.qdrant_client", mock_client):
        request = admin_pb2.GetStatsRequest()
        response = await admin_service.GetStats(request, mock_context)

        assert response.collection_name == "test_collection"
        assert response.total_documents == 100
        assert response.vector_size == 1024
        assert response.distance_metric == "cosine"
        assert response.status == "green"

        # Verify additional stats
        assert response.additional_stats is not None


@pytest.mark.asyncio
async def test_get_stats_client_unavailable(admin_service, mock_context):
    """Test GetStats returns error when Qdrant client unavailable."""
    with patch("src.globals.qdrant_client", None):
        request = admin_pb2.GetStatsRequest()
        response = await admin_service.GetStats(request, mock_context)

        # Verify error response
        mock_context.set_code.assert_called_once_with(grpc.StatusCode.UNAVAILABLE)
        mock_context.set_details.assert_called_once_with(
            "Qdrant client not available."
        )

        # Should return empty response
        assert response.collection_name == ""


@pytest.mark.asyncio
async def test_get_stats_qdrant_error(admin_service, mock_context):
    """Test GetStats handles Qdrant errors gracefully."""
    mock_client = AsyncMock()
    mock_client.get_stats = AsyncMock(return_value={"error": "Connection failed"})

    with patch("src.globals.qdrant_client", mock_client):
        request = admin_pb2.GetStatsRequest()
        response = await admin_service.GetStats(request, mock_context)

        # Verify error handling
        mock_context.set_code.assert_called_once_with(grpc.StatusCode.INTERNAL)
        assert "Failed to retrieve stats" in mock_context.set_details.call_args[0][0]


@pytest.mark.asyncio
async def test_get_stats_exception_handling(admin_service, mock_context):
    """Test GetStats handles unexpected exceptions."""
    mock_client = AsyncMock()
    mock_client.get_stats = AsyncMock(side_effect=Exception("Unexpected error"))

    with patch("src.globals.qdrant_client", mock_client):
        request = admin_pb2.GetStatsRequest()
        response = await admin_service.GetStats(request, mock_context)

        # Verify error handling
        mock_context.set_code.assert_called_once_with(grpc.StatusCode.INTERNAL)
        assert "internal error" in mock_context.set_details.call_args[0][0].lower()


@pytest.mark.asyncio
async def test_get_stats_additional_stats_format(admin_service, mock_context):
    """Test GetStats formats additional_stats as Protobuf Struct correctly."""
    mock_client = AsyncMock()
    mock_client.get_stats = AsyncMock(
        return_value={
            "collection_name": "test",
            "total_documents": 50,
            "vector_size": 512,
            "distance_metric": "euclidean",
            "status": "yellow",
            "optimizer_status": "indexing",
            "indexed_vectors_count": 25,
            "points_count": 50,
        }
    )

    with patch("src.globals.qdrant_client", mock_client):
        request = admin_pb2.GetStatsRequest()
        response = await admin_service.GetStats(request, mock_context)

        # Verify additional stats is a Struct
        assert isinstance(response.additional_stats, Struct)
        assert "optimizer_status" in response.additional_stats
        assert response.additional_stats["optimizer_status"] == "indexing"


def test_admin_service_inherits_servicer():
    """Test AdminService inherits from correct base class."""
    from protos import admin_pb2_grpc

    assert issubclass(AdminService, admin_pb2_grpc.AdminServiceServicer)


def test_admin_service_has_get_stats_method():
    """Test AdminService implements GetStats method."""
    service = AdminService()
    assert hasattr(service, "GetStats")
    assert callable(service.GetStats)
