"""Unit tests for QdrantClient retry functionality using retry_with_backoff utility.

This test file uses TDD approach to verify that:
1. update_single_point_vector uses retry_with_backoff correctly
2. update_batch_point_vectors uses retry_with_backoff correctly
3. Both methods maintain their existing functionality
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from common.config import get_settings
from qdrant_db import QdrantClient


class TestUpdateSinglePointVectorRetry:
    """Test suite for update_single_point_vector with retry functionality."""

    @pytest_asyncio.fixture
    async def mock_client(self):
        """Create a mock QdrantClient instance."""
        settings = get_settings()
        mock_async_client = AsyncMock()

        # Mock _initialize_collection to avoid DB calls in unit tests
        with patch.object(QdrantClient, "_initialize_collection", new=AsyncMock()):
            # Create client instance using async factory
            client = await QdrantClient.create(
                settings=settings,
                async_qdrant_client=mock_async_client,
            )

        return client

    @pytest.mark.asyncio
    async def test_update_single_point_vector_success_first_attempt(self, mock_client):
        """RED: Test successful update on first attempt uses retry utility."""
        # Setup
        mock_client.client.update_vectors = AsyncMock(return_value=None)

        # Execute
        result = await mock_client.update_single_point_vector(
            point_id="550e8400-e29b-41d4-a716-446655440000",
            vector_name="text_vector",
            vector_data=[0.1] * 1024,
        )

        # Verify
        assert result is True
        assert mock_client.client.update_vectors.call_count == 1

    @pytest.mark.asyncio
    async def test_update_single_point_vector_retries_on_transient_error(self, mock_client):
        """RED: Test that transient errors trigger retry using retry utility."""
        # Setup - fail twice with transient errors, then succeed
        mock_client.client.update_vectors = AsyncMock()
        mock_client.client.update_vectors.side_effect = [
            Exception("Connection timeout"),
            Exception("Network error"),
            None,  # Success on third attempt
        ]

        # Execute
        result = await mock_client.update_single_point_vector(
            point_id="550e8400-e29b-41d4-a716-446655440000",
            vector_name="text_vector",
            vector_data=[0.1] * 1024,
            max_retries=3,
            retry_delay=0.01,
        )

        # Verify
        assert result is True
        assert mock_client.client.update_vectors.call_count == 3

    @pytest.mark.asyncio
    async def test_update_single_point_vector_fails_on_non_transient_error(self, mock_client):
        """RED: Test that non-transient errors fail immediately without retry."""
        # Setup
        mock_client.client.update_vectors = AsyncMock(
            side_effect=ValueError("Invalid vector dimension")
        )

        # Execute
        result = await mock_client.update_single_point_vector(
            point_id="550e8400-e29b-41d4-a716-446655440000",
            vector_name="text_vector",
            vector_data=[0.1] * 1024,
        )

        # Verify - should fail and return False, only called once
        assert result is False
        assert mock_client.client.update_vectors.call_count == 1

    @pytest.mark.asyncio
    async def test_update_single_point_vector_max_retries_exceeded(self, mock_client):
        """RED: Test that exceeding max retries returns False."""
        # Setup - always fail with transient error
        mock_client.client.update_vectors = AsyncMock(
            side_effect=Exception("Connection timeout")
        )

        # Execute
        result = await mock_client.update_single_point_vector(
            point_id="550e8400-e29b-41d4-a716-446655440000",
            vector_name="text_vector",
            vector_data=[0.1] * 1024,
            max_retries=2,
            retry_delay=0.01,
        )

        # Verify - should fail after max_retries + 1 attempts
        assert result is False
        assert (
            mock_client.client.update_vectors.call_count == 3
        )  # 1 initial + 2 retries

    @pytest.mark.asyncio
    async def test_update_single_point_vector_validation_fails_no_retry(self, mock_client):
        """RED: Test that validation errors don't trigger retry."""
        # Execute with invalid vector name
        result = await mock_client.update_single_point_vector(
            point_id="550e8400-e29b-41d4-a716-446655440000",
            vector_name="invalid_vector",
            vector_data=[0.1] * 1024,
        )

        # Verify - should fail validation, no retry attempted
        assert result is False
        assert mock_client.client.update_vectors.call_count == 0


class TestUpdateBatchPointVectorsRetry:
    """Test suite for update_batch_point_vectors with retry functionality."""

    @pytest_asyncio.fixture
    async def mock_client(self):
        """Create a mock QdrantClient instance."""
        settings = get_settings()
        mock_async_client = AsyncMock()

        # Mock _initialize_collection to avoid DB calls in unit tests
        with patch.object(QdrantClient, "_initialize_collection", new=AsyncMock()):
            client = await QdrantClient.create(
                settings=settings,
                async_qdrant_client=mock_async_client,
            )

        return client

    @pytest.mark.asyncio
    async def test_update_batch_point_vectors_success_first_attempt(self, mock_client):
        """RED: Test successful batch update on first attempt."""
        # Setup
        mock_client.client.update_vectors = AsyncMock(return_value=None)

        updates = [
            {
                "point_id": "550e8400-e29b-41d4-a716-446655440001",
                "vector_name": "text_vector",
                "vector_data": [0.1] * 1024,
            },
            {
                "point_id": "550e8400-e29b-41d4-a716-446655440002",
                "vector_name": "text_vector",
                "vector_data": [0.2] * 1024,
            },
        ]

        # Execute
        result = await mock_client.update_batch_point_vectors(
            updates=updates,
            dedup_policy="last-wins",
        )

        # Verify
        assert result["success"] == 2
        assert result["failed"] == 0
        assert mock_client.client.update_vectors.call_count == 1

    @pytest.mark.asyncio
    async def test_update_batch_point_vectors_retries_on_transient_error(self, mock_client):
        """RED: Test that batch update retries on transient errors."""
        # Setup
        mock_client.client.update_vectors = AsyncMock()
        mock_client.client.update_vectors.side_effect = [
            Exception("Connection timeout"),
            Exception("Network unavailable"),
            None,  # Success on third attempt
        ]

        updates = [
            {
                "point_id": "550e8400-e29b-41d4-a716-446655440001",
                "vector_name": "text_vector",
                "vector_data": [0.1] * 1024,
            },
        ]

        # Execute
        result = await mock_client.update_batch_point_vectors(
            updates=updates,
            max_retries=3,
            retry_delay=0.01,
        )

        # Verify
        assert result["success"] == 1
        assert result["failed"] == 0
        assert mock_client.client.update_vectors.call_count == 3

    @pytest.mark.asyncio
    async def test_update_batch_point_vectors_deduplication_still_works(self, mock_client):
        """RED: Test that deduplication logic is preserved."""
        # Setup
        mock_client.client.update_vectors = AsyncMock(return_value=None)

        updates = [
            {
                "point_id": "550e8400-e29b-41d4-a716-446655440001",
                "vector_name": "text_vector",
                "vector_data": [0.1] * 1024,
            },
            {
                "point_id": "550e8400-e29b-41d4-a716-446655440001",  # Duplicate
                "vector_name": "text_vector",  # Duplicate
                "vector_data": [0.2] * 1024,  # Different data (last-wins)
            },
        ]

        # Execute
        result = await mock_client.update_batch_point_vectors(
            updates=updates,
            dedup_policy="last-wins",
        )

        # Verify - should deduplicate to 1 update
        assert result["duplicates_removed"] == 1
        assert result["success"] == 1
        assert mock_client.client.update_vectors.call_count == 1

    @pytest.mark.asyncio
    async def test_update_batch_point_vectors_max_retries_exceeded(self, mock_client):
        """RED: Test that batch update fails after max retries."""
        # Setup
        mock_client.client.update_vectors = AsyncMock(
            side_effect=Exception("Connection timeout")
        )

        updates = [
            {
                "point_id": "550e8400-e29b-41d4-a716-446655440001",
                "vector_name": "text_vector",
                "vector_data": [0.1] * 1024,
            },
        ]

        # Execute
        result = await mock_client.update_batch_point_vectors(
            updates=updates,
            max_retries=2,
            retry_delay=0.01,
        )

        # Verify - should fail all updates
        assert result["success"] == 0
        assert result["failed"] == 1
        assert mock_client.client.update_vectors.call_count == 3  # 1 + 2 retries

    @pytest.mark.asyncio
    async def test_update_batch_point_vectors_exponential_backoff(self, mock_client):
        """RED: Test that retry uses exponential backoff."""
        # Setup
        mock_client.client.update_vectors = AsyncMock()
        mock_client.client.update_vectors.side_effect = [
            Exception("timeout"),
            Exception("timeout"),
            None,
        ]

        updates = [
            {
                "point_id": "550e8400-e29b-41d4-a716-446655440001",
                "vector_name": "text_vector",
                "vector_data": [0.1] * 1024,
            },
        ]

        # Execute and measure time
        start_time = asyncio.get_event_loop().time()

        result = await mock_client.update_batch_point_vectors(
            updates=updates,
            max_retries=3,
            retry_delay=0.1,
        )

        end_time = asyncio.get_event_loop().time()
        elapsed = end_time - start_time

        # Verify - first retry: 0.1s, second retry: 0.2s (exponential)
        # Total should be at least 0.3s
        assert elapsed >= 0.3
        assert result["success"] == 1


class TestUnifiedSearch:
    """Test suite for unified search method."""

    @pytest_asyncio.fixture
    async def mock_qdrant_client(self):
        """Create a mock QdrantClient instance for search tests."""
        settings = get_settings()
        mock_async_client = AsyncMock()

        # Mock _initialize_collection to avoid DB calls in unit tests
        with patch.object(QdrantClient, "_initialize_collection", new=AsyncMock()):
            client = await QdrantClient.create(
                settings=settings,
                async_qdrant_client=mock_async_client,
            )

        return client

    @pytest.mark.asyncio
    async def test_search_text_only(self, mock_qdrant_client):
        """Test unified search with text embedding only."""
        text_embedding = [0.1] * 1024

        with patch.object(mock_qdrant_client, "search_single_vector") as mock_search:
            mock_search.return_value = [{"id": "test", "similarity_score": 0.9}]

            await mock_qdrant_client.search(text_embedding=text_embedding)

            mock_search.assert_called_once()
            call_args = mock_search.call_args
            assert call_args.kwargs["vector_name"] == "text_vector"

    @pytest.mark.asyncio
    async def test_search_image_only(self, mock_qdrant_client):
        """Test unified search with image embedding only."""
        image_embedding = [0.1] * 768

        with patch.object(mock_qdrant_client, "search_single_vector") as mock_search:
            mock_search.return_value = [{"id": "test", "similarity_score": 0.9}]

            await mock_qdrant_client.search(image_embedding=image_embedding)

            mock_search.assert_called_once()
            call_args = mock_search.call_args
            assert call_args.kwargs["vector_name"] == "image_vector"

    @pytest.mark.asyncio
    async def test_search_multimodal(self, mock_qdrant_client):
        """Test unified search with both text and image embeddings."""
        text_embedding = [0.1] * 1024
        image_embedding = [0.1] * 768

        with patch.object(mock_qdrant_client, "search_multi_vector") as mock_search:
            mock_search.return_value = [{"id": "test", "similarity_score": 0.9}]

            await mock_qdrant_client.search(
                text_embedding=text_embedding, image_embedding=image_embedding
            )

            mock_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_with_entity_type_filter(self, mock_qdrant_client):
        """Test unified search with entity type filter."""
        text_embedding = [0.1] * 1024

        with patch.object(mock_qdrant_client, "search_single_vector") as mock_search:
            mock_search.return_value = []

            await mock_qdrant_client.search(
                text_embedding=text_embedding, entity_type="character"
            )

            call_args = mock_search.call_args
            assert call_args.kwargs["filters"] is not None

    @pytest.mark.asyncio
    async def test_search_requires_embedding(self, mock_qdrant_client):
        """Test that search raises error without any embedding."""
        with pytest.raises(ValueError, match="At least one"):
            await mock_qdrant_client.search()


class TestMultivectorConfiguration:
    """Test suite for multivector collection configuration."""

    @pytest_asyncio.fixture
    async def mock_client(self):
        """Create a mock QdrantClient instance."""
        settings = get_settings()
        mock_async_client = AsyncMock()

        with patch.object(QdrantClient, "_initialize_collection", new=AsyncMock()):
            client = await QdrantClient.create(
                settings=settings,
                async_qdrant_client=mock_async_client,
            )

        return client

    def test_create_multi_vector_config_with_multivector(self, mock_client):
        """Test that image_vector gets multivector_config with MAX_SIM."""
        from qdrant_client.models import MultiVectorComparator

        config = mock_client._create_multi_vector_config()

        # text_vector should NOT have multivector_config
        assert config["text_vector"].multivector_config is None

        # image_vector SHOULD have multivector_config with MAX_SIM
        assert config["image_vector"].multivector_config is not None
        assert (
            config["image_vector"].multivector_config.comparator
            == MultiVectorComparator.MAX_SIM
        )
