"""Unit tests for QdrantClient retry functionality using retry_with_backoff utility.

This test file uses TDD approach to verify that:
1. update_single_vector uses retry_with_backoff correctly
2. update_batch_vectors uses retry_with_backoff correctly
3. Both methods maintain their existing functionality
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from src.config import get_settings
from src.vector.client.qdrant_client import QdrantClient


class TestUpdateSingleVectorRetry:
    """Test suite for update_single_vector with retry functionality."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock QdrantClient instance."""
        settings = get_settings()
        mock_async_client = AsyncMock()

        # Create client instance
        client = QdrantClient(
            settings=settings,
            async_qdrant_client=mock_async_client,
        )

        return client

    @pytest.mark.asyncio
    async def test_update_single_vector_success_first_attempt(self, mock_client):
        """RED: Test successful update on first attempt uses retry utility."""
        # Setup
        mock_client.client.update_vectors = AsyncMock(return_value=None)

        # Execute
        result = await mock_client.update_single_vector(
            anime_id="test_123",
            vector_name="title_vector",
            vector_data=[0.1] * 1024,
        )

        # Verify
        assert result is True
        assert mock_client.client.update_vectors.call_count == 1

    @pytest.mark.asyncio
    async def test_update_single_vector_retries_on_transient_error(self, mock_client):
        """RED: Test that transient errors trigger retry using retry utility."""
        # Setup - fail twice with transient errors, then succeed
        mock_client.client.update_vectors = AsyncMock()
        mock_client.client.update_vectors.side_effect = [
            Exception("Connection timeout"),
            Exception("Network error"),
            None,  # Success on third attempt
        ]

        # Execute
        result = await mock_client.update_single_vector(
            anime_id="test_123",
            vector_name="title_vector",
            vector_data=[0.1] * 1024,
            max_retries=3,
            retry_delay=0.01,
        )

        # Verify
        assert result is True
        assert mock_client.client.update_vectors.call_count == 3

    @pytest.mark.asyncio
    async def test_update_single_vector_fails_on_non_transient_error(self, mock_client):
        """RED: Test that non-transient errors fail immediately without retry."""
        # Setup
        mock_client.client.update_vectors = AsyncMock(
            side_effect=ValueError("Invalid vector dimension")
        )

        # Execute
        result = await mock_client.update_single_vector(
            anime_id="test_123",
            vector_name="title_vector",
            vector_data=[0.1] * 1024,
        )

        # Verify - should fail and return False, only called once
        assert result is False
        assert mock_client.client.update_vectors.call_count == 1

    @pytest.mark.asyncio
    async def test_update_single_vector_max_retries_exceeded(self, mock_client):
        """RED: Test that exceeding max retries returns False."""
        # Setup - always fail with transient error
        mock_client.client.update_vectors = AsyncMock(
            side_effect=Exception("Connection timeout")
        )

        # Execute
        result = await mock_client.update_single_vector(
            anime_id="test_123",
            vector_name="title_vector",
            vector_data=[0.1] * 1024,
            max_retries=2,
            retry_delay=0.01,
        )

        # Verify - should fail after max_retries + 1 attempts
        assert result is False
        assert mock_client.client.update_vectors.call_count == 3  # 1 initial + 2 retries

    @pytest.mark.asyncio
    async def test_update_single_vector_validation_fails_no_retry(self, mock_client):
        """RED: Test that validation errors don't trigger retry."""
        # Execute with invalid vector name
        result = await mock_client.update_single_vector(
            anime_id="test_123",
            vector_name="invalid_vector",
            vector_data=[0.1] * 1024,
        )

        # Verify - should fail validation, no retry attempted
        assert result is False
        assert mock_client.client.update_vectors.call_count == 0


class TestUpdateBatchVectorsRetry:
    """Test suite for update_batch_vectors with retry functionality."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock QdrantClient instance."""
        settings = get_settings()
        mock_async_client = AsyncMock()

        client = QdrantClient(
            settings=settings,
            async_qdrant_client=mock_async_client,
        )

        return client

    @pytest.mark.asyncio
    async def test_update_batch_vectors_success_first_attempt(self, mock_client):
        """RED: Test successful batch update on first attempt."""
        # Setup
        mock_client.client.update_vectors = AsyncMock(return_value=None)

        updates = [
            {
                "anime_id": "anime_1",
                "vector_name": "title_vector",
                "vector_data": [0.1] * 1024,
            },
            {
                "anime_id": "anime_2",
                "vector_name": "genre_vector",
                "vector_data": [0.2] * 1024,
            },
        ]

        # Execute
        result = await mock_client.update_batch_vectors(
            updates=updates,
            dedup_policy="last-wins",
        )

        # Verify
        assert result["success"] == 2
        assert result["failed"] == 0
        assert mock_client.client.update_vectors.call_count == 1

    @pytest.mark.asyncio
    async def test_update_batch_vectors_retries_on_transient_error(self, mock_client):
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
                "anime_id": "anime_1",
                "vector_name": "title_vector",
                "vector_data": [0.1] * 1024,
            },
        ]

        # Execute
        result = await mock_client.update_batch_vectors(
            updates=updates,
            max_retries=3,
            retry_delay=0.01,
        )

        # Verify
        assert result["success"] == 1
        assert result["failed"] == 0
        assert mock_client.client.update_vectors.call_count == 3

    @pytest.mark.asyncio
    async def test_update_batch_vectors_deduplication_still_works(self, mock_client):
        """RED: Test that deduplication logic is preserved."""
        # Setup
        mock_client.client.update_vectors = AsyncMock(return_value=None)

        updates = [
            {
                "anime_id": "anime_1",
                "vector_name": "title_vector",
                "vector_data": [0.1] * 1024,
            },
            {
                "anime_id": "anime_1",  # Duplicate
                "vector_name": "title_vector",  # Duplicate
                "vector_data": [0.2] * 1024,  # Different data (last-wins)
            },
        ]

        # Execute
        result = await mock_client.update_batch_vectors(
            updates=updates,
            dedup_policy="last-wins",
        )

        # Verify - should deduplicate to 1 update
        assert result["duplicates_removed"] == 1
        assert result["success"] == 1
        assert mock_client.client.update_vectors.call_count == 1

    @pytest.mark.asyncio
    async def test_update_batch_vectors_max_retries_exceeded(self, mock_client):
        """RED: Test that batch update fails after max retries."""
        # Setup
        mock_client.client.update_vectors = AsyncMock(
            side_effect=Exception("Connection timeout")
        )

        updates = [
            {
                "anime_id": "anime_1",
                "vector_name": "title_vector",
                "vector_data": [0.1] * 1024,
            },
        ]

        # Execute
        result = await mock_client.update_batch_vectors(
            updates=updates,
            max_retries=2,
            retry_delay=0.01,
        )

        # Verify - should fail all updates
        assert result["success"] == 0
        assert result["failed"] == 1
        assert mock_client.client.update_vectors.call_count == 3  # 1 + 2 retries

    @pytest.mark.asyncio
    async def test_update_batch_vectors_exponential_backoff(self, mock_client):
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
                "anime_id": "anime_1",
                "vector_name": "title_vector",
                "vector_data": [0.1] * 1024,
            },
        ]

        # Execute and measure time
        start_time = asyncio.get_event_loop().time()

        result = await mock_client.update_batch_vectors(
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
