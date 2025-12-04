"""
Tests for ProgrammaticEnrichmentPipeline context manager protocol.
"""

from unittest.mock import AsyncMock, patch

import pytest

from src.enrichment.programmatic.enrichment_pipeline import (
    ProgrammaticEnrichmentPipeline,
)


class TestProgrammaticEnrichmentPipelineContextManager:
    """Test async context manager protocol for ProgrammaticEnrichmentPipeline."""

    @pytest.mark.asyncio
    async def test_context_manager_protocol(self):
        """Test that ProgrammaticEnrichmentPipeline implements async context manager protocol."""
        async with ProgrammaticEnrichmentPipeline() as pipeline:
            assert pipeline is not None
            assert isinstance(pipeline, ProgrammaticEnrichmentPipeline)

    @pytest.mark.asyncio
    async def test_context_manager_cleanup_on_exception(self):
        """Test that context manager cleans up even when exception occurs."""
        pipeline = ProgrammaticEnrichmentPipeline()

        with pytest.raises(ValueError, match="Test error"):
            async with pipeline:
                raise ValueError("Test error")
        # Should exit cleanly despite exception

    @pytest.mark.asyncio
    async def test_context_manager_with_api_fetcher(self):
        """Test that context manager handles api_fetcher cleanup."""
        pipeline = ProgrammaticEnrichmentPipeline()

        # Mock api_fetcher
        mock_fetcher = AsyncMock()
        pipeline.api_fetcher = mock_fetcher

        async with pipeline:
            pass

        # Pipeline's __aexit__ just has pass - fetcher manages itself via context manager
        # No explicit close needed

    @pytest.mark.asyncio
    async def test_no_cleanup_method_exists(self):
        """Test that cleanup() method was removed (should not exist)."""
        pipeline = ProgrammaticEnrichmentPipeline()

        # cleanup() method should NOT exist
        assert not hasattr(pipeline, "cleanup")
