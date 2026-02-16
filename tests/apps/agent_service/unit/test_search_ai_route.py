"""Unit tests for SearchAI route request validation."""

import pytest

from agent.v1 import agent_search_pb2
from agent_service.routes.search_ai import AgentSearchService


class _RuntimeStub:
    """Minimal runtime stub for route construction in validation tests."""


@pytest.mark.asyncio
async def test_search_ai_rejects_image_url_query() -> None:
    """Reject HTTP(S) image queries before orchestrator execution."""
    service = AgentSearchService(runtime=_RuntimeStub())
    request = agent_search_pb2.SearchAIRequest(
        query="find this anime",
        image_query="https://example.com/image.png",
        max_turns=1,
    )

    response = await service.SearchAI(request, context=None)

    assert response.answer.startswith("Invalid image query")
    assert response.warnings
    assert "Image URLs are not accepted" in response.warnings[0]
