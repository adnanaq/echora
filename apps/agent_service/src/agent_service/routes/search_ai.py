"""Search RPC implementation for the internal agent service.

This module exposes the ``SearchAI`` method and converts orchestrator outputs
into protobuf responses consumed by the BFF.
"""

from __future__ import annotations

import grpc

from agent_core.schemas import AgentResponse
from agent.v1 import agent_search_pb2, agent_search_pb2_grpc

from ..main import AgentService
from ..utils.mappers import entity_type_to_proto
from ..utils.proto_utils import struct_from_dict


class AgentSearchService(agent_search_pb2_grpc.AgentSearchServiceServicer):
    """gRPC servicer implementing the ``SearchAI`` contract."""

    def __init__(self, runtime: AgentService) -> None:
        """Initializes the gRPC servicer.

        Args:
            runtime: Shared runtime dependency container.
        """
        self._rt = runtime

    async def SearchAI(
        self,
        request: agent_search_pb2.SearchAIRequest,
        context: grpc.aio.ServicerContext,
    ) -> agent_search_pb2.SearchAIResponse:
        """Handles one ``SearchAI`` RPC.

        Args:
            request: RPC request containing text/image query and limits.
            context: gRPC servicer context.

        Returns:
            Structured search response with answer, IDs, and evidence.
        """
        text_query = (request.query or "").strip()
        image_query = (
            (request.image_query or "").strip()
            if hasattr(request, "image_query")
            else ""
        )
        if not text_query and not image_query:
            return agent_search_pb2.SearchAIResponse(
                answer="Empty query (no text and no image).",
                confidence=0.0,
                warnings=["Empty query (no text and no image)."],
            )

        max_turns = (
            int(request.max_turns)
            if request.max_turns
            else self._rt.app_settings.agent.default_max_turns
        )

        # No-LLM fallback: do one qdrant search in anime lane.
        if self._rt.llm_client is None:
            intent = {
                "rationale": "LLM disabled; simple semantic search",
                "entity_type": "anime",
                "query": text_query or None,
                "image_query": image_query or None,
                "filters": {},
            }
            # Import here to avoid circular import issues.
            from agent_core.schemas import SearchIntent

            res = await self._rt.qdrant_executor.search(
                SearchIntent(**intent), limit=self._rt.app_settings.agent.qdrant_limit
            )
            refs = self._rt.qdrant_executor.extract_entity_refs(res.raw_data)
            evidence = {"mode": "no_llm", "summary": res.summary}
            return agent_search_pb2.SearchAIResponse(
                answer="Top semantic matches (LLM disabled).",
                result_entities=[
                    agent_search_pb2.EntityRef(
                        type=entity_type_to_proto(r.entity_type), id=r.id
                    )
                    for r in refs
                ],
                evidence=struct_from_dict(evidence),
                confidence=0.2,
                warnings=["AGENT_LLM_ENABLED=false; returning semantic matches only."],
            )

        orch = self._rt.build_orchestrator()
        resp: AgentResponse = await orch.run_search_ai(
            query=text_query,
            image_query=image_query or None,
            max_turns=max_turns,
        )

        return agent_search_pb2.SearchAIResponse(
            answer=resp.answer,
            source_entities=[
                agent_search_pb2.EntityRef(type=entity_type_to_proto(r.entity_type), id=r.id)
                for r in resp.source_entities
            ],
            result_entities=[
                agent_search_pb2.EntityRef(type=entity_type_to_proto(r.entity_type), id=r.id)
                for r in resp.result_entities
            ],
            evidence=struct_from_dict(resp.evidence),
            citations=list(resp.citations),
            confidence=float(resp.confidence),
            warnings=list(resp.warnings),
        )
