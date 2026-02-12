"""Search RPC implementation for the internal agent service.

This module exposes the ``SearchAI`` method and converts orchestrator outputs
into protobuf responses consumed by the BFF.
"""

from __future__ import annotations

import logging
import re
from uuid import uuid4

import grpc

from agent_core.schemas import AgentResponse
from agent.v1 import agent_search_pb2, agent_search_pb2_grpc

from ..main import AgentService
from ..utils.mappers import entity_type_to_proto, evidence_to_proto

logger = logging.getLogger(__name__)
_IMAGE_URL_PATTERN = re.compile(r"^https?://", re.IGNORECASE)


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
            logger.info("agent.rpc.search_ai.invalid_empty_query")
            return agent_search_pb2.SearchAIResponse(
                answer="Empty query (no text and no image).",
                warnings=["Empty query (no text and no image)."],
            )
        if image_query and _IMAGE_URL_PATTERN.match(image_query):
            logger.info("agent.rpc.search_ai.invalid_image_query_url")
            return agent_search_pb2.SearchAIResponse(
                answer="Invalid image query. Provide raw base64 or a data URL only.",
                warnings=[
                    "Image URLs are not accepted for security reasons. "
                    "Use raw base64 bytes or a data URL."
                ],
            )

        max_turns = (
            int(request.max_turns)
            if request.max_turns
            else self._rt.app_settings.agent.default_max_turns
        )
        request_id = uuid4().hex[:12]
        logger.info(
            "agent.rpc.search_ai.start request_id=%s query=%r has_image=%s max_turns=%d llm_enabled=%s",
            request_id,
            text_query[:200],
            bool(image_query),
            max_turns,
            self._rt.llm_client is not None,
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
            logger.debug(
                "agent.rpc.search_ai.no_llm request_id=%s result_entities=%d summary=%r",
                request_id,
                len(refs),
                res.summary[:300],
            )
            return agent_search_pb2.SearchAIResponse(
                answer="Top semantic matches (LLM disabled).",
                result_entities=[
                    agent_search_pb2.EntityRef(
                        type=entity_type_to_proto(r.entity_type), id=r.id
                    )
                    for r in refs
                ],
                evidence=evidence_to_proto(
                    {
                        "termination_reason": "no_llm_semantic_search",
                        "search_similarity_score": 0.0,
                        "llm_confidence": 0.0,
                        "last_summary": evidence["summary"],
                    }
                ),
                warnings=["AGENT_LLM_ENABLED=false; returning semantic matches only."],
            )

        orch = self._rt.build_orchestrator()
        resp: AgentResponse = await orch.run_search_ai(
            query=text_query,
            image_query=image_query or None,
            max_turns=max_turns,
            request_id=request_id,
        )
        logger.info(
            "agent.rpc.search_ai.done request_id=%s source_entities=%d result_entities=%d warnings=%s",
            request_id,
            len(resp.source_entities),
            len(resp.result_entities),
            resp.warnings,
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
            evidence=evidence_to_proto(resp.evidence),
            warnings=list(resp.warnings),
        )
