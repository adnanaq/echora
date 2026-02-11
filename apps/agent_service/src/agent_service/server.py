"""Internal gRPC server exposing ``SearchAI`` for the BFF."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import grpc
import instructor
from google.protobuf.struct_pb2 import Struct
from grpc_health.v1 import health, health_pb2, health_pb2_grpc
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient

from agent_core.agents import build_planner_agent, build_sufficiency_agent
from agent_core.context import RetrievedContextProvider
from agent_core.orchestrator import AgentOrchestrator
from agent_core.retrieval import QdrantExecutor
from agent_core.schemas import AgentResponse, EntityType
from common.config import get_settings
from agent.v1 import agent_search_pb2, agent_search_pb2_grpc
from qdrant_db import QdrantClient
from vector_processing import TextProcessor, VisionProcessor
from vector_processing.embedding_models.factory import EmbeddingModelFactory
from vector_processing.utils.image_downloader import ImageDownloader

from .settings import AgentServiceSettings

logger = logging.getLogger(__name__)


def _entity_type_to_proto(et: EntityType) -> int:
    """Maps internal entity enum values to protobuf enum values.

    Args:
        et: Internal entity type.

    Returns:
        Integer protobuf enum value.
    """
    match et:
        case EntityType.ANIME:
            return agent_search_pb2.ENTITY_TYPE_ANIME
        case EntityType.CHARACTER:
            return agent_search_pb2.ENTITY_TYPE_CHARACTER
        case EntityType.EPISODE:
            return agent_search_pb2.ENTITY_TYPE_EPISODE
        case EntityType.MANGA:
            return agent_search_pb2.ENTITY_TYPE_MANGA
    return agent_search_pb2.ENTITY_TYPE_UNSPECIFIED


def _struct_from_dict(d: dict[str, Any]) -> Struct:
    """Converts a Python dictionary into a protobuf ``Struct``.

    Args:
        d: JSON-like dictionary.

    Returns:
        Protobuf ``Struct`` value.
    """
    s = Struct()
    # Struct.update requires JSON-like scalars/containers.
    try:
        s.update(d)
    except Exception:
        # Best-effort: stringify unknown objects.
        s.update(
            {
                k: (
                    v
                    if isinstance(v, (str, int, float, bool, list, dict))
                    else str(v)
                )
                for k, v in d.items()
            }
        )
    return s


class _AgentRuntime:
    """Owns long-lived runtime dependencies used by request handlers."""

    def __init__(self, settings: AgentServiceSettings) -> None:
        """Initializes runtime with service settings."""
        self.settings = settings
        self._initialized = False

    async def init(self) -> None:
        """Initializes clients, embedding models, and optional LLM runtime."""
        if self._initialized:
            return

        app_settings = get_settings()

        # Qdrant client
        if app_settings.qdrant.qdrant_api_key:
            self.async_qdrant = AsyncQdrantClient(
                url=app_settings.qdrant.qdrant_url,
                api_key=app_settings.qdrant.qdrant_api_key,
            )
        else:
            self.async_qdrant = AsyncQdrantClient(url=app_settings.qdrant.qdrant_url)

        self.qdrant = await QdrantClient.create(
            config=app_settings.qdrant,
            async_qdrant_client=self.async_qdrant,
            url=app_settings.qdrant.qdrant_url,
            collection_name=app_settings.qdrant.qdrant_collection_name,
        )

        # Embedding (text only)
        text_model = EmbeddingModelFactory.create_text_model(app_settings.embedding)
        self.text_processor = TextProcessor(model=text_model, config=app_settings.embedding)

        # Vision embedding (for image queries)
        vision_model = EmbeddingModelFactory.create_vision_model(app_settings.embedding)
        downloader = ImageDownloader(cache_dir=app_settings.embedding.model_cache_dir)
        self.vision_processor = VisionProcessor(
            model=vision_model,
            downloader=downloader,
            config=app_settings.embedding,
        )

        self.qdrant_executor = QdrantExecutor(
            qdrant=self.qdrant,
            text_processor=self.text_processor,
            vision_processor=self.vision_processor,
        )

        # LLM client
        self.llm_client = None
        if self.settings.llm_enabled:
            openai_client = AsyncOpenAI(
                api_key=self.settings.openai_api_key,
                base_url=self.settings.openai_base_url,
            )
            # Atomic Agents expects an Instructor client.
            self.llm_client = instructor.from_openai(
                openai_client, mode=instructor.Mode.TOOLS
            )

        self._initialized = True

    async def close(self) -> None:
        """Closes runtime resources."""
        if getattr(self, "async_qdrant", None):
            try:
                await self.async_qdrant.close()
            except Exception:
                logger.exception("Failed to close AsyncQdrantClient")

    def build_orchestrator(self) -> AgentOrchestrator:
        """Builds a fully wired orchestrator for one request flow.

        Returns:
            Configured ``AgentOrchestrator`` instance.

        Raises:
            RuntimeError: If LLM is disabled and orchestrator construction is requested.
        """
        retrieved = RetrievedContextProvider()

        if self.llm_client is None:
            raise RuntimeError("LLM is disabled; orchestrator should not be used.")

        planner = build_planner_agent(
            client=self.llm_client,
            model=self.settings.openai_model,
            retrieved_context=retrieved,
        )
        suff = build_sufficiency_agent(
            client=self.llm_client,
            model=self.settings.openai_model,
            retrieved_context=retrieved,
        )
        return AgentOrchestrator(
            planner=planner,
            sufficiency=suff,
            qdrant=self.qdrant_executor,
            max_turns_default=self.settings.default_max_turns,
            qdrant_limit=self.settings.qdrant_limit,
        )


class AgentSearchService(agent_search_pb2_grpc.AgentSearchServiceServicer):
    """gRPC servicer implementing the ``SearchAI`` contract."""

    def __init__(self, runtime: _AgentRuntime) -> None:
        """Initializes the gRPC servicer.

        Args:
            runtime: Shared runtime dependency container.
        """
        self._rt = runtime

    async def SearchAI(self, request: agent_search_pb2.SearchAIRequest, context: grpc.aio.ServicerContext):
        """Handles one ``SearchAI`` RPC.

        Args:
            request: RPC request containing text/image query and limits.
            context: gRPC servicer context.

        Returns:
            Structured search response with answer, IDs, and evidence.
        """
        text_query = (request.query or "").strip()
        image_query = (request.image_query or "").strip() if hasattr(request, "image_query") else ""
        if not text_query and not image_query:
            return agent_search_pb2.SearchAIResponse(
                answer="Empty query (no text and no image).",
                confidence=0.0,
                warnings=["Empty query (no text and no image)."],
            )

        # TODO(observability): plumb request_id/trace_id from BFF via proto fields
        # and add structured logging spans per turn (planner/retrieval/sufficiency).

        max_turns = int(request.max_turns) if request.max_turns else self._rt.settings.default_max_turns

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

            res = await self._rt.qdrant_executor.search(SearchIntent(**intent), limit=self._rt.settings.qdrant_limit)
            refs = self._rt.qdrant_executor.extract_entity_refs(res.raw_data)
            evidence = {"mode": "no_llm", "summary": res.summary}
            return agent_search_pb2.SearchAIResponse(
                answer="Top semantic matches (LLM disabled).",
                result_entities=[
                    agent_search_pb2.EntityRef(type=_entity_type_to_proto(r.entity_type), id=r.id)
                    for r in refs
                ],
                evidence=_struct_from_dict(evidence),
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
                agent_search_pb2.EntityRef(type=_entity_type_to_proto(r.entity_type), id=r.id)
                for r in resp.source_entities
            ],
            result_entities=[
                agent_search_pb2.EntityRef(type=_entity_type_to_proto(r.entity_type), id=r.id)
                for r in resp.result_entities
            ],
            evidence=_struct_from_dict(resp.evidence),
            citations=list(resp.citations),
            confidence=float(resp.confidence),
            warnings=list(resp.warnings),
        )


async def serve() -> None:
    """Starts and runs the async gRPC server until termination."""
    svc_settings = AgentServiceSettings()
    rt = _AgentRuntime(svc_settings)
    await rt.init()

    server = grpc.aio.server(options=[
        ("grpc.max_send_message_length", 8 * 1024 * 1024),
        ("grpc.max_receive_message_length", 8 * 1024 * 1024),
    ])

    agent_search_pb2_grpc.add_AgentSearchServiceServicer_to_server(AgentSearchService(rt), server)

    # Health checks
    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)

    addr = f"{svc_settings.host}:{svc_settings.port}"
    server.add_insecure_port(addr)
    logger.info("Starting agent gRPC service on %s", addr)

    await server.start()
    try:
        await server.wait_for_termination()
    finally:
        await rt.close()


def main() -> None:
    """Entrypoint used by `python -m agent_service.main`."""
    asyncio.run(serve())
