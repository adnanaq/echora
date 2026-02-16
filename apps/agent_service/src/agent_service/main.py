"""Runtime container and gRPC bootstrap for the internal agent service.

This module contains both:
- long-lived runtime dependency initialization/lifecycle logic, and
- gRPC server startup/wiring for ``AgentSearchService`` and ``ServiceOps``.
"""

from __future__ import annotations

import asyncio
import logging

import grpc
import instructor
from grpc_health.v1 import health, health_pb2, health_pb2_grpc
from openai import AsyncOpenAI
from langfuse import get_client, Langfuse
from qdrant_client import AsyncQdrantClient

from agent.v1 import agent_search_pb2_grpc, service_ops_pb2_grpc
from agent_core.agents import (
    build_answer_agent,
    build_rewrite_agent,
    build_source_selector_agent,
    build_sufficiency_agent,
)
from agent_core.context import RetrievedContextProvider
from agent_core.orchestrator import AgentOrchestrator
from agent_core.retrieval import QdrantExecutor
from common.config import Settings, get_settings
from qdrant_db import QdrantClient
from vector_processing import TextProcessor, VisionProcessor
from vector_processing.embedding_models.factory import EmbeddingModelFactory
from vector_processing.utils.image_downloader import ImageDownloader

logger = logging.getLogger(__name__)


class AgentService:
    """Owns long-lived runtime dependencies used by gRPC request handlers."""

    def __init__(self, settings: Settings) -> None:
        """Initialize runtime with service settings.

        Args:
            settings: Common application settings from ``common.config``.
        """
        self.app_settings = settings
        self.langfuse = None
        self._initialized = False

    async def init(self) -> None:
        """Initialize clients, embedding models, and optional LLM runtime."""
        if self._initialized:
            return

        if self.app_settings.qdrant.qdrant_api_key:
            self.async_qdrant = AsyncQdrantClient(
                url=self.app_settings.qdrant.qdrant_url,
                api_key=self.app_settings.qdrant.qdrant_api_key,
            )
        else:
            self.async_qdrant = AsyncQdrantClient(
                url=self.app_settings.qdrant.qdrant_url
            )

        self.qdrant = await QdrantClient.create(
            config=self.app_settings.qdrant,
            async_qdrant_client=self.async_qdrant,
            url=self.app_settings.qdrant.qdrant_url,
            collection_name=self.app_settings.qdrant.qdrant_collection_name,
        )

        text_model = EmbeddingModelFactory.create_text_model(self.app_settings.embedding)
        self.text_processor = TextProcessor(
            model=text_model, config=self.app_settings.embedding
        )

        vision_model = EmbeddingModelFactory.create_vision_model(
            self.app_settings.embedding
        )
        downloader = ImageDownloader(
            cache_dir=self.app_settings.embedding.model_cache_dir
        )
        self.vision_processor = VisionProcessor(
            model=vision_model,
            downloader=downloader,
            config=self.app_settings.embedding,
        )

        self.qdrant_executor = QdrantExecutor(
            qdrant=self.qdrant,
            text_processor=self.text_processor,
            vision_processor=self.vision_processor,
        )

        if self.app_settings.agent.langfuse_enabled:
            self.langfuse = get_client()

        self.llm_client = None
        if self.app_settings.agent.llm_enabled:
            openai_client = AsyncOpenAI(
                api_key=self.app_settings.agent.openai_api_key,
                base_url=self.app_settings.agent.openai_base_url,
            )
            self.llm_client = instructor.from_openai(
                openai_client, mode=instructor.Mode.TOOLS
            )
        self._initialized = True

    async def close(self) -> None:
        """Close runtime resources."""
        if getattr(self, "async_qdrant", None):
            try:
                await self.async_qdrant.close()
            except Exception:
                logger.exception("Failed to close AsyncQdrantClient")
        if self.app_settings.agent.langfuse_enabled and self.langfuse is not None:
            self.langfuse.shutdown()

    def build_orchestrator(self) -> AgentOrchestrator:
        """Build a fully wired orchestrator for one request flow."""
        retrieved = RetrievedContextProvider()

        if self.llm_client is None:
            raise RuntimeError("LLM is disabled; orchestrator should not be used.")

        rewrite = build_rewrite_agent(
            client=self.llm_client,
            model=self.app_settings.agent.openai_model,
        )
        source_selector = build_source_selector_agent(
            client=self.llm_client,
            model=self.app_settings.agent.openai_model,
            retrieved_context=retrieved,
        )
        answer = build_answer_agent(
            client=self.llm_client,
            model=self.app_settings.agent.openai_model,
            retrieved_context=retrieved,
        )
        suff = build_sufficiency_agent(
            client=self.llm_client,
            model=self.app_settings.agent.openai_model,
            retrieved_context=retrieved,
        )
        return AgentOrchestrator(
            rewrite=rewrite,
            source_selector=source_selector,
            answer=answer,
            sufficiency=suff,
            qdrant=self.qdrant_executor,
            max_turns_default=self.app_settings.agent.default_max_turns,
            qdrant_limit=self.app_settings.agent.qdrant_limit,
            qdrant_context_top_k=self.app_settings.agent.qdrant_context_top_k,
            qdrant_min_score_text=self.app_settings.agent.qdrant_min_score_text,
            qdrant_min_score_image=self.app_settings.agent.qdrant_min_score_image,
            qdrant_min_score_multivector=self.app_settings.agent.qdrant_min_score_multivector,
        )


async def serve() -> None:
    """Start the gRPC server and block until termination."""
    # Initialize common settings first to configure logging.
    app_settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, app_settings.service.log_level),
        format=app_settings.service.log_format,
    )

    runtime = AgentService(app_settings)
    await runtime.init()

    # Imported here to avoid circular imports since handlers import runtime types.
    from .routes.search_ai import AgentSearchService
    from .routes.service_ops import ServiceOps

    server = grpc.aio.server(
        options=[
            ("grpc.max_send_message_length", 8 * 1024 * 1024),
            ("grpc.max_receive_message_length", 8 * 1024 * 1024),
        ]
    )

    agent_search_pb2_grpc.add_AgentSearchServiceServicer_to_server(
        AgentSearchService(runtime),
        server,
    )
    service_ops_pb2_grpc.add_ServiceOpsServicer_to_server(
        ServiceOps(runtime),
        server,
    )

    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)
    health_servicer.set(
        "agent.v1.AgentSearchService",
        health_pb2.HealthCheckResponse.SERVING,
    )
    health_servicer.set(
        "agent.v1.ServiceOps",
        health_pb2.HealthCheckResponse.SERVING,
    )

    addr = (
        f"{runtime.app_settings.agent.service_host}:"
        f"{runtime.app_settings.agent.service_port}"
    )
    server.add_insecure_port(addr)
    logger.info("Starting agent gRPC service on %s", addr)

    await server.start()
    try:
        await server.wait_for_termination()
    finally:
        await runtime.close()


def main() -> None:
    """Run the async ``serve`` coroutine from a synchronous entrypoint."""
    asyncio.run(serve())


if __name__ == "__main__":
    main()
