"""gRPC servicer adapter routing AgentService RPCs to individual agents."""

import grpc
from agent_proto.v1 import agent_service_pb2, agent_service_pb2_grpc
from agent_service.agents.review_cases import (
    ReviewCasesAgent,
    build_default_judgment_store,
)
from agent_service.runtime import AgentRuntime


class AgentRoutes(agent_service_pb2_grpc.AgentServiceServicer):
    """Delegate ``AgentService`` RPCs to dedicated domain agent modules."""

    def __init__(self, runtime: AgentRuntime) -> None:
        """Initialize agent instances from shared runtime configuration.

        Args:
            runtime: Shared agent service runtime configuration.
        """
        self.review_cases_agent = ReviewCasesAgent(
            build_default_judgment_store(runtime.agent_data_dir),
            typesafe_api_key=runtime.typesafe_api_key,
        )

    async def SuggestPageAnswers(  # noqa: N802
        self,
        request: agent_service_pb2.SuggestPageAnswersRequest,
        context: grpc.aio.ServicerContext,
    ) -> agent_service_pb2.SuggestPageAnswersResponse:
        """Delegate ``SuggestPageAnswers`` to ``ReviewCasesAgent``.

        Args:
            request: gRPC request containing the target review case.
            context: Active gRPC servicer context.

        Returns:
            Response with confident page suggestions.
        """
        return await self.review_cases_agent.suggest_page_answers(request, context)

    async def JudgeOpenCases(  # noqa: N802
        self,
        request: agent_service_pb2.JudgeOpenCasesRequest,
        context: grpc.aio.ServicerContext,
    ) -> agent_service_pb2.JudgeOpenCasesResponse:
        """Delegate ``JudgeOpenCases`` to ``ReviewCasesAgent``.

        Args:
            request: gRPC request containing open cases and synopses.
            context: Active gRPC servicer context.

        Returns:
            Summary counts of judged pairs and billed tokens.
        """
        return await self.review_cases_agent.judge_open_cases(request, context)
