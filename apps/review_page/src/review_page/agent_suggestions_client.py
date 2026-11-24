"""Read page answer suggestions from the AgentService gRPC server."""

import logging
from typing import Protocol

import grpc
from agent_proto.v1 import agent_service_pb2, agent_service_pb2_grpc
from review_page.models import AgentPageSuggestion
from work_matching import ReviewCase
from work_matching.review_case_messages import convert_case_to_message

logger = logging.getLogger(__name__)

_SUGGESTION_TIMEOUT_SECONDS = 5.0


class AgentSuggestionsClient(Protocol):
    """Fetch suggested placements for open pages of a review case."""

    def suggest_page_answers(self, case: ReviewCase) -> tuple[AgentPageSuggestion, ...]:
        """Return confident page suggestions for ``case``.

        Args:
            case: The open review case to request suggestions for.

        Returns:
            Tuple of ``AgentPageSuggestion`` items for the case's open pages.
        """


class GrpcAgentSuggestionsClient:
    """Call ``AgentService/SuggestPageAnswers`` over gRPC."""

    def __init__(
        self,
        target_address: str,
        *,
        timeout_seconds: float = _SUGGESTION_TIMEOUT_SECONDS,
    ) -> None:
        """Configure the gRPC target address and RPC timeout.

        Args:
            target_address: ``host:port`` address of the AgentService server.
            timeout_seconds: Maximum seconds to wait for ``SuggestPageAnswers``.
        """
        self._target_address = target_address
        self._timeout_seconds = timeout_seconds

    def suggest_page_answers(self, case: ReviewCase) -> tuple[AgentPageSuggestion, ...]:
        """Request Jev's page suggestions for ``case``; return empty on failure.

        Args:
            case: The open review case to request suggestions for.

        Returns:
            Tuple of ``AgentPageSuggestion`` items, or empty if the service is
            unavailable or returns no suggestions.
        """
        request = agent_service_pb2.SuggestPageAnswersRequest(
            review_case=convert_case_to_message(case)
        )
        try:
            with grpc.insecure_channel(self._target_address) as channel:
                stub = agent_service_pb2_grpc.AgentServiceStub(channel)
                response = stub.SuggestPageAnswers(
                    request, timeout=self._timeout_seconds
                )
        except grpc.RpcError as error:
            logger.warning("AgentService unavailable for %s: %s", case.case_name, error)
            return ()
        return tuple(
            AgentPageSuggestion(
                page_key=suggestion.page_key,
                answer_word=suggestion.answer_word,
                score=suggestion.score,
            )
            for suggestion in response.page_suggestions
        )
