"""Implement the AgentService gRPC handlers backed by match_suggestions."""

import logging
from pathlib import Path

import grpc
from agent_proto.v1 import agent_service_pb2, agent_service_pb2_grpc
from match_suggestions import (
    PAIR_JUDGMENTS_FILE_NAME,
    JsonlPairJudgmentStore,
    PairJudgmentStore,
    suggest_stored_page_answers,
)
from pydantic import ValidationError
from work_matching import ReviewCaseKind
from work_matching.review_case_messages import convert_message_to_case

logger = logging.getLogger(__name__)


class AgentServiceHandler(agent_service_pb2_grpc.AgentServiceServicer):
    """Handle AgentService RPC requests using stored Jev judgments."""

    def __init__(self, judgment_store: PairJudgmentStore) -> None:
        """Store the judgment store used to build page answer suggestions.

        Args:
            judgment_store: Store holding Jev's pairwise page judgments.
        """
        self._judgment_store = judgment_store

    async def SuggestPageAnswers(  # noqa: N802
        self,
        request: agent_service_pb2.SuggestPageAnswersRequest,
        context: grpc.aio.ServicerContext,
    ) -> agent_service_pb2.SuggestPageAnswersResponse:
        """Suggest places for open pages of an entry group review case.

        Args:
            request: gRPC request containing the target review case message.
            context: Active gRPC servicer context for status reporting.

        Returns:
            Response with confident page suggestions sorted by page key.
        """
        if not request.HasField("review_case"):
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "review_case must be provided",
            )
        try:
            review_case = convert_message_to_case(request.review_case)
        except (ValidationError, ValueError) as error:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"invalid review_case: {error}",
            )
        if (
            review_case.answer is not None
            or review_case.kind is ReviewCaseKind.LINKED_PAGE
        ):
            return agent_service_pb2.SuggestPageAnswersResponse()
        if not review_case.entries:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "entry group review_case must include at least one entry",
            )

        page_suggestions = suggest_stored_page_answers(
            review_case, self._judgment_store
        )
        return agent_service_pb2.SuggestPageAnswersResponse(
            page_suggestions=[
                agent_service_pb2.PageAnswerSuggestion(
                    page_key=suggestion.page_key,
                    answer_word=suggestion.answer_word,
                    score=suggestion.score,
                )
                for suggestion in page_suggestions
            ]
        )


def build_default_judgment_store(agent_data_dir: Path) -> JsonlPairJudgmentStore:
    """Construct the JSONL pair judgment store under the agent data directory.

    Args:
        agent_data_dir: Directory holding ``page_pair_judgments.jsonl``.

    Returns:
        Store backed by ``agent_data_dir / PAIR_JUDGMENTS_FILE_NAME``.
    """
    return JsonlPairJudgmentStore(agent_data_dir / PAIR_JUDGMENTS_FILE_NAME)
