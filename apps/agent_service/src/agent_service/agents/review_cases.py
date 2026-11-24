"""Jev agent handlers for review case suggestions and pair judging."""

import logging
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import Any

import grpc
from agent_proto.v1 import agent_service_pb2
from match_suggestions import (
    PAIR_JUDGMENTS_FILE_NAME,
    JsonlPairJudgmentStore,
    PairJudgmentStore,
    judge_pair_requests,
    list_pair_requests,
    suggest_stored_page_answers,
)
from pydantic import ValidationError
from typesafe_sdk import AsyncTypeSafeClient
from work_matching import ProviderPage, ReviewCaseKind
from work_matching.review_case_messages import convert_message_to_case

logger = logging.getLogger(__name__)

ClientFactory = Callable[[str], AbstractAsyncContextManager[Any]]


def build_default_typesafe_client(
    api_key: str,
) -> AbstractAsyncContextManager[Any]:
    """Create an async TypeSafe client context manager.

    Args:
        api_key: TypeSafe API key for Jev inference.

    Returns:
        Async context manager yielding an ``AsyncTypeSafeClient``.
    """
    return AsyncTypeSafeClient(api_key=api_key)


def build_default_judgment_store(agent_data_dir: Path) -> JsonlPairJudgmentStore:
    """Construct the JSONL pair judgment store under the agent data directory.

    Args:
        agent_data_dir: Directory holding ``page_pair_judgments.jsonl``.

    Returns:
        Store backed by ``agent_data_dir / PAIR_JUDGMENTS_FILE_NAME``.
    """
    return JsonlPairJudgmentStore(agent_data_dir / PAIR_JUDGMENTS_FILE_NAME)


class ReviewCasesAgent:
    """Suggest review-case placements and judge open review-case pairs with Jev."""

    def __init__(
        self,
        judgment_store: PairJudgmentStore,
        *,
        typesafe_api_key: str | None = None,
        client_factory: ClientFactory | None = None,
    ) -> None:
        """Initialize the review-cases agent with its store and TypeSafe settings.

        Args:
            judgment_store: Store holding Jev's pairwise page judgments.
            typesafe_api_key: Optional API key for live Jev pair judging.
            client_factory: Optional factory for async TypeSafe clients.
        """
        self._judgment_store = judgment_store
        self._typesafe_api_key = typesafe_api_key
        self._client_factory = client_factory or build_default_typesafe_client

    async def suggest_page_answers(
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

    async def judge_open_cases(
        self,
        request: agent_service_pb2.JudgeOpenCasesRequest,
        context: grpc.aio.ServicerContext,
    ) -> agent_service_pb2.JudgeOpenCasesResponse:
        """Judge unstored open review case pairs using provided synopses.

        Args:
            request: gRPC request carrying open review cases and synopses.
            context: Active gRPC servicer context for status reporting.

        Returns:
            Summary counts of judged pairs, failures, and billed input tokens.
        """
        try:
            review_cases = [
                convert_message_to_case(message) for message in request.review_cases
            ]
        except (ValidationError, ValueError) as error:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"invalid review_case: {error}",
            )
        if not self._typesafe_api_key:
            logger.info("JudgeOpenCases skipped: TYPESAFE_API_KEY is not set")
            return agent_service_pb2.JudgeOpenCasesResponse()

        async def _read_synopsis(page: ProviderPage) -> str | None:
            return request.synopses_by_page_key.get(page.key)

        pair_requests = await list_pair_requests(
            review_cases, self._judgment_store, _read_synopsis
        )
        if not pair_requests:
            return agent_service_pb2.JudgeOpenCasesResponse()
        async with self._client_factory(self._typesafe_api_key) as client:
            judging_summary = await judge_pair_requests(
                pair_requests, self._judgment_store, client
            )
        return agent_service_pb2.JudgeOpenCasesResponse(
            judged_pair_count=judging_summary.judged_pair_count,
            failed_pair_count=judging_summary.failed_pair_count,
            input_token_count=judging_summary.input_token_count,
        )
