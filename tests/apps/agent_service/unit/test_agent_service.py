"""Unit tests for AgentService gRPC handlers and health check setup."""

import asyncio
import socket
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import grpc
import pytest
from agent_proto.v1 import agent_service_pb2
from agent_service.main import _register_sigterm_shutdown, build_grpc_server, serve
from agent_service.service import AgentServiceHandler
from common.config import AgentConfig, ServiceConfig, Settings
from common.config.settings import Environment
from grpc_health.v1 import health_pb2, health_pb2_grpc
from match_suggestions import (
    PAIR_JUDGMENTS_FILE_NAME,
    QUESTION_VERSION,
    JsonlPairJudgmentStore,
    PairJudgment,
)
from review_page.agent_suggestions_client import GrpcAgentSuggestionsClient
from shared_proto.v1 import review_cases_pb2
from work_matching import (
    ProviderPage,
    ReviewCase,
    ReviewCaseKind,
    WorkEntry,
)
from work_matching.review_case_messages import convert_case_to_message


def _build_open_split_case() -> ReviewCase:
    first_page = ProviderPage(
        key="mal:anime:10",
        address="https://myanimelist.net/anime/10",
        titles=["Monster"],
        type="TV",
        year=2004,
    )
    second_page = ProviderPage(
        key="anilist:anime:19",
        address="https://anilist.co/anime/19",
        titles=["Monster"],
        type="TV",
        year=2004,
    )
    return ReviewCase(
        case_name="Monster [TV]",
        kind=ReviewCaseKind.SPLIT_WORK,
        reasons=(),
        decisions=(),
        entries={
            "mal:anime:10": WorkEntry(
                title="Monster",
                type="TV",
                year=2004,
                addresses_by_page_key={
                    "mal:anime:10": "https://myanimelist.net/anime/10"
                },
            ),
            "anilist:anime:19": WorkEntry(
                title="Monster",
                type="TV",
                year=2004,
                addresses_by_page_key={
                    "anilist:anime:19": "https://anilist.co/anime/19"
                },
            ),
        },
        pages=(first_page, second_page),
        placeable_page_keys=("mal:anime:10", "anilist:anime:19"),
        suggested_answer={"mal:anime:10": "work 1"},
    )


@pytest.mark.asyncio
async def test_suggest_page_answers_returns_stored_suggestions(tmp_path: Path) -> None:
    """Verify SuggestPageAnswers returns confident suggestions from stored judgments."""
    store = JsonlPairJudgmentStore(tmp_path / PAIR_JUDGMENTS_FILE_NAME)
    store.add_judgments(
        [
            PairJudgment(
                first_page_key="anilist:anime:19",
                second_page_key="mal:anime:10",
                question_version=QUESTION_VERSION,
                request_fingerprint="fp-1",
                is_same_franchise=0.95,
                is_same_work=0.92,
                is_first_work_part_of_second_work=0.05,
                is_second_work_part_of_first_work=0.04,
                judged_at=datetime.now(UTC),
            )
        ]
    )
    handler = AgentServiceHandler(store)
    review_case = _build_open_split_case()
    request = agent_service_pb2.SuggestPageAnswersRequest(
        review_case=convert_case_to_message(review_case)
    )
    context = MagicMock(spec=grpc.aio.ServicerContext)

    response = await handler.SuggestPageAnswers(request, context)

    assert len(response.page_suggestions) == 1
    assert response.page_suggestions[0].page_key == "anilist:anime:19"
    assert response.page_suggestions[0].answer_word == "work 1"
    assert response.page_suggestions[0].score == pytest.approx(0.92)


@pytest.mark.asyncio
async def test_suggest_page_answers_returns_empty_for_answered_case(
    tmp_path: Path,
) -> None:
    """Verify SuggestPageAnswers returns no suggestions when a case already has an answer."""
    store = JsonlPairJudgmentStore(tmp_path / PAIR_JUDGMENTS_FILE_NAME)
    handler = AgentServiceHandler(store)
    answered_case = _build_open_split_case().model_copy(
        update={"answer": {"mal:anime:10": "work 1", "anilist:anime:19": "work 1"}}
    )
    request = agent_service_pb2.SuggestPageAnswersRequest(
        review_case=convert_case_to_message(answered_case)
    )
    context = MagicMock(spec=grpc.aio.ServicerContext)

    response = await handler.SuggestPageAnswers(request, context)

    assert list(response.page_suggestions) == []


@pytest.mark.asyncio
async def test_suggest_page_answers_aborts_when_review_case_missing(
    tmp_path: Path,
) -> None:
    """Verify SuggestPageAnswers rejects requests missing review_case."""
    store = JsonlPairJudgmentStore(tmp_path / PAIR_JUDGMENTS_FILE_NAME)
    handler = AgentServiceHandler(store)
    request = agent_service_pb2.SuggestPageAnswersRequest()
    context = MagicMock(spec=grpc.aio.ServicerContext)
    context.abort = AsyncMock(side_effect=RuntimeError("aborted"))

    with pytest.raises(RuntimeError, match="aborted"):
        await handler.SuggestPageAnswers(request, context)

    context.abort.assert_awaited_once_with(
        grpc.StatusCode.INVALID_ARGUMENT,
        "review_case must be provided",
    )


@pytest.mark.asyncio
async def test_suggest_page_answers_aborts_on_invalid_review_case(
    tmp_path: Path,
) -> None:
    """Verify SuggestPageAnswers rejects malformed review_case payloads."""
    store = JsonlPairJudgmentStore(tmp_path / PAIR_JUDGMENTS_FILE_NAME)
    handler = AgentServiceHandler(store)
    invalid_message = review_cases_pb2.ReviewCase(case_name="bad")
    request = agent_service_pb2.SuggestPageAnswersRequest(review_case=invalid_message)
    context = MagicMock(spec=grpc.aio.ServicerContext)
    context.abort = AsyncMock(side_effect=RuntimeError("aborted"))

    with pytest.raises(RuntimeError, match="aborted"):
        await handler.SuggestPageAnswers(request, context)

    assert context.abort.await_args.args[0] == grpc.StatusCode.INVALID_ARGUMENT


@pytest.mark.asyncio
async def test_suggest_page_answers_rejects_invalid_review_case_message(
    tmp_path: Path,
) -> None:
    """Verify SuggestPageAnswers rejects protobuf review cases that fail validation."""
    store = JsonlPairJudgmentStore(tmp_path / PAIR_JUDGMENTS_FILE_NAME)
    handler = AgentServiceHandler(store)
    invalid_message = review_cases_pb2.ReviewCase(
        case_name="bad_case",
        doubled_provider_decisions=[
            review_cases_pb2.DoubledProviderDecision(
                provider="anilist",
                outcome="NOT_AN_OUTCOME",
            )
        ],
    )
    request = agent_service_pb2.SuggestPageAnswersRequest(review_case=invalid_message)
    context = MagicMock(spec=grpc.aio.ServicerContext)
    context.abort = AsyncMock(side_effect=RuntimeError("aborted"))

    with pytest.raises(RuntimeError, match="aborted"):
        await handler.SuggestPageAnswers(request, context)

    assert context.abort.await_args.args[0] == grpc.StatusCode.INVALID_ARGUMENT
    assert "invalid review_case" in context.abort.await_args.args[1]


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.mark.asyncio
async def test_grpc_end_to_end_suggest_page_answers_and_health_check(
    tmp_path: Path,
) -> None:
    """Verify AgentService serves SuggestPageAnswers and gRPC health checks end to end."""
    store = JsonlPairJudgmentStore(tmp_path / PAIR_JUDGMENTS_FILE_NAME)
    store.add_judgments(
        [
            PairJudgment(
                first_page_key="anilist:anime:19",
                second_page_key="mal:anime:10",
                question_version=QUESTION_VERSION,
                request_fingerprint="fp-1",
                is_same_franchise=0.95,
                is_same_work=0.92,
                is_first_work_part_of_second_work=0.05,
                is_second_work_part_of_first_work=0.04,
                judged_at=datetime.now(UTC),
            )
        ]
    )
    port = _pick_free_port()
    settings = Settings(
        environment=Environment.DEVELOPMENT,
        service=ServiceConfig(agent_service_host="127.0.0.1", agent_service_port=port),
        agent=AgentConfig(agent_data_dir=tmp_path),
    )
    server, listen_address = await build_grpc_server(settings)
    assert listen_address == f"127.0.0.1:{port}"
    await server.start()
    try:
        async with grpc.aio.insecure_channel(listen_address) as channel:
            health_stub = health_pb2_grpc.HealthStub(channel)
            health_response = await health_stub.Check(
                health_pb2.HealthCheckRequest(service="agent_service.v1.AgentService")
            )
            assert health_response.status == health_pb2.HealthCheckResponse.SERVING

        client = GrpcAgentSuggestionsClient(listen_address)
        suggestions = await asyncio.to_thread(
            client.suggest_page_answers, _build_open_split_case()
        )
        assert len(suggestions) == 1
        assert suggestions[0].page_key == "anilist:anime:19"
        assert suggestions[0].answer_word == "work 1"
        assert suggestions[0].score == pytest.approx(0.92)
    finally:
        await server.stop(grace=0)


def test_grpc_agent_suggestions_client_returns_empty_when_unreachable() -> None:
    """Verify GrpcAgentSuggestionsClient degrades gracefully when server is unreachable."""
    closed_port = _pick_free_port()
    client = GrpcAgentSuggestionsClient(f"127.0.0.1:{closed_port}", timeout_seconds=0.2)
    assert client.suggest_page_answers(_build_open_split_case()) == ()


@pytest.mark.asyncio
async def test_serve_and_sigterm_shutdown(tmp_path: Path) -> None:
    """Verify serve starts the server and SIGTERM registration stops it."""
    port = _pick_free_port()
    settings = Settings(
        environment=Environment.DEVELOPMENT,
        service=ServiceConfig(agent_service_host="127.0.0.1", agent_service_port=port),
        agent=AgentConfig(agent_data_dir=tmp_path),
    )
    fake_server = MagicMock(spec=grpc.aio.Server)
    fake_server.start = AsyncMock()
    fake_server.wait_for_termination = AsyncMock()
    fake_server.stop = AsyncMock()

    with (
        patch("agent_service.main.get_settings", return_value=settings),
        patch(
            "agent_service.main.build_grpc_server",
            AsyncMock(return_value=(fake_server, f"127.0.0.1:{port}")),
        ),
    ):
        await serve()

    fake_server.start.assert_awaited_once()
    fake_server.wait_for_termination.assert_awaited_once()

    mock_loop = MagicMock()
    with patch("asyncio.get_running_loop", return_value=mock_loop):
        _register_sigterm_shutdown(fake_server, grace=0)
    handle_sigterm = mock_loop.add_signal_handler.call_args.args[1]
    assert callable(handle_sigterm)
    handle_sigterm()
    await asyncio.sleep(0)
    fake_server.stop.assert_awaited_once_with(grace=0)


def test_register_sigterm_shutdown_handles_unsupported_platform() -> None:
    """Verify SIGTERM registration logs a warning instead of raising on unsupported loops."""
    fake_server = MagicMock(spec=grpc.aio.Server)
    mock_loop = MagicMock()
    mock_loop.add_signal_handler.side_effect = NotImplementedError

    with patch("asyncio.get_running_loop", return_value=mock_loop):
        _register_sigterm_shutdown(fake_server)


def test_main_invokes_asyncio_run() -> None:
    """Verify main delegates to asyncio.run(serve())."""
    from agent_service import main as main_module

    async def _noop() -> None:
        return None

    coro = _noop()
    try:
        with (
            patch.object(main_module, "serve", new=MagicMock(return_value=coro)),
            patch("asyncio.run") as run_mock,
        ):
            main_module.main()
        run_mock.assert_called_once_with(coro)
    finally:
        coro.close()
