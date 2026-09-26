from types import SimpleNamespace
from unittest.mock import MagicMock

import grpc
import pytest
from observability.interceptors import AioServerInterceptor
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode

_EXPORTER = InMemorySpanExporter()
_PROVIDER = TracerProvider()
_PROVIDER.add_span_processor(SimpleSpanProcessor(_EXPORTER))
trace.set_tracer_provider(_PROVIDER)


def _response(error_code: str | None = None) -> MagicMock:
    response = MagicMock(spec=["HasField", "error"])
    response.HasField.side_effect = lambda name: (
        name == "error" and error_code is not None
    )
    response.error.code = error_code
    return response


def _context(code: grpc.StatusCode | None = None) -> MagicMock:
    context = MagicMock()
    context.code.return_value = code
    context.invocation_metadata.return_value = ()
    return context


async def _call(behavior, context) -> None:
    async def continuation(_details):
        return grpc.unary_unary_rpc_method_handler(behavior)

    details = SimpleNamespace(method="/vector_service.v1.VectorSearchService/Search")
    handler = await AioServerInterceptor().intercept_service(continuation, details)
    await handler.unary_unary(object(), context)


@pytest.fixture(autouse=True)
def _clear_spans():
    _EXPORTER.clear()


def _only_span():
    (span,) = _EXPORTER.get_finished_spans()
    return span


@pytest.mark.asyncio
async def test_successful_call_records_service_and_ok_status_code():
    async def behavior(_request, _context):
        return _response()

    await _call(behavior, _context())

    span = _only_span()
    assert span.attributes["rpc.service"] == "vector_service.v1.VectorSearchService"
    assert span.attributes["rpc.method"] == "Search"
    assert span.attributes["rpc.grpc.status_code"] == grpc.StatusCode.OK.value[0]
    assert span.status.status_code is StatusCode.UNSET


@pytest.mark.asyncio
async def test_error_in_response_marks_span_as_error():
    async def behavior(_request, _context):
        return _response(error_code="MISSING_QUERY_INPUT")

    await _call(behavior, _context())

    span = _only_span()
    assert span.status.status_code is StatusCode.ERROR
    assert span.status.description == "MISSING_QUERY_INPUT"
    assert span.attributes["rpc.error_code"] == "MISSING_QUERY_INPUT"
    assert span.attributes["rpc.grpc.status_code"] == grpc.StatusCode.OK.value[0]


@pytest.mark.asyncio
async def test_grpc_status_error_marks_span_as_error():
    async def behavior(_request, _context):
        return _response()

    await _call(behavior, _context(grpc.StatusCode.NOT_FOUND))

    span = _only_span()
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes["rpc.grpc.status_code"] == grpc.StatusCode.NOT_FOUND.value[0]


@pytest.mark.asyncio
async def test_raised_exception_marks_span_as_error_with_unknown_code():
    async def behavior(_request, _context):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        await _call(behavior, _context())

    span = _only_span()
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes["rpc.grpc.status_code"] == grpc.StatusCode.UNKNOWN.value[0]
