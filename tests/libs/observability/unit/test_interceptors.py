from types import SimpleNamespace
from unittest.mock import create_autospec

import grpc
import pytest
from observability.interceptors import AioServerInterceptor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode
from shared_proto.v1 import error_pb2
from vector_proto.v1 import vector_search_pb2


def _response(error_code: str | None = None) -> vector_search_pb2.SearchResponse:
    if error_code is None:
        return vector_search_pb2.SearchResponse()
    return vector_search_pb2.SearchResponse(
        error=error_pb2.ErrorDetails(code=error_code)
    )


def _context(code: grpc.StatusCode | None = None) -> grpc.aio.ServicerContext:
    context = create_autospec(grpc.aio.ServicerContext, instance=True)
    context.code.return_value = code
    context.invocation_metadata.return_value = ()
    return context


async def _call(behavior, context) -> None:
    async def continuation(_details):
        return grpc.unary_unary_rpc_method_handler(behavior)

    details = SimpleNamespace(method="/vector_service.v1.VectorSearchService/Search")
    handler = await AioServerInterceptor().intercept_service(continuation, details)
    await handler.unary_unary(object(), context)


def _only_span(spans: InMemorySpanExporter):
    (span,) = spans.get_finished_spans()
    return span


@pytest.mark.asyncio
async def test_intercept_service_successful_call_records_service_and_ok_status(
    span_exporter: InMemorySpanExporter,
):
    async def behavior(_request, _context):
        return _response()

    await _call(behavior, _context())

    span = _only_span(span_exporter)
    assert span.attributes["rpc.service"] == "vector_service.v1.VectorSearchService"
    assert span.attributes["rpc.method"] == "Search"
    assert span.attributes["rpc.grpc.status_code"] == grpc.StatusCode.OK.value[0]
    assert span.status.status_code is StatusCode.UNSET


@pytest.mark.asyncio
async def test_intercept_service_error_in_response_marks_span_as_error(
    span_exporter: InMemorySpanExporter,
):
    async def behavior(_request, _context):
        return _response(error_code="MISSING_QUERY_INPUT")

    await _call(behavior, _context())

    span = _only_span(span_exporter)
    assert span.status.status_code is StatusCode.ERROR
    assert span.status.description == "MISSING_QUERY_INPUT"
    assert span.attributes["rpc.error_code"] == "MISSING_QUERY_INPUT"
    assert span.attributes["rpc.grpc.status_code"] == grpc.StatusCode.OK.value[0]


@pytest.mark.asyncio
async def test_intercept_service_grpc_error_status_marks_span_as_error(
    span_exporter: InMemorySpanExporter,
):
    async def behavior(_request, _context):
        return _response()

    await _call(behavior, _context(grpc.StatusCode.NOT_FOUND))

    span = _only_span(span_exporter)
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes["rpc.grpc.status_code"] == grpc.StatusCode.NOT_FOUND.value[0]


@pytest.mark.asyncio
async def test_intercept_service_raised_exception_marks_span_as_error_with_unknown_code(
    span_exporter: InMemorySpanExporter,
):
    async def behavior(_request, _context):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        await _call(behavior, _context())

    span = _only_span(span_exporter)
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes["rpc.grpc.status_code"] == grpc.StatusCode.UNKNOWN.value[0]
