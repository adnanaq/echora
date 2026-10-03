import asyncio

import pytest
from common.utils.request_batcher import BatchResultCountError, RequestBatcher
from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import HistogramDataPoint, InMemoryMetricReader
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode

_READER = InMemoryMetricReader()
metrics.set_meter_provider(MeterProvider(metric_readers=[_READER]))

_SPANS = InMemorySpanExporter()
_TRACER_PROVIDER = TracerProvider()
_TRACER_PROVIDER.add_span_processor(SimpleSpanProcessor(_SPANS))
trace.set_tracer_provider(_TRACER_PROVIDER)
_TRACER = trace.get_tracer("test")


class RecordingBatchFunction:
    def __init__(
        self, delay_seconds: float = 0.01, failure: Exception | None = None
    ) -> None:
        self.delay_seconds = delay_seconds
        self.failure = failure
        self.batches: list[list[str]] = []

    async def __call__(self, texts: list[str]) -> list[tuple[list[float], dict]]:
        self.batches.append(list(texts))
        await asyncio.sleep(self.delay_seconds)
        if self.failure is not None:
            raise self.failure
        return [([float(len(text))], {"text": text}) for text in texts]


async def test_concurrent_requests_share_one_batch_call():
    batch_function = RecordingBatchFunction()
    batcher = RequestBatcher(
        batch_function, max_batch_size=8, max_wait_seconds=0.005, concurrency=1
    )

    results = await asyncio.gather(
        *(batcher.submit(text) for text in ["a", "bb", "ccc"])
    )

    assert batch_function.batches == [["a", "bb", "ccc"]]
    assert results == [
        ([1.0], {"text": "a"}),
        ([2.0], {"text": "bb"}),
        ([3.0], {"text": "ccc"}),
    ]
    await batcher.close()


async def test_batches_never_exceed_the_maximum_size():
    batch_function = RecordingBatchFunction()
    batcher = RequestBatcher(
        batch_function, max_batch_size=4, max_wait_seconds=0.005, concurrency=1
    )

    texts = [f"text {number}" for number in range(10)]
    results = await asyncio.gather(*(batcher.submit(text) for text in texts))

    assert all(len(batch) <= 4 for batch in batch_function.batches)
    assert sorted(text for batch in batch_function.batches for text in batch) == sorted(
        texts
    )
    assert [sparse["text"] for _, sparse in results] == texts
    await batcher.close()


async def test_a_lone_request_is_not_delayed_without_a_wait():
    batch_function = RecordingBatchFunction(delay_seconds=0)
    batcher = RequestBatcher(
        batch_function, max_batch_size=8, max_wait_seconds=0, concurrency=1
    )

    loop = asyncio.get_running_loop()
    started = loop.time()
    await batcher.submit("pirate ship")

    assert loop.time() - started < 0.05
    assert batch_function.batches == [["pirate ship"]]
    await batcher.close()


async def test_requests_queued_during_a_call_form_the_next_batch():
    batch_function = RecordingBatchFunction(delay_seconds=0.05)
    batcher = RequestBatcher(
        batch_function, max_batch_size=8, max_wait_seconds=0, concurrency=1
    )

    first = asyncio.create_task(batcher.submit("first"))
    await asyncio.sleep(0.01)
    rest = [asyncio.create_task(batcher.submit(text)) for text in ["second", "third"]]
    await asyncio.gather(first, *rest)

    assert batch_function.batches == [["first"], ["second", "third"]]
    await batcher.close()


async def test_a_failed_batch_call_fails_every_request_in_it():
    batch_function = RecordingBatchFunction(failure=RuntimeError("batch failed"))
    batcher = RequestBatcher(
        batch_function, max_batch_size=8, max_wait_seconds=0.005, concurrency=1
    )

    results = await asyncio.gather(
        *(batcher.submit(text) for text in ["a", "b"]), return_exceptions=True
    )

    assert all(isinstance(result, RuntimeError) for result in results)
    await batcher.close()


async def test_batch_calls_run_at_the_same_time_up_to_the_concurrency():
    batch_function = RecordingBatchFunction(delay_seconds=0.05)
    batcher = RequestBatcher(
        batch_function, max_batch_size=1, max_wait_seconds=0, concurrency=2
    )

    loop = asyncio.get_running_loop()
    started = loop.time()
    await asyncio.gather(batcher.submit("a"), batcher.submit("b"))

    assert loop.time() - started < 0.09
    await batcher.close()


async def test_close_rejects_new_requests():
    batcher = RequestBatcher(
        RecordingBatchFunction(), max_batch_size=8, max_wait_seconds=0, concurrency=2
    )
    await batcher.submit("a")

    await batcher.close()

    with pytest.raises(RuntimeError, match="closed"):
        await batcher.submit("b")


def _histogram_points(metric_name: str, batcher: str) -> list[HistogramDataPoint]:
    data = _READER.get_metrics_data()
    if data is None:
        return []
    return [
        point
        for resource_metrics in data.resource_metrics
        for scope_metrics in resource_metrics.scope_metrics
        for metric in scope_metrics.metrics
        if metric.name == metric_name
        for point in metric.data.data_points
        if isinstance(point, HistogramDataPoint)
        and point.attributes.get("batcher") == batcher
    ]


async def _echo(items: list[str]) -> list[str]:
    await asyncio.sleep(0.01)
    return items


async def _slow_echo(items: list[str]) -> list[str]:
    await asyncio.sleep(0.05)
    return items


async def test_batch_size_is_recorded_per_batch_call():
    batcher = RequestBatcher(
        _echo, max_batch_size=8, max_wait_seconds=0.005, concurrency=1, name="sizes"
    )

    await asyncio.gather(*(batcher.submit(text) for text in ["a", "b", "c"]))
    await batcher.submit("d")

    (point,) = _histogram_points("echora_batcher_batch_size", "sizes")
    assert point.count == 2
    assert point.sum == 4
    await batcher.close()


async def test_queue_wait_is_recorded_for_every_request():
    batcher = RequestBatcher(
        _echo, max_batch_size=8, max_wait_seconds=0.005, concurrency=1, name="waits"
    )

    await asyncio.gather(*(batcher.submit(text) for text in ["a", "b", "c"]))

    (point,) = _histogram_points("echora_batcher_queue_wait_seconds", "waits")
    assert point.count == 3
    assert point.min >= 0
    await batcher.close()


async def test_queue_wait_includes_waiting_for_free_call_slot():
    batcher = RequestBatcher(
        _slow_echo, max_batch_size=1, max_wait_seconds=0, concurrency=1, name="slots"
    )

    await asyncio.gather(batcher.submit("first"), batcher.submit("second"))

    (point,) = _histogram_points("echora_batcher_queue_wait_seconds", "slots")
    assert point.count == 2
    assert point.max >= 0.04
    await batcher.close()


async def test_each_batcher_records_under_its_own_name():
    first = RequestBatcher(
        _echo, max_batch_size=8, max_wait_seconds=0, concurrency=1, name="model"
    )
    second = RequestBatcher(
        _echo, max_batch_size=8, max_wait_seconds=0, concurrency=1, name="qdrant"
    )

    await first.submit("a")
    await second.submit("b")
    await second.submit("c")

    (model_point,) = _histogram_points("echora_batcher_batch_size", "model")
    (qdrant_point,) = _histogram_points("echora_batcher_batch_size", "qdrant")
    assert model_point.count == 1
    assert qdrant_point.count == 2
    await first.close()
    await second.close()


async def _traced_echo(items: list[str]) -> list[str]:
    with _TRACER.start_as_current_span("downstream.call"):
        await asyncio.sleep(0.01)
    return items


async def _failing_call(items: list[str]) -> list[str]:
    raise RuntimeError("downstream failed")


async def _submit_in_span(batcher: RequestBatcher, name: str, item: str) -> str:
    with _TRACER.start_as_current_span(name):
        return await batcher.submit(item)


def _span(name: str) -> ReadableSpan:
    (span,) = [span for span in _SPANS.get_finished_spans() if span.name == name]
    return span


@pytest.fixture(autouse=True)
def _clear_spans() -> None:
    _SPANS.clear()


async def test_batch_call_runs_in_its_own_span_linked_to_each_caller():
    batcher = RequestBatcher(
        _traced_echo,
        max_batch_size=8,
        max_wait_seconds=0.005,
        concurrency=1,
        name="model",
    )

    await asyncio.gather(
        _submit_in_span(batcher, "search.first", "a"),
        _submit_in_span(batcher, "search.second", "b"),
    )

    batch = _span("batch.model")
    first, second = _span("search.first"), _span("search.second")
    assert batch.parent is None
    assert batch.context.trace_id not in {
        first.context.trace_id,
        second.context.trace_id,
    }
    assert {link.context.span_id for link in batch.links} == {
        first.context.span_id,
        second.context.span_id,
    }
    assert batch.attributes == {"batcher.name": "model", "batcher.batch_size": 2}
    await batcher.close()


async def test_each_caller_links_back_to_its_batch():
    batcher = RequestBatcher(
        _traced_echo,
        max_batch_size=8,
        max_wait_seconds=0.005,
        concurrency=1,
        name="model",
    )

    await asyncio.gather(
        _submit_in_span(batcher, "search.first", "a"),
        _submit_in_span(batcher, "search.second", "b"),
    )

    batch_context = _span("batch.model").context
    for name in ("search.first", "search.second"):
        assert [link.context.span_id for link in _span(name).links] == [
            batch_context.span_id
        ]


async def test_work_inside_batch_call_is_child_of_batch_span():
    batcher = RequestBatcher(
        _traced_echo, max_batch_size=8, max_wait_seconds=0, concurrency=1, name="model"
    )

    await _submit_in_span(batcher, "search.only", "a")

    downstream = _span("downstream.call")
    assert downstream.parent is not None
    assert downstream.parent.span_id == _span("batch.model").context.span_id


async def test_later_batches_do_not_join_first_callers_trace():
    batcher = RequestBatcher(
        _traced_echo, max_batch_size=8, max_wait_seconds=0, concurrency=1, name="model"
    )

    await _submit_in_span(batcher, "search.first", "a")
    await _submit_in_span(batcher, "search.second", "b")

    first_trace = _span("search.first").context.trace_id
    batch_traces = [
        span.context.trace_id
        for span in _SPANS.get_finished_spans()
        if span.name == "batch.model"
    ]
    assert len(batch_traces) == 2
    assert first_trace not in batch_traces


async def test_failed_batch_call_marks_batch_span_as_error():
    batcher = RequestBatcher(
        _failing_call, max_batch_size=8, max_wait_seconds=0, concurrency=1, name="model"
    )

    with pytest.raises(RuntimeError, match="downstream failed"):
        await _submit_in_span(batcher, "search.only", "a")

    batch = _span("batch.model")
    assert batch.status.status_code is StatusCode.ERROR
    assert [event.name for event in batch.events] == ["exception"]


async def test_requests_without_trace_still_get_batch_span():
    batcher = RequestBatcher(
        _traced_echo, max_batch_size=8, max_wait_seconds=0, concurrency=1, name="model"
    )

    assert await batcher.submit("a") == "a"

    assert _span("batch.model").links == ()


async def _returns_too_few(items: list[str]) -> list[str]:
    return items[:-1]


async def test_wrong_result_count_fails_every_request():
    batcher = RequestBatcher(
        _returns_too_few, max_batch_size=8, max_wait_seconds=0.005, concurrency=1
    )

    results = await asyncio.wait_for(
        asyncio.gather(
            *(batcher.submit(text) for text in ["a", "b"]), return_exceptions=True
        ),
        timeout=2,
    )

    assert all(isinstance(result, BatchResultCountError) for result in results)
    assert "1 results for 2 requests" in str(results[0])
    await batcher.close()


async def test_wrong_result_count_marks_batch_span_as_error():
    batcher = RequestBatcher(
        _returns_too_few,
        max_batch_size=8,
        max_wait_seconds=0,
        concurrency=1,
        name="model",
    )

    with pytest.raises(BatchResultCountError):
        await asyncio.wait_for(_submit_in_span(batcher, "search.only", "a"), timeout=2)

    assert _span("batch.model").status.status_code is StatusCode.ERROR
    await batcher.close()
