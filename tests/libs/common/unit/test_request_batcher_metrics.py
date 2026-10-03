import asyncio

from common.utils.request_batcher import RequestBatcher
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import HistogramDataPoint, InMemoryMetricReader

_READER = InMemoryMetricReader()
metrics.set_meter_provider(MeterProvider(metric_readers=[_READER]))


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


async def test_queue_wait_includes_waiting_for_a_free_call_slot():
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
