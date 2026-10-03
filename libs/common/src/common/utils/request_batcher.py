"""Group concurrent single requests into shared batch calls."""

import asyncio
from collections.abc import Awaitable, Callable

from opentelemetry import metrics

_meter = metrics.get_meter("echora.request_batcher")
_queue_wait = _meter.create_histogram(
    "echora_batcher_queue_wait_seconds",
    unit="s",
    description="Time a request waited in a batcher's queue before its batch call started",
)
_batch_size = _meter.create_histogram(
    "echora_batcher_batch_size",
    unit="{request}",
    description="Requests in each batch call",
)


class BatcherClosedError(RuntimeError):
    """Raised for a request made after, or still waiting when, the batcher closed."""

    def __init__(self) -> None:
        super().__init__("RequestBatcher is closed")


type _QueuedItem[ItemT, ResultT] = tuple[ItemT, asyncio.Future[ResultT], float]
"""A queued request: the item, the future for its result, and when it was queued."""


class RequestBatcher[ItemT, ResultT]:
    """Combine concurrent single requests into one call on a batch function.

    Each caller submits one item and waits for its own result. One dispatcher
    forms batches: it takes whatever is queued, up to ``max_batch_size`` items,
    and starts a batch call for it, with at most ``concurrency`` calls running
    at once. While every call slot is busy, items keep queueing, so batches grow
    by themselves under load. With ``max_wait_seconds`` at 0 a lone request is
    processed at once; a small wait lets the dispatcher gather more items first.

    Each batch call records its size, and each request the time it waited
    before its batch call started (``echora_batcher_batch_size`` and
    ``echora_batcher_queue_wait_seconds``, labelled with the batcher's name).
    """

    def __init__(
        self,
        process_batch: Callable[[list[ItemT]], Awaitable[list[ResultT]]],
        max_batch_size: int,
        max_wait_seconds: float,
        concurrency: int,
        name: str = "unnamed",
    ) -> None:
        """Create the batcher; the dispatcher starts on first use.

        Args:
            process_batch: Processes a list of items and returns one result
                per item, in the same order.
            max_batch_size: Most items in one batch call.
            max_wait_seconds: How long the dispatcher waits for more items
                before starting a batch that is not full.
            concurrency: Batch calls that may run at the same time.
            name: Label for this batcher's metrics, e.g. ``"text_embedding"``.
        """
        self._process_batch = process_batch
        self._max_batch_size = max_batch_size
        self._max_wait_seconds = max_wait_seconds
        self._concurrency = concurrency
        self._metric_attributes = {"batcher": name}
        self._queue: asyncio.Queue[_QueuedItem[ItemT, ResultT]] | None = None
        self._call_slots: asyncio.Semaphore | None = None
        self._dispatcher: asyncio.Task[None] | None = None
        self._running: set[asyncio.Task[None]] = set()
        self._closed = False

    async def submit(self, item: ItemT) -> ResultT:
        """Process one item as part of a shared batch.

        Returns:
            The result for this item.

        Raises:
            BatcherClosedError: When the batcher has been closed.
        """
        if self._closed:
            raise BatcherClosedError()
        queue = self._start_dispatcher()
        loop = asyncio.get_running_loop()
        result: asyncio.Future[ResultT] = loop.create_future()
        queue.put_nowait((item, result, loop.time()))
        return await result

    async def close(self) -> None:
        """Stop the dispatcher and running calls, and fail any item still waiting."""
        self._closed = True
        tasks = [*self._running, *([self._dispatcher] if self._dispatcher else [])]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._running.clear()
        while self._queue is not None and not self._queue.empty():
            _, result, _ = self._queue.get_nowait()
            if not result.done():
                result.set_exception(BatcherClosedError())

    def _start_dispatcher(self) -> asyncio.Queue[_QueuedItem[ItemT, ResultT]]:
        """Create the queue and dispatcher inside the running event loop."""
        if self._queue is None:
            self._queue = asyncio.Queue()
            self._call_slots = asyncio.Semaphore(self._concurrency)
            self._dispatcher = asyncio.create_task(
                self._dispatch(self._queue, self._call_slots),
                name="request-batcher",
            )
        return self._queue

    async def _dispatch(
        self,
        queue: asyncio.Queue[_QueuedItem[ItemT, ResultT]],
        call_slots: asyncio.Semaphore,
    ) -> None:
        """Form batches from the queue and start a batch call for each."""
        while True:
            batch = [await queue.get()]
            if self._max_wait_seconds > 0 and queue.qsize() + 1 < self._max_batch_size:
                await asyncio.sleep(self._max_wait_seconds)
            await call_slots.acquire()
            while len(batch) < self._max_batch_size and not queue.empty():
                batch.append(queue.get_nowait())
            call = asyncio.create_task(self._process_and_deliver(batch))
            self._running.add(call)
            call.add_done_callback(self._running.discard)
            call.add_done_callback(lambda _: call_slots.release())

    async def _process_and_deliver(
        self, batch: list[_QueuedItem[ItemT, ResultT]]
    ) -> None:
        """Process a batch and hand each caller its own result or the error."""
        self._record_batch_start(batch)
        try:
            results = await self._process_batch([item for item, _, _ in batch])
        except Exception as exc:
            for _, result, _ in batch:
                if not result.done():
                    result.set_exception(exc)
            return
        for (_, result, _), value in zip(batch, results, strict=True):
            if not result.done():
                result.set_result(value)

    def _record_batch_start(self, batch: list[_QueuedItem[ItemT, ResultT]]) -> None:
        """Record the batch size and how long each request waited for it."""
        started = asyncio.get_running_loop().time()
        _batch_size.record(len(batch), self._metric_attributes)
        for _, _, queued_at in batch:
            _queue_wait.record(started - queued_at, self._metric_attributes)
