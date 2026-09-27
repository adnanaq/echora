import asyncio

import pytest
from common.utils.request_batcher import RequestBatcher


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
    assert sorted(text for batch in batch_function.batches for text in batch) == sorted(texts)
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
