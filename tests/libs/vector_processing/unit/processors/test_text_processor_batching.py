import asyncio

from vector_processing.processors.text_processor import TextProcessor


def _sparse_model(text_model):
    text_model.encode_with_sparse.side_effect = lambda texts: (
        [[float(len(text))] * 1024 for text in texts],
        [{"indices": [len(text)], "values": [1.0]} for text in texts],
    )
    return text_model


async def test_encode_text_with_sparse_concurrent_searches_runs_one_model_pass(
    text_model, embedding_config
):
    batching_config = embedding_config.model_copy(
        update={"embed_batch_max_size": 16, "embed_batch_max_wait_ms": 5.0}
    )
    processor = TextProcessor(model=_sparse_model(text_model), config=batching_config)

    results = await asyncio.gather(
        *(processor.encode_text_with_sparse(text) for text in ["a", "bb", "ccc"])
    )

    assert text_model.encode_with_sparse.call_count == 1
    assert [sparse["indices"] for _, sparse in results] == [[1], [2], [3]]
    await processor.close()


async def test_encode_text_with_sparse_default_config_runs_one_model_pass_per_search(
    text_model, embedding_config
):
    processor = TextProcessor(model=_sparse_model(text_model), config=embedding_config)

    await asyncio.gather(
        *(processor.encode_text_with_sparse(text) for text in ["a", "bb", "ccc"])
    )

    assert text_model.encode_with_sparse.call_count == 3
    await processor.close()
