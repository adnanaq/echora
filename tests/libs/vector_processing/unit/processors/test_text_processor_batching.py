import asyncio

from vector_processing.processors.text_processor import TextProcessor


def _sparse_model(mock_text_model):
    mock_text_model.encode_with_sparse.side_effect = lambda texts: (
        [[float(len(text))] * 1024 for text in texts],
        [{"indices": [len(text)], "values": [1.0]} for text in texts],
    )
    return mock_text_model


async def test_text_processor_combines_concurrent_searches(
    mock_text_model, mock_settings
):
    mock_settings.embed_batch_max_size = 16
    mock_settings.embed_batch_max_wait_ms = 5.0
    processor = TextProcessor(
        model=_sparse_model(mock_text_model), config=mock_settings
    )

    results = await asyncio.gather(
        *(processor.encode_text_with_sparse(text) for text in ["a", "bb", "ccc"])
    )

    assert mock_text_model.encode_with_sparse.call_count == 1
    assert [sparse["indices"] for _, sparse in results] == [[1], [2], [3]]
    await processor.close()


async def test_text_processor_encodes_each_search_alone_by_default(
    mock_text_model, mock_settings
):
    processor = TextProcessor(
        model=_sparse_model(mock_text_model), config=mock_settings
    )

    await asyncio.gather(
        *(processor.encode_text_with_sparse(text) for text in ["a", "bb", "ccc"])
    )

    assert mock_text_model.encode_with_sparse.call_count == 3
    await processor.close()
