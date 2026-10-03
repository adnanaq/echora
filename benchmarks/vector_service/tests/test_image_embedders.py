from pathlib import Path

import pytest

from benchmarks.vector_service.quality.image_embedders import (
    SERVICE_MODEL,
    UnknownImageModelError,
    embeddings_store,
    load_embedder,
)


def test_embeddings_store_service_model_returns_original_file():
    assert embeddings_store(Path("r"), SERVICE_MODEL) == Path("r/image_embeddings.npz")


def test_embeddings_store_other_model_returns_own_file():
    store = embeddings_store(Path("r"), "openclip:ViT-SO400M-14-SigLIP2-378/webli")

    assert store == Path(
        "r/image_embeddings_openclip_vit_so400m_14_siglip2_378_webli.npz"
    )


def test_load_embedder_unknown_model_kind_raises_unknown_image_model():
    with pytest.raises(UnknownImageModelError, match="hf-clip"):
        load_embedder("tensorflow:some-model")
