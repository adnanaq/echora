from pathlib import Path

import pytest

from benchmarks.vector_service.quality.image_embedders import (
    SERVICE_MODEL,
    UnknownImageModelError,
    embeddings_store,
    load_embedder,
)


def test_service_model_keeps_the_original_embeddings_file():
    assert embeddings_store(Path("r"), SERVICE_MODEL) == Path("r/image_embeddings.npz")


def test_other_models_get_their_own_file():
    store = embeddings_store(Path("r"), "openclip:ViT-SO400M-14-SigLIP2-378/webli")

    assert store == Path(
        "r/image_embeddings_openclip_vit_so400m_14_siglip2_378_webli.npz"
    )


def test_unknown_model_kind_is_refused():
    with pytest.raises(UnknownImageModelError, match="hf-clip"):
        load_embedder("tensorflow:some-model")
