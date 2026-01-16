"""Unit tests for MultiVectorEmbeddingManager."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from common.config import Settings
from common.models.anime import Anime, AnimeRecord, AnimeStatus, AnimeType, Character, Episode
from vector_db_interface import VectorDocument
from vector_processing.processors.embedding_manager import MultiVectorEmbeddingManager
from vector_processing.processors.text_processor import TextProcessor
from vector_processing.processors.vision_processor import VisionProcessor


@pytest.fixture
def mock_text_processor():
    processor = MagicMock(spec=TextProcessor)
    processor.encode_text.return_value = [0.1] * 1024
    processor.encode_texts_batch.side_effect = lambda texts: [[0.1] * 1024 for _ in texts]
    processor._get_zero_embedding.return_value = [0.0] * 1024
    return processor


@pytest.fixture
def mock_vision_processor():
    processor = MagicMock(spec=VisionProcessor)
    # Mock the downloader
    processor.downloader = AsyncMock()
    processor.downloader.download_and_cache_image = AsyncMock(
        return_value="/tmp/cached_image.jpg"
    )
    # Mock encode_image
    processor.encode_image = MagicMock(return_value=[0.2] * 768)
    return processor


@pytest.fixture
def sample_record():
    return AnimeRecord(
        anime=Anime(
            id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
            title="Test Anime",
            type=AnimeType.TV,
            status=AnimeStatus.FINISHED,
            sources=["https://myanimelist.net/anime/1"],
            images={"covers": ["https://example.com/cover.jpg"]},
        ),
        characters=[
            Character(
                id="char_01ARZ3NDEKTSV4RRFFQ69G5FA1",
                name="Test Char",
                role="Main",
                anime_ids=["01ARZ3NDEKTSV4RRFFQ69G5FAV"],
                images=["https://example.com/char.jpg"],
            )
        ],
        episodes=[
            Episode(
                episode_number=1,
                title="Test Episode",
                anime_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
            )
        ],
    )


@pytest.fixture
def sample_record_no_images():
    """Record without any images for edge case testing."""
    return AnimeRecord(
        anime=Anime(
            id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
            title="Test Anime No Images",
            type=AnimeType.TV,
            status=AnimeStatus.FINISHED,
            sources=["https://myanimelist.net/anime/1"],
        ),
        characters=[
            Character(
                id="char_01ARZ3NDEKTSV4RRFFQ69G5FA1",
                name="Test Char",
                role="Main",
                anime_ids=["01ARZ3NDEKTSV4RRFFQ69G5FAV"],
            )
        ],
        episodes=[],
    )

@pytest.mark.asyncio
async def test_process_anime_vectors_structure(
    mock_text_processor, mock_vision_processor, sample_record
):
    """Test that process_anime_vectors returns correct document hierarchy."""
    manager = MultiVectorEmbeddingManager(
        text_processor=mock_text_processor,
        vision_processor=mock_vision_processor,
        settings=Settings(),
    )

    documents = await manager.process_anime_vectors(sample_record)

    # Should have 5 documents: 1 Anime, 1 Character, 1 Episode, 1 Anime Image, 1 Char Image
    assert len(documents) == 5

    # Analyze Anime Point (text only, no image_vector)
    anime_doc = next(d for d in documents if d.payload["type"] == "anime")
    assert anime_doc.id == sample_record.anime.id
    assert "text_vector" in anime_doc.vectors
    assert "image_vector" not in anime_doc.vectors  # Images are separate points now
    assert anime_doc.payload["title"] == "Test Anime"

    # Analyze Character Point (text only, no image_vector)
    char_doc = next(d for d in documents if d.payload["type"] == "character")
    assert char_doc.id == "char_01ARZ3NDEKTSV4RRFFQ69G5FA1"
    assert "text_vector" in char_doc.vectors
    assert "image_vector" not in char_doc.vectors  # Images are separate points now
    assert char_doc.payload["anime_ids"] == [sample_record.anime.id]

    # Analyze Episode Point
    ep_doc = next(d for d in documents if d.payload["type"] == "episode")
    assert ep_doc.id.startswith("ep_")  # Should use generated ID
    assert "text_vector" in ep_doc.vectors
    assert ep_doc.payload["anime_id"] == sample_record.anime.id

    # Analyze Image Points
    image_docs = [d for d in documents if d.payload["type"] == "image"]
    assert len(image_docs) == 2  # 1 anime image + 1 character image

    # Check anime image point
    anime_image = next(d for d in image_docs if d.payload.get("anime_id"))
    assert anime_image.id.startswith("img_")
    assert "image_vector" in anime_image.vectors
    assert "text_vector" not in anime_image.vectors
    assert anime_image.payload["anime_id"] == sample_record.anime.id

    # Check character image point
    char_image = next(d for d in image_docs if d.payload.get("character_id"))
    assert char_image.id.startswith("img_")
    assert "image_vector" in char_image.vectors
    assert char_image.payload["character_id"] == "char_01ARZ3NDEKTSV4RRFFQ69G5FA1"

@pytest.mark.asyncio
async def test_process_anime_vectors_missing_char_id(mock_text_processor, mock_vision_processor):
    """Test deterministic ID generation for characters without IDs."""
    record = AnimeRecord(
        anime=Anime(
            id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
            title="Test Anime",
            type=AnimeType.TV,
            status=AnimeStatus.FINISHED,
            sources=["https://myanimelist.net/anime/1"],
        ),
        characters=[
            Character(name="No ID Char", role="Main")  # Missing ID
        ]
    )

    manager = MultiVectorEmbeddingManager(
        text_processor=mock_text_processor,
        vision_processor=mock_vision_processor,
        settings=Settings()
    )

    documents = await manager.process_anime_vectors(record)

    char_doc = next(d for d in documents if d.payload["type"] == "character")

    # Should be deterministic ID starting with 'char_' or similar (impl detail)
    # The current impl uses generate_deterministic_id with "character" type
    # which prefixes "char_"? No, it uses prefix from map.
    # Let's check impl:
    #   generate_deterministic_id(seed, "character")
    #   if entity_type: prefix = ENTITY_PREFIXES.get...
    # We should assume it generates a string ID.
    assert isinstance(char_doc.id, str)
    assert len(char_doc.id) > 0
    # Verify consistency
    documents2 = await manager.process_anime_vectors(record)
    char_doc2 = next(d for d in documents2 if d.payload["type"] == "character")
    assert char_doc.id == char_doc2.id

@pytest.mark.asyncio
async def test_process_anime_batch(mock_text_processor, mock_vision_processor, sample_record):
    """Test batch processing flattens results correctly."""
    manager = MultiVectorEmbeddingManager(
        text_processor=mock_text_processor,
        vision_processor=mock_vision_processor,
        settings=Settings(),
    )

    # Process batch of 2 identical records
    batch_docs = await manager.process_anime_batch([sample_record, sample_record])

    # Should contain 2 * 5 = 10 documents (1 anime + 1 char + 1 ep + 2 images each)
    assert len(batch_docs) == 10
    assert isinstance(batch_docs[0], VectorDocument)


@pytest.mark.asyncio
async def test_process_anime_vectors_no_images(
    mock_text_processor, mock_vision_processor, sample_record_no_images
):
    """Test processing record with no images creates no image points."""
    manager = MultiVectorEmbeddingManager(
        text_processor=mock_text_processor,
        vision_processor=mock_vision_processor,
        settings=Settings(),
    )

    documents = await manager.process_anime_vectors(sample_record_no_images)

    # Should have 2 documents: 1 Anime, 1 Character (no episode, no images)
    assert len(documents) == 2

    # No image points should exist
    image_docs = [d for d in documents if d.payload["type"] == "image"]
    assert len(image_docs) == 0


@pytest.mark.asyncio
async def test_image_processing_failure_handled(mock_text_processor, mock_vision_processor):
    """Test that image processing failures are handled gracefully."""
    # Make image download fail
    mock_vision_processor.downloader.download_and_cache_image = AsyncMock(return_value=None)

    record = AnimeRecord(
        anime=Anime(
            id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
            title="Test Anime",
            type=AnimeType.TV,
            status=AnimeStatus.FINISHED,
            sources=["https://myanimelist.net/anime/1"],
            images={"covers": ["https://example.com/cover.jpg"]},
        ),
        characters=[],
        episodes=[],
    )

    manager = MultiVectorEmbeddingManager(
        text_processor=mock_text_processor,
        vision_processor=mock_vision_processor,
        settings=Settings(),
    )

    documents = await manager.process_anime_vectors(record)

    # Should still have anime point, but no image points due to failure
    assert len(documents) == 1
    assert documents[0].payload["type"] == "anime"


@pytest.mark.asyncio
async def test_duplicate_image_urls_deduplicated(mock_text_processor, mock_vision_processor):
    """Test that duplicate image URLs are deduplicated."""
    record = AnimeRecord(
        anime=Anime(
            id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
            title="Test Anime",
            type=AnimeType.TV,
            status=AnimeStatus.FINISHED,
            sources=["https://myanimelist.net/anime/1"],
            images={
                "covers": ["https://example.com/same.jpg"],
                "posters": ["https://example.com/same.jpg"],  # Duplicate
            },
        ),
        characters=[],
        episodes=[],
    )

    manager = MultiVectorEmbeddingManager(
        text_processor=mock_text_processor,
        vision_processor=mock_vision_processor,
        settings=Settings(),
    )

    documents = await manager.process_anime_vectors(record)

    # Should have 2 documents: 1 Anime + 1 Image (deduplicated)
    image_docs = [d for d in documents if d.payload["type"] == "image"]
    assert len(image_docs) == 1
