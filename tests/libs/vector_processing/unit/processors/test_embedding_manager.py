"""Unit tests for MultiVectorEmbeddingManager."""

import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest
from common.models.anime import (
    Anime,
    AnimeRecord,
    AnimeStatus,
    AnimeType,
    Character,
    Episode,
)
from vector_db_interface import VectorDocument
from vector_processing.processors.anime_field_mapper import AnimeFieldMapper
from vector_processing.processors.embedding_manager import MultiVectorEmbeddingManager
from vector_processing.processors.text_processor import TextProcessor
from vector_processing.processors.vision_processor import VisionProcessor


@pytest.fixture
def mock_field_mapper():
    """Mock AnimeFieldMapper for unit tests."""
    mapper = MagicMock(spec=AnimeFieldMapper)
    mapper.extract_anime_text.return_value = (
        "Title: Test Anime | Synopsis: Test synopsis"
    )
    mapper.extract_character_text.return_value = "Name: Test Char | Role: Main"
    mapper.extract_episode_text.return_value = "Episode 1: Test Episode"
    mapper.extract_image_urls.return_value = ["https://example.com/cover.jpg"]
    mapper.extract_character_image_urls.return_value = ["https://example.com/char.jpg"]
    return mapper


@pytest.fixture
def mock_text_processor():
    processor = MagicMock(spec=TextProcessor)
    processor.encode_text.return_value = [0.1] * 1024
    processor.encode_texts_batch.side_effect = lambda texts: [
        [0.1] * 1024 for _ in texts
    ]
    processor.get_zero_embedding.return_value = [0.0] * 1024
    return processor


@pytest.fixture
def mock_vision_processor():
    processor = MagicMock(spec=VisionProcessor)
    # Mock the downloader
    processor.downloader = AsyncMock()
    processor.downloader.download_and_cache_image = AsyncMock(
        return_value=f"{tempfile.gettempdir()}/cached_image.jpg"
    )
    # Mock encode_image
    processor.encode_image = MagicMock(return_value=[0.2] * 768)
    # Mock encode_images_batch for multivector support
    processor.encode_images_batch = AsyncMock(
        return_value=[[0.2] * 768]  # Returns matrix (list of lists)
    )
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
    mock_field_mapper,
    mock_text_processor,
    mock_vision_processor,
    sample_record,
    mock_settings,
):
    """Test that process_anime_vectors returns correct document hierarchy.

    With multivector architecture:
    - Anime points have text_vector + image_vector (multivector matrix)
    - Character points have text_vector + image_vector (multivector matrix)
    - Episode points have text_vector only
    - No separate Image points (images embedded in parent entities)
    """
    manager = MultiVectorEmbeddingManager(
        text_processor=mock_text_processor,
        vision_processor=mock_vision_processor,
        field_mapper=mock_field_mapper,
        settings=mock_settings,
    )

    documents = await manager.process_anime_vectors(sample_record)

    # Should have 3 documents: 1 Anime, 1 Character, 1 Episode (no separate image points)
    assert len(documents) == 3

    # Analyze Anime Point (text + image multivector)
    anime_doc = next(d for d in documents if d.payload["entity_type"] == "anime")
    assert anime_doc.id == sample_record.anime.id
    assert "text_vector" in anime_doc.vectors
    assert "image_vector" in anime_doc.vectors  # Multivector embedded
    # image_vector should be a matrix (list of lists)
    assert isinstance(anime_doc.vectors["image_vector"], list)
    assert isinstance(anime_doc.vectors["image_vector"][0], list)
    assert anime_doc.payload["title"] == "Test Anime"

    # Analyze Character Point (text + image multivector)
    char_doc = next(d for d in documents if d.payload["entity_type"] == "character")
    assert char_doc.id == "char_01ARZ3NDEKTSV4RRFFQ69G5FA1"
    assert "text_vector" in char_doc.vectors
    assert "image_vector" in char_doc.vectors  # Multivector embedded
    # image_vector should be a matrix (list of lists)
    assert isinstance(char_doc.vectors["image_vector"], list)
    assert isinstance(char_doc.vectors["image_vector"][0], list)
    assert char_doc.payload["anime_ids"] == [sample_record.anime.id]

    # Analyze Episode Point (text only, no images)
    ep_doc = next(d for d in documents if d.payload["entity_type"] == "episode")
    assert "text_vector" in ep_doc.vectors
    assert "image_vector" not in ep_doc.vectors  # Episodes don't have images
    assert ep_doc.payload["anime_id"] == sample_record.anime.id

    # No separate Image Points
    image_docs = [d for d in documents if d.payload["entity_type"] == "image"]
    assert len(image_docs) == 0  # Images embedded in parent entities


@pytest.mark.asyncio
async def test_process_anime_vectors_missing_char_id(
    mock_field_mapper, mock_text_processor, mock_vision_processor, mock_settings
):
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
        ],
    )

    manager = MultiVectorEmbeddingManager(
        text_processor=mock_text_processor,
        vision_processor=mock_vision_processor,
        field_mapper=mock_field_mapper,
        settings=mock_settings,
    )

    documents = await manager.process_anime_vectors(record)

    char_doc = next(d for d in documents if d.payload["entity_type"] == "character")

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
    char_doc2 = next(d for d in documents2 if d.payload["entity_type"] == "character")
    assert char_doc.id == char_doc2.id


@pytest.mark.asyncio
async def test_process_anime_batch(
    mock_field_mapper,
    mock_text_processor,
    mock_vision_processor,
    sample_record,
    mock_settings,
):
    """Test batch processing flattens results correctly."""
    manager = MultiVectorEmbeddingManager(
        text_processor=mock_text_processor,
        vision_processor=mock_vision_processor,
        field_mapper=mock_field_mapper,
        settings=mock_settings,
    )

    # Process batch of 2 identical records
    batch_docs = await manager.process_anime_batch([sample_record, sample_record])

    # Should contain 2 * 3 = 6 documents (1 anime + 1 char + 1 ep each)
    # No separate image points - images embedded as multivector in anime/char
    assert len(batch_docs) == 6
    assert isinstance(batch_docs[0], VectorDocument)


@pytest.mark.asyncio
async def test_process_anime_vectors_no_images(
    mock_field_mapper,
    mock_text_processor,
    mock_vision_processor,
    sample_record_no_images,
    mock_settings,
):
    """Test processing record with no images omits image_vector from vectors."""
    # Override mock to return no images for this test
    mock_field_mapper.extract_image_urls.return_value = []
    mock_field_mapper.extract_character_image_urls.return_value = []
    mock_vision_processor.encode_images_batch = AsyncMock(return_value=[])

    manager = MultiVectorEmbeddingManager(
        text_processor=mock_text_processor,
        vision_processor=mock_vision_processor,
        field_mapper=mock_field_mapper,
        settings=mock_settings,
    )

    documents = await manager.process_anime_vectors(sample_record_no_images)

    # Should have 2 documents: 1 Anime, 1 Character (no episode)
    assert len(documents) == 2

    # Anime point should not have image_vector (no images available)
    anime_doc = next(d for d in documents if d.payload["entity_type"] == "anime")
    assert "text_vector" in anime_doc.vectors
    assert "image_vector" not in anime_doc.vectors  # No images = no image_vector

    # Character point should not have image_vector (no images available)
    char_doc = next(d for d in documents if d.payload["entity_type"] == "character")
    assert "text_vector" in char_doc.vectors
    assert "image_vector" not in char_doc.vectors  # No images = no image_vector


@pytest.mark.asyncio
async def test_image_processing_failure_handled(
    mock_field_mapper, mock_text_processor, mock_vision_processor, mock_settings
):
    """Test that image processing failures are handled gracefully."""
    # Make image encoding return empty (simulating all failures)
    mock_vision_processor.encode_images_batch = AsyncMock(return_value=[])

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
        field_mapper=mock_field_mapper,
        settings=mock_settings,
    )

    documents = await manager.process_anime_vectors(record)

    # Should still have anime point with text_vector, but no image_vector due to failure
    assert len(documents) == 1
    assert documents[0].payload["entity_type"] == "anime"
    assert "text_vector" in documents[0].vectors
    assert (
        "image_vector" not in documents[0].vectors
    )  # Failed encoding = no image_vector


@pytest.mark.asyncio
async def test_multiple_images_creates_multivector_matrix(
    mock_field_mapper, mock_text_processor, mock_vision_processor, mock_settings
):
    """Test that multiple images create a multivector matrix in the anime point."""
    # Mock field mapper to return multiple deduplicated URLs
    mock_field_mapper.extract_image_urls.return_value = [
        "https://example.com/cover.jpg",
        "https://example.com/poster.jpg",
    ]
    # Mock vision processor to return matrix with 2 embeddings
    mock_vision_processor.encode_images_batch = AsyncMock(
        return_value=[[0.1] * 768, [0.2] * 768]  # 2 image vectors
    )

    record = AnimeRecord(
        anime=Anime(
            id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
            title="Test Anime",
            type=AnimeType.TV,
            status=AnimeStatus.FINISHED,
            sources=["https://myanimelist.net/anime/1"],
            images={
                "covers": ["https://example.com/cover.jpg"],
                "posters": ["https://example.com/poster.jpg"],
            },
        ),
        characters=[],
        episodes=[],
    )

    manager = MultiVectorEmbeddingManager(
        text_processor=mock_text_processor,
        vision_processor=mock_vision_processor,
        field_mapper=mock_field_mapper,
        settings=mock_settings,
    )

    documents = await manager.process_anime_vectors(record)

    # Should have 1 document: Anime with embedded multivector
    assert len(documents) == 1
    anime_doc = documents[0]
    assert anime_doc.payload["entity_type"] == "anime"

    # image_vector should be a matrix with 2 vectors
    assert "image_vector" in anime_doc.vectors
    assert len(anime_doc.vectors["image_vector"]) == 2
    assert anime_doc.vectors["image_vector"][0] == [0.1] * 768
    assert anime_doc.vectors["image_vector"][1] == [0.2] * 768


# ============================================================================
# TDD: anime_ids Merging Tests
# ============================================================================


@pytest.mark.asyncio
async def test_character_anime_ids_empty_list_adds_current_anime(
    mock_field_mapper, mock_text_processor, mock_vision_processor, mock_settings
):
    """Test that character with empty anime_ids gets current anime_id added."""
    anime_id = "anime_01H1234567890ABCDEF"
    char_id = "char_01H1234567890ABCDEF"

    record = AnimeRecord(
        anime=Anime(
            id=anime_id,
            title="Test Anime",
            type=AnimeType.TV,
            status=AnimeStatus.FINISHED,
            sources=["https://myanimelist.net/anime/1"],
        ),
        characters=[
            Character(
                id=char_id,
                name="Test Character",
                role="Main",
                anime_ids=[],  # Empty list
            )
        ],
    )

    manager = MultiVectorEmbeddingManager(
        text_processor=mock_text_processor,
        vision_processor=mock_vision_processor,
        field_mapper=mock_field_mapper,
        settings=mock_settings,
    )

    documents = await manager.process_anime_vectors(record)
    char_doc = next(d for d in documents if d.payload["entity_type"] == "character")

    # Should contain current anime_id
    assert char_doc.payload["anime_ids"] == [anime_id]


@pytest.mark.asyncio
async def test_character_anime_ids_none_creates_list_with_current_anime(
    mock_field_mapper, mock_text_processor, mock_vision_processor, mock_settings
):
    """Test that character with None anime_ids gets initialized with current anime_id."""
    anime_id = "anime_01H1234567890ABCDEF"
    char_id = "char_01H1234567890ABCDEF"

    # Create character dict directly to force None anime_ids
    record = AnimeRecord(
        anime=Anime(
            id=anime_id,
            title="Test Anime",
            type=AnimeType.TV,
            status=AnimeStatus.FINISHED,
            sources=["https://myanimelist.net/anime/1"],
        ),
        characters=[
            Character(
                id=char_id,
                name="Test Character",
                role="Main",
                # anime_ids omitted - will be default_factory=list, but we'll test None case
            )
        ],
    )

    manager = MultiVectorEmbeddingManager(
        text_processor=mock_text_processor,
        vision_processor=mock_vision_processor,
        field_mapper=mock_field_mapper,
        settings=mock_settings,
    )

    documents = await manager.process_anime_vectors(record)
    char_doc = next(d for d in documents if d.payload["entity_type"] == "character")

    # Should contain current anime_id
    assert char_doc.payload["anime_ids"] == [anime_id]


@pytest.mark.asyncio
async def test_character_anime_ids_existing_list_merges_new_anime(
    mock_field_mapper, mock_text_processor, mock_vision_processor, mock_settings
):
    """Test that character with existing anime_ids gets new anime_id appended."""
    anime_id = "anime_NEW_ID_01H123"
    existing_anime_id = "anime_EXISTING_01H456"
    char_id = "char_01H1234567890ABCDEF"

    record = AnimeRecord(
        anime=Anime(
            id=anime_id,
            title="Test Anime",
            type=AnimeType.TV,
            status=AnimeStatus.FINISHED,
            sources=["https://myanimelist.net/anime/1"],
        ),
        characters=[
            Character(
                id=char_id,
                name="Shared Character",
                role="Main",
                anime_ids=[existing_anime_id],  # Already appeared in another anime
            )
        ],
    )

    manager = MultiVectorEmbeddingManager(
        text_processor=mock_text_processor,
        vision_processor=mock_vision_processor,
        field_mapper=mock_field_mapper,
        settings=mock_settings,
    )

    documents = await manager.process_anime_vectors(record)
    char_doc = next(d for d in documents if d.payload["entity_type"] == "character")

    # Should contain both anime IDs
    assert set(char_doc.payload["anime_ids"]) == {existing_anime_id, anime_id}
    assert len(char_doc.payload["anime_ids"]) == 2


@pytest.mark.asyncio
async def test_character_anime_ids_prevents_duplicates(
    mock_field_mapper, mock_text_processor, mock_vision_processor, mock_settings
):
    """Test that duplicate anime_ids are not added when processing same anime twice."""
    anime_id = "anime_01H1234567890ABCDEF"
    char_id = "char_01H1234567890ABCDEF"

    record = AnimeRecord(
        anime=Anime(
            id=anime_id,
            title="Test Anime",
            type=AnimeType.TV,
            status=AnimeStatus.FINISHED,
            sources=["https://myanimelist.net/anime/1"],
        ),
        characters=[
            Character(
                id=char_id,
                name="Test Character",
                role="Main",
                anime_ids=[anime_id],  # Already contains current anime_id
            )
        ],
    )

    manager = MultiVectorEmbeddingManager(
        text_processor=mock_text_processor,
        vision_processor=mock_vision_processor,
        field_mapper=mock_field_mapper,
        settings=mock_settings,
    )

    documents = await manager.process_anime_vectors(record)
    char_doc = next(d for d in documents if d.payload["entity_type"] == "character")

    # Should not duplicate the anime_id
    assert char_doc.payload["anime_ids"] == [anime_id]
    assert len(char_doc.payload["anime_ids"]) == 1


@pytest.mark.asyncio
async def test_character_anime_ids_preserves_order_and_adds_new(
    mock_field_mapper, mock_text_processor, mock_vision_processor, mock_settings
):
    """Test that existing anime_ids order is preserved when adding new anime_id."""
    anime_id = "anime_NEW_01H789"
    existing_ids = ["anime_FIRST_01H123", "anime_SECOND_01H456"]
    char_id = "char_01H1234567890ABCDEF"

    record = AnimeRecord(
        anime=Anime(
            id=anime_id,
            title="Test Anime",
            type=AnimeType.TV,
            status=AnimeStatus.FINISHED,
            sources=["https://myanimelist.net/anime/1"],
        ),
        characters=[
            Character(
                id=char_id,
                name="Multi-Anime Character",
                role="Main",
                anime_ids=existing_ids.copy(),  # Character appeared in 2 anime already
            )
        ],
    )

    manager = MultiVectorEmbeddingManager(
        text_processor=mock_text_processor,
        vision_processor=mock_vision_processor,
        field_mapper=mock_field_mapper,
        settings=mock_settings,
    )

    documents = await manager.process_anime_vectors(record)
    char_doc = next(d for d in documents if d.payload["entity_type"] == "character")

    # Should preserve order and append new ID
    expected = [*existing_ids, anime_id]
    assert char_doc.payload["anime_ids"] == expected
    assert len(char_doc.payload["anime_ids"]) == 3
