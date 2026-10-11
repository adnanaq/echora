from unittest.mock import create_autospec

import pytest
from common.models.anime import (
    Anime,
    AnimeRecord,
    AnimeStatus,
    AnimeType,
    Character,
    CharacterRole,
    Episode,
)
from vector_db_interface import VectorDocument
from vector_processing.processors.anime_field_mapper import AnimeFieldMapper
from vector_processing.processors.embedding_manager import MultiVectorEmbeddingManager
from vector_processing.processors.text_processor import TextProcessor
from vector_processing.processors.vision_processor import VisionProcessor

pytestmark = pytest.mark.asyncio

_ANIME_ID = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
_CHARACTER_ID = "char_01ARZ3NDEKTSV4RRFFQ69G5FA1"
_SPARSE = {"indices": [1, 5], "values": [0.4, 0.9]}


@pytest.fixture
def text_processor() -> TextProcessor:
    processor = create_autospec(TextProcessor, instance=True)
    processor.encode_text_with_sparse.return_value = ([0.1] * 1024, None)
    processor.encode_texts_batch_with_sparse.side_effect = lambda texts: (
        [[0.1] * 1024 for _ in texts],
        [None] * len(texts),
    )
    processor.get_zero_embedding.return_value = [0.0] * 1024
    return processor


@pytest.fixture
def vision_processor() -> VisionProcessor:
    processor = create_autospec(VisionProcessor, instance=True)
    processor.encode_images_batch.side_effect = lambda urls: [[0.2] * 768 for _ in urls]
    return processor


@pytest.fixture
def manager(text_processor, vision_processor) -> MultiVectorEmbeddingManager:
    return MultiVectorEmbeddingManager(
        text_processor=text_processor,
        vision_processor=vision_processor,
        field_mapper=AnimeFieldMapper(),
    )


def _anime(**fields: object) -> Anime:
    return Anime(
        **{
            "id": _ANIME_ID,
            "title": "Test Anime",
            "type": AnimeType.TV,
            "status": AnimeStatus.FINISHED,
            "sources": ["https://myanimelist.net/anime/1"],
            **fields,
        }
    )


def _character(**fields: object) -> Character:
    return Character(
        **{
            "id": _CHARACTER_ID,
            "name": "Test Char",
            "roles": [CharacterRole.MAIN],
            **fields,
        }
    )


def _full_record() -> AnimeRecord:
    return AnimeRecord(
        anime=_anime(images={"covers": ["https://example.com/cover.jpg"]}),
        characters=[
            _character(anime_ids=[_ANIME_ID], images=["https://example.com/char.jpg"])
        ],
        episodes=[Episode(episode_number=1, title="Test Episode", anime_id=_ANIME_ID)],
    )


def _document(documents: list[VectorDocument], entity_type: str) -> VectorDocument:
    return next(doc for doc in documents if doc.payload["entity_type"] == entity_type)


def _character_anime_ids(documents: list[VectorDocument]) -> list[str]:
    return _document(documents, "character").payload["anime_ids"]


async def test_process_anime_vectors_full_record_returns_one_point_per_entity(
    manager,
) -> None:
    documents = await manager.process_anime_vectors(_full_record())

    assert sorted(doc.payload["entity_type"] for doc in documents) == [
        "anime",
        "character",
        "episode",
    ]


async def test_process_anime_vectors_anime_point_holds_text_and_image_vectors(
    manager,
) -> None:
    anime = _document(await manager.process_anime_vectors(_full_record()), "anime")

    assert anime.id == _ANIME_ID
    assert anime.payload["title"] == "Test Anime"
    assert anime.vectors["text_vector"] == [0.1] * 1024
    assert anime.vectors["image_vector"] == [[0.2] * 768]


async def test_process_anime_vectors_character_point_holds_text_and_image_vectors(
    manager,
) -> None:
    character = _document(
        await manager.process_anime_vectors(_full_record()), "character"
    )

    assert character.id == _CHARACTER_ID
    assert character.vectors["image_vector"] == [[0.2] * 768]
    assert character.payload["anime_ids"] == [_ANIME_ID]


async def test_process_anime_vectors_episode_point_holds_text_vector_only(
    manager,
) -> None:
    episode = _document(await manager.process_anime_vectors(_full_record()), "episode")

    assert set(episode.vectors) == {"text_vector"}
    assert episode.payload["anime_id"] == _ANIME_ID


async def test_process_anime_vectors_sparse_output_adds_sparse_vector_to_every_point(
    manager, text_processor
) -> None:
    text_processor.encode_text_with_sparse.return_value = ([0.1] * 1024, _SPARSE)
    text_processor.encode_texts_batch_with_sparse.side_effect = lambda texts: (
        [[0.1] * 1024 for _ in texts],
        [_SPARSE for _ in texts],
    )

    documents = await manager.process_anime_vectors(_full_record())

    assert [doc.vectors["text_sparse_vector"] for doc in documents] == [_SPARSE] * 3


async def test_process_anime_vectors_character_without_id_returns_same_id_each_time(
    manager,
) -> None:
    record = AnimeRecord(
        anime=_anime(),
        characters=[Character(name="No ID Char", roles=[CharacterRole.MAIN])],
    )

    first = _document(await manager.process_anime_vectors(record), "character")
    second = _document(await manager.process_anime_vectors(record), "character")

    assert first.id
    assert first.id == second.id


async def test_process_anime_vectors_without_images_omits_image_vectors(
    manager,
) -> None:
    record = AnimeRecord(anime=_anime(), characters=[_character()], episodes=[])

    documents = await manager.process_anime_vectors(record)

    assert len(documents) == 2
    assert all("image_vector" not in doc.vectors for doc in documents)


async def test_process_anime_vectors_failed_image_encoding_omits_image_vector(
    manager, vision_processor
) -> None:
    vision_processor.encode_images_batch.side_effect = lambda urls: []
    record = AnimeRecord(
        anime=_anime(images={"covers": ["https://example.com/cover.jpg"]})
    )

    [anime] = await manager.process_anime_vectors(record)

    assert set(anime.vectors) == {"text_vector"}


async def test_process_anime_vectors_several_images_returns_one_row_per_image(
    manager,
) -> None:
    record = AnimeRecord(
        anime=_anime(
            images={
                "covers": ["https://example.com/cover.jpg"],
                "posters": ["https://example.com/poster.jpg"],
            }
        )
    )

    [anime] = await manager.process_anime_vectors(record)

    assert anime.vectors["image_vector"] == [[0.2] * 768, [0.2] * 768]


async def test_process_anime_batch_two_records_returns_every_point(manager) -> None:
    documents = await manager.process_anime_batch([_full_record(), _full_record()])

    assert len(documents) == 6
    assert all(isinstance(doc, VectorDocument) for doc in documents)


@pytest.mark.parametrize(
    ("character", "expected"),
    [
        (_character(anime_ids=[]), [_ANIME_ID]),
        (_character(), [_ANIME_ID]),
        (_character(anime_ids=[_ANIME_ID]), [_ANIME_ID]),
        (_character(anime_ids=["anime_EXISTING"]), ["anime_EXISTING", _ANIME_ID]),
        (
            _character(anime_ids=["anime_FIRST", "anime_SECOND"]),
            ["anime_FIRST", "anime_SECOND", _ANIME_ID],
        ),
    ],
)
async def test_process_anime_vectors_character_anime_ids_include_current_anime_once(
    manager, character: Character, expected: list[str]
) -> None:
    record = AnimeRecord(anime=_anime(), characters=[character])

    assert _character_anime_ids(await manager.process_anime_vectors(record)) == expected
