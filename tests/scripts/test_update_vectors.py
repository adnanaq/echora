import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from qdrant_db.contracts import BatchOperationResult
from qdrant_db.errors import PermanentQdrantError
from update_vectors import update_vectors

pytestmark = pytest.mark.asyncio


def _anime_entry(anime_id: str | None, title: str) -> dict:
    return {
        "anime": {
            "id": anime_id,
            "title": title,
            "type": "TV",
            "status": "FINISHED",
            "sources": [],
        },
        "characters": [],
        "episodes": [],
    }


def _database_file(tmp_path: Path, entries: list) -> str:
    path = tmp_path / "anime_database.json"
    path.write_text(json.dumps({"data": entries}))
    return str(path)


def _embedding_manager() -> MagicMock:
    async def documents_for(records):
        return [
            SimpleNamespace(id=record.anime.id, payload={"entity_type": "anime"})
            for record in records
        ]

    manager = MagicMock()
    manager.process_anime_batch = AsyncMock(side_effect=documents_for)
    return manager


def _client_upserting_everything() -> MagicMock:
    async def upsert(documents, batch_size):
        return BatchOperationResult(
            total=len(documents), successful=len(documents), failed=0
        )

    client = MagicMock()
    client.add_documents = AsyncMock(side_effect=upsert)
    return client


async def test_upserts_every_valid_entry(tmp_path: Path) -> None:
    data_file = _database_file(
        tmp_path, [_anime_entry("a1", "First"), _anime_entry(None, "Second")]
    )
    client = _client_upserting_everything()

    summary = await update_vectors(
        client, _embedding_manager(), ["text_vector"], data_file=data_file
    )

    assert summary == {
        "total_anime": 2,
        "successful_count": 2,
        "failed_count": 0,
        "generation_failures": 0,
    }
    client.add_documents.assert_awaited_once()


async def test_skips_malformed_entries(tmp_path: Path) -> None:
    data_file = _database_file(
        tmp_path,
        [
            {"anime": None, "characters": [], "episodes": []},
            {"anime": "not a mapping", "characters": [], "episodes": []},
            {"anime": 123, "characters": [], "episodes": []},
            {"anime": [], "characters": [], "episodes": []},
            {"characters": [], "episodes": []},
            {"anime": {}, "characters": [], "episodes": []},
            _anime_entry("valid", "Valid"),
        ],
    )

    summary = await update_vectors(
        _client_upserting_everything(),
        _embedding_manager(),
        ["text_vector"],
        data_file=data_file,
    )

    assert summary["total_anime"] == 1
    assert summary["successful_count"] == 1


async def test_failed_upsert_marks_batch_failed_and_continues(tmp_path: Path) -> None:
    data_file = _database_file(
        tmp_path, [_anime_entry("a1", "First"), _anime_entry("a2", "Second")]
    )
    client = MagicMock()
    client.add_documents = AsyncMock(
        side_effect=[
            PermanentQdrantError("Failed to upsert documents"),
            BatchOperationResult(total=1, successful=1, failed=0),
        ]
    )

    summary = await update_vectors(
        client,
        _embedding_manager(),
        ["text_vector"],
        batch_size=1,
        data_file=data_file,
    )

    assert summary["successful_count"] == 1
    assert summary["failed_count"] == 1
    assert client.add_documents.await_count == 2
