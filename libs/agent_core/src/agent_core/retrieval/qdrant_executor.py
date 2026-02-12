"""Qdrant retrieval executor for text/image/vector lookup steps."""

from __future__ import annotations

import logging
from typing import Any

import base64
import os
import re
import tempfile

from agent_core.schemas import EntityRef, EntityType, RetrievalResult, SearchIntent
from qdrant_db import QdrantClient

from langfuse import observe

from vector_processing import TextProcessor, VisionProcessor

logger = logging.getLogger(__name__)


class QdrantExecutor:
    """Executes bounded retrieval steps against Qdrant."""

    def __init__(
        self,
        qdrant: QdrantClient,
        text_processor: TextProcessor,
        vision_processor: VisionProcessor,
    ) -> None:
        """Initializes dependencies for vector retrieval.

        Args:
            qdrant: Async Qdrant client wrapper.
            text_processor: Text embedding processor.
            vision_processor: Image embedding processor.
        """
        self._qdrant = qdrant
        self._text_processor = text_processor
        self._vision_processor = vision_processor

    @observe()
    async def search(self, intent: SearchIntent, limit: int = 10) -> RetrievalResult:
        """Executes a retrieval step based on source-selected intent.

        Args:
            intent: Structured search instruction from source selection.
            limit: Maximum number of points to return.

        Returns:
            RetrievalResult with summary text and raw payload rows.
        """
        if intent.entity_type == EntityType.MANGA:
            # Current Qdrant collection stores anime/character/episode points only.
            return RetrievalResult(
                summary="Manga search is not available in Qdrant yet.",
                raw_data=[],
                count=0,
            )

        # Special-case filter-only lookup by ids (canonical UUIDs = Qdrant point IDs).
        entity_ids = None
        if "id" in intent.filters and isinstance(intent.filters["id"], list):
            entity_ids = [str(entity_id) for entity_id in intent.filters["id"] if entity_id]

        if intent.query is None and entity_ids:
            raw: list[dict[str, Any]] = []
            for point_id in entity_ids[:limit]:
                payload = await self._qdrant.get_by_id(point_id, with_vectors=False)
                if payload:
                    raw.append({"id": point_id, **payload})
            return self._as_result(raw, note=f"Retrieved {len(raw)} points by id.")

        if intent.query is None and not intent.image_query:
            return RetrievalResult(
                summary="No query provided and no supported id filter present.",
                raw_data=[],
                count=0,
            )

        text_embedding = None
        if intent.query is not None:
            text_embedding = await self._text_processor.encode_text(intent.query)
            if text_embedding is None:
                return RetrievalResult(
                    summary="Failed to embed text query.",
                    raw_data=[],
                    count=0,
                )

        image_embedding = None
        if intent.image_query:
            image_embedding = await self._embed_image(intent.image_query)
            if image_embedding is None:
                return RetrievalResult(
                    summary="Failed to embed image query.",
                    raw_data=[],
                    count=0,
                )

        results = await self._qdrant.search(
            text_embedding=text_embedding,
            image_embedding=image_embedding,
            entity_type=intent.entity_type.value,
            limit=limit,
            filters=intent.filters or None,
        )
        return self._as_result(results)

    def _as_result(self, raw: list[dict[str, Any]], note: str | None = None) -> RetrievalResult:
        """Builds a compact retrieval summary from raw rows.

        Args:
            raw: Raw result rows from Qdrant.
            note: Optional prefix line for the summary.

        Returns:
            Structured retrieval result for agent context.
        """
        lines: list[str] = []
        if note:
            lines.append(note)
        for row in raw[:10]:
            title = row.get("title") or row.get("name") or row.get("id") or "item"
            entity_type = row.get("entity_type") or "unknown"
            score = row.get("similarity_score")
            if score is not None:
                lines.append(f"- {title} ({entity_type}) score={score:.3f}")
            else:
                lines.append(f"- {title} ({entity_type})")
        summary = "\n".join(lines) if lines else "No results."
        return RetrievalResult(summary=summary, raw_data=raw, count=len(raw))

    async def _embed_image(self, image_query: str) -> list[float] | None:
        """Embeds image input from a data URL or raw base64 string.

        Args:
            image_query: Image reference string.

        Returns:
            Image embedding vector when successful, else ``None``.
        """
        normalized_query = image_query.strip()
        lowered_query = normalized_query.lower()
        if lowered_query.startswith("http://") or lowered_query.startswith("https://"):
            logger.warning(
                "Rejected image URL input; only base64/data URLs are accepted for image_query."
            )
            return None

        # data:image/png;base64,....
        if normalized_query.startswith("data:"):
            try:
                data_url_header, encoded_data = normalized_query.split(",", 1)
                _ = data_url_header  # unused but keeps format explicit
                encoded_data = re.sub(r"\s+", "", encoded_data)
                data = base64.b64decode(encoded_data, validate=True)
            except Exception:
                logger.exception("Failed to parse data URL image_query")
                return None
            return await self._embed_bytes(data)

        # Raw base64 (best-effort). If it's not base64, this will likely fail.
        try:
            # Remove whitespace/newlines.
            encoded_data = re.sub(r"\s+", "", normalized_query)
            data = base64.b64decode(encoded_data, validate=True)
        except Exception:
            logger.exception("Failed to decode base64 image_query")
            return None
        return await self._embed_bytes(data)

    async def _embed_bytes(self, data: bytes) -> list[float] | None:
        """Embeds image bytes through a temporary file path.

        Args:
            data: Raw image bytes.

        Returns:
            Image embedding vector when successful, else ``None``.
        """
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".img") as temp_file:
                temp_file.write(data)
                tmp_path = temp_file.name
            return await self._vision_processor.encode_image(tmp_path)
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    @staticmethod
    def extract_entity_refs(raw: list[dict[str, Any]]) -> list[EntityRef]:
        """Extracts canonical ``EntityRef`` objects from raw payloads.

        Args:
            raw: Raw result rows from Qdrant.

        Returns:
            Validated entity references.
        """
        entity_refs: list[EntityRef] = []
        for row in raw:
            entity_type = row.get("entity_type")
            entity_id = row.get("id") or row.get("_id") or row.get("anime_id")
            if not entity_type or not entity_id:
                continue
            try:
                entity_refs.append(
                    EntityRef(entity_type=EntityType(entity_type), id=str(entity_id))
                )
            except Exception:
                continue
        return entity_refs
