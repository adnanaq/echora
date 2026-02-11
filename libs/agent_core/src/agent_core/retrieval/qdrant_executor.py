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
from vector_processing import TextProcessor

logger = logging.getLogger(__name__)


class QdrantExecutor:
    """Executes bounded retrieval steps against Qdrant."""

    def __init__(
        self,
        qdrant: QdrantClient,
        text_processor: TextProcessor,
        vision_processor=None,
    ) -> None:
        """Initializes dependencies for vector retrieval.

        Args:
            qdrant: Async Qdrant client wrapper.
            text_processor: Text embedding processor.
            vision_processor: Optional image embedding processor.
        """
        self._qdrant = qdrant
        self._text_processor = text_processor
        self._vision_processor = vision_processor

    async def search(self, intent: SearchIntent, limit: int = 10) -> RetrievalResult:
        """Executes a retrieval step based on planner-provided intent.

        Args:
            intent: Structured search instruction from planner.
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
        ids = None
        if "id" in intent.filters and isinstance(intent.filters["id"], list):
            ids = [str(x) for x in intent.filters["id"] if x]

        if intent.query is None and ids:
            raw: list[dict[str, Any]] = []
            for pid in ids[:limit]:
                payload = await self._qdrant.get_by_id(pid, with_vectors=False)
                if payload:
                    raw.append({"id": pid, **payload})
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
            if self._vision_processor is None:
                return RetrievalResult(
                    summary="Image query provided but vision processor is not configured.",
                    raw_data=[],
                    count=0,
                )
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
        for r in raw[:10]:
            title = r.get("title") or r.get("name") or r.get("id") or "item"
            et = r.get("entity_type") or "unknown"
            score = r.get("similarity_score")
            if score is not None:
                lines.append(f"- {title} ({et}) score={score:.3f}")
            else:
                lines.append(f"- {title} ({et})")
        summary = "\n".join(lines) if lines else "No results."
        return RetrievalResult(summary=summary, raw_data=raw, count=len(raw))

    async def _embed_image(self, image_query: str) -> list[float] | None:
        """Embeds image input from URL, data URL, or raw base64.

        Args:
            image_query: Image reference string.

        Returns:
            Image embedding vector when successful, else ``None``.
        """
        q = image_query.strip()
        if q.startswith("http://") or q.startswith("https://"):
            matrix = await self._vision_processor.encode_images_batch([q])
            return matrix[0] if matrix else None

        # data:image/png;base64,....
        if q.startswith("data:"):
            try:
                header, b64 = q.split(",", 1)
                _ = header  # unused but keeps format explicit
                data = base64.b64decode(b64, validate=False)
            except Exception:
                logger.exception("Failed to parse data URL image_query")
                return None
            return await self._embed_bytes(data)

        # Raw base64 (best-effort). If it's not base64, this will likely fail.
        try:
            # Remove whitespace/newlines.
            b64 = re.sub(r"\s+", "", q)
            data = base64.b64decode(b64, validate=False)
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
            with tempfile.NamedTemporaryFile(delete=False, suffix=".img") as f:
                f.write(data)
                tmp_path = f.name
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
        refs: list[EntityRef] = []
        for r in raw:
            et = r.get("entity_type")
            rid = r.get("id") or r.get("_id") or r.get("anime_id")
            if not et or not rid:
                continue
            try:
                refs.append(EntityRef(entity_type=EntityType(et), id=str(rid)))
            except Exception:
                continue
        return refs
