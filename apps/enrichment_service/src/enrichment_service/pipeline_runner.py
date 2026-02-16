"""Helpers for enrichment pipeline execution and artifact writing."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from enrichment.programmatic.enrichment_pipeline import ProgrammaticEnrichmentPipeline


def load_database(file_path: str | Path) -> dict[str, Any]:
    """Load a JSON anime database from disk.

    Args:
        file_path: Path to the JSON database file.

    Returns:
        Parsed database document.

    Raises:
        OSError: If the file cannot be read.
        json.JSONDecodeError: If the file content is not valid JSON.
    """
    path = Path(file_path)
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def get_anime_by_index(database: dict[str, Any], index: int) -> dict[str, Any]:
    """Select one anime entry by positional index.

    Args:
        database: Parsed database payload.
        index: Zero-based index into `database["data"]`.

    Returns:
        Selected anime entry.

    Raises:
        ValueError: If the index is outside the available range.
    """
    data = database.get("data", [])
    if not (0 <= index < len(data)):
        raise ValueError(
            f"Index {index} out of range for database size {len(data)}"
        )
    return data[index]


def get_anime_by_title(database: dict[str, Any], title: str) -> dict[str, Any]:
    """Select one anime entry by title match.

    Matching strategy is exact-title first, then partial-title match.

    Args:
        database: Parsed database payload.
        title: Title text to search for.

    Returns:
        Selected anime entry.

    Raises:
        ValueError: If no match or multiple matches are found.
    """

    data = database.get("data", [])
    title_lower = title.lower()
    exact = [entry for entry in data if entry.get("title", "").lower() == title_lower]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise ValueError(f"Multiple exact matches found for title '{title}'")
    partial = [entry for entry in data if title_lower in entry.get("title", "").lower()]
    if len(partial) == 1:
        return partial[0]
    if not partial:
        raise ValueError(f"No anime found matching title '{title}'")
    raise ValueError(f"Multiple matches found for title '{title}'")


def _artifact_name(title: str) -> str:
    """Build a normalized artifact file name from title and timestamp.

    Args:
        title: Source anime title.

    Returns:
        File-name-safe artifact name.
    """
    clean = "".join(c.lower() if c.isalnum() else "_" for c in title).strip("_")
    clean = "_".join(filter(None, clean.split("_")))
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{clean}_{ts}.json"


def _artifact_payload(
    *,
    request: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    """Create artifact payload wrapper with metadata.

    Args:
        request: Normalized request metadata used for enrichment.
        result: Raw enrichment output.

    Returns:
        JSON-serializable artifact envelope.
    """
    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "source_inputs": request,
        "data": result,
    }


async def run_pipeline_and_write_artifact(
    *,
    file_path: str | Path,
    index: int | None,
    title: str | None,
    agent_dir: str | None,
    skip_services: list[str] | None,
    only_services: list[str] | None,
    output_dir: str | Path = "assets/seed_data",
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Run enrichment for one anime entry and persist an artifact.

    Args:
        file_path: Source database JSON file path.
        index: Optional index selector for source entry.
        title: Optional title selector for source entry.
        agent_dir: Optional agent output directory passed to pipeline.
        skip_services: Optional service names to skip.
        only_services: Optional service names to run exclusively.
        output_dir: Directory where artifact files are written.

    Returns:
        Tuple of output artifact path, raw enrichment result, and full
        artifact payload.

    Raises:
        ValueError: If neither `index` nor `title` is provided, or selection fails.
        OSError: If input database cannot be read or output artifact cannot be written.
        Exception: Propagates pipeline execution failures.
    """
    if index is None and not title:
        raise ValueError("Either index or title must be provided")

    file_path_obj = Path(file_path)
    database = load_database(file_path_obj)
    if index is not None:
        anime_data = get_anime_by_index(database, index)
    else:
        anime_data = get_anime_by_title(database, title or "")

    async with ProgrammaticEnrichmentPipeline() as pipeline:
        result = await pipeline.enrich_anime(
            anime_data,
            agent_dir=agent_dir,
            skip_services=skip_services,
            only_services=only_services,
        )

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    title_for_name = anime_data.get("title", "unknown")
    output_path = output_root / _artifact_name(title_for_name)
    payload = _artifact_payload(
        request={
            "file_path": str(file_path_obj),
            "index": index,
            "title": title,
            "agent_dir": agent_dir,
            "skip_services": skip_services or [],
            "only_services": only_services or [],
        },
        result=result,
    )
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return str(output_path), result, payload
