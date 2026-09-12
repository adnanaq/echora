#!/usr/bin/env python3
"""
Stage 3: Multi-source relationship consolidation.

Every source helper already writes canonical, model-shaped payloads into the
agent's temp directory:

    related_anime:           dict[AnimeRelationType,         list[RelatedAnime]]
    related_source_material: dict[SourceMaterialRelationType, list[RelatedSourceMaterial]]

This stage merges those per-source payloads into one, so that a single
real-world work appears exactly once, under exactly one relation key, carrying
every source URL that mentioned it. Conflicts between sources are resolved by
the rules in docs/anime_relationship_and_format_type_mappings.md; the merge
itself lives in enrichment.pipeline.relationship_merger.

Usage:
    python scripts/process_stage3_relationships.py One_agent1
    python scripts/process_stage3_relationships.py One_agent1 --temp-dir custom_temp
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from enrichment.pipeline.identity import (
    OfflineDatabaseResolver,
    WorkIdentityResolver,
)
from enrichment.pipeline.relationship_merger import (
    PROVIDER_FILES,
    PROVIDER_PRIORITY,
    load_agent_providers,
    merge_agent_relationships,
    validate,
)

# Project root for resolving paths (works from anywhere)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_FILENAME = "stage3_relationships.json"

OFFLINE_DATABASE = PROJECT_ROOT / "assets" / "seed_data" / "anime-offline-database.json"


class MergedOutputInvalidError(ValueError):
    """Raised when merged entries do not satisfy the canonical models."""

    def __init__(self, errors: list[str]) -> None:
        """Build a message listing the first few validation failures.

        Args:
            errors: Human-readable errors returned by ``validate``.
        """
        detail = "\n".join(f"  {err}" for err in errors[:10])
        super().__init__(
            f"{len(errors)} merged entries failed model validation:\n{detail}"
        )


def build_resolver(path: Path) -> WorkIdentityResolver | None:
    """Load the offline database as a work identity resolver.

    Args:
        path: Path to the anime-offline-database JSON document.

    Returns:
        A resolver indexed over that document, or ``None`` when the file is
        absent or unreadable — the merge then falls back to fuzzy matching
        alone, which groups less accurately but still produces output.
    """
    if not path.exists():
        print(f"WARNING: {path} not found; merging without identity resolution.")
        return None
    try:
        resolver = OfflineDatabaseResolver.from_file(path)
    except (OSError, ValueError, KeyError) as exc:
        print(f"WARNING: could not load {path} ({exc}); merging without resolution.")
        return None
    print(f"Identity resolver: {len(resolver)} URLs indexed from {path.name}")
    return resolver


def process_all_relationships(
    temp_dir: str, resolver: WorkIdentityResolver | None = None
) -> dict[str, Any]:
    """Consolidate every source's relationship payload for one agent directory.

    Args:
        temp_dir: Agent directory holding the per-source ``*.jsonl`` files.
        resolver: Work identity resolver used to keep distinct works apart.
            When ``None`` the merge relies on fuzzy title matching alone.

    Returns:
        Mapping with ``related_anime`` and ``related_source_material``, each
        grouped by relation type and validated against the canonical models.

    Raises:
        FileNotFoundError: If the directory contains no recognised source files.
        MergedOutputInvalidError: If any merged entry fails model validation.
    """
    agent_dir = Path(temp_dir)
    records = load_agent_providers(agent_dir)
    if not records:
        raise FileNotFoundError(
            f"No source files found in {temp_dir}. Expected one or more of: "
            + ", ".join(PROVIDER_FILES[name] for name in PROVIDER_PRIORITY)
        )

    missing = [name for name in PROVIDER_PRIORITY if name not in records]
    raw_total = sum(
        len(entries)
        for record in records.values()
        for field in ("related_anime", "related_source_material")
        for entries in (record.get(field) or {}).values()
    )

    print(
        f"Sources found ({len(records)}/{len(PROVIDER_PRIORITY)}): {', '.join(records)}"
    )
    if missing:
        print(f"  not present: {', '.join(missing)}")
    print(f"Raw relationship entries across sources: {raw_total}")

    merged = merge_agent_relationships(agent_dir, resolver)

    anime_count = sum(len(v) for v in merged["related_anime"].values())
    manga_count = sum(len(v) for v in merged["related_source_material"].values())
    print(
        f"Merged into {anime_count + manga_count} distinct works "
        f"({anime_count} anime, {manga_count} source material)"
    )
    for relation, entries in merged["related_anime"].items():
        print(f"  {relation}: {len(entries)}")

    errors = validate(merged)
    if errors:
        # Writing anyway produces a file whose next consumer raises on load, and
        # a zero exit code says the stage succeeded. Fail here instead.
        raise MergedOutputInvalidError(errors)
    print("Model validation: OK")

    return merged


def write_output(merged: dict[str, Any], temp_dir: str) -> str:
    """Write the consolidated relationships to the agent directory.

    Args:
        merged: Output of :func:`process_all_relationships`.
        temp_dir: Agent directory to write into.

    Returns:
        Path of the file written.
    """
    output_file = os.path.join(temp_dir, OUTPUT_FILENAME)
    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(merged, handle, indent=2, ensure_ascii=False)
    return output_file


def resolve_temp_dir(args: argparse.Namespace) -> str:
    """Resolve the agent directory from either the agent_id or the legacy flag.

    Args:
        args: Parsed command line arguments.

    Returns:
        Absolute or relative path to the agent directory.

    Raises:
        SystemExit: If neither selector is given, or the directory is absent.
    """
    if args.agent_id:
        temp_base = Path(args.temp_dir)
        if not temp_base.is_absolute():
            temp_base = PROJECT_ROOT / temp_base
        temp_dir = str(temp_base / args.agent_id)
        if not os.path.exists(temp_dir):
            print(f"Error: Directory '{temp_dir}' does not exist")
            sys.exit(1)
        return temp_dir

    if args.current_anime:
        return os.path.dirname(args.current_anime)

    print("Error: provide an agent_id (e.g. One_agent1) or --current-anime")
    sys.exit(1)


def main() -> int:
    """Run stage 3 for one agent directory.

    Returns:
        ``0`` on success; failures exit non-zero via :func:`resolve_temp_dir`.
    """
    parser = argparse.ArgumentParser(
        description="Process Stage 3: Multi-source relationship consolidation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python process_stage3_relationships.py One_agent1
  python process_stage3_relationships.py One_agent1 --temp-dir custom_temp
        """,
    )
    parser.add_argument(
        "agent_id",
        nargs="?",
        help="Agent directory name to process (e.g., One_agent1)",
    )
    parser.add_argument(
        "--temp-dir", default="temp", help="Temporary directory path (default: temp)"
    )
    parser.add_argument(
        "--current-anime",
        type=str,
        help="Override current anime JSON file path (legacy, derives temp dir)",
    )
    parser.add_argument(
        "--offline-db",
        type=Path,
        default=OFFLINE_DATABASE,
        help=f"Offline database for identity resolution (default: {OFFLINE_DATABASE})",
    )
    args = parser.parse_args()

    temp_dir = resolve_temp_dir(args)
    merged = process_all_relationships(temp_dir, build_resolver(args.offline_db))
    output_file = write_output(merged, temp_dir)
    print(f"  - File saved: {output_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
