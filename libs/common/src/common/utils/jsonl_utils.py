"""Shared JSONL read/write utilities."""

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def append_jsonl(path: str, record: dict[str, Any]) -> None:
    """Append a single JSON record as a JSONL line. Logs and continues on failure."""
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")
    except Exception as e:
        logger.warning(f"Failed to append record to {path}: {e}")


def write_jsonl(path: str, records: list[dict[str, Any]]) -> None:
    """Write records as newline-delimited JSON, one record per line (overwrites)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
