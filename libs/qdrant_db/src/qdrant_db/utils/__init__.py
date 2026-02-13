"""Utilities for Qdrant client."""

from qdrant_db.utils.dedup import DuplicateKeyError, deduplicate_items
from qdrant_db.utils.retry import retry_with_backoff

__all__ = ["retry_with_backoff", "deduplicate_items", "DuplicateKeyError"]
