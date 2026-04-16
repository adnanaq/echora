"""Concrete ``IRepository`` implementations for the crawler framework.

Two implementations are provided:

- ``FileRepository`` — appends canonical output as a JSONL line via
  ``common.utils.jsonl_utils.append_jsonl``.
- ``NullRepository`` — discards output (useful when the caller handles
  persistence itself, or in unit tests).
"""

from typing import Any

from common.utils.jsonl_utils import append_jsonl
from enrichment.crawlers.framework.interfaces import IRepository
from enrichment.crawlers.utils import sanitize_output_path


class FileRepository(IRepository):
    """Appends canonical data to a JSONL file (one record per line).

    Each call to ``save`` appends a single JSON line to the file, creating
    the file and any missing parent directories if they do not yet exist.
    This matches the JSONL convention used throughout the enrichment pipeline
    and is safe for incremental writes.

    The output path is sanitised via ``sanitize_output_path`` before use,
    which rejects paths that would escape the working directory.

    Attributes:
        output_path: Resolved, sanitised absolute path for the output file.
    """

    def __init__(self, output_path: str):
        """Initialise with the destination file path.

        Args:
            output_path: Path to the JSONL file that will be written.  Must
                resolve to a location inside the current working directory.

        Raises:
            ValueError: If ``output_path`` escapes the working directory.
        """
        self.output_path = sanitize_output_path(output_path)

    def save(self, data: Any) -> None:
        """Append data as a single JSON line to the output file.

        Delegates to ``append_jsonl`` which handles directory creation and
        logs a warning (rather than raising) on I/O failure.

        Args:
            data: The canonical data to persist.  Must be JSON-serialisable.
        """
        append_jsonl(self.output_path, data)


class NullRepository(IRepository):
    """No-op repository that silently discards all data.

    Use this when the caller handles persistence itself (e.g. the helper
    collects results in memory) or when output is not needed (e.g. in tests
    that only verify return values).
    """

    def save(self, data: Any) -> None:
        """Discard data without performing any I/O.

        Args:
            data: Ignored.
        """
