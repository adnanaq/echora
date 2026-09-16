"""Abstract interfaces for the crawler framework.

Defines the pluggable persistence boundary that ``BaseCrawler`` depends on:

- ``IRepository`` — persistence layer (save canonical output).
"""

from abc import ABC, abstractmethod
from typing import Any


class IRepository(ABC):
    """Abstract persistence layer for saving crawled output.

    Concrete implementations write to different backends (local file,
    database, in-memory store) without requiring changes in crawler code.
    """

    @abstractmethod
    def save(self, data: Any) -> None:
        """Persist data to the underlying storage backend.

        Args:
            data: The canonical data to persist.  Typically a ``dict`` but
                the interface accepts ``Any`` to remain generic.
        """
