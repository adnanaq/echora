"""Type stubs for hishel._core._storages._sync_base module."""

import abc
import uuid
from typing import Callable, List, Optional, Union

# Import from parent module
from hishel import Entry, Request, Response

class SyncBaseStorage(abc.ABC):
    """Base class for synchronous storage backends."""

    @abc.abstractmethod
    def create_entry(
        self,
        request: Request,
        response: Response,
        key: str,
        id_: Optional[uuid.UUID] = None,
    ) -> Entry:
        """Create and store a new cache entry."""
        ...

    @abc.abstractmethod
    def get_entries(self, key: str) -> List[Entry]:
        """Retrieve all entries for a given cache key."""
        ...

    @abc.abstractmethod
    def update_entry(
        self,
        id: uuid.UUID,
        new_entry: Union[Entry, Callable[[Entry], Entry]],
    ) -> Optional[Entry]:
        """Update an existing cache entry."""
        ...

    @abc.abstractmethod
    def remove_entry(self, id: uuid.UUID) -> None:
        """Remove (soft delete) an entry."""
        ...

    def close(self) -> None:
        """Optional cleanup method."""
        ...

    def is_soft_deleted(self, entry: Entry) -> bool:
        """Check if entry is soft deleted."""
        ...

    def is_safe_to_hard_delete(self, entry: Entry) -> bool:
        """Check if entry can be safely hard deleted."""
        ...

    def mark_pair_as_deleted(self, entry: Entry) -> Entry:
        """Mark entry as soft deleted."""
        ...
