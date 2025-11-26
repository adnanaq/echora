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
        id: Optional[uuid.UUID] = None,
    ) -> Entry:
        """
        Create and store a cache entry for the given request/response under the provided key.

        Parameters:
            request (Request): The original request to associate with the entry.
            response (Response): The response to store alongside the request.
            key (str): Cache key under which the entry will be stored.
            id (Optional[uuid.UUID]): Optional UUID to assign as the entry's identifier; if omitted an identifier will be generated.

        Returns:
            Entry: The stored cache entry.
        """
        ...

    @abc.abstractmethod
    def get_entries(self, key: str) -> List[Entry]:
        """
        Return all stored Entry objects associated with the given cache key.
        
        Returns:
            List[Entry]: A list of Entry objects for `key`; empty list if no entries exist.
        """
        ...

    @abc.abstractmethod
    def update_entry(
        self,
        id: uuid.UUID,
        new_entry: Union[Entry, Callable[[Entry], Entry]],
    ) -> Optional[Entry]:
        """
        Update an existing cache entry identified by its UUID.
        
        Parameters:
            id (uuid.UUID): UUID of the entry to update.
            new_entry (Union[Entry, Callable[[Entry], Entry]]): Either an Entry to replace the existing one,
                or a callable that receives the current Entry and returns the updated Entry.
        
        Returns:
            Optional[Entry]: The updated Entry if an entry with `id` existed and was updated, `None` otherwise.
        """
        ...

    @abc.abstractmethod
    def remove_entry(self, id: uuid.UUID) -> None:
        """Remove (soft delete) an entry."""
        ...

    def close(self) -> None:
        """Optional cleanup method."""
        ...

    def is_soft_deleted(self, entry: Entry) -> bool:
        """
        Determine whether a cache entry has been marked as soft deleted.
        
        Parameters:
        	entry (Entry): The cache entry to inspect.
        
        Returns:
        	`true` if the entry is marked as soft deleted, `false` otherwise.
        """
        ...

    def is_safe_to_hard_delete(self, entry: Entry) -> bool:
        """
        Determine whether an entry is safe to permanently remove.
        
        Parameters:
        	entry (Entry): The cache entry to evaluate.
        
        Returns:
        	`true` if the entry can be permanently removed, `false` otherwise.
        """
        ...

    def mark_pair_as_deleted(self, entry: Entry) -> Entry:
        """
        Mark a cache entry as soft deleted and return the modified entry.
        
        Parameters:
            entry (Entry): The cache entry to mark as soft deleted.
        
        Returns:
            Entry: The entry instance marked as soft deleted (may be the same object or a modified copy).
        """
        ...