"""Type stubs for hishel._core._storages._async_base module."""

import abc
import uuid
from typing import Callable, List, Optional, Union

# Import from parent module
from hishel import Entry, Request, Response

class AsyncBaseStorage(abc.ABC):
    """Base class for asynchronous storage backends."""

    @abc.abstractmethod
    async def create_entry(
        self,
        request: Request,
        response: Response,
        key: str,
        id: Optional[uuid.UUID] = None,
    ) -> Entry:
        """
        Create and store a new cache Entry for the given request/response under the specified key.

        Parameters:
            request (Request): The request associated with the entry.
            response (Response): The response to be cached.
            key (str): The cache key under which the entry will be stored.
            id (Optional[uuid.UUID]): Optional UUID to assign to the new entry; if omitted, an identifier will be generated.

        Returns:
            Entry: The created cache entry.
        """
        ...

    @abc.abstractmethod
    async def get_entries(self, key: str) -> List[Entry]:
        """
        Retrieve all entries for a given cache key.
        
        Returns:
            entries (List[Entry]): List of `Entry` objects associated with `key`. Returns an empty list if no entries exist.
        """
        ...

    @abc.abstractmethod
    async def update_entry(
        self,
        id: uuid.UUID,
        new_entry: Union[Entry, Callable[[Entry], Entry]],
    ) -> Optional[Entry]:
        """
        Update an existing cache entry identified by `id`.
        
        If `new_entry` is an Entry instance it replaces the stored entry; if it is a callable, it is invoked with the current Entry and the returned Entry is stored. If no entry with `id` exists, nothing is changed.
        
        Parameters:
            id (uuid.UUID): Identifier of the entry to update.
            new_entry (Union[Entry, Callable[[Entry], Entry]]): Replacement Entry or a function that transforms the existing Entry into a new Entry.
        
        Returns:
            Optional[Entry]: The updated Entry if the entry was found and updated, `None` if no entry with `id` exists.
        """
        ...

    @abc.abstractmethod
    async def remove_entry(self, id: uuid.UUID) -> None:
        """
        Mark the cache entry identified by `id` as removed (soft delete) in the storage backend.
        
        Parameters:
            id (uuid.UUID): Identifier of the entry to mark as deleted.
        """
        ...

    async def close(self) -> None:
        """Optional cleanup method."""
        ...

    def is_soft_deleted(self, entry: Entry) -> bool:
        """
        Determine whether a cache entry is marked as soft deleted.
        
        Parameters:
            entry (Entry): Cache entry to inspect.
        
        Returns:
            True if the entry is marked as soft deleted, False otherwise.
        """
        ...

    def is_safe_to_hard_delete(self, entry: Entry) -> bool:
        """
        Determine whether a stored cache entry can be permanently (hard) deleted.
        
        Parameters:
        	entry (Entry): The cache entry to evaluate.
        
        Returns:
        	True if the entry can be hard deleted, False otherwise.
        """
        ...

    def mark_pair_as_deleted(self, entry: Entry) -> Entry:
        """
        Mark a cache entry as soft deleted.
        
        Parameters:
            entry (Entry): The entry to mark as deleted.
        
        Returns:
            Entry: The input entry with soft-deletion applied.
        """
        ...