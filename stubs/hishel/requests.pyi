"""Type stubs for hishel.requests module (v1.0)."""

from requests.adapters import HTTPAdapter

from . import SyncBaseStorage

class CacheAdapter(HTTPAdapter):
    """Cache adapter for requests library."""

    def __init__(
        self, storage: SyncBaseStorage | None = None, **kwargs: object
    ) -> None:
        """
        Initialize the CacheAdapter with an optional storage backend.

        Parameters:
            storage (Optional[SyncBaseStorage]): Storage implementation used to persist cached responses. If omitted, no storage backend is configured.
            **kwargs: Additional keyword arguments forwarded to the underlying HTTPAdapter initializer.
        """
        ...
