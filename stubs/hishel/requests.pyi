"""Type stubs for hishel.requests module (v1.0)."""

from typing import Optional

from requests.adapters import HTTPAdapter

from . import SyncBaseStorage

class CacheAdapter(HTTPAdapter):
    """Cache adapter for requests library."""

    def __init__(
        self, storage: Optional[SyncBaseStorage] = None, **kwargs: object
    ) -> None: ...
