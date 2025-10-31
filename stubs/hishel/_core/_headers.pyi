"""Type stubs for hishel._core._headers module."""

from typing import Dict, List, Tuple

class Headers:
    """HTTP headers wrapper."""

    _headers: List[Tuple[str, str]]  # List of (key, value) tuples for multivalue support

    def __init__(self, headers: Dict[str, str]) -> None:
        """Initialize Headers from dict."""
        ...

    def __getitem__(self, key: str) -> str:
        """Get header value."""
        ...

    def __setitem__(self, key: str, value: str) -> None:
        """Set header value."""
        ...

    def get(self, key: str, default: str | None = None) -> str | None:
        """Get header with default."""
        ...
