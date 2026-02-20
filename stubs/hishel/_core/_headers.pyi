"""Type stubs for hishel._core._headers module."""

from collections.abc import Iterable, Iterator

class Headers:
    """HTTP headers wrapper."""

    _headers: list[
        tuple[str, str]
    ]  # List of (key, value) tuples for multivalue support

    def __init__(self, headers: dict[str, str] | list[tuple[str, str]]) -> None:
        """
        Create a Headers instance populated from the given mapping of header names to values.

        Parameters:
            headers: Mapping of header names to header values, or list of (name, value) tuples.
        """
        ...

    def __getitem__(self, key: str) -> str:
        """
        Retrieve the value associated with the header name.

        Returns:
            str: The header value.
        """
        ...

    def __setitem__(self, key: str, value: str) -> None:
        """
        Set the header value for the given key.

        If the header already exists, replace its values with the provided single value.

        Parameters:
            key (str): Header name.
            value (str): Header value to set.
        """
        ...

    def get(self, key: str, default: str | None = None) -> str | None:
        """
        Return the header value for the given key or the provided default.

        Parameters:
            key (str): Header name to look up.
            default (str | None): Value to return if the header is not present.

        Returns:
            str | None: The header value if found, otherwise `default`.
        """
        ...

    def items(self) -> Iterable[tuple[str, str]]:
        """
        Return an iterable over (key, value) header pairs.

        Returns:
            Iterable of (header_name, header_value) tuples.
        """
        ...

    def __iter__(self) -> Iterator[str]:
        """
        Return an iterator over header names.

        Returns:
            Iterator of header names.
        """
        ...

    def keys(self) -> Iterable[str]:
        """
        Return an iterable over header names.

        Returns:
            Iterable of header names.
        """
        ...

    def values(self) -> Iterable[str]:
        """
        Return an iterable over header values.

        Returns:
            Iterable of header values.
        """
        ...
