"""Type stubs for hishel._core._headers module."""

class Headers:
    """HTTP headers wrapper."""

    _headers: list[
        tuple[str, str]
    ]  # List of (key, value) tuples for multivalue support

    def __init__(self, headers: dict[str, str]) -> None:
        """
        Create a Headers instance populated from the given mapping of header names to values.

        Parameters:
            headers (Dict[str, str]): Mapping of header names to header values used to initialize the internal list of (name, value) header pairs.
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
