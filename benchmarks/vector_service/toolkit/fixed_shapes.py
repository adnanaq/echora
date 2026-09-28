"""Fixed input sizes, for model runners that record a pass once per shape."""

from collections.abc import Sequence


class TooManyRowsError(ValueError):
    def __init__(self, rows: int, largest: int) -> None:
        super().__init__(f"{rows} rows do not fit the largest fixed size, {largest}")


def fixed_row_count(rows: int, row_sizes: Sequence[int]) -> int:
    """The smallest fixed size that holds ``rows`` rows."""
    for size in sorted(row_sizes):
        if size >= rows:
            return size
    raise TooManyRowsError(rows, max(row_sizes))
