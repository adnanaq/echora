import pytest

from benchmarks.vector_service.toolkit.fixed_shapes import fixed_row_count


@pytest.mark.parametrize(
    ("rows", "expected"), [(1, 8), (8, 8), (9, 16), (16, 16), (17, 32), (32, 32)]
)
def test_fixed_row_count_rows_round_up_to_next_fixed_size(rows, expected):
    assert fixed_row_count(rows, (8, 16, 32)) == expected


def test_fixed_row_count_more_rows_than_largest_size_raises_value_error():
    with pytest.raises(ValueError, match="rows"):
        fixed_row_count(33, (8, 16, 32))
