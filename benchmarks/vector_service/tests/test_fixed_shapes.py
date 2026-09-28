import pytest

from benchmarks.vector_service.toolkit.fixed_shapes import fixed_row_count


@pytest.mark.parametrize(
    ("rows", "expected"), [(1, 8), (8, 8), (9, 16), (16, 16), (17, 32), (32, 32)]
)
def test_rows_round_up_to_the_next_fixed_size(rows, expected):
    assert fixed_row_count(rows, (8, 16, 32)) == expected


def test_more_rows_than_the_largest_size_is_refused():
    with pytest.raises(ValueError, match="rows"):
        fixed_row_count(33, (8, 16, 32))
