import pytest
from work_matching.doubled_provider_decision import DoubledProviderOutcome
from work_matching.enum_names import read_member_by_name, write_member_name


def test_write_member_name_gives_member_name() -> None:
    assert write_member_name(DoubledProviderOutcome.SAME_PAGE) == "SAME_PAGE"


@pytest.mark.parametrize(
    "stored_value", ["SAME_PAGE", DoubledProviderOutcome.SAME_PAGE]
)
def test_read_member_by_name_name_or_member_gives_member(stored_value) -> None:
    read_outcome = read_member_by_name(DoubledProviderOutcome)
    assert read_outcome(stored_value) is DoubledProviderOutcome.SAME_PAGE


def test_read_member_by_name_sentence_raises_value_error() -> None:
    read_outcome = read_member_by_name(DoubledProviderOutcome)
    with pytest.raises(ValueError, match="SAME_PAGE"):
        read_outcome(DoubledProviderOutcome.SAME_PAGE.value + " (reworded)")
