import pytest
from review_page.review_case_messages import (
    convert_case_to_message,
    convert_message_to_case,
)


@pytest.mark.parametrize(
    "case_name", ["atashinchi_next", "gochuumon_movie", "nami", "aware_meisaku_kun"]
)
def test_convert_message_to_case_round_trip_gives_same_case(
    review_cases, case_name: str
) -> None:
    case = review_cases[case_name]
    assert (
        convert_message_to_case(convert_case_to_message(case)).model_dump_json()
        == case.model_dump_json()
    )


@pytest.mark.parametrize(
    "case_name", ["atashinchi_next", "gochuumon_movie", "nami", "aware_meisaku_kun"]
)
def test_convert_message_to_case_round_trip_keeps_entry_and_address_order(
    review_cases, case_name: str
) -> None:
    case = review_cases[case_name]
    converted_case = convert_message_to_case(convert_case_to_message(case))
    assert [
        (reference_page, list(entry.addresses_by_page_key))
        for reference_page, entry in converted_case.entries.items()
    ] == [
        (reference_page, list(entry.addresses_by_page_key))
        for reference_page, entry in case.entries.items()
    ]


def test_convert_message_to_case_round_trip_keeps_linking_entry_address_order(
    review_cases,
) -> None:
    case = review_cases["nami"]
    converted_case = convert_message_to_case(convert_case_to_message(case))
    assert [
        list(entry.addresses_by_page_key)
        for decision in converted_case.decisions
        for entry in decision.linking_entries
    ] == [
        list(entry.addresses_by_page_key)
        for decision in case.decisions
        for entry in decision.linking_entries
    ]


def test_convert_case_to_message_linked_page_fills_linked_page_decisions(
    review_cases,
) -> None:
    message = convert_case_to_message(review_cases["nami"])
    assert (
        len(message.linked_page_decisions),
        len(message.doubled_provider_decisions),
    ) == (1, 0)


def test_convert_case_to_message_answer_kept_as_json(review_cases) -> None:
    case = review_cases["nami"].model_copy(
        update={"answer": {"anilist:anime:102289": "skip"}}
    )
    message = convert_case_to_message(case)
    assert message.answer.struct_value["anilist:anime:102289"] == "skip"


def test_convert_case_to_message_open_case_leaves_answer_unset(review_cases) -> None:
    assert not convert_case_to_message(review_cases["nami"]).HasField("answer")
