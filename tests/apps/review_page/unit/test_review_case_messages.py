import pytest
from google.protobuf import json_format
from review_page.review_case_messages import (
    build_case_answer,
    convert_case_to_message,
    convert_message_to_case,
)
from work_matching import LinkedPageDecision, TargetEntrySuggestion


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
        if isinstance(decision, LinkedPageDecision)
        for entry in decision.linking_entries
    ] == [
        list(entry.addresses_by_page_key)
        for decision in case.decisions
        if isinstance(decision, LinkedPageDecision)
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


def test_build_case_answer_seen_saved_answer_given_sets_it() -> None:
    case_answer = build_case_answer(
        "split work: a",
        {"mal:anime:1": "work 1"},
        seen_saved_answers={"split work: a": None},
    )
    assert case_answer.HasField("seen_saved_answer")
    assert json_format.MessageToDict(case_answer.seen_saved_answer) is None


def test_build_case_answer_without_seen_saved_answer_leaves_it_unset() -> None:
    case_answer = build_case_answer("split work: a", {"mal:anime:1": "work 1"})
    assert not case_answer.HasField("seen_saved_answer")


def test_convert_message_to_case_round_trip_keeps_named_entries_and_suggestions(
    review_cases,
) -> None:
    case = review_cases["gochuumon_movie"].model_copy(
        update={
            "answer": {"anidb:anime:19188": "other work mal:anime:21273"},
            "named_entry_reference_pages": {"mal:anime:21273": "mal:anime:21273"},
            "target_entry_suggestions": (
                TargetEntrySuggestion(
                    entry_reference_page="mal:anime:21273",
                    title="Gochuumon wa Usagi desu ka?",
                    providers=("anidb", "mal"),
                ),
            ),
            "covered_outside_entries": (
                TargetEntrySuggestion(
                    entry_reference_page="anilist:anime:21273",
                    title="Gochuumon wa Usagi desu ka? Season 1",
                    providers=("anilist",),
                ),
            ),
        }
    )
    assert convert_message_to_case(convert_case_to_message(case)) == case
