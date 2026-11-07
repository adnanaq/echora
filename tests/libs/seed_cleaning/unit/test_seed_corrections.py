import json

import pytest
from seed_cleaning.seed_corrections import (
    SeedCorrection,
    follow_named_entry_pages,
    merge_corrections,
    write_corrections,
)


def _correction(entry_reference_page: str, page_key: str) -> SeedCorrection:
    return SeedCorrection(
        entry_reference_page=entry_reference_page,
        page_key=page_key,
        address=f"https://example.com/{page_key}",
        placement_word="work 1",
        reason="review answer",
    )


def test_write_corrections_same_corrections_give_same_file(tmp_path) -> None:
    corrections = [
        _correction("mal:anime:9", "kitsu:anime:1"),
        _correction("mal:anime:1", "kitsu:anime:2"),
        _correction("mal:anime:1", "kitsu:anime:1"),
    ]
    path = tmp_path / "corrections.jsonl"
    write_corrections(path, reversed(corrections))
    first_write = path.read_text(encoding="utf-8")
    write_corrections(path, corrections)
    assert path.read_text(encoding="utf-8") == first_write
    assert [
        (correction["entry_reference_page"], correction["page_key"])
        for correction in map(json.loads, first_write.splitlines())
    ] == [
        ("mal:anime:1", "kitsu:anime:1"),
        ("mal:anime:1", "kitsu:anime:2"),
        ("mal:anime:9", "kitsu:anime:1"),
    ]


def test_merge_corrections_second_pass_replaces_same_entry_page() -> None:
    first_pass = [
        _correction("mal:anime:1", "kitsu:anime:1"),
        _correction("mal:anime:1", "kitsu:anime:2"),
    ]
    second_pass = [
        _correction("mal:anime:1", "kitsu:anime:2").model_copy(
            update={"target_entry_reference_page": "mal:anime:9"}
        ),
        _correction("mal:anime:9", "kitsu:anime:9"),
    ]
    assert merge_corrections(first_pass, second_pass) == [
        first_pass[0],
        second_pass[0],
        second_pass[1],
    ]


def _named_move(page_key: str, named_page_key: str) -> SeedCorrection:
    return _correction("mal:anime:1", page_key).model_copy(
        update={
            "placement_word": "other work",
            "target_entry_reference_page": "mal:anime:9",
            "named_entry_page_key": named_page_key,
        }
    )


@pytest.mark.parametrize(
    ("named_page_move", "page_destination"),
    [
        ({"new_entry_key": "mal:anime:9 work 2"}, (None, "mal:anime:9 work 2")),
        ({"target_entry_reference_page": "mal:anime:5"}, ("mal:anime:5", None)),
        ({}, ("mal:anime:9", None)),
    ],
)
def test_follow_named_entry_pages_page_ends_up_with_named_page(
    named_page_move, page_destination
) -> None:
    named_page = _correction("mal:anime:9", "mal:anime:9").model_copy(
        update=named_page_move
    )
    followed = follow_named_entry_pages(
        [_named_move("kitsu:anime:2", "mal:anime:9"), named_page]
    )
    assert (
        followed[0].target_entry_reference_page,
        followed[0].new_entry_key,
    ) == page_destination
    assert followed[1] == named_page


def test_follow_named_entry_pages_named_page_sent_to_named_page_followed_in_turn() -> (
    None
):
    named_page = _correction("mal:anime:9", "mal:anime:9").model_copy(
        update={
            "placement_word": "other work",
            "target_entry_reference_page": "mal:anime:5",
            "named_entry_page_key": "mal:anime:5",
        }
    )
    final_page = _correction("mal:anime:5", "mal:anime:5").model_copy(
        update={"new_entry_key": "mal:anime:5 work 2"}
    )
    followed = follow_named_entry_pages(
        [_named_move("kitsu:anime:2", "mal:anime:9"), named_page, final_page]
    )
    assert [
        (correction.target_entry_reference_page, correction.new_entry_key)
        for correction in followed
    ] == [
        (None, "mal:anime:5 work 2"),
        (None, "mal:anime:5 work 2"),
        (None, "mal:anime:5 work 2"),
    ]
