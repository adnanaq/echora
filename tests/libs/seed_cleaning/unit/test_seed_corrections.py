import json

from seed_cleaning.seed_corrections import (
    SeedCorrection,
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
