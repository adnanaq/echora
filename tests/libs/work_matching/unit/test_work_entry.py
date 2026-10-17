import pytest
from pydantic import ValidationError
from work_matching.work_entry import WorkEntry


def _entry_titled(seed_entries: list[dict], title: str) -> dict:
    return next(entry for entry in seed_entries if entry["title"] == title)


def test_work_entry_from_seed_entry_reads_matching_fields(seed_entries) -> None:
    work = WorkEntry.from_seed_entry(_entry_titled(seed_entries, "One Piece"))
    assert (work.title, work.type, work.year, work.season) == (
        "One Piece",
        "TV",
        1999,
        "FALL",
    )


def test_work_entry_from_seed_entry_keys_pages_by_provider_page(seed_entries) -> None:
    work = WorkEntry.from_seed_entry(_entry_titled(seed_entries, "One Piece"))
    assert work.pages["mal:anime:21"] == "https://myanimelist.net/anime/21"
    assert work.pages["kitsu:anime:12"] == "https://kitsu.app/anime/12"
    assert work.pages["ann:anime:836"] == (
        "https://animenewsnetwork.com/encyclopedia/anime.php?id=836"
    )
    assert len(work.pages) == 10


def test_work_entry_from_seed_entry_undefined_season_and_missing_year_give_none(
    seed_entries,
) -> None:
    work = WorkEntry.from_seed_entry(
        _entry_titled(seed_entries, "2.5-jigen no Ririsa 2nd Season")
    )
    assert (work.year, work.season) == (None, None)


def test_work_entry_reference_page_follows_reference_order() -> None:
    work = WorkEntry(
        title="Test",
        type="TV",
        pages={
            "kitsu:anime:1": "https://kitsu.app/anime/1",
            "anilist:anime:2": "https://anilist.co/anime/2",
        },
    )
    assert work.reference_page == "anilist:anime:2"


def test_work_entry_reference_page_without_known_provider_takes_first_page() -> None:
    work = WorkEntry(
        title="Test",
        type="TV",
        pages={"https://example.com/a": "https://example.com/a"},
    )
    assert work.reference_page == "https://example.com/a"


def test_work_entry_unknown_type_rejected() -> None:
    with pytest.raises(ValidationError):
        WorkEntry.model_validate(
            {"title": "Test", "type": "MUSIC", "pages": {"mal:anime:1": "x"}}
        )


def test_work_entry_without_pages_rejected() -> None:
    with pytest.raises(ValidationError):
        WorkEntry(title="Test", type="TV", pages={})
