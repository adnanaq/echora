import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from clean_seed import build_entries_to_check_document, count_corrections
from seed_cleaning import SeedCorrection, SplitWorkCandidates
from work_matching import WorkEntry


def _entry(title: str, *addresses: str) -> WorkEntry:
    return WorkEntry.from_seed_entry(
        {"title": title, "type": "TV", "sources": list(addresses)}
    )


def test_build_entries_to_check_document_names_entries_by_reference_page() -> None:
    doubled = _entry(
        "Rock Star",
        "https://anilist.co/anime/120516",
        "https://anilist.co/anime/120517",
    )
    group = SplitWorkCandidates(
        entries=(
            _entry("CUE!", "https://animenewsnetwork.com/encyclopedia/anime.php?id=1"),
            _entry("Cue!", "https://myanimelist.net/anime/43735"),
        ),
        shared_provider=None,
    )
    document = build_entries_to_check_document("seed-schema-url", [doubled], [group])
    assert document["seed_schema"] == "seed-schema-url"
    assert document["doubled_provider_entries"] == [
        {
            "reference_page": "anilist:anime:120516",
            "title": "Rock Star",
            "doubled_providers": {
                "anilist": ("anilist:anime:120516", "anilist:anime:120517")
            },
        }
    ]
    [candidates] = document["split_work_candidates"]
    assert candidates["shared_provider"] is None
    assert [entry["reference_page"] for entry in candidates["entries"]] == [
        "ann:anime:1",
        "mal:anime:43735",
    ]


def _correction(page_key: str, placement_word: str, **fields) -> SeedCorrection:
    return SeedCorrection(
        entry_reference_page="mal:anime:1",
        page_key=page_key,
        address=f"https://example.com/{page_key}",
        placement_word=placement_word,
        reason="review answer",
        **fields,
    )


def test_count_corrections_counts_each_change_once() -> None:
    corrections = (
        _correction("kitsu:anime:1", "entry"),
        _correction("kitsu:anime:2", "entry", is_main_page=True),
        _correction("kitsu:anime:3", "entry", renamed_to="kitsu:anime:2"),
        _correction(
            "kitsu:anime:4", "other work", target_entry_reference_page="mal:anime:9"
        ),
        _correction("kitsu:anime:5", "work 2"),
        _correction("mal:anime:2", "entry", is_added_page=True),
        _correction("anidb:anime:2", "entry", is_added_page=True, is_part_page=True),
    )
    assert count_corrections(corrections, ()) == {
        "pages kept as main page": 1,
        "old addresses dropped": 1,
        "pages moved to a seed entry": 1,
        "pages moved to a new entry": 1,
        "pages added": 1,
        "pages added as parts": 1,
        "review cases answered": 0,
        "review cases open": 0,
    }
