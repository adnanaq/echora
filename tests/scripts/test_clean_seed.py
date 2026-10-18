import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from clean_seed import entries_to_check_document
from seed_cleaning import SplitWorkCandidates
from work_matching import WorkEntry


def _entry(title: str, *addresses: str) -> WorkEntry:
    return WorkEntry.from_seed_entry(
        {"title": title, "type": "TV", "sources": list(addresses)}
    )


def test_entries_to_check_document_names_entries_by_reference_page() -> None:
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
    document = entries_to_check_document("2026-40", [doubled], [group])
    assert document["release"] == "2026-40"
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
