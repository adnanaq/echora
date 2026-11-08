import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from review_page.seed_entries import load_seed_entries
from work_matching import KnownWorkEntriesFromSeed


def _write_seed(seed_file, page_number: int) -> None:
    seed_file.write_text(
        json.dumps(
            {
                "data": [
                    {
                        "title": "Gochuumon wa Usagi desu ka?",
                        "type": "TV",
                        "sources": [f"https://myanimelist.net/anime/{page_number}"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_load_seed_entries_unchanged_file_loaded_once(tmp_path) -> None:
    seed_file = tmp_path / "seed.json"
    _write_seed(seed_file, 21273)
    assert load_seed_entries(seed_file) is load_seed_entries(seed_file)


def test_load_seed_entries_requested_at_once_loaded_once(tmp_path) -> None:
    seed_file = tmp_path / "seed.json"
    _write_seed(seed_file, 21273)
    load_real_seed_file = KnownWorkEntriesFromSeed.load_seed_file

    def load_seed_file_slowly(path):
        time.sleep(0.2)
        return load_real_seed_file(path)

    with (
        patch.object(
            KnownWorkEntriesFromSeed,
            "load_seed_file",
            autospec=True,
            side_effect=load_seed_file_slowly,
        ) as load_seed_file,
        ThreadPoolExecutor(max_workers=4) as executor,
    ):
        loaded_entries = list(executor.map(load_seed_entries, [seed_file] * 4))
    assert (
        load_seed_file.call_count,
        all(entries is loaded_entries[0] for entries in loaded_entries),
    ) == (1, True)


def test_load_seed_entries_changed_file_loaded_again(tmp_path) -> None:
    seed_file = tmp_path / "seed.json"
    _write_seed(seed_file, 21273)
    first_entries = load_seed_entries(seed_file)
    _write_seed(seed_file, 29787)
    modified_at_ns = seed_file.stat().st_mtime_ns + 1_000_000_000
    os.utime(seed_file, ns=(modified_at_ns, modified_at_ns))
    second_entries = load_seed_entries(seed_file)
    assert (
        first_entries.find_by_page_key("mal:anime:21273") is not None,
        second_entries.find_by_page_key("mal:anime:21273"),
        second_entries.find_by_page_key("mal:anime:29787") is not None,
    ) == (True, None, True)
