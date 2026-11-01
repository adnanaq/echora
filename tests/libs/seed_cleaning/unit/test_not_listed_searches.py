from seed_cleaning.not_listed_searches import NotListedSearchFile


def test_not_listed_search_file_added_search_kept_for_same_release(tmp_path) -> None:
    path = tmp_path / "not_listed_searches.jsonl"
    NotListedSearchFile(path, "2026-43").add("mal:anime:21")
    assert "mal:anime:21" in NotListedSearchFile(path, "2026-43")


def test_not_listed_search_file_new_release_drops_earlier_release_searches(
    tmp_path,
) -> None:
    path = tmp_path / "not_listed_searches.jsonl"
    NotListedSearchFile(path, "2026-43").add("mal:anime:21")
    assert "mal:anime:21" not in NotListedSearchFile(path, "2026-44")
    assert "2026-43" not in path.read_text(encoding="utf-8")


def test_not_listed_search_file_partial_last_line_ignored(tmp_path) -> None:
    path = tmp_path / "not_listed_searches.jsonl"
    path.write_text(
        '{"release_name": "2026-43", "searched_by": "mal:anime:21"}\n'
        '{"release_name": "2026-43", "searched_by": "mal:an',
        encoding="utf-8",
    )
    not_listed_searches = NotListedSearchFile(path, "2026-43")
    not_listed_searches.add("mal:anime:30")
    assert (
        "mal:anime:21" in not_listed_searches,
        "mal:anime:30" in NotListedSearchFile(path, "2026-43"),
    ) == (True, True)
