import subprocess
import sys
from pathlib import Path

import pytest
from seed_cleaning.seed_cleaning_run import (
    read_seed_cleaning_run_result,
    run_seed_cleaning,
)
from work_matching import (
    KnownWorkEntriesFromSeed,
    ProviderPage,
    ProviderPageStore,
    ReviewCase,
    ReviewFile,
    ReviewFileLockedError,
)

MAL = "https://myanimelist.net/anime/1"
KITSU_ONE = "https://kitsu.app/anime/1"
KITSU_TWO = "https://kitsu.app/anime/2"
OTHER_MAL = "https://myanimelist.net/anime/9"
OTHER_KITSU_ONE = "https://kitsu.app/anime/91"
OTHER_KITSU_TWO = "https://kitsu.app/anime/92"
DOUBLED_CASE_NAME = "doubled provider: kitsu:anime:1 kitsu:anime:2 in mal:anime:1"


def _seed_entry(title: str, *addresses: str) -> dict:
    return {
        "title": title,
        "type": "TV",
        "sources": list(addresses),
        "animeSeason": {"season": "SPRING", "year": 2020},
    }


def _page(address: str, *linked_addresses: str) -> ProviderPage:
    return ProviderPage.read_canonical_record(
        address,
        {
            "title": "Show",
            "external_sources": [
                {"source": linked_address} for linked_address in linked_addresses
            ],
        },
    )


@pytest.fixture
def seed_document() -> dict:
    return {
        "$schema": "https://example.com/refs/tags/2026-43/schemas/seed.json",
        "data": [_seed_entry("Show", MAL, KITSU_ONE, KITSU_TWO)],
    }


@pytest.fixture
def store(tmp_path) -> ProviderPageStore:
    store = ProviderPageStore(tmp_path / "provider_pages.jsonl")
    store.add_pages(_page(KITSU_ONE, MAL), _page(KITSU_TWO, MAL))
    return store


@pytest.fixture
def data_dir(tmp_path) -> Path:
    return tmp_path / "seed_cleaning"


def _run(seed_document, store, data_dir, **options):
    return run_seed_cleaning(
        seed_document,
        Path("assets/seed_data/seed.json"),
        KnownWorkEntriesFromSeed(seed_document["data"]),
        store,
        data_dir,
        **options,
    )


def test_run_seed_cleaning_lists_cases_and_open_count(
    seed_document, store, data_dir
) -> None:
    result = _run(seed_document, store, data_dir)
    assert (result.case_names, result.correction_counts["review cases open"]) == (
        (DOUBLED_CASE_NAME,),
        1,
    )


def test_run_seed_cleaning_writes_result_file_read_back_unchanged(
    seed_document, store, data_dir
) -> None:
    result = _run(seed_document, store, data_dir)
    assert read_seed_cleaning_run_result(data_dir) == result


def test_run_seed_cleaning_keeps_seed_file(seed_document, store, data_dir) -> None:
    assert (
        _run(seed_document, store, data_dir).seed_file == "assets/seed_data/seed.json"
    )


def test_run_seed_cleaning_first_run_marks_no_case_new(
    seed_document, store, data_dir
) -> None:
    assert _run(seed_document, store, data_dir).new_case_names == ()


def test_run_seed_cleaning_case_missing_from_previous_run_marked_new(
    seed_document, store, data_dir
) -> None:
    _run(seed_document, store, data_dir)
    seed_document["data"].append(
        _seed_entry("Other", OTHER_MAL, OTHER_KITSU_ONE, OTHER_KITSU_TWO)
    )
    store.add_pages(
        _page(OTHER_KITSU_ONE, OTHER_MAL), _page(OTHER_KITSU_TWO, OTHER_MAL)
    )
    result = _run(seed_document, store, data_dir)
    assert result.new_case_names == (
        "doubled provider: kitsu:anime:91 kitsu:anime:92 in mal:anime:9",
    )


def test_run_seed_cleaning_keeps_answers_as_written_as_originals(
    seed_document, store, data_dir
) -> None:
    _run(seed_document, store, data_dir)
    answer = {"kitsu:anime:1": "work 1", "kitsu:anime:2": "work 1, main"}
    ReviewFile(data_dir / "review.jsonl").save_answers({DOUBLED_CASE_NAME: answer})
    result = _run(seed_document, store, data_dir)
    assert (
        result.original_answers,
        result.correction_counts["review cases answered"],
        result.correction_counts["pages kept as main page"],
    ) == ({DOUBLED_CASE_NAME: answer}, 1, 1)


def test_run_seed_cleaning_lists_answer_rules_refuse(
    seed_document, store, data_dir
) -> None:
    _run(seed_document, store, data_dir)
    ReviewFile(data_dir / "review.jsonl").save_answers(
        {DOUBLED_CASE_NAME: {"kitsu:anime:1": "work 1"}}
    )
    assert _run(seed_document, store, data_dir).rejected_answer_case_names == (
        DOUBLED_CASE_NAME,
    )


def test_run_seed_cleaning_lists_answers_whose_case_is_gone(
    seed_document, store, data_dir
) -> None:
    _run(seed_document, store, data_dir)
    gone_case = ReviewCase.model_validate_json(
        (data_dir / "review.jsonl").read_text(encoding="utf-8")
    ).model_copy(update={"case_name": "doubled provider: kitsu:anime:5 in mal:anime:5"})
    with (data_dir / "review.jsonl").open("a", encoding="utf-8") as review_file:
        review_file.write(gone_case.model_dump_json() + "\n")
    ReviewFile(data_dir / "review.jsonl").save_answers(
        {gone_case.case_name: {"kitsu:anime:5": "work 1"}}
    )
    assert _run(seed_document, store, data_dir).unused_answer_case_names == (
        gone_case.case_name,
    )


def test_run_seed_cleaning_review_file_locked_by_other_process_raises(
    seed_document, store, data_dir
) -> None:
    data_dir.mkdir(parents=True)
    holder = subprocess.Popen(  # noqa: S603  (fixed test arguments)
        [
            sys.executable,
            "-c",
            "import sys, time, filelock\n"
            "with filelock.FileLock(sys.argv[1]):\n"
            "    print('held', flush=True); time.sleep(5)",
            str(data_dir / "review.jsonl.lock"),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "held"
        with pytest.raises(ReviewFileLockedError):
            _run(seed_document, store, data_dir, seconds_to_wait_for_lock=0)
    finally:
        holder.kill()
        holder.wait()


def test_read_seed_cleaning_run_result_no_run_yet_gives_none(data_dir) -> None:
    assert read_seed_cleaning_run_result(data_dir) is None
