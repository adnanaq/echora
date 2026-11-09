import json
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
    PreviousAnswer,
    ProviderPage,
    ProviderPageStore,
    ReviewCase,
    ReviewFile,
    ReviewFileLockedError,
    WorkEntry,
)

MAL = "https://myanimelist.net/anime/1"
KITSU_ONE = "https://kitsu.app/anime/1"
KITSU_TWO = "https://kitsu.app/anime/2"
KITSU_THREE = "https://kitsu.app/anime/3"
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
    run_result = _run(seed_document, store, data_dir)
    assert (
        run_result.case_names,
        run_result.correction_counts["review cases open"],
    ) == (
        (DOUBLED_CASE_NAME,),
        1,
    )


def test_run_seed_cleaning_writes_result_file_read_back_unchanged(
    seed_document, store, data_dir
) -> None:
    run_result = _run(seed_document, store, data_dir)
    assert read_seed_cleaning_run_result(data_dir) == run_result


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
    run_result = _run(seed_document, store, data_dir)
    assert run_result.new_case_names == (
        "doubled provider: kitsu:anime:91 kitsu:anime:92 in mal:anime:9",
    )


def test_run_seed_cleaning_keeps_answers_as_written_as_originals(
    seed_document, store, data_dir
) -> None:
    _run(seed_document, store, data_dir)
    answer = {"kitsu:anime:1": "work 1", "kitsu:anime:2": "dropped"}
    ReviewFile(data_dir / "review.jsonl").save_answers({DOUBLED_CASE_NAME: answer})
    run_result = _run(seed_document, store, data_dir)
    assert (
        run_result.original_answers,
        run_result.correction_counts["review cases answered"],
        run_result.correction_counts["duplicate pages dropped"],
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
    ).model_copy(
        update={
            "case_name": "doubled provider: kitsu:anime:5 in mal:anime:5",
            "entries": {
                "mal:anime:5": WorkEntry(
                    title="Gone",
                    type="TV",
                    year=2020,
                    addresses_by_page_key={
                        "mal:anime:5": "https://myanimelist.net/anime/5",
                        "kitsu:anime:5": "https://kitsu.app/anime/5",
                    },
                )
            },
        }
    )
    with (data_dir / "review.jsonl").open("a", encoding="utf-8") as review_file:
        review_file.write(gone_case.model_dump_json() + "\n")
    ReviewFile(data_dir / "review.jsonl").save_answers(
        {gone_case.case_name: {"kitsu:anime:5": "work 1"}}
    )
    assert _run(seed_document, store, data_dir).unused_answer_case_names == (
        gone_case.case_name,
    )


def test_run_seed_cleaning_release_adding_doubled_page_keeps_previous_answer_applied(
    seed_document, store, data_dir
) -> None:
    _run(seed_document, store, data_dir)
    answer = {"kitsu:anime:1": "work 1", "kitsu:anime:2": "dropped"}
    ReviewFile(data_dir / "review.jsonl").save_answers({DOUBLED_CASE_NAME: answer})
    seed_document["data"][0]["sources"].append(KITSU_THREE)
    store.add_pages(_page(KITSU_THREE, MAL))
    run_result = _run(seed_document, store, data_dir)
    [carrying_case] = [
        ReviewCase.model_validate_json(line)
        for line in (data_dir / "review.jsonl").read_text(encoding="utf-8").splitlines()
        if json.loads(line)["case_name"] != DOUBLED_CASE_NAME
    ]
    dropped_page_keys = [
        correction["page_key"]
        for correction in map(
            json.loads,
            (data_dir / "corrections.jsonl").read_text(encoding="utf-8").splitlines(),
        )
        if correction["is_dropped_page"]
    ]
    assert run_result.unused_answer_case_names == ()
    assert run_result.correction_counts["open cases placed by a previous answer"] == 1
    assert carrying_case.previous_answer == PreviousAnswer(
        case_name=DOUBLED_CASE_NAME,
        answer=answer,
        added_page_keys=("kitsu:anime:3",),
        is_applied=True,
    )
    assert dropped_page_keys == ["kitsu:anime:2"]


def test_run_seed_cleaning_dead_pages_dropped_before_any_case(
    seed_document, store, data_dir
) -> None:
    seed_document["data"].append(_seed_entry("Lost", OTHER_MAL))
    store.add_pages(
        ProviderPage.build_dead_page(KITSU_TWO), ProviderPage.build_dead_page(OTHER_MAL)
    )
    run_result = _run(seed_document, store, data_dir)
    dead_page_corrections = [
        (correction["entry_reference_page"], correction["page_key"])
        for correction in map(
            json.loads,
            (data_dir / "corrections.jsonl").read_text(encoding="utf-8").splitlines(),
        )
        if correction["is_dead_page"]
    ]
    assert run_result.case_names == ()
    assert dead_page_corrections == [
        ("mal:anime:1", "kitsu:anime:2"),
        ("mal:anime:9", "mal:anime:9"),
    ]
    assert run_result.correction_counts["dead pages dropped"] == 2
    assert run_result.corrected_entry_counts["entries removed, no page left"] == 1


def test_run_seed_cleaning_dead_reference_page_corrections_name_seed_entry(
    store, data_dir
) -> None:
    seed_document = {
        "$schema": "https://example.com/refs/tags/2026-43/schemas/seed.json",
        "data": [_seed_entry("Other", OTHER_MAL, OTHER_KITSU_ONE, OTHER_KITSU_TWO)],
    }
    store.add_pages(
        ProviderPage.build_dead_page(OTHER_MAL),
        _page(OTHER_KITSU_ONE),
        _page(OTHER_KITSU_TWO),
    )
    [case_name] = _run(seed_document, store, data_dir).case_names
    ReviewFile(data_dir / "review.jsonl").save_answers(
        {case_name: {"kitsu:anime:91": "work 1", "kitsu:anime:92": "work 2"}}
    )
    run_result = _run(seed_document, store, data_dir)
    assert run_result.rejected_answer_case_names == ()
    assert [
        (
            correction["entry_reference_page"],
            correction["page_key"],
            correction["new_entry_key"],
            correction["is_dead_page"],
        )
        for correction in map(
            json.loads,
            (data_dir / "corrections.jsonl").read_text(encoding="utf-8").splitlines(),
        )
    ] == [
        ("mal:anime:9", "kitsu:anime:91", None, False),
        ("mal:anime:9", "kitsu:anime:92", "kitsu:anime:91 work 2", False),
        ("mal:anime:9", "mal:anime:9", None, True),
    ]


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
