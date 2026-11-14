import dataclasses
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest
from google.protobuf import json_format
from review_page.file_review_case_source import (
    FileReviewCaseSource,
    RunSeedFileUnknownError,
    load_seed_entries,
)
from review_page.models import InvalidNamedPageKeyReason
from review_page.review_case_loader import UnreadableReviewFileError
from review_page.review_case_source import (
    ANSWER_SAVED_ELSEWHERE_RULE_NAME,
    SeedFileUnavailableError,
)
from review_page.seed_cleaning_run_process import RUN_LOG_LIMIT_BYTES
from seed_cleaning import (
    REVIEW_ANSWER_REASON,
    SeedCleaningRunResult,
    SeedCorrection,
    describe_correction_reason,
    find_correction_change,
)
from shared_proto.v1 import review_cases_pb2
from work_matching import (
    KnownWorkEntriesFromSeed,
    PlacementRule,
    ReviewCase,
    ReviewFileLockedError,
)

ATASHINCHI_ANSWER = {
    "mal:anime:63013": "work 1",
    "anilist:anime:203146": "work 1",
    "anidb:anime:19895": "work 1",
}


def _case_answer(case_name: str, answer: object) -> review_cases_pb2.CaseAnswer:
    case_answer = review_cases_pb2.CaseAnswer(case_name=case_name)
    if answer is not None:
        json_format.ParseDict(answer, case_answer.answer)
    return case_answer


def _wait_for_run_end(source: FileReviewCaseSource) -> review_cases_pb2.RunState:
    for _ in range(100):
        run_state = source.get_run_state()
        if not run_state.is_running:
            return run_state
        time.sleep(0.05)
    raise AssertionError


@pytest.fixture
def data_dir(tmp_path, review_case_lines) -> Path:
    data_dir = tmp_path / "seed_cleaning"
    data_dir.mkdir()
    (data_dir / "review.jsonl").write_text("".join(review_case_lines), encoding="utf-8")
    return data_dir


@pytest.fixture
def run_script(tmp_path) -> Path:
    script = tmp_path / "clean_seed.py"
    script.write_text(
        "import json, sys, time\n"
        "from pathlib import Path\n"
        "data_dir = Path(sys.argv[sys.argv.index('--data-dir') + 1])\n"
        "(data_dir / 'run_arguments.json').write_text(json.dumps(sys.argv[1:]))\n"
        "time.sleep(0.3)\n"
        "sys.exit(int(data_dir.joinpath('exit_code').read_text())"
        " if data_dir.joinpath('exit_code').exists() else 0)\n",
        encoding="utf-8",
    )
    return script


@pytest.fixture
def source(data_dir, run_script) -> FileReviewCaseSource:
    return FileReviewCaseSource(
        data_dir, run_script, seed_file=Path("assets/seed_data/seed.json")
    )


@pytest.fixture
def seed_source(data_dir, run_script, tmp_path) -> FileReviewCaseSource:
    seed_file = tmp_path / "seed.json"
    seed_file.write_text(
        json.dumps(
            {
                "data": [
                    {
                        "title": "Gochuumon wa Usagi desu ka?",
                        "type": "TV",
                        "sources": ["https://myanimelist.net/anime/21273"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return FileReviewCaseSource(data_dir, run_script, seed_file=seed_file)


def _gochuumon_named_answer(named_entry_page_key: str) -> dict[str, str]:
    return {
        "anidb:anime:19188": f"other work {named_entry_page_key}",
        "anidb:anime:19792": "work 1",
        "mal:anime:61115": "work 1",
    }


def test_file_review_case_source_list_cases_gives_every_case(
    source, review_cases
) -> None:
    assert sorted(case.case_name for case in source.list_cases()) == sorted(
        case.case_name for case in review_cases.values()
    )


def test_file_review_case_source_get_case_gives_named_case(
    source, review_cases
) -> None:
    case_name = review_cases["nami"].case_name
    case = source.get_case(case_name)
    assert case is not None
    assert case.case_name == case_name


def test_file_review_case_source_get_case_unknown_name_gives_none(source) -> None:
    assert source.get_case("split work: none:anime:1") is None


def test_file_review_case_source_save_answers_valid_answer_saved(
    source, review_cases, data_dir
) -> None:
    case_name = review_cases["atashinchi_next"].case_name
    [result] = source.save_answers([_case_answer(case_name, ATASHINCHI_ANSWER)])
    saved = {
        case.case_name: case.answer
        for case in map(
            ReviewCase.model_validate_json,
            (data_dir / "review.jsonl").read_text(encoding="utf-8").splitlines(),
        )
    }
    assert (result.is_saved, saved[case_name]) == (True, ATASHINCHI_ANSWER)


def test_file_review_case_source_save_answers_named_seed_entry_saved(
    seed_source, review_cases
) -> None:
    case_name = review_cases["gochuumon_movie"].case_name
    [result] = seed_source.save_answers(
        [_case_answer(case_name, _gochuumon_named_answer("mal:anime:21273"))]
    )
    assert (result.is_saved, list(result.broken_rule_names)) == (True, [])


def _write_dead_page_correction(data_dir: Path, page_key: str) -> None:
    (data_dir / "corrections.jsonl").write_text(
        SeedCorrection(
            entry_reference_page=page_key,
            page_key=page_key,
            address=f"https://myanimelist.net/anime/{page_key.rsplit(':', 1)[1]}",
            placement_word=None,
            is_dead_page=True,
            reason="dead page",
        ).model_dump_json()
        + "\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("named_entry_page_key", "dead_page_key"),
    [("mal:anime:99999", None), ("mal:anime:21273", "mal:anime:21273")],
)
def test_file_review_case_source_save_answers_named_entry_outside_seed_not_saved(
    seed_source, review_cases, data_dir, named_entry_page_key, dead_page_key
) -> None:
    if dead_page_key:
        _write_dead_page_correction(data_dir, dead_page_key)
    case_name = review_cases["gochuumon_movie"].case_name
    [result] = seed_source.save_answers(
        [_case_answer(case_name, _gochuumon_named_answer(named_entry_page_key))]
    )
    assert (result.is_saved, list(result.broken_rule_names)) == (
        False,
        [PlacementRule.UNKNOWN_NAMED_ENTRY.name],
    )


@pytest.mark.parametrize(
    "seed_file", [None, Path("assets/seed_data/missing_seed.json")]
)
def test_file_review_case_source_save_answers_named_entry_without_seed_file_raises(
    data_dir, run_script, review_cases, review_case_lines, seed_file
) -> None:
    source = FileReviewCaseSource(data_dir, run_script, seed_file=seed_file)
    case_name = review_cases["gochuumon_movie"].case_name
    with pytest.raises(SeedFileUnavailableError):
        source.save_answers(
            [_case_answer(case_name, _gochuumon_named_answer("mal:anime:21273"))]
        )
    assert (data_dir / "review.jsonl").read_text(encoding="utf-8").splitlines(
        keepends=True
    ) == review_case_lines


def test_file_review_case_source_find_invalid_named_page_keys_tells_dead_from_unknown(
    seed_source, data_dir
) -> None:
    assert seed_source.find_invalid_named_page_keys(["mal:anime:21273"]) == {}
    _write_dead_page_correction(data_dir, "mal:anime:21273")
    assert seed_source.find_invalid_named_page_keys(
        ["mal:anime:21273", "mal:anime:99999"]
    ) == {
        "mal:anime:21273": InvalidNamedPageKeyReason.DEAD_PAGE,
        "mal:anime:99999": InvalidNamedPageKeyReason.UNKNOWN_PAGE,
    }


def test_file_review_case_source_find_named_entry_suggestions_gives_seed_entry_of_each_key(
    seed_source,
) -> None:
    named_entry_suggestions = seed_source.find_named_entry_suggestions(
        ["mal:anime:21273", "mal:anime:99999"]
    )
    assert {
        page_key: json_format.MessageToDict(entry_message)
        for page_key, entry_message in named_entry_suggestions.items()
    } == {
        "mal:anime:21273": {
            "entryReferencePage": "mal:anime:21273",
            "title": "Gochuumon wa Usagi desu ka?",
            "providers": ["mal"],
        }
    }


def test_file_review_case_source_save_answers_rule_breaking_answer_not_saved(
    source, review_cases, data_dir, review_case_lines
) -> None:
    case_name = review_cases["atashinchi_next"].case_name
    [result] = source.save_answers(
        [_case_answer(case_name, {"mal:anime:63013": "work 1"})]
    )
    assert (
        result.is_saved,
        list(result.broken_rule_names),
        list(result.broken_rule_sentences),
    ) == (
        False,
        [PlacementRule.MISSING_REQUIRED_PLACEMENT.name],
        [PlacementRule.MISSING_REQUIRED_PLACEMENT.value],
    )
    assert (data_dir / "review.jsonl").read_text(encoding="utf-8") == "".join(
        review_case_lines
    )


def test_file_review_case_source_save_answers_saves_valid_and_reports_rejected(
    source, review_cases
) -> None:
    results = source.save_answers(
        [
            _case_answer(review_cases["atashinchi_next"].case_name, ATASHINCHI_ANSWER),
            _case_answer(review_cases["nami"].case_name, {"anilist:anime:102289": "x"}),
        ]
    )
    assert [result.is_saved for result in results] == [True, False]
    saved_case = source.get_case(review_cases["atashinchi_next"].case_name)
    assert saved_case is not None
    assert saved_case.HasField("answer")


def test_file_review_case_source_save_answers_unset_answer_clears_saved_answer(
    source, review_cases, data_dir, review_case_lines
) -> None:
    case_name = review_cases["atashinchi_next"].case_name
    source.save_answers([_case_answer(case_name, ATASHINCHI_ANSWER)])
    [result] = source.save_answers([_case_answer(case_name, None)])
    assert result.is_saved
    assert (data_dir / "review.jsonl").read_text(encoding="utf-8") == "".join(
        review_case_lines
    )


def test_file_review_case_source_save_answers_saved_elsewhere_rejected_and_kept(
    source, review_cases
) -> None:
    case_name = review_cases["atashinchi_next"].case_name
    other_tab_answer = {**ATASHINCHI_ANSWER, "anidb:anime:19895": "work 2"}
    source.save_answers([_case_answer(case_name, other_tab_answer)])
    case_answer = _case_answer(case_name, ATASHINCHI_ANSWER)
    json_format.ParseDict(None, case_answer.seen_saved_answer)
    [result] = source.save_answers([case_answer])
    saved_case = source.get_case(case_name)
    assert (result.is_saved, list(result.broken_rule_names)) == (
        False,
        [ANSWER_SAVED_ELSEWHERE_RULE_NAME],
    )
    assert saved_case is not None
    assert json_format.MessageToDict(saved_case.answer) == other_tab_answer


def test_file_review_case_source_save_answers_seen_answer_in_other_key_order_saves(
    source, review_cases
) -> None:
    case_name = review_cases["atashinchi_next"].case_name
    source.save_answers([_case_answer(case_name, ATASHINCHI_ANSWER)])
    case_answer = _case_answer(case_name, None)
    json_format.ParseDict(
        dict(reversed(list(ATASHINCHI_ANSWER.items()))), case_answer.seen_saved_answer
    )
    [result] = source.save_answers([case_answer])
    assert result.is_saved


def test_file_review_case_source_save_answers_unknown_case_not_saved(source) -> None:
    [result] = source.save_answers(
        [_case_answer("split work: none:anime:1", {"none:anime:1": "work 1"})]
    )
    assert not result.is_saved
    assert result.broken_rule_sentences


def test_file_review_case_source_save_answers_dropped_case_rejected_others_saved(
    source, review_cases
) -> None:
    case_name = review_cases["atashinchi_next"].case_name
    results = source.save_answers(
        [
            _case_answer("split work: none:anime:1", {"none:anime:1": "work 1"}),
            _case_answer(case_name, ATASHINCHI_ANSWER),
        ]
    )
    saved_case = source.get_case(case_name)
    assert [result.is_saved for result in results] == [False, True]
    assert saved_case is not None and saved_case.HasField("answer")


def test_file_review_case_source_save_answers_while_run_holds_lock_raises(
    source, review_cases, data_dir
) -> None:
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
        assert source.get_run_state().is_running
        with pytest.raises(ReviewFileLockedError):
            source.save_answers(
                [_case_answer(review_cases["atashinchi_next"].case_name, None)]
            )
    finally:
        holder.kill()
        holder.wait()


def test_file_review_case_source_get_last_run_result_no_run_gives_none(
    source,
) -> None:
    assert source.get_last_run_result() is None


def test_file_review_case_source_get_last_run_result_reads_result_file(
    source, data_dir
) -> None:
    result = SeedCleaningRunResult(
        seed_file="assets/seed_data/seed.json",
        started_at=datetime(2026, 11, 1, tzinfo=UTC),
        finished_at=datetime(2026, 11, 1, tzinfo=UTC),
        correction_counts={"review cases open": 4},
        corrected_entry_counts={},
        linked_page_counts={},
        rejected_answer_case_names=(),
        conflicting_linked_page_keys=(),
        unused_answer_case_names=(),
        case_names=(),
        new_case_names=(),
        original_answers={},
    )
    (data_dir / "review_run.json").write_text(
        result.model_dump_json(), encoding="utf-8"
    )
    last_run_result = source.get_last_run_result()
    assert last_run_result is not None
    assert [
        (named_count.name, named_count.count)
        for named_count in last_run_result.correction_counts
    ] == [("review cases open", 4)]


def test_file_review_case_source_start_run_runs_script_without_fetch(
    source, data_dir
) -> None:
    assert source.start_run().is_running
    run_state = _wait_for_run_end(source)
    assert json.loads((data_dir / "run_arguments.json").read_text()) == [
        "assets/seed_data/seed.json",
        "--no-fetch",
        "--data-dir",
        str(data_dir),
    ]
    assert (run_state.is_running, run_state.last_exit_code) == (False, 0)


def test_file_review_case_source_start_run_failed_script_reports_exit_code(
    source, data_dir
) -> None:
    (data_dir / "exit_code").write_text("3")
    source.start_run()
    assert _wait_for_run_end(source).last_exit_code == 3


def test_file_review_case_source_start_run_while_running_starts_nothing_more(
    source, data_dir
) -> None:
    source.start_run()
    for _ in range(100):
        if (data_dir / "run_arguments.json").exists():
            break
        time.sleep(0.02)
    (data_dir / "run_arguments.json").unlink()
    source.start_run()
    _wait_for_run_end(source)
    assert not (data_dir / "run_arguments.json").exists()


def test_file_review_case_source_start_run_requested_at_once_starts_one_run(
    data_dir, tmp_path
) -> None:
    script = tmp_path / "counting_clean_seed.py"
    script.write_text(
        "import sys, time\n"
        "from pathlib import Path\n"
        "data_dir = Path(sys.argv[sys.argv.index('--data-dir') + 1])\n"
        "with (data_dir / 'started_runs.log').open('a') as log:\n"
        "    log.write('started\\n')\n"
        "time.sleep(0.5)\n",
        encoding="utf-8",
    )
    source = FileReviewCaseSource(
        data_dir, script, seed_file=Path("assets/seed_data/seed.json")
    )
    barrier = threading.Barrier(8)

    def start_run_after_barrier() -> None:
        barrier.wait()
        source.start_run()

    threads = [threading.Thread(target=start_run_after_barrier) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    _wait_for_run_end(source)
    started_runs = (data_dir / "started_runs.log").read_text().splitlines()
    assert started_runs == ["started"]


def test_file_review_case_source_get_run_state_asked_at_once_after_failed_run_reports_exit_code(
    source, data_dir
) -> None:
    (data_dir / "exit_code").write_text("3")
    source.start_run()
    time.sleep(1)
    barrier = threading.Barrier(8)
    run_states = []

    def read_run_state_after_barrier() -> None:
        barrier.wait()
        run_states.append(source.get_run_state())

    threads = [threading.Thread(target=read_run_state_after_barrier) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert [
        (run_state.is_running, run_state.last_exit_code) for run_state in run_states
    ] == [(False, 3)] * 8


def test_file_review_case_source_get_run_state_process_id_not_started_here_not_running(
    source, data_dir
) -> None:
    (data_dir / "review_run.pid").write_text(str(os.getpid()), encoding="utf-8")
    assert not source.get_run_state().is_running


def test_file_review_case_source_start_run_without_known_seed_file_raises(
    data_dir, run_script
) -> None:
    source = FileReviewCaseSource(data_dir, run_script)
    with pytest.raises(RunSeedFileUnknownError):
        source.start_run()


def test_file_review_case_source_start_run_uses_seed_file_of_last_run(
    data_dir, run_script
) -> None:
    (data_dir / "review_run.json").write_text(
        SeedCleaningRunResult(
            seed_file="assets/seed_data/last.json",
            started_at=datetime(2026, 11, 1, tzinfo=UTC),
            finished_at=datetime(2026, 11, 1, tzinfo=UTC),
            correction_counts={},
            corrected_entry_counts={},
            linked_page_counts={},
            rejected_answer_case_names=(),
            conflicting_linked_page_keys=(),
            unused_answer_case_names=(),
            case_names=(),
            new_case_names=(),
            original_answers={},
        ).model_dump_json(),
        encoding="utf-8",
    )
    source = FileReviewCaseSource(data_dir, run_script)
    source.start_run()
    _wait_for_run_end(source)
    assert json.loads((data_dir / "run_arguments.json").read_text())[0] == (
        "assets/seed_data/last.json"
    )


def test_file_review_case_source_save_answers_conflicting_linked_pages_warns(
    source, review_cases, data_dir
) -> None:
    nami = review_cases["nami"]
    [decision] = nami.decisions
    second_case = nami.model_copy(
        update={
            "case_name": "linked page: anilist:anime:5 from kitsu:anime:10561",
            "decisions": (dataclasses.replace(decision, page_key="anilist:anime:5"),),
        }
    )
    with (data_dir / "review.jsonl").open("a", encoding="utf-8") as review_file:
        review_file.write(second_case.model_dump_json() + "\n")
    first_result, second_result = source.save_answers(
        [
            _case_answer(nami.case_name, {"anilist:anime:102289": "add"}),
            _case_answer(second_case.case_name, {"anilist:anime:5": "add"}),
        ]
    )
    assert (first_result.is_saved, second_result.is_saved) == (True, True)
    assert list(first_result.warnings) == [
        "another answer also adds a whole anilist page to kitsu:anime:10561, so "
        f"the run adds neither: {second_case.case_name}"
    ]


def test_file_review_case_source_list_corrections_keeps_every_line(
    source, data_dir, correction_lines
) -> None:
    correction_lines = [
        *correction_lines,
        json.dumps(
            {
                "entry_reference_page": "mal:anime:36040",
                "page_key": "mal:anime:36040",
                "address": "https://myanimelist.net/anime/36040",
                "placement_word": "work 1",
                "partial_entry_reference_pages": ["anilist:anime:114448"],
                "partial_new_entry_keys": ["mal:anime:36040 work 3"],
                "reason": "REVIEW_ANSWER",
            }
        )
        + "\n",
        json.dumps(
            {
                "entry_reference_page": "kitsu:anime:46351",
                "page_key": "kitsu:anime:50917",
                "address": "https://kitsu.app/anime/50917",
                "placement_word": "dropped",
                "is_dropped_page": True,
                "reason": "REVIEW_ANSWER",
            }
        )
        + "\n",
    ]
    (data_dir / "corrections.jsonl").write_text(
        "".join(correction_lines), encoding="utf-8"
    )
    corrections = source.list_corrections()
    raw_fields = [
        {
            field: value
            for field, value in json_format.MessageToDict(
                correction, preserving_proto_field_name=True
            ).items()
            if field not in ("change", "reason_sentence", "is_from_answer")
        }
        for correction in corrections
    ]
    assert raw_fields == [
        {
            field: value
            for field, value in json.loads(line).items()
            if value not in (None, False)
        }
        for line in correction_lines
    ]


def test_file_review_case_source_list_corrections_line_separator_in_value_keeps_line_whole(
    source, data_dir, correction_lines
) -> None:
    first_fields = json.loads(correction_lines[0])
    first_fields["address"] += "\u2028\u0085"
    (data_dir / "corrections.jsonl").write_text(
        json.dumps(first_fields, ensure_ascii=False)
        + "\n"
        + "".join(correction_lines[1:]),
        encoding="utf-8",
    )
    corrections = source.list_corrections()
    assert (len(corrections), corrections[0].address) == (
        len(correction_lines),
        first_fields["address"],
    )


def test_file_review_case_source_list_corrections_fills_change_and_reason(
    source, data_dir, correction_lines
) -> None:
    (data_dir / "corrections.jsonl").write_text(
        "".join(correction_lines), encoding="utf-8"
    )
    expected = [
        (
            find_correction_change(correction),
            describe_correction_reason(correction),
            correction.reason == REVIEW_ANSWER_REASON,
        )
        for correction in map(SeedCorrection.model_validate_json, correction_lines)
    ]
    assert [
        (
            correction.change if correction.HasField("change") else None,
            correction.reason_sentence,
            correction.is_from_answer,
        )
        for correction in source.list_corrections()
    ] == expected


def test_file_review_case_source_list_corrections_before_any_run_gives_none(
    source,
) -> None:
    assert source.list_corrections() == []


def test_file_review_case_source_list_checked_entries_reads_both_passes(
    source, data_dir, entries_to_check_text
) -> None:
    (data_dir / "entries_to_check.json").write_text(
        entries_to_check_text, encoding="utf-8"
    )
    document = json.loads(entries_to_check_text)
    split_entries = [
        entry
        for group in document["split_work_candidates"]
        for entry in group["entries"]
    ]
    entries = {
        entry.entry_reference_page: entry for entry in source.list_checked_entries()
    }
    first_split_entry = split_entries[0]
    first_split_message = entries[first_split_entry["reference_page"]]
    assert (
        set(entries)
        == {entry["reference_page"] for entry in document["doubled_provider_entries"]}
        | {entry["reference_page"] for entry in split_entries},
        (
            first_split_message.title,
            first_split_message.type if first_split_message.HasField("type") else None,
            first_split_message.year if first_split_message.HasField("year") else None,
        ),
    ) == (
        True,
        (
            first_split_entry["title"],
            first_split_entry["type"],
            first_split_entry["year"],
        ),
    )


def test_file_review_case_source_list_checked_entries_before_any_run_gives_none(
    source,
) -> None:
    assert source.list_checked_entries() == []


def test_file_review_case_source_list_cases_broken_line_names_line(
    source, data_dir, review_case_lines
) -> None:
    lines = list(review_case_lines)
    lines[1] = '{"case_name": "hand edited", "answer": \n'
    (data_dir / "review.jsonl").write_text("".join(lines), encoding="utf-8")
    with pytest.raises(UnreadableReviewFileError, match="Line 2 of"):
        source.list_cases()


def test_file_review_case_source_start_run_full_run_log_keeps_previous_log(
    source, data_dir
) -> None:
    run_log = data_dir / "logs" / "review_page_runs.log"
    run_log.parent.mkdir()
    run_log.write_bytes(b"x" * (RUN_LOG_LIMIT_BYTES + 1))
    source.start_run()
    _wait_for_run_end(source)
    previous_log = run_log.with_name("review_page_runs.log.1")
    assert (
        previous_log.stat().st_size,
        run_log.stat().st_size < RUN_LOG_LIMIT_BYTES,
    ) == (
        RUN_LOG_LIMIT_BYTES + 1,
        True,
    )


def _write_seed(seed_file: Path, page_number: int) -> None:
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
