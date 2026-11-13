import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError
from work_matching import review_file
from work_matching.doubled_provider_check import decide_doubled_providers
from work_matching.errors import (
    DuplicateCaseError,
    ReviewFileLockedError,
    SavedAnswerChangedError,
    UnknownCaseError,
)
from work_matching.known_work_entries_from_seed import KnownWorkEntriesFromSeed
from work_matching.models import (
    LinkedPageAddition,
    PreviousAnswer,
)
from work_matching.page_placement import PagePlacement
from work_matching.review_cases import ReviewCase
from work_matching.review_file import (
    ReviewFile,
    hold_review_file_lock,
    is_review_file_locked,
    replace_file_lines,
)

SIGEWINNE = "animeplanet:anime:genshin-impact-character-teaser-sigewinne-everyday-correspondence"
VARESA = (
    "animeplanet:anime:genshin-impact-character-teaser-varesa-the-legend-of-a-warrior"
)
INVALID_ANSWERS = [
    {SIGEWINNE: "maybe"},
    {SIGEWINNE: "work 1, main", VARESA: "work 1, part"},
]


def test_review_file_attach_kept_answer_linked_page_answer_comes_back(
    tmp_path, linked_page_case
) -> None:
    path = tmp_path / "review.jsonl"
    answered_case = linked_page_case.model_copy(
        update={"answer": {"mal:anime:65144": "add"}}
    )
    ReviewFile(path).write_cases([answered_case])
    kept_case = ReviewFile(path).attach_kept_answer(linked_page_case)
    assert kept_case.accepted_answer == LinkedPageAddition(
        "anilist:anime:163794", is_partial_source=False
    )


def test_review_file_missing_file_keeps_no_answers(tmp_path, doubled_case) -> None:
    review = ReviewFile(tmp_path / "review.jsonl")
    assert review.attach_kept_answer(doubled_case) == doubled_case


def test_review_file_write_cases_puts_case_name_and_answer_first(
    tmp_path, doubled_case
) -> None:
    path = tmp_path / "review.jsonl"
    ReviewFile(path).write_cases([doubled_case])
    [line] = path.read_text(encoding="utf-8").splitlines()
    assert list(json.loads(line))[:3] == ["case_name", "answer", "suggested_answer"]


def test_review_file_attach_kept_answer_answer_written_in_file_comes_back(
    tmp_path, doubled_case
) -> None:
    path = tmp_path / "review.jsonl"
    ReviewFile(path).write_cases([doubled_case])
    answer = {SIGEWINNE: "work 1", VARESA: "other work"}
    _write_answer_in_file(path, doubled_case.case_name, answer)
    kept_case = ReviewFile(path).attach_kept_answer(doubled_case)
    assert kept_case.accepted_answer == {
        SIGEWINNE: PagePlacement("work 1"),
        VARESA: PagePlacement("other work"),
    }


def test_review_file_attach_kept_answer_provider_gaining_page_reopens_case(
    tmp_path, matching_cases, doubled_case
) -> None:
    path = tmp_path / "review.jsonl"
    ReviewFile(path).write_cases([doubled_case])
    _write_answer_in_file(
        path, doubled_case.case_name, {SIGEWINNE: "work 1", VARESA: "work 1"}
    )
    entries, pages = matching_cases["genshin_sigewinne"]
    third_page = "https://anime-planet.com/anime/genshin-impact-third-teaser"
    entry = entries[0].model_copy(
        update={
            "addresses_by_page_key": {
                **entries[0].addresses_by_page_key,
                "animeplanet:anime:genshin-impact-third-teaser": third_page,
            }
        }
    )
    decisions = decide_doubled_providers(entry, pages, KnownWorkEntriesFromSeed([]))
    later_case = ReviewCase.build_entry_group_case([entry], decisions, (), pages)
    assert later_case.case_name != doubled_case.case_name
    assert ReviewFile(path).attach_kept_answer(later_case).answer is None


def test_review_file_attach_kept_answer_doubled_pages_in_other_entry_reopens_case(
    tmp_path, matching_cases, doubled_case
) -> None:
    path = tmp_path / "review.jsonl"
    ReviewFile(path).write_cases([doubled_case])
    _write_answer_in_file(
        path, doubled_case.case_name, {SIGEWINNE: "work 1", VARESA: "work 1"}
    )
    entries, pages = matching_cases["genshin_sigewinne"]
    other_entry = entries[0].model_copy(
        update={
            "addresses_by_page_key": {"mal:anime:1": "https://myanimelist.net/anime/1"}
            | entries[0].addresses_by_page_key
        }
    )
    decisions = decide_doubled_providers(
        other_entry, pages, KnownWorkEntriesFromSeed([])
    )
    moved_case = ReviewCase.build_entry_group_case([other_entry], decisions, (), pages)
    assert moved_case.case_name != doubled_case.case_name
    assert ReviewFile(path).attach_kept_answer(moved_case).answer is None


def test_review_file_write_cases_answer_without_case_kept_and_reported(
    tmp_path, doubled_case, split_case
) -> None:
    path = tmp_path / "review.jsonl"
    ReviewFile(path).write_cases([doubled_case, split_case])
    _write_answer_in_file(
        path, doubled_case.case_name, {SIGEWINNE: "work 1", VARESA: "work 1"}
    )
    unused_case_names = ReviewFile(path).write_cases([split_case])
    cases = [json.loads(line)["case_name"] for line in path.read_text().splitlines()]
    assert (unused_case_names, cases) == (
        (doubled_case.case_name,),
        [split_case.case_name, doubled_case.case_name],
    )


def test_review_file_list_kept_cases_returns_answered_cases(
    tmp_path, doubled_case, split_case
) -> None:
    path = tmp_path / "review.jsonl"
    ReviewFile(path).write_cases([doubled_case, split_case])
    _write_answer_in_file(path, doubled_case.case_name, {SIGEWINNE: "work 1"})
    [kept_case] = ReviewFile(path).list_kept_cases()
    assert (kept_case.case_name, kept_case.entries) == (
        doubled_case.case_name,
        doubled_case.entries,
    )


def test_review_file_write_cases_previous_answer_kept_and_not_reported(
    tmp_path, doubled_case, split_case
) -> None:
    path = tmp_path / "review.jsonl"
    ReviewFile(path).write_cases([doubled_case])
    answer = {SIGEWINNE: "work 1", VARESA: "work 1"}
    _write_answer_in_file(path, doubled_case.case_name, answer)
    carrying_case = split_case.model_copy(
        update={
            "previous_answer": PreviousAnswer(
                case_name=doubled_case.case_name, answer=answer, is_applied=True
            )
        }
    )
    assert ReviewFile(path).write_cases([carrying_case]) == ()
    assert [
        json.loads(line)["case_name"] for line in path.read_text().splitlines()
    ] == [split_case.case_name, doubled_case.case_name]


def test_review_file_write_cases_previous_answer_dropped_once_case_answered(
    tmp_path, doubled_case, split_case
) -> None:
    path = tmp_path / "review.jsonl"
    ReviewFile(path).write_cases([doubled_case])
    answer = {SIGEWINNE: "work 1", VARESA: "work 1"}
    _write_answer_in_file(path, doubled_case.case_name, answer)
    answered_case = split_case.model_copy(
        update={
            "answer": {"mal:anime:1": "work 1", "anilist:anime:1": "work 2"},
            "previous_answer": PreviousAnswer(
                case_name=doubled_case.case_name, answer=answer, is_applied=True
            ),
        }
    )
    assert ReviewFile(path).write_cases([answered_case]) == ()
    assert [
        json.loads(line)["case_name"] for line in path.read_text().splitlines()
    ] == [split_case.case_name]


def test_review_file_write_cases_invalid_answer_without_case_kept_unchanged(
    tmp_path, doubled_case, split_case
) -> None:
    path = tmp_path / "review.jsonl"
    ReviewFile(path).write_cases([doubled_case, split_case])
    _write_answer_in_file(path, doubled_case.case_name, "maybe")
    answered_line = next(
        line for line in path.read_text().splitlines(keepends=True) if "maybe" in line
    )
    ReviewFile(path).write_cases([split_case])
    assert path.read_text().splitlines(keepends=True)[-1] == answered_line


def test_review_file_write_cases_open_case_without_case_dropped(
    tmp_path, doubled_case, split_case
) -> None:
    path = tmp_path / "review.jsonl"
    ReviewFile(path).write_cases([doubled_case, split_case])
    assert ReviewFile(path).write_cases([split_case]) == ()
    assert [
        json.loads(line)["case_name"] for line in path.read_text().splitlines()
    ] == [split_case.case_name]


@pytest.mark.parametrize("invalid_answer", INVALID_ANSWERS)
def test_review_file_write_cases_invalid_answer_kept_for_fixing(
    tmp_path, doubled_case, invalid_answer
) -> None:
    path = tmp_path / "review.jsonl"
    ReviewFile(path).write_cases([doubled_case])
    _write_answer_in_file(path, doubled_case.case_name, invalid_answer)
    review = ReviewFile(path)
    kept_case = review.attach_kept_answer(doubled_case)
    review.write_cases([kept_case])
    assert kept_case.accepted_answer is None
    assert json.loads(path.read_text())["answer"] == invalid_answer


def test_review_file_write_cases_leaves_only_review_file(
    tmp_path, doubled_case
) -> None:
    path = tmp_path / "cleaning" / "review.jsonl"
    ReviewFile(path).write_cases([doubled_case])
    ReviewFile(path).write_cases([doubled_case])
    assert list(path.parent.iterdir()) == [path]


def test_review_file_write_cases_failed_write_keeps_old_file(
    tmp_path, doubled_case, split_case
) -> None:
    path = tmp_path / "review.jsonl"
    ReviewFile(path).write_cases([doubled_case])
    written_text = path.read_text(encoding="utf-8")
    with (
        patch.object(review_file.os, "fsync", autospec=True, side_effect=OSError),
        pytest.raises(OSError),
    ):
        ReviewFile(path).write_cases([split_case])
    assert (path.read_text(encoding="utf-8"), list(tmp_path.iterdir())) == (
        written_text,
        [path],
    )


def test_review_file_blank_line_answers_read_and_line_not_written(
    tmp_path, doubled_case
) -> None:
    path = tmp_path / "review.jsonl"
    ReviewFile(path).write_cases([doubled_case])
    _write_answer_in_file(
        path, doubled_case.case_name, {SIGEWINNE: "work 1", VARESA: "work 2"}
    )
    path.write_text("\n" + path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    review = ReviewFile(path)
    kept_case = review.attach_kept_answer(doubled_case)
    review.write_cases([kept_case])
    assert kept_case.answer == {SIGEWINNE: "work 1", VARESA: "work 2"}
    assert "" not in path.read_text(encoding="utf-8").splitlines()


def test_review_file_write_cases_same_case_twice_raises(tmp_path, doubled_case) -> None:
    with pytest.raises(DuplicateCaseError):
        ReviewFile(tmp_path / "review.jsonl").write_cases([doubled_case, doubled_case])


def test_review_file_write_cases_answer_saved_after_opening_kept(
    tmp_path, doubled_case
) -> None:
    path = tmp_path / "review.jsonl"
    ReviewFile(path).write_cases([doubled_case])
    review = ReviewFile(path)
    _write_answer_in_file(
        path, doubled_case.case_name, {SIGEWINNE: "work 1", VARESA: "work 1"}
    )
    review.write_cases([doubled_case])
    assert json.loads(path.read_text(encoding="utf-8"))["answer"] == {
        SIGEWINNE: "work 1",
        VARESA: "work 1",
    }


def test_review_file_write_cases_syncs_file_and_folder(tmp_path, doubled_case) -> None:
    with patch.object(review_file.os, "fsync", wraps=os.fsync) as fsync:
        ReviewFile(tmp_path / "review.jsonl").write_cases([doubled_case])
    assert fsync.call_count == 2


def test_review_file_write_cases_system_without_folder_sync_writes_file(
    tmp_path, monkeypatch, doubled_case
) -> None:
    monkeypatch.delattr(review_file.os, "O_DIRECTORY")
    path = tmp_path / "review.jsonl"
    ReviewFile(path).write_cases([doubled_case])
    assert (
        json.loads(path.read_text(encoding="utf-8"))["case_name"]
        == doubled_case.case_name
    )


def test_review_file_write_cases_existing_file_keeps_mode(
    tmp_path, doubled_case
) -> None:
    path = tmp_path / "review.jsonl"
    ReviewFile(path).write_cases([doubled_case])
    path.chmod(0o640)
    ReviewFile(path).write_cases([doubled_case])
    assert path.stat().st_mode & 0o777 == 0o640


def test_review_file_write_cases_new_file_gets_default_mode(
    tmp_path, doubled_case
) -> None:
    reference = tmp_path / "reference.txt"
    reference.write_text("", encoding="utf-8")
    path = tmp_path / "review.jsonl"
    ReviewFile(path).write_cases([doubled_case])
    assert path.stat().st_mode & 0o777 == reference.stat().st_mode & 0o777


@pytest.mark.parametrize("invalid_answer", INVALID_ANSWERS)
def test_review_file_write_cases_answer_not_accepted_logs_warning(
    tmp_path, doubled_case, caplog, invalid_answer
) -> None:
    path = tmp_path / "review.jsonl"
    ReviewFile(path).write_cases([doubled_case])
    _write_answer_in_file(path, doubled_case.case_name, invalid_answer)
    ReviewFile(path).write_cases([doubled_case])
    assert doubled_case.case_name in caplog.text


def test_review_file_case_answered_twice_logs_warning(
    tmp_path, doubled_case, caplog
) -> None:
    path = tmp_path / "review.jsonl"
    ReviewFile(path).write_cases([doubled_case])
    _write_answer_in_file(
        path, doubled_case.case_name, {SIGEWINNE: "work 1", VARESA: "work 1"}
    )
    path.write_text(path.read_text(encoding="utf-8") * 2, encoding="utf-8")
    ReviewFile(path)
    assert doubled_case.case_name in caplog.text


@pytest.mark.parametrize("line", ["[1, 2]", '{"case_name": 5, "answer": "one work"}'])
def test_review_file_line_not_case_object_raises_naming_file_and_line(
    tmp_path, line: str
) -> None:
    path = tmp_path / "review.jsonl"
    path.write_text(f"{line}\n", encoding="utf-8")
    with pytest.raises(ValidationError) as error_info:
        ReviewFile(path)
    assert error_info.value.__notes__ == [f"{path}, line 1"]


def test_review_file_broken_line_raises_naming_file_and_line(tmp_path) -> None:
    path = tmp_path / "review.jsonl"
    path.write_text(
        '{"case_name": "split work: mal:anime:1"}\n{"case_name": \n', encoding="utf-8"
    )
    with pytest.raises(ValidationError) as error_info:
        ReviewFile(path)
    assert error_info.value.__notes__ == [f"{path}, line 2"]


def _write_answer_in_file(path, case_name: str, answer: object) -> None:
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for line in lines:
        if line["case_name"] == case_name:
            line["answer"] = answer
    path.write_text(
        "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines),
        encoding="utf-8",
    )


@pytest.fixture
def review_file_path(tmp_path, review_case_lines):
    path = tmp_path / "review.jsonl"
    path.write_text("".join(review_case_lines), encoding="utf-8")
    return path


def test_review_file_save_answers_writes_answer_of_named_case(
    review_file_path, review_cases
) -> None:
    case_name = review_cases["nami"].case_name
    ReviewFile(review_file_path).save_answers(
        {case_name: {"anilist:anime:102289": "skip"}}
    )
    saved_cases = {
        case.case_name: case
        for case in map(
            ReviewCase.model_validate_json,
            review_file_path.read_text(encoding="utf-8").splitlines(),
        )
    }
    assert saved_cases[case_name].answer == {"anilist:anime:102289": "skip"}


def test_review_file_save_answers_keeps_other_lines_unchanged(
    review_file_path, review_cases, review_case_lines
) -> None:
    case_name = review_cases["nami"].case_name
    ReviewFile(review_file_path).save_answers(
        {case_name: {"anilist:anime:102289": "skip"}}
    )
    assert [
        line
        for line in review_file_path.read_text(encoding="utf-8").splitlines(
            keepends=True
        )
        if case_name not in line
    ] == [line for line in review_case_lines if case_name not in line]


def test_review_file_save_answers_title_with_line_separator_keeps_line_whole(
    review_file_path, review_cases, review_case_lines
) -> None:
    nami_case_name = review_cases["nami"].case_name
    nami_line = next(line for line in review_case_lines if nami_case_name in line)
    nami_fields = json.loads(nami_line)
    for entry in nami_fields["entries"].values():
        entry["title"] += "\u2028\u0085Part 2"
    separated_line = json.dumps(nami_fields, ensure_ascii=False) + "\n"
    review_file_path.write_text(
        "".join(
            separated_line if line == nami_line else line for line in review_case_lines
        ),
        encoding="utf-8",
    )
    ReviewFile(review_file_path).save_answers(
        {review_cases["atashinchi_next"].case_name: None}
    )
    with review_file_path.open(encoding="utf-8") as saved_file:
        assert separated_line in saved_file.readlines()


def test_review_file_save_answers_seen_saved_answer_matching_saves(
    review_file_path, review_cases
) -> None:
    case_name = review_cases["nami"].case_name
    review_file = ReviewFile(review_file_path)
    review_file.save_answers({case_name: {"anilist:anime:102289": "skip"}})
    review_file.save_answers(
        {case_name: {"anilist:anime:102289": "add"}},
        seen_saved_answers={case_name: {"anilist:anime:102289": "skip"}},
    )
    assert review_file.get_answers()[case_name] == {"anilist:anime:102289": "add"}


def test_review_file_save_answers_seen_saved_answer_changed_raises_and_keeps_file(
    review_file_path, review_cases
) -> None:
    case_name = review_cases["nami"].case_name
    review_file = ReviewFile(review_file_path)
    review_file.save_answers({case_name: {"anilist:anime:102289": "skip"}})
    saved_text = review_file_path.read_text(encoding="utf-8")
    with pytest.raises(SavedAnswerChangedError) as error_info:
        review_file.save_answers(
            {case_name: {"anilist:anime:102289": "add"}},
            seen_saved_answers={case_name: None},
        )
    assert error_info.value.case_names == (case_name,)
    assert review_file_path.read_text(encoding="utf-8") == saved_text


def test_review_file_save_answers_case_on_two_lines_writes_answer_to_both(
    review_file_path, review_cases, review_case_lines
) -> None:
    case_name = review_cases["nami"].case_name
    nami_line = next(line for line in review_case_lines if case_name in line)
    review_file_path.write_text(
        "".join(review_case_lines) + nami_line, encoding="utf-8"
    )
    review_file = ReviewFile(review_file_path)
    review_file.save_answers({case_name: {"anilist:anime:102289": "skip"}})
    saved_answers = [
        json.loads(line)["answer"]
        for line in review_file_path.read_text(encoding="utf-8").splitlines()
        if case_name in line
    ]
    assert saved_answers == [{"anilist:anime:102289": "skip"}] * 2
    assert review_file.get_answers()[case_name] == {"anilist:anime:102289": "skip"}


def test_review_file_save_answers_none_clears_answer(
    review_file_path, review_cases, review_case_lines
) -> None:
    case_name = review_cases["nami"].case_name
    review_file = ReviewFile(review_file_path)
    review_file.save_answers({case_name: {"anilist:anime:102289": "skip"}})
    review_file.save_answers({case_name: None})
    assert review_file_path.read_text(encoding="utf-8") == "".join(review_case_lines)


def test_review_file_save_answers_unknown_case_raises_and_keeps_file(
    review_file_path, review_case_lines
) -> None:
    with pytest.raises(UnknownCaseError):
        ReviewFile(review_file_path).save_answers({"split work: none:anime:1": None})
    assert review_file_path.read_text(encoding="utf-8") == "".join(review_case_lines)


def test_review_file_save_answers_lock_held_by_other_process_raises(
    review_file_path, review_cases
) -> None:
    holder = subprocess.Popen(  # noqa: S603  (fixed test arguments)
        [
            sys.executable,
            "-c",
            "import sys, time, filelock\n"
            "with filelock.FileLock(sys.argv[1]):\n"
            "    print('held', flush=True); time.sleep(5)",
            f"{review_file_path}.lock",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "held"
        with pytest.raises(ReviewFileLockedError):
            ReviewFile(review_file_path).save_answers(
                {review_cases["nami"].case_name: None}
            )
    finally:
        holder.kill()
        holder.wait()


def test_replace_file_lines_existing_file_holds_only_new_lines(tmp_path) -> None:
    path = tmp_path / "corrections.jsonl"
    path.write_text("old line\n", encoding="utf-8")
    replace_file_lines(path, ["first\n", "second\n"])
    assert path.read_text(encoding="utf-8") == "first\nsecond\n"


def test_replace_file_lines_missing_folder_created(tmp_path) -> None:
    path = tmp_path / "seed_cleaning" / "corrections.jsonl"
    replace_file_lines(path, ["line\n"])
    assert path.read_text(encoding="utf-8") == "line\n"


def test_replace_file_lines_failed_write_keeps_old_file_and_removes_temporary_file(
    tmp_path,
) -> None:
    path = tmp_path / "corrections.jsonl"
    path.write_text("old line\n", encoding="utf-8")
    with (
        patch.object(review_file.os, "fsync", autospec=True, side_effect=OSError),
        pytest.raises(OSError),
    ):
        replace_file_lines(path, ["new line\n"])
    assert (path.read_text(encoding="utf-8"), list(tmp_path.iterdir())) == (
        "old line\n",
        [path],
    )


def test_replace_file_lines_existing_file_keeps_mode(tmp_path) -> None:
    path = tmp_path / "corrections.jsonl"
    path.write_text("old line\n", encoding="utf-8")
    path.chmod(0o640)
    replace_file_lines(path, ["new line\n"])
    assert path.stat().st_mode & 0o777 == 0o640


@pytest.fixture
def empty_review_file_path(tmp_path) -> Path:
    path = tmp_path / "review.jsonl"
    path.write_text("", encoding="utf-8")
    return path


@pytest.fixture
def lock_held_by_other_process(empty_review_file_path) -> Iterator[None]:
    holder = subprocess.Popen(  # noqa: S603  (fixed test arguments)
        [
            sys.executable,
            "-c",
            "import sys, time, filelock\n"
            "with filelock.FileLock(sys.argv[1]):\n"
            "    print('held', flush=True); time.sleep(5)",
            f"{empty_review_file_path}.lock",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert holder.stdout is not None
    assert holder.stdout.readline().strip() == "held"
    yield
    holder.kill()
    holder.wait()


def test_is_review_file_locked_free_file_reports_false(
    empty_review_file_path,
) -> None:
    assert not is_review_file_locked(empty_review_file_path)


def test_is_review_file_locked_held_by_other_process_reports_true(
    empty_review_file_path, lock_held_by_other_process
) -> None:
    assert is_review_file_locked(empty_review_file_path)


def test_hold_review_file_lock_held_by_other_process_raises(
    empty_review_file_path, lock_held_by_other_process
) -> None:
    with (
        pytest.raises(ReviewFileLockedError),
        hold_review_file_lock(empty_review_file_path),
    ):
        pass


def test_hold_review_file_lock_while_held_reports_locked_to_others(
    empty_review_file_path,
) -> None:
    with hold_review_file_lock(empty_review_file_path):
        checker = subprocess.run(  # noqa: S603  (fixed test arguments)
            [
                sys.executable,
                "-c",
                "import sys, filelock\n"
                "try:\n"
                "    filelock.FileLock(sys.argv[1]).acquire(timeout=0)\n"
                "    print('free')\n"
                "except filelock.Timeout:\n"
                "    print('locked')",
                f"{empty_review_file_path}.lock",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    assert checker.stdout.strip() == "locked"


def test_hold_review_file_lock_released_after_block(empty_review_file_path) -> None:
    with hold_review_file_lock(empty_review_file_path):
        pass
    assert not is_review_file_locked(empty_review_file_path)
