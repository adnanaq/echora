import json
import os
import subprocess
import sys
from unittest.mock import patch

import pytest
from pydantic import ValidationError
from work_matching import file_replacement
from work_matching.doubled_provider_check import decide_doubled_providers
from work_matching.known_work_entries_from_seed import KnownWorkEntriesFromSeed
from work_matching.linked_page_check import (
    LinkedPageAddition,
)
from work_matching.page_placement import PagePlacement
from work_matching.review_cases import ReviewCase
from work_matching.review_file import DuplicateCaseError, ReviewFile, UnknownCaseError
from work_matching.review_file_lock import ReviewFileLockedError

SIGEWINNE = "animeplanet:anime:genshin-impact-character-teaser-sigewinne-everyday-correspondence"
VARESA = (
    "animeplanet:anime:genshin-impact-character-teaser-varesa-the-legend-of-a-warrior"
)


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
        "anilist:anime:163794", is_part_page=False
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
    _answer_in_file(path, doubled_case.case_name, answer)
    kept = ReviewFile(path).attach_kept_answer(doubled_case)
    assert kept.accepted_answer == {
        SIGEWINNE: PagePlacement("work 1"),
        VARESA: PagePlacement("other work"),
    }


def test_review_file_attach_kept_answer_provider_gaining_page_reopens_case(
    tmp_path, matching_cases, doubled_case
) -> None:
    path = tmp_path / "review.jsonl"
    ReviewFile(path).write_cases([doubled_case])
    _answer_in_file(
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
    later_case = ReviewCase.from_entry_group([entry], decisions, (), pages)
    assert later_case.case_name != doubled_case.case_name
    assert ReviewFile(path).attach_kept_answer(later_case).answer is None


def test_review_file_attach_kept_answer_doubled_pages_in_other_entry_reopens_case(
    tmp_path, matching_cases, doubled_case
) -> None:
    path = tmp_path / "review.jsonl"
    ReviewFile(path).write_cases([doubled_case])
    _answer_in_file(
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
    moved_case = ReviewCase.from_entry_group([other_entry], decisions, (), pages)
    assert moved_case.case_name != doubled_case.case_name
    assert ReviewFile(path).attach_kept_answer(moved_case).answer is None


def test_review_file_write_cases_answer_without_case_kept_and_reported(
    tmp_path, doubled_case, split_case
) -> None:
    path = tmp_path / "review.jsonl"
    ReviewFile(path).write_cases([doubled_case, split_case])
    _answer_in_file(
        path, doubled_case.case_name, {SIGEWINNE: "work 1", VARESA: "work 1"}
    )
    unused = ReviewFile(path).write_cases([split_case])
    cases = [json.loads(line)["case_name"] for line in path.read_text().splitlines()]
    assert (unused, cases) == (
        (doubled_case.case_name,),
        [split_case.case_name, doubled_case.case_name],
    )


def test_review_file_write_cases_invalid_answer_without_case_kept_unchanged(
    tmp_path, doubled_case, split_case
) -> None:
    path = tmp_path / "review.jsonl"
    ReviewFile(path).write_cases([doubled_case, split_case])
    _answer_in_file(path, doubled_case.case_name, "maybe")
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


def test_review_file_write_cases_invalid_answer_kept_for_fixing(
    tmp_path, doubled_case
) -> None:
    path = tmp_path / "review.jsonl"
    ReviewFile(path).write_cases([doubled_case])
    _answer_in_file(path, doubled_case.case_name, {SIGEWINNE: "maybe"})
    review = ReviewFile(path)
    kept = review.attach_kept_answer(doubled_case)
    review.write_cases([kept])
    assert kept.accepted_answer is None
    assert json.loads(path.read_text())["answer"] == {SIGEWINNE: "maybe"}


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
    written = path.read_text(encoding="utf-8")
    with (
        patch.object(file_replacement.os, "fsync", autospec=True, side_effect=OSError),
        pytest.raises(OSError),
    ):
        ReviewFile(path).write_cases([split_case])
    assert (path.read_text(encoding="utf-8"), list(tmp_path.iterdir())) == (
        written,
        [path],
    )


def test_review_file_blank_line_answers_read_and_line_not_written(
    tmp_path, doubled_case
) -> None:
    path = tmp_path / "review.jsonl"
    ReviewFile(path).write_cases([doubled_case])
    _answer_in_file(
        path, doubled_case.case_name, {SIGEWINNE: "work 1", VARESA: "work 2"}
    )
    path.write_text("\n" + path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    review = ReviewFile(path)
    kept = review.attach_kept_answer(doubled_case)
    review.write_cases([kept])
    assert kept.answer == {SIGEWINNE: "work 1", VARESA: "work 2"}
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
    _answer_in_file(
        path, doubled_case.case_name, {SIGEWINNE: "work 1", VARESA: "work 1"}
    )
    review.write_cases([doubled_case])
    assert json.loads(path.read_text(encoding="utf-8"))["answer"] == {
        SIGEWINNE: "work 1",
        VARESA: "work 1",
    }


def test_review_file_write_cases_syncs_file_and_folder(tmp_path, doubled_case) -> None:
    with patch.object(file_replacement.os, "fsync", wraps=os.fsync) as fsync:
        ReviewFile(tmp_path / "review.jsonl").write_cases([doubled_case])
    assert fsync.call_count == 2


def test_review_file_write_cases_system_without_folder_sync_writes_file(
    tmp_path, monkeypatch, doubled_case
) -> None:
    monkeypatch.delattr(file_replacement.os, "O_DIRECTORY")
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


def test_review_file_write_cases_answer_not_accepted_logs_warning(
    tmp_path, doubled_case, caplog
) -> None:
    path = tmp_path / "review.jsonl"
    ReviewFile(path).write_cases([doubled_case])
    _answer_in_file(path, doubled_case.case_name, {SIGEWINNE: "maybe"})
    ReviewFile(path).write_cases([doubled_case])
    assert doubled_case.case_name in caplog.text


def test_review_file_case_answered_twice_logs_warning(
    tmp_path, doubled_case, caplog
) -> None:
    path = tmp_path / "review.jsonl"
    ReviewFile(path).write_cases([doubled_case])
    _answer_in_file(
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
    with pytest.raises(ValidationError) as raised:
        ReviewFile(path)
    assert raised.value.__notes__ == [f"{path}, line 1"]


def test_review_file_broken_line_raises_naming_file_and_line(tmp_path) -> None:
    path = tmp_path / "review.jsonl"
    path.write_text(
        '{"case_name": "split work: mal:anime:1"}\n{"case_name": \n', encoding="utf-8"
    )
    with pytest.raises(ValidationError) as raised:
        ReviewFile(path)
    assert raised.value.__notes__ == [f"{path}, line 2"]


def _answer_in_file(path, case_name: str, answer: object) -> None:
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
