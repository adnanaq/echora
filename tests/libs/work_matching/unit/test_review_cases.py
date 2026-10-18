import json
import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError
from work_matching import review_cases
from work_matching.doubled_provider_check import decide_doubled_providers
from work_matching.known_works_from_seed import KnownWorksFromSeed
from work_matching.provider_page import ProviderPage
from work_matching.review_cases import DuplicateCaseError, ReviewCase, ReviewFile
from work_matching.split_work_check import decide_split_work

SIGEWINNE = "animeplanet:anime:genshin-impact-character-teaser-sigewinne-everyday-correspondence"
VARESA = (
    "animeplanet:anime:genshin-impact-character-teaser-varesa-the-legend-of-a-warrior"
)


@pytest.fixture
def doubled_case(matching_cases) -> ReviewCase:
    entries, pages = matching_cases["genshin_sigewinne"]
    [decision] = decide_doubled_providers(entries[0], pages, KnownWorksFromSeed([]))
    return ReviewCase.from_doubled_provider(entries[0], decision, pages)


@pytest.fixture
def split_case(linked_page, work_entry) -> ReviewCase:
    entries = [
        work_entry("https://myanimelist.net/anime/1"),
        work_entry("https://anilist.co/anime/1"),
    ]
    pages = [linked_page("https://myanimelist.net/anime/1")]
    return ReviewCase.from_split_work(decide_split_work(entries, pages), pages)


def test_review_case_from_doubled_provider_names_case_by_pages_and_entry(
    doubled_case,
) -> None:
    assert (doubled_case.case, doubled_case.reason) == (
        f"doubled provider: {SIGEWINNE} {VARESA} in {SIGEWINNE}",
        "UNDECIDED_PAGES",
    )
    assert doubled_case.entries == {
        SIGEWINNE: (
            "Genshin Impact: Character Teaser - Sigewinne: Everyday Correspondence"
        )
    }
    assert [page.key for page in doubled_case.pages] == [SIGEWINNE, VARESA]


def test_review_case_from_doubled_provider_keeps_entry_and_found_pages(
    linked_page, work_entry
) -> None:
    entry = work_entry(
        "https://myanimelist.net/anime/1",
        "https://kitsu.app/anime/1",
        "https://kitsu.app/anime/2",
    )
    found = ProviderPage.from_canonical_record(
        "https://animeschedule.net/anime/show",
        {"title": "Show"},
        found_by="https://myanimelist.net/anime/1",
    )
    pages = [
        linked_page("https://kitsu.app/anime/1"),
        linked_page("https://kitsu.app/anime/2"),
        linked_page("https://myanimelist.net/anime/1"),
        linked_page("https://anilist.co/anime/9"),
        found,
    ]
    [decision] = decide_doubled_providers(entry, pages, KnownWorksFromSeed([]))
    case = ReviewCase.from_doubled_provider(entry, decision, pages)
    assert [page.key for page in case.pages] == [
        "https://animeschedule.net/anime/show",
        "kitsu:anime:1",
        "kitsu:anime:2",
        "mal:anime:1",
    ]


def test_review_case_from_split_work_keeps_found_pages(linked_page, work_entry) -> None:
    entries = [
        work_entry("https://myanimelist.net/anime/1"),
        work_entry("https://anilist.co/anime/1"),
    ]
    found = ProviderPage.from_canonical_record(
        "https://animeschedule.net/anime/show",
        {"title": "Show"},
        found_by="https://myanimelist.net/anime/1",
    )
    pages = [linked_page("https://myanimelist.net/anime/1"), found]
    case = ReviewCase.from_split_work(decide_split_work(entries, pages), pages)
    assert [page.key for page in case.pages] == [
        "https://animeschedule.net/anime/show",
        "mal:anime:1",
    ]


def test_review_case_from_split_work_keeps_renamed_page(work_entry) -> None:
    old_address = "https://www.anime-planet.com/anime/old"
    entries = [work_entry(old_address), work_entry("https://myanimelist.net/anime/1")]
    renamed = ProviderPage.from_canonical_record(
        old_address,
        {
            "title": "Show",
            "sources": [old_address, "https://www.anime-planet.com/anime/new"],
        },
    )
    case = ReviewCase.from_split_work(decide_split_work(entries, [renamed]), [renamed])
    assert [page.key for page in case.pages] == ["animeplanet:anime:old"]


def test_review_case_from_split_work_names_case_by_reference_pages(split_case) -> None:
    assert (split_case.case, split_case.reason) == (
        "split work: anilist:anime:1 mal:anime:1",
        "NOT_LINKED",
    )
    assert [page.key for page in split_case.pages] == ["mal:anime:1"]


@pytest.mark.parametrize(
    "answer",
    [
        {SIGEWINNE: "entry", VARESA: "other work"},
        {SIGEWINNE: "entry", VARESA: "work 2"},
        {SIGEWINNE: "entry", VARESA: "work 10"},
        {SIGEWINNE: "entry", VARESA: "entry"},
    ],
)
def test_review_case_accepted_answer_placement_for_every_page_gives_answer(
    doubled_case, answer: dict[str, str]
) -> None:
    assert doubled_case.model_copy(update={"answer": answer}).accepted_answer == answer


@pytest.mark.parametrize(
    "answer",
    [
        None,
        {SIGEWINNE: "entry"},
        {SIGEWINNE: "entry", VARESA: "other work", "mal:anime:1": "entry"},
        {SIGEWINNE: "entry", VARESA: "work 1"},
        {SIGEWINNE: "entry", VARESA: "maybe"},
        {SIGEWINNE: "entry", VARESA: "work 01"},
        {SIGEWINNE: "entry", VARESA: "work 0"},
        {SIGEWINNE: "entry", VARESA: "work 1٢"},
        {SIGEWINNE: 1, VARESA: ["entry"]},
        "one work",
    ],
)
def test_review_case_accepted_answer_invalid_doubled_provider_answer_gives_none(
    doubled_case, answer: object
) -> None:
    assert doubled_case.model_copy(update={"answer": answer}).accepted_answer is None


@pytest.mark.parametrize(
    ("answer", "accepted"),
    [
        ("one work", "one work"),
        ("separate works", "separate works"),
        ("maybe", None),
        ({"mal:anime:1": "entry"}, None),
    ],
)
def test_review_case_accepted_answer_split_work_takes_one_of_two_answers(
    split_case, answer: object, accepted: str | None
) -> None:
    assert split_case.model_copy(update={"answer": answer}).accepted_answer == accepted


def test_review_file_missing_file_keeps_no_answers(tmp_path, doubled_case) -> None:
    review = ReviewFile(tmp_path / "review.jsonl")
    assert review.with_kept_answer(doubled_case) == doubled_case


def test_review_file_write_cases_puts_case_and_answer_first(
    tmp_path, doubled_case
) -> None:
    path = tmp_path / "review.jsonl"
    ReviewFile(path).write_cases([doubled_case])
    [line] = path.read_text(encoding="utf-8").splitlines()
    assert list(json.loads(line))[:3] == ["case", "answer", "reason"]


def test_review_file_with_kept_answer_answer_written_in_file_comes_back(
    tmp_path, doubled_case
) -> None:
    path = tmp_path / "review.jsonl"
    ReviewFile(path).write_cases([doubled_case])
    answer = {SIGEWINNE: "entry", VARESA: "other work"}
    _answer_in_file(path, doubled_case.case, answer)
    kept = ReviewFile(path).with_kept_answer(doubled_case)
    assert kept.accepted_answer == answer


def test_review_file_with_kept_answer_provider_gaining_page_reopens_case(
    tmp_path, matching_cases, doubled_case
) -> None:
    path = tmp_path / "review.jsonl"
    ReviewFile(path).write_cases([doubled_case])
    _answer_in_file(path, doubled_case.case, {SIGEWINNE: "entry", VARESA: "entry"})
    entries, pages = matching_cases["genshin_sigewinne"]
    third_page = "https://anime-planet.com/anime/genshin-impact-third-teaser"
    entry = entries[0].model_copy(
        update={
            "pages": {
                **entries[0].pages,
                "animeplanet:anime:genshin-impact-third-teaser": third_page,
            }
        }
    )
    [decision] = decide_doubled_providers(entry, pages, KnownWorksFromSeed([]))
    later_case = ReviewCase.from_doubled_provider(entry, decision, pages)
    assert later_case.case != doubled_case.case
    assert ReviewFile(path).with_kept_answer(later_case).answer is None


def test_review_file_with_kept_answer_doubled_pages_in_other_entry_reopens_case(
    tmp_path, matching_cases, doubled_case
) -> None:
    path = tmp_path / "review.jsonl"
    ReviewFile(path).write_cases([doubled_case])
    _answer_in_file(path, doubled_case.case, {SIGEWINNE: "entry", VARESA: "entry"})
    entries, pages = matching_cases["genshin_sigewinne"]
    other_entry = entries[0].model_copy(
        update={
            "pages": {"mal:anime:1": "https://myanimelist.net/anime/1"}
            | entries[0].pages
        }
    )
    [decision] = decide_doubled_providers(other_entry, pages, KnownWorksFromSeed([]))
    moved_case = ReviewCase.from_doubled_provider(other_entry, decision, pages)
    assert moved_case.case != doubled_case.case
    assert ReviewFile(path).with_kept_answer(moved_case).answer is None


def test_review_file_write_cases_answer_without_case_kept_and_reported(
    tmp_path, doubled_case, split_case
) -> None:
    path = tmp_path / "review.jsonl"
    ReviewFile(path).write_cases([doubled_case, split_case])
    _answer_in_file(path, doubled_case.case, {SIGEWINNE: "entry", VARESA: "entry"})
    unused = ReviewFile(path).write_cases([split_case])
    cases = [json.loads(line)["case"] for line in path.read_text().splitlines()]
    assert (unused, cases) == (
        (doubled_case.case,),
        [split_case.case, doubled_case.case],
    )


def test_review_file_write_cases_invalid_answer_without_case_kept_unchanged(
    tmp_path, doubled_case, split_case
) -> None:
    path = tmp_path / "review.jsonl"
    ReviewFile(path).write_cases([doubled_case, split_case])
    _answer_in_file(path, doubled_case.case, "maybe")
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
    assert [json.loads(line)["case"] for line in path.read_text().splitlines()] == [
        split_case.case
    ]


def test_review_file_write_cases_invalid_answer_kept_for_fixing(
    tmp_path, doubled_case
) -> None:
    path = tmp_path / "review.jsonl"
    ReviewFile(path).write_cases([doubled_case])
    _answer_in_file(path, doubled_case.case, {SIGEWINNE: "maybe"})
    review = ReviewFile(path)
    kept = review.with_kept_answer(doubled_case)
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
        patch.object(review_cases.os, "fsync", autospec=True, side_effect=OSError),
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
    _answer_in_file(path, doubled_case.case, {SIGEWINNE: "entry", VARESA: "entry"})
    path.write_text("\n" + path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    review = ReviewFile(path)
    kept = review.with_kept_answer(doubled_case)
    review.write_cases([kept])
    assert kept.accepted_answer == {SIGEWINNE: "entry", VARESA: "entry"}
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
    _answer_in_file(path, doubled_case.case, {SIGEWINNE: "entry", VARESA: "entry"})
    review.write_cases([doubled_case])
    assert json.loads(path.read_text(encoding="utf-8"))["answer"] == {
        SIGEWINNE: "entry",
        VARESA: "entry",
    }


def test_review_file_write_cases_syncs_file_and_folder(tmp_path, doubled_case) -> None:
    with patch.object(review_cases.os, "fsync", wraps=os.fsync) as fsync:
        ReviewFile(tmp_path / "review.jsonl").write_cases([doubled_case])
    assert fsync.call_count == 2


def test_review_file_write_cases_system_without_folder_sync_writes_file(
    tmp_path, monkeypatch, doubled_case
) -> None:
    monkeypatch.delattr(review_cases.os, "O_DIRECTORY")
    path = tmp_path / "review.jsonl"
    ReviewFile(path).write_cases([doubled_case])
    assert json.loads(path.read_text(encoding="utf-8"))["case"] == doubled_case.case


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
    _answer_in_file(path, doubled_case.case, {SIGEWINNE: "maybe"})
    ReviewFile(path).write_cases([doubled_case])
    assert doubled_case.case in caplog.text


def test_review_file_case_answered_twice_logs_warning(
    tmp_path, doubled_case, caplog
) -> None:
    path = tmp_path / "review.jsonl"
    ReviewFile(path).write_cases([doubled_case])
    _answer_in_file(path, doubled_case.case, {SIGEWINNE: "entry", VARESA: "entry"})
    path.write_text(path.read_text(encoding="utf-8") * 2, encoding="utf-8")
    ReviewFile(path)
    assert doubled_case.case in caplog.text


@pytest.mark.parametrize("line", ["[1, 2]", '{"case": 5, "answer": "one work"}'])
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
        '{"case": "split work: mal:anime:1"}\n{"case": \n', encoding="utf-8"
    )
    with pytest.raises(ValidationError) as raised:
        ReviewFile(path)
    assert raised.value.__notes__ == [f"{path}, line 2"]


def _answer_in_file(path, case_name: str, answer: object) -> None:
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for line in lines:
        if line["case"] == case_name:
            line["answer"] = answer
    path.write_text(
        "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines),
        encoding="utf-8",
    )
