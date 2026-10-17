import json

import pytest
from pydantic import ValidationError
from work_matching.doubled_provider_check import decide_doubled_providers
from work_matching.known_works_from_seed import KnownWorksFromSeed
from work_matching.review_cases import ReviewCase, ReviewFile
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


def test_review_case_from_doubled_provider_names_case_by_pages(doubled_case) -> None:
    assert (doubled_case.case, doubled_case.reason) == (
        f"doubled provider: {SIGEWINNE} {VARESA}",
        "UNDECIDED_PAGES",
    )
    assert doubled_case.entries == {
        SIGEWINNE: (
            "Genshin Impact: Character Teaser - Sigewinne: Everyday Correspondence"
        )
    }
    assert [page.key for page in doubled_case.pages] == [SIGEWINNE, VARESA]


def test_review_case_from_split_work_names_case_by_reference_pages(split_case) -> None:
    assert (split_case.case, split_case.reason) == (
        "split work: anilist:anime:1 mal:anime:1",
        "NOT_LINKED",
    )
    assert [page.key for page in split_case.pages] == ["mal:anime:1"]


@pytest.mark.parametrize(
    "answer",
    [
        {SIGEWINNE: "entry", VARESA: "elsewhere"},
        {SIGEWINNE: "entry", VARESA: "work 2"},
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
        {SIGEWINNE: "entry", VARESA: "elsewhere", "mal:anime:1": "entry"},
        {SIGEWINNE: "entry", VARESA: "work 1"},
        {SIGEWINNE: "entry", VARESA: "maybe"},
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
    answer = {SIGEWINNE: "entry", VARESA: "elsewhere"}
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


def test_review_file_write_cases_failed_replace_leaves_no_temporary_file(
    tmp_path, doubled_case
) -> None:
    path = tmp_path / "review.jsonl"
    review = ReviewFile(path)
    path.mkdir()
    with pytest.raises(IsADirectoryError):
        review.write_cases([doubled_case])
    assert list(tmp_path.iterdir()) == [path]


def test_review_file_blank_line_skipped(tmp_path, doubled_case) -> None:
    path = tmp_path / "review.jsonl"
    ReviewFile(path).write_cases([doubled_case])
    _answer_in_file(path, doubled_case.case, {SIGEWINNE: "entry", VARESA: "entry"})
    path.write_text("\n" + path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    kept = ReviewFile(path).with_kept_answer(doubled_case)
    assert kept.accepted_answer == {SIGEWINNE: "entry", VARESA: "entry"}


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
