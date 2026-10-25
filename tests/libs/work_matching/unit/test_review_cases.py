import json
import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError
from work_matching import file_replacement
from work_matching.doubled_provider_check import decide_doubled_providers
from work_matching.entry_group import combine_group_entries
from work_matching.known_work_entries_from_seed import KnownWorkEntriesFromSeed
from work_matching.linked_page_check import (
    LinkedPageAddition,
    LinkingEntryDetails,
    decide_linked_page,
)
from work_matching.page_placement import PagePlacement
from work_matching.provider_page import ProviderPage
from work_matching.review_cases import DuplicateCaseError, ReviewCase, ReviewFile
from work_matching.split_work_check import list_split_work_review_reasons
from work_matching.work_entry import WorkEntry

SIGEWINNE = "animeplanet:anime:genshin-impact-character-teaser-sigewinne-everyday-correspondence"
VARESA = (
    "animeplanet:anime:genshin-impact-character-teaser-varesa-the-legend-of-a-warrior"
)


@pytest.fixture
def doubled_case(matching_cases) -> ReviewCase:
    entries, pages = matching_cases["genshin_sigewinne"]
    decisions = decide_doubled_providers(
        entries[0], pages, KnownWorkEntriesFromSeed([])
    )
    return ReviewCase.from_entry_group([entries[0]], decisions, (), pages)


def _build_group_case(entries, pages) -> ReviewCase:
    decisions = decide_doubled_providers(
        combine_group_entries(entries), pages, KnownWorkEntriesFromSeed([])
    )
    return ReviewCase.from_entry_group(
        entries, decisions, list_split_work_review_reasons(entries, pages), pages
    )


@pytest.fixture
def split_case(linked_page, work_entry) -> ReviewCase:
    entries = [
        work_entry("https://myanimelist.net/anime/1"),
        work_entry("https://anilist.co/anime/1"),
    ]
    return _build_group_case(entries, [linked_page("https://myanimelist.net/anime/1")])


@pytest.fixture
def linked_page_case() -> ReviewCase:
    entry = WorkEntry.from_seed_entry(
        {
            "title": "The Boxer",
            "type": "TV",
            "sources": ["https://anilist.co/anime/163794"],
        }
    )
    entry_page = ProviderPage(
        key="anilist:anime:163794",
        address="https://anilist.co/anime/163794",
        titles=("The Boxer",),
        type="TV",
        episode_count=12,
        linked_page_keys=("mal:anime:65144",),
    )
    linked_page = ProviderPage(
        key="mal:anime:65144",
        address="https://myanimelist.net/anime/65144",
        titles=("The Boxer",),
        type="TV",
        episode_count=1,
    )
    decision = decide_linked_page(
        linked_page,
        [
            LinkingEntryDetails(
                entry, entry_page.episode_count, frozenset({linked_page.key})
            )
        ],
        KnownWorkEntriesFromSeed([]),
    )
    return ReviewCase.from_linked_page(decision, [linked_page, entry_page])


def test_review_case_from_entry_group_single_entry_names_case_by_pages_and_entry(
    doubled_case,
) -> None:
    assert (doubled_case.case_name, doubled_case.reasons) == (
        f"doubled provider: {SIGEWINNE} {VARESA} in {SIGEWINNE}",
        ("UNDECIDED", "UNLINKED_PAGES"),
    )
    assert doubled_case.entries[SIGEWINNE].title == (
        "Genshin Impact: Character Teaser - Sigewinne: Everyday Correspondence"
    )
    assert [page.key for page in doubled_case.pages] == [SIGEWINNE, VARESA]


def test_review_case_from_entry_group_single_entry_keeps_entry_and_searched_pages(
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
        searched_by_address="https://myanimelist.net/anime/1",
    )
    pages = [
        linked_page("https://kitsu.app/anime/1"),
        linked_page("https://kitsu.app/anime/2"),
        linked_page("https://myanimelist.net/anime/1"),
        linked_page("https://anilist.co/anime/9"),
        found,
    ]
    decisions = decide_doubled_providers(entry, pages, KnownWorkEntriesFromSeed([]))
    case = ReviewCase.from_entry_group([entry], decisions, (), pages)
    assert [page.key for page in case.pages] == [
        "https://animeschedule.net/anime/show",
        "kitsu:anime:1",
        "kitsu:anime:2",
        "mal:anime:1",
    ]


def test_review_case_from_entry_group_single_entry_one_case_for_every_provider(
    matching_cases,
) -> None:
    entries, pages = matching_cases["ao_haru_ride_ova"]
    decisions = decide_doubled_providers(
        entries[0], pages, KnownWorkEntriesFromSeed([])
    )
    case = ReviewCase.from_entry_group([entries[0]], decisions, (), pages)
    assert case.case_name == (
        "doubled provider: anilist:anime:20837 anilist:anime:20900"
        " animeplanet:anime:blue-spring-ride-page-13"
        " animeplanet:anime:blue-spring-ride-unwritten in mal:anime:24151"
    )
    assert case.reasons == ("PART_PAGES", "UNDECIDED", "UNLINKED_PAGES")


def test_review_case_from_entry_group_single_entry_suggests_linked_places(
    matching_cases,
) -> None:
    entries, pages = matching_cases["ao_haru_ride_ova"]
    decisions = decide_doubled_providers(
        entries[0], pages, KnownWorkEntriesFromSeed([])
    )
    case = ReviewCase.from_entry_group([entries[0]], decisions, (), pages)
    assert case.suggested_answer["anilist:anime:20900"] == "work 1, part"
    assert "animeplanet:anime:blue-spring-ride-unwritten" not in case.suggested_answer


def test_review_case_from_entry_group_unlinked_entries_suggest_nothing(
    split_case,
) -> None:
    assert split_case.suggested_answer == {}


def test_review_case_from_entry_group_linked_entries_suggest_one_work(
    linked_page, work_entry
) -> None:
    entries = [
        work_entry("https://myanimelist.net/anime/1"),
        work_entry("https://anilist.co/anime/1"),
    ]
    pages = [
        linked_page("https://anilist.co/anime/1", "https://myanimelist.net/anime/1")
    ]
    assert _build_group_case(entries, pages).suggested_answer == {
        "mal:anime:1": "work 1",
        "anilist:anime:1": "work 1",
    }


def test_review_case_from_entry_group_single_entry_writes_linked_pages_sorted(
    matching_cases,
) -> None:
    entries, pages = matching_cases["puu_neko"]
    decisions = decide_doubled_providers(
        entries[0], pages, KnownWorkEntriesFromSeed([])
    )
    case = ReviewCase.from_entry_group([entries[0]], decisions, (), pages)
    written_decisions = json.loads(case.model_dump_json())["decisions"]
    assert all(
        linked_pages == sorted(linked_pages)
        for decision in written_decisions
        for linked_pages in decision["belonging_page_keys"].values()
    )


def test_review_case_from_entry_group_single_entry_writes_unrelated_linked_page_keys_sorted(
    linked_page, work_entry
) -> None:
    entry = work_entry(
        "https://myanimelist.net/anime/1",
        "https://kitsu.app/anime/1",
        "https://kitsu.app/anime/2",
    )
    linked_addresses = [
        f"https://myanimelist.net/anime/{number}" for number in range(10, 30)
    ]
    known_work_entries = KnownWorkEntriesFromSeed(
        [{"title": "Other", "type": "TV", "sources": linked_addresses}]
    )
    pages = [
        linked_page("https://kitsu.app/anime/1", "https://myanimelist.net/anime/1"),
        linked_page("https://kitsu.app/anime/2", *linked_addresses),
    ]
    decisions = decide_doubled_providers(entry, pages, known_work_entries)
    case = ReviewCase.from_entry_group([entry], decisions, (), pages)
    [decision] = json.loads(case.model_dump_json())["decisions"]
    written_links = decision["unrelated_pages"]["kitsu:anime:2"]["linked_page_keys"]
    assert written_links == sorted(written_links)


def test_review_case_from_entry_group_keeps_searched_pages(
    linked_page, work_entry
) -> None:
    entries = [
        work_entry("https://myanimelist.net/anime/1"),
        work_entry("https://anilist.co/anime/1"),
    ]
    found = ProviderPage.from_canonical_record(
        "https://animeschedule.net/anime/show",
        {"title": "Show"},
        searched_by_address="https://myanimelist.net/anime/1",
    )
    pages = [linked_page("https://myanimelist.net/anime/1"), found]
    case = _build_group_case(entries, pages)
    assert [page.key for page in case.pages] == [
        "https://animeschedule.net/anime/show",
        "mal:anime:1",
    ]


def test_review_case_from_entry_group_keeps_renamed_page(work_entry) -> None:
    old_address = "https://www.anime-planet.com/anime/old"
    entries = [work_entry(old_address), work_entry("https://myanimelist.net/anime/1")]
    renamed = ProviderPage.from_canonical_record(
        old_address,
        {
            "title": "Show",
            "sources": [old_address, "https://www.anime-planet.com/anime/new"],
        },
    )
    case = _build_group_case(entries, [renamed])
    assert [page.key for page in case.pages] == ["animeplanet:anime:old"]


def test_review_case_from_entry_group_names_group_case_by_every_page(
    split_case,
) -> None:
    assert (split_case.case_name, split_case.reasons) == (
        "split work: anilist:anime:1 mal:anime:1",
        ("NOT_TIED_BY_LINKS",),
    )
    assert [page.key for page in split_case.pages] == ["mal:anime:1"]


@pytest.mark.parametrize(
    ("answer", "placements"),
    [
        (
            {SIGEWINNE: "work 1", VARESA: "other work"},
            {SIGEWINNE: PagePlacement("work 1"), VARESA: PagePlacement("other work")},
        ),
        (
            {SIGEWINNE: "work 1", VARESA: "work 10"},
            {SIGEWINNE: PagePlacement("work 1"), VARESA: PagePlacement("work 10")},
        ),
        (
            {SIGEWINNE: "work 1, main", VARESA: "work 1"},
            {
                SIGEWINNE: PagePlacement("work 1", is_main_page=True),
                VARESA: PagePlacement("work 1"),
            },
        ),
    ],
)
def test_review_case_accepted_answer_placement_for_every_page_gives_placements(
    doubled_case, answer: dict[str, str], placements: dict[str, PagePlacement]
) -> None:
    accepted = doubled_case.model_copy(update={"answer": answer}).accepted_answer
    assert accepted == placements


@pytest.mark.parametrize(
    "answer",
    [
        None,
        {SIGEWINNE: "work 1"},
        {SIGEWINNE: "work 1", VARESA: "work 1"},
        {SIGEWINNE: "work 1", VARESA: "other work", "mal:anime:1": "work 1"},
        {SIGEWINNE: "work 1", VARESA: "work 1"},
        {SIGEWINNE: "work 1", VARESA: "maybe"},
        {SIGEWINNE: "work 1", VARESA: "work 01"},
        {SIGEWINNE: "work 1", VARESA: "work 0"},
        {SIGEWINNE: "work 1", VARESA: "work 1٢"},
        {SIGEWINNE: 1, VARESA: ["work 1"]},
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
        (
            {"mal:anime:1": "work 1", "anilist:anime:1": "work 1"},
            {
                "mal:anime:1": PagePlacement("work 1"),
                "anilist:anime:1": PagePlacement("work 1"),
            },
        ),
        (
            {"mal:anime:1": "work 1", "anilist:anime:1": "work 2"},
            {
                "mal:anime:1": PagePlacement("work 1"),
                "anilist:anime:1": PagePlacement("work 2"),
            },
        ),
        ("one work", None),
        ({"mal:anime:1": "work 1"}, None),
    ],
)
def test_review_case_accepted_answer_entry_group_places_every_page(
    split_case, answer: object, accepted: dict[str, PagePlacement] | None
) -> None:
    assert split_case.model_copy(update={"answer": answer}).accepted_answer == accepted


def test_review_case_from_linked_page_names_case_by_page_and_linking_entries(
    linked_page_case,
) -> None:
    assert (linked_page_case.case_name, linked_page_case.reasons) == (
        "linked page: mal:anime:65144 from anilist:anime:163794",
        ("EPISODE_COUNTS_DIFFER",),
    )
    assert list(linked_page_case.entries) == ["anilist:anime:163794"]
    assert [page.key for page in linked_page_case.pages] == [
        "anilist:anime:163794",
        "mal:anime:65144",
    ]


def test_review_case_from_linked_page_suggests_part_for_fewer_episodes(
    linked_page_case,
) -> None:
    assert linked_page_case.suggested_answer == {"mal:anime:65144": "add as part"}


@pytest.mark.parametrize(
    ("answer", "accepted"),
    [
        (
            {"mal:anime:65144": "add as part"},
            LinkedPageAddition("anilist:anime:163794", is_part_page=True),
        ),
        ({"mal:anime:65144": "skip"}, "skip"),
        ({"mal:anime:65144": "work 1"}, None),
        ("one work", None),
    ],
)
def test_review_case_accepted_answer_linked_page_reads_addition(
    linked_page_case, answer: object, accepted: object
) -> None:
    assert (
        linked_page_case.model_copy(update={"answer": answer}).accepted_answer
        == accepted
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
