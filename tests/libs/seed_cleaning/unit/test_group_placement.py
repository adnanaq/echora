import pytest
from seed_cleaning.corrected_entries import (
    build_corrected_entries,
)
from seed_cleaning.group_placement import correct_entry_groups
from seed_cleaning.seed_corrections import SeedCorrection
from work_matching import (
    KnownWorkEntriesFromSeed,
    ProviderPage,
    ProviderPageStore,
    ReviewFile,
    WorkEntry,
)

MAL = "https://myanimelist.net/anime/1"
KITSU_ONE = "https://kitsu.app/anime/1"
KITSU_TWO = "https://kitsu.app/anime/2"
TARGET_MAL = "https://myanimelist.net/anime/9"
OLD_PLANET = "https://www.anime-planet.com/anime/old-name"
NEW_PLANET = "https://www.anime-planet.com/anime/new-name"
SCHEDULE = "https://animeschedule.net/anime/show"


def _seed_entry(title: str, *addresses: str) -> dict:
    return {
        "title": title,
        "type": "TV",
        "sources": list(addresses),
        "animeSeason": {"season": "SPRING", "year": 2020},
    }


def _page(address: str, *linked_addresses: str, **details) -> ProviderPage:
    return ProviderPage.from_canonical_record(
        address,
        {
            "title": "Show",
            "external_sources": [
                {"source": linked_address} for linked_address in linked_addresses
            ],
        }
        | details,
    )


@pytest.fixture
def known_work_entries() -> KnownWorkEntriesFromSeed:
    return KnownWorkEntriesFromSeed(
        [
            _seed_entry("Show", MAL, KITSU_ONE, KITSU_TWO),
            _seed_entry("Show Movie", TARGET_MAL),
        ]
    )


@pytest.fixture
def entry(known_work_entries) -> WorkEntry:
    return next(iter(known_work_entries))


@pytest.fixture
def store(tmp_path) -> ProviderPageStore:
    return ProviderPageStore(tmp_path / "provider_pages.jsonl")


@pytest.fixture
def review_file(tmp_path) -> ReviewFile:
    return ReviewFile(tmp_path / "review.jsonl")


def _correct(entry, store, known_work_entries, review_file):
    return correct_entry_groups([[entry]], store, known_work_entries, review_file)


def test_correct_entry_groups_entry_without_review_moves_page_to_seed_entry(
    entry, store, known_work_entries, review_file
) -> None:
    store.add_pages(_page(KITSU_ONE, MAL), _page(KITSU_TWO, TARGET_MAL))
    result = _correct(entry, store, known_work_entries, review_file)
    assert result.review_cases == ()
    assert result.corrections == (
        SeedCorrection(
            entry_reference_page="mal:anime:1",
            page_key="kitsu:anime:1",
            address=KITSU_ONE,
            placement_word="work 1",
            reason="PAGES_OF_OTHER_WORKS",
        ),
        SeedCorrection(
            entry_reference_page="mal:anime:1",
            page_key="kitsu:anime:2",
            address=KITSU_TWO,
            placement_word="other work",
            target_entry_reference_page="mal:anime:9",
            reason="PAGES_OF_OTHER_WORKS",
        ),
    )


def test_correct_entry_groups_link_on_searched_page_settles_entry(
    entry, store, known_work_entries, review_file
) -> None:
    found = ProviderPage.from_canonical_record(
        SCHEDULE,
        {"title": "Show", "sources": [SCHEDULE, KITSU_ONE]},
        searched_by_address=MAL,
    )
    store.add_pages(found, _page(KITSU_ONE), _page(KITSU_TWO, TARGET_MAL))
    result = _correct(entry, store, known_work_entries, review_file)
    assert [correction.placement_word for correction in result.corrections] == [
        "work 1",
        "other work",
    ]


def test_correct_entry_groups_renamed_page_points_to_current_page(
    store, review_file
) -> None:
    works = KnownWorkEntriesFromSeed([_seed_entry("Show", MAL, NEW_PLANET, OLD_PLANET)])
    store.add_pages(
        _page(OLD_PLANET, sources=[OLD_PLANET, NEW_PLANET]), _page(NEW_PLANET)
    )
    result = _correct(next(iter(works)), store, works, review_file)
    assert [
        (correction.page_key, correction.renamed_to, correction.reason)
        for correction in result.corrections
    ] == [
        ("animeplanet:anime:new-name", None, "SAME_PAGE"),
        ("animeplanet:anime:old-name", "animeplanet:anime:new-name", "SAME_PAGE"),
    ]


def test_correct_entry_groups_duplicate_page_goes_to_review(
    entry, store, known_work_entries, review_file
) -> None:
    store.add_pages(_page(KITSU_ONE, MAL), _page(KITSU_TWO, MAL))
    result = _correct(entry, store, known_work_entries, review_file)
    assert result.corrections == ()
    assert [case.reasons for case in result.review_cases] == [("DUPLICATE_PAGES",)]


def test_correct_entry_groups_accepted_answer_gives_corrections(
    entry, store, known_work_entries, tmp_path
) -> None:
    store.add_pages(_page(KITSU_ONE, MAL), _page(KITSU_TWO, MAL))
    path = tmp_path / "review.jsonl"
    [case] = _correct(entry, store, known_work_entries, ReviewFile(path)).review_cases
    answered = case.model_copy(
        update={"answer": {"kitsu:anime:1": "work 1", "kitsu:anime:2": "work 1, main"}}
    )
    path.write_text(answered.model_dump_json() + "\n", encoding="utf-8")
    result = _correct(entry, store, known_work_entries, ReviewFile(path))
    assert [case.answer for case in result.review_cases] == [answered.answer]
    assert [
        (correction.page_key, correction.is_main_page, correction.reason)
        for correction in result.corrections
    ] == [
        ("kitsu:anime:1", False, "review answer"),
        ("kitsu:anime:2", True, "review answer"),
    ]


def test_correct_entry_groups_part_page_answer_marks_corrections_part(
    entry, store, known_work_entries, tmp_path
) -> None:
    store.add_pages(
        _page(KITSU_ONE, MAL, episode_count=1), _page(KITSU_TWO, MAL, episode_count=4)
    )
    path = tmp_path / "review.jsonl"
    [case] = _correct(entry, store, known_work_entries, ReviewFile(path)).review_cases
    answered = case.model_copy(
        update={
            "answer": {"kitsu:anime:1": "work 1, part", "kitsu:anime:2": "work 1, part"}
        }
    )
    path.write_text(answered.model_dump_json() + "\n", encoding="utf-8")
    result = _correct(entry, store, known_work_entries, ReviewFile(path))
    assert (case.reasons, case.suggested_answer) == (
        ("PART_PAGES",),
        {
            "kitsu:anime:1": "work 1, part",
            "kitsu:anime:2": "work 1, part",
            "mal:anime:1": "work 1",
        },
    )
    assert [
        (correction.page_key, correction.is_part_page)
        for correction in result.corrections
    ] == [("kitsu:anime:1", True), ("kitsu:anime:2", True)]


def test_correct_entry_groups_answer_places_renamed_single_page_by_listed_key(
    store, tmp_path
) -> None:
    works = KnownWorkEntriesFromSeed(
        [_seed_entry("Show", MAL, KITSU_ONE, KITSU_TWO, OLD_PLANET)]
    )
    entry = next(iter(works))
    store.add_pages(
        _page(KITSU_ONE, MAL),
        _page(KITSU_TWO, MAL),
        _page(OLD_PLANET, sources=[OLD_PLANET, NEW_PLANET]),
    )
    path = tmp_path / "review.jsonl"
    [case] = _correct(entry, store, works, ReviewFile(path)).review_cases
    answer = {
        "kitsu:anime:1": "work 1",
        "kitsu:anime:2": "work 2",
        "mal:anime:1": "work 1",
        "animeplanet:anime:old-name": "work 2",
    }
    path.write_text(
        case.model_copy(update={"answer": answer}).model_dump_json() + "\n",
        encoding="utf-8",
    )
    corrections = _correct(entry, store, works, ReviewFile(path)).corrections
    assert [
        (correction.placement_word, correction.renamed_to)
        for correction in corrections
        if correction.page_key == "animeplanet:anime:old-name"
    ] == [("work 2", None)]


def test_correct_entry_groups_reference_page_in_later_work_moves_first_work_out(
    entry, store, known_work_entries, tmp_path
) -> None:
    store.add_pages(_page(KITSU_ONE, MAL), _page(KITSU_TWO, MAL))
    path = tmp_path / "review.jsonl"
    [case] = _correct(entry, store, known_work_entries, ReviewFile(path)).review_cases
    answer = {
        "kitsu:anime:1": "work 1",
        "kitsu:anime:2": "work 2",
        "mal:anime:1": "work 2",
    }
    path.write_text(
        case.model_copy(update={"answer": answer}).model_dump_json() + "\n",
        encoding="utf-8",
    )
    corrections = _correct(entry, store, known_work_entries, ReviewFile(path))
    assert [
        (correction.page_key, correction.new_entry_key)
        for correction in corrections.corrections
    ] == [
        ("mal:anime:1", None),
        ("kitsu:anime:1", "mal:anime:1 work 1"),
        ("kitsu:anime:2", None),
    ]


def test_correct_entry_groups_answer_keeps_renamed_page_when_current_not_listed(
    store, tmp_path
) -> None:
    other_planet = "https://www.anime-planet.com/anime/other-name"
    works = KnownWorkEntriesFromSeed(
        [_seed_entry("Show", MAL, OLD_PLANET, other_planet)]
    )
    entry = next(iter(works))
    store.add_pages(
        _page(OLD_PLANET, MAL, sources=[OLD_PLANET, NEW_PLANET]), _page(other_planet)
    )
    path = tmp_path / "review.jsonl"
    [case] = _correct(entry, store, works, ReviewFile(path)).review_cases
    answer = {
        "animeplanet:anime:new-name": "work 1",
        "animeplanet:anime:other-name": "other work",
    }
    path.write_text(
        case.model_copy(update={"answer": answer}).model_dump_json() + "\n",
        encoding="utf-8",
    )
    corrections = _correct(entry, store, works, ReviewFile(path)).corrections
    [corrected] = build_corrected_entries(corrections, works, store)[:1]
    assert "animeplanet:anime:old-name" in corrected.entry.addresses_by_page_key
    assert [
        (correction.placement_word, correction.renamed_to)
        for correction in corrections
        if correction.page_key == "animeplanet:anime:old-name"
    ] == [("work 1", None)]


def test_correct_entry_groups_answer_other_work_into_entry_with_same_provider_starts_new_entry(
    store, tmp_path
) -> None:
    works = KnownWorkEntriesFromSeed(
        [
            _seed_entry("Show", MAL, KITSU_ONE, KITSU_TWO),
            _seed_entry("Show Movie", TARGET_MAL, "https://kitsu.app/anime/99"),
        ]
    )
    entry = next(iter(works))
    store.add_pages(_page(KITSU_ONE, MAL), _page(KITSU_TWO, TARGET_MAL))
    path = tmp_path / "review.jsonl"
    [case] = _correct(entry, store, works, ReviewFile(path)).review_cases
    answer = {"kitsu:anime:1": "work 1", "kitsu:anime:2": "other work"}
    path.write_text(
        case.model_copy(update={"answer": answer}).model_dump_json() + "\n",
        encoding="utf-8",
    )
    corrections = _correct(entry, store, works, ReviewFile(path)).corrections
    assert [
        (correction.placement_word, correction.target_entry_reference_page)
        for correction in corrections
        if correction.page_key == "kitsu:anime:2"
    ] == [("other work", None)]


def test_correct_entry_groups_page_linking_work_outside_seed_goes_to_review(
    entry, store, known_work_entries, review_file
) -> None:
    store.add_pages(
        _page(KITSU_ONE, MAL), _page(KITSU_TWO, "https://myanimelist.net/anime/77")
    )
    group_corrections = _correct(entry, store, known_work_entries, review_file)
    assert group_corrections.corrections == ()
    assert [case.suggested_answer for case in group_corrections.review_cases] == [
        {
            "kitsu:anime:1": "work 1",
            "kitsu:anime:2": "other work",
            "mal:anime:1": "work 1",
        }
    ]


def test_correct_entry_groups_moved_page_details_differing_go_to_review(
    entry, store, known_work_entries, review_file
) -> None:
    store.add_pages(
        _page(KITSU_ONE, MAL), _page(KITSU_TWO, TARGET_MAL, type="OVA", year=1987)
    )
    review_cases = _correct(entry, store, known_work_entries, review_file).review_cases
    assert [case.reasons for case in review_cases] == [
        ("PAGES_OF_OTHER_WORKS", "MOVE_NOT_SURE")
    ]


CUE_ANN = "https://animenewsnetwork.com/encyclopedia/anime.php?id=24629"
CUE_MAL = "https://myanimelist.net/anime/43735"
CUE_KITSU = "https://kitsu.app/anime/44979"


@pytest.fixture
def cue_entries() -> KnownWorkEntriesFromSeed:
    return KnownWorkEntriesFromSeed(
        [_seed_entry("CUE!", CUE_ANN), _seed_entry("Cue!", CUE_MAL, CUE_KITSU)]
    )


def _answer_group_case(entries, store, review_file_path, answer):
    [case] = correct_entry_groups(
        [entries], store, KnownWorkEntriesFromSeed([]), ReviewFile(review_file_path)
    ).review_cases
    review_file_path.write_text(
        case.model_copy(update={"answer": answer}).model_dump_json() + "\n",
        encoding="utf-8",
    )
    return correct_entry_groups(
        [entries], store, KnownWorkEntriesFromSeed([]), ReviewFile(review_file_path)
    )


def test_correct_entry_groups_group_without_doubled_provider_goes_to_review(
    cue_entries, store, review_file
) -> None:
    store.add_pages(_page(CUE_MAL, CUE_ANN))
    placed = correct_entry_groups([list(cue_entries)], store, cue_entries, review_file)
    assert placed.corrections == ()
    assert [
        (case.case_name, case.suggested_answer) for case in placed.review_cases
    ] == [
        (
            "split work: ann:anime:24629 kitsu:anime:44979 mal:anime:43735",
            {
                "ann:anime:24629": "work 1",
                "mal:anime:43735": "work 1",
                "kitsu:anime:44979": "work 1",
            },
        )
    ]


def test_correct_entry_groups_accepted_merge_moves_pages_into_surviving_entry(
    cue_entries, store, tmp_path
) -> None:
    store.add_pages(_page(CUE_MAL, CUE_ANN))
    answer = dict.fromkeys(
        ["ann:anime:24629", "mal:anime:43735", "kitsu:anime:44979"], "work 1"
    )
    corrections = _answer_group_case(
        list(cue_entries), store, tmp_path / "review.jsonl", answer
    ).corrections
    assert [
        (
            correction.page_key,
            correction.target_entry_reference_page,
            correction.new_entry_key,
        )
        for correction in corrections
    ] == [
        ("ann:anime:24629", "mal:anime:43735", None),
        ("mal:anime:43735", None, None),
        ("kitsu:anime:44979", None, None),
    ]
    assert [
        corrected.entry
        for corrected in build_corrected_entries(corrections, cue_entries, store)
        if corrected.seed_entry_reference_page == "ann:anime:24629"
    ] == [None]


def test_correct_entry_groups_later_work_survived_by_other_entry_takes_page(
    cue_entries, store, tmp_path
) -> None:
    answer = {
        "ann:anime:24629": "work 2",
        "mal:anime:43735": "work 1",
        "kitsu:anime:44979": "work 2",
    }
    corrections = _answer_group_case(
        list(cue_entries), store, tmp_path / "review.jsonl", answer
    ).corrections
    assert [
        (correction.page_key, correction.target_entry_reference_page)
        for correction in corrections
    ] == [
        ("ann:anime:24629", None),
        ("mal:anime:43735", None),
        ("kitsu:anime:44979", "ann:anime:24629"),
    ]


def test_correct_entry_groups_later_work_without_survivor_named_by_group(
    cue_entries, store, tmp_path
) -> None:
    answer = {
        "ann:anime:24629": "work 1",
        "mal:anime:43735": "work 1",
        "kitsu:anime:44979": "work 2",
    }
    corrections = _answer_group_case(
        list(cue_entries), store, tmp_path / "review.jsonl", answer
    ).corrections
    assert [
        (correction.page_key, correction.new_entry_key)
        for correction in corrections
        if correction.new_entry_key
    ] == [("kitsu:anime:44979", "mal:anime:43735 work 2")]


def test_correct_entry_groups_reference_page_marked_part_keeps_entry_whole(
    store, tmp_path
) -> None:
    planet = "https://www.anime-planet.com/anime/show"
    works = KnownWorkEntriesFromSeed(
        [_seed_entry("Show", MAL, KITSU_ONE, KITSU_TWO, planet)]
    )
    store.add_pages(_page(KITSU_ONE, MAL), _page(KITSU_TWO, MAL))
    path = tmp_path / "review.jsonl"
    entry = next(iter(works))
    [case] = _correct(entry, store, works, ReviewFile(path)).review_cases
    answer = dict.fromkeys(
        ["kitsu:anime:1", "kitsu:anime:2", "mal:anime:1"], "work 1, part"
    )
    path.write_text(
        case.model_copy(update={"answer": answer}).model_dump_json() + "\n",
        encoding="utf-8",
    )
    corrections = _correct(entry, store, works, ReviewFile(path)).corrections
    [corrected] = build_corrected_entries(corrections, works, store)
    assert corrected.entry is not None
    assert list(corrected.entry.addresses_by_page_key) == [
        "mal:anime:1",
        "kitsu:anime:1",
        "kitsu:anime:2",
        "animeplanet:anime:show",
    ]
