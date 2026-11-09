import pytest
from seed_cleaning.corrected_entries import (
    build_corrected_entries,
)
from seed_cleaning.corrected_seed import build_corrected_seed
from seed_cleaning.entry_groups import EntryGroup
from seed_cleaning.group_placement import correct_entry_groups
from seed_cleaning.seed_corrections import SeedCorrection
from work_matching import (
    KnownWorkEntriesFromSeed,
    PlacementRule,
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
    return ProviderPage.read_canonical_record(
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
    return correct_entry_groups(
        [EntryGroup((entry,))], store, known_work_entries, review_file
    )


def test_correct_entry_groups_entry_without_review_moves_page_to_seed_entry(
    entry, store, known_work_entries, review_file
) -> None:
    store.add_pages(_page(KITSU_ONE, MAL), _page(KITSU_TWO, TARGET_MAL))
    placement_corrections = _correct(entry, store, known_work_entries, review_file)
    assert placement_corrections.review_cases == ()
    assert placement_corrections.corrections == (
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
    found_page = ProviderPage.read_canonical_record(
        SCHEDULE,
        {"title": "Show", "sources": [SCHEDULE, KITSU_ONE]},
        searched_by_address=MAL,
    )
    store.add_pages(found_page, _page(KITSU_ONE), _page(KITSU_TWO, TARGET_MAL))
    placement_corrections = _correct(entry, store, known_work_entries, review_file)
    assert [
        correction.placement_word for correction in placement_corrections.corrections
    ] == [
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
    placement_corrections = _correct(next(iter(works)), store, works, review_file)
    assert [
        (correction.page_key, correction.renamed_to, correction.reason)
        for correction in placement_corrections.corrections
    ] == [
        ("animeplanet:anime:new-name", None, "SAME_PAGE"),
        ("animeplanet:anime:old-name", "animeplanet:anime:new-name", "SAME_PAGE"),
    ]


def test_correct_entry_groups_duplicate_page_goes_to_review(
    entry, store, known_work_entries, review_file
) -> None:
    store.add_pages(_page(KITSU_ONE, MAL), _page(KITSU_TWO, MAL))
    placement_corrections = _correct(entry, store, known_work_entries, review_file)
    assert placement_corrections.corrections == ()
    assert [case.reasons for case in placement_corrections.review_cases] == [
        ("DUPLICATE_PAGES",)
    ]


def test_correct_entry_groups_accepted_answer_gives_corrections(
    entry, store, known_work_entries, tmp_path
) -> None:
    store.add_pages(_page(KITSU_ONE, MAL), _page(KITSU_TWO, MAL))
    path = tmp_path / "review.jsonl"
    [case] = _correct(entry, store, known_work_entries, ReviewFile(path)).review_cases
    answered_case = case.model_copy(
        update={"answer": {"kitsu:anime:1": "work 1", "kitsu:anime:2": "dropped"}}
    )
    path.write_text(answered_case.model_dump_json() + "\n", encoding="utf-8")
    placement_corrections = _correct(entry, store, known_work_entries, ReviewFile(path))
    assert [case.answer for case in placement_corrections.review_cases] == [
        answered_case.answer
    ]
    assert [
        (correction.page_key, correction.is_dropped_page, correction.reason)
        for correction in placement_corrections.corrections
    ] == [
        ("kitsu:anime:1", False, "review answer"),
        ("kitsu:anime:2", True, "review answer"),
    ]


def test_correct_entry_groups_split_answer_records_partial_source(
    entry, store, known_work_entries, tmp_path
) -> None:
    store.add_pages(
        _page(KITSU_ONE, MAL, episode_count=1), _page(KITSU_TWO, MAL, episode_count=4)
    )
    path = tmp_path / "review.jsonl"
    [case] = _correct(entry, store, known_work_entries, ReviewFile(path)).review_cases
    answered_case = case.model_copy(
        update={
            "answer": {
                "kitsu:anime:1": "work 1",
                "kitsu:anime:2": "work 2",
                "mal:anime:1": "work 1, partial in work 2",
            }
        }
    )
    path.write_text(answered_case.model_dump_json() + "\n", encoding="utf-8")
    placement_corrections = _correct(entry, store, known_work_entries, ReviewFile(path))
    assert (case.reasons, case.suggested_answer) == (
        ("PART_PAGES",),
        {"kitsu:anime:1": "work 1", "kitsu:anime:2": "work 2"},
    )
    assert [
        (
            correction.page_key,
            correction.new_entry_key,
            correction.partial_new_entry_keys,
        )
        for correction in placement_corrections.corrections
    ] == [
        ("mal:anime:1", None, ("mal:anime:1 work 2",)),
        ("kitsu:anime:1", None, ()),
        ("kitsu:anime:2", "mal:anime:1 work 2", ()),
    ]


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
    [corrected_entry] = build_corrected_entries(corrections, works, store)[:1]
    assert "animeplanet:anime:old-name" in corrected_entry.entry.addresses_by_page_key
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
        [EntryGroup(tuple(entries))],
        store,
        KnownWorkEntriesFromSeed([]),
        ReviewFile(review_file_path),
    ).review_cases
    review_file_path.write_text(
        case.model_copy(update={"answer": answer}).model_dump_json() + "\n",
        encoding="utf-8",
    )
    return correct_entry_groups(
        [EntryGroup(tuple(entries))],
        store,
        KnownWorkEntriesFromSeed([]),
        ReviewFile(review_file_path),
    )


def test_correct_entry_groups_group_without_doubled_provider_goes_to_review(
    cue_entries, store, review_file
) -> None:
    store.add_pages(_page(CUE_MAL, CUE_ANN))
    placement_corrections = correct_entry_groups(
        [EntryGroup(tuple(cue_entries))], store, cue_entries, review_file
    )
    assert placement_corrections.corrections == ()
    assert [
        (case.case_name, case.suggested_answer)
        for case in placement_corrections.review_cases
    ] == [
        (
            "split work: ann:anime:24629 kitsu:anime:44979 mal:anime:43735",
            {
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
        corrected_entry.entry
        for corrected_entry in build_corrected_entries(corrections, cue_entries, store)
        if corrected_entry.seed_entry_reference_page == "ann:anime:24629"
    ] == [None]


def test_correct_entry_groups_previous_answer_keeps_added_entry_out_of_works(
    cue_entries, store, tmp_path
) -> None:
    review_file_path = tmp_path / "review.jsonl"
    _answer_group_case(
        list(cue_entries),
        store,
        review_file_path,
        {
            "ann:anime:24629": "work 1",
            "mal:anime:43735": "work 2",
            "kitsu:anime:44979": "work 2",
        },
    )
    added_entry = next(
        iter(
            KnownWorkEntriesFromSeed(
                [_seed_entry("Cue!", "https://anilist.co/anime/7")]
            )
        )
    )
    placement_corrections = correct_entry_groups(
        [EntryGroup((*cue_entries, added_entry))],
        store,
        KnownWorkEntriesFromSeed([]),
        ReviewFile(review_file_path),
    )
    [case] = placement_corrections.review_cases
    assert case.previous_answer is not None
    assert case.previous_answer.is_applied
    assert [
        (
            correction.page_key,
            correction.target_entry_reference_page,
            correction.new_entry_key,
        )
        for correction in placement_corrections.corrections
    ] == [
        ("ann:anime:24629", None, None),
        ("mal:anime:43735", None, None),
        ("kitsu:anime:44979", None, None),
    ]


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


def test_correct_entry_groups_split_answer_gives_each_work_its_entry(
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
    answer = {
        "kitsu:anime:1": "work 1",
        "kitsu:anime:2": "work 2",
        "mal:anime:1": "work 1, partial in work 2",
        "animeplanet:anime:show": "work 1",
    }
    path.write_text(
        case.model_copy(update={"answer": answer}).model_dump_json() + "\n",
        encoding="utf-8",
    )
    corrections = _correct(entry, store, works, ReviewFile(path)).corrections
    assert [
        (
            list(corrected_entry.entry.addresses_by_page_key),
            list(corrected_entry.entry.partial_addresses_by_page_key),
        )
        for corrected_entry in build_corrected_entries(corrections, works, store)
        if corrected_entry.entry is not None
    ] == [
        (["mal:anime:1", "kitsu:anime:1", "animeplanet:anime:show"], []),
        (["kitsu:anime:2"], ["mal:anime:1"]),
    ]


def test_correct_entry_groups_renamed_reference_page_in_later_work_keeps_entry(
    store, tmp_path
) -> None:
    ann_one = "https://animenewsnetwork.com/encyclopedia/anime.php?id=1"
    ann_two = "https://animenewsnetwork.com/encyclopedia/anime.php?id=2"
    simkl = "https://simkl.com/anime/1"
    works = KnownWorkEntriesFromSeed(
        [_seed_entry("Show", OLD_PLANET, NEW_PLANET, ann_one, ann_two, simkl)]
    )
    entry = next(iter(works))
    store.add_pages(
        _page(OLD_PLANET, sources=[OLD_PLANET, NEW_PLANET]), _page(NEW_PLANET)
    )
    path = tmp_path / "review.jsonl"
    [case] = _correct(entry, store, works, ReviewFile(path)).review_cases
    answer = {
        "animeplanet:anime:new-name": "work 2",
        "ann:anime:1": "work 1",
        "ann:anime:2": "work 2",
        "simkl:anime:1": "work 1",
    }
    path.write_text(
        case.model_copy(update={"answer": answer}).model_dump_json() + "\n",
        encoding="utf-8",
    )
    corrections = _correct(entry, store, works, ReviewFile(path)).corrections
    assert [
        (
            corrected_entry.seed_entry_reference_page,
            None
            if corrected_entry.entry is None
            else list(corrected_entry.entry.addresses_by_page_key),
        )
        for corrected_entry in build_corrected_entries(corrections, works, store)
    ] == [
        ("animeplanet:anime:old-name", ["animeplanet:anime:new-name", "ann:anime:2"]),
        (None, ["ann:anime:1", "simkl:anime:1"]),
    ]


def test_correct_entry_groups_answer_with_two_mal_pages_in_one_work_left_open(
    store, tmp_path
) -> None:
    works = KnownWorkEntriesFromSeed(
        [_seed_entry("Show", MAL), _seed_entry("Show", TARGET_MAL)]
    )
    path = tmp_path / "review.jsonl"
    [case] = correct_entry_groups(
        [EntryGroup(tuple(works))], store, works, ReviewFile(path)
    ).review_cases
    answer = {"mal:anime:1": "work 1", "mal:anime:9": "work 1"}
    path.write_text(
        case.model_copy(update={"answer": answer}).model_dump_json() + "\n",
        encoding="utf-8",
    )
    placement_corrections = correct_entry_groups(
        [EntryGroup(tuple(works))], store, works, ReviewFile(path)
    )
    assert placement_corrections.corrections == ()
    assert [case.accepted_answer for case in placement_corrections.review_cases] == [
        None
    ]


def test_correct_entry_groups_group_case_carries_evidence_and_shared_link_warning(
    cue_entries, store, review_file
) -> None:
    store.add_pages(_page(CUE_MAL, CUE_ANN))
    entry_group = EntryGroup(
        tuple(cue_entries),
        ("mal:anime:43735 links ann:anime:24629",),
        has_shared_tying_link=True,
    )
    [case] = correct_entry_groups(
        [entry_group], store, cue_entries, review_file
    ).review_cases
    assert case.candidate_evidence == ("mal:anime:43735 links ann:anime:24629",)
    assert "LINK_SHARED_WITH_OTHER_ENTRIES" in case.reasons


def test_correct_entry_groups_corrected_seed_names_pages_by_origin_and_entries_by_identity(
    cue_entries, store, tmp_path
) -> None:
    first_pass_corrections = [
        SeedCorrection(
            entry_reference_page="mal:anime:43735",
            page_key="kitsu:anime:44979",
            address=CUE_KITSU,
            placement_word="work 2",
            new_entry_key="mal:anime:43735 work 2",
            reason="review answer",
        )
    ]
    corrected_seed = build_corrected_seed(first_pass_corrections, cue_entries, store)
    corrected_entries = corrected_seed.known_work_entries
    group_entries = tuple(
        corrected_entries.find_by_page_key(page_key)
        for page_key in ("ann:anime:24629", "kitsu:anime:44979")
    )
    path = tmp_path / "review.jsonl"
    [case] = correct_entry_groups(
        [EntryGroup(group_entries)],
        store,
        corrected_entries,
        ReviewFile(path),
        corrected_seed,
    ).review_cases
    answer = {"ann:anime:24629": "work 1", "kitsu:anime:44979": "work 1"}
    path.write_text(
        case.model_copy(update={"answer": answer}).model_dump_json() + "\n",
        encoding="utf-8",
    )
    corrections = correct_entry_groups(
        [EntryGroup(group_entries)],
        store,
        corrected_entries,
        ReviewFile(path),
        corrected_seed,
    ).corrections
    assert [
        (
            correction.entry_reference_page,
            correction.page_key,
            correction.target_entry_reference_page,
            correction.new_entry_key,
        )
        for correction in corrections
    ] == [
        ("ann:anime:24629", "ann:anime:24629", None, "mal:anime:43735 work 2"),
        ("mal:anime:43735", "kitsu:anime:44979", None, "mal:anime:43735 work 2"),
    ]


SAINT_SEIYA_SEED_ENTRIES = [
    {
        "title": "Saint Seiya: Knights of the Zodiac",
        "type": "ONA",
        "animeSeason": {"season": "SUMMER", "year": 2019},
        "sources": [
            "https://anidb.net/anime/13347",
            "https://anilist.co/anime/102427",
            "https://kitsu.app/anime/13695",
            "https://livechart.me/anime/2822",
            "https://myanimelist.net/anime/36040",
        ],
    },
    {
        "title": "Saint Seiya: Knights of the Zodiac Part 2",
        "type": "ONA",
        "animeSeason": {"season": "WINTER", "year": 2020},
        "sources": [
            "https://anilist.co/anime/114448",
            "https://livechart.me/anime/9705",
        ],
    },
]


def test_correct_entry_groups_split_answer_keeps_parts_apart_with_partial_sources(
    store, tmp_path
) -> None:
    works = KnownWorkEntriesFromSeed(SAINT_SEIYA_SEED_ENTRIES)
    answer = {
        "anidb:anime:13347": "work 1, partial in work 2",
        "anilist:anime:102427": "work 1",
        "kitsu:anime:13695": "work 1, partial in work 2",
        "livechart:anime:2822": "work 1",
        "mal:anime:36040": "work 1, partial in work 2",
        "anilist:anime:114448": "work 2",
        "livechart:anime:9705": "work 2",
    }
    corrections = _answer_group_case(
        list(works), store, tmp_path / "review.jsonl", answer
    ).corrections
    assert {
        correction.page_key: (
            correction.target_entry_reference_page,
            correction.partial_entry_reference_pages,
        )
        for correction in corrections
        if correction.page_key.startswith(("anilist", "mal"))
    } == {
        "anilist:anime:102427": (None, ()),
        "mal:anime:36040": (None, ("anilist:anime:114448",)),
        "anilist:anime:114448": (None, ()),
    }


@pytest.fixture
def series_works() -> KnownWorkEntriesFromSeed:
    return KnownWorkEntriesFromSeed(
        [
            _seed_entry("Show", MAL, KITSU_ONE, KITSU_TWO),
            _seed_entry("Show Movie", TARGET_MAL),
        ]
    )


def _answer_named_entry_case(works, store, path, named_page_key):
    entry = next(iter(works))
    store.add_pages(_page(KITSU_ONE, MAL), _page(KITSU_TWO, MAL))
    [case] = _correct(entry, store, works, ReviewFile(path)).review_cases
    answer = {
        "kitsu:anime:1": "work 1",
        "kitsu:anime:2": f"other work {named_page_key}",
    }
    path.write_text(
        case.model_copy(update={"answer": answer}).model_dump_json() + "\n",
        encoding="utf-8",
    )
    return _correct(entry, store, works, ReviewFile(path))


def test_correct_entry_groups_answer_naming_seed_entry_moves_page_there(
    series_works, store, tmp_path
) -> None:
    corrections = _answer_named_entry_case(
        series_works, store, tmp_path / "review.jsonl", "mal:anime:9"
    ).corrections
    assert [
        (
            correction.placement_word,
            correction.target_entry_reference_page,
            correction.new_entry_key,
            correction.named_entry_page_key,
        )
        for correction in corrections
        if correction.page_key == "kitsu:anime:2"
    ] == [("other work", "mal:anime:9", None, "mal:anime:9")]


def test_correct_entry_groups_second_pass_answer_naming_moved_page_joins_its_new_entry(
    series_works, store, tmp_path
) -> None:
    corrected_seed = build_corrected_seed(
        [
            SeedCorrection(
                entry_reference_page="mal:anime:9",
                page_key="mal:anime:9",
                address=TARGET_MAL,
                placement_word="other work",
                new_entry_key="mal:anime:9 mal:anime:9",
                reason="review answer",
            )
        ],
        series_works,
        store,
    )
    corrected_works = corrected_seed.known_work_entries
    entry = corrected_works.find_by_page_key("mal:anime:1")
    store.add_pages(_page(KITSU_ONE, MAL), _page(KITSU_TWO, MAL))
    path = tmp_path / "review.jsonl"
    [case] = correct_entry_groups(
        [EntryGroup((entry,))], store, corrected_works, ReviewFile(path), corrected_seed
    ).review_cases
    answer = {"kitsu:anime:1": "work 1", "kitsu:anime:2": "other work mal:anime:9"}
    path.write_text(
        case.model_copy(update={"answer": answer}).model_dump_json() + "\n",
        encoding="utf-8",
    )
    corrections = correct_entry_groups(
        [EntryGroup((entry,))], store, corrected_works, ReviewFile(path), corrected_seed
    ).corrections
    assert [
        (correction.target_entry_reference_page, correction.new_entry_key)
        for correction in corrections
        if correction.page_key == "kitsu:anime:2"
    ] == [(None, "mal:anime:9 mal:anime:9")]


def test_correct_entry_groups_second_pass_dropped_moved_in_page_goes_nowhere(
    store, tmp_path
) -> None:
    works = KnownWorkEntriesFromSeed(
        [
            _seed_entry("Show", MAL, KITSU_ONE),
            _seed_entry("Show Movie", TARGET_MAL, KITSU_TWO),
        ]
    )
    corrected_seed = build_corrected_seed(
        [
            SeedCorrection(
                entry_reference_page="mal:anime:9",
                page_key="kitsu:anime:2",
                address=KITSU_TWO,
                placement_word="other work",
                target_entry_reference_page="mal:anime:1",
                reason="review answer",
            )
        ],
        works,
        store,
    )
    corrected_works = corrected_seed.known_work_entries
    entry = corrected_works.find_by_page_key("mal:anime:1")
    store.add_pages(_page(KITSU_ONE, MAL), _page(KITSU_TWO, MAL))
    path = tmp_path / "review.jsonl"
    [case] = correct_entry_groups(
        [EntryGroup((entry,))], store, corrected_works, ReviewFile(path), corrected_seed
    ).review_cases
    answer = {"kitsu:anime:1": "work 1", "kitsu:anime:2": "dropped"}
    path.write_text(
        case.model_copy(update={"answer": answer}).model_dump_json() + "\n",
        encoding="utf-8",
    )
    corrections = correct_entry_groups(
        [EntryGroup((entry,))], store, corrected_works, ReviewFile(path), corrected_seed
    ).corrections
    assert [
        (
            correction.is_dropped_page,
            correction.target_entry_reference_page,
            correction.new_entry_key,
        )
        for correction in corrections
        if correction.page_key == "kitsu:anime:2"
    ] == [(True, None, None)]


def test_correct_entry_groups_answer_naming_entry_outside_seed_left_open(
    series_works, store, tmp_path
) -> None:
    placement_corrections = _answer_named_entry_case(
        series_works, store, tmp_path / "review.jsonl", "mal:anime:77"
    )
    [case] = placement_corrections.review_cases
    assert placement_corrections.corrections == ()
    assert case.list_broken_answer_rules(case.answer) == (
        PlacementRule.UNKNOWN_NAMED_ENTRY,
    )


def test_correct_entry_groups_case_suggests_series_entries(
    series_works, store, review_file
) -> None:
    store.add_pages(_page(KITSU_ONE, MAL), _page(KITSU_TWO, MAL))
    [case] = _correct(
        next(iter(series_works)), store, series_works, review_file
    ).review_cases
    assert [
        (suggestion.entry_reference_page, suggestion.title)
        for suggestion in case.target_entry_suggestions
    ] == [("mal:anime:9", "Show Movie")]
