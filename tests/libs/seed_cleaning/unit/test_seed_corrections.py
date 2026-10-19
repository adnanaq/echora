import json

import pytest
from seed_cleaning.seed_corrections import (
    CorrectedEntry,
    SeedCorrection,
    UnknownSeedEntryError,
    build_corrected_entries,
    correct_doubled_provider_entries,
    write_corrections,
)
from work_matching import (
    KnownWorksFromSeed,
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


def _page(address: str, *links: str, **details) -> ProviderPage:
    return ProviderPage.from_canonical_record(
        address,
        {"title": "Show", "external_sources": [{"source": link} for link in links]}
        | details,
    )


@pytest.fixture
def known_works() -> KnownWorksFromSeed:
    return KnownWorksFromSeed(
        [
            _seed_entry("Show", MAL, KITSU_ONE, KITSU_TWO),
            _seed_entry("Show Movie", TARGET_MAL),
        ]
    )


@pytest.fixture
def entry(known_works) -> WorkEntry:
    return next(iter(known_works))


@pytest.fixture
def store(tmp_path) -> ProviderPageStore:
    return ProviderPageStore(tmp_path / "provider_pages.jsonl")


@pytest.fixture
def review_file(tmp_path) -> ReviewFile:
    return ReviewFile(tmp_path / "review.jsonl")


def _correct(entry, store, known_works, review_file):
    return correct_doubled_provider_entries([entry], store, known_works, review_file)


def test_correct_doubled_provider_entries_settled_entry_moves_page_to_seed_entry(
    entry, store, known_works, review_file
) -> None:
    store.add_pages(_page(KITSU_ONE, MAL), _page(KITSU_TWO, TARGET_MAL))
    result = _correct(entry, store, known_works, review_file)
    assert result.review_cases == ()
    assert result.corrections == (
        SeedCorrection(
            entry_reference_page="mal:anime:1",
            page_key="kitsu:anime:1",
            address=KITSU_ONE,
            work="entry",
            reason="ONE_LINKED",
        ),
        SeedCorrection(
            entry_reference_page="mal:anime:1",
            page_key="kitsu:anime:2",
            address=KITSU_TWO,
            work="other work",
            target_entry_reference_page="mal:anime:9",
            reason="ONE_LINKED",
        ),
    )


def test_correct_doubled_provider_entries_link_on_found_page_settles_entry(
    entry, store, known_works, review_file
) -> None:
    found = ProviderPage.from_canonical_record(
        SCHEDULE, {"title": "Show", "sources": [SCHEDULE, KITSU_ONE]}, found_by=MAL
    )
    store.add_pages(found, _page(KITSU_ONE), _page(KITSU_TWO, TARGET_MAL))
    result = _correct(entry, store, known_works, review_file)
    assert [correction.work for correction in result.corrections] == [
        "entry",
        "other work",
    ]


def test_correct_doubled_provider_entries_renamed_page_points_to_current_page(
    store, review_file
) -> None:
    works = KnownWorksFromSeed([_seed_entry("Show", MAL, NEW_PLANET, OLD_PLANET)])
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


def test_correct_doubled_provider_entries_duplicate_page_goes_to_review(
    entry, store, known_works, review_file
) -> None:
    store.add_pages(_page(KITSU_ONE, MAL), _page(KITSU_TWO, MAL))
    result = _correct(entry, store, known_works, review_file)
    assert result.corrections == ()
    assert [case.reasons for case in result.review_cases] == [("SHARED_LINK",)]


def test_correct_doubled_provider_entries_accepted_answer_gives_corrections(
    entry, store, known_works, tmp_path
) -> None:
    store.add_pages(_page(KITSU_ONE, MAL), _page(KITSU_TWO, MAL))
    path = tmp_path / "review.jsonl"
    [case] = _correct(entry, store, known_works, ReviewFile(path)).review_cases
    answered = case.model_copy(
        update={"answer": {"kitsu:anime:1": "entry", "kitsu:anime:2": "entry, main"}}
    )
    path.write_text(answered.model_dump_json() + "\n", encoding="utf-8")
    result = _correct(entry, store, known_works, ReviewFile(path))
    assert [case.answer for case in result.review_cases] == [answered.answer]
    assert [
        (correction.page_key, correction.is_main_page, correction.reason)
        for correction in result.corrections
    ] == [
        ("kitsu:anime:1", False, "review answer"),
        ("kitsu:anime:2", True, "review answer"),
    ]


def test_correct_doubled_provider_entries_answer_places_renamed_single_page_by_listed_key(
    store, tmp_path
) -> None:
    works = KnownWorksFromSeed(
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
        "kitsu:anime:1": "entry",
        "kitsu:anime:2": "work 2",
        "mal:anime:1": "entry",
        "animeplanet:anime:old-name": "work 2",
    }
    path.write_text(
        case.model_copy(update={"answer": answer}).model_dump_json() + "\n",
        encoding="utf-8",
    )
    corrections = _correct(entry, store, works, ReviewFile(path)).corrections
    assert [
        (correction.work, correction.renamed_to)
        for correction in corrections
        if correction.page_key == "animeplanet:anime:old-name"
    ] == [("work 2", None)]


def test_correct_doubled_provider_entries_answer_keeps_renamed_page_when_current_not_listed(
    store, tmp_path
) -> None:
    other_planet = "https://www.anime-planet.com/anime/other-name"
    works = KnownWorksFromSeed([_seed_entry("Show", MAL, OLD_PLANET, other_planet)])
    entry = next(iter(works))
    store.add_pages(
        _page(OLD_PLANET, MAL, sources=[OLD_PLANET, NEW_PLANET]), _page(other_planet)
    )
    path = tmp_path / "review.jsonl"
    [case] = _correct(entry, store, works, ReviewFile(path)).review_cases
    answer = {
        "animeplanet:anime:new-name": "entry",
        "animeplanet:anime:other-name": "other work",
    }
    path.write_text(
        case.model_copy(update={"answer": answer}).model_dump_json() + "\n",
        encoding="utf-8",
    )
    corrections = _correct(entry, store, works, ReviewFile(path)).corrections
    [corrected] = build_corrected_entries(corrections, works, store)[:1]
    assert "animeplanet:anime:old-name" in corrected.entry.pages
    assert [
        (correction.work, correction.renamed_to)
        for correction in corrections
        if correction.page_key == "animeplanet:anime:old-name"
    ] == [("entry", None)]


def test_correct_doubled_provider_entries_answer_other_work_into_entry_with_same_provider_starts_new_entry(
    store, tmp_path
) -> None:
    works = KnownWorksFromSeed(
        [
            _seed_entry("Show", MAL, KITSU_ONE, KITSU_TWO),
            _seed_entry("Show Movie", TARGET_MAL, "https://kitsu.app/anime/99"),
        ]
    )
    entry = next(iter(works))
    store.add_pages(_page(KITSU_ONE, MAL), _page(KITSU_TWO, TARGET_MAL))
    path = tmp_path / "review.jsonl"
    [case] = _correct(entry, store, works, ReviewFile(path)).review_cases
    answer = {"kitsu:anime:1": "entry", "kitsu:anime:2": "other work"}
    path.write_text(
        case.model_copy(update={"answer": answer}).model_dump_json() + "\n",
        encoding="utf-8",
    )
    corrections = _correct(entry, store, works, ReviewFile(path)).corrections
    assert [
        (correction.work, correction.target_entry_reference_page)
        for correction in corrections
        if correction.page_key == "kitsu:anime:2"
    ] == [("other work", None)]


def test_correct_doubled_provider_entries_page_linking_work_outside_seed_goes_to_review(
    entry, store, known_works, review_file
) -> None:
    store.add_pages(
        _page(KITSU_ONE, MAL), _page(KITSU_TWO, "https://myanimelist.net/anime/77")
    )
    doubled_provider_corrections = _correct(entry, store, known_works, review_file)
    assert doubled_provider_corrections.corrections == ()
    assert [
        case.suggested_answer for case in doubled_provider_corrections.review_cases
    ] == [
        {
            "kitsu:anime:1": "entry",
            "kitsu:anime:2": "other work",
            "mal:anime:1": "entry",
        }
    ]


def test_correct_doubled_provider_entries_moved_page_details_differing_go_to_review(
    entry, store, known_works, review_file
) -> None:
    store.add_pages(
        _page(KITSU_ONE, MAL), _page(KITSU_TWO, TARGET_MAL, type="OVA", year=1987)
    )
    review_cases = _correct(entry, store, known_works, review_file).review_cases
    assert [case.reasons for case in review_cases] == [("ONE_LINKED",)]


def test_build_corrected_entries_moved_page_leaves_entry_and_joins_seed_entry(
    entry, store, known_works, review_file
) -> None:
    store.add_pages(_page(KITSU_ONE, MAL), _page(KITSU_TWO, TARGET_MAL))
    corrections = _correct(entry, store, known_works, review_file).corrections
    assert build_corrected_entries(corrections, known_works, store) == [
        CorrectedEntry(
            seed_entry_reference_page="mal:anime:1",
            entry=entry.model_copy(
                update={"pages": {"mal:anime:1": MAL, "kitsu:anime:1": KITSU_ONE}}
            ),
        ),
        CorrectedEntry(
            seed_entry_reference_page="mal:anime:9",
            entry=WorkEntry(
                title="Show Movie",
                type="TV",
                year=2020,
                season="SPRING",
                pages={"mal:anime:9": TARGET_MAL, "kitsu:anime:2": KITSU_TWO},
            ),
        ),
    ]


def test_build_corrected_entries_page_moved_into_entry_corrected_itself_joins_corrected_entry(
    store, review_file
) -> None:
    works = KnownWorksFromSeed(
        [
            _seed_entry("Show", MAL, KITSU_ONE, KITSU_TWO),
            _seed_entry("Show Movie", TARGET_MAL, NEW_PLANET, OLD_PLANET),
        ]
    )
    store.add_pages(
        _page(KITSU_ONE, MAL),
        _page(KITSU_TWO, TARGET_MAL),
        _page(OLD_PLANET, sources=[OLD_PLANET, NEW_PLANET]),
        _page(NEW_PLANET),
    )
    corrections = correct_doubled_provider_entries(
        list(works), store, works, review_file
    ).corrections
    corrected = build_corrected_entries(corrections, works, store)
    [target] = [
        corrected_entry.entry
        for corrected_entry in corrected
        if corrected_entry.seed_entry_reference_page == "mal:anime:9"
    ]
    assert list(target.pages) == [
        "mal:anime:9",
        "animeplanet:anime:new-name",
        "kitsu:anime:2",
    ]


def test_build_corrected_entries_renamed_old_address_dropped(
    store, review_file
) -> None:
    works = KnownWorksFromSeed([_seed_entry("Show", MAL, NEW_PLANET, OLD_PLANET)])
    store.add_pages(
        _page(OLD_PLANET, sources=[OLD_PLANET, NEW_PLANET]), _page(NEW_PLANET)
    )
    corrections = _correct(next(iter(works)), store, works, review_file).corrections
    [corrected] = build_corrected_entries(corrections, works, store)
    assert list(corrected.entry.pages) == ["mal:anime:1", "animeplanet:anime:new-name"]


def test_build_corrected_entries_main_page_listed_first(
    entry, store, known_works
) -> None:
    corrections = [
        SeedCorrection(
            entry_reference_page="mal:anime:1",
            page_key=key,
            address=address,
            work="entry",
            is_main_page=key == "kitsu:anime:2",
            reason="review answer",
        )
        for key, address in (("kitsu:anime:1", KITSU_ONE), ("kitsu:anime:2", KITSU_TWO))
    ]
    [corrected] = build_corrected_entries(corrections, known_works, store)
    assert list(corrected.entry.pages) == [
        "kitsu:anime:2",
        "mal:anime:1",
        "kitsu:anime:1",
    ]


def test_build_corrected_entries_added_page_joins_entry(store, known_works) -> None:
    corrections = [
        SeedCorrection(
            entry_reference_page="mal:anime:9",
            page_key="anilist:anime:9",
            address="https://anilist.co/anime/9",
            work="entry",
            is_added_page=True,
            is_part_page=True,
            reason="review answer",
        )
    ]
    [corrected] = build_corrected_entries(corrections, known_works, store)
    assert (corrected.seed_entry_reference_page, corrected.entry.pages) == (
        "mal:anime:9",
        {"mal:anime:9": TARGET_MAL, "anilist:anime:9": "https://anilist.co/anime/9"},
    )


def test_build_corrected_entries_work_pages_become_new_entry_from_trusted_page(
    entry, store, known_works
) -> None:
    store.add_pages(
        _page(
            KITSU_TWO, title="Show: The Movie", type="MOVIE", year=2021, season="FALL"
        )
    )
    corrections = [
        SeedCorrection(
            entry_reference_page="mal:anime:1",
            page_key="kitsu:anime:2",
            address=KITSU_TWO,
            work="work 2",
            reason="review answer",
        )
    ]
    assert build_corrected_entries(corrections, known_works, store)[
        1
    ] == CorrectedEntry(
        seed_entry_reference_page=None,
        entry=WorkEntry(
            title="Show: The Movie",
            type="MOVIE",
            year=2021,
            season="FALL",
            pages={"kitsu:anime:2": KITSU_TWO},
        ),
    )


def test_build_corrected_entries_page_without_details_new_entry_takes_entry_title(
    entry, store, known_works, caplog
) -> None:
    ann = "https://animenewsnetwork.com/encyclopedia/anime.php?id=5"
    corrections = [
        SeedCorrection(
            entry_reference_page="mal:anime:1",
            page_key="ann:anime:5",
            address=ann,
            work="other work",
            reason="review answer",
        )
    ]
    assert build_corrected_entries(corrections, known_works, store)[
        1
    ] == CorrectedEntry(
        seed_entry_reference_page=None,
        entry=WorkEntry(title="Show", type="UNKNOWN", pages={"ann:anime:5": ann}),
    )
    assert "ann:anime:5" in caplog.text


def test_build_corrected_entries_entry_missing_from_seed_raises(
    store, known_works
) -> None:
    correction = SeedCorrection(
        entry_reference_page="mal:anime:404",
        page_key="kitsu:anime:1",
        address=KITSU_ONE,
        work="other work",
        reason="review answer",
    )
    with pytest.raises(UnknownSeedEntryError):
        build_corrected_entries([correction], known_works, store)


def test_write_corrections_same_corrections_give_same_file(
    entry, store, known_works, review_file, tmp_path
) -> None:
    store.add_pages(_page(KITSU_ONE, MAL), _page(KITSU_TWO, TARGET_MAL))
    corrections = _correct(entry, store, known_works, review_file).corrections
    path = tmp_path / "corrections.jsonl"
    write_corrections(path, reversed(corrections))
    first_write = path.read_text(encoding="utf-8")
    write_corrections(path, corrections)
    assert path.read_text(encoding="utf-8") == first_write
    assert [json.loads(line)["page_key"] for line in first_write.splitlines()] == [
        "kitsu:anime:1",
        "kitsu:anime:2",
    ]
