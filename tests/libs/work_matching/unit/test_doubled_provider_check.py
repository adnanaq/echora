from datetime import UTC, datetime

import pytest
from common.models.anime import AnimeType
from work_matching.doubled_provider_check import decide_doubled_providers
from work_matching.doubled_provider_decision import (
    DoubledProviderOutcome,
    DoubledProviderReviewReason,
    UnrelatedPage,
)
from work_matching.known_work_entries_from_seed import KnownWorkEntriesFromSeed
from work_matching.provider_page import ProviderPage
from work_matching.work_entry import WorkEntry

NO_OTHER_WORKS = KnownWorkEntriesFromSeed([])
MAL = "https://myanimelist.net/anime/1"
KITSU_ONE = "https://kitsu.app/anime/1"
KITSU_TWO = "https://kitsu.app/anime/2"
OTHER_MAL = "https://myanimelist.net/anime/9"
OTHER_ANILIST = "https://anilist.co/anime/9"


def _decisions(entry, pages, known_work_entries=NO_OTHER_WORKS):
    return {
        decision.provider: decision
        for decision in decide_doubled_providers(entry, pages, known_work_entries)
    }


def _verdict(decision):
    return decision.outcome, decision.review_reasons


def _anilist_page(number: int, episode_count: int, aired_from: datetime):
    return ProviderPage(
        key=f"anilist:anime:{number}",
        address=f"https://anilist.co/anime/{number}",
        type="ONA",
        year=aired_from.year,
        episode_count=episode_count,
        aired_from=aired_from,
        linked_page_keys=("mal:anime:17295",),
    )


def _other_work(**fields) -> dict:
    return {"title": "Other", "type": "TV", "sources": [OTHER_MAL]} | fields


def test_decide_doubled_providers_renamed_page_same_page(matching_cases) -> None:
    entries, pages = matching_cases["cocoon"]
    decision = _decisions(entries[0], pages)["animeplanet"]
    assert _verdict(decision) == (DoubledProviderOutcome.SAME_PAGE, ())
    assert decision.page_keys == ("animeplanet:anime:cocoon-one-summer-of-girlhood",)


def test_decide_doubled_providers_renamed_page_only_old_addresses_listed_current_address_missing(
    work_entry,
) -> None:
    new_address = "https://www.anime-planet.com/anime/new"
    old_addresses = [f"https://www.anime-planet.com/anime/old-{n}" for n in (1, 2)]
    pages = [
        ProviderPage.from_canonical_record(
            address, {"title": "Show", "sources": [address, new_address]}
        )
        for address in old_addresses
    ]
    decision = _decisions(work_entry(MAL, *old_addresses), pages)["animeplanet"]
    assert _verdict(decision) == (
        DoubledProviderOutcome.SAME_PAGE,
        (DoubledProviderReviewReason.CURRENT_ADDRESS_MISSING,),
    )


def test_decide_doubled_providers_pages_linked_apart_separate_works(
    matching_cases,
) -> None:
    entries, pages = matching_cases["puu_neko"]
    decisions = _decisions(entries[0], pages)
    assert {provider: decision.outcome for provider, decision in decisions.items()} == {
        "mal": DoubledProviderOutcome.SEPARATE_WORKS,
        "anilist": DoubledProviderOutcome.SEPARATE_WORKS,
        "kitsu": DoubledProviderOutcome.SEPARATE_WORKS,
    }
    assert decisions["kitsu"].belonging_page_keys == {
        "kitsu:anime:6190": frozenset(
            {
                "anilist:anime:10459",
                "mal:anime:10459",
            }
        ),
        "kitsu:anime:6543": frozenset(
            {
                "anidb:anime:8328",
                "anilist:anime:11589",
                "mal:anime:11589",
            }
        ),
    }


def test_decide_doubled_providers_pages_sharing_linked_page_aired_apart_part_pages(
    matching_cases,
) -> None:
    entries, pages = matching_cases["ao_haru_ride_ova"]
    decision = _decisions(entries[0], pages)["anilist"]
    assert _verdict(decision) == (DoubledProviderOutcome.PART_PAGES, ())
    assert set(decision.belonging_page_keys) == {
        "anilist:anime:20837",
        "anilist:anime:20900",
    }


def test_decide_doubled_providers_pages_sharing_linked_page_other_episode_counts_part_pages(
    work_entry,
) -> None:
    first_aired = datetime(2010, 9, 8, 15, tzinfo=UTC)
    entry = work_entry(
        "https://myanimelist.net/anime/17295",
        "https://anilist.co/anime/17295",
        "https://anilist.co/anime/206843",
        work_type="ONA",
        year=2010,
    )
    pages = [
        _anilist_page(17295, episode_count=1, aired_from=first_aired),
        _anilist_page(206843, episode_count=4, aired_from=first_aired),
    ]
    assert _verdict(_decisions(entry, pages)["anilist"]) == (
        DoubledProviderOutcome.PART_PAGES,
        (),
    )


def test_decide_doubled_providers_pages_sharing_linked_page_same_details_duplicate_pages(
    linked_page, work_entry
) -> None:
    aired_from = datetime(1957, 8, 19, 15, tzinfo=UTC)
    pages = [
        linked_page(address, MAL).model_copy(
            update={"episode_count": 1, "aired_from": aired_from}
        )
        for address in (KITSU_ONE, KITSU_TWO)
    ]
    decision = _decisions(work_entry(MAL, KITSU_ONE, KITSU_TWO), pages)["kitsu"]
    assert _verdict(decision) == (DoubledProviderOutcome.DUPLICATE_PAGES, ())
    assert decision.needs_review


def test_decide_doubled_providers_pages_sharing_link_unknown_details_duplicate_pages(
    linked_page, work_entry
) -> None:
    pages = [linked_page(KITSU_ONE, MAL), linked_page(KITSU_TWO, MAL)]
    decision = _decisions(work_entry(MAL, KITSU_ONE, KITSU_TWO), pages)["kitsu"]
    assert _verdict(decision) == (DoubledProviderOutcome.DUPLICATE_PAGES, ())


def test_decide_doubled_providers_pages_sharing_link_with_other_years_details_differ(
    linked_page, work_entry
) -> None:
    entry = work_entry("https://anilist.co/anime/1", KITSU_ONE, KITSU_TWO)
    pages = [
        linked_page(address, "https://anilist.co/anime/1").model_copy(
            update={"year": year}
        )
        for address, year in ((KITSU_ONE, 2019), (KITSU_TWO, 2020))
    ]
    assert _verdict(_decisions(entry, pages)["kitsu"]) == (
        DoubledProviderOutcome.DUPLICATE_PAGES,
        (DoubledProviderReviewReason.DETAILS_DIFFER,),
    )


def test_decide_doubled_providers_music_and_special_pages_sharing_link_duplicate_pages(
    linked_page, work_entry
) -> None:
    pages = [
        linked_page(KITSU_ONE, MAL).model_copy(update={"type": "MUSIC"}),
        linked_page(KITSU_TWO, MAL).model_copy(update={"type": "SPECIAL"}),
    ]
    decision = _decisions(work_entry(MAL, KITSU_ONE, KITSU_TWO), pages)["kitsu"]
    assert _verdict(decision) == (DoubledProviderOutcome.DUPLICATE_PAGES, ())


def test_decide_doubled_providers_page_without_links_undecided(matching_cases) -> None:
    entries, pages = matching_cases["ao_haru_ride_ova"]
    decision = _decisions(entries[0], pages)["animeplanet"]
    assert _verdict(decision) == (
        DoubledProviderOutcome.UNDECIDED,
        (DoubledProviderReviewReason.UNLINKED_PAGES,),
    )
    assert decision.belonging_page_keys == {
        "animeplanet:anime:blue-spring-ride-page-13": frozenset({"mal:anime:24151"})
    }
    assert decision.unlinked_page_keys == (
        "animeplanet:anime:blue-spring-ride-unwritten",
    )


def test_decide_doubled_providers_no_links_records_title_and_year_type_matches(
    matching_cases,
) -> None:
    entries, pages = matching_cases["genshin_sigewinne"]
    decision = _decisions(entries[0], pages)["animeplanet"]
    sigewinne = "animeplanet:anime:genshin-impact-character-teaser-sigewinne-everyday-correspondence"
    assert decision.needs_review
    assert (decision.title_matches, decision.year_type_matches) == (
        (sigewinne,),
        (sigewinne,),
    )


def test_decide_doubled_providers_no_links_records_season_matches(
    linked_page, work_entry
) -> None:
    entry = work_entry(
        "https://anime-planet.com/anime/first", "https://anime-planet.com/anime/second"
    ).model_copy(update={"season": "SUMMER"})
    pages = [
        linked_page(address).model_copy(update={"season": season})
        for address, season in (
            ("https://anime-planet.com/anime/first", "SUMMER"),
            ("https://anime-planet.com/anime/second", "FALL"),
        )
    ]
    decision = _decisions(entry, pages)["animeplanet"]
    assert decision.season_matches == ("animeplanet:anime:first",)


def test_decide_doubled_providers_applied_decision_records_no_matches(
    linked_page, work_entry
) -> None:
    known_work_entries = KnownWorkEntriesFromSeed([_other_work()])
    pages = [linked_page(KITSU_ONE, MAL), linked_page(KITSU_TWO, OTHER_MAL)]
    decision = _decisions(
        work_entry(MAL, KITSU_ONE, KITSU_TWO, year=None), pages, known_work_entries
    )["kitsu"]
    assert not decision.needs_review
    assert (
        decision.title_matches,
        decision.year_type_matches,
        decision.season_matches,
    ) == ((), (), ())


def test_decide_doubled_providers_unfetched_page_without_links_undecided(
    matching_cases,
) -> None:
    entries, pages = matching_cases["episode_count"]
    decision = _decisions(entries[0], pages)["ann"]
    assert decision.review_reasons == (DoubledProviderReviewReason.UNLINKED_PAGES,)
    assert decision.belonging_page_keys == {
        "ann:anime:32626": frozenset({"mal:anime:58573"})
    }
    assert decision.unlinked_page_keys == ("ann:anime:38845",)


def test_decide_doubled_providers_page_linking_known_work_moves_there(
    linked_page, work_entry
) -> None:
    other_work = _other_work(animeSeason={"year": 2020})
    known_work_entries = KnownWorkEntriesFromSeed([other_work])
    pages = [linked_page(KITSU_ONE, MAL), linked_page(KITSU_TWO, OTHER_MAL)]
    decision = _decisions(
        work_entry(MAL, KITSU_ONE, KITSU_TWO), pages, known_work_entries
    )["kitsu"]
    assert _verdict(decision) == (DoubledProviderOutcome.PAGES_OF_OTHER_WORKS, ())
    assert decision.unrelated_pages == {
        "kitsu:anime:2": UnrelatedPage(
            linked_page_keys=frozenset({"mal:anime:9"}),
            other_work_entries=(WorkEntry.from_seed_entry(other_work),),
            move_target_entry=WorkEntry.from_seed_entry(other_work),
        )
    }


def test_decide_doubled_providers_every_page_linking_other_work_no_page_linked_with_entry(
    linked_page, work_entry
) -> None:
    pages = [
        linked_page(KITSU_ONE, "https://myanimelist.net/anime/7"),
        linked_page(KITSU_TWO, "https://myanimelist.net/anime/8"),
    ]
    decision = _decisions(work_entry(MAL, KITSU_ONE, KITSU_TWO), pages)["kitsu"]
    assert _verdict(decision) == (
        DoubledProviderOutcome.PAGES_OF_OTHER_WORKS,
        (
            DoubledProviderReviewReason.NO_PAGE_LINKED_WITH_ENTRY,
            DoubledProviderReviewReason.MOVE_NOT_SURE,
        ),
    )
    assert decision.unrelated_pages["kitsu:anime:1"].other_work_entries == ()


@pytest.mark.parametrize(
    ("moved_page_details", "is_move_sure"),
    [
        ({"type": "MOVIE", "year": 2020}, False),
        ({"type": "TV", "year": 1987}, False),
        ({"type": "TV", "year": 2020}, True),
        ({"type": "TV SHORT"}, True),
    ],
)
def test_decide_doubled_providers_moved_page_details_must_agree_with_target(
    work_entry, moved_page_details: dict, is_move_sure: bool
) -> None:
    known_work_entries = KnownWorkEntriesFromSeed(
        [_other_work(animeSeason={"year": 2020})]
    )
    pages = [
        ProviderPage.from_canonical_record(
            KITSU_ONE, {"title": "Show", "sources": [KITSU_ONE, MAL]}
        ),
        ProviderPage.from_canonical_record(
            KITSU_TWO,
            {"title": "Other", "sources": [KITSU_TWO, OTHER_MAL]} | moved_page_details,
        ),
    ]
    decision = _decisions(
        work_entry(MAL, KITSU_ONE, KITSU_TWO), pages, known_work_entries
    )["kitsu"]
    assert (
        DoubledProviderReviewReason.MOVE_NOT_SURE not in decision.review_reasons
    ) is is_move_sure


def test_decide_doubled_providers_target_holding_same_provider_move_not_sure(
    linked_page, work_entry
) -> None:
    known_work_entries = KnownWorkEntriesFromSeed(
        [_other_work(sources=[OTHER_MAL, "https://kitsu.app/anime/99"])]
    )
    pages = [linked_page(KITSU_ONE, MAL), linked_page(KITSU_TWO, OTHER_MAL)]
    decision = _decisions(
        work_entry(MAL, KITSU_ONE, KITSU_TWO), pages, known_work_entries
    )["kitsu"]
    assert _verdict(decision) == (
        DoubledProviderOutcome.PAGES_OF_OTHER_WORKS,
        (DoubledProviderReviewReason.MOVE_NOT_SURE,),
    )


def test_decide_doubled_providers_page_linking_two_works_move_not_sure(
    linked_page, work_entry
) -> None:
    known_work_entries = KnownWorkEntriesFromSeed(
        [
            _other_work(),
            {"title": "Third", "type": "TV", "sources": [OTHER_ANILIST]},
        ]
    )
    pages = [
        linked_page(KITSU_ONE, MAL),
        linked_page(KITSU_TWO, OTHER_MAL, OTHER_ANILIST),
    ]
    decision = _decisions(
        work_entry(MAL, KITSU_ONE, KITSU_TWO), pages, known_work_entries
    )["kitsu"]
    assert decision.review_reasons == (DoubledProviderReviewReason.MOVE_NOT_SURE,)


def test_decide_doubled_providers_page_linking_work_outside_seed_move_not_sure(
    linked_page, work_entry
) -> None:
    pages = [
        linked_page(KITSU_ONE, MAL),
        linked_page(KITSU_TWO, "https://myanimelist.net/anime/77"),
    ]
    decision = _decisions(work_entry(MAL, KITSU_ONE, KITSU_TWO), pages)["kitsu"]
    assert decision.review_reasons == (DoubledProviderReviewReason.MOVE_NOT_SURE,)


def test_decide_doubled_providers_link_from_multi_work_page_lists_multi_work_link(
    linked_page, work_entry
) -> None:
    entry = work_entry(
        MAL, "https://myanimelist.net/anime/2", "https://anidb.net/anime/1"
    )
    anthology = linked_page(
        "https://anidb.net/anime/1", MAL, "https://myanimelist.net/anime/2"
    )
    assert _verdict(_decisions(entry, [anthology])["mal"]) == (
        DoubledProviderOutcome.UNDECIDED,
        (
            DoubledProviderReviewReason.UNLINKED_PAGES,
            DoubledProviderReviewReason.MULTI_WORK_LINK,
        ),
    )


def test_decide_doubled_providers_overlapping_links_undecided(
    linked_page, work_entry
) -> None:
    entry = work_entry(
        KITSU_ONE,
        KITSU_TWO,
        "https://kitsu.app/anime/3",
        "https://anilist.co/anime/1",
        "https://anidb.net/anime/1",
    )
    pages = [
        linked_page(KITSU_ONE, "https://anilist.co/anime/1"),
        linked_page(
            KITSU_TWO, "https://anilist.co/anime/1", "https://anidb.net/anime/1"
        ),
        linked_page("https://kitsu.app/anime/3", "https://anidb.net/anime/1"),
    ]
    assert _verdict(_decisions(entry, pages)["kitsu"]) == (
        DoubledProviderOutcome.UNDECIDED,
        (DoubledProviderReviewReason.OVERLAPPING_LINKS,),
    )


def test_decide_doubled_providers_mal_pages_sharing_linked_pages_unlinked(
    linked_page, work_entry
) -> None:
    entry = work_entry(
        MAL,
        "https://myanimelist.net/anime/2",
        "https://myanimelist.net/anime/3",
        "https://anilist.co/anime/1",
        KITSU_ONE,
    )
    pages = [
        linked_page(MAL, "https://anilist.co/anime/1"),
        linked_page(
            "https://myanimelist.net/anime/2", "https://anilist.co/anime/1", KITSU_ONE
        ),
        linked_page("https://myanimelist.net/anime/3", KITSU_ONE),
    ]
    assert _verdict(_decisions(entry, pages)["mal"]) == (
        DoubledProviderOutcome.UNDECIDED,
        (DoubledProviderReviewReason.UNLINKED_PAGES,),
    )


def test_decide_doubled_providers_page_linked_and_linking_other_work_contradicting(
    linked_page, work_entry
) -> None:
    entry = work_entry(MAL, "https://anilist.co/anime/1", KITSU_ONE, KITSU_TWO)
    pages = [
        linked_page(KITSU_ONE, MAL),
        linked_page(KITSU_TWO, "https://anilist.co/anime/1", OTHER_MAL),
    ]
    decision = _decisions(entry, pages)["kitsu"]
    assert _verdict(decision) == (
        DoubledProviderOutcome.UNDECIDED,
        (DoubledProviderReviewReason.CONTRADICTING_LINKS,),
    )
    assert decision.contradicting_page_keys == ("kitsu:anime:2",)


def test_decide_doubled_providers_dead_page_lists_dead_page(
    linked_page, work_entry
) -> None:
    entry = work_entry(MAL, "https://myanimelist.net/anime/2", KITSU_ONE)
    pages = [
        ProviderPage.from_dead_address(MAL),
        linked_page(KITSU_ONE, "https://myanimelist.net/anime/2"),
    ]
    assert _verdict(_decisions(entry, pages)["mal"]) == (
        DoubledProviderOutcome.UNDECIDED,
        (
            DoubledProviderReviewReason.UNLINKED_PAGES,
            DoubledProviderReviewReason.DEAD_PAGE,
        ),
    )


def test_decide_doubled_providers_multi_work_link_and_dead_page_lists_both(
    linked_page, work_entry
) -> None:
    entry = work_entry(
        MAL, "https://myanimelist.net/anime/2", "https://anidb.net/anime/1"
    )
    pages = [
        linked_page(
            "https://anidb.net/anime/1", MAL, "https://myanimelist.net/anime/2"
        ),
        ProviderPage.from_dead_address("https://myanimelist.net/anime/2"),
    ]
    assert _decisions(entry, pages)["mal"].review_reasons == (
        DoubledProviderReviewReason.UNLINKED_PAGES,
        DoubledProviderReviewReason.MULTI_WORK_LINK,
        DoubledProviderReviewReason.DEAD_PAGE,
    )


def test_decide_doubled_providers_dead_page_and_contradicting_links_lists_both(
    linked_page, work_entry
) -> None:
    entry = work_entry(MAL, "https://anilist.co/anime/1", KITSU_ONE, KITSU_TWO)
    pages = [
        ProviderPage.from_dead_address(KITSU_ONE),
        linked_page(KITSU_TWO, "https://anilist.co/anime/1", OTHER_MAL),
    ]
    assert _decisions(entry, pages)["kitsu"].review_reasons == (
        DoubledProviderReviewReason.CONTRADICTING_LINKS,
        DoubledProviderReviewReason.UNLINKED_PAGES,
        DoubledProviderReviewReason.DEAD_PAGE,
    )


def test_decide_doubled_providers_simkl_and_animecountdown_decided_together(
    work_entry,
) -> None:
    entry = work_entry(
        "https://simkl.com/anime/1",
        "https://simkl.com/anime/2",
        "https://animecountdown.com/1",
        "https://animecountdown.com/2",
    )
    decisions = decide_doubled_providers(entry, [], NO_OTHER_WORKS)
    assert [(decision.provider, decision.outcome) for decision in decisions] == [
        ("simkl", DoubledProviderOutcome.UNDECIDED)
    ]
    assert len(decisions[0].unlinked_page_keys) == 4


def test_decide_doubled_providers_simkl_and_animecountdown_other_numbers_undecided(
    work_entry,
) -> None:
    entry = work_entry("https://simkl.com/anime/1", "https://animecountdown.com/2")
    decisions = decide_doubled_providers(entry, [], NO_OTHER_WORKS)
    assert [(decision.provider, decision.review_reasons) for decision in decisions] == [
        ("simkl", (DoubledProviderReviewReason.UNLINKED_PAGES,))
    ]


def test_decide_doubled_providers_simkl_and_animecountdown_same_number_gives_nothing(
    work_entry,
) -> None:
    entry = work_entry("https://simkl.com/anime/1", "https://animecountdown.com/1")
    assert decide_doubled_providers(entry, [], NO_OTHER_WORKS) == []


def test_decide_doubled_providers_no_doubled_provider_gives_nothing(work_entry) -> None:
    entry = work_entry(MAL, KITSU_ONE)
    assert decide_doubled_providers(entry, [], NO_OTHER_WORKS) == []


def test_decide_doubled_providers_link_to_provider_entry_lacks_counts_as_no_link(
    linked_page, work_entry
) -> None:
    pages = [
        linked_page(KITSU_ONE, MAL),
        linked_page(KITSU_TWO, "https://anidb.net/anime/7"),
    ]
    decision = _decisions(work_entry(MAL, KITSU_ONE, KITSU_TWO), pages)["kitsu"]
    assert (decision.review_reasons, decision.unlinked_page_keys) == (
        (DoubledProviderReviewReason.UNLINKED_PAGES,),
        ("kitsu:anime:2",),
    )


def test_decide_doubled_providers_provider_type_falls_under_seed_type(
    linked_page, work_entry
) -> None:
    addresses = [
        f"https://anime-planet.com/anime/{slug}"
        for slug in ("short", "unknown", "movie")
    ]
    pages = [
        linked_page(address).model_copy(update={"type": provider_type})
        for address, provider_type in zip(addresses, ("TV SHORT", "UNKNOWN", "MOVIE"))
    ]
    decision = _decisions(work_entry(*addresses), pages)["animeplanet"]
    assert decision.year_type_matches == (
        "animeplanet:anime:short",
        "animeplanet:anime:unknown",
    )


@pytest.mark.parametrize(
    "canonical_type",
    [kind.value for kind in AnimeType if kind is not AnimeType.UNKNOWN],
)
def test_decide_doubled_providers_canonical_type_falls_under_one_seed_type(
    linked_page, work_entry, canonical_type: str
) -> None:
    addresses = [
        "https://anime-planet.com/anime/first",
        "https://anime-planet.com/anime/second",
    ]
    pages = [linked_page(addresses[0]).model_copy(update={"type": canonical_type})]
    seed_types = [
        seed_type
        for seed_type in ("TV", "MOVIE", "OVA", "ONA", "SPECIAL")
        if _decisions(work_entry(*addresses, work_type=seed_type), pages)[
            "animeplanet"
        ].year_type_matches
    ]
    assert len(seed_types) == 1


def test_decide_doubled_providers_link_to_own_provider_outside_entry_counts_as_no_link(
    linked_page, work_entry
) -> None:
    pages = [
        linked_page(KITSU_ONE, MAL),
        ProviderPage(
            key="kitsu:anime:2",
            address=KITSU_TWO,
            linked_page_keys=("kitsu:anime:9",),
        ),
    ]
    decision = _decisions(work_entry(MAL, KITSU_ONE, KITSU_TWO), pages)["kitsu"]
    assert (decision.review_reasons, decision.unlinked_page_keys) == (
        (DoubledProviderReviewReason.UNLINKED_PAGES,),
        ("kitsu:anime:2",),
    )


def test_decide_doubled_providers_renamed_page_only_old_address_keeps_details(
    linked_page, work_entry
) -> None:
    entry = work_entry(
        MAL,
        "https://anime-planet.com/anime/old",
        "https://anime-planet.com/anime/other",
    )
    pages = [
        linked_page(
            "https://anime-planet.com/anime/old",
            "https://anime-planet.com/anime/new",
            OTHER_MAL,
        ),
        linked_page("https://anime-planet.com/anime/other", MAL),
    ]
    decision = _decisions(entry, pages)["animeplanet"]
    assert (decision.outcome, set(decision.unrelated_pages)) == (
        DoubledProviderOutcome.PAGES_OF_OTHER_WORKS,
        {"animeplanet:anime:new"},
    )
