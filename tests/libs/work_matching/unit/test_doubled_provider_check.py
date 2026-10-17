from work_matching.doubled_provider_check import (
    Outcome,
    Reason,
    UnrelatedPage,
    decide_doubled_providers,
)
from work_matching.known_works_from_seed import KnownWorksFromSeed
from work_matching.provider_page import ProviderPage
from work_matching.work_entry import WorkEntry

NO_OTHER_WORKS = KnownWorksFromSeed([])


def _decisions(entry, pages, known_works=NO_OTHER_WORKS):
    return {
        decision.provider: decision
        for decision in decide_doubled_providers(entry, pages, known_works)
    }


def test_decide_doubled_providers_renamed_page_same_page(matching_cases) -> None:
    entries, pages = matching_cases["cocoon"]
    decision = _decisions(entries[0], pages)["animeplanet"]
    assert (decision.outcome, decision.reason) == (
        Outcome.SAME_PAGE,
        Reason.SAME_PAGE,
    )
    assert decision.pages == ("animeplanet:anime:cocoon-one-summer-of-girlhood",)


def test_decide_doubled_providers_pages_linked_apart_mixed_entry(
    matching_cases,
) -> None:
    entries, pages = matching_cases["puu_neko"]
    decisions = _decisions(entries[0], pages)
    assert {provider: decision.outcome for provider, decision in decisions.items()} == {
        "mal": Outcome.MIXED_ENTRY,
        "anilist": Outcome.MIXED_ENTRY,
        "kitsu": Outcome.MIXED_ENTRY,
    }
    assert decisions["kitsu"].belonging == {
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


def test_decide_doubled_providers_pages_sharing_linked_page_one_work(
    matching_cases,
) -> None:
    entries, pages = matching_cases["ao_haru_ride_ova"]
    decision = _decisions(entries[0], pages)["anilist"]
    assert (decision.outcome, decision.reason) == (Outcome.ONE_WORK, Reason.SHARED_LINK)
    assert set(decision.belonging) == {"anilist:anime:20837", "anilist:anime:20900"}


def test_decide_doubled_providers_pages_sharing_link_with_other_years_review(
    linked_page, work_entry
) -> None:
    entry = work_entry(
        "https://anilist.co/anime/1",
        "https://kitsu.app/anime/1",
        "https://kitsu.app/anime/2",
    )
    pages = [
        linked_page("https://kitsu.app/anime/1", "https://anilist.co/anime/1"),
        linked_page("https://kitsu.app/anime/2", "https://anilist.co/anime/1"),
    ]
    pages = [
        page.model_copy(update={"year": year})
        for page, year in zip(pages, (2019, 2020))
    ]
    decision = _decisions(entry, pages)["kitsu"]
    assert (decision.outcome, decision.reason) == (
        Outcome.REVIEW,
        Reason.DETAILS_DIFFER,
    )


def test_decide_doubled_providers_page_without_links_review(matching_cases) -> None:
    entries, pages = matching_cases["ao_haru_ride_ova"]
    decision = _decisions(entries[0], pages)["animeplanet"]
    assert (decision.outcome, decision.reason) == (
        Outcome.REVIEW,
        Reason.UNDECIDED_PAGES,
    )
    assert decision.belonging == {
        "animeplanet:anime:blue-spring-ride-page-13": frozenset({"mal:anime:24151"})
    }
    assert decision.undecided == ("animeplanet:anime:blue-spring-ride-unwritten",)


def test_decide_doubled_providers_no_links_records_title_and_year_type_matches(
    matching_cases,
) -> None:
    entries, pages = matching_cases["genshin_sigewinne"]
    decision = _decisions(entries[0], pages)["animeplanet"]
    sigewinne = "animeplanet:anime:genshin-impact-character-teaser-sigewinne-everyday-correspondence"
    assert decision.outcome is Outcome.REVIEW
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


def test_decide_doubled_providers_unfetched_page_without_links_review(
    matching_cases,
) -> None:
    entries, pages = matching_cases["episode_count"]
    decision = _decisions(entries[0], pages)["ann"]
    assert decision.reason is Reason.UNDECIDED_PAGES
    assert decision.belonging == {"ann:anime:32626": frozenset({"mal:anime:58573"})}
    assert decision.undecided == ("ann:anime:38845",)


def test_decide_doubled_providers_page_linking_known_work_unrelated(
    linked_page, work_entry
) -> None:
    other_work = {
        "title": "Other",
        "type": "TV",
        "animeSeason": {"year": 2020},
        "sources": ["https://myanimelist.net/anime/2"],
    }
    known_works = KnownWorksFromSeed([other_work])
    entry = work_entry(
        "https://myanimelist.net/anime/1",
        "https://kitsu.app/anime/1",
        "https://kitsu.app/anime/2",
    )
    pages = [
        linked_page("https://kitsu.app/anime/1", "https://myanimelist.net/anime/1"),
        linked_page("https://kitsu.app/anime/2", "https://myanimelist.net/anime/2"),
    ]
    decision = _decisions(entry, pages, known_works)["kitsu"]
    assert (decision.outcome, decision.reason) == (Outcome.ONE_WORK, Reason.ONE_LINKED)
    assert decision.unrelated == {
        "kitsu:anime:2": UnrelatedPage(
            links=frozenset({"mal:anime:2"}),
            other_works=(WorkEntry.from_seed_entry(other_work),),
        )
    }


def test_decide_doubled_providers_every_page_linking_elsewhere_review(
    linked_page, work_entry
) -> None:
    entry = work_entry(
        "https://myanimelist.net/anime/1",
        "https://kitsu.app/anime/1",
        "https://kitsu.app/anime/2",
    )
    pages = [
        linked_page("https://kitsu.app/anime/1", "https://myanimelist.net/anime/7"),
        linked_page("https://kitsu.app/anime/2", "https://myanimelist.net/anime/8"),
    ]
    decision = _decisions(entry, pages)["kitsu"]
    assert (decision.outcome, decision.reason) == (Outcome.REVIEW, Reason.NONE_LINKED)
    assert decision.unrelated["kitsu:anime:1"].other_works == ()


def test_decide_doubled_providers_link_from_multi_work_page_review(
    linked_page, work_entry
) -> None:
    entry = work_entry(
        "https://myanimelist.net/anime/1",
        "https://myanimelist.net/anime/2",
        "https://anidb.net/anime/1",
    )
    anthology = linked_page(
        "https://anidb.net/anime/1",
        "https://myanimelist.net/anime/1",
        "https://myanimelist.net/anime/2",
    )
    decision = _decisions(entry, [anthology])["mal"]
    assert (decision.outcome, decision.reason) == (
        Outcome.REVIEW,
        Reason.MULTI_WORK_LINK,
    )


def test_decide_doubled_providers_overlapping_links_review(
    linked_page, work_entry
) -> None:
    entry = work_entry(
        "https://myanimelist.net/anime/1",
        "https://myanimelist.net/anime/2",
        "https://myanimelist.net/anime/3",
        "https://anilist.co/anime/1",
        "https://kitsu.app/anime/1",
    )
    pages = [
        linked_page("https://myanimelist.net/anime/1", "https://anilist.co/anime/1"),
        linked_page(
            "https://myanimelist.net/anime/2",
            "https://anilist.co/anime/1",
            "https://kitsu.app/anime/1",
        ),
        linked_page("https://myanimelist.net/anime/3", "https://kitsu.app/anime/1"),
    ]
    decision = _decisions(entry, pages)["mal"]
    assert (decision.outcome, decision.reason) == (
        Outcome.REVIEW,
        Reason.OVERLAPPING_LINKS,
    )


def test_decide_doubled_providers_page_linked_and_linking_elsewhere_review(
    linked_page, work_entry
) -> None:
    entry = work_entry(
        "https://myanimelist.net/anime/1",
        "https://anilist.co/anime/1",
        "https://kitsu.app/anime/1",
        "https://kitsu.app/anime/2",
    )
    pages = [
        linked_page("https://kitsu.app/anime/1", "https://myanimelist.net/anime/1"),
        linked_page(
            "https://kitsu.app/anime/2",
            "https://anilist.co/anime/1",
            "https://myanimelist.net/anime/9",
        ),
    ]
    decision = _decisions(entry, pages)["kitsu"]
    assert (decision.outcome, decision.reason) == (
        Outcome.REVIEW,
        Reason.CONTRADICTING_LINKS,
    )


def test_decide_doubled_providers_dead_page_review(linked_page, work_entry) -> None:
    entry = work_entry(
        "https://myanimelist.net/anime/1",
        "https://myanimelist.net/anime/2",
        "https://kitsu.app/anime/1",
    )
    pages = [
        ProviderPage.dead_page("https://myanimelist.net/anime/1"),
        linked_page("https://kitsu.app/anime/1", "https://myanimelist.net/anime/2"),
    ]
    decision = _decisions(entry, pages)["mal"]
    assert (decision.outcome, decision.reason) == (Outcome.REVIEW, Reason.DEAD_PAGE)


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
        ("simkl", Outcome.REVIEW)
    ]
    assert len(decisions[0].undecided) == 4


def test_decide_doubled_providers_no_doubled_provider_gives_nothing(work_entry) -> None:
    entry = work_entry("https://myanimelist.net/anime/1", "https://kitsu.app/anime/1")
    assert decide_doubled_providers(entry, [], NO_OTHER_WORKS) == []
