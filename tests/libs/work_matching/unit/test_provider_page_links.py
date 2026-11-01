from work_matching.provider_page import ProviderPage
from work_matching.provider_page_links import ProviderPageLinks


def test_provider_page_links_link_counts_both_ways(linked_page) -> None:
    links = ProviderPageLinks(
        [linked_page("https://kitsu.app/anime/1", "https://myanimelist.net/anime/1")],
        ["mal:anime:1"],
    )
    assert links.get_linked_page_keys("kitsu:anime:1") == frozenset({"mal:anime:1"})
    assert links.get_linked_page_keys("mal:anime:1") == frozenset({"kitsu:anime:1"})


def test_provider_page_links_link_outside_candidates_ignored(linked_page) -> None:
    links = ProviderPageLinks(
        [linked_page("https://kitsu.app/anime/1", "https://myanimelist.net/anime/2")],
        ["mal:anime:1"],
    )
    assert links.get_linked_page_keys("kitsu:anime:1") == frozenset()


def test_provider_page_links_renamed_page_answers_to_current_key(
    matching_cases,
) -> None:
    _, pages = matching_cases["cocoon"]
    links = ProviderPageLinks(pages)
    old = "animeplanet:anime:cocoon-aru-natsu-no-shoujo-tachi-yori"
    assert (
        links.get_current_key(old) == "animeplanet:anime:cocoon-one-summer-of-girlhood"
    )
    assert links.get_current_key("mal:anime:55779") == "mal:anime:55779"


def test_provider_page_links_link_to_renamed_page_reaches_current_key(
    linked_page,
) -> None:
    renamed = linked_page(
        "https://anime-planet.com/anime/old", "https://anime-planet.com/anime/new"
    )
    kitsu = linked_page(
        "https://kitsu.app/anime/1", "https://anime-planet.com/anime/old"
    )
    links = ProviderPageLinks([renamed, kitsu], ["animeplanet:anime:new"])
    assert links.get_linked_page_keys("animeplanet:anime:new") == frozenset(
        {"kitsu:anime:1"}
    )


def test_provider_page_links_page_linking_two_pages_of_one_provider_set_aside(
    linked_page,
) -> None:
    anthology = linked_page(
        "https://anidb.net/anime/1",
        "https://myanimelist.net/anime/1",
        "https://myanimelist.net/anime/2",
    )
    links = ProviderPageLinks([anthology], ["mal:anime:1", "mal:anime:2"])
    assert links.get_linked_page_keys("mal:anime:1") == frozenset()
    assert links.get_linking_multi_work_page_keys("mal:anime:1") == frozenset(
        {"anidb:anime:1"}
    )


def test_provider_page_links_searched_page_links_count_for_searched_by_page(
    matching_cases,
) -> None:
    entries, pages = matching_cases["puu_neko"]
    links = ProviderPageLinks(pages, entries[0].addresses_by_page_key)
    assert links.get_linked_page_keys("mal:anime:10459") == frozenset(
        {
            "anidb:anime:8328",
            "ann:anime:12559",
            "anilist:anime:10459",
            "kitsu:anime:6190",
        }
    )
    assert links.get_linked_page_keys("https://animeschedule.net/anime/puu-neko") == (
        links.get_linked_page_keys("mal:anime:10459")
    )


def test_provider_page_links_searched_page_naming_other_page_of_searched_by_provider_set_aside() -> (
    None
):
    found = ProviderPage.from_canonical_record(
        "https://animeschedule.net/anime/islay",
        {"title": "Islay", "sources": ["https://myanimelist.net/anime/2"]},
        searched_by_address="https://myanimelist.net/anime/1",
    )
    links = ProviderPageLinks([found], ["mal:anime:1", "mal:anime:2"])
    assert links.get_linked_page_keys("mal:anime:1") == frozenset()
    assert links.get_linking_multi_work_page_keys("mal:anime:2") == frozenset(
        {"mal:anime:1"}
    )


def test_provider_page_links_searched_page_with_renamed_searched_by_page_counts_for_current_key(
    linked_page,
) -> None:
    renamed = linked_page(
        "https://anime-planet.com/anime/old", "https://anime-planet.com/anime/new"
    )
    found = ProviderPage.from_canonical_record(
        "https://animeschedule.net/anime/show",
        {"title": "Show", "sources": ["https://myanimelist.net/anime/1"]},
        searched_by_address="https://anime-planet.com/anime/old",
    )
    links = ProviderPageLinks(
        [renamed, found], ["animeplanet:anime:new", "mal:anime:1"]
    )
    assert links.get_linked_page_keys("animeplanet:anime:new") == frozenset(
        {"mal:anime:1"}
    )


def test_provider_page_links_link_to_own_provider_page_ignored() -> None:
    page = ProviderPage(
        key="kitsu:anime:1",
        address="https://kitsu.app/anime/1",
        linked_page_keys=("kitsu:anime:2",),
    )
    links = ProviderPageLinks([page], ["kitsu:anime:2"])
    assert links.get_linked_page_keys("kitsu:anime:1") == frozenset()


def test_provider_page_links_searched_page_counts_for_searched_by_page_not_given() -> (
    None
):
    found = ProviderPage.from_canonical_record(
        "https://animeschedule.net/anime/show",
        {"title": "Show", "sources": ["https://anilist.co/anime/1"]},
        searched_by_address="https://myanimelist.net/anime/1",
    )
    links = ProviderPageLinks([found], ["anilist:anime:1"])
    assert links.get_linked_page_keys("mal:anime:1") == frozenset({"anilist:anime:1"})


def test_provider_page_links_searched_page_through_two_searched_by_pages_links_each_with_other() -> (
    None
):
    found = ProviderPage.from_canonical_record(
        "https://animeschedule.net/anime/ranma",
        {
            "title": "Ranma",
            "sources": [
                "https://myanimelist.net/anime/210",
                "https://anilist.co/anime/210",
                "https://anidb.net/anime/193",
            ],
        },
        searched_by_address="https://myanimelist.net/anime/210",
    ).model_copy(update={"searched_by": ("anilist:anime:210", "mal:anime:210")})
    links = ProviderPageLinks(
        [found], ["mal:anime:210", "anilist:anime:210", "anidb:anime:193"]
    )
    assert links.get_linked_page_keys("mal:anime:210") == frozenset(
        {"anilist:anime:210", "anidb:anime:193"}
    )
    assert links.get_linked_page_keys("anilist:anime:210") == frozenset(
        {"mal:anime:210", "anidb:anime:193"}
    )


def test_provider_page_links_page_linked_with_several_mal_pages_ties_nothing(
    matching_cases,
) -> None:
    entries, pages = matching_cases["ajin_films"]
    links = ProviderPageLinks(
        pages, [key for entry in entries for key in entry.addresses_by_page_key]
    )
    assert links.get_linked_page_keys("anidb:anime:11265") == frozenset()
    assert links.get_linked_page_keys("mal:anime:30868") == frozenset(
        {"anilist:anime:21212", "ann:anime:17176", "kitsu:anime:10949"}
    )
