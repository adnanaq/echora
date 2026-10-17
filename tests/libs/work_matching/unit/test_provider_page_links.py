from work_matching.provider_page import ProviderPage
from work_matching.provider_page_links import ProviderPageLinks


def test_provider_page_links_link_counts_both_ways(linked_page) -> None:
    links = ProviderPageLinks(
        [linked_page("https://kitsu.app/anime/1", "https://myanimelist.net/anime/1")],
        ["mal:anime:1"],
    )
    assert links.linked_with("kitsu:anime:1") == frozenset({"mal:anime:1"})
    assert links.linked_with("mal:anime:1") == frozenset({"kitsu:anime:1"})


def test_provider_page_links_link_outside_group_ignored(linked_page) -> None:
    links = ProviderPageLinks(
        [linked_page("https://kitsu.app/anime/1", "https://myanimelist.net/anime/2")],
        ["mal:anime:1"],
    )
    assert links.linked_with("kitsu:anime:1") == frozenset()


def test_provider_page_links_renamed_page_answers_to_current_key(
    matching_cases,
) -> None:
    _, pages = matching_cases["cocoon"]
    links = ProviderPageLinks(pages)
    old = "animeplanet:anime:cocoon-aru-natsu-no-shoujo-tachi-yori"
    assert links.same_page(old) == "animeplanet:anime:cocoon-one-summer-of-girlhood"
    assert links.same_page("mal:anime:55779") == "mal:anime:55779"


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
    assert links.linked_with("animeplanet:anime:new") == frozenset({"kitsu:anime:1"})


def test_provider_page_links_page_linking_two_pages_of_one_provider_set_aside(
    linked_page,
) -> None:
    anthology = linked_page(
        "https://anidb.net/anime/1",
        "https://myanimelist.net/anime/1",
        "https://myanimelist.net/anime/2",
    )
    links = ProviderPageLinks([anthology], ["mal:anime:1", "mal:anime:2"])
    assert links.linked_with("mal:anime:1") == frozenset()
    assert links.linked_by_multi_work_pages("mal:anime:1") == frozenset(
        {"anidb:anime:1"}
    )


def test_provider_page_links_found_page_links_count_for_lookup_page(
    matching_cases,
) -> None:
    entries, pages = matching_cases["puu_neko"]
    links = ProviderPageLinks(pages, entries[0].pages)
    assert links.linked_with("mal:anime:10459") == frozenset(
        {
            "anidb:anime:8328",
            "ann:anime:12559",
            "anilist:anime:10459",
            "kitsu:anime:6190",
        }
    )
    assert links.linked_with("https://animeschedule.net/anime/puu-neko") == (
        links.linked_with("mal:anime:10459")
    )


def test_provider_page_links_found_page_naming_other_page_of_lookup_provider_set_aside() -> (
    None
):
    found = ProviderPage.from_canonical_record(
        "https://animeschedule.net/anime/islay",
        {"title": "Islay", "sources": ["https://myanimelist.net/anime/2"]},
        found_by="https://myanimelist.net/anime/1",
    )
    links = ProviderPageLinks([found], ["mal:anime:1", "mal:anime:2"])
    assert links.linked_with("mal:anime:1") == frozenset()
    assert links.linked_by_multi_work_pages("mal:anime:2") == frozenset({"mal:anime:1"})


def test_provider_page_links_found_page_with_renamed_lookup_page_counts_for_current_key(
    linked_page,
) -> None:
    renamed = linked_page(
        "https://anime-planet.com/anime/old", "https://anime-planet.com/anime/new"
    )
    found = ProviderPage.from_canonical_record(
        "https://animeschedule.net/anime/show",
        {"title": "Show", "sources": ["https://myanimelist.net/anime/1"]},
        found_by="https://anime-planet.com/anime/old",
    )
    links = ProviderPageLinks(
        [renamed, found], ["animeplanet:anime:new", "mal:anime:1"]
    )
    assert links.linked_with("animeplanet:anime:new") == frozenset({"mal:anime:1"})


def test_provider_page_links_link_to_own_provider_page_ignored() -> None:
    page = ProviderPage(
        key="kitsu:anime:1",
        address="https://kitsu.app/anime/1",
        links=("kitsu:anime:2",),
    )
    links = ProviderPageLinks([page], ["kitsu:anime:2"])
    assert links.linked_with("kitsu:anime:1") == frozenset()
