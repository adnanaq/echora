import pytest
from work_matching.known_works_from_seed import KnownWorksFromSeed
from work_matching.linked_page_check import (
    SKIP_ANSWER,
    LinkedPageAddition,
    LinkedPageDecision,
    LinkedPageOutcome,
    LinkedPageReason,
    LinkingEntry,
    decide_linked_page,
    find_unheld_linked_pages,
    get_entry_episode_count,
    read_linked_page_answer,
    suggest_linked_page_answer,
)
from work_matching.provider_page import ProviderPage
from work_matching.work_entry import WorkEntry

LINKED_PROVIDERS = frozenset({"mal", "anilist", "kitsu", "anidb"})

YAMATO_SEED_ENTRY = {
    "sources": [
        "https://anidb.net/anime/18902",
        "https://anilist.co/anime/182615",
        "https://anime-planet.com/anime/space-battleship-yamato-movie-new",
    ],
    "title": "Space Battleship Yamato Movie (New)",
    "type": "MOVIE",
    "animeSeason": {"season": "UNDEFINED"},
}
BOXER_SEED_ENTRY = {
    "sources": ["https://anilist.co/anime/163794"],
    "title": "The Boxer",
    "type": "TV",
    "animeSeason": {"season": "UNDEFINED"},
}
MORI_NO_YOUSEI_SEED_ENTRY = {
    "sources": [
        "https://anidb.net/anime/13091",
        "https://anisearch.com/anime/12064",
        "https://kitsu.app/anime/40741",
    ],
    "title": "Mori no Yousei: Kinoko no Musume",
    "type": "TV",
    "animeSeason": {"season": "UNDEFINED"},
}
DOUKYUUSEI_SEED_ENTRY = {
    "sources": [
        "https://animenewsnetwork.com/encyclopedia/anime.php?id=2125",
        "https://myanimelist.net/anime/3558",
    ],
    "title": "Doukyuusei: Natsu no Owari ni",
    "type": "OVA",
    "animeSeason": {"season": "SUMMER", "year": 1994},
}

YAMATO_ANILIST_PAGE = ProviderPage(
    key="anilist:anime:182615",
    address="https://anilist.co/anime/182615",
    titles=("Uchuu Senkan Yamato √2199",),
    type="MOVIE",
    episode_count=1,
    links=("mal:anime:65124",),
)
YAMATO_MAL_PAGE = ProviderPage(
    key="mal:anime:65124",
    address="https://myanimelist.net/anime/65124",
    titles=("Uchuu Senkan Yamato √2199",),
    type="MOVIE",
    episode_count=1,
)
BOXER_ANILIST_PAGE = ProviderPage(
    key="anilist:anime:163794",
    address="https://anilist.co/anime/163794",
    titles=("The Boxer",),
    type="TV",
    episode_count=12,
    links=("mal:anime:65144",),
)
BOXER_MAL_PAGE = ProviderPage(
    key="mal:anime:65144",
    address="https://myanimelist.net/anime/65144",
    titles=("The Boxer",),
    type="TV",
)
MORI_NO_YOUSEI_KITSU_PAGE = ProviderPage(
    key="kitsu:anime:40741",
    address="https://kitsu.app/anime/40741",
    titles=("Mori no Yousei: Kinoko no Musume",),
    type="TV",
    links=("mal:anime:35362",),
)
DOUKYUUSEI_MAL_PAGE = ProviderPage(
    key="mal:anime:3558",
    address="https://myanimelist.net/anime/3558",
    titles=("Doukyuusei: Natsu no Owari ni",),
    type="OVA",
    year=1994,
    links=("anidb:anime:1159", "ann:anime:2125"),
)


@pytest.fixture(scope="module")
def known_works() -> KnownWorksFromSeed:
    return KnownWorksFromSeed(
        [
            YAMATO_SEED_ENTRY,
            BOXER_SEED_ENTRY,
            MORI_NO_YOUSEI_SEED_ENTRY,
            DOUKYUUSEI_SEED_ENTRY,
        ]
    )


def _entry(seed_entry: dict) -> WorkEntry:
    return WorkEntry.from_seed_entry(seed_entry)


def _decide(
    page: ProviderPage,
    *entries: WorkEntry,
    entry_episode_count: int | None,
    other_linked_page_keys: frozenset[str] = frozenset(),
) -> LinkedPageDecision:
    return decide_linked_page(
        page,
        [
            LinkingEntry(
                entry,
                entry_episode_count,
                frozenset({page.key}) | other_linked_page_keys,
            )
            for entry in entries
        ],
    )


def test_find_unheld_linked_pages_link_to_page_no_entry_holds_returned(
    known_works,
) -> None:
    assert find_unheld_linked_pages(
        _entry(MORI_NO_YOUSEI_SEED_ENTRY),
        [MORI_NO_YOUSEI_KITSU_PAGE],
        known_works,
        LINKED_PROVIDERS,
    ) == frozenset({"mal:anime:35362"})


def test_find_unheld_linked_pages_page_held_by_seed_entry_left_out(known_works) -> None:
    page = BOXER_ANILIST_PAGE.model_copy(update={"links": ("anilist:anime:182615",)})
    assert (
        find_unheld_linked_pages(
            _entry(BOXER_SEED_ENTRY), [page], known_works, LINKED_PROVIDERS
        )
        == frozenset()
    )


def test_find_unheld_linked_pages_provider_not_given_left_out(known_works) -> None:
    assert find_unheld_linked_pages(
        _entry(DOUKYUUSEI_SEED_ENTRY),
        [DOUKYUUSEI_MAL_PAGE],
        known_works,
        LINKED_PROVIDERS,
    ) == frozenset({"anidb:anime:1159"})


def test_find_unheld_linked_pages_link_to_entry_own_page_left_out(known_works) -> None:
    entry = WorkEntry.from_seed_entry(
        {
            **BOXER_SEED_ENTRY,
            "sources": [
                *BOXER_SEED_ENTRY["sources"],
                "https://myanimelist.net/anime/65144",
            ],
        }
    )
    assert (
        find_unheld_linked_pages(
            entry, [BOXER_ANILIST_PAGE], known_works, LINKED_PROVIDERS
        )
        == frozenset()
    )


def test_get_entry_episode_count_most_trusted_page_with_count_returned() -> None:
    entry = _entry(YAMATO_SEED_ENTRY)
    anidb_page = ProviderPage(
        key="anidb:anime:18902",
        address="https://anidb.net/anime/18902",
        episode_count=2,
    )
    assert get_entry_episode_count(entry, [anidb_page, YAMATO_ANILIST_PAGE]) == 1


def test_get_entry_episode_count_trusted_page_without_count_skipped() -> None:
    entry = _entry(YAMATO_SEED_ENTRY)
    anidb_page = ProviderPage(
        key="anidb:anime:18902",
        address="https://anidb.net/anime/18902",
        episode_count=2,
    )
    anilist_page = YAMATO_ANILIST_PAGE.model_copy(update={"episode_count": None})
    assert get_entry_episode_count(entry, [anilist_page, anidb_page]) == 2


def test_get_entry_episode_count_no_page_with_count_returns_none() -> None:
    assert get_entry_episode_count(_entry(BOXER_SEED_ENTRY), []) is None


def test_decide_linked_page_page_agreeing_with_entry_added() -> None:
    decision = _decide(
        YAMATO_MAL_PAGE, _entry(YAMATO_SEED_ENTRY), entry_episode_count=1
    )
    assert (decision.reason, decision.outcome) == (
        LinkedPageReason.CHECKS_PASSED,
        LinkedPageOutcome.ADD,
    )


def test_decide_linked_page_unknown_episode_count_does_not_stop_addition() -> None:
    decision = _decide(BOXER_MAL_PAGE, _entry(BOXER_SEED_ENTRY), entry_episode_count=12)
    assert decision.outcome is LinkedPageOutcome.ADD


def test_decide_linked_page_dead_page_skipped() -> None:
    dead_page = ProviderPage.from_dead_address("https://myanimelist.net/anime/35362")
    decision = _decide(
        dead_page, _entry(MORI_NO_YOUSEI_SEED_ENTRY), entry_episode_count=None
    )
    assert (decision.reason, decision.outcome) == (
        LinkedPageReason.NOT_FOUND,
        LinkedPageOutcome.SKIP,
    )


def test_decide_linked_page_page_linked_from_two_entries_goes_to_review() -> None:
    decision = _decide(
        BOXER_MAL_PAGE,
        _entry(BOXER_SEED_ENTRY),
        _entry(YAMATO_SEED_ENTRY),
        entry_episode_count=None,
    )
    assert (decision.reason, decision.outcome) == (
        LinkedPageReason.SEVERAL_ENTRIES,
        LinkedPageOutcome.REVIEW,
    )


def test_decide_linked_page_entry_holding_provider_goes_to_review() -> None:
    anidb_page = ProviderPage(
        key="anidb:anime:20000", address="https://anidb.net/anime/20000", type="MOVIE"
    )
    decision = _decide(anidb_page, _entry(YAMATO_SEED_ENTRY), entry_episode_count=1)
    assert decision.reason is LinkedPageReason.PROVIDER_IN_ENTRY


def test_decide_linked_page_entry_linking_two_pages_of_provider_goes_to_review() -> (
    None
):
    decision = _decide(
        BOXER_MAL_PAGE,
        _entry(BOXER_SEED_ENTRY),
        entry_episode_count=12,
        other_linked_page_keys=frozenset({"mal:anime:1", "anidb:anime:1"}),
    )
    assert decision.reason is LinkedPageReason.SEVERAL_PAGES_OF_PROVIDER


def test_decide_linked_page_other_linked_page_of_other_provider_does_not_stop_addition() -> (
    None
):
    decision = _decide(
        BOXER_MAL_PAGE,
        _entry(BOXER_SEED_ENTRY),
        entry_episode_count=12,
        other_linked_page_keys=frozenset({"anidb:anime:1"}),
    )
    assert decision.outcome is LinkedPageOutcome.ADD


def test_decide_linked_page_year_differing_from_entry_goes_to_review() -> None:
    anidb_page = ProviderPage(
        key="anidb:anime:1159",
        address="https://anidb.net/anime/1159",
        type="OVA",
        year=2001,
    )
    decision = _decide(anidb_page, _entry(DOUKYUUSEI_SEED_ENTRY), entry_episode_count=4)
    assert decision.reason is LinkedPageReason.DETAILS_DIFFER


def test_decide_linked_page_episode_counts_differing_goes_to_review() -> None:
    decision = _decide(
        BOXER_MAL_PAGE.model_copy(update={"episode_count": 1}),
        _entry(BOXER_SEED_ENTRY),
        entry_episode_count=12,
    )
    assert (decision.reason, decision.entry_episode_count) == (
        LinkedPageReason.EPISODE_COUNTS_DIFFER,
        12,
    )


def test_decide_linked_page_linking_entries_sorted_by_reference_page() -> None:
    decision = _decide(
        BOXER_MAL_PAGE,
        _entry(YAMATO_SEED_ENTRY),
        _entry(BOXER_SEED_ENTRY),
        entry_episode_count=None,
    )
    assert [entry.reference_page for entry in decision.linking_entries] == [
        "anilist:anime:163794",
        "anilist:anime:182615",
    ]


def test_suggest_linked_page_answer_page_with_fewer_episodes_suggests_part() -> None:
    page = BOXER_MAL_PAGE.model_copy(update={"episode_count": 1})
    decision = _decide(page, _entry(BOXER_SEED_ENTRY), entry_episode_count=12)
    assert suggest_linked_page_answer(decision, page) == {
        "mal:anime:65144": "add as part"
    }


def test_suggest_linked_page_answer_page_with_more_episodes_suggests_nothing() -> None:
    page = BOXER_MAL_PAGE.model_copy(update={"episode_count": 24})
    decision = _decide(page, _entry(BOXER_SEED_ENTRY), entry_episode_count=12)
    assert suggest_linked_page_answer(decision, page) == {}


def test_suggest_linked_page_answer_several_entries_suggests_nothing() -> None:
    page = BOXER_MAL_PAGE.model_copy(update={"episode_count": 1})
    decision = _decide(
        page,
        _entry(BOXER_SEED_ENTRY),
        _entry(YAMATO_SEED_ENTRY),
        entry_episode_count=12,
    )
    assert suggest_linked_page_answer(decision, page) == {}


@pytest.mark.parametrize(
    ("answer_text", "expected"),
    [
        ("add", LinkedPageAddition("anilist:anime:163794", is_part_page=False)),
        ("add as part", LinkedPageAddition("anilist:anime:163794", is_part_page=True)),
        (
            "add to anilist:anime:163794",
            LinkedPageAddition("anilist:anime:163794", is_part_page=False),
        ),
        ("skip", SKIP_ANSWER),
        ("add to anilist:anime:1", None),
        ("move", None),
    ],
)
def test_read_linked_page_answer_one_linking_entry(
    answer_text: str, expected: object
) -> None:
    decision = _decide(BOXER_MAL_PAGE, _entry(BOXER_SEED_ENTRY), entry_episode_count=12)
    assert (
        read_linked_page_answer({"mal:anime:65144": answer_text}, decision) == expected
    )


@pytest.mark.parametrize(
    ("answer_text", "expected"),
    [
        ("add", None),
        (
            "add as part to anilist:anime:182615",
            LinkedPageAddition("anilist:anime:182615", is_part_page=True),
        ),
        (
            "add to anilist:anime:163794",
            LinkedPageAddition("anilist:anime:163794", is_part_page=False),
        ),
    ],
)
def test_read_linked_page_answer_several_linking_entries_needs_target(
    answer_text: str, expected: object
) -> None:
    decision = _decide(
        BOXER_MAL_PAGE,
        _entry(BOXER_SEED_ENTRY),
        _entry(YAMATO_SEED_ENTRY),
        entry_episode_count=None,
    )
    assert (
        read_linked_page_answer({"mal:anime:65144": answer_text}, decision) == expected
    )


def test_read_linked_page_answer_add_to_entry_holding_provider_rejected() -> None:
    anidb_page = ProviderPage(
        key="anidb:anime:20000", address="https://anidb.net/anime/20000", type="MOVIE"
    )
    decision = _decide(anidb_page, _entry(YAMATO_SEED_ENTRY), entry_episode_count=1)
    assert read_linked_page_answer({"anidb:anime:20000": "add"}, decision) is None


def test_read_linked_page_answer_part_to_entry_holding_provider_accepted() -> None:
    anidb_page = ProviderPage(
        key="anidb:anime:20000", address="https://anidb.net/anime/20000", type="MOVIE"
    )
    decision = _decide(anidb_page, _entry(YAMATO_SEED_ENTRY), entry_episode_count=1)
    assert read_linked_page_answer(
        {"anidb:anime:20000": "add as part"}, decision
    ) == LinkedPageAddition("anilist:anime:182615", is_part_page=True)


@pytest.mark.parametrize(
    "answer",
    [
        "add",
        {"mal:anime:1": "add"},
        {"mal:anime:65144": "add", "mal:anime:1": "add"},
        {"mal:anime:65144": 1},
    ],
)
def test_read_linked_page_answer_answer_not_naming_page_alone_rejected(
    answer: object,
) -> None:
    decision = _decide(BOXER_MAL_PAGE, _entry(BOXER_SEED_ENTRY), entry_episode_count=12)
    assert read_linked_page_answer(answer, decision) is None
