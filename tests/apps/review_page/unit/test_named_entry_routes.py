import pytest
from review_page.models import InvalidNamedPageKeyReason
from shared_proto.v1 import review_cases_pb2
from work_matching import TargetEntrySuggestion

GOCHUUMON_ENTRY = review_cases_pb2.TargetEntrySuggestion(
    entry_reference_page="mal:anime:21273",
    title="Gochuumon wa Usagi desu ka?",
    providers=["anilist", "mal"],
)


def _render_typed_option(client, query: str, case_name: str = "") -> str:
    return client.post(
        "/fragments/named-entry-option",
        data={
            "page_key": "anidb:anime:19188",
            "query": query,
            "case_name": case_name,
        },
    ).text


def test_render_named_entry_option_valid_address_offers_its_key(client) -> None:
    option = _render_typed_option(client, " https://myanimelist.net/anime/29787 ")
    assert (
        "Use mal:anime:29787" in option,
        '"named_entry": "mal:anime:29787"' in option,
        'aria-disabled="true"' in option,
    ) == (True, True, False)


@pytest.mark.parametrize(
    "invalid_reason",
    [InvalidNamedPageKeyReason.DEAD_PAGE, InvalidNamedPageKeyReason.UNKNOWN_PAGE],
)
def test_render_named_entry_option_invalid_key_disabled_with_reason(
    client, source, invalid_reason
) -> None:
    source.invalid_named_page_key_reasons = {"mal:anime:29787": invalid_reason}
    option = _render_typed_option(client, "mal:anime:29787")
    assert (
        'aria-disabled="true"' in option,
        invalid_reason.value in option,
        "hx-post" in option,
        "mal:anime:29787" in option,
    ) == (True, True, False, False)


def test_render_named_entry_option_text_not_page_key_disabled(client) -> None:
    option = _render_typed_option(client, "kusuriya:season 3")
    assert (
        'aria-disabled="true"' in option,
        InvalidNamedPageKeyReason.NOT_PAGE_KEY.value in option,
    ) == (True, True)


def test_render_named_entry_option_key_of_case_entry_disabled(
    client, review_cases
) -> None:
    option = _render_typed_option(
        client, "mal:anime:61115", review_cases["gochuumon_movie"].case_name
    )
    assert (
        'aria-disabled="true"' in option,
        InvalidNamedPageKeyReason.IN_CASE.value in option,
    ) == (True, True)


@pytest.mark.parametrize("query", ["   ", "part 2"])
def test_render_named_entry_option_search_words_render_nothing(client, query) -> None:
    assert _render_typed_option(client, query) == ""


def test_render_named_entry_option_found_entry_shows_title_and_reference_page(
    client, source
) -> None:
    source.named_entries = {"anilist:anime:21273": GOCHUUMON_ENTRY}
    option = _render_typed_option(client, "https://anilist.co/anime/21273/Gochiusa/")
    assert (
        "Gochuumon wa Usagi desu ka?" in option,
        "anilist:anime:21273 · seed entry mal:anime:21273" in option,
        '"named_entry": "anilist:anime:21273"' in option,
    ) == (True, True, True)


@pytest.mark.parametrize(
    ("query", "is_option_rendered"),
    [("mal:anime:21273", False), ("https://myanimelist.net/anime/21273", True)],
)
def test_render_named_entry_option_entry_suggested_for_typed_text_renders_nothing(
    client, source, review_cases, query, is_option_rendered
) -> None:
    case = review_cases["gochuumon_movie"].model_copy(
        update={
            "target_entry_suggestions": (
                TargetEntrySuggestion(
                    entry_reference_page="mal:anime:21273",
                    title="Gochuumon wa Usagi desu ka?",
                    providers=("mal",),
                ),
            )
        }
    )
    source.cases[case.case_name] = case
    source.named_entries = {"mal:anime:21273": GOCHUUMON_ENTRY}
    option = _render_typed_option(client, query, case.case_name)
    assert ("Gochuumon wa Usagi desu ka?" in option) is is_option_rendered
