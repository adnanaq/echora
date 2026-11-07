import pytest
from review_page.models import InvalidNamedPageKeyReason


def _render_typed_option(client, query: str, case_name: str = "") -> str:
    return client.post(
        "/fragments/named-entry-option",
        data={
            "page_key": "anidb:anime:19188",
            "mark": "",
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
