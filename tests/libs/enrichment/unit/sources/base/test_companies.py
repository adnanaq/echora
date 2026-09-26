"""Tests for the per-provider company list every mapper builds.

A provider can credit one company twice - Kitsu files Madhouse as producer and
studio on Death Note - so the collapse has to happen before the merge sees the
record, or the merge inherits a duplicate it did not create.
"""

from common.models.anime import CompanyEntry, CompanyRole
from enrichment.sources.base.companies import companies_from_roles


def test_a_company_credited_twice_becomes_one_entry_with_both_roles() -> None:
    companies = companies_from_roles(
        studios=[CompanyEntry(name="Madhouse")],
        producers=[CompanyEntry(name="Madhouse")],
    )
    assert len(companies) == 1
    assert companies[0].roles == [CompanyRole.STUDIO, CompanyRole.PRODUCER]


def test_each_role_list_keeps_its_own_companies() -> None:
    companies = companies_from_roles(
        studios=[CompanyEntry(name="Toei Animation")],
        producers=[CompanyEntry(name="Fuji TV")],
        licensors=[CompanyEntry(name="Funimation")],
    )
    assert {entry.name: entry.roles for entry in companies} == {
        "Toei Animation": [CompanyRole.STUDIO],
        "Fuji TV": [CompanyRole.PRODUCER],
        "Funimation": [CompanyRole.LICENSOR],
    }


def test_sources_from_both_credits_are_kept() -> None:
    companies = companies_from_roles(
        studios=[
            CompanyEntry(name="Madhouse", sources=["https://kitsu.app/producers/1"])
        ],
        producers=[
            CompanyEntry(name="Madhouse", sources=["https://kitsu.app/producers/2"])
        ],
    )
    assert companies[0].sources == [
        "https://kitsu.app/producers/1",
        "https://kitsu.app/producers/2",
    ]


def test_the_same_link_under_two_roles_is_stored_once() -> None:
    companies = companies_from_roles(
        studios=[
            CompanyEntry(name="Madhouse", sources=["https://kitsu.app/producers/1"])
        ],
        producers=[
            CompanyEntry(name="Madhouse", sources=["https://kitsu.app/producers/1/"])
        ],
    )
    assert companies[0].sources == ["https://kitsu.app/producers/1"]


def test_a_description_is_taken_from_whichever_credit_carries_one() -> None:
    companies = companies_from_roles(
        studios=[CompanyEntry(name="Madhouse")],
        producers=[CompanyEntry(name="Madhouse", description="Animation studio")],
    )
    assert companies[0].description == "Animation studio"


def test_companies_keep_the_order_they_were_first_seen_in() -> None:
    companies = companies_from_roles(
        studios=[CompanyEntry(name="Bones"), CompanyEntry(name="Madhouse")],
        producers=[CompanyEntry(name="Aniplex")],
    )
    assert [entry.name for entry in companies] == ["Bones", "Madhouse", "Aniplex"]


def test_no_roles_supplied_yields_no_companies() -> None:
    assert companies_from_roles() == []
