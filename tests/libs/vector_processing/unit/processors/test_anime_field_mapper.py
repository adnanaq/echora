"""Tests for the text a vector is built from.

This is the only consumer of the company fields, so a change there reaches the
embeddings rather than failing loudly.
"""

import pytest
from common.models.anime import (
    Anime,
    AnimeStatus,
    AnimeType,
    CompanyEntry,
    CompanyRole,
)
from vector_processing.processors.anime_field_mapper import AnimeFieldMapper


def _anime(**overrides) -> Anime:
    fields = {
        "title": "One Piece",
        "sources": ["https://myanimelist.net/anime/21"],
        "status": AnimeStatus.ONGOING,
        "type": AnimeType.TV,
    }
    fields.update(overrides)
    return Anime(**fields)


def _production_text(anime: Anime) -> str:
    text = AnimeFieldMapper().extract_anime_text(anime)
    return next((part for part in text.split("\n") if "Studios:" in part or
                 "Producers:" in part or "Licensors:" in part), "")


def test_each_role_is_labelled() -> None:
    text = _production_text(
        _anime(
            companies=[
                CompanyEntry(name="Toei Animation", roles=[CompanyRole.STUDIO]),
                CompanyEntry(name="Fuji TV", roles=[CompanyRole.PRODUCER]),
                CompanyEntry(name="Funimation", roles=[CompanyRole.LICENSOR]),
            ]
        )
    )
    assert "Studios: Toei Animation" in text
    assert "Producers: Fuji TV" in text
    assert "Licensors: Funimation" in text


def test_a_company_with_two_roles_appears_under_both() -> None:
    text = _production_text(
        _anime(
            companies=[
                CompanyEntry(
                    name="Madhouse",
                    roles=[CompanyRole.STUDIO, CompanyRole.PRODUCER],
                )
            ]
        )
    )
    assert "Studios: Madhouse" in text
    assert "Producers: Madhouse" in text


def test_companies_sharing_a_role_are_listed_together() -> None:
    text = _production_text(
        _anime(
            companies=[
                CompanyEntry(name="Toei Animation", roles=[CompanyRole.STUDIO]),
                CompanyEntry(name="TAP", roles=[CompanyRole.STUDIO]),
            ]
        )
    )
    assert "Studios: Toei Animation, TAP" in text


def test_unknown_roles_are_not_labelled() -> None:
    text = _production_text(
        _anime(companies=[CompanyEntry(name="Mystery Co", roles=[CompanyRole.UNKNOWN])])
    )
    assert text == ""


def test_no_companies_yields_no_production_line() -> None:
    assert _production_text(_anime()) == ""


@pytest.mark.parametrize("name", ["", None])
def test_a_company_without_a_name_is_skipped(name) -> None:
    text = _production_text(
        _anime(
            companies=[
                CompanyEntry(name=name or "", roles=[CompanyRole.STUDIO]),
                CompanyEntry(name="Bones", roles=[CompanyRole.STUDIO]),
            ]
        )
    )
    assert "Studios: Bones" in text
