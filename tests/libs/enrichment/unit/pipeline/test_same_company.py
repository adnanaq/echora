"""Tests for company name folding.

Cases are real spellings taken from the seven providers, not invented ones.
"""

from enrichment.pipeline.same_company import company_base, company_keys


def _merges(*names: str) -> bool:
    return len(set(company_keys(list(names)).values())) == 1


def test_case_and_punctuation_fold() -> None:
    assert _merges("MADHOUSE", "Madhouse")
    assert _merges("P.A. WORKS", "P.A. Works", "P.A.WORKS")
    assert _merges("BONES FILM", "Bones Film", "bones film")
    assert _merges("A Line", "A-Line")


def test_legal_suffixes_fold() -> None:
    assert _merges("Toei Animation Co., Ltd.", "Toei Animation")
    assert _merges("GONZO", "Gonzo", "Gonzo K.K.")
    assert _merges("MADHOUSE Inc.", "Madhouse")
    assert _merges("Nippon Television Network Corporation", "Nippon Television Network")
    assert _merges("Yumeta Co., Ltd.", "Yumeta Company")


def test_plurals_fold_when_both_spellings_are_present() -> None:
    assert _merges("Ashi Production", "Ashi Productions")
    assert _merges(
        "Mushi Production", "Mushi Production Co., Ltd.", "Mushi Productions"
    )
    assert _merges("Pony Canyon Enterprise", "Pony Canyon Enterprises")


def test_a_lone_plural_is_left_alone() -> None:
    assert company_keys(["BONES"])["BONES"] == "bones"
    assert company_keys(["Marvelous"])["Marvelous"] == "marvelous"


def test_different_companies_stay_apart() -> None:
    assert not _merges("Tsuburaya Entertainment", "Tsuburaya Productions")
    assert not _merges("Studio Deen", "Studio Ghibli")
    assert not _merges("Production I.G", "Production Reed")
    assert not _merges("Toei Animation", "Toei Company")


def test_base_drops_nothing_meaningful() -> None:
    assert company_base("Studio Ghibli") == "studio ghibli"
    assert company_base("8bit") == "8bit"
    assert company_base("  Bones  ") == "bones"
