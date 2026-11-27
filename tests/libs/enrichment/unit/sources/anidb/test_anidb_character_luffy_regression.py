"""Regression test for AniDB character page extraction on Luffy (anidb.net/character/474)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from enrichment.sources.anidb.anidb_character_crawler import _extract_from_html


def test_anidb_character_474_luffy_regression(data_regression: Any) -> None:
    """Extract picture, seiyuu, profile rows, traits, and animeography for Luffy (474)."""
    html = (Path(__file__).parent / "fixtures" / "anidb_character_474.html").read_text(
        encoding="utf-8"
    )
    page = _extract_from_html(html)
    assert page is not None
    assert page.picture == "https://cdn-eu.anidb.net/images/main/201474.jpg"
    assert page.date_of_birth == "05.05"
    assert page.age == "19"
    assert page.blood_type == "F"
    assert page.height == "174 cm"
    assert len(page.seiyuu) == 1
    assert page.seiyuu[0]["name"] == "Tanaka Mayumi"
    assert page.seiyuu[0]["url"] == "https://anidb.net/creator/529"
    assert page.seiyuu[0]["anime_url"] == "https://anidb.net/anime/69"
    data_regression.check(page.model_dump(mode="json"))
