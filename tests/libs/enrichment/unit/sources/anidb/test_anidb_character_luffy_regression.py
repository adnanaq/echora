from __future__ import annotations

from pathlib import Path
from typing import Any

from enrichment.sources.anidb.anidb_character_crawler import _extract_from_html


def test_anidb_character_474_luffy_regression(
    original_datadir: Path,
    data_regression: Any,
) -> None:
    html = (original_datadir / "anidb_character_474.html").read_text(encoding="utf-8")
    page = _extract_from_html(html)
    assert page is not None
    assert page.name_main == "Monkey D. Luffy"
    assert page.name_kanji == "モンキー・D・ルフィ"
    assert page.picture == "https://cdn.anidb.net/images/main/77294.jpg"
    assert page.date_of_birth == "05.05.????"
    assert page.age == "17-19"
    assert page.blood_type == "B"
    assert page.height == "172-174"
    assert any(
        entry["name"] == "Tanaka Mayumi"
        and entry["url"] == "https://anidb.net/creator/322"
        for entry in page.seiyuu
    )
    assert any(
        entry["title"] == "One Piece" and entry["url"] == "https://anidb.net/anime/69"
        for entry in page.animeography
    )
    data_regression.check(page.model_dump(mode="json"))
