"""Parse MAL episode pages and episode list pages into MAL episode models, with no I/O.

The crawler reads these from pages it fetches, and the mapper from stored page responses,
so both share one parser:

- ``extract_episode_from_html`` reads one episode page into its raw fields;
- ``build_episode_from_raw`` turns those fields into a ``MalEpisode``;
- ``extract_episode_list_from_html`` reads every row of an episode list page.
"""

import re
from typing import Any, cast

from enrichment.sources.base.utils import parse_iso_date
from enrichment.sources.mal.mal_base import parse_duration_seconds
from enrichment.sources.mal.mal_models import (
    EpisodeCharacterRef,
    EpisodeStaffRef,
    EpisodeVARef,
    MalEpisode,
)
from lxml import etree

# XPath of each part of an episode page, and of an episode list page's rows.
MAL_EPISODE_XPATHS: dict[str, str] = {
    "title_header": "//h2[starts-with(normalize-space(.), '#')]",
    "subtitle": "//h2[starts-with(normalize-space(.), '#')]/following-sibling::p[1]",
    "info_box": "//div[contains(@class,'di-tc') and contains(@class,'ar')]",
    "synopsis_container": "//h2[normalize-space()='Synopsis']/parent::div",
    "char_tables": (
        "//table[contains(@class,'fl-l')]"
        "[preceding::h2[1][contains(normalize-space(),'Characters')]]"
    ),
    "staff_tables": (
        "//table[contains(@class,'fl-l')][preceding::h2[1][normalize-space()='Staff']]"
    ),
    "episode_list_rows": "//tr[contains(@class,'episode-list-data')]",
}


def _read_element_text(el: Any) -> str:
    """Read an element's text with its children's text, stripped.

    Args:
        el: An lxml element.

    Returns:
        The element's whole text.
    """
    return "".join(el.itertext()).strip()


def extract_episode_from_html(html: str) -> dict[str, Any] | None:
    """Extract raw field dict from a MAL episode detail page.

    Args:
        html: Full HTML of a MAL episode detail page.

    Returns:
        Dict with ``title_header``, ``subtitle_raw``, ``duration_raw``,
        ``aired_raw``, ``synopsis_raw``, ``characters``, and ``staff`` keys,
        or None if the page title header cannot be found.
    """
    if not html:
        return None
    tree = etree.fromstring(html, etree.HTMLParser(encoding="utf-8"))

    title_els = cast(list[Any], tree.xpath(MAL_EPISODE_XPATHS["title_header"]))
    if not title_els:
        return None
    title_header = _read_element_text(title_els[0])

    sub_els = cast(list[Any], tree.xpath(MAL_EPISODE_XPATHS["subtitle"]))
    subtitle_raw = _read_element_text(sub_els[0]) if sub_els else None

    info_els = cast(list[Any], tree.xpath(MAL_EPISODE_XPATHS["info_box"]))
    info_text = _read_element_text(info_els[0]) if info_els else ""
    dur_m = re.search(r"Duration:\s*(\d{1,2}:\d{2}:\d{2})", info_text)
    air_m = re.search(r"Aired:\s*([A-Za-z]+ \d{1,2},\s*\d{4})", info_text)
    duration_raw = dur_m.group(1) if dur_m else None
    aired_raw = air_m.group(1) if air_m else None

    syn_els = cast(list[Any], tree.xpath(MAL_EPISODE_XPATHS["synopsis_container"]))
    synopsis_raw: str | None = None
    if syn_els:
        syn_text = _read_element_text(syn_els[0])
        syn_m = re.search(r"Synopsis\s*(.*)", syn_text, re.DOTALL)
        synopsis_raw = syn_m.group(1).strip() if syn_m else None

    raw: dict[str, Any] = {
        "title_header": title_header,
        "subtitle_raw": subtitle_raw,
        "duration_raw": duration_raw,
        "aired_raw": aired_raw,
        "synopsis_raw": synopsis_raw,
        "characters": _extract_character_tables(tree),
        "staff": _extract_staff_tables(tree),
    }
    return raw


def _extract_character_tables(tree: Any) -> list[dict[str, Any]]:
    """Extract the raw fields of each character table on an episode page.

    A table without a character link is left out.

    Args:
        tree: Parsed HTML tree of a MAL episode detail page.

    Returns:
        One dict per character with ``char_name``, ``char_url``, ``role`` and
        ``voice_actors_html``.
    """
    characters: list[dict[str, Any]] = []
    for character_table in cast(
        list[Any], tree.xpath(MAL_EPISODE_XPATHS["char_tables"])
    ):
        name_els = cast(
            list[Any],
            character_table.xpath(
                ".//a[contains(@class,'fw-b')][contains(@href,'/character/')]"
            ),
        )
        if not name_els:
            continue
        role_m = re.search(
            r"\b(Main|Supporting)\b", _read_element_text(character_table)
        )
        va_els = cast(list[Any], character_table.xpath(".//p[contains(@class,'pb8')]"))
        characters.append(
            {
                "char_name": _read_element_text(name_els[0]),
                "char_url": name_els[0].get("href", ""),
                "role": role_m.group(1) if role_m else None,
                "voice_actors_html": etree.tostring(va_els[0], encoding="unicode")
                if va_els
                else "",
            }
        )
    return characters


def _extract_staff_tables(tree: Any) -> list[dict[str, Any]]:
    """Extract the raw fields of each staff table on an episode page.

    A table without a person link is left out.

    Args:
        tree: Parsed HTML tree of a MAL episode detail page.

    Returns:
        One dict per staff member with ``name``, ``person_url`` and ``role``.
    """
    staff: list[dict[str, Any]] = []
    for staff_table in cast(list[Any], tree.xpath(MAL_EPISODE_XPATHS["staff_tables"])):
        name_els = cast(
            list[Any],
            staff_table.xpath(
                ".//a[contains(@class,'fw-b')][contains(@href,'/people/')]"
            ),
        )
        if not name_els:
            continue
        role_els = cast(list[Any], staff_table.xpath(".//p[contains(@class,'pr12')]"))
        staff.append(
            {
                "name": _read_element_text(name_els[0]),
                "person_url": name_els[0].get("href", ""),
                "role": _read_element_text(role_els[0]) if role_els else None,
            }
        )
    return staff


def _parse_title_info(
    header_text: str | None,
    subtitle_text: str | None,
    episode_number: int,
) -> tuple[str, str | None, str | None, bool, bool]:
    """Parse MAL episode header into (title, title_japanese, title_romaji, is_filler, is_recap).

    header_text comes from ``//h2[starts-with(normalize-space(.), '#')]`` and
    subtitle_text from its first p sibling. lxml itertext() emits the filler/recap
    badge text as trailing whitespace — whitespace collapsing handles it.

    Args:
        header_text: Raw h2 text, e.g. "#1 - I'm Luffy! Filler"
        subtitle_text: Raw subtitle text, e.g. "Romaji Title (Japanese Title)"
        episode_number: Fallback episode number for title when header_text is None.

    Returns:
        Tuple of (title, title_japanese, title_romaji, is_filler, is_recap).
    """
    is_filler = False
    is_recap = False
    title = f"Episode {episode_number}"

    if header_text:
        header_text = " ".join(header_text.split())
        m = re.match(r"^#\d+\s*-\s*(.*)", header_text)
        raw_title = m.group(1) if m else header_text

        filler_m = re.search(r"\s+\bFiller\b\s*$", raw_title, re.IGNORECASE)
        recap_m = re.search(r"\s+\bRecap\b\s*$", raw_title, re.IGNORECASE)
        if filler_m:
            is_filler = True
            raw_title = raw_title[: filler_m.start()]
        elif recap_m:
            is_recap = True
            raw_title = raw_title[: recap_m.start()]

        title = raw_title.strip()

    title_romaji, title_japanese = _split_romaji_and_japanese_titles(subtitle_text)
    return title, title_japanese, title_romaji, is_filler, is_recap


def _split_romaji_and_japanese_titles(
    subtitle_text: str | None,
) -> tuple[str | None, str | None]:
    """Split MAL's ``Romaji Title (Japanese Title)`` text into its two titles.

    Args:
        subtitle_text: The text under an episode title, or ``None`` when absent.

    Returns:
        Tuple of ``(title_romaji, title_japanese)``; a title MAL does not give is
        ``None``.
    """
    if not subtitle_text or not subtitle_text.strip():
        return None, None
    subtitle_text = " ".join(subtitle_text.split())
    japanese_match = re.search(r"\(([^)]+)\)\s*$", subtitle_text)
    if japanese_match is None:
        return subtitle_text, None
    title_romaji = subtitle_text[: japanese_match.start()].strip() or None
    return title_romaji, japanese_match.group(1).strip()


def extract_episode_list_from_html(html: str) -> list[MalEpisode]:
    """Extract every episode row of a MAL episode list page.

    Each row gives the episode's number, its page URL with the work's slug, its
    titles, its air date, and MAL's ``Filler`` or ``Recap`` mark.

    Args:
        html: Full HTML of a MAL episode list page.

    Returns:
        One ``MalEpisode`` per row with an episode number and a page link, in list
        order.
    """
    if not html:
        return []
    tree = etree.fromstring(html.encode(), etree.HTMLParser(encoding="utf-8"))
    episodes: list[MalEpisode] = []
    for row in cast(list[Any], tree.xpath(MAL_EPISODE_XPATHS["episode_list_rows"])):
        number_text = str(row.xpath("string(td[contains(@class,'episode-number')])"))
        title_links = cast(
            list[Any], row.xpath("td[contains(@class,'episode-title')]/a[@href]")
        )
        if not number_text.strip().isdigit() or not title_links:
            continue
        marks = {
            mark.strip()
            for mark in cast(
                list[str],
                row.xpath(
                    "td[contains(@class,'episode-title')]"
                    "/span[contains(@class,'icon-episode-type-bg')]/text()"
                ),
            )
        }
        title_romaji, title_japanese = _split_romaji_and_japanese_titles(
            str(
                row.xpath(
                    "string(td[contains(@class,'episode-title')]"
                    "/span[contains(@class,'di-ib')])"
                )
            )
        )
        episode_number = int(number_text.strip())
        episodes.append(
            MalEpisode(
                episode_number=episode_number,
                source=title_links[0].get("href"),
                title=_read_element_text(title_links[0]) or f"Episode {episode_number}",
                title_japanese=title_japanese,
                title_romaji=title_romaji,
                aired=parse_iso_date(
                    str(row.xpath("string(td[contains(@class,'episode-aired')])"))
                ),
                filler="Filler" in marks,
                recap="Recap" in marks,
            )
        )
    return episodes


def _parse_episode_characters(
    raw_items: list[dict[str, Any]] | None,
) -> list[EpisodeCharacterRef]:
    """Parse raw character dicts from HTML extraction into EpisodeCharacterRef list.

    Args:
        raw_items: List of dicts with ``char_name``, ``char_url``, ``role``,
            and ``voice_actors_html`` keys, as produced by
            ``extract_episode_from_html``.

    Returns:
        List of EpisodeCharacterRef, skipping entries with missing name/URL
        or non-character URLs.
    """
    if not raw_items:
        return []

    characters: list[EpisodeCharacterRef] = []
    for item in raw_items:
        name = (item.get("char_name") or "").strip()
        url = (item.get("char_url") or "").strip()
        if not name or not url:
            continue

        char_id_match = re.search(r"/character/(\d+)", url)
        if not char_id_match:
            continue
        char_id = int(char_id_match.group(1))

        role = item.get("role") or "Supporting"

        voice_actors_html = item.get("voice_actors_html") or ""
        voice_actors: list[EpisodeVARef] = []
        if voice_actors_html:
            va_matches = re.finditer(
                r'<a[^>]+href="[^"]*/people/(\d+)[^"]*"[^>]*>(.*?)</a>\s*\(([^)]+)\)',
                voice_actors_html,
                re.DOTALL,
            )
            for m in va_matches:
                person_id = int(m.group(1))
                va_name = re.sub(r"<[^>]+>", "", m.group(2)).strip()
                language = m.group(3).strip()
                voice_actors.append(
                    EpisodeVARef(
                        person_id=person_id,
                        name=va_name,
                        language=language,
                    )
                )

        characters.append(
            EpisodeCharacterRef(
                mal_id=char_id, name=name, role=role, voice_actors=voice_actors
            )
        )

    return characters


def _parse_episode_staff(
    raw_items: list[dict[str, Any]] | None,
) -> list[EpisodeStaffRef]:
    """Parse raw staff dicts from HTML extraction into EpisodeStaffRef list.

    Args:
        raw_items: List of dicts with ``name``, ``person_url``, and ``role``
            keys, as produced by ``extract_episode_from_html``.

    Returns:
        List of EpisodeStaffRef. Entries with no role are included with
        ``role=None`` — MAL leaves the role field blank for some credits
        (e.g. English dub talent). Entries with missing name or non-people
        URLs are skipped.
    """
    if not raw_items:
        return []

    staff: list[EpisodeStaffRef] = []
    for item in raw_items:
        name = (item.get("name") or "").strip()
        url = (item.get("person_url") or "").strip()
        if not name or not url:
            continue

        person_match = re.search(r"/people/(\d+)", url)
        if not person_match:
            continue
        person_id = int(person_match.group(1))

        role = (item.get("role") or "").strip() or None
        staff.append(EpisodeStaffRef(person_id=person_id, name=name, role=role))

    return staff


def build_episode_from_raw(
    raw: dict[str, Any], episode_number: int, url: str
) -> MalEpisode:
    """Construct a MalEpisode from a raw extraction dict.

    Args:
        raw: Raw fields from ``extract_episode_from_html``; a saved ``_url`` wins
            over ``url``.
        episode_number: The episode's number, used when the header holds no title.
        url: The episode page's address.

    Returns:
        The episode as a ``MalEpisode``.
    """
    saved_url = raw.pop("_url", url)

    title, title_japanese, title_romaji, is_filler, is_recap = _parse_title_info(
        raw.get("title_header"),
        raw.get("subtitle_raw"),
        episode_number,
    )
    aired = parse_iso_date(raw.get("aired_raw"))
    duration = parse_duration_seconds(raw.get("duration_raw"))
    syn_raw = raw.get("synopsis_raw")
    synopsis = " ".join(syn_raw.split()).strip() if syn_raw else None
    if synopsis:
        source_idx = synopsis.find("(Source:")
        if source_idx != -1:
            synopsis = synopsis[:source_idx].strip() or None
    if synopsis and "doesn't seem to have a synopsis" in synopsis:
        synopsis = None
    characters = _parse_episode_characters(raw.get("characters"))
    staff = _parse_episode_staff(raw.get("staff"))

    return MalEpisode(
        episode_number=episode_number,
        source=saved_url,
        title=title,
        title_japanese=title_japanese,
        title_romaji=title_romaji,
        synopsis=synopsis,
        aired=aired,
        duration=duration,
        filler=is_filler,
        recap=is_recap,
        characters=characters,
        staff=staff,
    )
