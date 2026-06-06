"""AniDB XML parser — stateless, no I/O.

Single public function: parse_anime_xml(xml_content) -> AniDBAnime.

Extraction rules:
- Field names follow AniDBAnime model (XML tag names where canonical name differs)
- picture/seiyuu picture stored as raw filenames — CDN prefix added in mapper
- status/season/year NOT computed here — belong in mapper
- findall("seiyuu") not find() — XML allows multiple seiyuu per character
"""

import logging
from typing import TypedDict
from xml.etree.ElementTree import Element

import defusedxml.ElementTree as ET

from enrichment.sources.anidb.anidb_models import (
    AniDBAnime,
    AniDBCategory,
    AniDBCharacter,
    AniDBCreator,
    AniDBEpisode,
    AniDBExternalResource,
    AniDBRatings,
    AniDBRelatedAnime,
    AniDBSeiyuu,
)

logger = logging.getLogger(__name__)

_XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"
_LANG_NORMALIZE = {"x-jat": "romaji"}
_CRUNCHYROLL_RESOURCE_TYPE = "28"


class _TitlesDict(TypedDict):
    title: str | None
    title_english: str | None
    title_japanese: str | None
    synonyms: list[str]
    title_others: dict[str, str]


def parse_anime_xml(xml_content: str) -> AniDBAnime:
    """Parse AniDB HTTP XML API response into an AniDBAnime model.

    Args:
        xml_content: Raw XML string from the AniDB HTTP API.

    Returns:
        Fully populated AniDBAnime model with all available fields extracted.

    Raises:
        ValueError: If the XML cannot be parsed, the root element is not
            ``<anime>``, or the ``id`` attribute is missing or non-numeric.
    """
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        raise ValueError(f"AniDB XML parse error: {e}") from e

    if root.tag != "anime":
        raise ValueError(f"Expected <anime> root element, got <{root.tag}>")

    anime_id_raw = root.get("id")
    if not anime_id_raw or not anime_id_raw.isdigit():
        raise ValueError(f"Missing or invalid 'id' attribute on <anime>: {anime_id_raw!r}")

    return AniDBAnime(
        id=int(anime_id_raw),
        restricted=root.get("restricted", "false").lower() == "true",
        type=_text(root, "type"),
        episode_count=_int(root, "episodecount"),
        start_date=_text(root, "startdate"),
        end_date=_text(root, "enddate"),
        description=_text(root, "description"),
        picture=_text(root, "picture"),
        url=_text(root, "url"),
        **_parse_titles(root),
        categories=_parse_categories(root),
        characters=_parse_characters(root),
        creators=_parse_creators(root),
        episodes=_parse_episodes(root),
        related_anime=_parse_related_anime(root),
        resources=_parse_resources(root),
        tags=_parse_tags(root),
        ratings=_parse_ratings(root),
    )


# =============================================================================
# TITLE PARSING
# =============================================================================


def _parse_titles(root: Element) -> _TitlesDict:
    """Extract all title variants from the ``<titles>`` element.

    Classifies each ``<title>`` by its ``type`` attribute:
        - ``main``    → primary display title
        - ``official`` + ``lang="en"`` → English title
        - ``official`` + ``lang="ja"`` → Japanese title
        - ``official`` + other lang   → goes into title_others dict (BCP 47 key)
        - ``synonym`` / ``short``     → informal synonyms list

    Args:
        root: Root ``<anime>`` XML element.

    Returns:
        Dict with keys ``title``, ``title_english``, ``title_japanese``,
        ``synonyms``, and ``title_others``, ready to be unpacked into
        AniDBAnime constructor via ``**``.
    """
    title: str | None = None
    title_english: str | None = None
    title_japanese: str | None = None
    synonyms: list[str] = []
    title_others: dict[str, str] = {}

    titles_elem = root.find("titles")
    if titles_elem is not None:
        for title_elem in titles_elem.findall("title"):
            title_type = title_elem.get("type", "")
            lang = title_elem.get(_XML_LANG, "")
            text = title_elem.text

            if not text:
                continue

            if title_type == "main":
                title = text
            elif title_type == "official":
                if lang == "en":
                    title_english = text
                elif lang == "ja":
                    title_japanese = text
                else:
                    title_others[lang] = text
            elif title_type in ("synonym", "short"):
                synonyms.append(text)

    return {
        "title": title,
        "title_english": title_english,
        "title_japanese": title_japanese,
        "synonyms": synonyms,
        "title_others": title_others,
    }


# =============================================================================
# ARRAY PARSERS
# =============================================================================


def _parse_categories(root: Element) -> list[AniDBCategory]:
    """Extract categories from the ``<categories>`` element.

    Each ``<category>`` must have a ``<name>`` child element; entries without
    one are skipped.

    Args:
        root: Root ``<anime>`` XML element.

    Returns:
        List of AniDBCategory models. Empty list if element is absent.
    """
    categories_elem = root.find("categories")
    if categories_elem is None:
        return []

    result = []
    for category in categories_elem.findall("category"):
        name_elem = category.find("name")
        if name_elem is None or not name_elem.text:
            continue
        result.append(
            AniDBCategory(
                id=category.get("id"),
                name=name_elem.text,
                weight=int(category.get("weight", 0)),
                hentai=category.get("hentai", "false").lower() == "true",
            )
        )
    return result


def _parse_characters(root: Element) -> list[AniDBCharacter]:
    """Extract characters from the ``<characters>`` element.

    Uses ``findall("seiyuu")`` instead of ``find("seiyuu")`` because the XML
    schema allows multiple ``<seiyuu>`` elements per character.

    Args:
        root: Root ``<anime>`` XML element.

    Returns:
        List of AniDBCharacter models. Empty list if element is absent.
    """
    characters_elem = root.find("characters")
    if characters_elem is None:
        return []

    result = []
    for character in characters_elem.findall("character"):
        character_type_elem = character.find("charactertype")
        seiyuu_list = [
            AniDBSeiyuu(
                id=_safe_int(seiyuu_elem.get("id")),
                name=seiyuu_elem.text,
                picture=seiyuu_elem.get("picture"),
            )
            for seiyuu_elem in character.findall("seiyuu")
        ]
        rating, rating_votes = _rating_pair(character.find("rating"))
        result.append(
            AniDBCharacter(
                id=_safe_int(character.get("id")),
                type=character.get("type"),
                name=_text(character, "name"),
                gender=_text(character, "gender"),
                character_type=character_type_elem.text if character_type_elem is not None else None,
                character_type_id=(
                    _safe_int(character_type_elem.get("id"))
                    if character_type_elem is not None
                    else None
                ),
                description=_text(character, "description"),
                picture=_text(character, "picture"),
                rating=rating,
                rating_votes=rating_votes,
                seiyuu=seiyuu_list,
            )
        )
    return result


def _parse_creators(root: Element) -> list[AniDBCreator]:
    """Extract creators from the ``<creators>`` element.

    Each ``<name>`` child represents one creator entry. The ``type`` attribute
    holds the role string (e.g. "Direction", "Original Work", "Music").

    Args:
        root: Root ``<anime>`` XML element.

    Returns:
        List of AniDBCreator models. Empty list if element is absent.
    """
    creators_elem = root.find("creators")
    if creators_elem is None:
        return []

    return [
        AniDBCreator(
            id=_safe_int(creator.get("id")),
            name=creator.text,
            role=creator.get("type"),
        )
        for creator in creators_elem.findall("name")
    ]


def _parse_episodes(root: Element) -> list[AniDBEpisode]:
    """Extract episodes from the ``<episodes>`` element.

    Episode number parsing depends on ``episode_type``:
        - Type 1 (regular) → integer episode number
        - All others        → raw string (e.g. "S1", "C3", "T1")

    Streaming links are extracted from per-episode ``<resources>`` (type 28 =
    Crunchyroll, episode-level identifier used to build watch URLs).

    Args:
        root: Root ``<anime>`` XML element.

    Returns:
        List of AniDBEpisode models for all episode types. Empty list if
        element is absent. Callers should filter by ``episode_type`` as needed.
    """
    episodes_elem = root.find("episodes")
    if episodes_elem is None:
        return []

    result = []
    for episode in episodes_elem.findall("episode"):
        epno_elem = episode.find("epno")
        length_elem = episode.find("length")

        episode_type: int | None = None
        episode_number: int | str | None = None
        if epno_elem is not None:
            episode_type = _safe_int(epno_elem.get("type"))
            if epno_elem.text:
                if episode_type == 1:
                    try:
                        episode_number = int(epno_elem.text)
                    except ValueError:
                        episode_number = epno_elem.text
                else:
                    episode_number = epno_elem.text

        titles: dict[str, str] = {}
        for title_elem in episode.findall("title"):
            lang = title_elem.get(_XML_LANG, "unknown")
            if title_elem.text:
                titles[_LANG_NORMALIZE.get(lang, lang)] = title_elem.text

        streaming: dict[str, str] = {}
        episode_resources_elem = episode.find("resources")
        if episode_resources_elem is not None:
            for resource in episode_resources_elem.findall("resource"):
                if resource.get("type") == _CRUNCHYROLL_RESOURCE_TYPE:
                    external_entity = resource.find("externalentity")
                    if external_entity is not None:
                        identifier_elem = external_entity.find("identifier")
                        if identifier_elem is not None and identifier_elem.text:
                            streaming["crunchyroll"] = (
                                f"https://www.crunchyroll.com/watch/{identifier_elem.text}"
                            )

        rating, rating_votes = _rating_pair(episode.find("rating"))
        result.append(
            AniDBEpisode(
                id=_safe_int(episode.get("id")),
                episode_number=episode_number,
                episode_type=episode_type,
                length=(
                    int(length_elem.text)
                    if length_elem is not None and length_elem.text and length_elem.text.isdigit()
                    else None
                ),
                airdate=_text(episode, "airdate"),
                rating=rating,
                rating_votes=rating_votes,
                summary=_text(episode, "summary"),
                titles=titles,
                streaming=streaming,
            )
        )
    return result


def _parse_related_anime(root: Element) -> list[AniDBRelatedAnime]:
    """Extract related anime entries from the ``<relatedanime>`` element.

    Each ``<anime>`` child requires both a numeric ``id`` attribute and a
    ``type`` attribute (relation string). Entries missing either are skipped.

    Args:
        root: Root ``<anime>`` XML element.

    Returns:
        List of AniDBRelatedAnime models. Empty list if element is absent.
    """
    related_elem = root.find("relatedanime")
    if related_elem is None:
        return []

    result = []
    for related_anime in related_elem.findall("anime"):
        related_id_raw = related_anime.get("id")
        relation_type = related_anime.get("type")
        if not related_id_raw or not related_id_raw.isdigit() or not relation_type:
            continue
        result.append(
            AniDBRelatedAnime(
                id=int(related_id_raw),
                relation_type=relation_type,
                title=related_anime.text.strip() if related_anime.text else None,
            )
        )
    return result


def _parse_resources(root: Element) -> list[AniDBExternalResource]:
    """Extract external resource entries from the ``<resources>`` element.

    Each ``<resource>`` has a numeric ``type`` attribute identifying the
    platform (e.g. "1" = ANN, "2" = MAL, "6" = Wikipedia EN). Within each
    resource, ``<externalentity>`` children may contain ``<url>`` and/or
    ``<identifier>`` elements.

    Args:
        root: Root ``<anime>`` XML element.

    Returns:
        List of AniDBExternalResource models for all resource types found.
        Empty list if element is absent.
    """
    resources_elem = root.find("resources")
    if resources_elem is None:
        return []

    result = []
    for resource in resources_elem.findall("resource"):
        resource_type = resource.get("type")
        if not resource_type:
            continue

        urls: list[str] = []
        identifiers: list[str] = []
        for external_entity in resource.findall("externalentity"):
            url_elem = external_entity.find("url")
            if url_elem is not None and url_elem.text:
                urls.append(url_elem.text)
            for identifier_elem in external_entity.findall("identifier"):
                if identifier_elem.text:
                    identifiers.append(identifier_elem.text)

        result.append(
            AniDBExternalResource(type=resource_type, urls=urls, identifiers=identifiers)
        )
    return result


def _parse_tags(root: Element) -> list[str]:
    """Extract tag name strings from the ``<tags>`` element.

    Only the ``<name>`` child text of each ``<tag>`` is collected; weight,
    id, and hentai attributes are ignored here (categories cover hentai).

    Args:
        root: Root ``<anime>`` XML element.

    Returns:
        List of tag name strings. Empty list if element is absent.
    """
    tags_elem = root.find("tags")
    if tags_elem is None:
        return []
    return [
        name_elem.text
        for tag in tags_elem.findall("tag")
        if (name_elem := tag.find("name")) is not None and name_elem.text
    ]


def _parse_ratings(root: Element) -> AniDBRatings | None:
    """Extract permanent, temporary, and review ratings from ``<ratings>``.

    Each sub-element (``<permanent>``, ``<temporary>``, ``<review>``) carries
    the score as element text and the vote count as a ``count`` attribute.

    Args:
        root: Root ``<anime>`` XML element.

    Returns:
        AniDBRatings model if the element exists, otherwise None.
    """
    ratings_elem = root.find("ratings")
    if ratings_elem is None:
        return None

    def _rating_value(tag: str) -> tuple[float | None, int]:
        elem = ratings_elem.find(tag)
        if elem is None:
            return None, 0
        score = float(elem.text) if elem.text else None
        count = int(elem.get("count", 0))
        return score, count

    permanent, permanent_count = _rating_value("permanent")
    temporary, temporary_count = _rating_value("temporary")
    review, review_count = _rating_value("review")

    return AniDBRatings(
        permanent=permanent,
        permanent_count=permanent_count,
        temporary=temporary,
        temporary_count=temporary_count,
        review=review,
        review_count=review_count,
    )


# =============================================================================
# HELPERS
# =============================================================================


def _text(elem: Element, tag: str) -> str | None:
    """Return the text content of a direct child element, or None if absent.

    Args:
        elem: Parent XML element to search within.
        tag: Tag name of the child element to find.

    Returns:
        Text content of the child element, or None if the child does not exist.
    """
    child = elem.find(tag)
    return child.text if child is not None else None


def _int(elem: Element, tag: str, default: int = 0) -> int:
    """Return the integer text content of a direct child element.

    Args:
        elem: Parent XML element to search within.
        tag: Tag name of the child element to find.
        default: Value to return if the child is absent or non-numeric.

    Returns:
        Parsed integer value, or ``default`` if not found or not parseable.
    """
    child = elem.find(tag)
    if child is not None and child.text and child.text.isdigit():
        return int(child.text)
    return default


def _safe_int(value: str | None) -> int | None:
    """Parse a string attribute value to int, or None if absent or non-numeric."""
    return int(value) if value and value.isdigit() else None


def _rating_pair(elem: Element | None) -> tuple[float | None, int]:
    """Extract score and vote count from a ``<rating votes="N">score</rating>`` element."""
    if elem is None:
        return None, 0
    return (float(elem.text) if elem.text else None), int(elem.get("votes", 0))
