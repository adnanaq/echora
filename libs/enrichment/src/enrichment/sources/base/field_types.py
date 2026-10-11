"""Field types shared by the provider models.

Each type converts what a provider sends into the value its model stores, so
crawlers pass on the text they extracted and mappers read clean fields. Only
conversions that more than one provider needs live here; a conversion one
provider needs is a validator in that provider's model.
"""

import re
from typing import Annotated

import lxml.html
from pydantic import BeforeValidator

# Tags that end a line or a block of text; one between two words leaves a space.
_LINE_TAGS = (
    "br",
    "p",
    "div",
    "li",
    "ul",
    "ol",
    "dl",
    "dt",
    "dd",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "tr",
    "td",
    "table",
    "blockquote",
)
# Stands in for a line or block tag while the text is read out of the parsed HTML.
_LINE_TAG_MARK = "\ue000"  # a Unicode private-use character
_LINE_TAG_MARKS = re.compile(f"{_LINE_TAG_MARK}+")
# A whole number, its thousands grouped by "," or "."; digits that are part of
# a decimal ("8.5") do not match.
_WHOLE_NUMBER = re.compile(r"(?<![\d.,])(\d{1,3}(?:[.,]\d{3})+|\d+)(?![.,]?\d)")


def strip_text_or_none(value: object) -> object:
    """Strip surrounding whitespace from text and turn empty text into None.

    >>> strip_text_or_none("  One Piece ")
    'One Piece'
    >>> strip_text_or_none("   ") is None
    True
    >>> strip_text_or_none(12)
    12
    """
    if isinstance(value, str):
        return value.strip() or None
    return value


def _space_for_line_tags(match: re.Match[str]) -> str:
    """Replace the place of line or block tags with a space only between words.

    ``"worse.<br>Next"`` becomes ``"worse. Next"`` rather than
    ``"worse.Next"``; when whitespace already sits beside the tags, nothing is
    added.
    """
    text = match.string
    before = text[match.start() - 1] if match.start() > 0 else " "
    after = text[match.end()] if match.end() < len(text) else " "
    return "" if before.isspace() or after.isspace() else " "


def html_to_plain_text(value: object) -> object:
    """Turn provider HTML into plain text; empty text becomes None.

    The text is parsed with ``lxml.html``, which removes the tags and decodes
    entities; text that was UTF-8 read as cp1252 (``"Ã©"`` for ``"é"``) is then
    repaired. Line breaks and spacing inside the text are kept.

    The repair applies only when the whole text reads back as UTF-8, so
    correctly written accents are kept. Text that really means ``"Ã©"`` would
    be changed too; no provider text checked so far does.

    >>> html_to_plain_text("<p>Luffy &amp; Zoro</p>")
    'Luffy & Zoro'
    >>> html_to_plain_text("worse.<br><br>(Source: ANN)")
    'worse. (Source: ANN)'
    >>> html_to_plain_text("season of <i>Log Horizon</i>.")
    'season of Log Horizon.'
    >>> html_to_plain_text("<br>") is None
    True
    >>> html_to_plain_text("PokÃ©mon")
    'Pokémon'
    >>> html_to_plain_text("Pokémon")
    'Pokémon'
    """
    if not isinstance(value, str) or not value.strip():
        return strip_text_or_none(value)
    fragment = lxml.html.fragment_fromstring(value, create_parent="div")
    for element in fragment.iter(*_LINE_TAGS):
        element.text = _LINE_TAG_MARK + (element.text or "")
        element.tail = _LINE_TAG_MARK + (element.tail or "")
    text = fragment.text_content()
    text = _LINE_TAG_MARKS.sub(_space_for_line_tags, text).strip()
    try:
        text = text.encode("cp1252").decode("utf-8")
    except UnicodeEncodeError, UnicodeDecodeError:
        pass
    return text or None


def number_from_display_text(value: object) -> object:
    """Read a count or rank from display text such as ``"Rank #1,005"``.

    Anime-Planet groups thousands with ``,`` and AniSearch with ``.``. A
    decimal is not a count, so text whose only number is a decimal becomes
    None, as does text without a number.

    >>> number_from_display_text("Rank #1,005")
    1005
    >>> number_from_display_text("#1.087")
    1087
    >>> number_from_display_text("1,234,567 users")
    1234567
    >>> number_from_display_text("Rank") is None
    True
    >>> number_from_display_text("8.5") is None
    True
    >>> number_from_display_text(7)
    7
    """
    if not isinstance(value, str):
        return value
    match = _WHOLE_NUMBER.search(value)
    return int(re.sub(r"[.,]", "", match.group(1))) if match else None


StrippedText = Annotated[str | None, BeforeValidator(strip_text_or_none)]
PlainText = Annotated[str | None, BeforeValidator(html_to_plain_text)]
NumberFromText = Annotated[int | None, BeforeValidator(number_from_display_text)]
