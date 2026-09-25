"""Undo what Google's in-text ads insert into a page's own content.

After a page loads, Google's in-text ad feature changes it in two ways that
reach our extractors:

**It turns chosen words into ad links.** It picks "Anime" in the "Anime Roles"
heading, turning

    <h3>Anime Roles</h3>

into

    <h3><a href="#" class="google-anno" ...><svg>...</svg>&nbsp;<span
    class="google-anno-t">Anime</span></a> Roles</h3>

so an extractor looking for that heading finds nothing and drops every role.
The replaced word is kept inside the link, so it is put back exactly.

**It inserts search-suggestion chips into the text.** A chip such as

    <div class="google-anno-skip google-anno-sc" aria-label="Anime &amp; Manga">
    ...<span>Anime</span>...<span>&amp; Manga</span></div>

lands inside the character's description, right after its last sentence, and
its words were read as part of the description. A chip
is not the page's content, so it is removed whole. No chip contains another
``div``, so the first closing ``div`` after one is its own.

Restoring the page once, before extraction, protects every extractor on it,
including against words or places the ad script has not picked yet, rather
than each extractor learning to tolerate ad markup.

The script's other additions, a side panel (``id="google-anno-sa"``) and
``goog-rentries`` containers, were found only outside every section we parse.
They contain nested ``div`` elements, so a pattern could not remove them safely,
and there is nothing to gain; they are left alone.
"""

from __future__ import annotations

import re

_AD_LINK = re.compile(
    r'<a\b[^>]*\bclass="google-anno"[^>]*>(.*?)</a>', re.DOTALL | re.IGNORECASE
)
_AD_LINKED_WORD = re.compile(
    r'<span\b[^>]*\bclass="google-anno-t"[^>]*>(.*?)</span>',
    re.DOTALL | re.IGNORECASE,
)
_SUGGESTION_CHIP = re.compile(
    r'<div\b[^>]*\bclass="[^"]*\bgoogle-anno-sc\b[^"]*"[^>]*>.*?</div>',
    re.DOTALL | re.IGNORECASE,
)
_MARKUP_AND_SPACING = re.compile(r"<[^>]+>|&nbsp;")


def remove_ad_annotations(html: str) -> str:
    """Put back every ad-linked word and remove every suggestion chip.

    Args:
        html: A rendered page, possibly changed by Google's in-text ads.

    Returns:
        The page with each ad link replaced by its original word and each
        suggestion chip removed. A page without either is returned unchanged.
    """

    def original_word(ad_link: re.Match[str]) -> str:
        word = _AD_LINKED_WORD.search(ad_link.group(1))
        if word:
            return word.group(1)
        return _MARKUP_AND_SPACING.sub("", ad_link.group(1))

    return _SUGGESTION_CHIP.sub("", _AD_LINK.sub(original_word, html))
