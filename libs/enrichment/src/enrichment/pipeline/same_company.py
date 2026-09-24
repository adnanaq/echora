"""Decide when two differently-written company names are the same company.

Providers write one company several ways, and so does a single provider: Kitsu
stores ``MADHOUSE`` as producer 5 and ``Madhouse`` as producer 917, the second
carrying an auto-suffixed slug because the name already existed. Across 276
anime and all seven providers, 235 names arrive under more than one spelling,
and 234 of those span providers, so this is a merge-time problem rather than
one any single mapper can fix.

Three layers, each measured:

* ``fold_for_comparison`` - case, NFKC and punctuation, shared with categories
  and synonyms. Catches ``MADHOUSE`` against ``Madhouse`` and ``P.A.WORKS``
  against ``P.A. Works``.
* legal suffixes - ``Toei Animation Co., Ltd.`` against ``Toei Animation``,
  ``Gonzo K.K.`` against ``GONZO``. Over Kitsu's 1,975 producers this merges 52
  groups.
* plurals, pairwise - ``Ashi Production`` against ``Ashi Productions``.

Industry words are deliberately not stripped. Removing ``production`` or
``entertainment`` merges ``Tsuburaya Entertainment`` with ``Tsuburaya
Productions``, which are different companies.

The plural rule fires only when both spellings are present among the names
being compared. Dropping a trailing ``s`` unconditionally would turn ``BONES``
into ``bone``, which is safe today only because no company is called ``Bone``.
It also cannot live in ``fold_for_comparison``: that fold is shared, and
changing it moved categories on 12 of 75 sample anime and broke the
hand-checked ``superpowers`` entry in ``same_word``.
"""

from __future__ import annotations

import re

from enrichment.utils.text_utils import fold_for_comparison

# Forms of incorporation, never part of the name a person would use.
_LEGAL_SUFFIX = re.compile(
    r"\b(co\.?,? ?ltd\.?|ltd|inc|k ?k|corp|corporation|llc|gmbh|company)\b"
)


def company_base(name: str) -> str:
    """Return the name with case, punctuation and legal suffixes removed.

    Args:
        name: A company name as a provider wrote it.

    Returns:
        A comparison key, empty when the name carries no comparable characters.
    """
    return " ".join(_LEGAL_SUFFIX.sub(" ", fold_for_comparison(name)).split())


def company_keys(names: list[str]) -> dict[str, str]:
    """Map each name to the key it shares with other spellings of itself.

    Args:
        names: Every company name being compared, usually one anime's credits
            across all providers. Plurals are paired within this set only.

    Returns:
        Each name against its key. Two names that mean the same company share
        one.
    """
    bases = {name: company_base(name) for name in names}
    present = set(bases.values())
    keys: dict[str, str] = {}
    for name, base in bases.items():
        singular = base[:-1] if base.endswith("s") and not base.endswith("ss") else base
        keys[name] = singular if singular != base and singular in present else base
    return keys
