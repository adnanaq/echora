"""Compare a provider page's year and type with a work entry's.

Providers name types more finely than the seed, and fill years in at different
times, so a page is compared with an entry through the seed's coarser type, and
a missing year or unknown type agrees with any.
"""

from work_matching.provider_page import ProviderPage
from work_matching.work_entry import (
    WorkEntry,
    WorkType,
    is_type_matching,
    is_year_matching,
)

# The seed type each provider type falls under. A provider type missing here,
# such as UNKNOWN, agrees with every seed type.
_SEED_TYPE_BY_PROVIDER_TYPE: dict[str, WorkType] = {
    "TV": "TV",
    "TV SHORT": "TV",
    "MOVIE": "MOVIE",
    "OVA": "OVA",
    "ONA": "ONA",
    "SPECIAL": "SPECIAL",
    "TV SPECIAL": "SPECIAL",
    "MUSIC": "SPECIAL",
    "CM": "SPECIAL",
    "PV": "SPECIAL",
    "OTHER": "SPECIAL",
}


def get_seed_type(page: ProviderPage) -> WorkType | None:
    """Return the seed type a provider page's type falls under, or None when unknown.

    Providers name types more finely than the seed (``TV SHORT``, ``MUSIC``), so
    comparing a page with an entry, or building an entry from a page, needs the
    seed's coarser type.
    """
    return _SEED_TYPE_BY_PROVIDER_TYPE.get(page.type or "")


def is_page_details_matching(page: ProviderPage, entry: WorkEntry) -> bool:
    """Report whether a page's year and type agree with an entry's.

    A page joining an entry, by a move or as an added linked page, must agree.
    A disagreement shows the page's link may be a wrong mapping. The rules:

    - A missing year agrees with any year, because providers fill years in at
      different times.
    - An unknown type agrees with any type, for the same reason.
    - The page's type is compared as the seed's coarser type (``get_seed_type``).

    Args:
        page: The provider page's fetched details.
        entry: The work entry the page would join.

    Returns:
        True when both the year and the type agree.
    """
    return is_year_matching(page.year, entry.year) and is_type_matching(
        get_seed_type(page), entry.type
    )
