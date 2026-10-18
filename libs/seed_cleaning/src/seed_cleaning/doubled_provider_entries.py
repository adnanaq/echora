"""Find the seed entries that hold a doubled provider.

A doubled provider is an entry with two or more pages from one provider. The
pages may be one page under two addresses, or pages of different works that
the seed joined by mistake; the doubled-provider rules decide which.
"""

from collections.abc import Iterable

from work_matching import WorkEntry, doubled_provider_pages


def doubled_provider_entries(works: Iterable[WorkEntry]) -> list[WorkEntry]:
    """Return the entries holding a doubled provider, in the order given.

    The rule is the doubled-provider rules' own (``doubled_provider_pages``), so
    seed cleaning checks exactly the pages ``decide_doubled_providers`` decides:

    - two pages of one provider count only when their ids differ;
    - Simkl and AnimeCountdown count as one provider, because they number a work
      the same, so a Simkl page and an AnimeCountdown page with the same number
      are one page.

    Args:
        works: The works to check, such as every entry of a seed file.
    """
    return [work for work in works if doubled_provider_pages(work.pages)]
