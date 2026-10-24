"""Find the seed entries that hold a doubled provider.

A doubled provider is an entry with two or more pages from one provider. The
pages may be one page under two addresses, or pages of different works that
the seed joined by mistake; the doubled-provider rules decide which.
"""

from collections.abc import Iterable

from work_matching import WorkEntry, find_doubled_provider_page_keys


def find_doubled_provider_entries(entries: Iterable[WorkEntry]) -> list[WorkEntry]:
    """Return the entries holding a doubled provider, in the order given.

    The rule is the doubled-provider rules' own
    (``find_doubled_provider_page_keys``), so seed cleaning checks exactly the
    pages ``decide_doubled_providers`` decides:

    - two pages of one provider count only when their ids differ;
    - Simkl and AnimeCountdown count as one provider, because they number a work
      the same, so a Simkl page and an AnimeCountdown page with the same number
      are one page.

    Args:
        entries: The entries to check, such as every entry of a seed file.
    """
    return [
        entry
        for entry in entries
        if find_doubled_provider_page_keys(entry.addresses_by_page_key)
    ]
