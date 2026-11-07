"""Find an entry's doubled providers holding two or more distinct pages.

Seed cleaning lists entries to check with ``find_doubled_provider_page_keys``,
and ``decide_doubled_providers`` evaluates the providers returned by that lookup.

Each item gives the rule, then the reason for it.

- A provider is doubled when its pages carry more than one identifier. Two pages
  of one provider with different identifiers represent distinct provider pages.
- Simkl and AnimeCountdown count as one provider named ``simkl``. Both platforms
  number a work identically, so a Simkl page and an AnimeCountdown page with the
  same number refer to the same work.
"""

from collections import defaultdict
from collections.abc import Iterable

from work_matching.provider_page_key import read_page_id, read_page_provider

# Simkl and AnimeCountdown number a work the same, so they are decided together,
# under Simkl's name: a Simkl and an AnimeCountdown page with different numbers
# are two works, just as two Simkl pages are.
_PROVIDER_DECIDED_UNDER = {"animecountdown": "simkl"}


def find_doubled_provider_page_keys(
    page_keys: Iterable[str],
) -> dict[str, tuple[str, ...]]:
    """Return the page keys of each doubled provider among these page keys.

    A provider is doubled when its pages carry more than one identifier. Pages
    of providers decided together count as one provider under
    ``resolve_decided_provider``. A Simkl page and an AnimeCountdown page with
    the same number therefore count as a single page rather than a doubled
    provider. Seed cleaning uses this function so it checks the exact pages
    that ``decide_doubled_providers`` evaluates.

    Args:
        page_keys: Page keys, such as the keys of an entry's pages.

    Returns:
        Each doubled provider, by the name its decision falls under, with its
        page keys in the order given.
    """
    page_keys_by_provider: defaultdict[str, list[str]] = defaultdict(list)
    for page_key in page_keys:
        if provider := resolve_decided_provider(page_key):
            page_keys_by_provider[provider].append(page_key)
    return {
        provider: tuple(provider_page_keys)
        for provider, provider_page_keys in page_keys_by_provider.items()
        if has_several_page_ids(provider_page_keys)
    }


def resolve_decided_provider(page_key: str) -> str | None:
    """Resolve the provider name under which a page key is decided.

    AnimeCountdown numbers works identically to Simkl, so AnimeCountdown pages
    are grouped under ``simkl``.

    Args:
        page_key: A page key, as ``resolve_page_key`` gives it.

    Returns:
        The provider name, or None when the key is not a known provider's page.
    """
    provider = read_page_provider(page_key)
    return _PROVIDER_DECIDED_UNDER.get(provider, provider) if provider else None


def has_several_page_ids(page_keys: Iterable[str]) -> bool:
    """Report whether a decision group's page keys carry more than one identifier.

    Pages of one provider with different identifiers are distinct pages. Pages
    of providers decided together follow the same rule because they share a
    numbering scheme.

    Args:
        page_keys: Page keys belonging to one decided provider group.

    Returns:
        True when the page keys contain at least two distinct identifiers.
    """
    return len(set(map(read_page_id, page_keys))) > 1
