"""Find an entry's doubled providers: providers with two or more pages in the entry.

Seed cleaning lists the entries to check with ``find_doubled_provider_page_keys``,
and ``decide_doubled_providers`` decides exactly the providers it returns. The
rules:

| Rule | Reason for the rule |
| -- | -- |
| A provider is doubled when its pages carry more than one id. | Two pages of one provider with different ids are different pages. |
| Simkl and AnimeCountdown count as one provider, named ``simkl``. | They number a work the same, so a Simkl page and an AnimeCountdown page with the same number are one page. |
"""

from collections import defaultdict
from collections.abc import Iterable

from work_matching.provider_page_key import get_page_id, get_page_provider

# Simkl and AnimeCountdown number a work the same, so they are decided together,
# under Simkl's name: a Simkl and an AnimeCountdown page with different numbers
# are two works, just as two Simkl pages are.
_PROVIDER_DECIDED_UNDER = {"animecountdown": "simkl"}


def find_doubled_provider_page_keys(
    page_keys: Iterable[str],
) -> dict[str, tuple[str, ...]]:
    """Return the page keys of each doubled provider among these page keys.

    A provider is doubled when its pages carry more than one id: two pages of
    one provider with different ids are different pages. Pages of providers
    decided together count as one provider (see ``get_decided_provider``), so a
    Simkl page and an AnimeCountdown page with the same number are one page,
    not a doubled provider. Seed cleaning lists the entries to check with this rule,
    so it checks exactly the pages ``decide_doubled_providers`` decides.

    Args:
        page_keys: Page keys, such as the keys of an entry's pages.

    Returns:
        Each doubled provider, by the name its decision falls under, with its
        page keys in the order given.
    """
    page_keys_by_provider: defaultdict[str, list[str]] = defaultdict(list)
    for key in page_keys:
        if provider := get_decided_provider(key):
            page_keys_by_provider[provider].append(key)
    return {
        provider: tuple(page_keys)
        for provider, page_keys in page_keys_by_provider.items()
        if has_several_page_ids(page_keys)
    }


def get_decided_provider(key: str) -> str | None:
    """Return the provider whose decision a page key falls under, if any.

    AnimeCountdown numbers a work the same as Simkl: an entry listing one lists
    the other with the same number. So its pages fall under ``simkl``.

    Args:
        key: A page key, as ``get_page_key`` gives it.

    Returns:
        The provider name, or None when the key is not a known provider's page.
    """
    provider = get_page_provider(key)
    return _PROVIDER_DECIDED_UNDER.get(provider, provider) if provider else None


def has_several_page_ids(page_keys: Iterable[str]) -> bool:
    """Report whether a decision group's page keys carry more than one id.

    Pages of one provider with different ids are different pages, and so are
    pages of providers decided together, since they number a work the same.
    """
    return len(set(map(get_page_id, page_keys))) > 1
