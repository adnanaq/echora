"""Drop the pages that providers say do not exist from the seed entries listing them.

The fetch stores a page as dead only when its provider says the page does not
exist, never on a failed request. A dead page has no details and no links, so no
rule can place it and no reviewer can decide anything about it. The rules:

- Every dead page a seed entry lists gets a correction dropping it
  (``is_dead_page``), and the run checks the entries without those pages.
- An entry whose every page is dead has nothing left, so the corrected seed
  removes it.
- A page the store holds as live is kept, even when its provider deleted it
  after the fetch. Only a new fetch of that page can tell.
"""

from collections.abc import Iterable

from seed_cleaning.seed_corrections import DEAD_PAGE_REASON, SeedCorrection
from work_matching import ProviderPageStore, WorkEntry


def list_dead_page_corrections(
    entries: Iterable[WorkEntry], store: ProviderPageStore
) -> list[SeedCorrection]:
    """List a correction dropping each dead page the entries list.

    Args:
        entries: The seed entries, such as every entry of the seed release.
        store: The fetched pages, which record whether a page is dead.

    Returns:
        One correction per dead page, in entry order and then page order.
    """
    return [
        SeedCorrection(
            entry_reference_page=entry.reference_page_key,
            page_key=page_key,
            address=address,
            placement_word=None,
            is_dead_page=True,
            reason=DEAD_PAGE_REASON,
        )
        for entry in entries
        for page_key, address in entry.addresses_by_page_key.items()
        if (page := store.get_page(page_key)) is not None and page.is_dead
    ]
