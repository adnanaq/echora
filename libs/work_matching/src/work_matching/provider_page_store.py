"""Persist fetched provider page details in a JSON Lines file across cache expirations.

The HTTP and crawler caches drop a page after about a week. Seed cleaning and
discovery read these details again on every seed release, long after that, so
each fetched page's details are kept in a JSON Lines file, one page per line.
"""

import logging
from collections import defaultdict
from pathlib import Path

from pydantic import ValidationError
from work_matching.provider_page import ProviderPage
from work_matching.work_entry import WorkEntry

logger = logging.getLogger(__name__)


class ProviderPageStore:
    """The details of fetched provider pages, by page key, kept in one JSON Lines file.

    Each rule reads ``<rule>: <behavior>``:

    - Append-only updates: A page fetched again is added as a new line at the
      end of the file, and the old line stays. When the file is read, the last
      line for a page key is the one kept, so the newest details win.
    - Cumulative search provenance: A searched page added again, searched by
      another page, keeps every page that searched for the page, so the order
      of searches changes nothing.
    - Partial line recovery: Every complete line ends with a line break. A last
      line without one is a partial line, left by an interrupted write. The
      partial line is skipped when the file is read, and truncated from the file
      before the next pages are added.
    - Explicit writes: Opening the store only reads the file; only ``add_pages``
      changes the file.
    - Single-writer access: Only one process at a time may add pages, because
      adding does not lock the file.
    """

    def __init__(self, path: str | Path) -> None:
        """Open the store and read the pages already in the store file, if present.

        Args:
            path: The JSON Lines file that holds the pages.

        Raises:
            ValidationError: If a complete line does not hold a valid page. The
                error's note names the file and the line number.
        """
        self._path = Path(path)
        self._pages: dict[str, ProviderPage] = {}
        # The keys of the searched pages held for each page they were searched
        # by, so finding them takes one dictionary read, not a scan of every page.
        self._searched_page_keys_by_searched_by_page_key: defaultdict[str, set[str]] = (
            defaultdict(set)
        )
        # Where the partial last line starts in the file, in bytes, or None
        # when the file has none. add_pages cuts the file at this point first.
        self._partial_line_offset: int | None = None
        if self._path.exists():
            self._read_pages()

    def __len__(self) -> int:
        """Return how many pages the store holds.

        Returns:
            The count of stored provider pages.
        """
        return len(self._pages)

    def __contains__(self, page_key: object) -> bool:
        """Report whether the store holds a page with this page key.

        Args:
            page_key: The page key to check.

        Returns:
            True when ``page_key`` is stored.
        """
        return page_key in self._pages

    def get_page(self, page_key: str) -> ProviderPage | None:
        """Return the details of the page with this page key, or None if not held.

        Args:
            page_key: The page key to look up.

        Returns:
            The stored provider page, or None when not in the store.
        """
        return self._pages.get(page_key)

    def has_page_searched_by(self, searched_by_page_key: str) -> bool:
        """Report whether the store holds a page searched by this page.

        A searched page, such as AnimeSchedule's, is stored under its own key,
        so this method tells a fetcher whether the search by a page was done.

        Args:
            searched_by_page_key: The page key of the page the search used.

        Returns:
            True when at least one stored page was searched by ``searched_by_page_key``.
        """
        return searched_by_page_key in self._searched_page_keys_by_searched_by_page_key

    def get_pages_searched_by(
        self, searched_by_page_key: str
    ) -> tuple[ProviderPage, ...]:
        """Return the pages searched by this page, sorted by page key.

        Args:
            searched_by_page_key: The page key of the page the search used.

        Returns:
            The stored pages found by searching with ``searched_by_page_key``.
        """
        return tuple(
            self._pages[page_key]
            for page_key in sorted(
                self._searched_page_keys_by_searched_by_page_key.get(
                    searched_by_page_key, ()
                )
            )
        )

    def get_entry_pages(self, entry: WorkEntry) -> list[ProviderPage]:
        """Return an entry's stored pages, then the pages searched by them.

        The rules read an entry's links from both: a searched page, such as
        AnimeSchedule's searched by MAL ID, carries links that the page used for
        the search lacks.

        Args:
            entry: The work entry whose pages to return.

        Returns:
            The entry's stored pages in the entry's order, then the searched
            pages of each, sorted by page key. Pages not stored are left out.
        """
        return [
            *(
                page
                for page_key in entry.addresses_by_page_key
                if (page := self.get_page(page_key))
            ),
            *(
                searched_page
                for page_key in entry.addresses_by_page_key
                for searched_page in self.get_pages_searched_by(page_key)
            ),
        ]

    def add_pages(self, *pages: ProviderPage) -> None:
        """Add pages to the store and write them to the end of the file in one write.

        A partial last line left by an interrupted write is truncated first, so
        the new lines do not join onto the partial line.

        Args:
            *pages: The pages to add. A page whose key the store already holds
                replaces the details held for that key, keeping the pages that
                searched for the held page.
        """
        pages_by_page_key: dict[str, ProviderPage] = {}
        for page in pages:
            earlier_page = pages_by_page_key.get(page.key) or self._pages.get(page.key)
            pages_by_page_key[page.key] = _merge_searched_by_page_keys(
                earlier_page, page
            )
        pages = tuple(pages_by_page_key.values())
        if not pages:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._partial_line_offset is not None:
            with self._path.open("rb+") as page_file:
                page_file.truncate(self._partial_line_offset)
            self._partial_line_offset = None
        with self._path.open("a", encoding="utf-8") as page_file:
            page_file.writelines(f"{page.model_dump_json()}\n" for page in pages)
        self._pages.update((page.key, page) for page in pages)
        for page in pages:
            self._index_searched_pages(page)

    def _read_pages(self) -> None:
        """Read every complete line, and note where a partial last line starts.

        Blank lines are skipped. The file is read as bytes, so the position of
        a partial last line is known in bytes, which is what truncating the file
        needs.
        """
        complete_lines_size = 0
        with self._path.open("rb") as page_file:
            for line_number, line in enumerate(page_file, start=1):
                if not line.endswith(b"\n"):
                    logger.warning(
                        f"Skipping a partial last line in {self._path}, left by an "
                        "interrupted write; adding pages cuts it off"
                    )
                    self._partial_line_offset = complete_lines_size
                    return
                complete_lines_size += len(line)
                if line.strip():
                    self._load_page_from_line(line, line_number)

    def _load_page_from_line(self, line: bytes, line_number: int) -> None:
        """Turn one line of the file into a page and add the page to the store.

        Each line is one page's details saved as JSON. If an earlier line had
        the same page key, this page replaces the earlier page, so the newest
        details win.

        Args:
            line: Raw UTF-8 JSON line bytes from the store file.
            line_number: The 1-based line number in the store file.

        Raises:
            ValidationError: If the line is not a valid page. The error's note
                names the file and the line number, so the line can be found
                and fixed.
        """
        try:
            page = ProviderPage.model_validate_json(line)
        except ValidationError as error:
            error.add_note(f"{self._path}, line {line_number}")
            raise
        self._pages[page.key] = page
        self._index_searched_pages(page)

    def _index_searched_pages(self, page: ProviderPage) -> None:
        """Record a searched page under each page that searched for the page.

        Args:
            page: The provider page to index by ``searched_by`` keys.
        """
        for searched_by_page_key in page.searched_by:
            self._searched_page_keys_by_searched_by_page_key[searched_by_page_key].add(
                page.key
            )


def _merge_searched_by_page_keys(
    earlier_page: ProviderPage | None, page: ProviderPage
) -> ProviderPage:
    """Return the page with the earlier version's ``searched_by`` keys added.

    The newer page's details win, but a searched page keeps every page that
    searched for the page, whichever version recorded the search.

    Args:
        earlier_page: Previously stored or queued version of the page, if any.
        page: Newly added version of the page.

    Returns:
        The newly added page with the union of ``searched_by`` keys.
    """
    if earlier_page is None or set(earlier_page.searched_by) <= set(page.searched_by):
        return page
    searched_by_page_keys = tuple(
        sorted({*earlier_page.searched_by, *page.searched_by})
    )
    return page.model_copy(update={"searched_by": searched_by_page_keys})
