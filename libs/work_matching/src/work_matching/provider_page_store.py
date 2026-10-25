"""Keep fetched provider pages' details in a file, so they outlast the cache.

The HTTP and crawler caches drop a page after about a week. Seed cleaning and
discovery read these details again on every seed release, long after that, so
each fetched page's details are kept in a JSON Lines file, one page per line.

A real line of the file is shown in the README, under "A stored page".
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

    The rules for the file:

    - A page fetched again is added as a new line at the end of the file, and
      the old line stays. When the file is read, the last line for a page key
      is the one kept, so the newest details win.
    - A searched page added again, searched by another page, keeps every page
      it was searched by, so the order of searches changes nothing.
    - Every complete line ends with a line break. A last line without one is a
      partial line, left by a write that was interrupted. It is skipped when
      the file is read, and cut off the file before the next pages are added.
    - Opening the store only reads the file; only ``add_pages`` changes it.
    - Only one process at a time may add pages, because adding does not lock
      the file.
    """

    def __init__(self, path: str | Path) -> None:
        """Open the store and read the pages already in its file, if there is one.

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
        self._searched_page_keys_by_searched_by_key: defaultdict[str, set[str]] = (
            defaultdict(set)
        )
        # Where the partial last line starts in the file, in bytes, or None
        # when the file has none. add_pages cuts the file at this point first.
        self._partial_line_offset: int | None = None
        if self._path.exists():
            self._read_pages()

    def __len__(self) -> int:
        """Return how many pages the store holds."""
        return len(self._pages)

    def __contains__(self, key: object) -> bool:
        """Report whether the store holds a page with this page key."""
        return key in self._pages

    def get_page(self, key: str) -> ProviderPage | None:
        """Return the details of the page with this page key, or None if not held."""
        return self._pages.get(key)

    def has_page_searched_by(self, searched_by_key: str) -> bool:
        """Report whether the store holds a page searched by this page.

        A searched page, such as AnimeSchedule's, is stored under its own key,
        so this is how a fetcher tells that the search by a page was done.

        Args:
            searched_by_key: The page key of the page the search used.
        """
        return searched_by_key in self._searched_page_keys_by_searched_by_key

    def get_pages_searched_by(self, searched_by_key: str) -> tuple[ProviderPage, ...]:
        """Return the pages searched by this page, sorted by page key.

        Args:
            searched_by_key: The page key of the page the search used.
        """
        return tuple(
            self._pages[key]
            for key in sorted(
                self._searched_page_keys_by_searched_by_key.get(searched_by_key, ())
            )
        )

    def get_entry_pages(self, entry: WorkEntry) -> list[ProviderPage]:
        """Return an entry's stored pages, then the pages searched by them.

        The rules read an entry's links from both: a searched page, such as
        AnimeSchedule's searched by MAL id, carries links the page it was
        searched by lacks.

        Args:
            entry: The work entry whose pages to return.

        Returns:
            The entry's stored pages in the entry's order, then the searched
            pages of each, sorted by page key. Pages not stored are left out.
        """
        return [
            *(
                page
                for key in entry.addresses_by_page_key
                if (page := self.get_page(key))
            ),
            *(
                searched_page
                for key in entry.addresses_by_page_key
                for searched_page in self.get_pages_searched_by(key)
            ),
        ]

    def add_pages(self, *pages: ProviderPage) -> None:
        """Add pages to the store and write them to the end of the file in one write.

        A partial last line left by an interrupted write is cut off first, so
        the new lines do not join onto it.

        Args:
            *pages: The pages to add. A page whose key the store already holds
                replaces the details held for it, keeping the pages the held
                page was searched by.
        """
        pages_by_key: dict[str, ProviderPage] = {}
        for page in pages:
            earlier = pages_by_key.get(page.key) or self._pages.get(page.key)
            pages_by_key[page.key] = _merge_searched_by_keys(earlier, page)
        pages = tuple(pages_by_key.values())
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
        a partial last line is known in bytes, which is what cutting the file
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
        """Turn one line of the file into a page and add it to the store.

        Each line is one page's details saved as JSON. If an earlier line had
        the same page key, this page replaces it, so the newest details win.

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
        """Record a searched page under each page it was searched by."""
        for searched_by_key in page.searched_by:
            self._searched_page_keys_by_searched_by_key[searched_by_key].add(page.key)


def _merge_searched_by_keys(
    earlier: ProviderPage | None, page: ProviderPage
) -> ProviderPage:
    """Return the page with the earlier version's ``searched_by`` keys added to its own.

    The newer page's details win, but a searched page keeps every page it was
    searched by, whichever version recorded it.
    """
    if earlier is None or set(earlier.searched_by) <= set(page.searched_by):
        return page
    searched_by_keys = tuple(sorted({*earlier.searched_by, *page.searched_by}))
    return page.model_copy(update={"searched_by": searched_by_keys})
