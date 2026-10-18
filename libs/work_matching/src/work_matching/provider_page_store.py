"""Keep fetched provider pages' details in a file, so they outlast the cache.

The HTTP and crawler caches drop a page after about a week. Seed cleaning and
discovery read these details again on every seed release, long after that, so
each fetched page's details are kept in a JSON Lines file, one page per line.
"""

import logging
from collections import defaultdict
from pathlib import Path

from pydantic import ValidationError
from work_matching.provider_page import ProviderPage

logger = logging.getLogger(__name__)


class ProviderPageStore:
    """The details of fetched provider pages, by page key, kept in one JSON Lines file.

    The rules for the file:

    - A page fetched again is added as a new line at the end of the file, and
      the old line stays. When the file is read, the last line for a page key
      is the one kept, so the newest details win.
    - A found page added again through another lookup page keeps every lookup
      page it was found through, so the order of lookups changes nothing.
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
        # The keys of the found pages held for each lookup page, so finding them
        # takes one look-up rather than a scan of every page.
        self._found_page_keys_by_lookup_page: defaultdict[str, set[str]] = defaultdict(
            set
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

    def has_page_found_by(self, lookup_key: str) -> bool:
        """Report whether the store holds a page found through this lookup page.

        A found page, such as AnimeSchedule's, is stored under its own key, so
        this is how a fetcher tells that the lookup through a page was done.

        Args:
            lookup_key: The page key of the page the lookup went through.
        """
        return lookup_key in self._found_page_keys_by_lookup_page

    def get_pages_found_by(self, lookup_key: str) -> tuple[ProviderPage, ...]:
        """Return the pages found through this lookup page, sorted by page key.

        Args:
            lookup_key: The page key of the page the lookup went through.
        """
        return tuple(
            self._pages[key]
            for key in sorted(self._found_page_keys_by_lookup_page.get(lookup_key, ()))
        )

    def add_pages(self, *pages: ProviderPage) -> None:
        """Add pages to the store and write them to the end of the file in one write.

        A partial last line left by an interrupted write is cut off first, so
        the new lines do not join onto it.

        Args:
            *pages: The pages to add. A page whose key the store already holds
                replaces the details held for it, keeping the lookup pages the
                held page was found through.
        """
        pages_by_key: dict[str, ProviderPage] = {}
        for page in pages:
            earlier = pages_by_key.get(page.key) or self._pages.get(page.key)
            pages_by_key[page.key] = _merge_lookup_pages(earlier, page)
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
            self._index_lookup_pages(page)

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
        self._index_lookup_pages(page)

    def _index_lookup_pages(self, page: ProviderPage) -> None:
        """Record a found page under each lookup page it was found through."""
        for lookup_key in page.found_by:
            self._found_page_keys_by_lookup_page[lookup_key].add(page.key)


def _merge_lookup_pages(
    earlier: ProviderPage | None, page: ProviderPage
) -> ProviderPage:
    """Return the page with the lookup pages of the earlier version added to its own.

    The newer page's details win, but a found page keeps every lookup page it
    was found through, whichever version recorded it.
    """
    if earlier is None or set(earlier.found_by) <= set(page.found_by):
        return page
    lookup_keys = tuple(sorted({*earlier.found_by, *page.found_by}))
    return page.model_copy(update={"found_by": lookup_keys})
