"""Fetched provider pages kept in a JSON Lines file, so they outlive the HTTP cache.

The HTTP and crawler caches keep a page for about a week. Seed cleaning and
discovery decide from these details later and again on every seed release, so
the details are kept here, one page per line.
"""

import logging
from pathlib import Path

from pydantic import ValidationError

from work_matching.provider_page import ProviderPage

logger = logging.getLogger(__name__)


class ProviderPageStore:
    """Provider pages by key, read from and appended to one JSON Lines file.

    A page fetched again is appended rather than rewritten, and the last line
    for a page wins when the file is read. Every line ends with a line end, so
    a last line without one was cut off by an interrupted write: it is ignored
    when the store opens and removed from the file by the next add. Opening a
    store never changes the file.

    One process at a time writes the file: appends are not locked.
    """

    def __init__(self, path: str | Path) -> None:
        """Read every page already in the file, if it exists.

        Args:
            path: The JSON Lines file holding the pages.

        Raises:
            ValidationError: A complete line is not a page. Its note names the
                file and line.
        """
        self._path = Path(path)
        self._pages: dict[str, ProviderPage] = {}
        # Where an unfinished last line starts, until the next add removes it.
        self._unfinished_line_start: int | None = None
        if self._path.exists():
            self._read_pages()

    def __len__(self) -> int:
        """Return how many pages are kept."""
        return len(self._pages)

    def __contains__(self, key: object) -> bool:
        """Report whether a page key is kept."""
        return key in self._pages

    def get_page(self, key: str) -> ProviderPage | None:
        """Return the kept page with this key, or None."""
        return self._pages.get(key)

    def add_pages(self, *pages: ProviderPage) -> None:
        """Keep pages, appending them to the file in one write.

        Args:
            *pages: The pages to keep; a page already kept is replaced.
        """
        if not pages:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._unfinished_line_start is not None:
            with self._path.open("rb+") as output:
                output.truncate(self._unfinished_line_start)
            self._unfinished_line_start = None
        with self._path.open("a", encoding="utf-8") as output:
            output.writelines(f"{page.model_dump_json()}\n" for page in pages)
        self._pages.update((page.key, page) for page in pages)

    def _read_pages(self) -> None:
        """Read every complete line, noting where an unfinished last line starts."""
        complete_length = 0
        with self._path.open("rb") as lines:
            for number, line in enumerate(lines, start=1):
                if not line.endswith(b"\n"):
                    logger.warning(
                        f"Ignoring an unfinished last line in {self._path}; "
                        "the next add removes it"
                    )
                    self._unfinished_line_start = complete_length
                    return
                complete_length += len(line)
                if line.strip():
                    self._keep_line(line, number)

    def _keep_line(self, line: bytes, number: int) -> None:
        """Keep the page one line holds, naming the file and line if it is broken."""
        try:
            page = ProviderPage.model_validate_json(line)
        except ValidationError as error:
            error.add_note(f"{self._path}, line {number}")
            raise
        self._pages[page.key] = page
