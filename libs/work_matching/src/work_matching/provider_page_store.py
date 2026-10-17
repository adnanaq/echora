"""Fetched provider pages kept in a JSON Lines file, so they outlive the HTTP cache.

The HTTP and crawler caches keep a page for about a week. Seed cleaning and
discovery decide from these details later and again on every seed release, so
the details are kept here, one page per line.
"""

import logging
from pathlib import Path

from work_matching.provider_page import ProviderPage

logger = logging.getLogger(__name__)


class ProviderPageStore:
    """Provider pages by key, read from and appended to one JSON Lines file.

    A page fetched again is appended rather than rewritten, and the last line
    for a page wins when the file is read. Every line ends with a line end, so
    a last line without one was cut off by an interrupted write: it is dropped
    from the file when the store opens.
    """

    def __init__(self, path: str | Path) -> None:
        """Read every page already in the file, if it exists.

        Args:
            path: The JSON Lines file holding the pages.
        """
        self._path = Path(path)
        self._pages: dict[str, ProviderPage] = {}
        if self._path.exists():
            self._read_pages()

    def __len__(self) -> int:
        """Return how many pages are kept."""
        return len(self._pages)

    def __contains__(self, key: object) -> bool:
        """Report whether a page key is kept."""
        return key in self._pages

    def get(self, key: str) -> ProviderPage | None:
        """Return the kept page with this key, or None."""
        return self._pages.get(key)

    def add(self, *pages: ProviderPage) -> None:
        """Keep pages, appending them to the file in one write.

        Args:
            *pages: The pages to keep; a page already kept is replaced.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as output:
            output.writelines(f"{page.model_dump_json()}\n" for page in pages)
        self._pages.update((page.key, page) for page in pages)

    def _read_pages(self) -> None:
        """Read every complete line, cutting an unfinished last line off the file."""
        complete_length = 0
        with self._path.open("rb+") as lines:
            for line in lines:
                if not line.endswith(b"\n"):
                    logger.warning(
                        f"Dropping an unfinished last line from {self._path}"
                    )
                    lines.truncate(complete_length)
                    break
                complete_length += len(line)
                if line.strip():
                    page = ProviderPage.model_validate_json(line)
                    self._pages[page.key] = page
