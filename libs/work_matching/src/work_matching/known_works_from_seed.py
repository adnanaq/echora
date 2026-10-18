"""The known-works lookup backed by an anime-offline-database seed file."""

import json
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Self

from work_matching.comparable_title import fold_title_for_comparison
from work_matching.known_works import KnownWorks
from work_matching.work_entry import WorkEntry, WorkType


class KnownWorksFromSeed(KnownWorks):
    """Every entry of a seed file as a known work, indexed by page and by title.

    Each page should belong to one entry. A page listed in more than one entry
    is kept by the first and listed in ``pages_in_several_entries``, so seed
    cleaning can report it. The seed format allows
    an entry without pages; such an entry cannot be found and is listed in
    ``entries_without_pages`` instead. Feed this the cleaned seed once there is
    one.
    """

    def __init__(self, entries: Iterable[dict[str, Any]]) -> None:
        """Index every entry by its provider pages and by its title and type.

        Args:
            entries: The ``data`` list of an anime-offline-database document.
        """
        self._entries: list[WorkEntry] = []
        self._entry_by_page_key: dict[str, WorkEntry] = {}
        self._entries_by_title_and_type: defaultdict[
            tuple[str, WorkType], list[WorkEntry]
        ] = defaultdict(list)
        pages_in_several_entries: set[str] = set()
        entries_without_pages: list[str] = []
        for entry in entries:
            if not entry["sources"]:
                entries_without_pages.append(entry["title"])
                continue
            work = WorkEntry.from_seed_entry(entry)
            self._entries.append(work)
            for key in work.pages:
                if self._entry_by_page_key.setdefault(key, work) is not work:
                    pages_in_several_entries.add(key)
            self._entries_by_title_and_type[
                (fold_title_for_comparison(work.title), work.type)
            ].append(work)
        self._pages_in_several_entries = frozenset(pages_in_several_entries)
        self._entries_without_pages = tuple(entries_without_pages)

    def __len__(self) -> int:
        """Return how many works are known."""
        return len(self._entries)

    def __iter__(self) -> Iterator[WorkEntry]:
        """Yield every known work in seed file order, so the seed is read once."""
        return iter(self._entries)

    @property
    def pages_in_several_entries(self) -> frozenset[str]:
        """Keys of pages found in more than one entry."""
        return self._pages_in_several_entries

    @property
    def entries_without_pages(self) -> tuple[str, ...]:
        """Titles of the seed entries left out because they have no pages."""
        return self._entries_without_pages

    @classmethod
    def from_file(cls, path: str | Path) -> Self:
        """Load and index a seed file.

        Args:
            path: Path to an anime-offline-database JSON document.

        Returns:
            The lookup over every entry in the file.
        """
        with Path(path).open(encoding="utf-8") as seed_file:
            return cls(json.load(seed_file)["data"])

    def find_by_page_key(self, key: str) -> WorkEntry | None:
        """Return the work that has the page with this key, or None.

        Args:
            key: A page key, as ``get_page_key`` gives it.
        """
        return self._entry_by_page_key.get(key)

    def find_by_title(
        self, title: str, work_type: WorkType, year: int | None
    ) -> list[WorkEntry]:
        """Return the works that look like this title, type and year.

        Args:
            title: A title, compared after case, punctuation, accents and
                season wording are folded.
            work_type: The work's type; only works of this type are returned.
            year: The work's year. A work matches when the years are equal or
                either side has no year.

        Returns:
            The matching works, in seed file order. Several can match, so a caller
            must not take the first as the answer.
        """
        return [
            work
            for work in self._entries_by_title_and_type.get(
                (fold_title_for_comparison(title), work_type), ()
            )
            if year is None or work.year is None or work.year == year
        ]
