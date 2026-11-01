"""The lookup of known work entries backed by an anime-offline-database seed file."""

import json
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Self

from work_matching.comparable_title import fold_title_for_comparison
from work_matching.known_work_entries import KnownWorkEntries
from work_matching.work_entry import WorkEntry, WorkType


class KnownWorkEntriesFromSeed(KnownWorkEntries):
    """Hold every seed entry as a known work entry, indexed by page key and title.

    - Each page should belong to one entry. A page listed in more than one
      entry is kept by the first and listed in ``page_keys_in_several_entries``,
      so seed cleaning can report it.
    - The seed format allows an entry without pages. Such an entry cannot be
      found, so its title is listed in ``titles_of_entries_without_pages``.
    - Feed this the cleaned seed once there is one.
    """

    def __init__(self, seed_entries: Iterable[dict[str, Any]]) -> None:
        """Index every entry by its provider pages and by its title and type.

        Args:
            seed_entries: The ``data`` list of an anime-offline-database document.
        """
        titles_of_entries_without_pages: list[str] = []
        work_entries: list[WorkEntry] = []
        for seed_entry in seed_entries:
            if seed_entry["sources"]:
                work_entries.append(WorkEntry.from_seed_entry(seed_entry))
            else:
                titles_of_entries_without_pages.append(seed_entry["title"])
        self._index_work_entries(work_entries)
        self._titles_of_entries_without_pages = tuple(titles_of_entries_without_pages)

    def __len__(self) -> int:
        """Return how many work entries are known."""
        return len(self._entries)

    def __iter__(self) -> Iterator[WorkEntry]:
        """Yield every known work entry in seed file order, so the seed is read once."""
        return iter(self._entries)

    @property
    def page_keys_in_several_entries(self) -> frozenset[str]:
        """Return the keys of pages found in more than one entry."""
        return self._page_keys_in_several_entries

    @property
    def titles_of_entries_without_pages(self) -> tuple[str, ...]:
        """Titles of the seed entries left out because they have no pages."""
        return self._titles_of_entries_without_pages

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

    @classmethod
    def from_work_entries(cls, work_entries: Iterable[WorkEntry]) -> Self:
        """Index work entries already built, such as seed entries as corrections leave them.

        Args:
            work_entries: The work entries, in the order they are iterated.

        Returns:
            The lookup over exactly these entries; none is without pages, since a
            work entry always holds one.
        """
        known_work_entries = cls.__new__(cls)
        known_work_entries._index_work_entries(work_entries)
        known_work_entries._titles_of_entries_without_pages = ()
        return known_work_entries

    def find_by_page_key(self, key: str) -> WorkEntry | None:
        """Return the work entry holding the page with this key, or None.

        Args:
            key: A page key, as ``get_page_key`` gives it.
        """
        return self._entry_by_page_key.get(key)

    def find_by_title(
        self, title: str, work_type: WorkType, year: int | None
    ) -> list[WorkEntry]:
        """Return the work entries that look like this title, type and year.

        Args:
            title: A title, compared after case, punctuation, accents and
                season wording are folded.
            work_type: The work's type; only work entries of this type are returned.
            year: The work's year. A work entry matches when the years are
                equal or either side has no year.

        Returns:
            The matching work entries, in seed file order. Several can match,
            so a caller must not take the first as the answer.
        """
        return [
            work_entry
            for work_entry in self._entries_by_title_and_type.get(
                (fold_title_for_comparison(title), work_type), ()
            )
            if year is None or work_entry.year is None or work_entry.year == year
        ]

    def _index_work_entries(self, work_entries: Iterable[WorkEntry]) -> None:
        """Index work entries by page key and by folded title and type.

        A page listed in more than one entry is kept by the first and reported
        in ``page_keys_in_several_entries``.
        """
        self._entries: list[WorkEntry] = []
        self._entry_by_page_key: dict[str, WorkEntry] = {}
        self._entries_by_title_and_type: defaultdict[
            tuple[str, WorkType], list[WorkEntry]
        ] = defaultdict(list)
        page_keys_in_several_entries: set[str] = set()
        for work_entry in work_entries:
            self._entries.append(work_entry)
            for key in work_entry.addresses_by_page_key:
                if (
                    self._entry_by_page_key.setdefault(key, work_entry)
                    is not work_entry
                ):
                    page_keys_in_several_entries.add(key)
            self._entries_by_title_and_type[
                (fold_title_for_comparison(work_entry.title), work_entry.type)
            ].append(work_entry)
        self._page_keys_in_several_entries = frozenset(page_keys_in_several_entries)
