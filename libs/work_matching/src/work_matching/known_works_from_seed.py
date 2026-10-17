"""The known-works lookup backed by an anime-offline-database seed file."""

import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Self

from work_matching.comparable_title import comparable_title
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
        self._by_page: dict[str, WorkEntry] = {}
        self._by_title: defaultdict[tuple[str, WorkType], list[WorkEntry]] = (
            defaultdict(list)
        )
        in_several_entries: set[str] = set()
        without_pages: list[str] = []
        count = 0
        for entry in entries:
            if not entry["sources"]:
                without_pages.append(entry["title"])
                continue
            work = WorkEntry.from_seed_entry(entry)
            count += 1
            for key in work.pages:
                if self._by_page.setdefault(key, work) is not work:
                    in_several_entries.add(key)
            self._by_title[(comparable_title(work.title), work.type)].append(work)
        self._count = count
        self._pages_in_several_entries = frozenset(in_several_entries)
        self._entries_without_pages = tuple(without_pages)

    def __len__(self) -> int:
        """Return how many works are known."""
        return self._count

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
        with Path(path).open(encoding="utf-8") as handle:
            return cls(json.load(handle)["data"])

    def find_by_page_key(self, key: str) -> WorkEntry | None:
        """Return the work that has the page with this key, or None.

        Args:
            key: A page key, as ``provider_page_key`` gives it.
        """
        return self._by_page.get(key)

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
            for work in self._by_title.get((comparable_title(title), work_type), ())
            if year is None or work.year is None or work.year == year
        ]
