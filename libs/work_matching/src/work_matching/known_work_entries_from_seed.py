"""Index known work entries backed by an anime-offline-database seed file."""

import json
from bisect import bisect_left
from collections import defaultdict
from collections.abc import Iterable, Iterator
from itertools import islice, takewhile
from pathlib import Path
from typing import Any, Final, Self

from work_matching.comparable_title import fold_title_for_comparison
from work_matching.known_work_entries import KnownWorkEntries
from work_matching.work_entry import WorkEntry, WorkType

# The fewest words a title prefix shared by sibling titles of a series holds.
# Measured on the review cases of a release: one or two words, such as "Ano"
# or "Bucket no", also start unrelated titles; three words kept series.
MIN_SERIES_ROOT_WORDS: Final = 3


class KnownWorkEntriesFromSeed(KnownWorkEntries):
    """Index every seed entry as a known work entry by page key and title.

    - Each page should belong to one entry. A page listed in more than one
      entry is kept by the first and listed in ``multi_entry_page_keys``,
      so seed cleaning can report the conflict.
    - The seed format allows an entry without pages. Such an entry cannot be
      found, so the entry title is listed in ``pageless_entry_titles``.
    """

    def __init__(self, seed_entries: Iterable[dict[str, Any]]) -> None:
        """Index every entry by provider pages and by title and type.

        Args:
            seed_entries: The ``data`` list of an anime-offline-database document.
        """
        pageless_entry_titles: list[str] = []
        work_entries: list[WorkEntry] = []
        for seed_entry in seed_entries:
            if seed_entry["sources"]:
                work_entries.append(WorkEntry.read_seed_entry(seed_entry))
            else:
                pageless_entry_titles.append(seed_entry["title"])
        self._index_work_entries(work_entries)
        self._pageless_entry_titles = tuple(pageless_entry_titles)

    @classmethod
    def load_seed_file(cls, path: str | Path) -> Self:
        """Load and index a seed file.

        Args:
            path: Path to an anime-offline-database JSON document.

        Returns:
            The lookup over every entry in the file.
        """
        with Path(path).open(encoding="utf-8") as seed_file:
            return cls(json.load(seed_file)["data"])

    @classmethod
    def index_work_entries(cls, work_entries: Iterable[WorkEntry]) -> Self:
        """Index work entries already built, such as seed entries after corrections.

        Args:
            work_entries: The work entries, in the order they are iterated.

        Returns:
            The lookup over these entries; none is without pages, since a
            work entry always holds at least one page.
        """
        known_work_entries = cls.__new__(cls)
        known_work_entries._index_work_entries(work_entries)
        known_work_entries._pageless_entry_titles = ()
        return known_work_entries

    @property
    def multi_entry_page_keys(self) -> frozenset[str]:
        """Return the keys of pages found in more than one entry."""
        return self._multi_entry_page_keys

    @property
    def pageless_entry_titles(self) -> tuple[str, ...]:
        """Return the titles of the seed entries left out because they have no pages."""
        return self._pageless_entry_titles

    def __len__(self) -> int:
        """Return how many work entries are known.

        Returns:
            The count of indexed work entries.
        """
        return len(self._entries)

    def __iter__(self) -> Iterator[WorkEntry]:
        """Yield every known work entry in seed file order, so the seed is read once.

        Yields:
            Each known ``WorkEntry`` in seed file order.
        """
        return iter(self._entries)

    def find_by_page_key(self, page_key: str) -> WorkEntry | None:
        """Return the work entry holding the page with this key, or None.

        Args:
            page_key: A page key, as ``resolve_page_key`` gives it.

        Returns:
            The matching work entry, or None when no known entry holds the page.
        """
        return self._entry_by_page_key.get(page_key)

    def find_by_title(
        self, title: str, work_type: WorkType, year: int | None
    ) -> list[WorkEntry]:
        """Return the work entries that look like this title, type, and year.

        Args:
            title: A title, compared after case, punctuation, accents, and
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

    def find_by_series(self, title: str) -> list[WorkEntry]:
        """Return the work entries whose titles belong to the same series as this one.

        Each item gives the kind of title found, then how it is found.

        - Shorter titles: the title's word prefixes, looked up directly.
        - Longer titles: titles starting with the title and a space, which sit
          together in the sorted titles and are found with a binary search.
        - Sibling titles: titles starting with the title's longest word prefix
          of at least ``MIN_SERIES_ROOT_WORDS`` words that another title starts
          with, such as two movies of one series. A shorter prefix, such as
          ``Ano``, also starts unrelated titles, so that prefix is not used.

        Args:
            title: A title, folded like ``find_by_title``'s.

        Returns:
            The entries with the same title, then those with shorter titles
            of the series, longest first, then those with longer titles, then
            the siblings, both in title order.
        """
        folded_title = fold_title_for_comparison(title)
        if not folded_title:
            return []
        words = folded_title.split()
        shorter_titles = [
            " ".join(words[:count]) for count in range(len(words) - 1, 0, -1)
        ]
        longer_titles = self._list_titles_starting_with(folded_title + " ")
        sibling_titles = next(
            (
                found_titles
                for count in range(len(words) - 1, MIN_SERIES_ROOT_WORDS - 1, -1)
                if (
                    found_titles := [
                        sorted_title
                        for sorted_title in self._list_titles_starting_with(
                            " ".join(words[:count]) + " "
                        )
                        if sorted_title != folded_title
                        and not sorted_title.startswith(folded_title + " ")
                    ]
                )
            ),
            [],
        )
        return [
            work_entry
            for series_title in (
                folded_title,
                *shorter_titles,
                *longer_titles,
                *sibling_titles,
            )
            for work_entry in self._entries_by_folded_title.get(series_title, ())
        ]

    def _list_titles_starting_with(self, title_start: str) -> list[str]:
        """List the folded titles starting with some text, with a binary search.

        Args:
            title_start: The text the titles start with.

        Returns:
            The matching folded titles, in title order.
        """
        return list(
            takewhile(
                lambda sorted_title: sorted_title.startswith(title_start),
                islice(
                    self._sorted_folded_titles,
                    bisect_left(self._sorted_folded_titles, title_start),
                    None,
                ),
            )
        )

    def _index_work_entries(self, work_entries: Iterable[WorkEntry]) -> None:
        """Index work entries by page key, by folded title and type, and by folded title.

        A page listed in more than one entry is kept by the first and reported
        in ``multi_entry_page_keys``.

        Args:
            work_entries: The work entries to index.
        """
        self._entries: list[WorkEntry] = []
        self._entry_by_page_key: dict[str, WorkEntry] = {}
        self._entries_by_title_and_type: defaultdict[
            tuple[str, WorkType], list[WorkEntry]
        ] = defaultdict(list)
        self._entries_by_folded_title: defaultdict[str, list[WorkEntry]] = defaultdict(
            list
        )
        multi_entry_page_keys: set[str] = set()
        for work_entry in work_entries:
            self._entries.append(work_entry)
            for page_key in work_entry.addresses_by_page_key:
                if (
                    self._entry_by_page_key.setdefault(page_key, work_entry)
                    is not work_entry
                ):
                    multi_entry_page_keys.add(page_key)
            folded_title = fold_title_for_comparison(work_entry.title)
            self._entries_by_title_and_type[folded_title, work_entry.type].append(
                work_entry
            )
            self._entries_by_folded_title[folded_title].append(work_entry)
        self._multi_entry_page_keys = frozenset(multi_entry_page_keys)
        # Sorted once, so ``find_by_series`` finds longer titles by binary search.
        self._sorted_folded_titles = sorted(self._entries_by_folded_title)
