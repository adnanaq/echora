"""Define the lookup protocol for known work entries matched against provider pages.

Seed cleaning, seed link filling, and discovery all query the work entries
already known through this protocol, allowing entries to come from the seed
file or from enriched works without changing the callers.
"""

from typing import Protocol, runtime_checkable

from work_matching.provider_page_key import resolve_page_key
from work_matching.work_entry import WorkEntry, WorkType


@runtime_checkable
class KnownWorkEntries(Protocol):
    """Provide work entries already known, found by provider page or by title, type, and year.

    An implementation provides ``find_by_page_key``, ``find_by_title``, and
    ``find_by_series``. A class subclassing this protocol inherits
    ``find_by_page_address``, built on ``find_by_page_key``.
    """

    def find_by_page_address(self, address: str) -> WorkEntry | None:
        """Return the work entry holding the page at this address, or None.

        Args:
            address: A provider page address, in any of its spellings.

        Returns:
            The matching work entry, or None when no known entry holds the page.
        """
        return self.find_by_page_key(resolve_page_key(address))

    def find_by_page_key(self, page_key: str) -> WorkEntry | None:
        """Return the work entry holding the page with this key, or None.

        Args:
            page_key: A page key, as ``resolve_page_key`` gives it.

        Returns:
            The matching work entry, or None when no known entry holds the page.
        """

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
            The matching work entries, in the order they became known.
            Several can match, so a caller must not take the first as the
            answer.
        """

    def find_by_series(self, title: str) -> list[WorkEntry]:
        """Return the work entries whose titles belong to the same series as this one.

        Two titles belong to one series when, after folding, one of these holds:

        - One is the other or starts with the other followed by more words,
          as "Kusuriya no Hitorigoto" and "Kusuriya no Hitorigoto Season 2".
        - Both start with the title's longest word prefix of a few words that
          another title also starts with, as "Kaguya-sama wa Kokurasetai
          Movie" and "Kaguya-sama wa Kokurasetai: Ultra Romantic".

        Args:
            title: A title, folded like ``find_by_title``'s.

        Returns:
            The entries with the same title, then those with shorter titles
            of the series, longest first, then those with longer titles, then
            those sharing the prefix, in title order. Titles can share words by
            chance, so these are suggestions for a person, never sure matches.
        """
