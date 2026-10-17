"""The known-works lookup that matching compares provider pages against.

Seed cleaning, seed link filling and discovery all ask the same questions
of the works already known. They ask them through this interface, so the works
can come from the seed now and from enriched works later without changing them.
"""

from typing import Protocol, runtime_checkable

from work_matching.provider_page_key import provider_page_key
from work_matching.work_entry import WorkEntry, WorkType


@runtime_checkable
class KnownWorks(Protocol):
    """Works already known, found by provider page or by title, type and year.

    A class subclassing this one inherits ``find_by_page``, built on its
    ``find_by_page_key``.
    """

    def find_by_page(self, address: str) -> WorkEntry | None:
        """Return the work that has this provider page, or None.

        Args:
            address: A provider page address, in any of its spellings.
        """
        return self.find_by_page_key(provider_page_key(address))

    def find_by_page_key(self, key: str) -> WorkEntry | None:
        """Return the work that has the page with this key, or None.

        Args:
            key: A page key, as ``provider_page_key`` gives it.
        """

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
            The matching works, in the order the works became known. Several
            can match, so a caller must not take the first as the answer.
        """
