"""Define domain exceptions raised across work matching and review file operations."""

from collections.abc import Iterable
from enum import Enum
from pathlib import Path
from typing import ClassVar


class WorkMatchingError(Exception):
    """Report a domain error during work matching or review file operations."""


class ReviewCaseNamesError(WorkMatchingError):
    """Report an error affecting one or more named review cases."""

    MESSAGE_PREFIX: ClassVar[str] = "Review cases failed validation"

    def __init__(self, case_names: Iterable[str]) -> None:
        """Store the affected case names and format the error message.

        Args:
            case_names: The review case names affected by the error.
        """
        self.case_names = tuple(case_names)
        super().__init__(f"{self.MESSAGE_PREFIX}: {', '.join(self.case_names)}")


class DuplicateCaseError(ReviewCaseNamesError):
    """Report that two review cases written in one run share the same case name."""

    MESSAGE_PREFIX: ClassVar[str] = "Review cases given more than once"


class UnknownCaseError(ReviewCaseNamesError):
    """Report that answers were given for case names absent from the review file."""

    MESSAGE_PREFIX: ClassVar[str] = "Review cases not in the review file"


class SavedAnswerChangedError(ReviewCaseNamesError):
    """Report that cases hold a different saved answer than when editing began."""

    MESSAGE_PREFIX: ClassVar[str] = "Saved answers changed since their edits began"


class ReviewFileLockedError(WorkMatchingError):
    """Report that another process holds the review file lock."""

    def __init__(self, review_file_path: Path) -> None:
        """Name the locked review file in the error message.

        Args:
            review_file_path: The review file whose lock is held.
        """
        super().__init__(
            f"Another process is writing the review file: {review_file_path}"
        )


class NoLinkingEntryError(WorkMatchingError):
    """Report that a linked page was passed to a decision check without a linking entry.

    A linked page is discovered through an entry's links, so a caller passing no
    linking entry has lost track of where the linked page originated.
    """

    def __init__(self, page_key: str) -> None:
        """Name the linked page key missing a linking entry.

        Args:
            page_key: The key of the linked page.
        """
        super().__init__(f"No entry links the page to decide: {page_key}")


class UnknownMemberNameError(WorkMatchingError, ValueError):
    """Report that a stored value names no member of the target enum."""

    def __init__(self, stored_value: object, enum_type: type[Enum]) -> None:
        """Name the stored value and the valid enum member names.

        Args:
            stored_value: The stored value that failed lookup.
            enum_type: The enum type the value was read against.
        """
        member_names = ", ".join(enum_type.__members__)
        super().__init__(
            f"{stored_value!r} is not a {enum_type.__name__} name; "
            f"expected one of {member_names}"
        )
