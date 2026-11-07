"""Store enum members by name in the review file and other saved records.

Members such as review reasons have sentences as values, which the reviewer
reads. Each item reads ``<rule>: <reason>``.

- Saved records hold the member name: a reworded sentence would otherwise make
  every saved line unreadable, and the answers kept in those lines would be
  lost.
- Reading accepts a member name or a member: a name that is no member is
  refused with the names that are, so a renamed member is reported rather than
  read wrongly.
"""

from collections.abc import Callable
from enum import Enum


class UnknownMemberNameError(ValueError):
    """A saved value names no member of the enum it is read as."""

    def __init__(self, stored_value: object, enum_type: type[Enum]) -> None:
        """Name the saved value and the member names it should be one of.

        Args:
            stored_value: The value as saved.
            enum_type: The enum the value is read as.
        """
        member_names = ", ".join(enum_type.__members__)
        super().__init__(
            f"{stored_value!r} is not a {enum_type.__name__} name; "
            f"expected one of {member_names}"
        )


def read_member_by_name[EnumMember: Enum](
    enum_type: type[EnumMember],
) -> Callable[[object], EnumMember]:
    """Build a Pydantic before validator reading one enum's members by name.

    Args:
        enum_type: The enum whose members are stored by name.

    Returns:
        A function turning a stored name, or a member, into the member.
    """

    def read_member(stored_value: object) -> EnumMember:
        """Return the member a stored name names.

        Args:
            stored_value: A member name as saved, or a member.

        Returns:
            The member.

        Raises:
            UnknownMemberNameError: The stored value names no member.
        """
        if isinstance(stored_value, enum_type):
            return stored_value
        if isinstance(stored_value, str) and stored_value in enum_type.__members__:
            return enum_type[stored_value]
        raise UnknownMemberNameError(stored_value, enum_type)

    return read_member


def write_member_name(member: Enum) -> str:
    """Return the name a member is saved under.

    Args:
        member: The enum member to save.

    Returns:
        The member's name.
    """
    return member.name
