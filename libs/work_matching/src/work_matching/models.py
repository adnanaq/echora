"""Define data models and validation rule enums for work matching and review cases.

Each model group reads ``<group>: <purpose>``:

- ``PlacementRule`` and ``LinkedPageAnswerRule``: Validation rules that review
  answers must satisfy, with user-facing explanations as enum values.
- ``DoubledProviderOutcome``, ``DoubledProviderReviewReason``, ``UnrelatedPage``,
  and ``DoubledProviderDecision``: Classification of a doubled provider's pages
  in an entry and the reasons requiring human review.
- ``LinkedPageOutcome``, ``LinkedPageReason``, ``LinkingEntryDetails``,
  ``LinkedPageDecision``, ``LinkedPageAddition``, and ``LinkedPageAnswerCheck``:
  Decisions and validated answers for linked pages not held by any known work.
- ``PreviousAnswer``: An answered case's answer carried over to the replacement
  case after a seed release adds pages.
- ``ChoiceOption``, ``GroupPageChoices``, ``WorkPartialSources``, and
  ``LinkedPageConflict``: Interactive choice options and rule disabling states
  offered by the review page.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from typing import Annotated

from pydantic import BaseModel, BeforeValidator, ConfigDict, JsonValue, PlainSerializer
from work_matching.answer_words import describe_answer_word
from work_matching.errors import UnknownMemberNameError
from work_matching.work_entry import WorkEntry


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


class PlacementRule(StrEnum):
    """Define a validation rule that an entry group's placement answer must satisfy."""

    INVALID_PLACEMENT_MAP = "the answer must map each page key to a placement"
    INVALID_PAGE_PLACEMENT = (
        "a placement must be 'work N', 'work N, partial in <works>', "
        "'other work', 'other work <page_key>', or 'dropped'"
    )
    PAGE_NOT_IN_CASE = "the answer includes a page that does not belong to this case"
    MISSING_REQUIRED_PLACEMENT = "every required page must be assigned to a work"
    WORK_WITH_TWO_PAGES_OF_ONE_PROVIDER = (
        "each work can include at most one page of each provider"
    )
    DROPPED_PAGE_WITHOUT_KEPT_PAGE = (
        "a page can be dropped only while another page of its provider has a "
        "work in this case"
    )
    PARTIAL_SOURCE_OF_OWN_WORK = (
        "a page cannot be a partial source of the work it is a source of"
    )
    PARTIAL_SOURCE_OF_EMPTY_WORK = (
        "a page can be a partial source only of a work that holds pages"
    )
    NAMED_ENTRY_IN_CASE = (
        "the target entry specified after 'other work' is already part of "
        "this case; assign the page to 'work N' instead"
    )
    UNKNOWN_NAMED_ENTRY = (
        "no seed entry holds the page key specified after 'other work'"
    )
    WORK_WITHOUT_REFERENCE_PAGE = (
        "a work combining pages from multiple entries must include at least "
        "one entry's reference page"
    )


class LinkedPageAnswerRule(StrEnum):
    """Define a validation rule that a linked-page review answer must satisfy."""

    WRONG_LINKED_PAGE_KEY = (
        "the answer must map only the linked page's key to a decision"
    )
    INVALID_LINKED_PAGE_ANSWER = (
        "the decision must be 'add', 'add as partial', 'add to <entry>', "
        "'add as partial to <entry>', or 'skip'"
    )
    MISSING_TARGET_ENTRY = (
        "multiple entries link to this page; specify the target entry with 'to <entry>'"
    )
    TARGET_ENTRY_NOT_LINKING = "the specified target entry does not link to this page"
    PROVIDER_ALREADY_IN_ENTRY = (
        "the target entry already has a page from this provider, so this page "
        "can only be added with 'add as partial'"
    )


class DoubledProviderOutcome(StrEnum):
    """Name what a doubled provider's pages are, relative to the entry holding them."""

    SAME_PAGE = (
        "the addresses are one page: an old address leads to the page's current address"
    )
    DUPLICATE_PAGES = (
        "every page belongs to the entry's work and covers all of it: the provider "
        "lists the work more than once"
    )
    PART_PAGES = (
        "every page belongs to the entry's work, but their air dates or episode "
        "counts differ, so each page covers part of the work"
    )
    PAGES_OF_OTHER_WORKS = (
        "at most one page belongs to the entry's work; every other page belongs to "
        "a work outside the entry"
    )
    SEPARATE_WORKS = (
        "each page is linked with a separate set of the entry's pages, so the entry "
        "holds several works"
    )
    UNDECIDED = "the links do not tell which work every page belongs to"


class DoubledProviderReviewReason(StrEnum):
    """Name why a person must check a doubled-provider decision."""

    CURRENT_ADDRESS_MISSING = (
        "the entry lists only old addresses of the page, so dropping them would "
        "lose the page"
    )
    NO_PAGE_LINKED_WITH_ENTRY = (
        "no page is linked with the entry, so the entry would lose every page of "
        "the provider"
    )
    MOVE_NOT_SURE = (
        "a page belonging to another work cannot be moved automatically: its links "
        "match no seed entry or multiple entries, the target entry already has a "
        "page from this provider, or their year or type differs"
    )
    DETAILS_DIFFER = "the pages belong to one work, but their years or types differ"
    CONTRADICTING_LINKS = (
        "a page is linked with the entry and also links a different work"
    )
    UNLINKED_PAGES = "a page has no link to the entry and no link to another work"
    OVERLAPPING_LINKS = (
        "the pages share some linked pages, but no page is linked with all of them"
    )
    MULTI_WORK_LINK = "a page naming several pages of one provider links these pages"
    DEAD_PAGE = "a page no longer exists on the provider's site"


_ALWAYS_REVIEWED_OUTCOMES = frozenset(
    {
        DoubledProviderOutcome.DUPLICATE_PAGES,
        DoubledProviderOutcome.PART_PAGES,
        DoubledProviderOutcome.SEPARATE_WORKS,
        DoubledProviderOutcome.UNDECIDED,
    }
)

PageKeySet = Annotated[
    frozenset[str], PlainSerializer(sorted, return_type=list[str], when_used="json")
]


@dataclass(frozen=True, slots=True)
class UnrelatedPage:
    """Hold what a doubled page's links say about the work it belongs to instead."""

    linked_page_keys: PageKeySet
    other_work_entries: tuple[WorkEntry, ...]
    move_target_entry: WorkEntry | None = None


@dataclass(frozen=True, slots=True)
class DoubledProviderDecision:
    """Hold what one doubled provider's pages are, and why a person must check them."""

    provider: str
    page_keys: tuple[str, ...]
    outcome: Annotated[
        DoubledProviderOutcome,
        BeforeValidator(read_member_by_name(DoubledProviderOutcome)),
        PlainSerializer(write_member_name, return_type=str),
    ]
    review_reasons: tuple[
        Annotated[
            DoubledProviderReviewReason,
            BeforeValidator(read_member_by_name(DoubledProviderReviewReason)),
            PlainSerializer(write_member_name, return_type=str),
        ],
        ...,
    ] = ()
    belonging_page_keys: Mapping[str, PageKeySet] = field(default_factory=dict)
    unrelated_pages: Mapping[str, UnrelatedPage] = field(default_factory=dict)
    contradicting_page_keys: tuple[str, ...] = ()
    unlinked_page_keys: tuple[str, ...] = ()
    title_matches: tuple[str, ...] = ()
    year_type_matches: tuple[str, ...] = ()
    season_matches: tuple[str, ...] = ()

    @property
    def is_review_needed(self) -> bool:
        """Report whether a person must check the decision before it is applied.

        Returns:
            True when a review reason applies, or when the outcome is one a
            person always checks.
        """
        return bool(self.review_reasons) or self.outcome in _ALWAYS_REVIEWED_OUTCOMES


class LinkedPageOutcome(StrEnum):
    """Define what happens to a linked page."""

    ADD = "add"
    REVIEW = "review"
    SKIP = "skip"


class LinkedPageReason(StrEnum):
    """Define the rule deciding what happens to a linked page."""

    CHECKS_PASSED = (
        "the page exists, one entry links it, the entry holds no page of its "
        "provider, and their year, type and episode count agree"
    )
    NOT_FOUND = "the provider says the page does not exist"
    HELD_UNDER_OTHER_ADDRESS = (
        "a seed entry already holds this page under a different address"
    )
    SEVERAL_ENTRIES = "two or more entries link the page"
    PROVIDER_IN_ENTRY = "the entry already holds a page of the page's provider"
    SEVERAL_PAGES_OF_PROVIDER = (
        "the entry links to multiple unclaimed pages from this provider"
    )
    DETAILS_DIFFER = "the page's year or type differs from the entry's"
    EPISODE_COUNTS_DIFFER = "the page and the entry list different episode counts"

    @property
    def outcome(self) -> LinkedPageOutcome:
        """Return what a decision with this reason does with the linked page.

        Returns:
            ``ADD``, ``REVIEW`` or ``SKIP``.
        """
        return _OUTCOME_BY_REASON[self]


_OUTCOME_BY_REASON: dict[LinkedPageReason, LinkedPageOutcome] = {
    LinkedPageReason.CHECKS_PASSED: LinkedPageOutcome.ADD,
    LinkedPageReason.NOT_FOUND: LinkedPageOutcome.SKIP,
    LinkedPageReason.HELD_UNDER_OTHER_ADDRESS: LinkedPageOutcome.SKIP,
    LinkedPageReason.SEVERAL_ENTRIES: LinkedPageOutcome.REVIEW,
    LinkedPageReason.PROVIDER_IN_ENTRY: LinkedPageOutcome.REVIEW,
    LinkedPageReason.SEVERAL_PAGES_OF_PROVIDER: LinkedPageOutcome.REVIEW,
    LinkedPageReason.DETAILS_DIFFER: LinkedPageOutcome.REVIEW,
    LinkedPageReason.EPISODE_COUNTS_DIFFER: LinkedPageOutcome.REVIEW,
}


@dataclass(frozen=True, slots=True)
class LinkingEntryDetails:
    """Hold an entry linking a page no known work holds, with the details the checks read."""

    entry: WorkEntry
    episode_count: int | None
    linked_page_keys: frozenset[str]


@dataclass(frozen=True, slots=True)
class LinkedPageDecision:
    """Hold what happens to one linked page and why."""

    page_key: str
    reason: Annotated[
        LinkedPageReason,
        BeforeValidator(read_member_by_name(LinkedPageReason)),
        PlainSerializer(write_member_name, return_type=str),
    ]
    linking_entries: tuple[WorkEntry, ...]
    entry_episode_count: int | None = None

    @property
    def outcome(self) -> LinkedPageOutcome:
        """Return what the decision does with the linked page.

        Returns:
            ``ADD``, ``REVIEW`` or ``SKIP``; a person answers a ``REVIEW``.
        """
        return self.reason.outcome


@dataclass(frozen=True, slots=True)
class LinkedPageAddition:
    """Hold an accepted addition naming the entry a linked page joins and whether as a partial source."""

    entry_reference_page: str
    is_partial_source: bool


@dataclass(frozen=True, slots=True)
class LinkedPageAnswerCheck:
    """Hold a linked-page answer once read, with the accepted answer or the rule it breaks."""

    accepted_answer: LinkedPageAddition | str | None
    broken_rules: tuple[LinkedPageAnswerRule, ...]

    @property
    def is_valid(self) -> bool:
        """Report whether the answer breaks no rule."""
        return not self.broken_rules


class PreviousAnswer(BaseModel):
    """Hold an answered case's answer, carried to the case that replaces it."""

    model_config = ConfigDict(frozen=True)

    case_name: str
    answer: JsonValue
    added_page_keys: tuple[str, ...] = ()
    is_applied: bool


@dataclass(frozen=True, slots=True)
class ChoiceOption:
    """Hold one choice a reviewer can make and the rule that disables it, if any."""

    value: str
    disabling_rule: PlacementRule | LinkedPageAnswerRule | None = None

    @property
    def is_disabled(self) -> bool:
        """Report whether the choice is disabled by a rule.

        Returns:
            True when ``disabling_rule`` is set.
        """
        return self.disabling_rule is not None

    @property
    def description(self) -> str:
        """Return what the choice does, as ``describe_answer_word`` says.

        Returns:
            The user-facing meaning of the choice value.
        """
        return describe_answer_word(self.value)


@dataclass(frozen=True, slots=True)
class GroupPageChoices:
    """Hold one page of an entry group's case, with its current place and the choices left."""

    page_key: str
    work: str | None
    named_entry_page_key: str | None
    partial_works: tuple[str, ...]
    is_required: bool
    work_options: tuple[ChoiceOption, ...]


@dataclass(frozen=True, slots=True)
class WorkPartialSources:
    """Hold one work of an entry group's case, with its partial sources and the pages that can be one."""

    work: str
    partial_source_page_keys: tuple[str, ...]
    page_key_options: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LinkedPageConflict:
    """Hold linked-page answers adding whole pages of one provider to one entry."""

    entry_reference_page: str
    provider: str
    case_names: tuple[str, ...]
