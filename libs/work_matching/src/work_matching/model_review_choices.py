"""Hold the choices the review page offers a reviewer, and the rules disabling them.

``review_choices`` builds these from the answer rules, so the page and a run
never disagree. Each model reads ``<model>: <what it holds>``.

- ``ChoiceRule``: a reason a choice is not offered that no answer rule states.
- ``ChoiceOption``: one choice and the rule that disables it, if any.
- ``GroupPageChoices``: one page of an entry group's case, with its current
  place and the work and mark choices left.
- ``LinkedPageConflict``: linked-page answers adding whole pages of one
  provider to one entry.
"""

from dataclasses import dataclass
from enum import StrEnum

from work_matching.answer_rules import LinkedPageAnswerRule, PlacementRule
from work_matching.answer_words import describe_answer_word


class ChoiceRule(StrEnum):
    """Define a reason a choice is not offered that no answer rule states.

    The answer rules (``PlacementRule``) check a whole answer when it is
    saved. These reasons explain a single dropdown option to the reviewer.
    """

    MARK_WITHOUT_WORK = (
        "choose a work for the page first, since a mark applies within a work"
    )
    MARK_IN_OTHER_WORK = (
        "a page moved to another work cannot be marked 'main' or 'part' in this case"
    )


@dataclass(frozen=True, slots=True)
class ChoiceOption:
    """Hold one choice a reviewer can make and the rule that disables it, if any."""

    # As written in the answer: ``work 2``, ``part``, ``add to <entry>``.
    value: str
    disabling_rule: PlacementRule | LinkedPageAnswerRule | ChoiceRule | None = None

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
    """Hold one page of an entry group's case: its current place and the choices left."""

    page_key: str
    # ``work N`` or ``other work``; None while the page has no place.
    work: str | None
    # The page key written after ``other work``; None when no entry is named.
    named_entry_page_key: str | None
    # ``NO_MARK``, ``MAIN_MARK`` or ``PART_MARK``.
    mark: str
    # Whether the answer must place this page to be saved.
    is_required: bool
    work_options: tuple[ChoiceOption, ...]
    mark_options: tuple[ChoiceOption, ...]


@dataclass(frozen=True, slots=True)
class LinkedPageConflict:
    """Hold linked-page answers adding whole pages of one provider to one entry."""

    entry_reference_page: str
    provider: str
    case_names: tuple[str, ...]
