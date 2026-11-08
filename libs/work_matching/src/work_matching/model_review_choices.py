"""Hold the choices the review page offers a reviewer, and the rules disabling them.

``review_choices`` builds these from the answer rules, so the page and a run
never disagree. Each model reads ``<model>: <what it holds>``.

- ``ChoiceOption``: one choice and the rule that disables it, if any.
- ``GroupPageChoices``: one page of an entry group's case, with its current
  place, the works it is a partial source of, and the work choices left.
- ``WorkPartialSources``: one work of an entry group's case, with its partial
  sources and the pages that can become one.
- ``LinkedPageConflict``: linked-page answers adding whole pages of one
  provider to one entry.
"""

from dataclasses import dataclass

from work_matching.answer_rules import LinkedPageAnswerRule, PlacementRule
from work_matching.answer_words import describe_answer_word


@dataclass(frozen=True, slots=True)
class ChoiceOption:
    """Hold one choice a reviewer can make and the rule that disables it, if any."""

    # The value as an answer writes it, such as ``work 2``, ``dropped`` or
    # ``add to <entry>``.
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
    # ``work N``, ``other work`` or ``dropped``; None while the page has no place.
    work: str | None
    # The page key written after ``other work``; None when no entry is named.
    named_entry_page_key: str | None
    # The works the page is also a partial source of, in answer words.
    partial_works: tuple[str, ...]
    # Whether the answer must place this page to be saved.
    is_required: bool
    work_options: tuple[ChoiceOption, ...]


@dataclass(frozen=True, slots=True)
class WorkPartialSources:
    """Hold one work of an entry group's case, with its partial sources and the pages that can be one."""

    # ``work N`` for a work of the case, or ``other work <page key>`` for a
    # seed entry outside the case.
    work: str
    # The pages that are partial sources of the work, in answer order.
    partial_source_page_keys: tuple[str, ...]
    # The pages that can become a partial source of the work: pages that are
    # a source of another work of the case and not yet a partial source here.
    page_key_options: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LinkedPageConflict:
    """Hold linked-page answers adding whole pages of one provider to one entry."""

    entry_reference_page: str
    provider: str
    case_names: tuple[str, ...]
