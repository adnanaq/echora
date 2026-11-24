"""Turn Jev's answers about a case's page pairs into a dropdown word per open page.

An open page is a page of the case's entries that the links leave without a
place. Searched pages, such as AnimeSchedule's, follow the page they were found
by and are never placed. Each item gives the rule, then the reason for it.

- The works are the ones the case's suggested answer names. Without any, the
  most trusted page is ``work 1``, the work the others are compared with.
- A page joins a work when Jev calls it the same work as that work's pages and
  as no other work. A suggested place, including one from a previous answer,
  is never changed.
- A page is a partial source of a work when Jev calls one of them part of the
  other, and of no other work. Partial sources go both ways, so the page gets
  a new work that is ``partial in`` the work, and the work's pages are
  ``partial in`` the new work.
- No partial source when Jev calls each work part of the other, or when the
  work holds a page of the same provider that Jev calls a different work. A
  provider listing that work on a page of its own does not cover it.
- A page apart from every work gets a new work. Open pages Jev calls the same
  work share one new work.
- Two pages of one provider in one work are both left open, because no work
  holds two pages of one provider.
- Each suggestion's score is its weakest pair, the answer least sure of it.
"""

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations

from match_suggestions.model_pair_judgment import PairJudgment, build_page_key_pair
from match_suggestions.pair_judgment_store import PairJudgmentStore
from match_suggestions.pair_questions import QUESTION_VERSION
from match_suggestions.pair_relation import (
    PairRelation,
    decide_pair_relation,
    read_part_probability,
)
from networkx.utils import UnionFind
from work_matching import (
    PagePlacement,
    ProviderPage,
    ReviewCase,
    ReviewCaseKind,
    rank_by_provider_trust,
    read_page_provider,
    read_placements,
    write_placement,
)
from work_matching.page_placement import build_page_keys_by_work_provider

# Jev calling each work of a pair part of the other at this level contradicts
# a partial source, since two works cannot each contain the other.
_CONTRADICTED_PART_PROBABILITY = 0.5


@dataclass(frozen=True, slots=True)
class PageSuggestion:
    """Hold the dropdown word suggested for one page and how sure Jev is of it."""

    page_key: str
    # A place as a review answer writes it, such as ``work 2, partial in work 1``.
    answer_word: str
    # The probability of the weakest pair behind the suggestion, from 0.7 to 1.
    score: float


@dataclass(frozen=True, slots=True)
class _OpenPagePlace:
    """Hold where Jev puts one open page relative to the case's works."""

    relation: PairRelation
    # The work the page joins or is a partial source of; None for a new work.
    work: str | None
    score: float


class _CasePairs:
    """Read Jev's relations between the pages of one case."""

    def __init__(
        self,
        pages_by_key: Mapping[str, ProviderPage],
        judgments_by_pair: Mapping[tuple[str, str], PairJudgment],
    ) -> None:
        """Hold the case's pages and the answers of the current questions.

        Args:
            pages_by_key: The case's pages with details, by page key.
            judgments_by_pair: Stored answers by page pair, of any version.
        """
        self._pages_by_key = pages_by_key
        self._judgments_by_pair = {
            page_key_pair: pair_judgment
            for page_key_pair, pair_judgment in judgments_by_pair.items()
            if pair_judgment.question_version == QUESTION_VERSION
        }

    def find_judgment(self, page_key: str, other_page_key: str) -> PairJudgment | None:
        """Return the answer about two pages, or None when they were not asked.

        Args:
            page_key: One page.
            other_page_key: The other page.

        Returns:
            Jev's answer about the pair, or None.
        """
        return self._judgments_by_pair.get(
            build_page_key_pair(page_key, other_page_key)
        )

    def find_relation(self, page_key: str, other_page_key: str) -> PairRelation | None:
        """Return the relation of two pages, or None when they were not asked.

        Args:
            page_key: One page.
            other_page_key: The other page.

        Returns:
            The relation the decision table gives, or None.
        """
        pair_judgment = self.find_judgment(page_key, other_page_key)
        if pair_judgment is None:
            return None
        first_page_key, second_page_key = pair_judgment.page_key_pair
        return decide_pair_relation(
            pair_judgment,
            self._pages_by_key[first_page_key],
            self._pages_by_key[second_page_key],
        )


def list_open_page_keys(review_case: ReviewCase) -> tuple[str, ...]:
    """List the case's pages that Jev can place: placeable, unplaced and not searched.

    Args:
        review_case: The review case.

    Returns:
        The open page keys, sorted.
    """
    pages_by_key = list_placeable_pages(review_case)
    placed_page_keys = {
        page_key
        for work_page_keys in build_suggested_works(review_case).values()
        for page_key in work_page_keys
    }
    return tuple(
        sorted(
            pages_by_key.keys() - placed_page_keys - review_case.suggested_answer.keys()
        )
    )


def build_suggested_works(review_case: ReviewCase) -> dict[str, frozenset[str]]:
    """Group the pages the suggested answer places by their work.

    Args:
        review_case: The review case.

    Returns:
        The pages of each ``work N``, or the most trusted page alone as
        ``work 1`` when the suggested answer names no work.
    """
    pages_by_work: defaultdict[str, set[str]] = defaultdict(set)
    for page_key, placement in read_placements(review_case.suggested_answer).items():
        if placement.is_case_work:
            pages_by_work[placement.placement_word].add(page_key)
    placeable_page_keys = list_placeable_pages(review_case)
    if not pages_by_work and placeable_page_keys:
        pages_by_work["work 1"].add(
            min(sorted(placeable_page_keys), key=rank_by_provider_trust)
        )
    return {work: frozenset(page_keys) for work, page_keys in pages_by_work.items()}


def suggest_page_answers(
    review_case: ReviewCase, judgments_by_pair: Mapping[tuple[str, str], PairJudgment]
) -> tuple[PageSuggestion, ...]:
    """Suggest a dropdown word for each page Jev is sure of, by the rules above.

    Args:
        review_case: The review case.
        judgments_by_pair: Stored answers about the case's page pairs.

    Returns:
        One suggestion per page Jev places, sorted by page key; empty for an
        answered or linked-page case.
    """
    if review_case.answer is not None or review_case.kind is ReviewCaseKind.LINKED_PAGE:
        return ()
    pages_by_key = list_placeable_pages(review_case)
    case_pairs = _CasePairs(pages_by_key, judgments_by_pair)
    pages_by_work = build_suggested_works(review_case)
    if not pages_by_work:
        return ()
    places_by_page = {
        page_key: open_page_place
        for page_key in list_open_page_keys(review_case)
        if (open_page_place := _place_open_page(page_key, pages_by_work, case_pairs))
    }
    answer_words_by_page = _write_open_page_words(
        places_by_page, pages_by_work, pages_by_key, case_pairs
    )
    answer_words_by_page = _leave_out_doubled_providers(
        answer_words_by_page, pages_by_work
    )
    page_suggestions = [
        PageSuggestion(page_key, answer_word, places_by_page[page_key].score)
        for page_key, answer_word in answer_words_by_page.items()
    ]
    page_suggestions += _suggest_work_pages(
        review_case, page_suggestions, pages_by_work
    )
    return tuple(
        sorted(page_suggestions, key=lambda page_suggestion: page_suggestion.page_key)
    )


def suggest_stored_page_answers(
    review_case: ReviewCase, store: PairJudgmentStore
) -> tuple[PageSuggestion, ...]:
    """Suggest a dropdown word for each open page from the answers the store holds.

    Args:
        review_case: The review case.
        store: Jev's stored answers.

    Returns:
        The suggestions ``suggest_page_answers`` gives for the case's stored pairs.
    """
    page_key_pairs = [
        build_page_key_pair(page_key, other_page_key)
        for page_key, other_page_key in combinations(
            list_placeable_pages(review_case), 2
        )
    ]
    return suggest_page_answers(review_case, store.find_judgments(page_key_pairs))


def list_placeable_pages(review_case: ReviewCase) -> dict[str, ProviderPage]:
    """Return the case's placeable pages that have details and were not found by a search.

    Args:
        review_case: The review case.

    Returns:
        The pages by page key, in ``ReviewCase.placeable_page_keys`` order, so
        suggestions name pages by the keys the review page offers.
    """
    pages_by_key = {
        page.key: page for page in review_case.pages if not page.searched_by
    }
    return {
        page_key: pages_by_key[page_key]
        for page_key in review_case.placeable_page_keys
        if page_key in pages_by_key
    }


def _place_open_page(
    page_key: str, pages_by_work: Mapping[str, frozenset[str]], case_pairs: _CasePairs
) -> _OpenPagePlace | None:
    """Place one open page relative to the case's works, or None when Jev is unsure.

    Args:
        page_key: The open page.
        pages_by_work: The pages of each work.
        case_pairs: Jev's relations between the case's pages.

    Returns:
        The page's place, or None.
    """
    same_works, partial_works = [], []
    is_apart_from_every_work = True
    for work, work_page_keys in pages_by_work.items():
        relations = {
            case_pairs.find_relation(page_key, work_page_key)
            for work_page_key in work_page_keys
        }
        if PairRelation.SAME_WORK in relations and relations <= {
            PairRelation.SAME_WORK,
            None,
        }:
            same_works.append(work)
        if (
            PairRelation.PARTIAL_IN in relations
            and PairRelation.SAME_WORK not in relations
            and not _is_partial_source_contradicted(
                page_key, work_page_keys, case_pairs
            )
        ):
            partial_works.append(work)
        is_apart_from_every_work &= relations == {PairRelation.DIFFERENT_WORKS}
    if len(same_works) == 1 and not partial_works:
        return _score_place(
            page_key, PairRelation.SAME_WORK, same_works[0], pages_by_work, case_pairs
        )
    if len(partial_works) == 1 and not same_works:
        return _score_place(
            page_key,
            PairRelation.PARTIAL_IN,
            partial_works[0],
            pages_by_work,
            case_pairs,
        )
    if not same_works and not partial_works and is_apart_from_every_work:
        return _score_place(
            page_key, PairRelation.DIFFERENT_WORKS, None, pages_by_work, case_pairs
        )
    return None


def _is_partial_source_contradicted(
    page_key: str, work_page_keys: frozenset[str], case_pairs: _CasePairs
) -> bool:
    """Report whether Jev's other answers rule out a page being a partial source of a work.

    Args:
        page_key: The open page.
        work_page_keys: The work's pages.
        case_pairs: Jev's relations between the case's pages.

    Returns:
        True when Jev calls each work part of the other, or calls a page of the
        same provider in the work a different work.
    """
    for work_page_key in work_page_keys:
        pair_judgment = case_pairs.find_judgment(page_key, work_page_key)
        if pair_judgment is None:
            continue
        if (
            min(
                pair_judgment.is_first_work_part_of_second_work,
                pair_judgment.is_second_work_part_of_first_work,
            )
            >= _CONTRADICTED_PART_PROBABILITY
        ):
            return True
        if read_page_provider(work_page_key) == read_page_provider(page_key) and (
            case_pairs.find_relation(page_key, work_page_key)
            is PairRelation.DIFFERENT_WORKS
        ):
            return True
    return False


def _score_place(
    page_key: str,
    relation: PairRelation,
    work: str | None,
    pages_by_work: Mapping[str, frozenset[str]],
    case_pairs: _CasePairs,
) -> _OpenPagePlace:
    """Score a place by its weakest pair.

    Args:
        page_key: The open page.
        relation: How the page relates to ``work``, or to every work.
        work: The work the page joins or is a partial source of, or None.
        pages_by_work: The pages of each work.
        case_pairs: Jev's relations between the case's pages.

    Returns:
        The place with the lowest probability among the pairs behind it.
    """
    compared_page_keys = (
        pages_by_work[work] if work else frozenset().union(*pages_by_work.values())
    )
    probabilities = []
    for compared_page_key in compared_page_keys:
        pair_judgment = case_pairs.find_judgment(page_key, compared_page_key)
        if pair_judgment is None:
            continue
        if relation is PairRelation.SAME_WORK:
            probabilities.append(pair_judgment.is_same_work)
        elif relation is PairRelation.DIFFERENT_WORKS:
            probabilities.append(1 - pair_judgment.is_same_work)
        elif (
            case_pairs.find_relation(page_key, compared_page_key)
            is PairRelation.PARTIAL_IN
        ):
            probabilities.append(read_part_probability(pair_judgment))
    return _OpenPagePlace(relation, work, round(min(probabilities), 4))


def _write_open_page_words(
    places_by_page: Mapping[str, _OpenPagePlace],
    pages_by_work: Mapping[str, frozenset[str]],
    pages_by_key: Mapping[str, ProviderPage],
    case_pairs: _CasePairs,
) -> dict[str, str]:
    """Write each placed open page's dropdown word, giving new works their numbers.

    Open pages that are not the same work as a case work become new works. Two
    of them share a new work when Jev calls them the same work and they relate
    to the case's works the same way. New works are numbered after the case's
    works, the earliest aired first.

    Args:
        places_by_page: Each placed open page's place.
        pages_by_work: The pages of each work.
        pages_by_key: The case's pages with details.
        case_pairs: Jev's relations between the case's pages.

    Returns:
        Each placed open page's answer word.
    """
    answer_words_by_page = {
        page_key: open_page_place.work
        for page_key, open_page_place in places_by_page.items()
        if open_page_place.relation is PairRelation.SAME_WORK and open_page_place.work
    }
    new_work_page_keys = sorted(places_by_page.keys() - answer_words_by_page.keys())
    new_work_sets = UnionFind(new_work_page_keys)
    for page_key, other_page_key in combinations(new_work_page_keys, 2):
        place, other_place = places_by_page[page_key], places_by_page[other_page_key]
        if case_pairs.find_relation(
            page_key, other_page_key
        ) is PairRelation.SAME_WORK and (
            (place.relation, place.work) == (other_place.relation, other_place.work)
        ):
            new_work_sets.union(page_key, other_page_key)
    next_work_number = max(int(work.split()[1]) for work in pages_by_work) + 1
    for new_work_number, page_keys in enumerate(
        sorted(
            new_work_sets.to_sets(),
            key=lambda page_keys: _read_first_aired(page_keys, pages_by_key),
        ),
        start=next_work_number,
    ):
        for page_key in page_keys:
            covering_work = places_by_page[page_key].work
            answer_words_by_page[page_key] = write_placement(
                PagePlacement(
                    f"work {new_work_number}",
                    partial_works=(covering_work,) if covering_work else (),
                )
            )
    return answer_words_by_page


def _read_first_aired(
    page_keys: set[str], pages_by_key: Mapping[str, ProviderPage]
) -> str:
    """Read the earliest first air date of a set of pages, for ordering new works.

    Args:
        page_keys: The pages of one new work.
        pages_by_key: The case's pages with details.

    Returns:
        The earliest first air date as ISO text, then the lowest page key.
    """
    return min(
        f"{aired_from.isoformat() if (aired_from := pages_by_key[page_key].aired_from) else '9999'} {page_key}"
        for page_key in page_keys
    )


def _leave_out_doubled_providers(
    answer_words_by_page: Mapping[str, str], pages_by_work: Mapping[str, frozenset[str]]
) -> dict[str, str]:
    """Leave out suggested pages that would put two pages of one provider in one work.

    Args:
        answer_words_by_page: Each suggested open page's answer word.
        pages_by_work: The pages of each work.

    Returns:
        The suggestions that keep one page per provider in every work.
    """
    placements = {
        page_key: PagePlacement(work)
        for work, work_page_keys in pages_by_work.items()
        for page_key in work_page_keys
    } | read_placements(answer_words_by_page)
    doubled_page_keys = {
        page_key
        for page_keys in build_page_keys_by_work_provider(placements).values()
        if len(page_keys) > 1
        for page_key in page_keys
    }
    return {
        page_key: answer_word
        for page_key, answer_word in answer_words_by_page.items()
        if page_key not in doubled_page_keys
    }


def _suggest_work_pages(
    review_case: ReviewCase,
    open_page_suggestions: list[PageSuggestion],
    pages_by_work: Mapping[str, frozenset[str]],
) -> list[PageSuggestion]:
    """Suggest the case works' pages that the open pages' suggestions change or need.

    A work's pages become ``partial in`` each new work that is partial in the
    work. The most trusted page standing in for ``work 1``, which the
    suggested answer does not place, is suggested whenever any page is.

    Args:
        review_case: The review case.
        open_page_suggestions: The suggestions for open pages.
        pages_by_work: The pages of each work.

    Returns:
        The suggestions for the works' pages, each scored by the weakest
        suggestion it follows from.
    """
    suggested_placements = read_placements(review_case.suggested_answer)
    new_works_by_work: defaultdict[str, list[PageSuggestion]] = defaultdict(list)
    for page_suggestion in open_page_suggestions:
        placement = read_placements(
            {page_suggestion.page_key: page_suggestion.answer_word}
        )[page_suggestion.page_key]
        for partial_work in placement.partial_works:
            new_works_by_work[partial_work].append(page_suggestion)
    work_page_suggestions = []
    for work, work_page_keys in pages_by_work.items():
        partial_suggestions = new_works_by_work.get(work, [])
        for page_key in sorted(work_page_keys):
            suggested_placement = suggested_placements.get(page_key)
            if suggested_placement is None and not open_page_suggestions:
                continue
            if suggested_placement is not None and not partial_suggestions:
                continue
            partial_works = (
                *(suggested_placement.partial_works if suggested_placement else ()),
                *(
                    page_suggestion.answer_word.split(",")[0]
                    for page_suggestion in partial_suggestions
                ),
            )
            followed_suggestions = partial_suggestions or open_page_suggestions
            work_page_suggestions.append(
                PageSuggestion(
                    page_key,
                    write_placement(
                        PagePlacement(
                            work, partial_works=tuple(dict.fromkeys(partial_works))
                        )
                    ),
                    min(
                        page_suggestion.score
                        for page_suggestion in followed_suggestions
                    ),
                )
            )
    return work_page_suggestions
