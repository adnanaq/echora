"""Look up the seed entries a review answer names, and suggest ones to name.

An answer can send a page to an existing seed entry with ``other work <page
key>``. Each item gives the rule, then the reason for it.

- A named page key is looked up in the seed (``KnownWorkEntries``) before the
  answer is checked, and the case keeps each found entry's reference page
  (``ReviewCase.named_entry_reference_pages``). The answer checks then stay
  functions of the case alone, and a key no seed entry holds is refused.
- The run looks the names up while building its cases, and the review page
  looks up an answer's names before saving it, with the same function, so
  both refuse the same answers.
- Suggestions are the seed entries of the same series as the case's entries
  (``KnownWorkEntries.find_by_series``), leaving out the case's own entries.
  Titles can share words by chance, so suggestions are offered to the
  reviewer and never applied without an answer.
- At most ``MAX_TARGET_ENTRY_SUGGESTIONS`` are kept, so a long series does not
  swell every case of the review file. The closest titles are kept first, and a
  title is closer when it shares more leading words with a case entry's title.
  Titles are ranked across all the case's entries, so one entry's long series
  cannot crowd out another entry's close matches.
"""

from typing import Final

from pydantic import JsonValue
from work_matching.comparable_title import fold_title_for_comparison
from work_matching.known_work_entries import KnownWorkEntries
from work_matching.page_placement import read_placement
from work_matching.provider_page_key import read_page_provider
from work_matching.review_cases import ReviewCase, TargetEntrySuggestion
from work_matching.work_entry import WorkEntry

MAX_TARGET_ENTRY_SUGGESTIONS: Final = 12


def list_named_entry_page_keys(answer: JsonValue) -> list[str]:
    """List the page keys an answer names after ``other work``, in answer order.

    Args:
        answer: An answer as written in the review file, of any case kind.

    Returns:
        Each named key once; none when the answer names no entry.
    """
    if not isinstance(answer, dict):
        return []
    named_entry_page_keys = (
        placement.named_entry_page_key
        for placement_text in answer.values()
        if isinstance(placement_text, str)
        and (placement := read_placement(placement_text))
    )
    return list(
        dict.fromkeys(page_key for page_key in named_entry_page_keys if page_key)
    )


def resolve_named_entries(
    case: ReviewCase,
    known_work_entries: KnownWorkEntries,
    answer: JsonValue = None,
) -> ReviewCase:
    """Return the case with the seed entries its answer names looked up.

    Args:
        case: The review case.
        known_work_entries: The seed's entries.
        answer: The answer whose names to look up; None looks up the case's
            own answer.

    Returns:
        The case, its ``named_entry_reference_pages`` holding each named key a
        seed entry holds, mapped to that entry's reference page.
    """
    named_entry_reference_pages = {
        page_key: work_entry.reference_page_key
        for page_key in list_named_entry_page_keys(
            case.answer if answer is None else answer
        )
        if (work_entry := known_work_entries.find_by_page_key(page_key))
    }
    return case.model_copy(
        update={"named_entry_reference_pages": named_entry_reference_pages}
    )


def suggest_target_entries(
    case: ReviewCase, known_work_entries: KnownWorkEntries
) -> tuple[TargetEntrySuggestion, ...]:
    """Suggest the seed entries of the case's series that a page could move to.

    Args:
        case: An entry group's review case.
        known_work_entries: The seed's entries.

    Returns:
        Each series entry outside the case once, with its title and the
        providers it already holds, closest titles first, at most
        ``MAX_TARGET_ENTRY_SUGGESTIONS``.
    """
    series_entries: dict[str, WorkEntry] = {}
    shared_word_counts: dict[str, int] = {}
    for case_entry in case.entries.values():
        for work_entry in known_work_entries.find_by_series(case_entry.title):
            reference_page = work_entry.reference_page_key
            if reference_page in case.entries:
                continue
            series_entries.setdefault(reference_page, work_entry)
            shared_word_counts[reference_page] = max(
                shared_word_counts.get(reference_page, 0),
                _count_shared_leading_words(case_entry.title, work_entry.title),
            )
    closest_entries = sorted(
        series_entries.values(),
        key=lambda work_entry: -shared_word_counts[work_entry.reference_page_key],
    )[:MAX_TARGET_ENTRY_SUGGESTIONS]
    return tuple(
        TargetEntrySuggestion(
            entry_reference_page=work_entry.reference_page_key,
            title=work_entry.title,
            providers=tuple(
                sorted(
                    {
                        provider
                        for page_key in work_entry.addresses_by_page_key
                        if (provider := read_page_provider(page_key))
                    }
                )
            ),
        )
        for work_entry in closest_entries
    )


def _count_shared_leading_words(first_title: str, second_title: str) -> int:
    """Count the leading words two titles share once folded for comparison.

    Args:
        first_title: A title, as written.
        second_title: Another title, as written.

    Returns:
        How many words both folded titles start with, in the same order.
    """
    shared_word_count = 0
    for first_word, second_word in zip(
        fold_title_for_comparison(first_title).split(),
        fold_title_for_comparison(second_title).split(),
        strict=False,
    ):
        if first_word != second_word:
            break
        shared_word_count += 1
    return shared_word_count
