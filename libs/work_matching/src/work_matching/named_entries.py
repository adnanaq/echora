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
- At most ``MAX_TARGET_ENTRY_SUGGESTIONS`` are kept, the closest titles first,
  so a long series does not swell every case of the review file.
"""

from typing import Final

from pydantic import JsonValue
from work_matching.known_work_entries import KnownWorkEntries
from work_matching.page_placement import read_placement
from work_matching.provider_page_key import get_page_provider
from work_matching.review_cases import ReviewCase, TargetEntrySuggestion

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
        for place_text in answer.values()
        if isinstance(place_text, str) and (placement := read_placement(place_text))
    )
    return list(dict.fromkeys(key for key in named_entry_page_keys if key))


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
        page_key: work_entry.reference_page
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
    series_entries = {
        work_entry.reference_page: work_entry
        for case_entry in case.entries.values()
        for work_entry in known_work_entries.find_by_series(case_entry.title)
        if work_entry.reference_page not in case.entries
    }
    return tuple(
        TargetEntrySuggestion(
            reference_page=reference_page,
            title=work_entry.title,
            providers=tuple(
                sorted(
                    {
                        provider
                        for page_key in work_entry.addresses_by_page_key
                        if (provider := get_page_provider(page_key))
                    }
                )
            ),
        )
        for reference_page, work_entry in list(series_entries.items())[
            :MAX_TARGET_ENTRY_SUGGESTIONS
        ]
    )
