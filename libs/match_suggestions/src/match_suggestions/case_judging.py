"""Ask Jev about the page pairs of open review cases and keep its answers.

The batch is split in two, so a caller can count and price the requests
before any is sent. Each item gives the rule, then the reason for it.

- A pair is asked only when it holds an open page, since only an open page's
  dropdown is suggested. Pairs of two placed pages are never read.
- A pair is asked again only when what would be sent changed, a new synopsis
  or new wording included. The stored answer is reused otherwise.
- Synopses come from a reader the caller passes, so this library imports
  nothing from enrichment.
- A failed request is counted and never stored, so a failure is never read as
  an answer. The pair is asked again on the next run.
- Each answer is stored as soon as it arrives, so a stopped batch keeps what
  it was given.
"""

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from match_suggestions.model_pair_judgment import PairJudgment, build_page_key_pair
from match_suggestions.page_suggestions import list_open_page_keys, list_placeable_pages
from match_suggestions.pair_judgment_store import PairJudgmentStore
from match_suggestions.pair_questions import (
    JEV_MODEL,
    PAIR_QUESTIONS,
    QUESTION_VERSION,
    build_pair_state,
    build_request_fingerprint,
)
from typesafe_sdk import AsyncTypeSafeClient, TypeSafeError
from work_matching import ProviderPage, ReviewCase, ReviewCaseKind

logger = logging.getLogger(__name__)

SynopsisReader = Callable[[ProviderPage], Awaitable[str | None]]


@dataclass(frozen=True, slots=True)
class PairRequest:
    """Hold one pair of pages to ask Jev about and exactly what would be sent."""

    first_page: ProviderPage
    second_page: ProviderPage
    pair_state: dict[str, Any]
    request_fingerprint: str

    @property
    def page_key_pair(self) -> tuple[str, str]:
        """Return the pair's page keys, the lower key first."""
        return (self.first_page.key, self.second_page.key)


# Characters of JSON per input token TypeSafe bills, from the measurement: the
# labelled pairs averaged about 1,100 tokens for 2,900 characters.
_CHARACTERS_PER_INPUT_TOKEN = 2.6


@dataclass(frozen=True, slots=True)
class JudgingSummary:
    """Count the pairs a batch answered and failed, and the input tokens billed."""

    judged_pair_count: int
    failed_pair_count: int
    input_token_count: int


async def list_pair_requests(
    review_cases: Iterable[ReviewCase],
    store: PairJudgmentStore,
    read_synopsis: SynopsisReader,
) -> tuple[PairRequest, ...]:
    """List the pairs of the open cases that have no answer for what would be sent.

    Args:
        review_cases: The review cases, open and answered.
        store: The answers already given.
        read_synopsis: Reads a page's English synopsis, or None.

    Returns:
        One request per pair to ask, sorted by the pair's page keys.
    """
    pages_by_pair: dict[tuple[str, str], tuple[ProviderPage, ProviderPage]] = {}
    for review_case in review_cases:
        if (
            review_case.answer is not None
            or review_case.kind is ReviewCaseKind.LINKED_PAGE
        ):
            continue
        pages_by_key = list_placeable_pages(review_case)
        for open_page_key in list_open_page_keys(review_case):
            for page_key in pages_by_key.keys() - {open_page_key}:
                first_page_key, second_page_key = build_page_key_pair(
                    open_page_key, page_key
                )
                pages_by_pair[(first_page_key, second_page_key)] = (
                    pages_by_key[first_page_key],
                    pages_by_key[second_page_key],
                )
    synopses = await _read_synopses(
        {
            page.key: page for page_pair in pages_by_pair.values() for page in page_pair
        }.values(),
        read_synopsis,
    )
    stored_judgments = store.find_judgments(pages_by_pair)
    pair_requests = []
    for page_key_pair, (first_page, second_page) in sorted(pages_by_pair.items()):
        pair_state = build_pair_state(first_page, second_page, synopses)
        request_fingerprint = build_request_fingerprint(pair_state)
        stored_judgment = stored_judgments.get(page_key_pair)
        if (
            stored_judgment is None
            or stored_judgment.request_fingerprint != request_fingerprint
        ):
            pair_requests.append(
                PairRequest(first_page, second_page, pair_state, request_fingerprint)
            )
    return tuple(pair_requests)


def estimate_input_tokens(pair_requests: Iterable[PairRequest]) -> int:
    """Estimate the input tokens asking these pairs would bill, before any is sent.

    Args:
        pair_requests: The pairs to ask.

    Returns:
        The estimated input tokens, the state and questions of every request.
    """
    question_characters = len(
        json.dumps(
            {
                name: question.model_dump(mode="json")
                for name, question in PAIR_QUESTIONS.items()
            }
        )
    )
    request_characters = sum(
        len(json.dumps(pair_request.pair_state)) + question_characters
        for pair_request in pair_requests
    )
    return round(request_characters / _CHARACTERS_PER_INPUT_TOKEN)


async def judge_pair_requests(
    pair_requests: Iterable[PairRequest],
    store: PairJudgmentStore,
    client: AsyncTypeSafeClient,
    *,
    max_concurrent_requests: int = 8,
) -> JudgingSummary:
    """Ask Jev every listed pair and store each answer as it arrives.

    Args:
        pair_requests: The pairs ``list_pair_requests`` listed.
        store: Where the answers are kept.
        client: TypeSafe's client, which retries rate limits and server errors.
        max_concurrent_requests: The most requests waiting for an answer at once.

    Returns:
        How many pairs were answered and failed, and the input tokens billed.
    """
    request_slots = asyncio.Semaphore(max_concurrent_requests)

    async def judge_pair(pair_request: PairRequest) -> int | None:
        async with request_slots:
            try:
                system_one_response = await client.system_one(
                    pair_request.pair_state, PAIR_QUESTIONS, model=JEV_MODEL
                )
            except TypeSafeError as error:
                logger.warning(
                    f"Jev gave no answer for {pair_request.page_key_pair}: {error}"
                )
                return None
        probabilities = {
            name: answer.noul for name, answer in system_one_response.nouls.items()
        }
        store.add_judgments(
            [
                PairJudgment(
                    first_page_key=pair_request.first_page.key,
                    second_page_key=pair_request.second_page.key,
                    question_version=QUESTION_VERSION,
                    request_fingerprint=pair_request.request_fingerprint,
                    judged_at=datetime.now(UTC),
                    **probabilities,
                )
            ]
        )
        return system_one_response.usage.input_tokens or 0

    input_tokens_by_request = await asyncio.gather(*map(judge_pair, pair_requests))
    billed_input_tokens = [
        input_tokens
        for input_tokens in input_tokens_by_request
        if input_tokens is not None
    ]
    return JudgingSummary(
        len(billed_input_tokens),
        len(input_tokens_by_request) - len(billed_input_tokens),
        sum(billed_input_tokens),
    )


async def _read_synopses(
    pages: Iterable[ProviderPage], read_synopsis: SynopsisReader
) -> dict[str, str]:
    """Read the synopsis of each page once, one page at a time.

    One read at a time leaves pacing to the reader, which knows each provider's limits.

    Args:
        pages: The pages to read.
        read_synopsis: Reads a page's English synopsis, or None.

    Returns:
        Each synopsis found, by page key.
    """
    synopses = {}
    for page in pages:
        if synopsis := await read_synopsis(page):
            synopses[page.key] = synopsis
    return synopses
