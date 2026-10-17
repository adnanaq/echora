"""Decide which provider pages belong to the same work.

Seed cleaning uses this now; seed link filling and discovery will use it later.
Terms are as in ``GLOSSARY.md``. This library imports nothing from enrichment.
"""

from work_matching.comparable_title import comparable_title
from work_matching.doubled_provider_check import (
    DoubledProviderDecision,
    Outcome,
    Reason,
    UnrelatedPage,
    decide_doubled_providers,
)
from work_matching.known_works import KnownWorks
from work_matching.known_works_from_seed import KnownWorksFromSeed
from work_matching.provider_page import ProviderPage, pages_by_current_key
from work_matching.provider_page_key import provider_of_key, provider_page_key
from work_matching.provider_page_links import ProviderPageLinks
from work_matching.provider_page_store import ProviderPageStore
from work_matching.review_cases import DuplicateCaseError, ReviewCase, ReviewFile
from work_matching.split_work_check import (
    SplitWorkDecision,
    SplitWorkReason,
    decide_split_work,
)
from work_matching.work_entry import WorkEntry

__all__ = [
    "DoubledProviderDecision",
    "DuplicateCaseError",
    "KnownWorks",
    "KnownWorksFromSeed",
    "Outcome",
    "ProviderPage",
    "ProviderPageLinks",
    "ProviderPageStore",
    "Reason",
    "ReviewCase",
    "ReviewFile",
    "SplitWorkDecision",
    "SplitWorkReason",
    "UnrelatedPage",
    "WorkEntry",
    "comparable_title",
    "decide_doubled_providers",
    "decide_split_work",
    "pages_by_current_key",
    "provider_of_key",
    "provider_page_key",
]
