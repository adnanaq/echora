from enrichment.pipeline.fetch_summary import (
    FetchOutcome,
    FetchSummary,
    describe_fetch_summary,
)


def test_describe_fetch_summary_counts_then_first_failures_listed() -> None:
    fetch_summary = FetchSummary(
        outcomes={FetchOutcome.FETCHED: 3, FetchOutcome.FAILED: 2},
        errors_by_request_label={
            "anidb:anime:1": "banned",
            "anidb:anime:2": "banned",
        },
    )
    assert describe_fetch_summary("Linked pages", fetch_summary, 1) == [
        "Linked pages failed: 2",
        "Linked pages fetched: 3",
        "Linked pages failed, retried next run: 2",
        "  anidb:anime:1: banned",
    ]
