"""Define domain and storage errors raised by the review page application."""

from pathlib import Path


class ReviewPageError(RuntimeError):
    """Base class for review page domain and storage errors."""


class SeedFileUnavailableError(ReviewPageError):
    """Report that the seed file an answer's named entries are checked against cannot be read."""

    def __init__(self) -> None:
        """Explain how to make a seed file available."""
        super().__init__(
            "The seed file that named entries are checked against is missing or "
            "unknown, so nothing was saved. Set REVIEW_PAGE_SEED_FILE, or run "
            "scripts/clean_seed.py once so its result names the file."
        )


class RunSeedFileUnknownError(ReviewPageError):
    """Indicate that no seed file is configured or recorded for a pipeline run."""

    def __init__(self) -> None:
        """Initialize the error with instructions for configuring a seed file."""
        super().__init__(
            "No seed file to clean: set REVIEW_PAGE_SEED_FILE, or run "
            "scripts/clean_seed.py once so its result names the file"
        )


class EnrichmentServiceUnavailableError(ReviewPageError):
    """Indicate that EnrichmentService could not be reached to start a run."""

    def __init__(self, target_address: str) -> None:
        """Initialize the error with the unreachable EnrichmentService address.

        Args:
            target_address: ``host:port`` address of the EnrichmentService server.
        """
        super().__init__(
            f"Enrichment service is unavailable ({target_address}). "
            "Start enrichment-service and try again."
        )


class UnreadableReviewFileError(ReviewPageError):
    """Indicate that a line in the review file failed JSON or schema validation."""

    def __init__(self, review_file_path: Path, line_number: int) -> None:
        """Initialize the error with the corrupted file name and 1-based line number.

        Args:
            review_file_path: Path to the unreadable review JSONL file.
            line_number: The 1-based line number where validation failed.
        """
        super().__init__(
            f"Line {line_number} of {review_file_path.name} is not a valid review "
            "case. When the file comes from an older version, run "
            "scripts/clean_seed.py --no-fetch from the terminal, which rebuilds "
            "the file and keeps its saved answers. Otherwise fix or remove that "
            "line, then reload the page."
        )
