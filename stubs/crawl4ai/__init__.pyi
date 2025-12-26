from collections.abc import Iterator
from typing import Any, Generic, TypeVar

from .types import RunManyReturn

CrawlResultT = TypeVar("CrawlResultT", bound="CrawlResult")

class AsyncWebCrawler:
    async def __aenter__(self) -> AsyncWebCrawler: ...
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None: ...
    async def arun(self, url: str, config: CrawlerRunConfig) -> RunManyReturn: ...
    async def arun_many(
        self, urls: list[str], config: CrawlerRunConfig
    ) -> RunManyReturn: ...

class CrawlerRunConfig:
    def __init__(
        self,
        extraction_strategy: JsonCssExtractionStrategy | None = None,
        wait_until: str | None = None,
        wait_for_images: bool = False,
        scan_full_page: bool = False,
        adjust_viewport_to_content: bool = False,
        delay_before_return_html: float = 0.0,
        session_id: str | None = None,
        js_code: str | None = None,
        wait_for: str | None = None,
        js_only: bool = False,
        page_timeout: int = 60000,
    ) -> None:
        """
        Initialize a crawler run configuration with extraction options and runtime behaviors.

        Parameters:
            extraction_strategy: Optional extraction schema specifying fields to extract from the page.
            wait_until: Optional name of a page lifecycle event to wait for before extracting (e.g., "load", "networkidle").
            wait_for_images: Whether to wait for images to finish loading before returning results.
            scan_full_page: Whether to scan and extract content from the full page (including content loaded outside the initial viewport).
            adjust_viewport_to_content: Whether to adjust the browser viewport to fit page content before extraction.
            delay_before_return_html: Delay in seconds to wait after extraction before returning the HTML.
            session_id: Optional identifier to reuse or group browser sessions across runs.
            js_code: Optional JavaScript code to execute in the page context prior to extraction.
            wait_for: Optional CSS selector or expression to wait for before extracting.
            js_only: If true, perform extraction using JavaScript execution results only (skip DOM-based extraction).
            page_timeout: Maximum time in milliseconds to wait for the page load or specified waits before timing out.
        """
        ...

class CrawlResult:
    success: bool
    extracted_content: str | None
    error_message: str | None
    url: str
    html: str | None

class CrawlResultContainer(Generic[CrawlResultT]):
    def __init__(self, results: CrawlResultT | list[CrawlResultT]) -> None: ...
    def __iter__(self) -> Iterator[CrawlResultT]: ...

class JsonCssExtractionStrategy:
    def __init__(self, schema: dict[str, Any]) -> None: ...
