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
    ) -> None: ...

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
