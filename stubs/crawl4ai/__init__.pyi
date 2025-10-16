from typing import Any, List, Dict, Optional, Generic, TypeVar, Union, Iterator
from .types import RunManyReturn

CrawlResultT = TypeVar("CrawlResultT", bound="CrawlResult")

class AsyncWebCrawler:
    async def __aenter__(self) -> "AsyncWebCrawler": ...
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None: ...
    async def arun(self, url: str, config: "CrawlerRunConfig") -> "RunManyReturn": ...
    async def arun_many(self, urls: List[str], config: "CrawlerRunConfig") -> "RunManyReturn": ...

class CrawlerRunConfig:
    def __init__(
        self,
        extraction_strategy: Optional["JsonCssExtractionStrategy"] = None,
        wait_until: Optional[str] = None,
        wait_for_images: bool = False,
        scan_full_page: bool = False,
        adjust_viewport_to_content: bool = False,
        delay_before_return_html: float = 0.0,
        session_id: Optional[str] = None,
        js_code: Optional[str] = None,
        wait_for: Optional[str] = None,
        js_only: bool = False,
    ) -> None: ...

class CrawlResult:
    success: bool
    extracted_content: Optional[str]
    error_message: Optional[str]
    url: str
    html: Optional[str]

class CrawlResultContainer(Generic[CrawlResultT]):
    def __init__(self, results: Union[CrawlResultT, List[CrawlResultT]]) -> None: ...
    def __iter__(self) -> Iterator[CrawlResultT]: ...

class JsonCssExtractionStrategy:
    def __init__(self, schema: Dict[str, Any]) -> None: ...
