from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from crawl4ai import CrawlResult

RunManyReturn = List["CrawlResult"]
