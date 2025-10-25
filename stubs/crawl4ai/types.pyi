from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from crawl4ai import CrawlResult

RunManyReturn = list["CrawlResult"]
