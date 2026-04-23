"""Public API for the crawler framework.

Re-exports the core abstractions so callers only need to import from
``enrichment.crawlers.framework`` without knowing the internal module layout.
"""

from enrichment.sources.base.framework.crawler import BaseCrawler
from enrichment.sources.base.framework.interfaces import IRepository, ITransport
from enrichment.sources.base.framework.repository import FileRepository, NullRepository
from enrichment.sources.base.framework.transport import DockerTransport

__all__ = [
    "BaseCrawler",
    "DockerTransport",
    "FileRepository",
    "IRepository",
    "ITransport",
    "NullRepository",
]
