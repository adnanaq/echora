"""Base crawler implementing the template-method pattern.

All source-specific crawlers inherit from ``BaseCrawler`` and override the
abstract steps.  The concrete ``crawl()`` method orchestrates the fixed
lifecycle so subclasses never need to handle error logging, ``None``-guarding,
or repository persistence themselves.

Lifecycle (in order):
    1. ``normalize_identifier`` — convert slug / ID / path to a canonical URL.
    2. ``fetch_raw_data`` — perform network I/O via ``self.transport``.
    3. ``post_process_raw_data`` — optional hook to clean the raw dict.
    4. ``build_source_model`` — construct the source-specific Pydantic model.
    5. ``map_to_canonical`` — translate the source model to a canonical dict.
    6. ``repository.save`` — persist if a repository was provided.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from enrichment.sources.base.framework.interfaces import IRepository, ITransport
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Type variables for source and canonical models
T_Source = TypeVar("T_Source", bound=BaseModel)
T_Canonical = TypeVar("T_Canonical")


class BaseCrawler(Generic[T_Source, T_Canonical], ABC):  # noqa: UP046
    """Abstract base class for all source crawlers.

    Generic over two type parameters:

    - ``T_Source``: a Pydantic ``BaseModel`` that represents the raw scraped
      data in source-specific terms (e.g. ``MalAnime``).
    - ``T_Canonical``: the output type returned to callers — always
      ``dict[str, Any]`` in practice, but left generic for flexibility.

    Subclasses must implement ``normalize_identifier``, ``fetch_raw_data``,
    ``build_source_model``, and ``map_to_canonical``.  The optional
    ``post_process_raw_data`` hook can be overridden when the raw dict needs
    cleaning before the Pydantic model is constructed.

    Attributes:
        transport: Network transport used to fetch pages.
        repository: Persistence layer for saving canonical output.  When
            ``None``, the save step is skipped.
    """

    def __init__(self, transport: ITransport, repository: IRepository | None = None):
        """Initialise with a transport and an optional repository.

        Args:
            transport: Network transport (e.g. ``DockerTransport``).
            repository: Where to persist results after mapping.  Pass
                ``NullRepository()`` (or ``None``) to skip persistence.
        """
        self.transport = transport
        self.repository = repository

    async def crawl(self, identifier: str) -> T_Canonical | None:
        """Execute the full crawl lifecycle for one identifier.

        Orchestrates the five steps: normalise → fetch → post-process →
        build source model → map to canonical → save.  Any unhandled
        exception in any step is caught, logged at ``ERROR`` level, and
        causes ``None`` to be returned so callers never need to guard against
        exceptions from this method.

        Args:
            identifier: A slug, path, numeric ID, or full URL that uniquely
                identifies the resource to crawl.  The exact format is
                source-specific; ``normalize_identifier`` converts it to a
                canonical URL.

        Returns:
            The canonical output dict, or ``None`` if any step failed.
        """
        try:
            # 1. Normalize
            url = self.normalize_identifier(identifier)

            # 2. Fetch raw data
            raw_data = await self.fetch_raw_data(url)
            if not raw_data:
                return None

            # 3. Post-process raw data (if needed)
            processed_raw = await self.post_process_raw_data(raw_data, url)

            # 4. Build source-specific Pydantic model
            source_model = self.build_source_model(processed_raw, url)

            # 5. Map to canonical dictionary
            canonical = self.map_to_canonical(source_model)

            # 6. Persist if repository exists
            if self.repository:
                self.repository.save(canonical)

        except Exception:
            logger.exception(f"Failed to crawl {identifier}")
            return None
        else:
            return canonical

    @abstractmethod
    def get_extraction_schema(self) -> dict[str, Any]:
        """Return the primary XPath/CSS extraction schema for this crawler.

        Used by ``fetch_raw_data`` to configure the transport extraction
        strategy. Exposing it on the class makes schema inspection and
        cache-dependency hashing discoverable without knowledge of the
        concrete subclass.
        """

    @abstractmethod
    def normalize_identifier(self, identifier: str) -> str:
        """Convert a slug, path, or ID into a full canonical URL.

        Args:
            identifier: Raw identifier supplied by the caller.

        Returns:
            Fully-qualified URL suitable for passing to ``fetch_raw_data``.

        Raises:
            ValueError: If ``identifier`` cannot be resolved to a valid URL.
        """

    @abstractmethod
    async def fetch_raw_data(self, url: str) -> dict[str, Any] | None:
        """Perform actual network fetching via ``self.transport``.

        Implementations typically call ``self.transport.fetch_single`` or
        ``self.transport.fetch_batch`` and return the raw result dict.

        Args:
            url: Canonical URL produced by ``normalize_identifier``.

        Returns:
            Raw extraction dict from the transport layer, or ``None`` if the
            fetch produced no usable data.
        """

    async def post_process_raw_data(
        self, raw_data: dict[str, Any], url: str
    ) -> dict[str, Any]:
        """Optional async hook to clean or enrich the raw dict before model construction.

        The default implementation is a pass-through.  Override to strip
        unwanted keys, merge auxiliary data, fetch related pages, or normalise
        field names before ``build_source_model`` is called.

        Args:
            raw_data: Raw dict returned by ``fetch_raw_data``.
            url: Canonical URL (may be needed for context, e.g. extracting an
                ID from the URL path, or constructing sub-page URLs).

        Returns:
            Processed dict forwarded to ``build_source_model``.
        """
        return raw_data

    @abstractmethod
    def build_source_model(self, processed_raw: dict[str, Any], url: str) -> T_Source:
        """Construct the source-specific Pydantic model from the processed dict.

        Args:
            processed_raw: Dict returned by ``post_process_raw_data``.
            url: Canonical URL (useful for injecting the source URL into the
                model as an identifier field).

        Returns:
            Validated Pydantic model representing the scraped data.
        """

    @abstractmethod
    def map_to_canonical(self, source_model: T_Source) -> T_Canonical:
        """Translate the source model into the canonical data structure.

        This is where all source-specific normalisation happens (e.g.
        mapping ``"Currently Airing"`` → ``"ONGOING"``).  The output is what
        callers receive from ``crawl()`` and what is passed to ``repository.save``.

        Args:
            source_model: Validated source-specific Pydantic model.

        Returns:
            Canonical representation of the crawled data.
        """
