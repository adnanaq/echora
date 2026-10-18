from typing import Any

import pytest
from enrichment.sources.base.exceptions import ServiceNetworkError
from enrichment.sources.base.framework.crawler import BaseCrawler
from enrichment.sources.base.framework.interfaces import IRepository
from pydantic import BaseModel, ValidationError


class TitledPage(BaseModel):
    title: str


class SavedRecords(IRepository):
    def __init__(self) -> None:
        self.records: list[Any] = []

    def save(self, data: Any) -> None:
        self.records.append(data)


class ScriptedCrawler(BaseCrawler[TitledPage, dict[str, Any]]):
    def __init__(
        self,
        repository: IRepository,
        raw_data: dict[str, Any] | None = None,
        fetch_error: Exception | None = None,
    ) -> None:
        super().__init__(repository)
        self._raw_data = raw_data
        self._fetch_error = fetch_error

    def get_extraction_schema(self) -> dict[str, Any]:
        return {}

    def normalize_identifier(self, identifier: str) -> str:
        return f"https://example.com/anime/{identifier}"

    async def fetch_raw_data(self, url: str) -> dict[str, Any] | None:
        if self._fetch_error is not None:
            raise self._fetch_error
        return self._raw_data

    def build_source_model(self, processed_raw: dict[str, Any], url: str) -> TitledPage:
        return TitledPage.model_validate(processed_raw)

    def map_to_canonical(self, source_model: TitledPage) -> dict[str, Any]:
        return {"title": source_model.title}


async def test_crawl_page_found_returns_canonical_and_saves_it() -> None:
    repository = SavedRecords()

    canonical = await ScriptedCrawler(repository, raw_data={"title": "Bebop"}).crawl(
        "1"
    )

    assert (canonical, repository.records) == ({"title": "Bebop"}, [canonical])


async def test_crawl_missing_page_returns_none_and_saves_nothing() -> None:
    repository = SavedRecords()

    assert await ScriptedCrawler(repository).crawl("1") is None
    assert repository.records == []


async def test_crawl_fetch_failure_raises() -> None:
    crawler = ScriptedCrawler(
        SavedRecords(),
        fetch_error=ServiceNetworkError(service="example", cause="timed out"),
    )

    with pytest.raises(ServiceNetworkError):
        await crawler.crawl("1")


async def test_crawl_page_model_rejecting_data_raises() -> None:
    crawler = ScriptedCrawler(SavedRecords(), raw_data={"name": "Bebop"})

    with pytest.raises(ValidationError):
        await crawler.crawl("1")
