import time
import uuid
from collections.abc import AsyncIterator, Callable
from unittest.mock import AsyncMock

import pytest
from hishel import AsyncBaseStorage, Entry, EntryMeta, Headers, Request, Response


class InMemoryStorage(AsyncBaseStorage):
    def __init__(self) -> None:
        self.entries: dict[str, list[Entry]] = {}

    async def create_entry(
        self,
        request: Request,
        response: Response,
        key: str,
        id_: uuid.UUID | None = None,
    ) -> Entry:
        stream = response.stream
        if isinstance(stream, AsyncIterator):
            body = b"".join([chunk async for chunk in stream])
        else:
            body = b"".join(stream)

        async def replay() -> AsyncIterator[bytes]:
            yield body

        entry = Entry(
            id=id_ or uuid.uuid4(),
            request=Request(
                method=request.method,
                url=request.url,
                headers=Headers(dict(request.headers)),
                stream=request.stream,
                metadata=request.metadata,
            ),
            meta=EntryMeta(created_at=time.time()),
            response=Response(
                status_code=response.status_code,
                headers=response.headers,
                stream=replay(),
            ),
            cache_key=key.encode(),
        )
        self.entries.setdefault(key, []).insert(0, entry)
        return entry

    async def get_entries(self, key: str) -> list[Entry]:
        return self.entries.get(key, [])

    async def update_entry(
        self, id: uuid.UUID, new_entry: Entry | Callable[[Entry], Entry]
    ) -> Entry | None:
        return None

    async def remove_entry(self, id: uuid.UUID) -> None:
        return None


@pytest.fixture
def storage() -> AsyncMock:
    return AsyncMock(wraps=InMemoryStorage())
