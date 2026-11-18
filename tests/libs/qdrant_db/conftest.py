import pytest
import pytest_asyncio
from qdrant_client import AsyncQdrantClient
from qdrant_db import QdrantClient


@pytest_asyncio.fixture(scope="session")
async def client(settings):
    async_qdrant_client = None
    try:
        if settings.qdrant.qdrant_api_key:
            async_qdrant_client = AsyncQdrantClient(
                url=settings.qdrant.qdrant_url,
                api_key=settings.qdrant.qdrant_api_key,
            )
        else:
            async_qdrant_client = AsyncQdrantClient(url=settings.qdrant.qdrant_url)

        client = await QdrantClient.create(
            config=settings.qdrant,
            async_qdrant_client=async_qdrant_client,
        )
    except Exception as e:
        pytest.skip(f"Failed to create test collection: {e}")

    yield client

    try:
        await client.delete_collection()
    except Exception:
        pass

    try:
        if async_qdrant_client:
            await async_qdrant_client.close()
    except Exception:
        pass
