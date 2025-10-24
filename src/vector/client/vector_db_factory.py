from ...config import Settings
from ..processors.embedding_manager import MultiVectorEmbeddingManager
from .milvus_client import MilvusClient
from .qdrant_client import QdrantClient
from .marqo_client import MarqoClient
from .vector_db_client import VectorDBClient


class VectorDBClientFactory:
    """Factory for creating VectorDBClient instances based on configuration."""

    @staticmethod
    def get_client(
        embedding_manager: MultiVectorEmbeddingManager,
        settings: Settings,
        url: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> VectorDBClient:
        """Returns an initialized VectorDBClient instance.

        Args:
            embedding_manager: The MultiVectorEmbeddingManager instance.
            settings: The application settings.
            url: Optional URL for the vector database. Defaults to settings if None.
            collection_name: Optional collection name. Defaults to settings if None.

        Returns:
            An instance of a class implementing VectorDBClient.

        Raises:
            ValueError: If an unsupported vector database provider is specified.
        """
        provider = settings.vector_db_provider.lower()

        if provider == "qdrant":
            return QdrantClient(embedding_manager=embedding_manager, url=url, collection_name=collection_name, settings=settings)
        elif provider == "milvus":
            return MilvusClient(embedding_manager=embedding_manager, url=url, collection_name=collection_name, settings=settings)
        elif provider == "marqo":
            return MarqoClient(embedding_manager=embedding_manager, url=url, collection_name=collection_name, settings=settings)
        # Add other vector database clients here as they are implemented
        # elif provider == "pinecone":
        #     return PineconeClient(embedding_manager=embedding_manager, url=url, collection_name=collection_name, settings=settings)
        # elif provider == "weaviate":
        #     return WeaviateClient(embedding_manager=embedding_manager, url=url, collection_name=collection_name, settings=settings)
        else:
            raise ValueError(f"Unsupported vector database provider: {provider}")
