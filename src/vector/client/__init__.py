"""Database Client Infrastructure

Vector database operations and search coordination for anime content.
"""

from .qdrant_client import QdrantClient

__all__ = ["QdrantClient"]
