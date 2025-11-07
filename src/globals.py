"""
Global instances shared across the application.

This module provides a single source of truth for globally shared instances
like the Qdrant client and query parser agent. These are initialized in main.py
during startup and accessed by service implementations.
"""
from typing import Optional
from src.vector.client.qdrant_client import QdrantClient
from src.poc.atomic_agents_poc import AnimeQueryAgent

# Global instances - initialized by main.py during startup
qdrant_client: Optional[QdrantClient] = None
query_parser_agent: Optional[AnimeQueryAgent] = None
