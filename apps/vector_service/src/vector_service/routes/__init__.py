"""Export vector_service route adapters.

This module exposes adapter classes that `main.py` registers with the gRPC
server for vector admin and search APIs.
"""

from .adapter import VectorAdminRoutes, VectorSearchRoutes

__all__ = ["VectorAdminRoutes", "VectorSearchRoutes"]
