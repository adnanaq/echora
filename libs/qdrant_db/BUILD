"""
Qdrant DB Client - Qdrant implementation of VectorDBClient interface.

Provides production-ready Qdrant vector database operations with:
- Multi-vector support
- Batch operations with retry logic
- Async/await patterns
- Type-safe operations
"""

python_distribution(
    name="qdrant_db_dist",
    dependencies=[
        "//libs/qdrant_db/src/qdrant_db",
    ],
    provides=python_artifact(
        name="echora-qdrant-db",
        version="0.1.0",
        description="Qdrant vector database client implementing VectorDBClient interface",
    ),
    generate_setup=True,
)
