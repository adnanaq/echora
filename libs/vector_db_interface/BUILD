"""
Vector DB Interface - Provider-agnostic interface for vector databases.

Defines VectorDocument and VectorDBClient interface that all
vector database implementations must follow.
"""

python_distribution(
    name="vector_db_interface_dist",
    dependencies=[
        "//libs/vector_db_interface/src/vector_db_interface",
    ],
    provides=python_artifact(
        name="echora-vector-db-interface",
        version="0.1.0",
        description="Provider-agnostic interface for vector database operations",
    ),
    generate_setup=True,
)
