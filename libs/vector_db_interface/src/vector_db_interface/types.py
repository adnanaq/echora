"""Provider-agnostic data types shared across vector DB interfaces."""

from dataclasses import dataclass
from typing import Any, TypedDict


class SparseVectorData(TypedDict):
    """Sparse vector payload represented as explicit index/value pairs.

    Attributes:
        indices: Dimension indices for non-zero values.
        values: Non-zero values aligned by position with ``indices``.
    """

    indices: list[int]
    values: list[float]


@dataclass
class VectorDocument:
    """Provider-agnostic representation of a document with vectors.

    Attributes:
        id: Unique identifier for the document
        vectors: Named vectors for multi-vector search. Supports single vectors
            (e.g., {"text": [0.1, 0.2, ...]}) or multivectors for hierarchical
            embeddings (e.g., {"episodes": [[0.1, ...], [0.2, ...]]}), and sparse
            vectors (e.g., {"text_sparse": {"indices": [1, 7], "values": [0.2, 1.1]}})
        payload: Metadata and searchable fields
    """

    id: str
    vectors: dict[str, list[float] | list[list[float]] | SparseVectorData]
    payload: dict[str, Any]
