"""Provider-agnostic data types shared across vector DB interfaces."""

from dataclasses import dataclass, field
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


@dataclass
class SearchHit:
    """Provider-agnostic search result returned by a vector search operation.

    Attributes:
        id: Point identifier.
        score: Vector similarity score from the search index.
        reranking_score: Cross-encoder relevance score if reranking was applied.
        payload: Metadata fields stored alongside the vector.
    """

    id: str
    score: float
    reranking_score: float | None = None
    payload: dict[str, Any] = field(default_factory=dict)
