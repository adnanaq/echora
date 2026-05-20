"""Typed contracts for strict Qdrant client request/response flows."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

DedupPolicy = Literal["last-wins", "fail"]
PayloadUpdateMode = Literal["merge", "overwrite"]
FusionMethod = Literal["rrf", "dbsf"]
FilterOperator = Literal["eq", "in", "range"]


def _is_scalar_filter_value(value: Any) -> bool:
    """Return whether value is an allowed scalar for filtering.

    Args:
        value: Candidate filter value.

    Returns:
        ``True`` when ``value`` is one of ``str``, ``int``, ``float``, or
        ``bool``.
    """
    return isinstance(value, str | int | float | bool)


class SearchRange(BaseModel):
    """Inclusive/exclusive numeric range condition."""

    model_config = ConfigDict(extra="forbid")

    gt: float | int | None = None
    gte: float | int | None = None
    lt: float | int | None = None
    lte: float | int | None = None

    @model_validator(mode="after")
    def validate_has_bound(self) -> "SearchRange":
        """Ensure at least one range boundary is set.

        Returns:
            The validated model instance.

        Raises:
            ValueError: If no range boundary is provided.
        """
        if all(value is None for value in (self.gt, self.gte, self.lt, self.lte)):
            raise ValueError("Range filter requires at least one bound")
        return self


class SearchFilterCondition(BaseModel):
    """Explicit filter condition for Qdrant payload filtering."""

    model_config = ConfigDict(extra="forbid")

    field: str = Field(min_length=1)
    operator: FilterOperator
    value: Any

    @model_validator(mode="after")
    def validate_value(self) -> "SearchFilterCondition":
        """Validate operator/value compatibility.

        Returns:
            The validated model instance.

        Raises:
            ValueError: If filter values do not match operator requirements.
        """
        if self.operator == "eq":
            if not _is_scalar_filter_value(self.value):
                raise ValueError("eq filter value must be scalar")
            return self

        if self.operator == "in":
            if not isinstance(self.value, list) or len(self.value) == 0:
                raise ValueError("in filter value must be a non-empty list")
            if not all(_is_scalar_filter_value(item) for item in self.value):
                raise ValueError("in filter values must be scalar")
            return self

        if self.operator == "range":
            if isinstance(self.value, SearchRange):
                return self
            self.value = SearchRange.model_validate(self.value)
            return self

        raise ValueError(f"Unsupported filter operator: {self.operator}")


class SparseVectorData(BaseModel):
    """Sparse vector represented by index/value pairs for keyword-based search.

    Sparse vectors are used for keyword/term-based search in Qdrant, typically
    generated from tokenizers or BM25 algorithms. Unlike dense vectors, sparse
    vectors only store non-zero dimensions using index/value pairs, making them
    memory-efficient for high-dimensional vocabulary spaces.

    This model enforces strict validation to ensure compatibility with Qdrant's
    sparse vector requirements, catching errors that the qdrant-client library
    does not validate.

    Attributes:
        indices: List of dimension indices where values are non-zero. Must be
            unique, non-negative, and within u32 bounds (0 to 4,294,967,295).
            Order is not enforced - Qdrant sorts internally.
        values: List of float values corresponding to each index. Must have
            same length as indices.

    Raises:
        ValueError: If validation constraints are violated (see validate_sparse_shape).

    Examples:
        Create a sparse vector for keyword search:
            >>> sparse = SparseVectorData(
            ...     indices=[42, 108, 555],
            ...     values=[0.8, 0.5, 0.3]
            ... )

        Use in search request:
            >>> from qdrant_db.contracts import SearchRequest
            >>> request = SearchRequest(
            ...     sparse_embedding=SparseVectorData(
            ...         indices=[10, 20, 30],
            ...         values=[0.9, 0.6, 0.4]
            ...     ),
            ...     limit=10
            ... )

        Convert from dict format:
            >>> data = {"indices": [1, 5, 10], "values": [0.5, 0.3, 0.2]}
            >>> sparse = SparseVectorData(**data)

    Note:
        The qdrant-client library accepts invalid sparse vectors (duplicates,
        negative indices, values exceeding u32) without validation. This model
        prevents such data from reaching the Qdrant server and causing runtime
        errors.

    See Also:
        - SearchRequest: For using sparse vectors in search queries
        - BatchVectorUpdateItem: For batch vector updates including sparse vectors
    """

    model_config = ConfigDict(extra="forbid")

    indices: list[int] = Field(min_length=1)
    values: list[float] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_sparse_shape(self) -> "SparseVectorData":
        """Validate sparse vector indices/values alignment and constraints.

        This validator ensures that sparse vector data conforms to Qdrant's
        requirements:
        - Indices and values arrays must have matching lengths
        - All indices must be non-negative (>= 0)
        - All indices must be unique (no duplicates)
        - All indices must fit within u32 bounds (0 to 4,294,967,295)

        Note:
            Indices do not need to be sorted - Qdrant handles sorting internally.
            The qdrant-client library does not validate these constraints, so
            validation here prevents runtime errors when data reaches the server.

        Returns:
            SparseVectorData: The validated sparse vector instance.

        Raises:
            ValueError: If indices and values lengths don't match.
            ValueError: If any index is negative.
            ValueError: If indices contain duplicates.
            ValueError: If any index exceeds the u32 maximum (4,294,967,295).

        Examples:
            Valid sparse vector:
                >>> SparseVectorData(indices=[1, 5, 10], values=[0.5, 0.3, 0.2])

            Invalid - duplicate indices:
                >>> SparseVectorData(indices=[1, 1, 2], values=[0.5, 0.3, 0.2])
                ValueError: indices must be unique (no duplicates)

            Invalid - exceeds u32 max:
                >>> SparseVectorData(indices=[4294967296], values=[0.5])
                ValueError: indices must not exceed u32 maximum (4,294,967,295)
        """
        if len(self.indices) != len(self.values):
            raise ValueError("indices and values must have the same length")
        if any(index < 0 for index in self.indices):
            raise ValueError("indices must be non-negative")
        if len(self.indices) != len(set(self.indices)):
            raise ValueError("indices must be unique (no duplicates)")
        if any(index > 4294967295 for index in self.indices):
            raise ValueError("indices must not exceed u32 maximum (4,294,967,295)")
        return self


class SearchRequest(BaseModel):
    """Search request across dense (text/image) and sparse embeddings."""

    model_config = ConfigDict(extra="forbid")

    text_embedding: list[float] | None = None
    image_embedding: list[float] | None = None
    sparse_embedding: SparseVectorData | None = None
    expanded_text_embeddings: list[list[float]] | None = Field(
        default=None,
        description=(
            "Additional text embedding variants for query expansion. "
            "Each variant is fused alongside the primary text_embedding via RRF. "
            "Intended for LLM-generated query rephrasing — requires text_embedding to be set."
        ),
    )
    entity_type: str | None = None
    limit: int = Field(default=10, ge=1, le=1000)
    score_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum similarity score for returned hits. Results below this value "
            "are dropped server-side. None means no threshold — Qdrant always returns "
            "up to `limit` results. Applied to both single-vector and fusion searches."
        ),
    )
    filters: list[SearchFilterCondition] = Field(default_factory=list)
    fusion_method: FusionMethod = "rrf"

    # Reranking support
    query_text: str | None = Field(
        default=None,
        description="Original query text (required for reranking if enabled in config)",
    )

    @model_validator(mode="after")
    def validate_embeddings(self) -> "SearchRequest":
        """Validate embedding requirements and element types.

        Returns:
            The validated request model.

        Raises:
            ValueError: If embeddings are missing, empty, or non-numeric.
        """
        if (
            self.text_embedding is None
            and self.image_embedding is None
            and self.sparse_embedding is None
        ):
            raise ValueError(
                "At least one of text_embedding, image_embedding, or sparse_embedding is required"
            )

        for name, vector in (
            ("text_embedding", self.text_embedding),
            ("image_embedding", self.image_embedding),
        ):
            if vector is not None and len(vector) == 0:
                raise ValueError(f"{name} must not be empty")

        if self.expanded_text_embeddings is not None:
            if self.text_embedding is None:
                raise ValueError(
                    "expanded_text_embeddings requires text_embedding to be set"
                )
            for i, expansion in enumerate(self.expanded_text_embeddings):
                if len(expansion) == 0:
                    raise ValueError(f"expanded_text_embeddings[{i}] must not be empty")

        return self


class SearchHit(BaseModel):
    """Normalized search hit returned by QdrantClient."""

    model_config = ConfigDict(extra="forbid")

    id: str
    score: float  # Vector similarity score
    reranking_score: float | None = (
        None  # Cross-encoder relevance score (if reranking applied)
    )
    payload: dict[str, Any] = Field(default_factory=dict)


class BatchVectorUpdateItem(BaseModel):
    """Single vector update request."""

    model_config = ConfigDict(extra="forbid")

    point_id: str = Field(min_length=1)
    vector_name: str = Field(min_length=1)
    vector_data: list[float] | list[list[float]] | SparseVectorData

    @model_validator(mode="after")
    def validate_vector_data(self) -> "BatchVectorUpdateItem":
        """Validate vector payload shape.

        Returns:
            The validated update item.

        Raises:
            ValueError: If vector payload is empty or not list-like.
        """
        if isinstance(self.vector_data, SparseVectorData):
            return self
        if not isinstance(self.vector_data, list) or len(self.vector_data) == 0:
            raise ValueError("vector_data must be a non-empty list")
        return self


class BatchPayloadUpdateItem(BaseModel):
    """Single payload update request for a batch payload operation.

    Attributes:
        point_id: ID of the point whose payload should be updated.
        payload: Payload fields to merge into or overwrite on the point.
        key: Optional nested key path for partial payload updates.
    """

    model_config = ConfigDict(extra="forbid")

    point_id: str = Field(min_length=1)
    payload: dict[str, Any]
    key: str | None = None


class OperationErrorDetail(BaseModel):
    """Per-item failure detail for a failed batch operation entry.

    Attributes:
        point_id: ID of the point that failed.
        message: Human-readable failure reason.
    """

    model_config = ConfigDict(extra="forbid")

    point_id: str
    message: str


class BatchOperationResult(BaseModel):
    """Normalized result summary for a batch write operation.

    Attributes:
        total: Total number of items processed.
        successful: Count of items that succeeded.
        failed: Count of items that failed.
        duplicates_removed: Count of duplicate items removed before processing.
        errors: Per-item failure details when individual items failed.
    """

    model_config = ConfigDict(extra="forbid")

    total: int
    successful: int
    failed: int
    duplicates_removed: int = 0
    errors: list[OperationErrorDetail] = Field(default_factory=list)


class CollectionStats(BaseModel):
    """Normalized collection statistics returned by get_stats().

    Attributes:
        collection_name: Name of the Qdrant collection.
        total_documents: Exact point count from a count query.
        vector_size: Primary text vector dimension.
        distance_metric: Configured distance metric (e.g. ``cosine``).
        points_count: Point count reported by collection info (may be approximate).
        indexed_vectors_count: Count of indexed vectors from collection info.
    """

    model_config = ConfigDict(extra="forbid")

    collection_name: str
    total_documents: int
    vector_size: int
    distance_metric: str
    points_count: int | None = None
    indexed_vectors_count: int | None = None
