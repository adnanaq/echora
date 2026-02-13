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
    """Sparse vector represented by index/value pairs."""

    model_config = ConfigDict(extra="forbid")

    indices: list[int] = Field(min_length=1)
    values: list[float] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_sparse_shape(self) -> "SparseVectorData":
        """Validate sparse vector indices/values alignment.

        Returns:
            The validated sparse vector.

        Raises:
            ValueError: If sparse vector indices are invalid or not aligned.
        """
        if len(self.indices) != len(self.values):
            raise ValueError("indices and values must have the same length")
        if any(index < 0 for index in self.indices):
            raise ValueError("indices must be non-negative")
        return self


class SearchRequest(BaseModel):
    """Search request across dense (text/image) and sparse embeddings."""

    model_config = ConfigDict(extra="forbid")

    text_embedding: list[float] | None = None
    image_embedding: list[float] | None = None
    sparse_embedding: SparseVectorData | None = None
    entity_type: str | None = None
    limit: int = Field(default=10, ge=1, le=1000)
    filters: list[SearchFilterCondition] = Field(default_factory=list)
    fusion_method: FusionMethod = "rrf"

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

        return self


class SearchHit(BaseModel):
    """Normalized search hit returned by QdrantClient."""

    model_config = ConfigDict(extra="forbid")

    id: str
    score: float
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
    """Single payload update request."""

    model_config = ConfigDict(extra="forbid")

    point_id: str = Field(min_length=1)
    payload: dict[str, Any]
    key: str | None = None


class OperationErrorDetail(BaseModel):
    """Per-item failure detail for batch operations."""

    model_config = ConfigDict(extra="forbid")

    point_id: str
    message: str


class BatchOperationResult(BaseModel):
    """Normalized batch operation result."""

    model_config = ConfigDict(extra="forbid")

    total: int
    successful: int
    failed: int
    duplicates_removed: int = 0
    errors: list[OperationErrorDetail] = Field(default_factory=list)


class CollectionStats(BaseModel):
    """Normalized collection stats model."""

    model_config = ConfigDict(extra="forbid")

    collection_name: str
    total_documents: int
    vector_size: int
    distance_metric: str
    points_count: int | None = None
    indexed_vectors_count: int | None = None
