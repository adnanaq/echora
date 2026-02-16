"""Domain errors for strict-contract Qdrant operations."""


class QdrantOperationError(RuntimeError):
    """Base class for Qdrant domain errors."""


class ConfigurationError(QdrantOperationError):
    """Raised when client/configuration is invalid."""


class CollectionCompatibilityError(ConfigurationError):
    """Raised when existing collection schema is incompatible with config."""


class ValidationError(QdrantOperationError, ValueError):
    """Raised when request/update payloads are invalid."""


class DuplicateUpdateError(ValidationError):
    """Raised when duplicate updates are supplied with fail policy."""


class TransientQdrantError(QdrantOperationError):
    """Raised for retryable/transient backend failures."""


class PermanentQdrantError(QdrantOperationError):
    """Raised for non-retryable backend failures."""
