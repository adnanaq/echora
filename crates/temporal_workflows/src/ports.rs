//! Trait ports decoupling Temporal activities from gRPC and Temporal client transport.

use std::fmt;
use std::sync::Arc;

use futures::future::BoxFuture;

use crate::refresh_anime_types::{
    ListEntityPagesOutput, MergeAnimeOutput, RecordRefreshInput, RefreshPageKey, RefreshPageOutput,
    StartEntityWorkflowsInput,
};

/// Result of a single `LoadAnime` -> `MergeAnime` -> `SaveAnime` attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MergeAttemptOutcome {
    /// Merge and save succeeded (or no changes were needed).
    Saved(MergeAnimeOutput),
    /// `SaveAnime` returned `ABORTED` due to an optimistic concurrency conflict.
    VersionConflict(String),
}

/// Error classification used by activities to choose Temporal retry behavior.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PortErrorKind {
    /// Permanent error (`INVALID_ARGUMENT`, `NOT_FOUND`, `FAILED_PRECONDITION`, etc.).
    NonRetryable,
    /// Transient error (`UNAVAILABLE`, `DEADLINE_EXCEEDED`, `INTERNAL`, transport).
    Retryable,
}

/// Error returned by [`CatalogueDataPort`] or [`WorkflowStarterPort`] implementations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PortError {
    /// Retry classification.
    pub kind: PortErrorKind,
    /// Human-readable error message.
    pub message: String,
}

impl PortError {
    /// Creates a non-retryable port error (mapped to `ApplicationFailure::non_retryable`).
    ///
    /// # Examples
    ///
    /// ```
    /// use temporal_workflows::{PortError, PortErrorKind};
    ///
    /// let err = PortError::non_retryable("invalid anime UUID");
    /// assert_eq!(err.kind, PortErrorKind::NonRetryable);
    /// ```
    #[must_use]
    pub fn non_retryable(message: impl Into<String>) -> Self {
        Self {
            kind: PortErrorKind::NonRetryable,
            message: message.into(),
        }
    }

    /// Creates a retryable port error.
    ///
    /// # Examples
    ///
    /// ```
    /// use temporal_workflows::{PortError, PortErrorKind};
    ///
    /// let err = PortError::retryable("upstream unavailable");
    /// assert_eq!(err.kind, PortErrorKind::Retryable);
    /// ```
    #[must_use]
    pub fn retryable(message: impl Into<String>) -> Self {
        Self {
            kind: PortErrorKind::Retryable,
            message: message.into(),
        }
    }
}

impl fmt::Display for PortError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for PortError {}

/// Returns `true` when a gRPC status code is permanent and must not be retried.
///
/// Permanent codes: `InvalidArgument` (3), `NotFound` (5), `PermissionDenied` (7),
/// `ResourceExhausted` (8), `FailedPrecondition` (9), `Unimplemented` (12).
///
/// # Examples
///
/// ```
/// use temporal_workflows::is_non_retryable_grpc_code;
///
/// assert!(is_non_retryable_grpc_code(3));
/// assert!(is_non_retryable_grpc_code(5));
/// assert!(is_non_retryable_grpc_code(7));
/// assert!(is_non_retryable_grpc_code(8));
/// assert!(is_non_retryable_grpc_code(9));
/// assert!(is_non_retryable_grpc_code(12));
/// assert!(!is_non_retryable_grpc_code(14));
/// ```
#[must_use]
pub const fn is_non_retryable_grpc_code(tonic_code_i32: i32) -> bool {
    matches!(tonic_code_i32, 3 | 5 | 7 | 8 | 9 | 12)
}

/// Port wrapping `PostgresService` and `EnrichmentService` gRPC calls used by activities.
pub trait CatalogueDataPort: Send + Sync {
    /// Lists an entity's linked provider pages via `PostgresService.ListEntityPages`.
    fn list_entity_pages(
        &self,
        anime_id: String,
    ) -> BoxFuture<'_, Result<ListEntityPagesOutput, PortError>>;

    /// Fetches a page via `EnrichmentService.FetchPage` and stores it via `PostgresService.StorePage`.
    fn refresh_page(
        &self,
        page_key: RefreshPageKey,
        max_age_seconds: i64,
    ) -> BoxFuture<'_, Result<RefreshPageOutput, PortError>>;

    /// Executes one `LoadAnime` -> `MergeAnime` -> `SaveAnime` attempt.
    fn merge_anime_attempt(
        &self,
        anime_id: String,
        merge_version: i32,
    ) -> BoxFuture<'_, Result<MergeAttemptOutcome, PortError>>;

    /// Records the final refresh outcome via `PostgresService.RecordRefresh`.
    fn record_refresh(&self, input: RecordRefreshInput) -> BoxFuture<'_, Result<(), PortError>>;
}

/// Port wrapping top-level Temporal workflow starts with `USE_EXISTING` conflict policy.
pub trait WorkflowStarterPort: Send + Sync {
    /// Starts `RefreshEpisodeWorkflow` and/or `WriteSynopsisWorkflow` as top-level workflows.
    fn start_entity_workflows(
        &self,
        input: StartEntityWorkflowsInput,
    ) -> BoxFuture<'_, Result<usize, PortError>>;
}

/// Bundle of external service and Temporal starter ports injected into worker activities.
#[derive(Clone)]
pub struct UpdateServicePorts {
    /// Data/enrichment gRPC port.
    pub catalogue: Arc<dyn CatalogueDataPort>,
    /// Temporal top-level workflow starter port.
    pub starter: Arc<dyn WorkflowStarterPort>,
}

impl UpdateServicePorts {
    /// Creates a port bundle from concrete `catalogue` and `starter` implementations.
    ///
    /// # Examples
    ///
    /// ```
    /// use temporal_workflows::UpdateServicePorts;
    ///
    /// let ports = UpdateServicePorts::noop();
    /// assert!(std::sync::Arc::strong_count(&ports.catalogue) >= 1);
    /// ```
    #[must_use]
    pub fn new(
        catalogue: Arc<dyn CatalogueDataPort>,
        starter: Arc<dyn WorkflowStarterPort>,
    ) -> Self {
        Self { catalogue, starter }
    }

    /// Creates a placeholder port bundle for unit tests that only inspect worker registration.
    #[must_use]
    pub fn noop() -> Self {
        Self {
            catalogue: Arc::new(NoopPort),
            starter: Arc::new(NoopPort),
        }
    }
}

struct NoopPort;

impl CatalogueDataPort for NoopPort {
    fn list_entity_pages(
        &self,
        _anime_id: String,
    ) -> BoxFuture<'_, Result<ListEntityPagesOutput, PortError>> {
        Box::pin(async {
            Err(PortError::non_retryable(
                "CatalogueDataPort is not configured",
            ))
        })
    }

    fn refresh_page(
        &self,
        _page_key: RefreshPageKey,
        _max_age_seconds: i64,
    ) -> BoxFuture<'_, Result<RefreshPageOutput, PortError>> {
        Box::pin(async {
            Err(PortError::non_retryable(
                "CatalogueDataPort is not configured",
            ))
        })
    }

    fn merge_anime_attempt(
        &self,
        _anime_id: String,
        _merge_version: i32,
    ) -> BoxFuture<'_, Result<MergeAttemptOutcome, PortError>> {
        Box::pin(async {
            Err(PortError::non_retryable(
                "CatalogueDataPort is not configured",
            ))
        })
    }

    fn record_refresh(&self, _input: RecordRefreshInput) -> BoxFuture<'_, Result<(), PortError>> {
        Box::pin(async {
            Err(PortError::non_retryable(
                "CatalogueDataPort is not configured",
            ))
        })
    }
}

impl WorkflowStarterPort for NoopPort {
    fn start_entity_workflows(
        &self,
        _input: StartEntityWorkflowsInput,
    ) -> BoxFuture<'_, Result<usize, PortError>> {
        Box::pin(async {
            Err(PortError::non_retryable(
                "WorkflowStarterPort is not configured",
            ))
        })
    }
}
