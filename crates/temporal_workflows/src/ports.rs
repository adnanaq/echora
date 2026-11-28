//! Service port traits decoupling Temporal activities from gRPC and Temporal client I/O.

use std::backtrace::Backtrace;
use std::fmt;
use std::sync::Arc;

use futures::future::BoxFuture;
use temporalio_sdk::ApplicationFailure;
use temporalio_sdk::activities::ActivityError;

use crate::refresh_anime_types::{
    ListEntityPagesOutput, MergeAnimeOutput, RecordRefreshInput, RefreshPageKey, RefreshPageOutput,
    StartEntityWorkflowsInput,
};

/// Result of one `LoadAnime` -> `MergeAnime` -> `SaveAnime` attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MergeAttemptOutcome {
    /// Merge and save succeeded (or no fields changed).
    Saved(MergeAnimeOutput),
    /// `SaveAnime` returned `ABORTED` due to an optimistic concurrency conflict.
    VersionConflict(String),
}

/// Internal classification driving Temporal retry behavior for [`PortError`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PortErrorKind {
    /// Permanent error (`INVALID_ARGUMENT`, `NOT_FOUND`, `FAILED_PRECONDITION`, etc.).
    NonRetryable,
    /// Transient error (`UNAVAILABLE`, `DEADLINE_EXCEEDED`, `INTERNAL`, transport).
    Retryable,
    /// Provider rejected requests with a block/ban; fail fast and record page failure.
    ProviderBlocked,
}

/// Canonical error returned by [`CatalogueDataPort`] or [`WorkflowStarterPort`].
///
/// Carries a retry classification, contextual message, and optional backtrace so
/// activities can convert directly into [`ActivityError`] via `From`.
pub struct PortError {
    kind: PortErrorKind,
    message: String,
    backtrace: Backtrace,
}

impl PortError {
    /// Creates a permanent error converted into `ApplicationFailure::non_retryable`.
    ///
    /// # Examples
    ///
    /// ```
    /// use temporal_workflows::PortError;
    ///
    /// let error = PortError::non_retryable("invalid anime UUID");
    /// assert!(error.is_non_retryable());
    /// assert!(!error.is_retryable());
    /// ```
    #[must_use]
    pub fn non_retryable(message: impl Into<String>) -> Self {
        Self {
            kind: PortErrorKind::NonRetryable,
            message: message.into(),
            backtrace: Backtrace::capture(),
        }
    }

    /// Creates a transient error converted into a retryable `ApplicationFailure`.
    ///
    /// # Examples
    ///
    /// ```
    /// use temporal_workflows::PortError;
    ///
    /// let error = PortError::retryable("upstream unavailable");
    /// assert!(error.is_retryable());
    /// assert!(!error.is_non_retryable());
    /// ```
    #[must_use]
    pub fn retryable(message: impl Into<String>) -> Self {
        Self {
            kind: PortErrorKind::Retryable,
            message: message.into(),
            backtrace: Backtrace::capture(),
        }
    }

    /// Creates a provider-block error signaling `refresh_page` to fail fast without retries.
    ///
    /// # Examples
    ///
    /// ```
    /// use temporal_workflows::PortError;
    ///
    /// let error = PortError::provider_blocked("anisearch returned HTTP 423");
    /// assert!(error.is_provider_blocked());
    /// ```
    #[must_use]
    pub fn provider_blocked(message: impl Into<String>) -> Self {
        Self {
            kind: PortErrorKind::ProviderBlocked,
            message: message.into(),
            backtrace: Backtrace::capture(),
        }
    }

    /// Returns `true` when Temporal must not retry this failure.
    #[must_use]
    pub const fn is_non_retryable(&self) -> bool {
        matches!(self.kind, PortErrorKind::NonRetryable)
    }

    /// Returns `true` when this failure is transient and safe for Temporal to retry.
    #[must_use]
    pub const fn is_retryable(&self) -> bool {
        matches!(self.kind, PortErrorKind::Retryable)
    }

    /// Returns `true` when the upstream provider blocked the request.
    #[must_use]
    pub const fn is_provider_blocked(&self) -> bool {
        matches!(self.kind, PortErrorKind::ProviderBlocked)
    }
    /// Converts this error into an [`ActivityError`] preserving its retry classification.
    #[must_use]
    pub fn into_activity_error(self) -> ActivityError {
        ActivityError::application(ApplicationFailure::from(self))
    }
}

impl fmt::Debug for PortError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PortError")
            .field("kind", &self.kind)
            .field("message", &self.message)
            .finish_non_exhaustive()
    }
}

impl fmt::Display for PortError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)?;
        if self.backtrace.status() == std::backtrace::BacktraceStatus::Captured {
            write!(f, "\n{}", self.backtrace)?;
        }
        Ok(())
    }
}

impl std::error::Error for PortError {}

impl From<PortError> for ApplicationFailure {
    fn from(error: PortError) -> Self {
        match error.kind {
            PortErrorKind::NonRetryable | PortErrorKind::ProviderBlocked => {
                Self::non_retryable(error)
            }
            PortErrorKind::Retryable => Self::new(error),
        }
    }
}

/// Port wrapping `PostgresService` and `EnrichmentService` gRPC calls used by activities.
pub trait CatalogueDataPort: Send + Sync {
    /// Lists an entity's linked provider pages and stored `merge_version`.
    fn list_entity_pages(
        &self,
        anime_id: String,
    ) -> BoxFuture<'_, Result<ListEntityPagesOutput, PortError>>;

    /// Fetches a page via `EnrichmentService.FetchPage` and stores it via `PostgresService.StorePage`.
    fn fetch_and_store_page(
        &self,
        page_key: RefreshPageKey,
        max_age_seconds: i64,
    ) -> BoxFuture<'_, Result<RefreshPageOutput, PortError>>;

    /// Records a failed page fetch via `PostgresService.StorePage` with empty responses.
    fn record_page_failure(&self, page_key: RefreshPageKey)
    -> BoxFuture<'_, Result<(), PortError>>;

    /// Runs one `LoadAnime` -> `MergeAnime` -> `SaveAnime` attempt.
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
    catalogue: Arc<dyn CatalogueDataPort>,
    starter: Arc<dyn WorkflowStarterPort>,
}

impl UpdateServicePorts {
    /// Creates a port bundle from concrete `catalogue` and `starter` implementations.
    #[must_use]
    pub fn new(
        catalogue: impl CatalogueDataPort + 'static,
        starter: impl WorkflowStarterPort + 'static,
    ) -> Self {
        Self {
            catalogue: Arc::new(catalogue),
            starter: Arc::new(starter),
        }
    }

    /// Returns the catalogue data port.
    #[must_use]
    pub fn catalogue(&self) -> &dyn CatalogueDataPort {
        self.catalogue.as_ref()
    }

    /// Returns the workflow starter port.
    #[must_use]
    pub fn starter(&self) -> &dyn WorkflowStarterPort {
        self.starter.as_ref()
    }

    /// Creates a placeholder port bundle for tests that only inspect worker registration.
    #[cfg(feature = "test-util")]
    #[must_use]
    pub fn noop() -> Self {
        Self::new(NoopPort, NoopPort)
    }
}

impl fmt::Debug for UpdateServicePorts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("UpdateServicePorts").finish_non_exhaustive()
    }
}

#[cfg(feature = "test-util")]
struct NoopPort;

#[cfg(feature = "test-util")]
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

    fn fetch_and_store_page(
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

    fn record_page_failure(
        &self,
        _page_key: RefreshPageKey,
    ) -> BoxFuture<'_, Result<(), PortError>> {
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

#[cfg(feature = "test-util")]
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
