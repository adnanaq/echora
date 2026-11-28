//! Service port traits decoupling Temporal activities from gRPC and Temporal client I/O.

use std::backtrace::Backtrace;
use std::error::Error as StdError;
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use futures::future::BoxFuture;
use temporalio_sdk::ApplicationFailure;
use temporalio_sdk::activities::ActivityError;
use uuid::Uuid;

use crate::refresh_anime_types::{
    ListEntityPagesOutput, MergeAnimeOutput, RefreshOutcome, RefreshPageKey, RefreshPageOutput,
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
/// Carries a retry classification, contextual message, optional upstream cause, and
/// captured backtrace so activities convert directly into [`ActivityError`].
pub struct PortError {
    kind: PortErrorKind,
    message: String,
    source: Option<Box<dyn StdError + Send + Sync>>,
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
        Self::new(PortErrorKind::NonRetryable, message)
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
        Self::new(PortErrorKind::Retryable, message)
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
        Self::new(PortErrorKind::ProviderBlocked, message)
    }

    fn new(kind: PortErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            source: None,
            backtrace: Backtrace::capture(),
        }
    }

    /// Attaches an upstream error as [`std::error::Error::source`].
    ///
    /// # Examples
    ///
    /// ```
    /// use std::error::Error as _;
    /// use std::io;
    ///
    /// use temporal_workflows::PortError;
    ///
    /// let io_err = io::Error::other("connection reset");
    /// let error = PortError::retryable("rpc failed").with_source(io_err);
    /// assert!(error.source().is_some());
    /// ```
    #[must_use]
    pub fn with_source(mut self, source: impl StdError + Send + Sync + 'static) -> Self {
        self.source = Some(Box::new(source));
        self
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
            .field("has_source", &self.source.is_some())
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

impl StdError for PortError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        self.source
            .as_deref()
            .map(|err| err as &(dyn StdError + 'static))
    }
}

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
    ///
    /// # Errors
    ///
    /// Returns a non-retryable [`PortError`] if the entity does not exist or its stored
    /// page metadata cannot be decoded, or a retryable [`PortError`] on transient RPC errors.
    fn list_entity_pages(
        &self,
        anime_id: Uuid,
    ) -> BoxFuture<'_, Result<ListEntityPagesOutput, PortError>>;

    /// Fetches a page via `EnrichmentService.FetchPage` and stores it via `PostgresService.StorePage`.
    ///
    /// # Errors
    ///
    /// Returns [`PortError::provider_blocked`] when the provider is blocked, a non-retryable
    /// [`PortError`] on permanent gRPC statuses or malformed responses, or a retryable
    /// [`PortError`] on transient upstream failures.
    fn fetch_and_store_page(
        &self,
        page_key: RefreshPageKey,
        max_age: Duration,
    ) -> BoxFuture<'_, Result<RefreshPageOutput, PortError>>;

    /// Records a failed page fetch via `PostgresService.StorePage` with empty responses.
    ///
    /// # Errors
    ///
    /// Returns a non-retryable [`PortError`] on invalid arguments or missing targets, or a
    /// retryable [`PortError`] on transient `PostgresService` failures.
    fn record_page_failure(
        &self,
        page_key: RefreshPageKey,
        failed_at: SystemTime,
    ) -> BoxFuture<'_, Result<(), PortError>>;

    /// Runs one `LoadAnime` -> `MergeAnime` -> `SaveAnime` attempt.
    ///
    /// # Errors
    ///
    /// Returns a non-retryable [`PortError`] on permanent validation or missing-entity errors,
    /// or a retryable [`PortError`] on transient `PostgresService` or `EnrichmentService` errors.
    fn merge_anime_attempt(
        &self,
        anime_id: Uuid,
        merge_version: i32,
    ) -> BoxFuture<'_, Result<MergeAttemptOutcome, PortError>>;

    /// Records the final refresh outcome via `PostgresService.RecordRefresh`.
    ///
    /// # Errors
    ///
    /// Returns a non-retryable [`PortError`] when the entity is missing or arguments are
    /// invalid, or a retryable [`PortError`] on transient `PostgresService` failures.
    fn record_refresh(
        &self,
        anime_id: Uuid,
        outcome: RefreshOutcome,
        refresh_after_days: i64,
    ) -> BoxFuture<'_, Result<(), PortError>>;
}

/// Port wrapping top-level Temporal workflow starts with `USE_EXISTING` conflict policy.
pub trait WorkflowStarterPort: Send + Sync {
    /// Starts `RefreshEpisodeWorkflow` and/or `WriteSynopsisWorkflow` as top-level workflows.
    ///
    /// # Errors
    ///
    /// Returns a non-retryable [`PortError`] if workflow input serialization fails, or a
    /// retryable [`PortError`] if Temporal rejects or fails to accept any start request.
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
fn not_configured<T: Send + 'static>(
    port_name: &'static str,
) -> BoxFuture<'static, Result<T, PortError>> {
    Box::pin(async move {
        Err(PortError::non_retryable(format!(
            "{port_name} is not configured"
        )))
    })
}

#[cfg(feature = "test-util")]
impl CatalogueDataPort for NoopPort {
    fn list_entity_pages(
        &self,
        _anime_id: Uuid,
    ) -> BoxFuture<'_, Result<ListEntityPagesOutput, PortError>> {
        not_configured("CatalogueDataPort")
    }

    fn fetch_and_store_page(
        &self,
        _page_key: RefreshPageKey,
        _max_age: Duration,
    ) -> BoxFuture<'_, Result<RefreshPageOutput, PortError>> {
        not_configured("CatalogueDataPort")
    }

    fn record_page_failure(
        &self,
        _page_key: RefreshPageKey,
        _failed_at: SystemTime,
    ) -> BoxFuture<'_, Result<(), PortError>> {
        not_configured("CatalogueDataPort")
    }

    fn merge_anime_attempt(
        &self,
        _anime_id: Uuid,
        _merge_version: i32,
    ) -> BoxFuture<'_, Result<MergeAttemptOutcome, PortError>> {
        not_configured("CatalogueDataPort")
    }

    fn record_refresh(
        &self,
        _anime_id: Uuid,
        _outcome: RefreshOutcome,
        _refresh_after_days: i64,
    ) -> BoxFuture<'_, Result<(), PortError>> {
        not_configured("CatalogueDataPort")
    }
}

#[cfg(feature = "test-util")]
impl WorkflowStarterPort for NoopPort {
    fn start_entity_workflows(
        &self,
        _input: StartEntityWorkflowsInput,
    ) -> BoxFuture<'_, Result<usize, PortError>> {
        not_configured("WorkflowStarterPort")
    }
}
