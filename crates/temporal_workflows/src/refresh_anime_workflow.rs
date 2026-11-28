//! Deterministic `RefreshAnimeWorkflow` orchestrating page fetches, merges, and downstream starts.

use std::collections::BTreeMap;
use std::future::Future;
use std::num::NonZeroU32;
use std::pin::Pin;
use std::task::Poll;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use echora_domain::PageKey;
use futures::FutureExt as _;
use temporalio_common::{Priority, RetryPolicy};
use temporalio_macros::{workflow, workflow_methods};
use temporalio_sdk::workflows::select;
use temporalio_sdk::{
    ActivityOptions, CancellableFuture, SyncWorkflowContext, WorkflowContext, WorkflowContextView,
    WorkflowResult,
};
use temporalio_workflow::{
    ActivityExecutionError, TimerResult, WorkflowCancellationToken, WorkflowTermination,
};
use tracing::{Level, event};

use crate::activities::{ProviderActivities, UpdateServiceActivities};
use crate::refresh_anime_types::{
    EntityPageSummary, FetchRequestedSignal, ListEntityPagesOutput, MAX_PAGE_RETRY_BACKOFF_HOURS,
    MergeAnimeOutput, RecordRefreshInput, RefreshAnimeInput, RefreshAnimeOutcome, RefreshOutcome,
    RefreshPageInput, RefreshPageOutput, RefreshSettings, RefreshTrigger, SECONDS_PER_DAY,
    StartEntityWorkflowsInput,
};
use crate::{UPDATE_SERVICE_TASK_QUEUE, WorkerQueueSpec, has_page_fetcher};

/// Seconds in one hour, used to convert backoff hours into seconds.
const SECONDS_PER_HOUR: i64 = 3_600;

/// Largest exponent for `2^failures` hours before clamping to [`MAX_PAGE_RETRY_BACKOFF_HOURS`].
///
/// `2^10 = 1024` already exceeds 168 hours (one week), preventing integer shift overflow.
const MAX_BACKOFF_EXPONENT: u32 = 10;

/// Start-to-close timeout for `list_pages` (`docs/architecture/update_service.md`).
const LIST_PAGES_TIMEOUT: Duration = Duration::from_secs(30);

/// Start-to-close timeout for `refresh_page` (browser-crawled pages can take minutes).
const REFRESH_PAGE_TIMEOUT: Duration = Duration::from_secs(300);

/// Heartbeat timeout for `refresh_page`, matching `docs/architecture/update_service.md`.
const REFRESH_PAGE_HEARTBEAT_TIMEOUT: Duration = Duration::from_secs(60);

/// Start-to-close timeout for `record_page_failure`.
const RECORD_PAGE_FAILURE_TIMEOUT: Duration = Duration::from_secs(30);

/// Start-to-close timeout for `merge_anime`, including in-attempt `ABORTED` retries.
const MERGE_ANIME_TIMEOUT: Duration = Duration::from_secs(120);

/// Start-to-close timeout for `start_entity_workflows` batches.
const START_WORKFLOWS_TIMEOUT: Duration = Duration::from_secs(60);

/// Start-to-close timeout for `record_refresh`.
const RECORD_REFRESH_TIMEOUT: Duration = Duration::from_secs(30);

/// Maximum attempts per activity under Temporal's exponential retry policy.
const STANDARD_MAX_ATTEMPTS: u32 = 5;

/// Highest Temporal priority key supported by server default range (`1..=5`).
const MIN_PRIORITY_KEY: u32 = 1;

/// Lowest Temporal priority key supported by server default range (`1..=5`).
const MAX_PRIORITY_KEY: u32 = 5;

/// Returns the standard 5-attempt exponential retry policy for Update Service activities.
fn standard_retry_policy() -> RetryPolicy {
    RetryPolicy::builder()
        .maximum_attempts(STANDARD_MAX_ATTEMPTS)
        .build()
}

/// Resolves an optional priority key override against `trigger`'s default priority key.
#[must_use]
fn resolve_priority_key(priority_key: Option<NonZeroU32>, trigger: RefreshTrigger) -> u32 {
    priority_key.map_or_else(
        || trigger.default_priority_key(),
        |key| key.get().clamp(MIN_PRIORITY_KEY, MAX_PRIORITY_KEY),
    )
}

/// Builds Temporal `Priority` metadata for a task (`1` = highest, `3` = default).
///
/// # Examples
///
/// ```
/// use std::num::NonZeroU32;
///
/// use temporal_workflows::{RefreshTrigger, build_priority};
///
/// let priority = build_priority(None, RefreshTrigger::Sweep);
/// assert_eq!(priority.priority_key, Some(3));
/// assert_eq!(priority.fairness_key.as_deref(), Some("sweep"));
///
/// let clamped = build_priority(NonZeroU32::new(9), RefreshTrigger::User);
/// assert_eq!(clamped.priority_key, Some(5));
/// ```
#[must_use]
pub fn build_priority(priority_key: Option<NonZeroU32>, trigger: RefreshTrigger) -> Priority {
    Priority::builder()
        .priority_key(resolve_priority_key(priority_key, trigger))
        .fairness_key(trigger.fairness_key())
        .build()
}

/// Selects the highest (numerically lowest) priority key among two optional overrides.
///
/// # Examples
///
/// ```
/// use std::num::NonZeroU32;
///
/// use temporal_workflows::merge_priority;
///
/// assert_eq!(
///     merge_priority(NonZeroU32::new(3), NonZeroU32::new(1)),
///     NonZeroU32::new(1)
/// );
/// assert_eq!(merge_priority(None, NonZeroU32::new(2)), NonZeroU32::new(2));
/// ```
#[must_use]
pub fn merge_priority(
    current: Option<NonZeroU32>,
    incoming: Option<NonZeroU32>,
) -> Option<NonZeroU32> {
    match (current, incoming) {
        (Some(left), Some(right)) => Some(left.min(right)),
        (Some(priority), None) | (None, Some(priority)) => Some(priority),
        (None, None) => None,
    }
}

/// Builds standard `ActivityOptions` for activities running on `update-service`.
fn build_update_service_activity_options(
    timeout: Duration,
    priority_key: Option<NonZeroU32>,
    trigger: RefreshTrigger,
    cancellation_token: Option<WorkflowCancellationToken>,
) -> ActivityOptions {
    ActivityOptions::with_start_to_close_timeout(timeout)
        .task_queue(UPDATE_SERVICE_TASK_QUEUE)
        .retry_policy(standard_retry_policy())
        .priority(build_priority(priority_key, trigger))
        .maybe_cancellation_token(cancellation_token)
        .build()
}

/// Converts a workflow `SystemTime` into UTC epoch seconds.
fn to_epoch_seconds(timestamp: Option<SystemTime>) -> i64 {
    timestamp
        .and_then(|ts| ts.duration_since(UNIX_EPOCH).ok())
        .and_then(|duration| i64::try_from(duration.as_secs()).ok())
        .unwrap_or(0)
}

/// Computes the exponential failure retry backoff in seconds, capped at one week (168h).
///
/// # Examples
///
/// ```
/// use temporal_workflows::compute_page_backoff_seconds;
///
/// assert_eq!(compute_page_backoff_seconds(1), 7_200);
/// assert_eq!(compute_page_backoff_seconds(20), 168 * 3_600);
/// ```
#[must_use]
pub fn compute_page_backoff_seconds(fetch_failures: i32) -> i64 {
    if fetch_failures <= 0 {
        return 0;
    }
    let exponent = u32::try_from(fetch_failures).unwrap_or(MAX_BACKOFF_EXPONENT);
    let hours = 1_i64
        .checked_shl(exponent.min(MAX_BACKOFF_EXPONENT))
        .unwrap_or(MAX_PAGE_RETRY_BACKOFF_HOURS)
        .min(MAX_PAGE_RETRY_BACKOFF_HOURS);
    hours * SECONDS_PER_HOUR
}

/// Evaluates whether a source page is due for refresh at `now_epoch_seconds`.
///
/// # Examples
///
/// ```
/// use echora_domain::{EntityType, PageKey, Provider};
/// use temporal_workflows::{EntityPageSummary, is_page_due};
///
/// let page = EntityPageSummary {
///     page_key: PageKey::new(Provider::Mal, EntityType::Anime, "1").expect("valid page key"),
///     is_partial_source: false,
///     fetched_at_epoch_seconds: None,
///     is_dead: false,
///     fetch_failed_at_epoch_seconds: None,
///     fetch_failures: 0,
/// };
/// assert!(is_page_due(&page, 1_700_000_000, 7));
/// ```
#[must_use]
pub fn is_page_due(
    page: &EntityPageSummary,
    now_epoch_seconds: i64,
    refresh_after_days: i64,
) -> bool {
    if page.is_partial_source || page.is_dead {
        return false;
    }

    if let Some(failed_at) = page.fetch_failed_at_epoch_seconds
        && page.fetch_failures > 0
    {
        let retry_at = failed_at.saturating_add(compute_page_backoff_seconds(page.fetch_failures));
        if now_epoch_seconds < retry_at {
            return false;
        }
    }

    match page.fetched_at_epoch_seconds {
        None => true,
        Some(fetched_at) => {
            now_epoch_seconds.saturating_sub(fetched_at)
                >= refresh_after_days.saturating_mul(SECONDS_PER_DAY)
        }
    }
}

/// Evaluates whether a page should be refreshed due to a `fetch_requested` signal.
///
/// # Examples
///
/// ```
/// use std::num::NonZeroU32;
/// use std::time::Duration;
///
/// use echora_domain::{EntityType, PageKey, Provider};
/// use temporal_workflows::{EntityPageSummary, FetchRequestedSignal, matches_fetch_request};
///
/// let page = EntityPageSummary {
///     page_key: PageKey::new(Provider::Kitsu, EntityType::Anime, "1").expect("valid page key"),
///     is_partial_source: false,
///     fetched_at_epoch_seconds: Some(1_700_000_000),
///     is_dead: false,
///     fetch_failed_at_epoch_seconds: None,
///     fetch_failures: 0,
/// };
/// let signal = FetchRequestedSignal {
///     providers: vec![Provider::Kitsu],
///     max_age: Duration::ZERO,
///     priority: NonZeroU32::new(1),
/// };
/// assert!(matches_fetch_request(&page, &signal, 1_700_000_010));
/// ```
#[must_use]
pub fn matches_fetch_request(
    page: &EntityPageSummary,
    request: &FetchRequestedSignal,
    now_epoch_seconds: i64,
) -> bool {
    if page.is_partial_source || page.is_dead {
        return false;
    }
    if !request.providers.is_empty() && !request.providers.contains(&page.page_key.provider()) {
        return false;
    }
    if request.max_age.is_zero() {
        return true;
    }
    let max_age_seconds = i64::try_from(request.max_age.as_secs()).unwrap_or(i64::MAX);
    match page.fetched_at_epoch_seconds {
        None => true,
        Some(fetched_at) => now_epoch_seconds.saturating_sub(fetched_at) >= max_age_seconds,
    }
}

/// Selects the deduplicated list of [`RefreshPageInput`] tasks to run.
///
/// Pages of a provider outside [`crate::PAGE_FETCH_PROVIDERS`] are never selected, since no
/// worker polls a task queue for their fetches.
///
/// # Examples
///
/// ```
/// use echora_domain::{EntityType, PageKey, Provider};
/// use temporal_workflows::{EntityPageSummary, select_pages_to_refresh};
///
/// let build_page = |provider, address| EntityPageSummary {
///     page_key: PageKey::new(provider, EntityType::Anime, address).expect("valid page key"),
///     is_partial_source: false,
///     fetched_at_epoch_seconds: None,
///     is_dead: false,
///     fetch_failed_at_epoch_seconds: None,
///     fetch_failures: 0,
/// };
/// let pages = [build_page(Provider::Mal, "1"), build_page(Provider::Ann, "2")];
/// let selected = select_pages_to_refresh(&pages, 1_700_000_000, 7, &[], true);
/// assert_eq!(selected.len(), 1);
/// assert_eq!(selected[0].page_key.provider(), Provider::Mal);
/// ```
#[must_use]
pub fn select_pages_to_refresh(
    pages: &[EntityPageSummary],
    now_epoch_seconds: i64,
    refresh_after_days: i64,
    requests: &[FetchRequestedSignal],
    include_due_pages: bool,
) -> Vec<RefreshPageInput> {
    let mut selected: BTreeMap<PageKey, Duration> = BTreeMap::new();
    let default_max_age = Duration::from_secs(
        u64::try_from(refresh_after_days.max(0).saturating_mul(SECONDS_PER_DAY)).unwrap_or(0),
    );

    for page in pages
        .iter()
        .filter(|page| has_page_fetcher(page.page_key.provider()))
    {
        if include_due_pages && is_page_due(page, now_epoch_seconds, refresh_after_days) {
            selected.insert(page.page_key.clone(), default_max_age);
        }
        for request in requests {
            if matches_fetch_request(page, request, now_epoch_seconds) {
                selected
                    .entry(page.page_key.clone())
                    .and_modify(|existing| *existing = (*existing).min(request.max_age))
                    .or_insert(request.max_age);
            }
        }
    }

    selected
        .into_iter()
        .map(|(page_key, max_age)| RefreshPageInput { page_key, max_age })
        .collect()
}

/// Tracks accumulated progress and merge state across a work refresh run.
struct RefreshProgress {
    pages_refreshed: usize,
    pages_failed: usize,
    merges_run: usize,
    episode_workflows_started: usize,
    pending_synopsis: bool,
    synopsis_workflow_started: bool,
    pending_merge: bool,
}

impl RefreshProgress {
    fn new() -> Self {
        Self {
            pages_refreshed: 0,
            pages_failed: 0,
            merges_run: 0,
            episode_workflows_started: 0,
            pending_synopsis: false,
            synopsis_workflow_started: false,
            pending_merge: false,
        }
    }
}

/// Error returned from the inner refresh body distinguishing `record_refresh` failures.
enum RefreshBodyError {
    /// A step before `record_refresh` failed or was cancelled; `record_refresh(Failed)` still runs.
    NeedsFailureRecord(WorkflowTermination),
    /// `record_refresh` itself failed; propagate without calling it a second time.
    RecordRefreshFailed(WorkflowTermination),
}

#[allow(
    missing_docs,
    reason = "temporalio_macros generates undocumented associated constants"
)]
mod workflow_impl {
    use super::{
        FetchRequestedSignal, LIST_PAGES_TIMEOUT, ListEntityPagesOutput, NonZeroU32,
        RefreshAnimeInput, RefreshAnimeOutcome, RefreshBodyError, RefreshOutcome, RefreshProgress,
        RefreshTrigger, SyncWorkflowContext, UpdateServiceActivities, WorkflowCancellationToken,
        WorkflowContext, WorkflowContextView, WorkflowResult, WorkflowTermination,
        build_update_service_activity_options, call_record_refresh, finish_cycle, merge_priority,
        resolve_priority_key, run_pending_merge, run_refresh_page_batch, select_pages_to_refresh,
        to_epoch_seconds, workflow, workflow_methods,
    };
    use tracing::{Level, event};

    /// Durable workflow refreshing one work (`refresh-anime-<anime_id>`).
    #[workflow]
    #[derive(Debug)]
    pub struct RefreshAnimeWorkflow {
        input: RefreshAnimeInput,
        pub(super) current_priority: Option<NonZeroU32>,
        pub(super) merge_requested: bool,
        pending_fetch_requests: Vec<FetchRequestedSignal>,
    }

    #[workflow_methods]
    impl RefreshAnimeWorkflow {
        /// Initializes `RefreshAnimeWorkflow` from `input`.
        #[init]
        fn new(_ctx: &WorkflowContextView, input: RefreshAnimeInput) -> Self {
            let initial_priority =
                NonZeroU32::new(resolve_priority_key(input.priority, input.trigger));
            Self {
                input,
                current_priority: initial_priority,
                merge_requested: false,
                pending_fetch_requests: Vec::new(),
            }
        }

        /// Requests specific providers to refresh with a custom `max_age` and priority.
        #[signal]
        pub fn fetch_requested(
            &mut self,
            _ctx: &mut SyncWorkflowContext<Self>,
            signal: FetchRequestedSignal,
        ) {
            self.current_priority = merge_priority(self.current_priority, signal.priority);
            self.pending_fetch_requests.push(signal);
        }

        /// Marks that the work needs a merge even if no provider page changed.
        #[signal]
        pub fn merge_requested(&mut self, _ctx: &mut SyncWorkflowContext<Self>) {
            self.merge_requested = true;
        }

        /// Runs the full refresh lifecycle and records the outcome via `record_refresh`.
        ///
        /// Every failure or cancellation path records `RefreshOutcome::Failed` before returning.
        /// Any `merge_requested` or `fetch_requested` signal arriving during the final activities
        /// triggers another cycle before the workflow completes.
        ///
        /// # Errors
        ///
        /// Returns a workflow failure or cancellation when `list_pages`, `merge_anime`,
        /// `start_entity_workflows`, `record_page_failure`, or `record_refresh` fails, or when
        /// the workflow is cancelled.
        #[run]
        pub async fn run(ctx: &mut WorkflowContext<Self>) -> WorkflowResult<RefreshAnimeOutcome> {
            let input = ctx.state(|state| state.input.clone());
            match run_refresh_cycles(ctx, &input).await {
                Ok(outcome) => Ok(outcome),
                Err(RefreshBodyError::RecordRefreshFailed(termination)) => Err(termination),
                Err(RefreshBodyError::NeedsFailureRecord(termination)) => {
                    if let Err(record_err) = call_record_refresh(
                        ctx,
                        &input,
                        RefreshOutcome::Failed,
                        Some(WorkflowCancellationToken::new()),
                    )
                    .await
                    {
                        event!(
                            name: "update_service.record_refresh.cleanup_failed",
                            Level::WARN,
                            anime.id = %input.anime_id,
                            error.message = %record_err,
                            "failed to record Failed refresh outcome for {{anime.id}}: {{error.message}}",
                        );
                    }
                    Err(termination)
                }
            }
        }
    }

    /// Executes refresh cycles until no `merge_requested` or `fetch_requested` signals remain.
    async fn run_refresh_cycles(
        ctx: &WorkflowContext<RefreshAnimeWorkflow>,
        input: &RefreshAnimeInput,
    ) -> Result<RefreshAnimeOutcome, RefreshBodyError> {
        let mut progress = RefreshProgress::new();
        let mut is_first_cycle = true;
        let mut pages_relinked = input.pages_relinked;

        loop {
            let priority_key = ctx.state(|state| state.current_priority);
            let list_output: ListEntityPagesOutput = ctx
                .execute_activity(
                    UpdateServiceActivities::list_pages,
                    input.anime_id,
                    build_update_service_activity_options(
                        LIST_PAGES_TIMEOUT,
                        priority_key,
                        input.trigger,
                        None,
                    ),
                )
                .await
                .map_err(|err| {
                    RefreshBodyError::NeedsFailureRecord(WorkflowTermination::from(err))
                })?;

            let settings = list_output.refresh_settings;
            let needs_remerge = is_first_cycle
                && list_output
                    .entity_merge_version
                    .is_none_or(|stored_version| stored_version < settings.merge_version);

            let pending_requests =
                ctx.state_mut(|state| std::mem::take(&mut state.pending_fetch_requests));
            let include_due_pages = is_first_cycle && input.trigger != RefreshTrigger::Remerge;
            let now_epoch_seconds = to_epoch_seconds(ctx.workflow_time());
            let pages_to_refresh = select_pages_to_refresh(
                &list_output.pages,
                now_epoch_seconds,
                settings.refresh_after_days,
                &pending_requests,
                include_due_pages,
            );
            is_first_cycle = false;

            progress.pending_merge |=
                pages_relinked || needs_remerge || ctx.state(|state| state.merge_requested);
            pages_relinked = false;
            ctx.state_mut(|state| state.merge_requested = false);

            if !pages_to_refresh.is_empty() {
                run_refresh_page_batch(ctx, input, pages_to_refresh, settings, &mut progress)
                    .await
                    .map_err(RefreshBodyError::NeedsFailureRecord)?;
            }

            while progress.pending_merge || ctx.state(|state| state.merge_requested) {
                run_pending_merge(ctx, input, &mut progress)
                    .await
                    .map_err(RefreshBodyError::NeedsFailureRecord)?;
            }

            let final_outcome = finish_cycle(ctx, input, &mut progress).await?;

            let has_more_work = ctx
                .state(|state| state.merge_requested || !state.pending_fetch_requests.is_empty());
            if !has_more_work {
                return Ok(RefreshAnimeOutcome {
                    anime_id: input.anime_id,
                    pages_refreshed: progress.pages_refreshed,
                    pages_failed: progress.pages_failed,
                    merges_run: progress.merges_run,
                    episode_workflows_started: progress.episode_workflows_started,
                    synopsis_workflow_started: progress.synopsis_workflow_started,
                    final_outcome,
                });
            }
        }
    }
}

#[doc(inline)]
pub use workflow_impl::RefreshAnimeWorkflow;

/// Starts `WriteSynopsisWorkflow` when provider synopses changed, then calls `record_refresh`.
async fn finish_cycle(
    ctx: &WorkflowContext<RefreshAnimeWorkflow>,
    input: &RefreshAnimeInput,
    progress: &mut RefreshProgress,
) -> Result<RefreshOutcome, RefreshBodyError> {
    if progress.pending_synopsis {
        progress.pending_synopsis = false;
        let started = start_downstream_workflows(ctx, input, Vec::new(), true)
            .await
            .map_err(RefreshBodyError::NeedsFailureRecord)?;
        progress.synopsis_workflow_started |= started > 0;
    }

    let final_outcome = if progress.pages_failed > 0 {
        RefreshOutcome::PagesFailed
    } else {
        RefreshOutcome::Succeeded
    };
    call_record_refresh(ctx, input, final_outcome, None)
        .await
        .map_err(RefreshBodyError::RecordRefreshFailed)?;
    Ok(final_outcome)
}

/// Pending `refresh_page` activity paired with its page key for workflow-side failure recording.
struct PendingPageFetch<F> {
    page_key: PageKey,
    future: Pin<Box<F>>,
}

/// Runs one concurrent batch of `refresh_page` activities with early-merge and late-coalesce timers.
async fn run_refresh_page_batch(
    ctx: &WorkflowContext<RefreshAnimeWorkflow>,
    input: &RefreshAnimeInput,
    pages_to_refresh: Vec<RefreshPageInput>,
    settings: RefreshSettings,
    progress: &mut RefreshProgress,
) -> WorkflowResult<()> {
    let page_timeout = resolve_refresh_page_timeout(settings);
    let merge_first_wait = Duration::from_secs(settings.merge_first_wait_seconds);
    let merge_coalesce_wait = Duration::from_secs(settings.merge_coalesce_wait_seconds);

    let mut page_futures = Vec::with_capacity(pages_to_refresh.len());
    for page_input in pages_to_refresh {
        let page_key = page_input.page_key.clone();
        let task_queue = WorkerQueueSpec::Provider(page_key.provider()).task_queue();
        let activity_priority_key = ctx.state(|state| state.current_priority);
        let options = ActivityOptions::with_start_to_close_timeout(page_timeout)
            .task_queue(task_queue)
            .heartbeat_timeout(REFRESH_PAGE_HEARTBEAT_TIMEOUT)
            .retry_policy(standard_retry_policy())
            .priority(build_priority(activity_priority_key, input.trigger))
            .build();
        page_futures.push(PendingPageFetch {
            page_key,
            future: Box::pin(ctx.execute_activity(
                ProviderActivities::refresh_page,
                page_input,
                options,
            )),
        });
    }

    let mut merge_timer: Option<Pin<Box<dyn CancellableFuture<Output = TimerResult> + '_>>> = None;

    while !page_futures.is_empty() {
        let batch_event = wait_next_batch_event(ctx, &mut page_futures, &mut merge_timer).await;
        match batch_event {
            BatchEvent::Cancelled => {
                cancel_batch_handles(&page_futures, merge_timer.as_deref());
                return Err(WorkflowTermination::cancelled());
            }
            BatchEvent::PageCompleted(page_key, page_result) => {
                let was_stored = match handle_page_completion(
                    ctx,
                    input,
                    page_key,
                    *page_result,
                    progress,
                )
                .await
                {
                    Ok(was_stored) => was_stored,
                    Err(err) => {
                        cancel_batch_handles(&page_futures, merge_timer.as_deref());
                        return Err(err);
                    }
                };
                if page_futures.is_empty() {
                    if let Some(timer) = merge_timer.take() {
                        timer.cancel();
                    }
                } else if merge_timer.is_none() {
                    if progress.merges_run == 0 && was_stored && !merge_first_wait.is_zero() {
                        merge_timer = Some(Box::pin(ctx.timer(merge_first_wait)));
                    } else if progress.merges_run > 0
                        && progress.pending_merge
                        && !merge_coalesce_wait.is_zero()
                    {
                        merge_timer = Some(Box::pin(ctx.timer(merge_coalesce_wait)));
                    }
                }
            }
            BatchEvent::TriggerMerge => {
                if let Some(timer) = merge_timer.take() {
                    timer.cancel();
                }
                if (progress.pending_merge || ctx.state(|state| state.merge_requested))
                    && let Err(err) = run_pending_merge(ctx, input, progress).await
                {
                    cancel_batch_handles(&page_futures, merge_timer.as_deref());
                    return Err(err);
                }
            }
        }
    }

    Ok(())
}

/// Returns the `refresh_page` start-to-close timeout, honoring test overrides under `test-util`.
fn resolve_refresh_page_timeout(settings: RefreshSettings) -> Duration {
    #[cfg(feature = "test-util")]
    if let Some(seconds) = settings.refresh_page_timeout_seconds {
        return Duration::from_secs(seconds);
    }
    let _ = settings;
    REFRESH_PAGE_TIMEOUT
}

enum BatchEvent {
    Cancelled,
    PageCompleted(
        PageKey,
        Box<Result<RefreshPageOutput, ActivityExecutionError>>,
    ),
    TriggerMerge,
}

/// Waits deterministically for workflow cancellation, the next page result, a merge timer, or `merge_requested`.
async fn wait_next_batch_event<'a, F>(
    ctx: &'a WorkflowContext<RefreshAnimeWorkflow>,
    page_futures: &mut Vec<PendingPageFetch<F>>,
    merge_timer: &mut Option<Pin<Box<dyn CancellableFuture<Output = TimerResult> + 'a>>>,
) -> BatchEvent
where
    F: CancellableFuture<Output = Result<RefreshPageOutput, ActivityExecutionError>>,
{
    let mut cancelled = std::pin::pin!(ctx.cancelled());
    let mut merge_signal = std::pin::pin!(ctx.wait_condition(|state| state.merge_requested));
    let mut next_page = std::pin::pin!(poll_any_ready(page_futures).fuse());

    if let Some(timer) = merge_timer.as_mut() {
        select! {
            _ = cancelled => BatchEvent::Cancelled,
            (page_key, result) = next_page => BatchEvent::PageCompleted(page_key, Box::new(result)),
            _ = timer.as_mut() => {
                *merge_timer = None;
                BatchEvent::TriggerMerge
            },
            _ = merge_signal => BatchEvent::TriggerMerge,
        }
    } else {
        select! {
            _ = cancelled => BatchEvent::Cancelled,
            (page_key, result) = next_page => BatchEvent::PageCompleted(page_key, Box::new(result)),
            _ = merge_signal => BatchEvent::TriggerMerge,
        }
    }
}

/// Cancels all in-flight page activities and any active merge timer cleanly.
fn cancel_batch_handles<F>(
    page_futures: &[PendingPageFetch<F>],
    merge_timer: Option<&(dyn CancellableFuture<Output = TimerResult> + '_)>,
) where
    F: CancellableFuture,
{
    if let Some(timer) = merge_timer {
        timer.cancel();
    }
    for pending in page_futures {
        pending.future.cancel();
    }
}

/// Polls pinned page activity futures in deterministic, order-preserving sequence.
///
/// Uses a manual poll loop instead of `FuturesUnordered` because `FuturesUnordered` allocates
/// its own wakers, which trips Temporal's `SdkGuardedFuture` nondeterminism detector.
async fn poll_any_ready<F>(pending: &mut Vec<PendingPageFetch<F>>) -> (PageKey, F::Output)
where
    F: Future,
{
    std::future::poll_fn(|cx| {
        for index in 0..pending.len() {
            if let Poll::Ready(output) = pending[index].future.as_mut().poll(cx) {
                let completed = pending.remove(index);
                return Poll::Ready((completed.page_key, output));
            }
        }
        Poll::Pending
    })
    .await
}

/// Records page failure on non-cancelled activity errors; returns `true` when the page was stored.
async fn handle_page_completion(
    ctx: &WorkflowContext<RefreshAnimeWorkflow>,
    input: &RefreshAnimeInput,
    page_key: PageKey,
    page_result: Result<RefreshPageOutput, ActivityExecutionError>,
    progress: &mut RefreshProgress,
) -> WorkflowResult<bool> {
    match page_result {
        Ok(output) => {
            progress.pages_refreshed += 1;
            if output.has_failed {
                progress.pages_failed += 1;
                Ok(false)
            } else {
                if output.has_changed {
                    progress.pending_merge = true;
                }
                Ok(true)
            }
        }
        Err(ActivityExecutionError::Cancelled(_)) => Err(WorkflowTermination::cancelled()),
        Err(activity_err) => {
            event!(
                name: "update_service.refresh_page.recording_failure",
                Level::WARN,
                anime.id = %input.anime_id,
                provider = %page_key.provider(),
                page.address = %page_key.address(),
                error.message = %activity_err,
                "recording page fetch failure for {{provider}}:{{page.address}} on {{anime.id}}: {{error.message}}",
            );
            let priority_key = ctx.state(|state| state.current_priority);
            ctx.execute_activity(
                UpdateServiceActivities::record_page_failure,
                page_key,
                build_update_service_activity_options(
                    RECORD_PAGE_FAILURE_TIMEOUT,
                    priority_key,
                    input.trigger,
                    None,
                ),
            )
            .await?;
            progress.pages_refreshed += 1;
            progress.pages_failed += 1;
            Ok(false)
        }
    }
}

/// Clears merge flags, runs `merge_anime`, and starts episode refreshes for changed episodes.
async fn run_pending_merge(
    ctx: &WorkflowContext<RefreshAnimeWorkflow>,
    input: &RefreshAnimeInput,
    progress: &mut RefreshProgress,
) -> WorkflowResult<()> {
    progress.pending_merge = false;
    ctx.state_mut(|state| state.merge_requested = false);
    let priority_key = ctx.state(|state| state.current_priority);

    let merge_output: MergeAnimeOutput = ctx
        .execute_activity(
            UpdateServiceActivities::merge_anime,
            input.anime_id,
            build_update_service_activity_options(
                MERGE_ANIME_TIMEOUT,
                priority_key,
                input.trigger,
                None,
            ),
        )
        .await?;

    progress.merges_run += 1;
    if merge_output.has_changed_synopses {
        progress.pending_synopsis = true;
    }

    if !merge_output.changed_episode_ids.is_empty() {
        let started =
            start_downstream_workflows(ctx, input, merge_output.changed_episode_ids, false).await?;
        progress.episode_workflows_started += started;
    }

    Ok(())
}

/// Starts downstream episode and/or synopsis workflows via `start_entity_workflows`.
async fn start_downstream_workflows(
    ctx: &WorkflowContext<RefreshAnimeWorkflow>,
    input: &RefreshAnimeInput,
    episode_ids: Vec<String>,
    start_synopsis: bool,
) -> WorkflowResult<usize> {
    let priority_key = ctx.state(|state| state.current_priority);
    let started = ctx
        .execute_activity(
            UpdateServiceActivities::start_entity_workflows,
            StartEntityWorkflowsInput {
                anime_id: input.anime_id,
                episode_ids,
                start_synopsis,
                trigger: input.trigger,
                priority: priority_key,
            },
            build_update_service_activity_options(
                START_WORKFLOWS_TIMEOUT,
                priority_key,
                input.trigger,
                None,
            ),
        )
        .await?;
    Ok(started)
}

/// Records the final refresh outcome via `UpdateServiceActivities::record_refresh`.
async fn call_record_refresh(
    ctx: &WorkflowContext<RefreshAnimeWorkflow>,
    input: &RefreshAnimeInput,
    outcome: RefreshOutcome,
    cancellation_token: Option<WorkflowCancellationToken>,
) -> WorkflowResult<()> {
    let priority_key = ctx.state(|state| state.current_priority);
    ctx.execute_activity(
        UpdateServiceActivities::record_refresh,
        RecordRefreshInput {
            anime_id: input.anime_id,
            outcome,
        },
        build_update_service_activity_options(
            RECORD_REFRESH_TIMEOUT,
            priority_key,
            input.trigger,
            cancellation_token,
        ),
    )
    .await?;
    Ok(())
}
