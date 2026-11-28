//! Deterministic `RefreshAnimeWorkflow` orchestrating page fetches, merges, and downstream starts.

use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::task::Poll;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use futures::future::{Either, select};
use temporalio_common::{Priority, RetryPolicy};
use temporalio_macros::{workflow, workflow_methods};
use temporalio_sdk::{
    ActivityOptions, SyncWorkflowContext, WorkflowContext, WorkflowContextView, WorkflowResult,
};

use crate::UPDATE_SERVICE_TASK_QUEUE;
use crate::activities::{ProviderActivities, UpdateServiceActivities};
use crate::refresh_anime_types::{
    DEFAULT_MERGE_COALESCE_WAIT_SECONDS, DEFAULT_MERGE_FIRST_WAIT_MINUTES,
    DEFAULT_REFRESH_AFTER_DAYS, EntityPageSummary, FetchRequestedSignal, ListEntityPagesOutput,
    MAX_PAGE_RETRY_BACKOFF_HOURS, MergeAnimeInput, MergeAnimeOutput, RecordRefreshInput,
    RefreshAnimeInput, RefreshAnimeOutcome, RefreshOutcome, RefreshPageInput, RefreshPageKey,
    RefreshPageOutput, RefreshTrigger, StartEntityWorkflowsInput,
};

/// Seconds in one hour, used to convert backoff hours into seconds.
const SECONDS_PER_HOUR: i64 = 3_600;

/// Seconds in one day, used to compare `fetched_at` against `REFRESH_AFTER_DAYS`.
const SECONDS_PER_DAY: i64 = 86_400;

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

/// Default Temporal priority key when no trigger or caller override is supplied.
const DEFAULT_PRIORITY_KEY: u32 = 3;

/// Returns the standard 5-attempt exponential retry policy for Update Service activities.
fn standard_retry_policy() -> RetryPolicy {
    RetryPolicy::builder()
        .maximum_attempts(STANDARD_MAX_ATTEMPTS)
        .build()
}

/// Builds Temporal `Priority` metadata for a task (`1` = highest, `3` = default).
///
/// # Examples
///
/// ```
/// use temporal_workflows::{RefreshTrigger, build_priority};
///
/// let priority = build_priority(0, RefreshTrigger::Sweep);
/// assert_eq!(priority.priority_key, Some(3));
/// assert_eq!(priority.fairness_key.as_deref(), Some("sweep"));
///
/// let clamped = build_priority(9, RefreshTrigger::User);
/// assert_eq!(clamped.priority_key, Some(5));
/// ```
#[must_use]
pub fn build_priority(priority_key: u32, trigger: RefreshTrigger) -> Priority {
    let resolved = if priority_key == 0 {
        trigger.default_priority_key()
    } else {
        priority_key.clamp(MIN_PRIORITY_KEY, MAX_PRIORITY_KEY)
    };
    Priority::builder()
        .priority_key(resolved)
        .fairness_key(trigger.fairness_key())
        .build()
}

/// Selects the highest (numerically lowest) priority key among two values (`0` means unset).
///
/// # Examples
///
/// ```
/// use temporal_workflows::merge_priority;
///
/// assert_eq!(merge_priority(3, 1), 1);
/// assert_eq!(merge_priority(0, 2), 2);
/// ```
#[must_use]
pub const fn merge_priority(current: u32, incoming: u32) -> u32 {
    match (current, incoming) {
        (0, other) | (other, 0) => other,
        (left, right) => {
            if left < right {
                left
            } else {
                right
            }
        }
    }
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
/// use temporal_workflows::{EntityType, EntityPageSummary, Provider, RefreshPageKey, is_page_due};
///
/// let page = EntityPageSummary {
///     page_key: RefreshPageKey {
///         provider: Provider::Mal,
///         entity_type: EntityType::Anime,
///         address: "1".to_owned(),
///     },
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
/// use temporal_workflows::{
///     EntityType, EntityPageSummary, FetchRequestedSignal, Provider, RefreshPageKey,
///     matches_fetch_request,
/// };
///
/// let page = EntityPageSummary {
///     page_key: RefreshPageKey {
///         provider: Provider::Kitsu,
///         entity_type: EntityType::Anime,
///         address: "1".to_owned(),
///     },
///     is_partial_source: false,
///     fetched_at_epoch_seconds: Some(1_700_000_000),
///     is_dead: false,
///     fetch_failed_at_epoch_seconds: None,
///     fetch_failures: 0,
/// };
/// let signal = FetchRequestedSignal {
///     providers: vec![Provider::Kitsu],
///     max_age_seconds: 0,
///     priority: 1,
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
    if !request.providers.is_empty() && !request.providers.contains(&page.page_key.provider) {
        return false;
    }
    if request.max_age_seconds <= 0 {
        return true;
    }
    match page.fetched_at_epoch_seconds {
        None => true,
        Some(fetched_at) => now_epoch_seconds.saturating_sub(fetched_at) >= request.max_age_seconds,
    }
}

/// Selects the deduplicated list of `(RefreshPageKey, max_age_seconds)` tasks to run.
///
/// # Examples
///
/// ```
/// use temporal_workflows::{
///     EntityType, EntityPageSummary, Provider, RefreshPageKey, select_pages_to_refresh,
/// };
///
/// let page = EntityPageSummary {
///     page_key: RefreshPageKey {
///         provider: Provider::Mal,
///         entity_type: EntityType::Anime,
///         address: "1".to_owned(),
///     },
///     is_partial_source: false,
///     fetched_at_epoch_seconds: None,
///     is_dead: false,
///     fetch_failed_at_epoch_seconds: None,
///     fetch_failures: 0,
/// };
/// let selected = select_pages_to_refresh(&[page], 1_700_000_000, 7, &[], true);
/// assert_eq!(selected.len(), 1);
/// ```
#[must_use]
pub fn select_pages_to_refresh(
    pages: &[EntityPageSummary],
    now_epoch_seconds: i64,
    refresh_after_days: i64,
    requests: &[FetchRequestedSignal],
    include_due_pages: bool,
) -> Vec<RefreshPageInput> {
    let mut selected: BTreeMap<RefreshPageKey, i64> = BTreeMap::new();
    let default_max_age = refresh_after_days.saturating_mul(SECONDS_PER_DAY);

    for page in pages {
        if include_due_pages && is_page_due(page, now_epoch_seconds, refresh_after_days) {
            selected.insert(page.page_key.clone(), default_max_age);
        }
        for request in requests {
            if matches_fetch_request(page, request, now_epoch_seconds) {
                selected
                    .entry(page.page_key.clone())
                    .and_modify(|existing| *existing = (*existing).min(request.max_age_seconds))
                    .or_insert(request.max_age_seconds);
            }
        }
    }

    selected
        .into_iter()
        .map(|(page_key, max_age_seconds)| RefreshPageInput {
            page_key,
            max_age_seconds,
        })
        .collect()
}

/// Tracks accumulated progress and merge state across a work refresh run.
struct RefreshProgress {
    pages_refreshed: usize,
    pages_failed: usize,
    merges_run: usize,
    episode_workflows_started: usize,
    any_synopses_changed: bool,
    pending_merge: bool,
}

impl RefreshProgress {
    fn new() -> Self {
        Self {
            pages_refreshed: 0,
            pages_failed: 0,
            merges_run: 0,
            episode_workflows_started: 0,
            any_synopses_changed: false,
            pending_merge: false,
        }
    }
}

/// Timings resolved from [`RefreshAnimeInput`] with defaults applied.
struct RefreshTimings {
    refresh_after_days: i64,
    merge_first_wait: Duration,
    merge_coalesce_wait: Duration,
}

impl RefreshTimings {
    fn from_input(input: &RefreshAnimeInput) -> Self {
        let refresh_after_days = if input.refresh_after_days > 0 {
            input.refresh_after_days
        } else {
            DEFAULT_REFRESH_AFTER_DAYS
        };
        let merge_first_wait = Duration::from_secs(
            input
                .merge_first_wait_seconds
                .unwrap_or(DEFAULT_MERGE_FIRST_WAIT_MINUTES * 60),
        );
        let merge_coalesce_wait = Duration::from_secs(
            input
                .merge_coalesce_wait_seconds
                .unwrap_or(DEFAULT_MERGE_COALESCE_WAIT_SECONDS),
        );
        Self {
            refresh_after_days,
            merge_first_wait,
            merge_coalesce_wait,
        }
    }
}

#[allow(
    missing_docs,
    reason = "temporalio_macros generates undocumented associated constants"
)]
mod workflow_impl {
    use super::{
        ActivityOptions, DEFAULT_PRIORITY_KEY, FetchRequestedSignal, LIST_PAGES_TIMEOUT,
        ListEntityPagesOutput, RefreshAnimeInput, RefreshAnimeOutcome, RefreshProgress,
        RefreshTimings, RefreshTrigger, SyncWorkflowContext, UPDATE_SERVICE_TASK_QUEUE,
        UpdateServiceActivities, WorkflowContext, WorkflowContextView, WorkflowResult,
        build_priority, finish_refresh, merge_priority, run_merge_or_record_failure,
        run_refresh_page_batch, select_pages_to_refresh, standard_retry_policy, to_epoch_seconds,
        workflow, workflow_methods,
    };

    /// Durable workflow refreshing one work (`refresh-anime-<anime_id>`).
    #[workflow]
    #[derive(Debug)]
    pub struct RefreshAnimeWorkflow {
        input: RefreshAnimeInput,
        pub(super) current_priority: u32,
        pub(super) merge_requested: bool,
        pending_fetch_requests: Vec<FetchRequestedSignal>,
    }

    #[workflow_methods]
    impl RefreshAnimeWorkflow {
        /// Initializes `RefreshAnimeWorkflow` from `input`.
        #[init]
        fn new(_ctx: &WorkflowContextView, input: RefreshAnimeInput) -> Self {
            let initial_priority = if input.priority == 0 {
                let trigger_default = input.trigger.default_priority_key();
                if trigger_default == 0 {
                    DEFAULT_PRIORITY_KEY
                } else {
                    trigger_default
                }
            } else {
                input.priority
            };
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

        /// Runs the full refresh lifecycle and finishes with `record_refresh`.
        ///
        /// # Errors
        ///
        /// Returns a workflow failure when `list_pages`, `merge_anime`,
        /// `start_entity_workflows`, or `record_refresh` fails.
        #[run]
        pub async fn run(ctx: &mut WorkflowContext<Self>) -> WorkflowResult<RefreshAnimeOutcome> {
            let input = ctx.state(|state| state.input.clone());
            let timings = RefreshTimings::from_input(&input);
            let mut progress = RefreshProgress::new();
            let mut is_first_cycle = true;
            let mut pages_relinked = input.pages_relinked;

            loop {
                let priority_key = ctx.state(|state| state.current_priority);
                let list_output: ListEntityPagesOutput = ctx
                    .execute_activity(
                        UpdateServiceActivities::list_pages,
                        input.anime_id.clone(),
                        ActivityOptions::with_start_to_close_timeout(LIST_PAGES_TIMEOUT)
                            .task_queue(UPDATE_SERVICE_TASK_QUEUE)
                            .retry_policy(standard_retry_policy())
                            .priority(build_priority(priority_key, input.trigger))
                            .build(),
                    )
                    .await?;

                let needs_remerge = is_first_cycle
                    && input.merge_version > 0
                    && list_output
                        .entity_merge_version
                        .is_none_or(|stored_version| stored_version < input.merge_version);

                let pending_requests =
                    ctx.state_mut(|state| std::mem::take(&mut state.pending_fetch_requests));
                let include_due_pages = is_first_cycle && input.trigger != RefreshTrigger::Remerge;
                let now_epoch_seconds = to_epoch_seconds(ctx.workflow_time());
                let pages_to_refresh = select_pages_to_refresh(
                    &list_output.pages,
                    now_epoch_seconds,
                    timings.refresh_after_days,
                    &pending_requests,
                    include_due_pages,
                );
                is_first_cycle = false;

                progress.pending_merge |=
                    pages_relinked || needs_remerge || ctx.state(|state| state.merge_requested);
                pages_relinked = false;
                ctx.state_mut(|state| state.merge_requested = false);

                if !pages_to_refresh.is_empty() {
                    run_refresh_page_batch(ctx, &input, pages_to_refresh, &timings, &mut progress)
                        .await?;
                }

                while progress.pending_merge || ctx.state(|state| state.merge_requested) {
                    progress.pending_merge = false;
                    ctx.state_mut(|state| state.merge_requested = false);
                    let merge_priority_key = ctx.state(|state| state.current_priority);
                    run_merge_or_record_failure(
                        ctx,
                        &input,
                        merge_priority_key,
                        timings.refresh_after_days,
                        &mut progress,
                    )
                    .await?;
                }

                let has_more_work = ctx.state(|state| {
                    state.merge_requested || !state.pending_fetch_requests.is_empty()
                });
                if !has_more_work {
                    break;
                }
            }

            finish_refresh(ctx, &input, &progress, timings.refresh_after_days).await
        }
    }
}

#[doc(inline)]
pub use workflow_impl::RefreshAnimeWorkflow;

/// Starts `WriteSynopsisWorkflow` when provider synopses changed, then calls `record_refresh`.
async fn finish_refresh(
    ctx: &WorkflowContext<RefreshAnimeWorkflow>,
    input: &RefreshAnimeInput,
    progress: &RefreshProgress,
    refresh_after_days: i64,
) -> WorkflowResult<RefreshAnimeOutcome> {
    let mut synopsis_workflow_started = false;
    if progress.any_synopses_changed {
        let synopsis_priority_key = ctx.state(|state| state.current_priority);
        let started = ctx
            .execute_activity(
                UpdateServiceActivities::start_entity_workflows,
                StartEntityWorkflowsInput {
                    anime_id: input.anime_id.clone(),
                    episode_ids: Vec::new(),
                    start_synopsis: true,
                    trigger: input.trigger,
                    priority: synopsis_priority_key,
                },
                ActivityOptions::with_start_to_close_timeout(START_WORKFLOWS_TIMEOUT)
                    .task_queue(UPDATE_SERVICE_TASK_QUEUE)
                    .retry_policy(standard_retry_policy())
                    .priority(build_priority(synopsis_priority_key, input.trigger))
                    .build(),
            )
            .await?;
        synopsis_workflow_started = started > 0;
    }

    let final_outcome = if progress.pages_failed > 0 {
        RefreshOutcome::PagesFailed
    } else {
        RefreshOutcome::Succeeded
    };
    call_record_refresh(ctx, input, final_outcome, refresh_after_days).await?;

    Ok(RefreshAnimeOutcome {
        anime_id: input.anime_id.clone(),
        pages_refreshed: progress.pages_refreshed,
        pages_failed: progress.pages_failed,
        merges_run: progress.merges_run,
        episode_workflows_started: progress.episode_workflows_started,
        synopsis_workflow_started,
        final_outcome,
    })
}

/// Runs one concurrent batch of `refresh_page` activities with early-merge and late-coalesce timers.
async fn run_refresh_page_batch(
    ctx: &WorkflowContext<RefreshAnimeWorkflow>,
    input: &RefreshAnimeInput,
    pages_to_refresh: Vec<RefreshPageInput>,
    timings: &RefreshTimings,
    progress: &mut RefreshProgress,
) -> WorkflowResult<()> {
    let mut page_futures = Vec::with_capacity(pages_to_refresh.len());
    for page_input in pages_to_refresh {
        let task_queue = page_input.page_key.provider.task_queue();
        let activity_priority_key = ctx.state(|state| state.current_priority);
        let options = ActivityOptions::with_start_to_close_timeout(REFRESH_PAGE_TIMEOUT)
            .task_queue(task_queue)
            .heartbeat_timeout(REFRESH_PAGE_HEARTBEAT_TIMEOUT)
            .retry_policy(standard_retry_policy())
            .priority(build_priority(activity_priority_key, input.trigger))
            .build();
        page_futures.push(Box::pin(ctx.execute_activity(
            ProviderActivities::refresh_page,
            page_input,
            options,
        )));
    }

    let mut merge_timer = None;

    while !page_futures.is_empty() {
        if let Some(timer) = merge_timer.take() {
            let next_page = {
                let mut next_fut = std::pin::pin!(poll_any_ready(&mut page_futures));
                match select(next_fut.as_mut(), timer).await {
                    Either::Left((page_result, remaining_timer)) => {
                        merge_timer = Some(remaining_timer);
                        Some(page_result)
                    }
                    Either::Right((_elapsed, _pending_next)) => None,
                }
            };
            if let Some(Some(page_result)) = next_page {
                handle_page_completion(page_result, progress);
            } else if next_page.is_none()
                && (progress.pending_merge || ctx.state(|state| state.merge_requested))
            {
                progress.pending_merge = false;
                ctx.state_mut(|state| state.merge_requested = false);
                let merge_priority_key = ctx.state(|state| state.current_priority);
                run_merge_or_record_failure(
                    ctx,
                    input,
                    merge_priority_key,
                    timings.refresh_after_days,
                    progress,
                )
                .await?;
            }
        } else if let Some(page_result) = poll_any_ready(&mut page_futures).await {
            let was_stored = handle_page_completion(page_result, progress);
            if !page_futures.is_empty() {
                if progress.merges_run == 0 && was_stored && !timings.merge_first_wait.is_zero() {
                    merge_timer = Some(Box::pin(ctx.timer(timings.merge_first_wait)));
                } else if progress.merges_run > 0
                    && progress.pending_merge
                    && !timings.merge_coalesce_wait.is_zero()
                {
                    merge_timer = Some(Box::pin(ctx.timer(timings.merge_coalesce_wait)));
                }
            }
        }
    }

    Ok(())
}

/// Polls pinned activity futures in deterministic, order-preserving sequence.
async fn poll_any_ready<F: Future + Unpin>(pending: &mut Vec<F>) -> Option<F::Output> {
    std::future::poll_fn(|cx| {
        if pending.is_empty() {
            return Poll::Ready(None);
        }
        for index in 0..pending.len() {
            if let Poll::Ready(output) = Pin::new(&mut pending[index]).poll(cx) {
                pending.remove(index);
                return Poll::Ready(Some(output));
            }
        }
        Poll::Pending
    })
    .await
}

/// Updates progress after one `refresh_page` completes; returns `true` when the page was stored.
fn handle_page_completion<E>(
    page_result: Result<RefreshPageOutput, E>,
    progress: &mut RefreshProgress,
) -> bool {
    progress.pages_refreshed += 1;
    if let Ok(output) = page_result {
        if output.has_failed {
            progress.pages_failed += 1;
            false
        } else {
            if output.has_changed {
                progress.pending_merge = true;
            }
            true
        }
    } else {
        progress.pages_failed += 1;
        false
    }
}

/// Runs `run_merge_and_episodes` and records a `Failed` refresh outcome before propagating errors.
async fn run_merge_or_record_failure(
    ctx: &WorkflowContext<RefreshAnimeWorkflow>,
    input: &RefreshAnimeInput,
    priority_key: u32,
    refresh_after_days: i64,
    progress: &mut RefreshProgress,
) -> WorkflowResult<()> {
    if let Err(err) = run_merge_and_episodes(ctx, input, priority_key, progress).await {
        let _ = call_record_refresh(ctx, input, RefreshOutcome::Failed, refresh_after_days).await;
        return Err(err);
    }
    Ok(())
}

/// Runs `merge_anime` and starts `RefreshEpisodeWorkflow` for created/updated episodes.
async fn run_merge_and_episodes(
    ctx: &WorkflowContext<RefreshAnimeWorkflow>,
    input: &RefreshAnimeInput,
    priority_key: u32,
    progress: &mut RefreshProgress,
) -> WorkflowResult<()> {
    let merge_output: MergeAnimeOutput = ctx
        .execute_activity(
            UpdateServiceActivities::merge_anime,
            MergeAnimeInput {
                anime_id: input.anime_id.clone(),
                merge_version: input.merge_version,
            },
            ActivityOptions::with_start_to_close_timeout(MERGE_ANIME_TIMEOUT)
                .task_queue(UPDATE_SERVICE_TASK_QUEUE)
                .retry_policy(standard_retry_policy())
                .priority(build_priority(priority_key, input.trigger))
                .build(),
        )
        .await?;

    progress.merges_run += 1;
    if merge_output.has_changed_synopses {
        progress.any_synopses_changed = true;
    }

    if !merge_output.changed_episode_ids.is_empty() {
        let episode_priority_key = ctx.state(|state| state.current_priority);
        let started = ctx
            .execute_activity(
                UpdateServiceActivities::start_entity_workflows,
                StartEntityWorkflowsInput {
                    anime_id: input.anime_id.clone(),
                    episode_ids: merge_output.changed_episode_ids,
                    start_synopsis: false,
                    trigger: input.trigger,
                    priority: episode_priority_key,
                },
                ActivityOptions::with_start_to_close_timeout(START_WORKFLOWS_TIMEOUT)
                    .task_queue(UPDATE_SERVICE_TASK_QUEUE)
                    .retry_policy(standard_retry_policy())
                    .priority(build_priority(episode_priority_key, input.trigger))
                    .build(),
            )
            .await?;
        progress.episode_workflows_started += started;
    }

    Ok(())
}

/// Records the final refresh outcome via `UpdateServiceActivities::record_refresh`.
async fn call_record_refresh(
    ctx: &WorkflowContext<RefreshAnimeWorkflow>,
    input: &RefreshAnimeInput,
    outcome: RefreshOutcome,
    refresh_after_days: i64,
) -> WorkflowResult<()> {
    let priority_key = ctx.state(|state| state.current_priority);
    ctx.execute_activity(
        UpdateServiceActivities::record_refresh,
        RecordRefreshInput {
            anime_id: input.anime_id.clone(),
            outcome,
            refresh_after_days,
        },
        ActivityOptions::with_start_to_close_timeout(RECORD_REFRESH_TIMEOUT)
            .task_queue(UPDATE_SERVICE_TASK_QUEUE)
            .retry_policy(standard_retry_policy())
            .priority(build_priority(priority_key, input.trigger))
            .build(),
    )
    .await?;
    Ok(())
}
