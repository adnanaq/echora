//! Deterministic `RefreshAnimeWorkflow` orchestrating page refreshes, merges, and downstream starts.

use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::task::Poll;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use futures::future::{Either, select};
use temporalio_common::Priority;
use temporalio_common::protos::temporal::api::common::v1::RetryPolicy;
use temporalio_macros::{workflow, workflow_methods};
use temporalio_sdk::{
    ActivityOptions, SyncWorkflowContext, WorkflowContext, WorkflowContextView, WorkflowResult,
};

use crate::activities::{ProviderActivities, UpdateServiceActivities};
use crate::provider_task_queue;
use crate::refresh_anime_types::{
    DEFAULT_MERGE_FIRST_WAIT_MINUTES, DEFAULT_REFRESH_AFTER_DAYS, EntityPageSummary,
    FetchRequestedSignal, ListEntityPagesOutput, MAX_PAGE_RETRY_BACKOFF_HOURS, MergeAnimeInput,
    MergeAnimeOutput, RecordRefreshInput, RefreshAnimeInput, RefreshAnimeOutcome,
    RefreshOutcomeKind, RefreshPageInput, RefreshPageKey, RefreshPageOutput,
    StartEntityWorkflowsInput,
};
use crate::{AGENT_SERVICE_TASK_QUEUE, UPDATE_SERVICE_TASK_QUEUE};

const SECONDS_PER_HOUR: i64 = 3_600;
const SECONDS_PER_DAY: i64 = 86_400;
const MAX_BACKOFF_EXPONENT: u32 = 10;

const LIST_PAGES_TIMEOUT: Duration = Duration::from_secs(30);
const REFRESH_PAGE_TIMEOUT: Duration = Duration::from_secs(300);
const REFRESH_PAGE_HEARTBEAT_TIMEOUT: Duration = Duration::from_secs(60);
const MERGE_ANIME_TIMEOUT: Duration = Duration::from_secs(120);
const START_WORKFLOWS_TIMEOUT: Duration = Duration::from_secs(60);
const RECORD_REFRESH_TIMEOUT: Duration = Duration::from_secs(30);

/// Returns the standard 5-attempt exponential retry policy for Update Service activities.
fn standard_retry_policy() -> RetryPolicy {
    RetryPolicy {
        maximum_attempts: 5,
        ..RetryPolicy::default()
    }
}

/// Computes `priority` metadata for an activity invocation (`1` = highest, `3` = default).
fn build_priority(priority_key: u32) -> Priority {
    let clamped = if priority_key == 0 {
        3
    } else {
        priority_key.clamp(1, 5)
    };
    let mut priority = Priority::default();
    priority.priority_key = Some(clamped);
    priority
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
        .and_then(|dur| i64::try_from(dur.as_secs()).ok())
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
    let exp = u32::try_from(fetch_failures).unwrap_or(MAX_BACKOFF_EXPONENT);
    let hours = 1_i64
        .checked_shl(exp.min(MAX_BACKOFF_EXPONENT))
        .unwrap_or(MAX_PAGE_RETRY_BACKOFF_HOURS)
        .min(MAX_PAGE_RETRY_BACKOFF_HOURS);
    hours * SECONDS_PER_HOUR
}

/// Evaluates whether a source page is due for refresh at `now_epoch_seconds`.
///
/// # Examples
///
/// ```
/// use temporal_workflows::{EntityPageSummary, RefreshPageKey, is_page_due};
///
/// let page = EntityPageSummary {
///     page_key: RefreshPageKey {
///         provider: "mal".to_owned(),
///         entity_type: "anime".to_owned(),
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
///     EntityPageSummary, FetchRequestedSignal, RefreshPageKey, matches_fetch_request,
/// };
///
/// let page = EntityPageSummary {
///     page_key: RefreshPageKey {
///         provider: "kitsu".to_owned(),
///         entity_type: "anime".to_owned(),
///         address: "1".to_owned(),
///     },
///     is_partial_source: false,
///     fetched_at_epoch_seconds: Some(1_700_000_000),
///     is_dead: false,
///     fetch_failed_at_epoch_seconds: None,
///     fetch_failures: 0,
/// };
/// let signal = FetchRequestedSignal {
///     providers: vec!["kitsu".to_owned()],
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
    if !request.providers.is_empty()
        && !request
            .providers
            .iter()
            .any(|provider| provider == &page.page_key.provider)
    {
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
/// use temporal_workflows::{EntityPageSummary, RefreshPageKey, select_pages_to_refresh};
///
/// let page = EntityPageSummary {
///     page_key: RefreshPageKey {
///         provider: "mal".to_owned(),
///         entity_type: "anime".to_owned(),
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
        for req in requests {
            if matches_fetch_request(page, req, now_epoch_seconds) {
                selected
                    .entry(page.page_key.clone())
                    .and_modify(|existing| *existing = (*existing).min(req.max_age_seconds))
                    .or_insert(req.max_age_seconds);
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

#[allow(
    missing_docs,
    reason = "temporalio_macros generates undocumented associated constants"
)]
mod workflow_impl {
    use super::{
        ActivityOptions, DEFAULT_MERGE_FIRST_WAIT_MINUTES, DEFAULT_REFRESH_AFTER_DAYS, Duration,
        FetchRequestedSignal, LIST_PAGES_TIMEOUT, ListEntityPagesOutput, RefreshAnimeInput,
        RefreshAnimeOutcome, RefreshOutcomeKind, SyncWorkflowContext, UPDATE_SERVICE_TASK_QUEUE,
        UpdateServiceActivities, WorkflowContext, WorkflowContextView, WorkflowResult,
        build_priority, call_record_refresh, finish_refresh, merge_priority,
        run_merge_and_episodes, run_refresh_page_batch, select_pages_to_refresh,
        standard_retry_policy, to_epoch_seconds, workflow, workflow_methods,
    };

    /// Durable workflow refreshing one work (`refresh-anime-<anime_id>`).
    #[workflow]
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
                3
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
        #[allow(
            clippy::too_many_lines,
            reason = "workflow run loop coordinates page fetch, merge wait timer, and signal drain"
        )]
        pub async fn run(ctx: &mut WorkflowContext<Self>) -> WorkflowResult<RefreshAnimeOutcome> {
            let input = ctx.state(|state| state.input.clone());
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

            let mut pages_refreshed = 0_usize;
            let mut pages_failed = 0_usize;
            let mut merges_run = 0_usize;
            let mut episode_workflows_started = 0_usize;
            let mut any_synopses_changed = false;
            let mut first_cycle = true;
            let mut pages_relinked = input.pages_relinked;

            loop {
                let priority_key = ctx.state(|state| state.current_priority);
                let list_output: ListEntityPagesOutput = ctx
                    .execute_activity(
                        UpdateServiceActivities::list_pages,
                        input.anime_id.clone(),
                        ActivityOptions::with_start_to_close_timeout(LIST_PAGES_TIMEOUT)
                            .task_queue(UPDATE_SERVICE_TASK_QUEUE.to_owned())
                            .retry_policy(standard_retry_policy())
                            .priority(build_priority(priority_key))
                            .build(),
                    )
                    .await?;

                let needs_remerge = first_cycle
                    && input.merge_version > 0
                    && list_output
                        .entity_merge_version
                        .is_none_or(|ver| ver < input.merge_version);

                let pending_requests =
                    ctx.state_mut(|state| std::mem::take(&mut state.pending_fetch_requests));
                let now_epoch_seconds = to_epoch_seconds(ctx.workflow_time());
                let pages_to_refresh = select_pages_to_refresh(
                    &list_output.pages,
                    now_epoch_seconds,
                    refresh_after_days,
                    &pending_requests,
                    first_cycle,
                );
                first_cycle = false;

                let mut pending_merge =
                    pages_relinked || needs_remerge || ctx.state(|state| state.merge_requested);
                pages_relinked = false;
                ctx.state_mut(|state| state.merge_requested = false);

                if !pages_to_refresh.is_empty() {
                    run_refresh_page_batch(
                        ctx,
                        &input,
                        pages_to_refresh,
                        merge_first_wait,
                        refresh_after_days,
                        &mut pages_refreshed,
                        &mut pages_failed,
                        &mut pending_merge,
                        &mut merges_run,
                        &mut episode_workflows_started,
                        &mut any_synopses_changed,
                    )
                    .await?;
                }

                if pending_merge || ctx.state(|state| state.merge_requested) {
                    ctx.state_mut(|state| state.merge_requested = false);
                    let merge_priority = ctx.state(|state| state.current_priority);
                    if let Err(err) = run_merge_and_episodes(
                        ctx,
                        &input,
                        merge_priority,
                        &mut merges_run,
                        &mut episode_workflows_started,
                        &mut any_synopses_changed,
                    )
                    .await
                    {
                        let _ = call_record_refresh(
                            ctx,
                            &input.anime_id,
                            RefreshOutcomeKind::Failed,
                            refresh_after_days,
                        )
                        .await;
                        return Err(err);
                    }
                }

                let more_work = ctx.state(|state| {
                    state.merge_requested || !state.pending_fetch_requests.is_empty()
                });
                if !more_work {
                    break;
                }
            }

            finish_refresh(
                ctx,
                &input,
                pages_refreshed,
                pages_failed,
                merges_run,
                episode_workflows_started,
                any_synopses_changed,
                refresh_after_days,
            )
            .await
        }
    }
}

#[doc(inline)]
pub use workflow_impl::RefreshAnimeWorkflow;

/// Starts `WriteSynopsisWorkflow` when provider synopses changed, then calls `record_refresh`.
#[allow(
    clippy::too_many_arguments,
    reason = "gathers refresh summary fields for final recording"
)]
async fn finish_refresh(
    ctx: &WorkflowContext<RefreshAnimeWorkflow>,
    input: &RefreshAnimeInput,
    pages_refreshed: usize,
    pages_failed: usize,
    merges_run: usize,
    episode_workflows_started: usize,
    any_synopses_changed: bool,
    refresh_after_days: i64,
) -> WorkflowResult<RefreshAnimeOutcome> {
    let mut synopsis_workflow_started = false;
    if any_synopses_changed {
        let syn_priority = ctx.state(|state| state.current_priority);
        let started = ctx
            .execute_activity(
                UpdateServiceActivities::start_entity_workflows,
                StartEntityWorkflowsInput {
                    anime_id: input.anime_id.clone(),
                    episode_ids: Vec::new(),
                    start_synopsis: true,
                    trigger: input.trigger.clone(),
                    priority: syn_priority,
                },
                ActivityOptions::with_start_to_close_timeout(START_WORKFLOWS_TIMEOUT)
                    .task_queue(AGENT_SERVICE_TASK_QUEUE.to_owned())
                    .retry_policy(standard_retry_policy())
                    .priority(build_priority(syn_priority))
                    .build(),
            )
            .await?;
        synopsis_workflow_started = started > 0;
    }

    let final_outcome = if pages_failed > 0 {
        RefreshOutcomeKind::PagesFailed
    } else {
        RefreshOutcomeKind::Succeeded
    };
    call_record_refresh(ctx, &input.anime_id, final_outcome, refresh_after_days).await?;

    Ok(RefreshAnimeOutcome {
        anime_id: input.anime_id.clone(),
        pages_refreshed,
        pages_failed,
        merges_run,
        episode_workflows_started,
        synopsis_workflow_started,
        final_outcome,
    })
}

/// Runs one concurrent batch of `refresh_page` activities and triggers an early merge
/// if `merge_first_wait` expires before all pages settle.
#[allow(
    clippy::too_many_arguments,
    reason = "workflow helper updates counters and merge state across a batch"
)]
async fn run_refresh_page_batch(
    ctx: &WorkflowContext<RefreshAnimeWorkflow>,
    input: &RefreshAnimeInput,
    pages_to_refresh: Vec<RefreshPageInput>,
    merge_first_wait: Duration,
    refresh_after_days: i64,
    pages_refreshed: &mut usize,
    pages_failed: &mut usize,
    pending_merge: &mut bool,
    merges_run: &mut usize,
    episode_workflows_started: &mut usize,
    any_synopses_changed: &mut bool,
) -> WorkflowResult<()> {
    let mut page_futures = Vec::with_capacity(pages_to_refresh.len());
    for page_input in pages_to_refresh {
        let task_queue = provider_task_queue(&page_input.page_key.provider);
        let act_priority = ctx.state(|state| state.current_priority);
        let opts = ActivityOptions::with_start_to_close_timeout(REFRESH_PAGE_TIMEOUT)
            .task_queue(task_queue)
            .heartbeat_timeout(REFRESH_PAGE_HEARTBEAT_TIMEOUT)
            .retry_policy(standard_retry_policy())
            .priority(build_priority(act_priority))
            .build();
        page_futures.push(Box::pin(ctx.execute_activity(
            ProviderActivities::refresh_page,
            page_input,
            opts,
        )));
    }

    let mut first_wait_armed = false;
    let mut first_wait_fired = *merges_run > 0 || merge_first_wait.is_zero();
    let mut timer_fut = None;

    while !page_futures.is_empty() {
        if let Some(timer) = timer_fut.take() {
            let next_res = {
                let mut next_fut = std::pin::pin!(poll_any_ready(&mut page_futures));
                match select(next_fut.as_mut(), timer).await {
                    Either::Left((maybe_page_res, uncompleted_timer)) => {
                        timer_fut = Some(uncompleted_timer);
                        Some(maybe_page_res)
                    }
                    Either::Right((_elapsed, _pending_next)) => None,
                }
            };
            if let Some(maybe_page_res) = next_res {
                if let Some(page_res) = maybe_page_res {
                    handle_page_completion(page_res, pages_refreshed, pages_failed, pending_merge);
                }
            } else {
                first_wait_fired = true;
                if *pending_merge || ctx.state(|state| state.merge_requested) {
                    *pending_merge = false;
                    ctx.state_mut(|state| state.merge_requested = false);
                    let merge_priority = ctx.state(|state| state.current_priority);
                    if let Err(err) = run_merge_and_episodes(
                        ctx,
                        input,
                        merge_priority,
                        merges_run,
                        episode_workflows_started,
                        any_synopses_changed,
                    )
                    .await
                    {
                        let _ = call_record_refresh(
                            ctx,
                            &input.anime_id,
                            RefreshOutcomeKind::Failed,
                            refresh_after_days,
                        )
                        .await;
                        return Err(err);
                    }
                }
            }
        } else if let Some(page_res) = poll_any_ready(&mut page_futures).await {
            handle_page_completion(page_res, pages_refreshed, pages_failed, pending_merge);
            if !first_wait_armed && !first_wait_fired && *merges_run == 0 {
                first_wait_armed = true;
                timer_fut = Some(Box::pin(ctx.timer(merge_first_wait)));
            }
        }
    }

    Ok(())
}

/// Polls a vector of pinned activity futures in deterministic order using the SDK waker.
async fn poll_any_ready<F: Future + Unpin>(pending: &mut Vec<F>) -> Option<F::Output> {
    std::future::poll_fn(|cx| {
        if pending.is_empty() {
            return Poll::Ready(None);
        }
        for idx in 0..pending.len() {
            if let Poll::Ready(output) = Pin::new(&mut pending[idx]).poll(cx) {
                pending.swap_remove(idx);
                return Poll::Ready(Some(output));
            }
        }
        Poll::Pending
    })
    .await
}

/// Updates counters and merge flag after one `refresh_page` activity finishes or fails.
fn handle_page_completion<E>(
    page_res: Result<RefreshPageOutput, E>,
    pages_refreshed: &mut usize,
    pages_failed: &mut usize,
    pending_merge: &mut bool,
) {
    *pages_refreshed += 1;
    match page_res {
        Ok(output) => {
            if output.has_failed {
                *pages_failed += 1;
            } else if output.has_changed {
                *pending_merge = true;
            }
        }
        Err(_) => {
            *pages_failed += 1;
        }
    }
}

/// Runs `merge_anime` and starts `RefreshEpisodeWorkflow` for created/updated episodes.
async fn run_merge_and_episodes(
    ctx: &WorkflowContext<RefreshAnimeWorkflow>,
    input: &RefreshAnimeInput,
    priority_key: u32,
    merges_run: &mut usize,
    episode_workflows_started: &mut usize,
    any_synopses_changed: &mut bool,
) -> WorkflowResult<()> {
    let merge_output: MergeAnimeOutput = ctx
        .execute_activity(
            UpdateServiceActivities::merge_anime,
            MergeAnimeInput {
                anime_id: input.anime_id.clone(),
                merge_version: input.merge_version,
            },
            ActivityOptions::with_start_to_close_timeout(MERGE_ANIME_TIMEOUT)
                .task_queue(UPDATE_SERVICE_TASK_QUEUE.to_owned())
                .retry_policy(standard_retry_policy())
                .priority(build_priority(priority_key))
                .build(),
        )
        .await?;

    *merges_run += 1;
    if merge_output.has_changed_synopses {
        *any_synopses_changed = true;
    }

    if !merge_output.changed_episode_ids.is_empty() {
        let ep_priority = ctx.state(|state| state.current_priority);
        let started = ctx
            .execute_activity(
                UpdateServiceActivities::start_entity_workflows,
                StartEntityWorkflowsInput {
                    anime_id: input.anime_id.clone(),
                    episode_ids: merge_output.changed_episode_ids,
                    start_synopsis: false,
                    trigger: input.trigger.clone(),
                    priority: ep_priority,
                },
                ActivityOptions::with_start_to_close_timeout(START_WORKFLOWS_TIMEOUT)
                    .task_queue(UPDATE_SERVICE_TASK_QUEUE.to_owned())
                    .retry_policy(standard_retry_policy())
                    .priority(build_priority(ep_priority))
                    .build(),
            )
            .await?;
        *episode_workflows_started += started;
    }

    Ok(())
}

/// Records the final refresh outcome via `UpdateServiceActivities::record_refresh`.
async fn call_record_refresh(
    ctx: &WorkflowContext<RefreshAnimeWorkflow>,
    anime_id: &str,
    outcome: RefreshOutcomeKind,
    refresh_after_days: i64,
) -> WorkflowResult<()> {
    let priority_key = ctx.state(|state| state.current_priority);
    ctx.execute_activity(
        UpdateServiceActivities::record_refresh,
        RecordRefreshInput {
            anime_id: anime_id.to_owned(),
            outcome,
            refresh_after_days,
        },
        ActivityOptions::with_start_to_close_timeout(RECORD_REFRESH_TIMEOUT)
            .task_queue(UPDATE_SERVICE_TASK_QUEUE.to_owned())
            .retry_policy(standard_retry_policy())
            .priority(build_priority(priority_key))
            .build(),
    )
    .await?;
    Ok(())
}
