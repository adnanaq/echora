//! Serializable workflow, signal, and activity payloads for `RefreshAnimeWorkflow`.

use serde::{Deserialize, Serialize};

/// Default days before a stored page or work counts as due for refresh.
pub const DEFAULT_REFRESH_AFTER_DAYS: i64 = 7;

/// Default minutes to wait after the first stored page before running the first merge.
pub const DEFAULT_MERGE_FIRST_WAIT_MINUTES: u64 = 10;

/// Maximum exponential backoff hours (7 days = 168 hours) for failed page retries.
pub const MAX_PAGE_RETRY_BACKOFF_HOURS: i64 = 168;

/// Builds the fixed Temporal workflow ID for a work refresh.
///
/// # Examples
///
/// ```
/// use temporal_workflows::build_refresh_anime_workflow_id;
///
/// assert_eq!(
///     build_refresh_anime_workflow_id("anime-123"),
///     "refresh-anime-anime-123"
/// );
/// ```
#[must_use]
pub fn build_refresh_anime_workflow_id(anime_id: &str) -> String {
    format!("refresh-anime-{anime_id}")
}

/// Builds the fixed Temporal workflow ID for an episode refresh.
///
/// # Examples
///
/// ```
/// use temporal_workflows::build_refresh_episode_workflow_id;
///
/// assert_eq!(
///     build_refresh_episode_workflow_id("ep-456"),
///     "refresh-episode-ep-456"
/// );
/// ```
#[must_use]
pub fn build_refresh_episode_workflow_id(episode_id: &str) -> String {
    format!("refresh-episode-{episode_id}")
}

/// Builds the fixed Temporal workflow ID for a work's synopsis workflow.
///
/// # Examples
///
/// ```
/// use temporal_workflows::build_write_synopsis_workflow_id;
///
/// assert_eq!(
///     build_write_synopsis_workflow_id("anime-123"),
///     "synopsis-anime-anime-123"
/// );
/// ```
#[must_use]
pub fn build_write_synopsis_workflow_id(anime_id: &str) -> String {
    format!("synopsis-anime-{anime_id}")
}

/// Claim-check reference identifying one provider page without its body.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Deserialize, Serialize)]
pub struct RefreshPageKey {
    /// Provider slug (for example `"mal"` or `"anilist"`).
    pub provider: String,
    /// Entity kind string (`"anime"`, `"episode"`, or `"character"`).
    pub entity_type: String,
    /// Canonical provider address or numeric ID.
    pub address: String,
}

/// Page metadata returned by the `list_pages` activity.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct EntityPageSummary {
    /// Provider page reference.
    pub page_key: RefreshPageKey,
    /// Whether this page is a partial source (never fetched or merged for this work).
    #[serde(default)]
    pub is_partial_source: bool,
    /// UTC epoch seconds when the page was last fetched, or `None` if never fetched.
    #[serde(default)]
    pub fetched_at_epoch_seconds: Option<i64>,
    /// Whether the page has been confirmed dead.
    #[serde(default)]
    pub is_dead: bool,
    /// UTC epoch seconds of the most recent failed fetch attempt.
    #[serde(default)]
    pub fetch_failed_at_epoch_seconds: Option<i64>,
    /// Consecutive failed fetch count since the last success.
    #[serde(default)]
    pub fetch_failures: i32,
}

/// Output returned by the `list_pages` activity.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize)]
pub struct ListEntityPagesOutput {
    /// All linked pages of the entity.
    #[serde(default)]
    pub pages: Vec<EntityPageSummary>,
    /// Stored `merge_version` of the entity (`None` when never merged).
    #[serde(default)]
    pub entity_merge_version: Option<i32>,
}

/// Input passed to the `refresh_page` activity.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct RefreshPageInput {
    /// Provider page to fetch and store.
    pub page_key: RefreshPageKey,
    /// Maximum acceptable cache age in seconds (`0` bypasses the HTTP cache).
    #[serde(default)]
    pub max_age_seconds: i64,
}

/// Claim-check output returned by the `refresh_page` activity.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct RefreshPageOutput {
    /// Provider page that was refreshed.
    pub page_key: RefreshPageKey,
    /// Whether the stored response hash changed.
    #[serde(default)]
    pub has_changed: bool,
    /// Whether every fetch attempt failed (recorded on the page; does not fail the refresh).
    #[serde(default)]
    pub has_failed: bool,
}

/// Input passed to the `merge_anime` activity.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct MergeAnimeInput {
    /// Target work UUID.
    pub anime_id: String,
    /// Current merge code version written on save.
    #[serde(default)]
    pub merge_version: i32,
}

/// Claim-check output returned by the `merge_anime` activity.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize)]
pub struct MergeAnimeOutput {
    /// Whether the saved record changed.
    #[serde(default)]
    pub has_changed: bool,
    /// Episode IDs created or updated by `SaveAnime` that need an episode refresh.
    #[serde(default)]
    pub changed_episode_ids: Vec<String>,
    /// Whether any provider synopsis changed compared to the current synopsis input hash.
    #[serde(default)]
    pub has_changed_synopses: bool,
}

/// Input passed to `start_entity_workflows` to launch top-level downstream workflows.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct StartEntityWorkflowsInput {
    /// Parent work UUID (used when starting `WriteSynopsisWorkflow`).
    pub anime_id: String,
    /// Episode UUIDs whose `RefreshEpisodeWorkflow` should be started.
    #[serde(default)]
    pub episode_ids: Vec<String>,
    /// Whether to start `WriteSynopsisWorkflow` for `anime_id`.
    #[serde(default)]
    pub start_synopsis: bool,
    /// `RefreshTrigger` search attribute value inherited from the starter.
    #[serde(default)]
    pub trigger: String,
    /// Temporal priority key (`1` = highest, `5` = lowest) inherited by started workflows.
    #[serde(default)]
    pub priority: u32,
}

/// Outcome recorded at the end of a refresh via `record_refresh`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub enum RefreshOutcomeKind {
    /// Every due page succeeded or no pages were due.
    Succeeded,
    /// Merge completed, but one or more source pages failed every attempt.
    PagesFailed,
    /// Merge or save failed.
    Failed,
}

/// Input passed to the `record_refresh` activity.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct RecordRefreshInput {
    /// Target work UUID.
    pub anime_id: String,
    /// Final refresh outcome.
    pub outcome: RefreshOutcomeKind,
    /// `REFRESH_AFTER_DAYS` setting from `update_service`.
    #[serde(default)]
    pub refresh_after_days: i64,
}

/// Signal payload requesting specific provider pages (`fetch_requested`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize)]
pub struct FetchRequestedSignal {
    /// Provider slugs requested to refresh (empty means all due source pages).
    #[serde(default)]
    pub providers: Vec<String>,
    /// Maximum acceptable cache age in seconds (`0` always fetches).
    #[serde(default)]
    pub max_age_seconds: i64,
    /// Priority key requested by the caller (`1` = highest).
    #[serde(default)]
    pub priority: u32,
}

/// Input passed when starting `RefreshAnimeWorkflow`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize)]
pub struct RefreshAnimeInput {
    /// Target work UUID.
    pub anime_id: String,
    /// `RefreshTrigger` search attribute value (`user`, `episode`, `sweep`, `seed`, `remerge`).
    #[serde(default)]
    pub trigger: String,
    /// Initial priority key (`1..=5`, `0` uses default `3`).
    #[serde(default)]
    pub priority: u32,
    /// `REFRESH_AFTER_DAYS` setting (`0` uses [`DEFAULT_REFRESH_AFTER_DAYS`]).
    #[serde(default)]
    pub refresh_after_days: i64,
    /// Current `MERGE_VERSION` from settings; triggers a re-merge when stored version is older.
    #[serde(default)]
    pub merge_version: i32,
    /// Seconds to wait after the first stored page before running the first merge
    /// (`None` uses [`DEFAULT_MERGE_FIRST_WAIT_MINUTES`] * 60).
    #[serde(default)]
    pub merge_first_wait_seconds: Option<u64>,
    /// Whether the work's pages were recently relinked and require a merge even without fetches.
    #[serde(default)]
    pub pages_relinked: bool,
}

/// Final summary returned by `RefreshAnimeWorkflow::run`.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct RefreshAnimeOutcome {
    /// Target work UUID.
    pub anime_id: String,
    /// Number of `refresh_page` calls completed.
    pub pages_refreshed: usize,
    /// Number of `refresh_page` calls that ended in page failure.
    pub pages_failed: usize,
    /// Number of `merge_anime` executions performed.
    pub merges_run: usize,
    /// Number of episode refreshes requested via `start_entity_workflows`.
    pub episode_workflows_started: usize,
    /// Whether `WriteSynopsisWorkflow` was started after all pages settled.
    pub synopsis_workflow_started: bool,
    /// Final outcome recorded via `record_refresh`.
    pub final_outcome: RefreshOutcomeKind,
}
