//! Serializable workflow, signal, and activity payloads for `RefreshAnimeWorkflow`.

use std::fmt;
use std::num::NonZeroU32;
use std::str::FromStr;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::Provider;

/// Seconds in one day, shared across due-page calculations and gRPC duration conversions.
pub const SECONDS_PER_DAY: i64 = 86_400;

/// Default days before a stored page or work counts as due for refresh.
///
/// Matches `REFRESH_AFTER_DAYS` in `docs/architecture/update_service.md`.
pub const DEFAULT_REFRESH_AFTER_DAYS: i64 = 7;

/// Default `MERGE_VERSION` written on entity saves when unset.
///
/// Matches `MERGE_VERSION` in `docs/architecture/update_service.md`.
pub const DEFAULT_MERGE_VERSION: i32 = 1;

/// Default minutes to wait after the first stored page before running the first merge.
///
/// Matches `MERGE_FIRST_WAIT_MINUTES` in `docs/architecture/update_service.md`, so fast
/// providers finish together while slower queues do not delay initial visibility.
pub const DEFAULT_MERGE_FIRST_WAIT_MINUTES: u64 = 10;

/// Default seconds to coalesce late-arriving pages after the first merge has run.
///
/// Keeps pages finishing close together grouped into one follow-up merge without making
/// a fast late page wait indefinitely for a slow provider.
pub const DEFAULT_MERGE_COALESCE_WAIT_SECONDS: u64 = 5;

/// Maximum exponential backoff hours (7 days = 168 hours) for failed page retries.
///
/// Matches the 1-week cap in `docs/architecture/update_service.md` and `postgres_db`.
pub const MAX_PAGE_RETRY_BACKOFF_HOURS: i64 = 168;

mod duration_seconds {
    use std::time::Duration;

    use serde::{Deserialize as _, Deserializer, Serializer};

    pub(crate) fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u64(duration.as_secs())
    }

    pub(crate) fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let seconds = i64::deserialize(deserializer)?;
        let non_negative = u64::try_from(seconds.max(0)).unwrap_or(0);
        Ok(Duration::from_secs(non_negative))
    }
}

/// Error returned when parsing a provider, entity type, or trigger slug fails.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseSlugError {
    expected: &'static str,
    input: String,
}

impl ParseSlugError {
    /// Creates a parse error describing `expected` and the rejected `input`.
    #[must_use]
    pub(crate) fn new(expected: &'static str, input: impl Into<String>) -> Self {
        Self {
            expected,
            input: input.into(),
        }
    }
}

impl fmt::Display for ParseSlugError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "unknown {}: {:?}", self.expected, self.input)
    }
}

impl std::error::Error for ParseSlugError {}

/// Type of catalogue entity tracked by refresh workflows and stored provider pages.
///
/// Matches `shared_proto.v1.EntityType` and `postgres_db::EntityType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Deserialize, Serialize)]
#[serde(try_from = "String", into = "&'static str")]
pub enum EntityType {
    /// An anime work.
    Anime,
    /// An episode belonging to a work.
    Episode,
    /// A character appearing in works.
    Character,
}

impl EntityType {
    /// Every supported catalogue entity type, each listed once.
    pub const ALL: [Self; 3] = [Self::Anime, Self::Episode, Self::Character];

    /// Returns the lowercase wire slug for this entity type.
    ///
    /// # Examples
    ///
    /// ```
    /// use temporal_workflows::EntityType;
    ///
    /// assert_eq!(EntityType::Anime.as_str(), "anime");
    /// ```
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Anime => "anime",
            Self::Episode => "episode",
            Self::Character => "character",
        }
    }
}

impl From<EntityType> for &'static str {
    fn from(value: EntityType) -> Self {
        value.as_str()
    }
}

impl TryFrom<String> for EntityType {
    type Error = ParseSlugError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        value.parse()
    }
}

impl fmt::Display for EntityType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for EntityType {
    type Err = ParseSlugError;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|entity_type| entity_type.as_str() == input)
            .ok_or_else(|| ParseSlugError::new("entity type", input))
    }
}

/// Origin that started a refresh workflow, mapped to priority and fairness keys.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Deserialize, Serialize)]
#[serde(try_from = "String", into = "&'static str")]
pub enum RefreshTrigger {
    /// Work opened by a user (priority 1, fairness `user`).
    User,
    /// Episode tracking around expected air time (priority 2, fairness `episode`).
    Episode,
    /// Requested merge from an approval, page move, or new AI text (priority 2, fairness `review`).
    Review,
    /// Weekly catalogue sweep or daily score refresh (priority 3, fairness `sweep`).
    #[default]
    Sweep,
    /// Cleaned seed release load (priority 3, fairness `seed`).
    Seed,
    /// Re-merge after raising `MERGE_VERSION` (priority 4, fairness `remerge`).
    Remerge,
}

impl RefreshTrigger {
    /// Every supported refresh trigger, each listed once.
    pub const ALL: [Self; 6] = [
        Self::User,
        Self::Episode,
        Self::Review,
        Self::Sweep,
        Self::Seed,
        Self::Remerge,
    ];

    /// Returns the keyword value stored in the `RefreshTrigger` search attribute.
    ///
    /// # Examples
    ///
    /// ```
    /// use temporal_workflows::RefreshTrigger;
    ///
    /// assert_eq!(RefreshTrigger::User.as_str(), "user");
    /// assert_eq!(RefreshTrigger::Remerge.as_str(), "remerge");
    /// ```
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Episode => "episode",
            Self::Review => "review",
            Self::Sweep => "sweep",
            Self::Seed => "seed",
            Self::Remerge => "remerge",
        }
    }

    /// Returns the default Temporal priority key (`1` = highest, `4` = lowest used).
    ///
    /// # Examples
    ///
    /// ```
    /// use temporal_workflows::RefreshTrigger;
    ///
    /// assert_eq!(RefreshTrigger::User.default_priority_key(), 1);
    /// assert_eq!(RefreshTrigger::Sweep.default_priority_key(), 3);
    /// assert_eq!(RefreshTrigger::Remerge.default_priority_key(), 4);
    /// ```
    #[must_use]
    pub const fn default_priority_key(self) -> u32 {
        match self {
            Self::User => 1,
            Self::Episode | Self::Review => 2,
            Self::Sweep | Self::Seed => 3,
            Self::Remerge => 4,
        }
    }

    /// Returns the Temporal task-queue fairness key for this trigger.
    ///
    /// # Examples
    ///
    /// ```
    /// use temporal_workflows::RefreshTrigger;
    ///
    /// assert_eq!(RefreshTrigger::Seed.fairness_key(), "seed");
    /// ```
    #[must_use]
    pub const fn fairness_key(self) -> &'static str {
        self.as_str()
    }
}

impl From<RefreshTrigger> for &'static str {
    fn from(value: RefreshTrigger) -> Self {
        value.as_str()
    }
}

impl TryFrom<String> for RefreshTrigger {
    type Error = ParseSlugError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        value.parse()
    }
}

impl fmt::Display for RefreshTrigger {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for RefreshTrigger {
    type Err = ParseSlugError;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|trigger| trigger.as_str() == input)
            .ok_or_else(|| ParseSlugError::new("refresh trigger", input))
    }
}

/// Worker-level refresh policy settings applied deterministically via `list_pages`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub struct RefreshSettings {
    /// Age in days before a stored page or work counts as due (`REFRESH_AFTER_DAYS`).
    pub refresh_after_days: i64,
    /// Current mapping and merge code version (`MERGE_VERSION`).
    pub merge_version: i32,
    /// Seconds to wait after the first stored page before running the first merge.
    pub merge_first_wait_seconds: u64,
    /// Seconds to coalesce late-arriving pages after the first merge has run.
    pub merge_coalesce_wait_seconds: u64,
    #[cfg(feature = "test-util")]
    #[serde(default)]
    pub(crate) refresh_page_timeout_seconds: Option<u64>,
}

impl RefreshSettings {
    /// Creates worker refresh settings from explicit day, version, and merge-wait values.
    #[must_use]
    pub const fn new(
        refresh_after_days: i64,
        merge_version: i32,
        merge_first_wait_seconds: u64,
        merge_coalesce_wait_seconds: u64,
    ) -> Self {
        Self {
            refresh_after_days,
            merge_version,
            merge_first_wait_seconds,
            merge_coalesce_wait_seconds,
            #[cfg(feature = "test-util")]
            refresh_page_timeout_seconds: None,
        }
    }

    /// Overrides the `refresh_page` start-to-close timeout for fast timeout tests.
    #[cfg(feature = "test-util")]
    #[must_use]
    pub const fn with_refresh_page_timeout(mut self, timeout_seconds: u64) -> Self {
        self.refresh_page_timeout_seconds = Some(timeout_seconds);
        self
    }
}

impl Default for RefreshSettings {
    fn default() -> Self {
        Self::new(
            DEFAULT_REFRESH_AFTER_DAYS,
            DEFAULT_MERGE_VERSION,
            DEFAULT_MERGE_FIRST_WAIT_MINUTES * 60,
            DEFAULT_MERGE_COALESCE_WAIT_SECONDS,
        )
    }
}

/// Builds the fixed Temporal workflow ID for a work refresh.
///
/// # Examples
///
/// ```
/// use temporal_workflows::build_refresh_anime_workflow_id;
/// use uuid::Uuid;
///
/// let anime_id = Uuid::nil();
/// assert_eq!(
///     build_refresh_anime_workflow_id(anime_id),
///     "refresh-anime-00000000-0000-0000-0000-000000000000"
/// );
/// ```
#[must_use]
pub fn build_refresh_anime_workflow_id(anime_id: Uuid) -> String {
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
/// use uuid::Uuid;
///
/// assert_eq!(
///     build_write_synopsis_workflow_id(Uuid::nil()),
///     "synopsis-anime-00000000-0000-0000-0000-000000000000"
/// );
/// ```
#[must_use]
pub fn build_write_synopsis_workflow_id(anime_id: Uuid) -> String {
    format!("synopsis-anime-{anime_id}")
}

/// Claim-check reference identifying one provider page without its body.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Deserialize, Serialize)]
pub struct RefreshPageKey {
    /// Metadata provider owning the page.
    pub provider: Provider,
    /// Catalogue entity type (`anime`, `episode`, or `character`).
    pub entity_type: EntityType,
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
    /// Worker refresh policy settings recorded deterministically in workflow history.
    #[serde(default)]
    pub refresh_settings: RefreshSettings,
}

/// Input passed to the `refresh_page` activity.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct RefreshPageInput {
    /// Provider page to fetch and store.
    pub page_key: RefreshPageKey,
    /// Maximum acceptable cache age (`Duration::ZERO` bypasses the HTTP cache).
    #[serde(default, rename = "max_age_seconds", with = "duration_seconds")]
    pub max_age: Duration,
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

/// Claim-check output returned by the `merge_anime` activity.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize)]
pub struct MergeAnimeOutput {
    /// Whether the saved record changed.
    #[serde(default)]
    pub has_changed: bool,
    /// Episode IDs created or updated by `SaveAnime` that need an episode refresh.
    #[serde(default)]
    pub changed_episode_ids: Vec<String>,
    /// Whether provider synopses changed compared to the current synopsis input hash.
    #[serde(default)]
    pub has_changed_synopses: bool,
}

/// Input passed to `start_entity_workflows` to launch top-level downstream workflows.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct StartEntityWorkflowsInput {
    /// Parent work UUID (used when starting `WriteSynopsisWorkflow`).
    pub anime_id: Uuid,
    /// Episode UUIDs whose `RefreshEpisodeWorkflow` should be started.
    #[serde(default)]
    pub episode_ids: Vec<String>,
    /// Whether to start `WriteSynopsisWorkflow` for `anime_id`.
    #[serde(default)]
    pub start_synopsis: bool,
    /// `RefreshTrigger` search attribute value inherited from the starter.
    #[serde(default)]
    pub trigger: RefreshTrigger,
    /// Temporal priority key (`1` = highest, `5` = lowest) inherited by started workflows.
    #[serde(default)]
    pub priority: Option<NonZeroU32>,
}

/// Outcome recorded at the end of a refresh via `record_refresh`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub enum RefreshOutcome {
    /// Every due page succeeded or no pages were due.
    Succeeded,
    /// Merge completed, but one or more source pages failed every attempt.
    PagesFailed,
    /// Merge, save, or refresh orchestration failed.
    Failed,
}

/// Input passed to the `record_refresh` activity.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct RecordRefreshInput {
    /// Target work UUID.
    pub anime_id: Uuid,
    /// Final refresh outcome.
    pub outcome: RefreshOutcome,
}

/// Signal payload requesting specific provider pages (`fetch_requested`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize)]
pub struct FetchRequestedSignal {
    /// Providers requested to refresh (empty means all non-dead source pages).
    #[serde(default)]
    pub providers: Vec<Provider>,
    /// Maximum acceptable cache age (`Duration::ZERO` always fetches).
    #[serde(default, rename = "max_age_seconds", with = "duration_seconds")]
    pub max_age: Duration,
    /// Priority key requested by the caller (`1` = highest, `None` keeps current priority).
    #[serde(default)]
    pub priority: Option<NonZeroU32>,
}

/// Input passed when starting `RefreshAnimeWorkflow`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize)]
pub struct RefreshAnimeInput {
    /// Target work UUID.
    pub anime_id: Uuid,
    /// `RefreshTrigger` search attribute value (`user`, `episode`, `review`, `sweep`, `seed`, `remerge`).
    #[serde(default)]
    pub trigger: RefreshTrigger,
    /// Initial priority key (`1..=5`, `None` uses `trigger.default_priority_key()`).
    #[serde(default)]
    pub priority: Option<NonZeroU32>,
    /// Whether the work's pages were recently relinked and require a merge even without fetches.
    #[serde(default)]
    pub pages_relinked: bool,
}

/// Final summary returned by `RefreshAnimeWorkflow::run`.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct RefreshAnimeOutcome {
    /// Target work UUID.
    pub anime_id: Uuid,
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
    pub final_outcome: RefreshOutcome,
}
