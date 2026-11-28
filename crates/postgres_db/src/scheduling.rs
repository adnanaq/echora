//! Scheduling queries: due entities, linked pages, refresh outcomes and upcoming episodes.
//!
//! The Update Service finds its work through these queries:
//!
//! - the weekly sweep pages [`list_due_anime`], [`list_due_episodes`] and
//!   [`list_due_characters`] from a fixed cutoff, and reads [`list_requested_merges`] first;
//! - a refresh reads [`list_entity_pages`] and ends with [`record_refresh`];
//! - episode tracking reads [`list_upcoming_episodes`], the daily score refresh
//!   [`list_score_anime`], and the budgeted fetcher [`list_due_pages`];
//! - the vector service's comparison job pages [`list_entity_versions`].
//!
//! The due queries, [`list_entity_pages`] and [`list_requested_merges`] read the tables, since an
//! entity that was never merged is due. The other reads go through the `*_visible` views, which
//! hold only merged, undeleted rows.

use chrono::{DateTime, Duration, Utc};
use sqlx::PgPool;
use sqlx::postgres::types::PgInterval;
use uuid::Uuid;

use crate::{EntityType, Error, PageKey, Provider, begin_read_snapshot};

/// Longest wait, in hours, before a failed page or a failed refresh is tried again.
///
/// One week, so a page or entity that keeps failing is retried weekly instead of on every
/// sweep. The plan sets this limit for both the page retry and the refresh backoff.
const MAX_BACKOFF_HOURS: i32 = 168;

/// Largest exponent used in the `2^failures` backoff before the week limit applies.
///
/// `2^10` hours already passes [`MAX_BACKOFF_HOURS`], so a larger failure count changes
/// nothing, and capping the exponent keeps `power(2, failures)` inside a 32-bit integer.
const MAX_BACKOFF_EXPONENT: i32 = 10;

/// How a refresh ended, passed to [`record_refresh`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefreshOutcome {
    /// Every page was fetched or was not due, whether or not the merge changed anything.
    Succeeded,
    /// The merge ran, but some pages failed every fetch attempt.
    PagesFailed,
    /// The merge or the save failed.
    Failed,
}

/// Linked provider pages and stored merge version of an entity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EntityPagesRecord {
    /// Linked provider pages with their last fetch state.
    pub pages: Vec<EntityPageRecord>,
    /// Merge code version that last saved the entity, or `None` when never merged.
    pub merge_version: Option<i32>,
}

/// A provider page linked to an entity, with its last fetch state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EntityPageRecord {
    /// The page's provider, entity type and address.
    pub page_key: PageKey,
    /// Whether the page is a partial source, linked for reference and never merged.
    pub is_partial_source: bool,
    /// When the page was last fetched, or `None` when it never was.
    pub fetched_at: Option<DateTime<Utc>>,
    /// Whether the page was confirmed dead.
    pub is_dead: bool,
    /// When the last failed fetch happened, or `None` after a successful fetch.
    pub fetch_failed_at: Option<DateTime<Utc>>,
    /// Failed fetches since the last successful one.
    pub fetch_failures: i32,
}

/// Largest `older_after_years` a due query accepts.
///
/// No anime first aired more than a thousand years ago, so a larger value means the same as
/// this one. Bounding it keeps `extract(year FROM now())::int - older_after_years` inside
/// PostgreSQL's integer range, so a caller's mistake is refused rather than failing the query.
pub const MAX_OLDER_AFTER_YEARS: i32 = 1_000;

/// Largest day window `ListUpcomingEpisodes` and `ListScoreAnime` accept.
///
/// A window of a hundred years covers every schedule and lookback the services use. Bounding
/// it keeps `make_interval(days => ...)` added to `now()` inside PostgreSQL's timestamp range.
pub const MAX_WINDOW_DAYS: i32 = 36_500;

/// Parameters of one page of a due query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DueEntitiesQuery {
    /// The sweep's fixed cutoff, compared with `next_refresh_at`.
    pub due_before: DateTime<Utc>,
    /// Entities first aired more than this many years ago refresh only when opened.
    pub older_after_years: i32,
    /// Most rows to return.
    pub page_size: i32,
    /// Keyset cursor: `next_refresh_at` of the previous page's last row.
    pub after_next_refresh_at: Option<DateTime<Utc>>,
    /// Keyset cursor: ID of the previous page's last row.
    pub after_id: Option<Uuid>,
    /// The current merge code version; entities merged by older code are always due.
    pub merge_version: i32,
}

/// A due entity and its keyset cursor value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DueEntityRecord {
    /// The entity's ID.
    pub entity_id: Uuid,
    /// When the entity is next due, or `None` when it is due now.
    pub next_refresh_at: Option<DateTime<Utc>>,
}

/// An episode expected to air within the tracking window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UpcomingEpisodeRecord {
    /// The work's ID.
    pub anime_id: Uuid,
    /// The episode's ID, or `None` when no episode row exists for the announced number yet.
    pub episode_id: Option<Uuid>,
    /// The episode number.
    pub episode_number: i32,
    /// When the episode is expected to air.
    pub expected_air_at: DateTime<Utc>,
}

/// A visible entity's ID and version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EntityVersionRecord {
    /// The entity's ID.
    pub entity_id: Uuid,
    /// The entity's version.
    pub version: i64,
}

/// A budgeted provider's due page and the work it is merged into.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DuePageRecord {
    /// The page's provider, entity type and address.
    pub page_key: PageKey,
    /// The work whose merge reads the page.
    pub anime_id: Uuid,
}

/// Lists an entity's linked provider pages, fetch state, and stored merge version.
///
/// Reads the link tables, not the visible views, so an entity that was never merged can be
/// refreshed. The entity check and the page rows come from one read-only snapshot, so an
/// entity deleted between them is not reported as present with no pages. A work lists its
/// source pages first, then its partial-source pages; episodes and characters have source
/// pages only. Each group is ordered by provider and address.
///
/// # Examples
///
/// ```no_run
/// # use uuid::Uuid;
/// # async fn example(pool: &sqlx::PgPool, anime_id: Uuid) -> Result<(), postgres_db::Error> {
/// let entity_pages =
///     postgres_db::list_entity_pages(pool, postgres_db::EntityType::Anime, anime_id).await?;
/// let due_pages = entity_pages
///     .pages
///     .iter()
///     .filter(|page| !page.is_partial_source && !page.is_dead);
/// assert!(due_pages.count() <= entity_pages.pages.len());
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns [`Error::not_found`] when the entity does not exist or is deleted, an
/// invalid-argument error when a stored page key no longer passes validation, or a database
/// error when a query fails.
pub async fn list_entity_pages(
    pool: &PgPool,
    entity_type: EntityType,
    entity_id: Uuid,
) -> Result<EntityPagesRecord, Error> {
    let mut tx = begin_read_snapshot(pool).await?;
    let entity_row = sqlx::query!(
        r#"
        SELECT merge_version
        FROM anime
        WHERE $2 = 'anime' AND id = $1 AND deleted_at IS NULL
        UNION ALL
        SELECT merge_version
        FROM episode
        WHERE $2 = 'episode' AND id = $1 AND deleted_at IS NULL
        UNION ALL
        SELECT merge_version
        FROM character
        WHERE $2 = 'character' AND id = $1 AND deleted_at IS NULL
        "#,
        entity_id,
        entity_type.as_str()
    )
    .fetch_optional(&mut *tx)
    .await?;
    let Some(entity_row) = entity_row else {
        return Err(Error::not_found());
    };

    let page_rows = sqlx::query!(
        r#"
        SELECT page.provider AS "provider!: Provider",
               page.entity_type AS "entity_type!: EntityType",
               page.address AS "address!",
               link.is_partial_source AS "is_partial_source!",
               page.fetched_at,
               page.is_dead,
               page.fetch_failed_at,
               page.fetch_failures
        FROM (
            SELECT provider, entity_type, address, false AS is_partial_source
            FROM anime_source WHERE $2 = 'anime' AND anime_id = $1
            UNION ALL
            SELECT provider, entity_type, address, true
            FROM anime_partial_source WHERE $2 = 'anime' AND anime_id = $1
            UNION ALL
            SELECT provider, entity_type, address, false
            FROM episode_source WHERE $2 = 'episode' AND episode_id = $1
            UNION ALL
            SELECT provider, entity_type, address, false
            FROM character_source WHERE $2 = 'character' AND character_id = $1
        ) link
        JOIN provider_page page USING (provider, entity_type, address)
        ORDER BY link.is_partial_source, page.provider, page.address
        "#,
        entity_id,
        entity_type.as_str()
    )
    .fetch_all(&mut *tx)
    .await?;

    let pages = page_rows
        .into_iter()
        .map(|page_row| {
            Ok(EntityPageRecord {
                page_key: PageKey::new(page_row.provider, page_row.entity_type, page_row.address)?,
                is_partial_source: page_row.is_partial_source,
                fetched_at: page_row.fetched_at,
                is_dead: page_row.is_dead,
                fetch_failed_at: page_row.fetch_failed_at,
                fetch_failures: page_row.fetch_failures,
            })
        })
        .collect::<Result<Vec<_>, Error>>()?;

    Ok(EntityPagesRecord {
        pages,
        merge_version: entity_row.merge_version,
    })
}

/// Records how a refresh ended and sets the entity's `next_refresh_at`.
///
/// Each outcome writes one statement, timed by the transaction's `now()`:
///
/// - `Succeeded`: `refreshed_at = now()`, `refresh_failures = 0`, and
///   `next_refresh_at = now() + refresh_after`.
/// - `PagesFailed`: the same, except `next_refresh_at` is the earliest time a failed source
///   page may be fetched again, `2^fetch_failures` hours after its failure and at most a week.
///   Dead and partial-source pages are left out. It is never later than
///   `now() + refresh_after`, so a page's long backoff does not delay the entity's healthy
///   pages past their normal refresh, and with no failed page left it is that time.
/// - `Failed`: `refresh_failed_at = now()`, one more `refresh_failures`, and
///   `next_refresh_at = now() + 2^refresh_failures` hours, at most a week.
///
/// `refresh_after` is the Update Service's `REFRESH_AFTER_DAYS`.
///
/// # Examples
///
/// ```no_run
/// # use chrono::Duration;
/// # use uuid::Uuid;
/// # async fn example(pool: &sqlx::PgPool, anime_id: Uuid) -> Result<(), postgres_db::Error> {
/// postgres_db::record_refresh(
///     pool,
///     postgres_db::EntityType::Anime,
///     anime_id,
///     postgres_db::RefreshOutcome::Succeeded,
///     Duration::days(7),
/// )
/// .await?;
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns an invalid-argument error when `refresh_after` is negative, [`Error::not_found`]
/// when the entity does not exist or is deleted, or a database error when the update fails.
pub async fn record_refresh(
    pool: &PgPool,
    entity_type: EntityType,
    entity_id: Uuid,
    outcome: RefreshOutcome,
    refresh_after: Duration,
) -> Result<(), Error> {
    if refresh_after < Duration::zero() {
        return Err(Error::invalid_argument(
            "refresh_after must not be negative",
        ));
    }
    let refresh_interval = PgInterval::try_from(refresh_after).map_err(|conversion_error| {
        Error::invalid_argument(format!(
            "refresh_after is not a PostgreSQL interval: {conversion_error}"
        ))
    })?;

    let rows_affected = match outcome {
        RefreshOutcome::Succeeded | RefreshOutcome::PagesFailed => {
            let has_failed_pages = outcome == RefreshOutcome::PagesFailed;
            record_finished_refresh(
                pool,
                entity_type,
                entity_id,
                has_failed_pages,
                refresh_interval,
            )
            .await?
        }
        RefreshOutcome::Failed => record_failed_refresh(pool, entity_type, entity_id).await?,
    };

    if rows_affected == 0 {
        return Err(Error::not_found());
    }
    Ok(())
}

/// Writes a refresh that merged, with or without failed pages, and returns the rows updated.
async fn record_finished_refresh(
    pool: &PgPool,
    entity_type: EntityType,
    entity_id: Uuid,
    has_failed_pages: bool,
    refresh_interval: PgInterval,
) -> Result<u64, Error> {
    let update_outcome = match entity_type {
        EntityType::Anime => {
            sqlx::query!(
                r#"
                UPDATE anime
                SET refreshed_at = now(),
                    refresh_failures = 0,
                    next_refresh_at = CASE
                        WHEN $2 THEN least(coalesce(
                            (SELECT min(page.fetch_failed_at + make_interval(
                                        hours => least(power(2, least(page.fetch_failures, $4))::int, $5)))
                             FROM anime_source link
                             JOIN provider_page page USING (provider, entity_type, address)
                             WHERE link.anime_id = anime.id
                               AND NOT page.is_dead
                               AND page.fetch_failed_at IS NOT NULL),
                            now() + $3), now() + $3)
                        ELSE now() + $3
                    END
                WHERE id = $1 AND deleted_at IS NULL
                "#,
                entity_id,
                has_failed_pages,
                refresh_interval,
                MAX_BACKOFF_EXPONENT,
                MAX_BACKOFF_HOURS
            )
            .execute(pool)
            .await?
        }
        EntityType::Episode => {
            sqlx::query!(
                r#"
                UPDATE episode
                SET refreshed_at = now(),
                    refresh_failures = 0,
                    next_refresh_at = CASE
                        WHEN $2 THEN least(coalesce(
                            (SELECT min(page.fetch_failed_at + make_interval(
                                        hours => least(power(2, least(page.fetch_failures, $4))::int, $5)))
                             FROM episode_source link
                             JOIN provider_page page USING (provider, entity_type, address)
                             WHERE link.episode_id = episode.id
                               AND NOT page.is_dead
                               AND page.fetch_failed_at IS NOT NULL),
                            now() + $3), now() + $3)
                        ELSE now() + $3
                    END
                WHERE id = $1 AND deleted_at IS NULL
                "#,
                entity_id,
                has_failed_pages,
                refresh_interval,
                MAX_BACKOFF_EXPONENT,
                MAX_BACKOFF_HOURS
            )
            .execute(pool)
            .await?
        }
        EntityType::Character => {
            sqlx::query!(
                r#"
                UPDATE character
                SET refreshed_at = now(),
                    refresh_failures = 0,
                    next_refresh_at = CASE
                        WHEN $2 THEN least(coalesce(
                            (SELECT min(page.fetch_failed_at + make_interval(
                                        hours => least(power(2, least(page.fetch_failures, $4))::int, $5)))
                             FROM character_source link
                             JOIN provider_page page USING (provider, entity_type, address)
                             WHERE link.character_id = character.id
                               AND NOT page.is_dead
                               AND page.fetch_failed_at IS NOT NULL),
                            now() + $3), now() + $3)
                        ELSE now() + $3
                    END
                WHERE id = $1 AND deleted_at IS NULL
                "#,
                entity_id,
                has_failed_pages,
                refresh_interval,
                MAX_BACKOFF_EXPONENT,
                MAX_BACKOFF_HOURS
            )
            .execute(pool)
            .await?
        }
    };
    Ok(update_outcome.rows_affected())
}

/// Writes a failed refresh with its exponential backoff and returns the rows updated.
async fn record_failed_refresh(
    pool: &PgPool,
    entity_type: EntityType,
    entity_id: Uuid,
) -> Result<u64, Error> {
    let update_outcome = match entity_type {
        EntityType::Anime => {
            sqlx::query!(
                r#"
                UPDATE anime
                SET refresh_failed_at = now(),
                    refresh_failures = refresh_failures + 1,
                    next_refresh_at = now() + make_interval(
                        hours => least(power(2, least(refresh_failures + 1, $2))::int, $3))
                WHERE id = $1 AND deleted_at IS NULL
                "#,
                entity_id,
                MAX_BACKOFF_EXPONENT,
                MAX_BACKOFF_HOURS
            )
            .execute(pool)
            .await?
        }
        EntityType::Episode => {
            sqlx::query!(
                r#"
                UPDATE episode
                SET refresh_failed_at = now(),
                    refresh_failures = refresh_failures + 1,
                    next_refresh_at = now() + make_interval(
                        hours => least(power(2, least(refresh_failures + 1, $2))::int, $3))
                WHERE id = $1 AND deleted_at IS NULL
                "#,
                entity_id,
                MAX_BACKOFF_EXPONENT,
                MAX_BACKOFF_HOURS
            )
            .execute(pool)
            .await?
        }
        EntityType::Character => {
            sqlx::query!(
                r#"
                UPDATE character
                SET refresh_failed_at = now(),
                    refresh_failures = refresh_failures + 1,
                    next_refresh_at = now() + make_interval(
                        hours => least(power(2, least(refresh_failures + 1, $2))::int, $3))
                WHERE id = $1 AND deleted_at IS NULL
                "#,
                entity_id,
                MAX_BACKOFF_EXPONENT,
                MAX_BACKOFF_HOURS
            )
            .execute(pool)
            .await?
        }
    };
    Ok(update_outcome.rows_affected())
}

/// Checks the parameters every due query shares.
fn validate_due_query(query: &DueEntitiesQuery) -> Result<(), Error> {
    if query.page_size <= 0 {
        return Err(Error::invalid_argument("page_size must be positive"));
    }
    if !(0..=MAX_OLDER_AFTER_YEARS).contains(&query.older_after_years) {
        return Err(Error::invalid_argument(format!(
            "older_after_years must be between 0 and {MAX_OLDER_AFTER_YEARS}"
        )));
    }
    Ok(())
}

/// Lists works due for refresh, paged by `(next_refresh_at, id)` from a fixed cutoff.
///
/// A work is due when its `next_refresh_at` is before `due_before` and it is not older than
/// `older_after_years`, or when older merge code merged it, whatever its age. A work that was
/// never merged has no `next_refresh_at` and comes first. Deleted works are left out.
///
/// # Examples
///
/// ```no_run
/// # use chrono::Utc;
/// # async fn example(pool: &sqlx::PgPool) -> Result<(), postgres_db::Error> {
/// let query = postgres_db::DueEntitiesQuery {
///     due_before: Utc::now(),
///     older_after_years: 30,
///     page_size: 100,
///     after_next_refresh_at: None,
///     after_id: None,
///     merge_version: 1,
/// };
/// let first_page = postgres_db::list_due_anime(pool, &query).await?;
/// if let Some(last_row) = first_page.last() {
///     let next_query = postgres_db::DueEntitiesQuery {
///         after_next_refresh_at: last_row.next_refresh_at,
///         after_id: Some(last_row.entity_id),
///         ..query
///     };
///     let next_page = postgres_db::list_due_anime(pool, &next_query).await?;
///     assert!(next_page.iter().all(|row| row.entity_id != last_row.entity_id));
/// }
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns an invalid-argument error when `page_size` is not positive or `older_after_years`
/// is negative or above [`MAX_OLDER_AFTER_YEARS`], or a database error when the query fails.
pub async fn list_due_anime(
    pool: &PgPool,
    query: &DueEntitiesQuery,
) -> Result<Vec<DueEntityRecord>, Error> {
    validate_due_query(query)?;
    let due_records = sqlx::query_as!(
        DueEntityRecord,
        r#"
        SELECT id AS entity_id, next_refresh_at
        FROM anime
        WHERE deleted_at IS NULL
          AND (
                (coalesce(next_refresh_at, '-infinity'::timestamptz) < $1
                 AND (year IS NULL OR year > extract(year FROM now())::int - $2))
             OR coalesce(merge_version, 0) < $6
          )
          AND (coalesce(next_refresh_at, '-infinity'::timestamptz), id)
              > (coalesce($3, '-infinity'::timestamptz), coalesce($4, '00000000-0000-0000-0000-000000000000'::uuid))
        ORDER BY coalesce(next_refresh_at, '-infinity'::timestamptz), id
        LIMIT $5
        "#,
        query.due_before,
        query.older_after_years,
        query.after_next_refresh_at,
        query.after_id,
        i64::from(query.page_size),
        query.merge_version
    )
    .fetch_all(pool)
    .await?;
    Ok(due_records)
}

/// Lists due episodes of ongoing and upcoming works, paged like [`list_due_anime`].
///
/// An episode is older when its work is. Episodes of finished works are left out; they refresh
/// when opened or when their work's save changes them, unless older merge code merged them.
///
/// # Examples
///
/// ```no_run
/// # use chrono::Utc;
/// # async fn example(pool: &sqlx::PgPool) -> Result<(), postgres_db::Error> {
/// let query = postgres_db::DueEntitiesQuery {
///     due_before: Utc::now(),
///     older_after_years: 30,
///     page_size: 100,
///     after_next_refresh_at: None,
///     after_id: None,
///     merge_version: 1,
/// };
/// let due_episodes = postgres_db::list_due_episodes(pool, &query).await?;
/// assert!(due_episodes.len() <= 100);
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns an invalid-argument error when `page_size` is not positive or `older_after_years`
/// is negative or above [`MAX_OLDER_AFTER_YEARS`], or a database error when the query fails.
pub async fn list_due_episodes(
    pool: &PgPool,
    query: &DueEntitiesQuery,
) -> Result<Vec<DueEntityRecord>, Error> {
    validate_due_query(query)?;
    let due_records = sqlx::query_as!(
        DueEntityRecord,
        r#"
        SELECT episode.id AS entity_id, episode.next_refresh_at
        FROM episode
        JOIN anime ON anime.id = episode.anime_id
        WHERE episode.deleted_at IS NULL
          AND anime.deleted_at IS NULL
          AND (
                (coalesce(episode.next_refresh_at, '-infinity'::timestamptz) < $1
                 AND anime.status IN ('ONGOING', 'UPCOMING')
                 AND (anime.year IS NULL OR anime.year > extract(year FROM now())::int - $2))
             OR coalesce(episode.merge_version, 0) < $6
          )
          AND (coalesce(episode.next_refresh_at, '-infinity'::timestamptz), episode.id)
              > (coalesce($3, '-infinity'::timestamptz), coalesce($4, '00000000-0000-0000-0000-000000000000'::uuid))
        ORDER BY coalesce(episode.next_refresh_at, '-infinity'::timestamptz), episode.id
        LIMIT $5
        "#,
        query.due_before,
        query.older_after_years,
        query.after_next_refresh_at,
        query.after_id,
        i64::from(query.page_size),
        query.merge_version
    )
    .fetch_all(pool)
    .await?;
    Ok(due_records)
}

/// Lists due characters of ongoing and upcoming works, paged like [`list_due_anime`].
///
/// A character's works are found through its `character_animeography` pages. A character is
/// due through a work that the sweep refreshes itself: an undeleted work that is ongoing or
/// upcoming and not older. A character whose ongoing works are all older is left out, as those
/// works are, whatever its finished works' age. Unless older merge code merged it, a character
/// with no such work is left out.
///
/// # Examples
///
/// ```no_run
/// # use chrono::Utc;
/// # async fn example(pool: &sqlx::PgPool) -> Result<(), postgres_db::Error> {
/// let query = postgres_db::DueEntitiesQuery {
///     due_before: Utc::now(),
///     older_after_years: 30,
///     page_size: 100,
///     after_next_refresh_at: None,
///     after_id: None,
///     merge_version: 1,
/// };
/// let due_characters = postgres_db::list_due_characters(pool, &query).await?;
/// assert!(due_characters.len() <= 100);
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns an invalid-argument error when `page_size` is not positive or `older_after_years`
/// is negative or above [`MAX_OLDER_AFTER_YEARS`], or a database error when the query fails.
pub async fn list_due_characters(
    pool: &PgPool,
    query: &DueEntitiesQuery,
) -> Result<Vec<DueEntityRecord>, Error> {
    validate_due_query(query)?;
    let due_records = sqlx::query_as!(
        DueEntityRecord,
        r#"
        SELECT character.id AS entity_id, character.next_refresh_at
        FROM character
        WHERE character.deleted_at IS NULL
          AND (
                (coalesce(character.next_refresh_at, '-infinity'::timestamptz) < $1
                 AND EXISTS (
                     SELECT 1
                     FROM character_animeography appearance
                     JOIN anime_source work_page
                       ON work_page.provider = appearance.anime_page_provider
                      AND work_page.entity_type = 'anime'
                      AND work_page.address = appearance.anime_page_address
                     JOIN anime ON anime.id = work_page.anime_id
                     WHERE appearance.character_id = character.id
                       AND anime.deleted_at IS NULL
                       AND anime.status IN ('ONGOING', 'UPCOMING')
                       AND (anime.year IS NULL OR anime.year > extract(year FROM now())::int - $2)
                 ))
             OR coalesce(character.merge_version, 0) < $6
          )
          AND (coalesce(character.next_refresh_at, '-infinity'::timestamptz), character.id)
              > (coalesce($3, '-infinity'::timestamptz), coalesce($4, '00000000-0000-0000-0000-000000000000'::uuid))
        ORDER BY coalesce(character.next_refresh_at, '-infinity'::timestamptz), character.id
        LIMIT $5
        "#,
        query.due_before,
        query.older_after_years,
        query.after_next_refresh_at,
        query.after_id,
        i64::from(query.page_size),
        query.merge_version
    )
    .fetch_all(pool)
    .await?;
    Ok(due_records)
}

/// Lists episodes of ongoing and upcoming works expected to air within `window_days`.
///
/// Two kinds of episode are returned, both from merged, undeleted works:
///
/// - Merged episodes whose `aired_at` falls in the window and whose aired notification is not
///   written yet. The window starts 24 hours back, so a delayed episode stays tracked.
/// - The next episode a work's broadcast announces, at its announced time, unless a merged
///   episode with that number already has an air date. It carries the episode's ID when a row
///   for the number exists but has no air date yet, and no ID when no row exists.
///
/// Rows are ordered by expected air time.
///
/// # Examples
///
/// ```no_run
/// # async fn example(pool: &sqlx::PgPool) -> Result<(), postgres_db::Error> {
/// let upcoming_episodes = postgres_db::list_upcoming_episodes(pool, 7).await?;
/// assert!(upcoming_episodes
///     .windows(2)
///     .all(|pair| pair[0].expected_air_at <= pair[1].expected_air_at));
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns an invalid-argument error when `window_days` is negative or above
/// [`MAX_WINDOW_DAYS`], or a database error when the query fails.
pub async fn list_upcoming_episodes(
    pool: &PgPool,
    window_days: i32,
) -> Result<Vec<UpcomingEpisodeRecord>, Error> {
    if !(0..=MAX_WINDOW_DAYS).contains(&window_days) {
        return Err(Error::invalid_argument(format!(
            "window_days must be between 0 and {MAX_WINDOW_DAYS}"
        )));
    }

    let upcoming_episodes = sqlx::query_as!(
        UpcomingEpisodeRecord,
        r#"
        SELECT episode.anime_id AS "anime_id!",
               episode.id AS "episode_id?",
               episode.episode_number AS "episode_number!",
               episode.aired_at AS "expected_air_at!"
        FROM episode_visible episode
        JOIN anime_visible anime ON anime.id = episode.anime_id
        WHERE anime.status IN ('ONGOING', 'UPCOMING')
          AND episode.aired_at BETWEEN now() - interval '24 hours' AND now() + make_interval(days => $1)
          AND episode.aired_notified_at IS NULL
        UNION ALL
        SELECT broadcast.anime_id AS "anime_id!",
               (SELECT pending.id
                FROM episode pending
                WHERE pending.anime_id = broadcast.anime_id
                  AND pending.episode_number = broadcast.next_episode_number
                  AND pending.deleted_at IS NULL) AS "episode_id?",
               broadcast.next_episode_number AS "episode_number!",
               broadcast.next_episode_at AS "expected_air_at!"
        FROM anime_broadcast broadcast
        JOIN anime_visible anime ON anime.id = broadcast.anime_id
        WHERE anime.status IN ('ONGOING', 'UPCOMING')
          AND broadcast.next_episode_number IS NOT NULL
          AND broadcast.next_episode_at BETWEEN now() - interval '24 hours' AND now() + make_interval(days => $1)
          AND NOT EXISTS (
              SELECT 1
              FROM episode_visible dated
              WHERE dated.anime_id = broadcast.anime_id
                AND dated.episode_number = broadcast.next_episode_number
                AND dated.aired_at IS NOT NULL
          )
        ORDER BY 4, 1, 3
        "#,
        window_days
    )
    .fetch_all(pool)
    .await?;
    Ok(upcoming_episodes)
}

/// Pages the works whose scores refresh daily, by ID.
///
/// A work qualifies when it is ongoing, or finished with an `aired_to` within `lookback_days`.
/// A finished work without an end date is left to the weekly sweep. Only merged, undeleted
/// works are read.
///
/// # Examples
///
/// ```no_run
/// # async fn example(pool: &sqlx::PgPool) -> Result<(), postgres_db::Error> {
/// let anime_ids = postgres_db::list_score_anime(pool, 90, 100, None).await?;
/// if let Some(&last_id) = anime_ids.last() {
///     let next_page = postgres_db::list_score_anime(pool, 90, 100, Some(last_id)).await?;
///     assert!(next_page.iter().all(|&anime_id| anime_id > last_id));
/// }
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns an invalid-argument error when `lookback_days` is negative or above
/// [`MAX_WINDOW_DAYS`] or `page_size` is not positive, or a database error when the query
/// fails.
pub async fn list_score_anime(
    pool: &PgPool,
    lookback_days: i32,
    page_size: i32,
    after_id: Option<Uuid>,
) -> Result<Vec<Uuid>, Error> {
    if !(0..=MAX_WINDOW_DAYS).contains(&lookback_days) {
        return Err(Error::invalid_argument(format!(
            "lookback_days must be between 0 and {MAX_WINDOW_DAYS}"
        )));
    }
    if page_size <= 0 {
        return Err(Error::invalid_argument("page_size must be positive"));
    }

    let anime_ids = sqlx::query_scalar!(
        r#"
        SELECT id AS "id!"
        FROM anime_visible
        WHERE (status = 'ONGOING'
               OR (status = 'FINISHED' AND aired_to >= now() - make_interval(days => $1)))
          AND id > coalesce($2, '00000000-0000-0000-0000-000000000000'::uuid)
        ORDER BY id
        LIMIT $3
        "#,
        lookback_days,
        after_id,
        i64::from(page_size)
    )
    .fetch_all(pool)
    .await?;
    Ok(anime_ids)
}

/// Lists entities whose merge the database requested, oldest request first.
///
/// Returns undeleted entities with `merge_requested_at` set, whatever their age, status or
/// `next_refresh_at`, through each table's partial index on `merge_requested_at`.
///
/// # Examples
///
/// ```no_run
/// # async fn example(pool: &sqlx::PgPool) -> Result<(), postgres_db::Error> {
/// let anime_ids =
///     postgres_db::list_requested_merges(pool, postgres_db::EntityType::Anime, 100).await?;
/// assert!(anime_ids.len() <= 100);
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns an invalid-argument error when `limit` is not positive, or a database error when
/// the query fails.
pub async fn list_requested_merges(
    pool: &PgPool,
    entity_type: EntityType,
    limit: i32,
) -> Result<Vec<Uuid>, Error> {
    if limit <= 0 {
        return Err(Error::invalid_argument("limit must be positive"));
    }
    let entity_ids = sqlx::query_scalar!(
        r#"
        SELECT requested.id AS "id!"
        FROM (
            SELECT id, merge_requested_at FROM anime
            WHERE $1 = 'anime' AND merge_requested_at IS NOT NULL AND deleted_at IS NULL
            UNION ALL
            SELECT id, merge_requested_at FROM episode
            WHERE $1 = 'episode' AND merge_requested_at IS NOT NULL AND deleted_at IS NULL
            UNION ALL
            SELECT id, merge_requested_at FROM character
            WHERE $1 = 'character' AND merge_requested_at IS NOT NULL AND deleted_at IS NULL
        ) requested
        ORDER BY requested.merge_requested_at, requested.id
        LIMIT $2
        "#,
        entity_type.as_str(),
        i64::from(limit)
    )
    .fetch_all(pool)
    .await?;
    Ok(entity_ids)
}

/// Pages one kind's visible entities by ID, with their versions.
///
/// Reads the `*_visible` views, so unmerged and deleted entities are left out. The vector
/// service's comparison job uses it to find points whose version differs.
///
/// # Examples
///
/// ```no_run
/// # async fn example(pool: &sqlx::PgPool) -> Result<(), postgres_db::Error> {
/// let versions =
///     postgres_db::list_entity_versions(pool, postgres_db::EntityType::Anime, 100, None).await?;
/// assert!(versions.windows(2).all(|pair| pair[0].entity_id < pair[1].entity_id));
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns an invalid-argument error when `page_size` is not positive, or a database error
/// when the query fails.
pub async fn list_entity_versions(
    pool: &PgPool,
    entity_type: EntityType,
    page_size: i32,
    after_id: Option<Uuid>,
) -> Result<Vec<EntityVersionRecord>, Error> {
    if page_size <= 0 {
        return Err(Error::invalid_argument("page_size must be positive"));
    }
    let versions = sqlx::query_as!(
        EntityVersionRecord,
        r#"
        SELECT visible.id AS "entity_id!", visible.version AS "version!"
        FROM (
            SELECT id, version FROM anime_visible WHERE $1 = 'anime'
            UNION ALL
            SELECT id, version FROM episode_visible WHERE $1 = 'episode'
            UNION ALL
            SELECT id, version FROM character_visible WHERE $1 = 'character'
        ) visible
        WHERE visible.id > coalesce($2, '00000000-0000-0000-0000-000000000000'::uuid)
        ORDER BY visible.id
        LIMIT $3
        "#,
        entity_type.as_str(),
        after_id,
        i64::from(page_size)
    )
    .fetch_all(pool)
    .await?;
    Ok(versions)
}

/// Lists one budgeted provider's due source pages, never-fetched first, then oldest fetch.
///
/// Only source pages of undeleted works are listed, since a partial-source page is never
/// merged. Dead pages are left out, and a page whose last fetch failed waits
/// `2^fetch_failures` hours, at most a week, after the failure.
///
/// # Examples
///
/// ```no_run
/// # async fn example(pool: &sqlx::PgPool) -> Result<(), postgres_db::Error> {
/// let due_pages = postgres_db::list_due_pages(pool, postgres_db::Provider::AniDb, 50).await?;
/// assert!(due_pages.len() <= 50);
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns an invalid-argument error when `page_size` is not positive or a stored page key no
/// longer passes validation, or a database error when the query fails.
pub async fn list_due_pages(
    pool: &PgPool,
    provider: Provider,
    page_size: i32,
) -> Result<Vec<DuePageRecord>, Error> {
    if page_size <= 0 {
        return Err(Error::invalid_argument("page_size must be positive"));
    }

    let page_rows = sqlx::query!(
        r#"
        SELECT page.provider AS "provider: Provider",
               page.entity_type AS "entity_type: EntityType",
               page.address,
               link.anime_id
        FROM provider_page page
        JOIN anime_source link USING (provider, entity_type, address)
        JOIN anime ON anime.id = link.anime_id
        WHERE page.provider = $1
          AND NOT page.is_dead
          AND anime.deleted_at IS NULL
          AND (page.fetch_failed_at IS NULL
               OR page.fetch_failed_at < now() - make_interval(
                      hours => least(power(2, least(page.fetch_failures, $3))::int, $4)))
        ORDER BY page.fetched_at NULLS FIRST, page.address
        LIMIT $2
        "#,
        provider.as_str(),
        i64::from(page_size),
        MAX_BACKOFF_EXPONENT,
        MAX_BACKOFF_HOURS
    )
    .fetch_all(pool)
    .await?;

    page_rows
        .into_iter()
        .map(|page_row| {
            Ok(DuePageRecord {
                page_key: PageKey::new(page_row.provider, page_row.entity_type, page_row.address)?,
                anime_id: page_row.anime_id,
            })
        })
        .collect()
}
