//! Scheduling, due-entity paging, page listing, and refresh recording queries.

use chrono::{DateTime, Duration, Utc};
use sqlx::{PgPool, Postgres, QueryBuilder, Row};
use uuid::Uuid;

use crate::{EntityType, Error, PageKey, Provider, read_transaction_time};

/// Maximum backoff interval (7 days = 168 hours) for failed page or refresh retries.
const MAX_BACKOFF_HOURS: i32 = 168;

/// Cap exponent on retry backoff so `2^failures` never overflows 32-bit integers.
const MAX_BACKOFF_SHIFT: i32 = 10;

/// Outcome of an entity refresh run passed to [`record_refresh`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefreshOutcome {
    /// Every due page succeeded or was not due.
    Succeeded,
    /// Merge ran, but one or more source pages failed every retry attempt.
    PagesFailed,
    /// Merge or save failed; schedules exponential entity backoff.
    Failed,
}

/// Provider page state returned by [`list_entity_pages`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EntityPageRecord {
    /// Page identifier (`provider`, `entity_type`, `address`).
    pub page_key: PageKey,
    /// Whether the page is linked as a partial source rather than a merged source.
    pub is_partial_source: bool,
    /// Last successful fetch time, or `None` when never fetched.
    pub fetched_at: Option<DateTime<Utc>>,
    /// Whether the page has been confirmed dead.
    pub is_dead: bool,
    /// Timestamp of the most recent failed fetch attempt, if any.
    pub fetch_failed_at: Option<DateTime<Utc>>,
    /// Consecutive fetch failure count since the last success.
    pub fetch_failures: i32,
}

/// Parameters for keyset-paged due entity queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DueEntitiesQuery {
    /// Fixed sweep cutoff compared against `next_refresh_at`.
    pub due_before: DateTime<Utc>,
    /// Entities first aired more than this many years ago refresh only on view.
    pub older_after_years: i32,
    /// Maximum number of rows to return in this page.
    pub page_size: i32,
    /// Keyset cursor timestamp from the last row of the previous page.
    pub after_next_refresh_at: Option<DateTime<Utc>>,
    /// Keyset cursor UUID from the last row of the previous page.
    pub after_id: Option<Uuid>,
    /// Current merge code version; rows with older `merge_version` are always due.
    pub merge_version: i32,
}

/// Due entity identifier and cursor value returned by `list_due_*`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DueEntityRecord {
    /// Entity UUID.
    pub entity_id: Uuid,
    /// Scheduled next refresh time, or `None` when immediately due.
    pub next_refresh_at: Option<DateTime<Utc>>,
}

/// Episode expected to air within the tracking window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UpcomingEpisodeRecord {
    /// Parent anime UUID.
    pub anime_id: Uuid,
    /// Episode UUID when already listed, or `None` when inferred from broadcast.
    pub episode_id: Option<Uuid>,
    /// 1-based episode number (or 0 for special episode 0).
    pub episode_number: i32,
    /// Expected UTC air timestamp.
    pub expected_air_at: DateTime<Utc>,
}

/// Visible entity identifier and version returned by [`list_entity_versions`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EntityVersionRecord {
    /// Entity UUID.
    pub entity_id: Uuid,
    /// Optimistic concurrency version stored on the entity.
    pub version: i64,
}

/// Due source page and owning work returned by [`list_due_pages`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DuePageRecord {
    /// Page identifier (`provider`, `entity_type`, `address`).
    pub page_key: PageKey,
    /// Owning work UUID whose merge consumes this page.
    pub anime_id: Uuid,
}

/// Returns the base table name for `entity_type`.
#[must_use]
const fn entity_table_name(entity_type: EntityType) -> &'static str {
    match entity_type {
        EntityType::Anime => "anime",
        EntityType::Episode => "episode",
        EntityType::Character => "character",
    }
}

/// Returns the visible view name for `entity_type`.
#[must_use]
const fn entity_visible_view_name(entity_type: EntityType) -> &'static str {
    match entity_type {
        EntityType::Anime => "anime_visible",
        EntityType::Episode => "episode_visible",
        EntityType::Character => "character_visible",
    }
}

/// Ensures an entity exists and is not soft-deleted.
async fn ensure_entity_exists(
    pool: &PgPool,
    entity_type: EntityType,
    entity_id: Uuid,
) -> Result<(), Error> {
    let table = entity_table_name(entity_type);
    let mut query: QueryBuilder<'_, Postgres> = QueryBuilder::new("SELECT 1 FROM ");
    query.push(table);
    query.push(" WHERE id = ");
    query.push_bind(entity_id);
    query.push(" AND deleted_at IS NULL");

    let exists = query.build().fetch_optional(pool).await?.is_some();
    if !exists {
        return Err(Error::not_found());
    }
    Ok(())
}

/// Lists all linked provider pages and their fetch metadata for an entity.
///
/// Reads base link tables directly so unmerged entities can be refreshed.
///
/// # Examples
///
/// ```no_run
/// # use uuid::Uuid;
/// # async fn example(pool: &sqlx::PgPool, anime_id: Uuid) -> Result<(), postgres_db::Error> {
/// let pages = postgres_db::list_entity_pages(pool, postgres_db::EntityType::Anime, anime_id).await?;
/// assert!(pages.iter().all(|page| !page.is_dead || page.is_partial_source || page.is_dead));
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns [`Error::not_found`] when the entity does not exist or is deleted,
/// or a database error if querying PostgreSQL fails.
pub async fn list_entity_pages(
    pool: &PgPool,
    entity_type: EntityType,
    entity_id: Uuid,
) -> Result<Vec<EntityPageRecord>, Error> {
    ensure_entity_exists(pool, entity_type, entity_id).await?;

    let rows = match entity_type {
        EntityType::Anime => {
            sqlx::query(
                "SELECT p.provider, p.entity_type, p.address, false AS is_partial_source,
                        p.fetched_at, p.is_dead, p.fetch_failed_at, p.fetch_failures
                 FROM anime_source s
                 JOIN provider_page p USING (provider, entity_type, address)
                 WHERE s.anime_id = $1
                 UNION ALL
                 SELECT p.provider, p.entity_type, p.address, true AS is_partial_source,
                        p.fetched_at, p.is_dead, p.fetch_failed_at, p.fetch_failures
                 FROM anime_partial_source ps
                 JOIN provider_page p USING (provider, entity_type, address)
                 WHERE ps.anime_id = $1
                 ORDER BY is_partial_source ASC, provider ASC, address ASC",
            )
            .bind(entity_id)
            .fetch_all(pool)
            .await?
        }
        EntityType::Episode => {
            sqlx::query(
                "SELECT p.provider, p.entity_type, p.address, false AS is_partial_source,
                        p.fetched_at, p.is_dead, p.fetch_failed_at, p.fetch_failures
                 FROM episode_source s
                 JOIN provider_page p USING (provider, entity_type, address)
                 WHERE s.episode_id = $1
                 ORDER BY p.provider ASC, p.address ASC",
            )
            .bind(entity_id)
            .fetch_all(pool)
            .await?
        }
        EntityType::Character => {
            sqlx::query(
                "SELECT p.provider, p.entity_type, p.address, false AS is_partial_source,
                        p.fetched_at, p.is_dead, p.fetch_failed_at, p.fetch_failures
                 FROM character_source s
                 JOIN provider_page p USING (provider, entity_type, address)
                 WHERE s.character_id = $1
                 ORDER BY p.provider ASC, p.address ASC",
            )
            .bind(entity_id)
            .fetch_all(pool)
            .await?
        }
    };

    rows.into_iter()
        .map(|row| {
            let provider: Provider = row.try_get::<String, _>("provider")?.parse()?;
            let row_entity_type: EntityType = row.try_get::<String, _>("entity_type")?.parse()?;
            let address: String = row.try_get("address")?;
            let page_key = PageKey::new(provider, row_entity_type, address)?;
            Ok(EntityPageRecord {
                page_key,
                is_partial_source: row.try_get("is_partial_source")?,
                fetched_at: row.try_get("fetched_at")?,
                is_dead: row.try_get("is_dead")?,
                fetch_failed_at: row.try_get("fetch_failed_at")?,
                fetch_failures: row.try_get("fetch_failures")?,
            })
        })
        .collect()
}

/// Computes the earliest retry timestamp among failed source pages of an entity.
async fn find_earliest_failed_page_retry(
    tx: &mut sqlx::Transaction<'_, Postgres>,
    entity_type: EntityType,
    entity_id: Uuid,
) -> Result<Option<DateTime<Utc>>, Error> {
    let (link_table, fk_col) = match entity_type {
        EntityType::Anime => ("anime_source", "anime_id"),
        EntityType::Episode => ("episode_source", "episode_id"),
        EntityType::Character => ("character_source", "character_id"),
    };

    let mut query: QueryBuilder<'_, Postgres> = QueryBuilder::new(
        "SELECT MIN(p.fetch_failed_at + make_interval(hours => LEAST(POWER(2, LEAST(p.fetch_failures, ",
    );
    query.push_bind(MAX_BACKOFF_SHIFT);
    query.push("))::int, ");
    query.push_bind(MAX_BACKOFF_HOURS);
    query.push("))) FROM ");
    query.push(link_table);
    query.push(" s JOIN provider_page p USING (provider, entity_type, address) WHERE s.");
    query.push(fk_col);
    query.push(" = ");
    query.push_bind(entity_id);
    query.push(" AND NOT p.is_dead AND p.fetch_failed_at IS NOT NULL");

    let earliest: Option<DateTime<Utc>> = query.build_query_scalar().fetch_one(&mut **tx).await?;
    Ok(earliest)
}

/// Records the outcome of an entity refresh and updates `next_refresh_at`.
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
/// Returns [`Error::invalid_argument`] when `refresh_after` is negative,
/// [`Error::not_found`] when the entity does not exist or is deleted, or a
/// database error if updating PostgreSQL fails.
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

    let mut tx = pool.begin().await?;
    let tx_now = read_transaction_time(&mut tx).await?;
    let table = entity_table_name(entity_type);

    let rows_affected = match outcome {
        RefreshOutcome::Succeeded => {
            let next_refresh_at = tx_now + refresh_after;
            let mut query: QueryBuilder<'_, Postgres> = QueryBuilder::new("UPDATE ");
            query.push(table);
            query.push(" SET refreshed_at = ");
            query.push_bind(tx_now);
            query.push(", refresh_failures = 0, next_refresh_at = ");
            query.push_bind(next_refresh_at);
            query.push(" WHERE id = ");
            query.push_bind(entity_id);
            query.push(" AND deleted_at IS NULL");
            query.build().execute(&mut *tx).await?.rows_affected()
        }
        RefreshOutcome::PagesFailed => {
            let earliest_retry =
                find_earliest_failed_page_retry(&mut tx, entity_type, entity_id).await?;
            let next_refresh_at = earliest_retry.unwrap_or_else(|| tx_now + refresh_after);
            let mut query: QueryBuilder<'_, Postgres> = QueryBuilder::new("UPDATE ");
            query.push(table);
            query.push(" SET refreshed_at = ");
            query.push_bind(tx_now);
            query.push(", refresh_failures = 0, next_refresh_at = ");
            query.push_bind(next_refresh_at);
            query.push(" WHERE id = ");
            query.push_bind(entity_id);
            query.push(" AND deleted_at IS NULL");
            query.build().execute(&mut *tx).await?.rows_affected()
        }
        RefreshOutcome::Failed => {
            let mut query: QueryBuilder<'_, Postgres> = QueryBuilder::new("UPDATE ");
            query.push(table);
            query.push(" SET refresh_failed_at = ");
            query.push_bind(tx_now);
            query.push(", refresh_failures = refresh_failures + 1, next_refresh_at = ");
            query.push_bind(tx_now);
            query.push(" + make_interval(hours => LEAST(POWER(2, LEAST(refresh_failures + 1, ");
            query.push_bind(MAX_BACKOFF_SHIFT);
            query.push("))::int, ");
            query.push_bind(MAX_BACKOFF_HOURS);
            query.push(")) WHERE id = ");
            query.push_bind(entity_id);
            query.push(" AND deleted_at IS NULL");
            query.build().execute(&mut *tx).await?.rows_affected()
        }
    };

    if rows_affected == 0 {
        return Err(Error::not_found());
    }

    tx.commit().await?;
    Ok(())
}

/// Validates keyset-paged due-entity parameters.
fn validate_due_query(query: &DueEntitiesQuery) -> Result<(), Error> {
    if query.page_size <= 0 {
        return Err(Error::invalid_argument("page_size must be positive"));
    }
    if query.older_after_years < 0 {
        return Err(Error::invalid_argument(
            "older_after_years must not be negative",
        ));
    }
    Ok(())
}

/// Maps SQL `(id, next_refresh_at)` rows into [`DueEntityRecord`]s.
fn map_due_rows(rows: Vec<(Uuid, Option<DateTime<Utc>>)>) -> Vec<DueEntityRecord> {
    rows.into_iter()
        .map(|(entity_id, next_refresh_at)| DueEntityRecord {
            entity_id,
            next_refresh_at,
        })
        .collect()
}

/// Lists works due for refresh using keyset pagination on `(next_refresh_at, id)`.
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
/// let due = postgres_db::list_due_anime(pool, &query).await?;
/// assert!(due.len() <= 100);
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns [`Error::invalid_argument`] when `page_size <= 0` or `older_after_years < 0`,
/// or a database error if querying PostgreSQL fails.
pub async fn list_due_anime(
    pool: &PgPool,
    query: &DueEntitiesQuery,
) -> Result<Vec<DueEntityRecord>, Error> {
    validate_due_query(query)?;
    let rows: Vec<(Uuid, Option<DateTime<Utc>>)> = sqlx::query_as(
        "SELECT id, next_refresh_at
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
         LIMIT $5",
    )
    .bind(query.due_before)
    .bind(query.older_after_years)
    .bind(query.after_next_refresh_at)
    .bind(query.after_id)
    .bind(i64::from(query.page_size))
    .bind(query.merge_version)
    .fetch_all(pool)
    .await?;
    Ok(map_due_rows(rows))
}

/// Lists episodes of ongoing and upcoming works due for refresh.
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
/// let due = postgres_db::list_due_episodes(pool, &query).await?;
/// assert!(due.len() <= 100);
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns [`Error::invalid_argument`] when `page_size <= 0` or `older_after_years < 0`,
/// or a database error if querying PostgreSQL fails.
pub async fn list_due_episodes(
    pool: &PgPool,
    query: &DueEntitiesQuery,
) -> Result<Vec<DueEntityRecord>, Error> {
    validate_due_query(query)?;
    let rows: Vec<(Uuid, Option<DateTime<Utc>>)> = sqlx::query_as(
        "SELECT e.id, e.next_refresh_at
         FROM episode e
         JOIN anime a ON a.id = e.anime_id
         WHERE e.deleted_at IS NULL
           AND a.deleted_at IS NULL
           AND a.status IN ('ONGOING', 'UPCOMING')
           AND (
                 (coalesce(e.next_refresh_at, '-infinity'::timestamptz) < $1
                  AND (a.year IS NULL OR a.year > extract(year FROM now())::int - $2))
              OR coalesce(e.merge_version, 0) < $6
           )
           AND (coalesce(e.next_refresh_at, '-infinity'::timestamptz), e.id)
               > (coalesce($3, '-infinity'::timestamptz), coalesce($4, '00000000-0000-0000-0000-000000000000'::uuid))
         ORDER BY coalesce(e.next_refresh_at, '-infinity'::timestamptz), e.id
         LIMIT $5",
    )
    .bind(query.due_before)
    .bind(query.older_after_years)
    .bind(query.after_next_refresh_at)
    .bind(query.after_id)
    .bind(i64::from(query.page_size))
    .bind(query.merge_version)
    .fetch_all(pool)
    .await?;
    Ok(map_due_rows(rows))
}

/// Lists characters of ongoing and upcoming works due for refresh.
///
/// A character counts as ongoing when any of its undeleted works is ongoing or
/// upcoming, and counts as older only when every undeleted work is older.
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
/// let due = postgres_db::list_due_characters(pool, &query).await?;
/// assert!(due.len() <= 100);
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns [`Error::invalid_argument`] when `page_size <= 0` or `older_after_years < 0`,
/// or a database error if querying PostgreSQL fails.
pub async fn list_due_characters(
    pool: &PgPool,
    query: &DueEntitiesQuery,
) -> Result<Vec<DueEntityRecord>, Error> {
    validate_due_query(query)?;
    let rows: Vec<(Uuid, Option<DateTime<Utc>>)> = sqlx::query_as(
        "SELECT c.id, c.next_refresh_at
         FROM character c
         WHERE c.deleted_at IS NULL
           AND EXISTS (
               SELECT 1
               FROM character_animeography ca
               JOIN anime_source ap
                 ON ap.provider = ca.anime_page_provider
                AND ap.entity_type = 'anime'
                AND ap.address = ca.anime_page_address
               JOIN anime a ON a.id = ap.anime_id
               WHERE ca.character_id = c.id
                 AND a.deleted_at IS NULL
                 AND a.status IN ('ONGOING', 'UPCOMING')
           )
           AND (
                 (coalesce(c.next_refresh_at, '-infinity'::timestamptz) < $1
                  AND EXISTS (
                      SELECT 1
                      FROM character_animeography ca
                      JOIN anime_source ap
                        ON ap.provider = ca.anime_page_provider
                       AND ap.entity_type = 'anime'
                       AND ap.address = ca.anime_page_address
                      JOIN anime a ON a.id = ap.anime_id
                      WHERE ca.character_id = c.id
                        AND a.deleted_at IS NULL
                        AND (a.year IS NULL OR a.year > extract(year FROM now())::int - $2)
                  ))
              OR coalesce(c.merge_version, 0) < $6
           )
           AND (coalesce(c.next_refresh_at, '-infinity'::timestamptz), c.id)
               > (coalesce($3, '-infinity'::timestamptz), coalesce($4, '00000000-0000-0000-0000-000000000000'::uuid))
         ORDER BY coalesce(c.next_refresh_at, '-infinity'::timestamptz), c.id
         LIMIT $5",
    )
    .bind(query.due_before)
    .bind(query.older_after_years)
    .bind(query.after_next_refresh_at)
    .bind(query.after_id)
    .bind(i64::from(query.page_size))
    .bind(query.merge_version)
    .fetch_all(pool)
    .await?;
    Ok(map_due_rows(rows))
}

/// Lists episodes of ongoing and upcoming works airing within `window_days`.
///
/// Looks back 24 hours for expected episodes not yet marked aired, and includes
/// the next episode announced on `anime_broadcast` when no episode row exists yet.
///
/// # Examples
///
/// ```no_run
/// # async fn example(pool: &sqlx::PgPool) -> Result<(), postgres_db::Error> {
/// let episodes = postgres_db::list_upcoming_episodes(pool, 7).await?;
/// assert!(episodes.windows(2).all(|pair| pair[0].expected_air_at <= pair[1].expected_air_at));
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns [`Error::invalid_argument`] when `window_days < 0`, or a database
/// error if querying PostgreSQL fails.
pub async fn list_upcoming_episodes(
    pool: &PgPool,
    window_days: i32,
) -> Result<Vec<UpcomingEpisodeRecord>, Error> {
    if window_days < 0 {
        return Err(Error::invalid_argument("window_days must not be negative"));
    }

    let rows = sqlx::query(
        "SELECT e.id AS episode_id, e.anime_id, e.episode_number, e.aired_at AS expected_air_at
         FROM episode e
         JOIN anime a ON a.id = e.anime_id
         WHERE a.status IN ('ONGOING', 'UPCOMING')
           AND a.deleted_at IS NULL
           AND e.deleted_at IS NULL
           AND e.aired_at BETWEEN now() - interval '24 hours' AND now() + make_interval(days => $1)
           AND e.aired_notified_at IS NULL
         UNION ALL
         SELECT NULL::uuid AS episode_id, b.anime_id, b.next_episode_number AS episode_number,
                b.next_episode_at AS expected_air_at
         FROM anime_broadcast b
         JOIN anime a ON a.id = b.anime_id
         WHERE a.status IN ('ONGOING', 'UPCOMING')
           AND a.deleted_at IS NULL
           AND b.next_episode_number IS NOT NULL
           AND b.next_episode_at BETWEEN now() - interval '24 hours' AND now() + make_interval(days => $1)
           AND NOT EXISTS (
               SELECT 1
               FROM episode e
               WHERE e.anime_id = b.anime_id
                 AND e.episode_number = b.next_episode_number
                 AND e.deleted_at IS NULL
           )
         ORDER BY expected_air_at",
    )
    .bind(window_days)
    .fetch_all(pool)
    .await?;

    rows.into_iter()
        .map(|row| {
            Ok(UpcomingEpisodeRecord {
                anime_id: row.try_get("anime_id")?,
                episode_id: row.try_get("episode_id")?,
                episode_number: row.try_get("episode_number")?,
                expected_air_at: row.try_get("expected_air_at")?,
            })
        })
        .collect()
}

/// Pages through works whose scores refresh daily (ongoing or recently finished).
///
/// # Examples
///
/// ```no_run
/// # async fn example(pool: &sqlx::PgPool) -> Result<(), postgres_db::Error> {
/// let anime_ids = postgres_db::list_score_anime(pool, 90, 100, None).await?;
/// assert!(anime_ids.len() <= 100);
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns [`Error::invalid_argument`] when `lookback_days < 0` or
/// `page_size <= 0`, or a database error if querying PostgreSQL fails.
pub async fn list_score_anime(
    pool: &PgPool,
    lookback_days: i32,
    page_size: i32,
    after_id: Option<Uuid>,
) -> Result<Vec<Uuid>, Error> {
    if lookback_days < 0 {
        return Err(Error::invalid_argument(
            "lookback_days must not be negative",
        ));
    }
    if page_size <= 0 {
        return Err(Error::invalid_argument("page_size must be positive"));
    }

    let anime_ids: Vec<Uuid> = sqlx::query_scalar(
        "SELECT id
         FROM anime
         WHERE deleted_at IS NULL
           AND (
                 status = 'ONGOING'
              OR (status = 'FINISHED' AND aired_to >= now() - make_interval(days => $1))
           )
           AND id > coalesce($2, '00000000-0000-0000-0000-000000000000'::uuid)
         ORDER BY id
         LIMIT $3",
    )
    .bind(lookback_days)
    .bind(after_id)
    .bind(i64::from(page_size))
    .fetch_all(pool)
    .await?;

    Ok(anime_ids)
}

/// Lists entities of `entity_type` whose `merge_requested_at` timestamp is set.
///
/// # Examples
///
/// ```no_run
/// # async fn example(pool: &sqlx::PgPool) -> Result<(), postgres_db::Error> {
/// let ids = postgres_db::list_requested_merges(pool, postgres_db::EntityType::Anime, 100).await?;
/// assert!(ids.len() <= 100);
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns [`Error::invalid_argument`] when `limit <= 0`, or a database error if
/// querying PostgreSQL fails.
pub async fn list_requested_merges(
    pool: &PgPool,
    entity_type: EntityType,
    limit: i32,
) -> Result<Vec<Uuid>, Error> {
    if limit <= 0 {
        return Err(Error::invalid_argument("limit must be positive"));
    }

    let table = entity_table_name(entity_type);
    let mut query: QueryBuilder<'_, Postgres> = QueryBuilder::new("SELECT id FROM ");
    query.push(table);
    query.push(" WHERE merge_requested_at IS NOT NULL AND deleted_at IS NULL ORDER BY merge_requested_at LIMIT ");
    query.push_bind(i64::from(limit));

    let ids: Vec<Uuid> = query.build_query_scalar().fetch_all(pool).await?;
    Ok(ids)
}

/// Pages through visible entities of `entity_type` with their current version.
///
/// # Examples
///
/// ```no_run
/// # async fn example(pool: &sqlx::PgPool) -> Result<(), postgres_db::Error> {
/// let versions = postgres_db::list_entity_versions(
///     pool,
///     postgres_db::EntityType::Anime,
///     100,
///     None,
/// )
/// .await?;
/// assert!(versions.len() <= 100);
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns [`Error::invalid_argument`] when `page_size <= 0`, or a database
/// error if querying PostgreSQL fails.
pub async fn list_entity_versions(
    pool: &PgPool,
    entity_type: EntityType,
    page_size: i32,
    after_id: Option<Uuid>,
) -> Result<Vec<EntityVersionRecord>, Error> {
    if page_size <= 0 {
        return Err(Error::invalid_argument("page_size must be positive"));
    }

    let view = entity_visible_view_name(entity_type);
    let mut query: QueryBuilder<'_, Postgres> = QueryBuilder::new("SELECT id, version FROM ");
    query.push(view);
    query.push(" WHERE id > coalesce(");
    query.push_bind(after_id);
    query.push(", '00000000-0000-0000-0000-000000000000'::uuid) ORDER BY id LIMIT ");
    query.push_bind(i64::from(page_size));

    let rows: Vec<(Uuid, i64)> = query.build_query_as().fetch_all(pool).await?;
    Ok(rows
        .into_iter()
        .map(|(entity_id, version)| EntityVersionRecord { entity_id, version })
        .collect())
}

/// Lists due source pages of a budgeted provider in due order (`fetched_at NULLS FIRST`).
///
/// # Examples
///
/// ```no_run
/// # async fn example(pool: &sqlx::PgPool) -> Result<(), postgres_db::Error> {
/// let pages = postgres_db::list_due_pages(pool, postgres_db::Provider::AniDb, 50).await?;
/// assert!(pages.len() <= 50);
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns [`Error::invalid_argument`] when `page_size <= 0`, or a database
/// error if querying PostgreSQL fails.
pub async fn list_due_pages(
    pool: &PgPool,
    provider: Provider,
    page_size: i32,
) -> Result<Vec<DuePageRecord>, Error> {
    if page_size <= 0 {
        return Err(Error::invalid_argument("page_size must be positive"));
    }

    let rows = sqlx::query(
        "SELECT p.provider, p.entity_type, p.address, ap.anime_id
         FROM provider_page p
         JOIN anime_source ap USING (provider, entity_type, address)
         JOIN anime a ON a.id = ap.anime_id
         WHERE p.provider = $1
           AND NOT p.is_dead
           AND a.deleted_at IS NULL
           AND (
                 p.fetch_failed_at IS NULL
              OR p.fetch_failed_at < now() - make_interval(hours => LEAST(POWER(2, LEAST(p.fetch_failures, $3))::int, $4))
           )
         ORDER BY p.fetched_at NULLS FIRST, p.address
         LIMIT $2",
    )
    .bind(provider.as_str())
    .bind(i64::from(page_size))
    .bind(MAX_BACKOFF_SHIFT)
    .bind(MAX_BACKOFF_HOURS)
    .fetch_all(pool)
    .await?;

    rows.into_iter()
        .map(|row| {
            let row_provider: Provider = row.try_get::<String, _>("provider")?.parse()?;
            let entity_type: EntityType = row.try_get::<String, _>("entity_type")?.parse()?;
            let address: String = row.try_get("address")?;
            let page_key = PageKey::new(row_provider, entity_type, address)?;
            Ok(DuePageRecord {
                page_key,
                anime_id: row.try_get("anime_id")?,
            })
        })
        .collect()
}
