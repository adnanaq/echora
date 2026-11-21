//! Work loading, version-checked saving, and transactional outbox writes.

use std::collections::HashMap;
use std::io::Read as _;

use chrono::{DateTime, Utc};
use flate2::read::GzDecoder;
use prost::Message as _;
use sqlx::{PgPool, Postgres, Transaction};
use tracing::Level;
use uuid::Uuid;

use crate::anime_details::{replace_anime_details, validate_anime_details};
use crate::anime_relations::{replace_relations_and_graph, validate_relations_and_graph};
use crate::anime_schedule_and_stats::{replace_schedule_and_stats, validate_schedule_and_stats};
use crate::episode::{
    SyncWorkEpisodesOutcome, load_known_episodes, sync_work_episodes,
    validate_work_episode_references,
};
use crate::read_transaction_time;
use crate::{
    AnimeDetailsInput, AnimeRelationsAndGraphInput, AnimeScheduleAndStatsInput,
    EpisodeReferenceInput, Error, PageKey, PageKind, PageResponse, Provider,
};

/// Current canonical hash format version stored in `anime.hash_format`.
///
/// Bumping this constant causes unchanged records to re-save once so the new
/// hash format is persisted.
pub const ANIME_HASH_FORMAT: i16 = 5;

/// Minimum valid release year stored in `anime.year`.
pub(crate) const MIN_ANIME_YEAR: i16 = 1900;

/// Minimum valid calendar month (`1` = January).
const MIN_MONTH: i16 = 1;

/// Maximum valid calendar month (`12` = December).
const MAX_MONTH: i16 = 12;

/// Minimum valid rating score on the 10-point scale.
const MIN_SCORE: f64 = 0.0;

/// Maximum valid rating score on the 10-point scale.
const MAX_SCORE: f64 = 10.0;

/// Stored provider page with its decompressed responses returned by [`load_anime`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadedSourcePage {
    /// Page identifier (`provider`, `page_kind`, `address`).
    pub page_key: PageKey,
    /// When the provider page was last fetched.
    pub fetched_at: DateTime<Utc>,
    /// Whether the page is confirmed dead.
    pub is_dead: bool,
    /// Whether the most recent fetch saw a provider not-found status.
    pub is_not_found: bool,
    /// Decompressed raw responses in deterministic `request_name` order.
    pub responses: Vec<PageResponse>,
}

/// Approved per-field override returned by [`load_anime`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldCorrectionRecord {
    /// Dot-separated field path on `Anime` (for example `"episode_count"`).
    pub field_path: String,
    /// Protobuf message type name stored in `google.protobuf.Any.type_url`.
    pub value_message: String,
    /// Serialized protobuf value bytes.
    pub value: Vec<u8>,
    /// Reviewer or service that approved the correction.
    pub approved_by: String,
    /// When the correction was approved.
    pub approved_at: DateTime<Utc>,
}

/// Merge inputs returned by [`load_anime`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadAnimeOutcome {
    /// Current optimistic-locking version of the work.
    pub version: i64,
    /// Linked source pages with their raw responses.
    pub source_pages: Vec<LoadedSourcePage>,
    /// Linked partial-source page keys (never fetched for this work).
    pub partial_source_pages: Vec<PageKey>,
    /// Approved per-field corrections to apply after merging.
    pub corrections: Vec<FieldCorrectionRecord>,
    /// Undeleted episodes the work already lists, so the merge keeps their IDs.
    pub known_episodes: Vec<EpisodeReferenceInput>,
}

/// Canonical work columns and merge metadata passed to [`save_anime`].
#[derive(Debug, Clone, PartialEq)]
pub struct SaveAnimeInput {
    /// UUID of the existing work row to update.
    pub anime_id: Uuid,
    /// Optimistic concurrency version expected in PostgreSQL.
    pub expected_version: i64,
    /// Primary display title.
    pub title: String,
    /// English title, if known.
    pub title_english: Option<String>,
    /// Japanese title, if known.
    pub title_japanese: Option<String>,
    /// Canonical `anime_type` lookup value (for example `"TV"`).
    pub anime_type: String,
    /// Canonical `anime_status` lookup value (for example `"ONGOING"`).
    pub status: String,
    /// Canonical `anime_rating` lookup value (for example `"PG-13 - Teens 13 or older"`).
    pub rating: String,
    /// Canonical `anime_season` lookup value, if known.
    pub season: Option<String>,
    /// Canonical `source_material_type` lookup value, if known.
    pub source_material: Option<String>,
    /// Release year (`>= 1900`), if known.
    pub year: Option<i16>,
    /// Premiere month (`1..=12`), if known.
    pub month: Option<i16>,
    /// Total episode count (`>= 0`).
    pub episode_count: i32,
    /// Episode duration in seconds (`>= 0`), if known.
    pub duration_seconds: Option<i32>,
    /// Explicit NSFW flag, if known.
    pub is_nsfw: Option<bool>,
    /// ISO 3166-1 alpha-2 country code, if known.
    pub country_of_origin: Option<String>,
    /// Approved or AI-written synopsis, if present.
    pub synopsis: Option<String>,
    /// Background notes, if known.
    pub background: Option<String>,
    /// Premiere timestamp, if known.
    pub aired_from: Option<DateTime<Utc>>,
    /// Finale timestamp, if known.
    pub aired_to: Option<DateTime<Utc>>,
    /// Confidence-weighted score (`MIN_SCORE..=MAX_SCORE`), if known.
    pub score_weighted: Option<f64>,
    /// Arithmetic mean score (`MIN_SCORE..=MAX_SCORE`), if known.
    pub score_mean: Option<f64>,
    /// Median score (`MIN_SCORE..=MAX_SCORE`), if known.
    pub score_median: Option<f64>,
    /// Canonical hash of the merged record.
    pub merged_record_hash: String,
    /// Format identifier of `merged_record_hash`.
    pub hash_format: i16,
    /// Version of the mapping and merge code that produced this save.
    pub merge_version: i32,
    /// Optional W3C `traceparent` header propagated into outbox rows.
    pub traceparent: Option<String>,
    /// Work names, classification lists, themes, and images.
    pub details: AnimeDetailsInput,
    /// Theme songs, trailers, links, streaming, statistics, broadcast, and hiatus.
    pub schedule_and_stats: AnimeScheduleAndStatsInput,
    /// Relations, relation claims, source material, and companies.
    pub relations_and_graph: AnimeRelationsAndGraphInput,
    /// Episodes listed by the work's merge.
    pub episodes: Vec<EpisodeReferenceInput>,
}

/// Result returned by [`save_anime`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SaveAnimeOutcome {
    /// Whether the canonical hash differed and database state was updated.
    pub has_changed: bool,
    /// Work version after the save attempt.
    pub version: i64,
    /// Newly created episode IDs to refresh next.
    pub created_episode_ids: Vec<Uuid>,
    /// Existing episode IDs whose list entries changed.
    pub updated_episode_ids: Vec<Uuid>,
}

#[derive(PartialEq, prost::Message)]
pub(crate) struct ProtoTimestamp {
    #[prost(int64, tag = "1")]
    pub(crate) seconds: i64,
    #[prost(int32, tag = "2")]
    pub(crate) nanos: i32,
}

#[derive(PartialEq, prost::Message)]
struct ProtoCatalogueEvent {
    #[prost(string, tag = "1")]
    event_id: String,
    #[prost(message, optional, tag = "2")]
    saved_at: Option<ProtoTimestamp>,
    #[prost(string, tag = "3")]
    entity_id: String,
    #[prost(int64, tag = "4")]
    version: i64,
    #[prost(string, optional, tag = "5")]
    anime_id: Option<String>,
}

#[derive(PartialEq, prost::Message)]
struct ProtoAnimeStatusChangedNotification {
    #[prost(string, tag = "1")]
    event_id: String,
    #[prost(message, optional, tag = "2")]
    saved_at: Option<ProtoTimestamp>,
    #[prost(string, tag = "3")]
    anime_id: String,
    #[prost(string, tag = "4")]
    anime_title: String,
    #[prost(int32, tag = "5")]
    previous_status: i32,
    #[prost(int32, tag = "6")]
    current_status: i32,
}

/// Loads a work's version, linked pages, and approved corrections for merging.
///
/// # Examples
///
/// ```no_run
/// # use sqlx::PgPool;
/// # use uuid::Uuid;
/// # use postgres_db::load_anime;
/// # async fn example(pool: &PgPool, anime_id: Uuid) -> Result<(), postgres_db::Error> {
/// let loaded = load_anime(pool, anime_id).await?;
/// assert!(loaded.version >= 1);
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns [`Error::is_not_found`] when `anime_id` does not exist or is
/// deleted, or a database error when any query or decompression fails.
pub async fn load_anime(pool: &PgPool, anime_id: Uuid) -> Result<LoadAnimeOutcome, Error> {
    let mut tx = pool.begin().await?;
    // Read version, linked page metadata, response bodies, and corrections from
    // one consistent snapshot so concurrent store_page calls cannot mix fields.
    sqlx::query!("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        .execute(&mut *tx)
        .await?;

    let version = sqlx::query_scalar!(
        r#"
        SELECT version
        FROM anime
        WHERE id = $1 AND deleted_at IS NULL
        "#,
        anime_id
    )
    .fetch_optional(&mut *tx)
    .await?
    .ok_or_else(Error::not_found)?;

    let (source_pages, partial_source_pages) = load_linked_pages(&mut tx, anime_id).await?;
    let corrections = load_anime_corrections(&mut tx, anime_id).await?;
    let known_episodes = load_known_episodes(&mut tx, anime_id).await?;
    tx.commit().await?;

    Ok(LoadAnimeOutcome {
        version,
        source_pages,
        partial_source_pages,
        corrections,
        known_episodes,
    })
}

/// Loads a work's fetched source pages and its partial-source page keys.
///
/// A source page that was never fetched is left out, since the merge has nothing to read.
///
/// # Errors
///
/// Returns a database error when a query fails, or an invalid-argument error when
/// a stored page key no longer passes validation.
async fn load_linked_pages(
    tx: &mut Transaction<'_, Postgres>,
    anime_id: Uuid,
) -> Result<(Vec<LoadedSourcePage>, Vec<PageKey>), Error> {
    let linked_rows = sqlx::query!(
        r#"
        SELECT
            app.provider,
            app.page_kind,
            app.address,
            app.role,
            pp.fetched_at,
            pp.is_dead,
            pp.missing_since
        FROM anime_provider_page app
        JOIN provider_page pp
          ON pp.provider = app.provider
         AND pp.page_kind = app.page_kind
         AND pp.address = app.address
        WHERE app.anime_id = $1
        ORDER BY app.role, app.provider, app.address
        "#,
        anime_id
    )
    .fetch_all(&mut **tx)
    .await?;

    let mut responses_by_page = load_source_responses_by_page(tx, anime_id).await?;
    let mut source_pages = Vec::new();
    let mut partial_source_pages = Vec::new();

    for row in linked_rows {
        let provider: Provider = row.provider.parse()?;
        let page_kind = parse_page_kind(&row.page_kind)?;
        let page_key = PageKey::new(provider, page_kind, &row.address)?;
        if row.role == "partial_source" {
            partial_source_pages.push(page_key);
        } else if let Some(fetched_at) = row.fetched_at {
            let responses = responses_by_page
                .remove(&(row.provider, row.page_kind, row.address))
                .unwrap_or_default();
            source_pages.push(LoadedSourcePage {
                page_key,
                fetched_at,
                is_dead: row.is_dead,
                is_not_found: row.missing_since.is_some(),
                responses,
            });
        }
    }

    Ok((source_pages, partial_source_pages))
}

/// Loads the decompressed responses of a work's source pages, grouped by page.
///
/// # Errors
///
/// Returns a database error when the query fails or a stored body cannot be decompressed.
async fn load_source_responses_by_page(
    tx: &mut Transaction<'_, Postgres>,
    anime_id: Uuid,
) -> Result<HashMap<(String, String, String), Vec<PageResponse>>, Error> {
    let response_rows = sqlx::query!(
        r#"
        SELECT
            ppr.provider,
            ppr.page_kind,
            ppr.address,
            ppr.request_name,
            ppr.content_type,
            ppr.body
        FROM anime_provider_page app
        JOIN provider_page_response ppr
          ON ppr.provider = app.provider
         AND ppr.page_kind = app.page_kind
         AND ppr.address = app.address
        WHERE app.anime_id = $1 AND app.role = 'source'
        ORDER BY ppr.provider, ppr.address, ppr.request_name
        "#,
        anime_id
    )
    .fetch_all(&mut **tx)
    .await?;

    let mut responses_by_page: HashMap<(String, String, String), Vec<PageResponse>> =
        HashMap::new();
    for row in response_rows {
        let body = decompress_body(&row.body)?;
        responses_by_page
            .entry((row.provider, row.page_kind, row.address))
            .or_default()
            .push(PageResponse {
                request_name: row.request_name,
                content_type: row.content_type,
                body,
            });
    }
    Ok(responses_by_page)
}

/// Loads a work's approved field corrections in field-path order.
///
/// # Errors
///
/// Returns a database error when a query fails.
async fn load_anime_corrections(
    tx: &mut Transaction<'_, Postgres>,
    anime_id: Uuid,
) -> Result<Vec<FieldCorrectionRecord>, Error> {
    let correction_rows = sqlx::query!(
        r#"
        SELECT field_path, value_message, value, approved_by, approved_at
        FROM anime_field_correction
        WHERE anime_id = $1
        ORDER BY field_path
        "#,
        anime_id
    )
    .fetch_all(&mut **tx)
    .await?;

    Ok(correction_rows
        .into_iter()
        .map(|row| FieldCorrectionRecord {
            field_path: row.field_path,
            value_message: row.value_message,
            value: row.value,
            approved_by: row.approved_by,
            approved_at: row.approved_at,
        })
        .collect())
}

/// Saves a merged work if `expected_version` matches and its hash changed.
///
/// # Examples
///
/// ```no_run
/// # use sqlx::PgPool;
/// # use postgres_db::{save_anime, SaveAnimeInput};
/// # async fn example(pool: &PgPool, input: &SaveAnimeInput) -> Result<(), postgres_db::Error> {
/// let outcome = save_anime(pool, input).await?;
/// assert!(outcome.version >= 1);
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns [`Error::is_invalid_argument`] for out-of-range fields,
/// [`Error::is_not_found`] when the work does not exist or is deleted, or
/// [`Error::is_version_conflict`] when the stored version differs from
/// `expected_version`.
pub async fn save_anime(pool: &PgPool, input: &SaveAnimeInput) -> Result<SaveAnimeOutcome, Error> {
    validate_save_input(input)?;

    let mut tx = pool.begin().await?;
    let outcome = save_anime_in_tx(&mut tx, input).await?;
    tx.commit().await?;

    tracing::event!(
        name: "postgres_db.anime.saved",
        Level::INFO,
        anime.id = %input.anime_id,
        anime.version = outcome.version,
        anime.has_changed = outcome.has_changed,
        "saved anime {{anime.id}} at version {{anime.version}}",
    );

    Ok(outcome)
}

/// Saves a merged work inside `tx` when its version matches and its hash changed.
///
/// An unchanged hash clears `merge_requested_at` and records `merge_version`, so the
/// entity stops being due for a merge-code upgrade, and writes nothing else.
///
/// # Errors
///
/// Returns a not-found error for a missing or deleted work, a version conflict when
/// `expected_version` is stale, an invalid-argument error for invalid child rows, or a
/// database error when a query fails.
async fn save_anime_in_tx(
    tx: &mut Transaction<'_, Postgres>,
    input: &SaveAnimeInput,
) -> Result<SaveAnimeOutcome, Error> {
    let existing_anime = sqlx::query!(
        r#"
        SELECT
            version,
            status,
            merged_at,
            merged_record_hash,
            hash_format
        FROM anime
        WHERE id = $1 AND deleted_at IS NULL
        FOR NO KEY UPDATE
        "#,
        input.anime_id
    )
    .fetch_optional(&mut **tx)
    .await?
    .ok_or_else(Error::not_found)?;

    if existing_anime.version != input.expected_version {
        return Err(Error::version_conflict(format!(
            "expected version {}, stored version {}",
            input.expected_version, existing_anime.version
        )));
    }

    if existing_anime.merged_record_hash.as_deref() == Some(input.merged_record_hash.as_str())
        && existing_anime.hash_format == Some(input.hash_format)
    {
        sqlx::query!(
            r#"
            UPDATE anime
            SET merge_requested_at = NULL,
                merge_version = $2
            WHERE id = $1
              AND (merge_requested_at IS NOT NULL OR merge_version IS DISTINCT FROM $2)
            "#,
            input.anime_id,
            input.merge_version
        )
        .execute(&mut **tx)
        .await?;

        return Ok(SaveAnimeOutcome {
            has_changed: false,
            version: existing_anime.version,
            created_episode_ids: Vec::new(),
            updated_episode_ids: Vec::new(),
        });
    }

    let next_version = existing_anime.version + 1;
    let now = read_transaction_time(tx).await?;
    let is_first_merge = existing_anime.merged_at.is_none();

    update_anime_row(tx, input, next_version, now).await?;
    replace_anime_details(tx, input.anime_id, &input.details).await?;
    replace_schedule_and_stats(tx, input.anime_id, &input.schedule_and_stats).await?;
    replace_relations_and_graph(tx, input.anime_id, &input.relations_and_graph).await?;
    let SyncWorkEpisodesOutcome {
        created_episode_ids,
        updated_episode_ids,
    } = sync_work_episodes(tx, input.anime_id, &input.episodes).await?;
    emit_anime_outbox_events(
        tx,
        input,
        next_version,
        now,
        is_first_merge,
        existing_anime.status.as_deref(),
    )
    .await?;

    Ok(SaveAnimeOutcome {
        has_changed: true,
        version: next_version,
        created_episode_ids,
        updated_episode_ids,
    })
}

/// Writes the work's own columns, its new version and its merge metadata.
///
/// # Errors
///
/// Returns a database error when a query fails.
async fn update_anime_row(
    tx: &mut Transaction<'_, Postgres>,
    input: &SaveAnimeInput,
    next_version: i64,
    now: DateTime<Utc>,
) -> Result<(), Error> {
    sqlx::query!(
        r#"
        UPDATE anime
        SET
            title = $2,
            title_english = $3,
            title_japanese = $4,
            type = $5,
            status = $6,
            rating = $7,
            season = $8,
            source_material = $9,
            year = $10,
            month = $11,
            episode_count = $12,
            duration_seconds = $13,
            is_nsfw = $14,
            country_of_origin = $15,
            synopsis = $16,
            background = $17,
            aired_from = $18,
            aired_to = $19,
            score_weighted = $20::float8::numeric(4, 2),
            score_mean = $21::float8::numeric(4, 2),
            score_median = $22::float8::numeric(4, 2),
            version = $23,
            merged_record_hash = $24,
            hash_format = $25,
            merge_version = $26,
            merge_requested_at = NULL,
            merged_at = COALESCE(merged_at, $27),
            updated_at = $27
        WHERE id = $1
        "#,
        input.anime_id,
        input.title,
        input.title_english,
        input.title_japanese,
        input.anime_type,
        input.status,
        input.rating,
        input.season,
        input.source_material,
        input.year,
        input.month,
        input.episode_count,
        input.duration_seconds,
        input.is_nsfw,
        input.country_of_origin,
        input.synopsis,
        input.background,
        input.aired_from,
        input.aired_to,
        input.score_weighted,
        input.score_mean,
        input.score_median,
        next_version,
        input.merged_record_hash,
        input.hash_format,
        input.merge_version,
        now,
    )
    .execute(&mut **tx)
    .await?;
    Ok(())
}

/// Writes the work's catalogue event, and its status notification when the status changed.
///
/// The first merge writes `anime.created`, and later merges write `anime.updated`.
///
/// # Errors
///
/// Returns an invalid-argument error for an unknown status, or a database error when
/// an insert fails.
async fn emit_anime_outbox_events(
    tx: &mut Transaction<'_, Postgres>,
    input: &SaveAnimeInput,
    next_version: i64,
    now: DateTime<Utc>,
    is_first_merge: bool,
    previous_status: Option<&str>,
) -> Result<(), Error> {
    let catalogue_subject = if is_first_merge {
        "anime.created"
    } else {
        "anime.updated"
    };
    let catalogue_event_id = Uuid::now_v7();
    let catalogue_payload =
        encode_catalogue_event(catalogue_event_id, now, input.anime_id, next_version);
    insert_outbox_row(
        tx,
        catalogue_event_id,
        catalogue_subject,
        input.anime_id,
        next_version,
        &catalogue_payload,
        input.traceparent.as_deref(),
    )
    .await?;

    if !is_first_merge
        && let Some(prev_status_str) = previous_status
        && prev_status_str != input.status
    {
        let prev_status = status_to_proto_i32(prev_status_str)?;
        let curr_status = status_to_proto_i32(&input.status)?;
        let notif_event_id = Uuid::now_v7();
        let notif_payload = encode_status_changed_notification(
            notif_event_id,
            now,
            input.anime_id,
            &input.title,
            prev_status,
            curr_status,
        );
        insert_outbox_row(
            tx,
            notif_event_id,
            "notification.anime.status.changed",
            input.anime_id,
            next_version,
            &notif_payload,
            input.traceparent.as_deref(),
        )
        .await?;
    }
    Ok(())
}

/// Inserts one outbox row with its payload and trace context.
///
/// # Errors
///
/// Returns a database error when a query fails.
async fn insert_outbox_row(
    tx: &mut Transaction<'_, Postgres>,
    event_id: Uuid,
    subject: &str,
    entity_id: Uuid,
    version: i64,
    payload: &[u8],
    traceparent: Option<&str>,
) -> Result<(), Error> {
    sqlx::query!(
        r#"
        INSERT INTO outbox (id, subject, entity_id, version, payload, traceparent)
        VALUES ($1, $2, $3, $4, $5, $6)
        "#,
        event_id,
        subject,
        entity_id,
        version,
        payload,
        traceparent,
    )
    .execute(&mut **tx)
    .await?;
    Ok(())
}

/// Checks that `country` is an ISO 3166-1 alpha-2 code.
///
/// A code is two uppercase ASCII letters, such as `JP`.
fn is_country_code(country: &str) -> bool {
    country.len() == 2 && country.bytes().all(|byte| byte.is_ascii_uppercase())
}

/// Checks a work save's own fields and every child list before any write.
///
/// # Errors
///
/// Returns an invalid-argument error naming the first field that is out of range or empty.
fn validate_save_input(input: &SaveAnimeInput) -> Result<(), Error> {
    if input.title.trim().is_empty() {
        return Err(Error::invalid_argument("anime title must not be empty"));
    }
    if let Some(year) = input.year
        && year < MIN_ANIME_YEAR
    {
        return Err(Error::invalid_argument(format!(
            "year {year} must be >= {MIN_ANIME_YEAR}"
        )));
    }
    if let Some(month) = input.month
        && !(MIN_MONTH..=MAX_MONTH).contains(&month)
    {
        return Err(Error::invalid_argument(format!(
            "month {month} must be between 1 and 12"
        )));
    }
    if input.episode_count < 0 {
        return Err(Error::invalid_argument("episode_count must be >= 0"));
    }
    if let Some(duration) = input.duration_seconds
        && duration < 0
    {
        return Err(Error::invalid_argument("duration_seconds must be >= 0"));
    }
    if let Some(country) = input.country_of_origin.as_deref()
        && !is_country_code(country)
    {
        return Err(Error::invalid_argument(
            "country_of_origin must be an ISO 3166-1 alpha-2 code of two uppercase letters",
        ));
    }
    for (name, score) in [
        ("score_weighted", input.score_weighted),
        ("score_mean", input.score_mean),
        ("score_median", input.score_median),
    ] {
        if let Some(val) = score
            && !(MIN_SCORE..=MAX_SCORE).contains(&val)
        {
            return Err(Error::invalid_argument(format!(
                "{name} must be between 0 and 10"
            )));
        }
    }
    validate_anime_details(&input.details)?;
    validate_schedule_and_stats(&input.schedule_and_stats)?;
    validate_relations_and_graph(&input.relations_and_graph)?;
    validate_work_episode_references(&input.episodes)?;
    Ok(())
}

/// Writes a `CatalogueEvent` outbox row for a work or an episode.
///
/// Episode events carry their work's ID in `anime_id`.
///
/// # Errors
///
/// Returns a database error when a query fails.
pub(crate) async fn insert_catalogue_outbox_event(
    tx: &mut Transaction<'_, Postgres>,
    subject: &str,
    entity_id: Uuid,
    version: i64,
    anime_id: Option<Uuid>,
    saved_at: DateTime<Utc>,
    traceparent: Option<&str>,
) -> Result<(), Error> {
    let event_id = Uuid::now_v7();
    let payload = ProtoCatalogueEvent {
        event_id: event_id.to_string(),
        saved_at: Some(to_proto_timestamp(saved_at)),
        entity_id: entity_id.to_string(),
        version,
        anime_id: anime_id.map(|id| id.to_string()),
    }
    .encode_to_vec();
    insert_outbox_row(
        tx,
        event_id,
        subject,
        entity_id,
        version,
        &payload,
        traceparent,
    )
    .await
}

/// Converts stored `page_kind` lookup text into a [`PageKind`].
///
/// # Errors
///
/// Returns an invalid-argument error for text no migration defines.
pub(crate) fn parse_page_kind(page_kind_text: &str) -> Result<PageKind, Error> {
    match page_kind_text {
        "anime" => Ok(PageKind::Anime),
        "episode" => Ok(PageKind::Episode),
        "character" => Ok(PageKind::Character),
        other => Err(Error::invalid_argument(format!(
            "unknown stored page_kind {other:?}"
        ))),
    }
}

/// Converts stored `anime_status` lookup text into its proto enum number.
///
/// # Errors
///
/// Returns an invalid-argument error for text no migration defines.
fn status_to_proto_i32(status: &str) -> Result<i32, Error> {
    match status {
        "FINISHED" => Ok(1),
        "UPCOMING" => Ok(2),
        "ONGOING" => Ok(3),
        "UNKNOWN" => Ok(4),
        "CANCELLED" => Ok(5),
        other => Err(Error::invalid_argument(format!(
            "unknown anime_status value {other:?}"
        ))),
    }
}

/// Decompresses a stored gzip response body.
///
/// # Errors
///
/// Returns an invalid-argument error when the body is not valid gzip.
pub(crate) fn decompress_body(compressed: &[u8]) -> Result<Vec<u8>, Error> {
    let mut decoder = GzDecoder::new(compressed);
    let mut out = Vec::new();
    decoder
        .read_to_end(&mut out)
        .map_err(|err| Error::invalid_argument(format!("invalid gzip page body: {err}")))?;
    Ok(out)
}

/// Converts a UTC timestamp into a protobuf `Timestamp`.
pub(crate) fn to_proto_timestamp(timestamp: DateTime<Utc>) -> ProtoTimestamp {
    ProtoTimestamp {
        seconds: timestamp.timestamp(),
        nanos: i32::try_from(timestamp.timestamp_subsec_nanos()).unwrap_or(0),
    }
}

/// Encodes a `CatalogueEvent` payload for an outbox row.
fn encode_catalogue_event(
    event_id: Uuid,
    saved_at: DateTime<Utc>,
    entity_id: Uuid,
    version: i64,
) -> Vec<u8> {
    ProtoCatalogueEvent {
        event_id: event_id.to_string(),
        saved_at: Some(to_proto_timestamp(saved_at)),
        entity_id: entity_id.to_string(),
        version,
        anime_id: None,
    }
    .encode_to_vec()
}

/// Encodes an `AnimeStatusChangedNotification` payload for an outbox row.
fn encode_status_changed_notification(
    event_id: Uuid,
    saved_at: DateTime<Utc>,
    anime_id: Uuid,
    anime_title: &str,
    previous_status: i32,
    current_status: i32,
) -> Vec<u8> {
    ProtoAnimeStatusChangedNotification {
        event_id: event_id.to_string(),
        saved_at: Some(to_proto_timestamp(saved_at)),
        anime_id: anime_id.to_string(),
        anime_title: anime_title.to_owned(),
        previous_status,
        current_status,
    }
    .encode_to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider_page::{
        FetchOutcome, PageKey, PageObservation, PageResponse, Provider, record_page_observation,
    };
    use crate::{AnimeImageInput, AnimeThemeInput, AnimeTitleInput};

    fn sample_save_input(anime_id: Uuid, expected_version: i64) -> SaveAnimeInput {
        SaveAnimeInput {
            anime_id,
            expected_version,
            title: "Monster".to_owned(),
            title_english: Some("Monster".to_owned()),
            title_japanese: Some("モンスター".to_owned()),
            anime_type: "TV".to_owned(),
            status: "FINISHED".to_owned(),
            rating: "R - 17+ (violence & profanity)".to_owned(),
            season: Some("SPRING".to_owned()),
            source_material: Some("MANGA".to_owned()),
            year: Some(2004),
            month: Some(4),
            episode_count: 74,
            duration_seconds: Some(1440),
            is_nsfw: Some(false),
            country_of_origin: Some("JP".to_owned()),
            synopsis: Some("Dr. Kenzo Tenma saves a young boy's life.".to_owned()),
            background: Some("Adapted from Naoki Urasawa's manga.".to_owned()),
            aired_from: None,
            aired_to: None,
            score_weighted: Some(8.88),
            score_mean: Some(8.85),
            score_median: Some(9.0),
            episodes: Vec::new(),
            merged_record_hash: "hash_v1".to_owned(),
            hash_format: ANIME_HASH_FORMAT,
            merge_version: 1,
            traceparent: Some("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01".to_owned()),
            details: AnimeDetailsInput {
                titles: vec![AnimeTitleInput {
                    language: "en".to_owned(),
                    title: "Monster".to_owned(),
                }],
                synonyms: vec!["MONSTER".to_owned()],
                genres: vec!["Mystery".to_owned(), "Thriller".to_owned()],
                tags: vec!["Psychological".to_owned()],
                demographics: vec!["Seinen".to_owned()],
                content_warnings: vec!["Violence".to_owned()],
                themes: vec![AnimeThemeInput {
                    name: "Psychological".to_owned(),
                    description: None,
                }],
                images: vec![AnimeImageInput {
                    image_type: "cover".to_owned(),
                    url: "https://example.com/monster.jpg".to_owned(),
                }],
            },
            schedule_and_stats: AnimeScheduleAndStatsInput::default(),
            relations_and_graph: AnimeRelationsAndGraphInput::default(),
        }
    }

    #[test]
    fn validate_save_input_checks_all_bounds() {
        let base = sample_save_input(Uuid::now_v7(), 1);
        assert!(validate_save_input(&base).is_ok());

        let mut empty_title = base.clone();
        empty_title.title = "   ".to_owned();
        assert!(matches!(validate_save_input(&empty_title), Err(e) if e.is_invalid_argument()));

        let mut bad_year = base.clone();
        bad_year.year = Some(1899);
        assert!(matches!(validate_save_input(&bad_year), Err(e) if e.is_invalid_argument()));

        let mut low_month = base.clone();
        low_month.month = Some(0);
        assert!(matches!(validate_save_input(&low_month), Err(e) if e.is_invalid_argument()));

        let mut high_month = base.clone();
        high_month.month = Some(13);
        assert!(matches!(validate_save_input(&high_month), Err(e) if e.is_invalid_argument()));

        let mut negative_episodes = base.clone();
        negative_episodes.episode_count = -1;
        assert!(matches!(
            validate_save_input(&negative_episodes),
            Err(e) if e.is_invalid_argument()
        ));

        let mut negative_duration = base.clone();
        negative_duration.duration_seconds = Some(-10);
        assert!(matches!(
            validate_save_input(&negative_duration),
            Err(e) if e.is_invalid_argument()
        ));

        let mut long_country = base.clone();
        long_country.country_of_origin = Some("JPN".to_owned());
        assert!(matches!(validate_save_input(&long_country), Err(e) if e.is_invalid_argument()));

        let mut lowercase_country = base.clone();
        lowercase_country.country_of_origin = Some("jp".to_owned());
        assert!(matches!(
            validate_save_input(&lowercase_country),
            Err(e) if e.is_invalid_argument()
        ));

        let mut nan_score = base.clone();
        nan_score.score_weighted = Some(f64::NAN);
        assert!(matches!(validate_save_input(&nan_score), Err(e) if e.is_invalid_argument()));

        let mut negative_score = base.clone();
        negative_score.score_mean = Some(-0.1);
        assert!(matches!(validate_save_input(&negative_score), Err(e) if e.is_invalid_argument()));

        let mut high_score = base;
        high_score.score_median = Some(10.1);
        assert!(matches!(validate_save_input(&high_score), Err(e) if e.is_invalid_argument()));
    }

    #[test]
    fn helpers_reject_unknown_values_and_corrupt_gzip() {
        assert_eq!(parse_page_kind("anime").ok(), Some(PageKind::Anime));
        assert_eq!(parse_page_kind("episode").ok(), Some(PageKind::Episode));
        assert_eq!(parse_page_kind("character").ok(), Some(PageKind::Character));
        assert!(matches!(parse_page_kind("unknown"), Err(e) if e.is_invalid_argument()));

        assert_eq!(status_to_proto_i32("FINISHED").ok(), Some(1));
        assert_eq!(status_to_proto_i32("UPCOMING").ok(), Some(2));
        assert_eq!(status_to_proto_i32("ONGOING").ok(), Some(3));
        assert_eq!(status_to_proto_i32("UNKNOWN").ok(), Some(4));
        assert_eq!(status_to_proto_i32("CANCELLED").ok(), Some(5));
        assert!(matches!(status_to_proto_i32("BOGUS"), Err(e) if e.is_invalid_argument()));

        assert!(matches!(decompress_body(b"not-gzip"), Err(e) if e.is_invalid_argument()));
    }

    #[test]
    fn proto_encoders_roundtrip_wire_format() {
        let now = DateTime::from_timestamp(1_700_000_000, 123_456_789).unwrap();
        let event_id = Uuid::now_v7();
        let anime_id = Uuid::now_v7();

        let catalogue_payload = encode_catalogue_event(event_id, now, anime_id, 7);
        let decoded_catalogue_event =
            ProtoCatalogueEvent::decode(catalogue_payload.as_slice()).unwrap();
        assert_eq!(decoded_catalogue_event.event_id, event_id.to_string());
        assert_eq!(decoded_catalogue_event.entity_id, anime_id.to_string());
        assert_eq!(decoded_catalogue_event.version, 7);
        assert_eq!(decoded_catalogue_event.anime_id, None);
        assert_eq!(
            decoded_catalogue_event.saved_at,
            Some(ProtoTimestamp {
                seconds: 1_700_000_000,
                nanos: 123_456_789,
            })
        );

        let notif_bytes =
            encode_status_changed_notification(event_id, now, anime_id, "Monster", 3, 1);
        let decoded_notif =
            ProtoAnimeStatusChangedNotification::decode(notif_bytes.as_slice()).unwrap();
        assert_eq!(decoded_notif.anime_id, anime_id.to_string());
        assert_eq!(decoded_notif.anime_title, "Monster");
        assert_eq!(decoded_notif.previous_status, 3);
        assert_eq!(decoded_notif.current_status, 1);
    }

    #[sqlx::test]
    async fn load_and_save_anime_full_lifecycle(pool: PgPool) {
        let missing_id = Uuid::now_v7();
        assert!(matches!(load_anime(&pool, missing_id).await, Err(e) if e.is_not_found()));
        assert!(matches!(
            save_anime(&pool, &sample_save_input(missing_id, 1)).await,
            Err(e) if e.is_not_found()
        ));

        let anime_id = Uuid::now_v7();
        sqlx::query!(
            "INSERT INTO anime (id, title, type, status, rating, version, merge_requested_at)
             VALUES ($1, 'Placeholder', 'TV', 'ONGOING', 'PG-13 - Teens 13 or older', 1, now())",
            anime_id,
        )
        .execute(&pool)
        .await
        .unwrap();

        let mal_key = PageKey::new(Provider::Mal, PageKind::Anime, "19").unwrap();
        record_page_observation(
            &pool,
            PageObservation {
                page_key: mal_key.clone(),
                observed_at: Utc::now(),
                outcome: FetchOutcome::Success {
                    responses: vec![
                        PageResponse {
                            request_name: "zeta".to_owned(),
                            content_type: "application/json".to_owned(),
                            body: br#"{"z":1}"#.to_vec(),
                        },
                        PageResponse {
                            request_name: "alpha".to_owned(),
                            content_type: "application/json".to_owned(),
                            body: br#"{"a":2}"#.to_vec(),
                        },
                    ],
                },
            },
        )
        .await
        .unwrap();

        sqlx::query!(
            "INSERT INTO provider_page (provider, page_kind, address) VALUES ('anilist', 'anime', '19')"
        )
        .execute(&pool)
        .await
        .unwrap();

        sqlx::query!(
            "INSERT INTO anime_provider_page (anime_id, provider, page_kind, address, role)
             VALUES ($1, 'mal', 'anime', '19', 'source'),
                    ($1, 'anilist', 'anime', '19', 'partial_source')",
            anime_id,
        )
        .execute(&pool)
        .await
        .unwrap();

        sqlx::query!(
            "INSERT INTO anime_field_correction (anime_id, field_path, value_message, value, approved_by)
             VALUES ($1, 'episode_count', 'google.protobuf.Int32Value', $2, 'admin')",
            anime_id,
            b"\x08\x4a".as_slice(),
        )
        .execute(&pool)
        .await
        .unwrap();

        let loaded = load_anime(&pool, anime_id).await.unwrap();
        assert_eq!(loaded.version, 1);
        assert_eq!(loaded.source_pages.len(), 1);
        assert_eq!(loaded.source_pages[0].responses[0].request_name, "alpha");
        assert_eq!(loaded.source_pages[0].responses[0].body, br#"{"a":2}"#);
        assert_eq!(loaded.source_pages[0].responses[1].request_name, "zeta");
        assert_eq!(loaded.partial_source_pages.len(), 1);
        assert_eq!(loaded.corrections.len(), 1);
        assert_eq!(loaded.corrections[0].field_path, "episode_count");

        let conflict = save_anime(&pool, &sample_save_input(anime_id, 99)).await;
        assert!(matches!(conflict, Err(e) if e.is_version_conflict()));

        let first_input = sample_save_input(anime_id, 1);
        let first_save = save_anime(&pool, &first_input).await.unwrap();
        assert!(first_save.has_changed);
        assert_eq!(first_save.version, 2);

        let subjects: Vec<String> = sqlx::query_scalar!(
            "SELECT subject FROM outbox WHERE entity_id = $1 ORDER BY subject",
            anime_id,
        )
        .fetch_all(&pool)
        .await
        .unwrap();
        assert_eq!(subjects, vec!["anime.created".to_owned()]);

        sqlx::query!(
            "UPDATE anime SET merge_requested_at = now() WHERE id = $1",
            anime_id,
        )
        .execute(&pool)
        .await
        .unwrap();

        let unchanged_input = sample_save_input(anime_id, 2);
        let unchanged_save = save_anime(&pool, &unchanged_input).await.unwrap();
        assert!(!unchanged_save.has_changed);
        assert_eq!(unchanged_save.version, 2);

        let merge_req: Option<DateTime<Utc>> = sqlx::query_scalar!(
            "SELECT merge_requested_at FROM anime WHERE id = $1",
            anime_id,
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(merge_req.is_none());

        let mut second_input = sample_save_input(anime_id, 2);
        second_input.merged_record_hash = "hash_v2".to_owned();
        second_input.status = "CANCELLED".to_owned();
        let second_save = save_anime(&pool, &second_input).await.unwrap();
        assert!(second_save.has_changed);
        assert_eq!(second_save.version, 3);

        let all_subjects: Vec<String> = sqlx::query_scalar!(
            "SELECT subject FROM outbox WHERE entity_id = $1 ORDER BY subject",
            anime_id,
        )
        .fetch_all(&pool)
        .await
        .unwrap();
        assert_eq!(
            all_subjects,
            vec![
                "anime.created".to_owned(),
                "anime.updated".to_owned(),
                "notification.anime.status.changed".to_owned(),
            ]
        );
    }
}
