//! Episode loading, version-checked saving, aired notifications, and work-list sync.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use chrono::{DateTime, Timelike as _, Utc};
use prost::Message as _;
use sqlx::{PgPool, Postgres, Transaction};
use tracing::Level;
use uuid::Uuid;

use crate::anime::{
    FieldCorrectionRecord, LoadedSourcePage, ProtoTimestamp, decompress_body,
    insert_catalogue_outbox_event, to_proto_timestamp,
};
use crate::provider_page::record_page_sources;
use crate::{EntityType, Error, PageKey, PageResponse, PageSource, Provider};
use crate::{begin_read_snapshot, read_transaction_time};

/// Current canonical hash format version stored in `episode.hash_format`.
pub const EPISODE_HASH_FORMAT: i16 = 1;

/// Lowest episode score that `episode.score numeric(5, 2)` cannot store.
///
/// PostgreSQL rounds the score to two decimals, and 999.995 rounds to 1000.00, which needs
/// a fourth integer digit. A score at or above this value fails the save with a numeric
/// overflow, so the save refuses it first. Changing the column's precision changes this value.
const EPISODE_SCORE_OVERFLOW: f64 = 999.995;

/// One provider's facts about an episode, sliced out of a work's page.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpisodeListEntryInput {
    /// Provider whose work page lists the episode.
    pub provider: Provider,
    /// Episode title reported by the provider, if any.
    pub title: Option<String>,
    /// Air timestamp reported by the provider, if known.
    pub aired_at: Option<DateTime<Utc>>,
    /// Episode duration in seconds (`>= 0`), if known.
    pub duration_seconds: Option<i32>,
    /// Filler flag reported by the provider, if known.
    pub is_filler: Option<bool>,
    /// Recap flag reported by the provider, if known.
    pub is_recap: Option<bool>,
}

/// Episode reference passed from `SaveAnime` to create or update episode list state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpisodeReferenceInput {
    /// Episode UUID assigned by the work merge (existing or newly generated `UUIDv7`).
    pub episode_id: Uuid,
    /// Episode number within the work (`>= 0`).
    pub episode_number: i32,
    /// Display title from the work's episode list.
    pub title: String,
    /// Episode provider page keys linked to this episode.
    pub page_keys: Vec<PageKey>,
    /// Per-provider episode list entries from the work's pages.
    pub list_entries: Vec<EpisodeListEntryInput>,
}

/// Localized episode title stored in `episode_title`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpisodeTitleInput {
    /// BCP 47 language code (primary key component).
    pub language: String,
    /// Localized title text.
    pub title: String,
}

/// Episode streaming link stored in `episode_streaming`, with its platform in
/// `external_source`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpisodeStreamingInput {
    /// Streaming platform name, the key of `Episode.streaming`.
    pub platform: String,
    /// Direct episode URL on `platform`.
    pub source: String,
}

/// Character appearance in an episode stored in `episode_character`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpisodeCharacterInput {
    /// Provider of the character's page.
    pub character_page_provider: String,
    /// Address of the character's page on `character_page_provider`.
    pub character_page_address: String,
    /// Canonical `character_role` lookup value (for example `"MAIN"`).
    pub role: String,
}

/// Merge inputs returned by [`load_episode`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadEpisodeOutcome {
    /// Current optimistic-locking version of the episode.
    pub version: i64,
    /// Owning work UUID.
    pub anime_id: Uuid,
    /// Episode number the work's episode list assigned, which the merge keeps.
    pub episode_number: i32,
    /// Linked episode provider pages with their raw responses.
    pub pages: Vec<LoadedSourcePage>,
    /// Approved per-field corrections to apply after merging.
    pub corrections: Vec<FieldCorrectionRecord>,
    /// Per-provider episode list entries from the work's merge.
    pub list_entries: Vec<EpisodeListEntryInput>,
}

/// Canonical episode columns and merge metadata passed to [`save_episode`].
#[derive(Debug, Clone, PartialEq)]
pub struct SaveEpisodeInput {
    /// UUID of the existing episode row to update.
    pub episode_id: Uuid,
    /// Optimistic concurrency version expected in PostgreSQL.
    pub expected_version: i64,
    /// Episode number (`>= 0`).
    pub episode_number: i32,
    /// Season number (`> 0`), if known.
    pub season_number: Option<i32>,
    /// Primary display title, blank while no provider has one; stored as NULL then.
    pub title: String,
    /// Japanese title, if known.
    pub title_japanese: Option<String>,
    /// Romaji title, if known.
    pub title_romaji: Option<String>,
    /// Synopsis text, if known.
    pub synopsis: Option<String>,
    /// Extended description text, if known.
    pub description: Option<String>,
    /// Air timestamp, if known.
    pub aired_at: Option<DateTime<Utc>>,
    /// Duration in seconds (`>= 0`), if known.
    pub duration_seconds: Option<i32>,
    /// Whether the episode is filler.
    pub is_filler: bool,
    /// Whether the episode is a recap.
    pub is_recap: bool,
    /// Episode score (`>= 0`), if known.
    pub score: Option<f64>,
    /// Localized titles.
    pub titles: Vec<EpisodeTitleInput>,
    /// Episode image URLs.
    pub images: Vec<String>,
    /// Episode streaming links.
    pub streaming: Vec<EpisodeStreamingInput>,
    /// Character appearances in this episode.
    pub characters: Vec<EpisodeCharacterInput>,
    /// Canonical hash of the merged episode record.
    pub merged_record_hash: String,
    /// Format identifier of `merged_record_hash`.
    pub hash_format: i16,
    /// Version of the mapping and merge code that produced this save.
    pub merge_version: i32,
    /// Optional W3C `traceparent` header propagated into outbox rows.
    pub traceparent: Option<String>,
    /// URL the merge chose for each page the entity holds as a source.
    pub page_sources: Vec<PageSource>,
}

/// Result returned by [`save_episode`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SaveEpisodeOutcome {
    /// Whether the canonical hash differed and database state was updated.
    pub has_changed: bool,
    /// Episode version after the save attempt.
    pub version: i64,
}

/// Result returned by [`mark_episode_aired`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MarkEpisodeAiredOutcome {
    /// Whether `notification.episode.aired` was written by this call.
    pub has_notified: bool,
}

#[derive(PartialEq, prost::Message)]
struct ProtoEpisodeAiredNotification {
    #[prost(string, tag = "1")]
    event_id: String,
    #[prost(message, optional, tag = "2")]
    saved_at: Option<ProtoTimestamp>,
    #[prost(string, tag = "3")]
    anime_id: String,
    #[prost(string, tag = "4")]
    anime_title: String,
    #[prost(string, tag = "5")]
    episode_id: String,
    #[prost(int32, tag = "6")]
    episode_number: i32,
    #[prost(message, optional, tag = "7")]
    aired_at: Option<ProtoTimestamp>,
}

/// The episodes a work's save created and the existing ones whose list entries changed.
#[derive(Debug)]
pub(crate) struct SyncAnimeEpisodesOutcome {
    /// Episodes the save created, to refresh next.
    pub created_episode_ids: Vec<Uuid>,
    /// Existing episodes whose list entries changed.
    pub updated_episode_ids: Vec<Uuid>,
}

/// Loads an episode's version, linked pages, list entries, and approved corrections.
///
/// # Examples
///
/// ```no_run
/// # use sqlx::PgPool;
/// # use uuid::Uuid;
/// # use postgres_db::load_episode;
/// # async fn example(
/// #     pool: &PgPool,
/// #     episode_id: Uuid,
/// #     anime_id: Uuid,
/// # ) -> Result<(), postgres_db::Error> {
/// // An episode its work's save just created, before the episode's own first merge.
/// let outcome = load_episode(pool, episode_id).await?;
/// assert_eq!(outcome.version, 1);
/// assert_eq!(outcome.anime_id, anime_id);
/// assert_eq!(outcome.pages, Vec::new(), "its pages are linked but not fetched yet");
/// # Ok(())
/// # }
/// ```
/// # Errors
///
/// Returns [`Error::is_not_found`] when `episode_id` does not exist or is
/// deleted, or a database error when any query or decompression fails.
pub async fn load_episode(pool: &PgPool, episode_id: Uuid) -> Result<LoadEpisodeOutcome, Error> {
    let mut tx = begin_read_snapshot(pool).await?;

    let episode_row = sqlx::query!(
        r#"
        SELECT version, anime_id, episode_number
        FROM episode
        WHERE id = $1 AND deleted_at IS NULL
        "#,
        episode_id
    )
    .fetch_optional(&mut *tx)
    .await?
    .ok_or_else(Error::not_found)?;

    let pages = load_episode_pages(&mut tx, episode_id).await?;
    let corrections = load_episode_corrections(&mut tx, episode_id).await?;
    let list_entries = load_episode_list_entries(&mut tx, episode_id).await?;
    tx.commit().await?;

    Ok(LoadEpisodeOutcome {
        version: episode_row.version,
        anime_id: episode_row.anime_id,
        episode_number: episode_row.episode_number,
        pages,
        corrections,
        list_entries,
    })
}

/// Loads an episode's fetched pages with their responses.
///
/// A linked page that was never fetched is left out, since the merge has nothing to read.
///
/// # Errors
///
/// Returns a database error when a query fails or a body cannot be decompressed, or an
/// invalid-argument error when a stored page key no longer passes validation.
async fn load_episode_pages(
    tx: &mut Transaction<'_, Postgres>,
    episode_id: Uuid,
) -> Result<Vec<LoadedSourcePage>, Error> {
    let linked_rows = sqlx::query!(
        r#"
        SELECT
            source.provider,
            source.entity_type,
            source.address,
            pp.fetched_at,
            pp.is_dead,
            pp.missing_since
        FROM episode_source source
        JOIN provider_page pp
          ON pp.provider = source.provider
         AND pp.entity_type = source.entity_type
         AND pp.address = source.address
        WHERE source.episode_id = $1
        ORDER BY source.provider, source.address
        "#,
        episode_id
    )
    .fetch_all(&mut **tx)
    .await?;

    let mut responses_by_page = load_episode_responses_by_page(tx, episode_id).await?;
    let mut pages = Vec::with_capacity(linked_rows.len());

    for row in linked_rows {
        let Some(fetched_at) = row.fetched_at else {
            continue;
        };
        let provider: Provider = row.provider.parse()?;
        let entity_type: EntityType = row.entity_type.parse()?;
        let page_key = PageKey::new(provider, entity_type, &row.address)?;
        let responses = responses_by_page
            .remove(&(row.provider, row.address))
            .unwrap_or_default();
        pages.push(LoadedSourcePage {
            page_key,
            fetched_at,
            is_dead: row.is_dead,
            is_not_found: row.missing_since.is_some(),
            responses,
        });
    }

    Ok(pages)
}

/// Loads the decompressed responses of an episode's pages, grouped by page.
///
/// # Errors
///
/// Returns a database error when the query fails or a stored body cannot be decompressed.
async fn load_episode_responses_by_page(
    tx: &mut Transaction<'_, Postgres>,
    episode_id: Uuid,
) -> Result<HashMap<(String, String), Vec<PageResponse>>, Error> {
    let response_rows = sqlx::query!(
        r#"
        SELECT
            ppr.provider,
            ppr.address,
            ppr.request_name,
            ppr.content_type,
            ppr.body
        FROM episode_source source
        JOIN provider_page_response ppr
          ON ppr.provider = source.provider
         AND ppr.entity_type = source.entity_type
         AND ppr.address = source.address
        WHERE source.episode_id = $1
        ORDER BY ppr.provider, ppr.address, ppr.request_name
        "#,
        episode_id
    )
    .fetch_all(&mut **tx)
    .await?;

    let mut responses_by_page: HashMap<(String, String), Vec<PageResponse>> = HashMap::new();
    for row in response_rows {
        let body = decompress_body(&row.body)?;
        responses_by_page
            .entry((row.provider, row.address))
            .or_default()
            .push(PageResponse {
                request_name: row.request_name,
                content_type: row.content_type,
                body,
            });
    }
    Ok(responses_by_page)
}

/// Loads an episode's approved field corrections in field-path order.
///
/// # Errors
///
/// Returns a database error when a query fails.
async fn load_episode_corrections(
    tx: &mut Transaction<'_, Postgres>,
    episode_id: Uuid,
) -> Result<Vec<FieldCorrectionRecord>, Error> {
    let rows = sqlx::query!(
        r#"
        SELECT
            field_path,
            value_message,
            value,
            approved_by,
            approved_at
        FROM episode_field_correction
        WHERE episode_id = $1
        ORDER BY field_path
        "#,
        episode_id
    )
    .fetch_all(&mut **tx)
    .await?;

    Ok(rows
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

/// Loads what each provider's work page says about an episode.
///
/// # Errors
///
/// Returns a database error when the query fails, or an invalid-argument error for a
/// stored provider that is not a known provider.
async fn load_episode_list_entries(
    tx: &mut Transaction<'_, Postgres>,
    episode_id: Uuid,
) -> Result<Vec<EpisodeListEntryInput>, Error> {
    let rows = sqlx::query!(
        r#"
        SELECT
            provider,
            title,
            aired_at,
            duration_seconds,
            is_filler,
            is_recap
        FROM episode_list_entry
        WHERE episode_id = $1
        ORDER BY provider
        "#,
        episode_id
    )
    .fetch_all(&mut **tx)
    .await?;

    rows.into_iter()
        .map(|row| {
            Ok(EpisodeListEntryInput {
                provider: row.provider.parse()?,
                title: row.title,
                aired_at: row.aired_at,
                duration_seconds: row.duration_seconds,
                is_filler: row.is_filler,
                is_recap: row.is_recap,
            })
        })
        .collect()
}

/// Validates and persists a merged episode when its canonical hash changed.
///
/// An unchanged hash clears `merge_requested_at` and records `merge_version`, so the
/// episode stops being due for a merge-code upgrade, and writes nothing else.
///
/// # Examples
///
/// ```no_run
/// # use sqlx::PgPool;
/// # use uuid::Uuid;
/// # use postgres_db::{EPISODE_HASH_FORMAT, SaveEpisodeInput, SaveEpisodeOutcome, save_episode};
/// # async fn example(pool: &PgPool, episode_id: Uuid) -> Result<(), postgres_db::Error> {
/// let input = SaveEpisodeInput {
///     episode_id,
///     expected_version: 1,
///     episode_number: 1,
///     season_number: Some(1),
///     title: "Asteroid Blues".to_owned(),
///     title_japanese: None,
///     title_romaji: None,
///     synopsis: None,
///     description: None,
///     aired_at: None,
///     duration_seconds: Some(1440),
///     is_filler: false,
///     is_recap: false,
///     score: Some(8.75),
///     titles: Vec::new(),
///     images: Vec::new(),
///     streaming: Vec::new(),
///     characters: Vec::new(),
///     merged_record_hash: "blake3-hash-value".to_owned(),
///     hash_format: EPISODE_HASH_FORMAT,
///     merge_version: 1,
///     traceparent: None,
///     page_sources: Vec::new(),
/// };
/// let first_save = save_episode(pool, &input).await?;
/// assert_eq!(first_save, SaveEpisodeOutcome { has_changed: true, version: 2 });
///
/// // The same merged record saved again writes nothing and keeps the version.
/// let repeated_input = SaveEpisodeInput { expected_version: 2, ..input };
/// let repeated_save = save_episode(pool, &repeated_input).await?;
/// assert_eq!(repeated_save, SaveEpisodeOutcome { has_changed: false, version: 2 });
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns [`Error::is_invalid_argument`] for out-of-range fields,
/// [`Error::is_not_found`] when `episode_id` does not exist or is deleted,
/// [`Error::is_version_conflict`] when `expected_version` does not match, or a
/// database error if persistence fails.
pub async fn save_episode(
    pool: &PgPool,
    input: &SaveEpisodeInput,
) -> Result<SaveEpisodeOutcome, Error> {
    validate_save_episode_input(input)?;

    let mut tx = pool.begin().await?;
    // Page rows, then shared rows, come before the episode's row lock, in the order a work's
    // save writes them.
    record_page_sources(
        &mut tx,
        EntityType::Episode,
        input.episode_id,
        &input.page_sources,
    )
    .await?;
    upsert_episode_shared_rows(&mut tx, input).await?;
    let existing = sqlx::query!(
        r#"
        SELECT
            anime_id,
            version,
            merged_record_hash,
            hash_format,
            merged_at
        FROM episode
        WHERE id = $1 AND deleted_at IS NULL
        FOR UPDATE
        "#,
        input.episode_id
    )
    .fetch_optional(&mut *tx)
    .await?
    .ok_or_else(Error::not_found)?;

    if existing.version != input.expected_version {
        return Err(Error::version_conflict(format!(
            "expected version {}, stored version {}",
            input.expected_version, existing.version
        )));
    }

    if existing.merged_record_hash.as_deref() == Some(input.merged_record_hash.as_str())
        && existing.hash_format == Some(input.hash_format)
    {
        sqlx::query!(
            r#"
            UPDATE episode
            SET merge_requested_at = NULL,
                merge_version = $2
            WHERE id = $1
              AND (merge_requested_at IS NOT NULL OR merge_version IS DISTINCT FROM $2)
            "#,
            input.episode_id,
            input.merge_version
        )
        .execute(&mut *tx)
        .await?;
        tx.commit().await?;

        return Ok(SaveEpisodeOutcome {
            has_changed: false,
            version: existing.version,
        });
    }

    let next_version = existing.version + 1;
    let now = read_transaction_time(&mut tx).await?;
    let is_first_merge = existing.merged_at.is_none();

    update_episode_row(&mut tx, input, next_version, now).await?;
    replace_episode_titles(&mut tx, input.episode_id, &input.titles).await?;
    replace_episode_images(&mut tx, input.episode_id, &input.images).await?;
    replace_episode_streaming(&mut tx, input.episode_id, &input.streaming).await?;
    replace_episode_characters(&mut tx, input.episode_id, &input.characters).await?;

    let subject = if is_first_merge {
        "episode.created"
    } else {
        "episode.updated"
    };
    insert_catalogue_outbox_event(
        &mut tx,
        subject,
        input.episode_id,
        next_version,
        Some(existing.anime_id),
        now,
        input.traceparent.as_deref(),
    )
    .await?;

    tx.commit().await?;
    tracing::event!(
        name: "postgres_db.episode.saved",
        Level::INFO,
        episode.id = %input.episode_id,
        episode.version = next_version,
        outbox.subject = subject,
        "saved episode {{episode.id}} at version {{episode.version}} with {{outbox.subject}}",
    );

    Ok(SaveEpisodeOutcome {
        has_changed: true,
        version: next_version,
    })
}

/// Checks an episode save's fields and child lists before any write.
///
/// # Errors
///
/// Returns an invalid-argument error naming the first field that is out of range or empty.
fn validate_save_episode_input(input: &SaveEpisodeInput) -> Result<(), Error> {
    if input.episode_number < 0 {
        return Err(Error::invalid_argument("episode_number must be >= 0"));
    }
    if input
        .season_number
        .is_some_and(|season_number| season_number <= 0)
    {
        return Err(Error::invalid_argument("season_number must be > 0"));
    }
    if input.duration_seconds.is_some_and(|seconds| seconds < 0) {
        return Err(Error::invalid_argument("duration_seconds must be >= 0"));
    }
    if let Some(score) = input.score
        && !(0.0..EPISODE_SCORE_OVERFLOW).contains(&score)
    {
        return Err(Error::invalid_argument(format!(
            "episode score must be at least 0 and below {EPISODE_SCORE_OVERFLOW}, got {score}"
        )));
    }
    if input.merged_record_hash.is_empty() {
        return Err(Error::invalid_argument(
            "merged_record_hash must not be empty",
        ));
    }
    if input.merge_version < 0 {
        return Err(Error::invalid_argument("merge_version must be >= 0"));
    }
    for title_entry in &input.titles {
        if title_entry.language.trim().is_empty() || title_entry.title.trim().is_empty() {
            return Err(Error::invalid_argument(
                "episode_title language and title must not be empty",
            ));
        }
    }
    for url in &input.images {
        if url.trim().is_empty() {
            return Err(Error::invalid_argument(
                "episode_image url must not be empty",
            ));
        }
    }
    let mut source_by_platform: BTreeMap<&str, &str> = BTreeMap::new();
    let mut platform_by_source: BTreeMap<&str, &str> = BTreeMap::new();
    for streaming_entry in &input.streaming {
        let platform = streaming_entry.platform.as_str();
        let source = streaming_entry.source.as_str();
        if platform.trim().is_empty() || source.trim().is_empty() {
            return Err(Error::invalid_argument(
                "episode_streaming platform and source must not be empty",
            ));
        }
        if source_by_platform
            .insert(platform, source)
            .is_some_and(|known_source| known_source != source)
        {
            return Err(Error::invalid_argument(format!(
                "episode_streaming platform {platform:?} is given two sources"
            )));
        }
        if platform_by_source
            .insert(source, platform)
            .is_some_and(|known_platform| known_platform != platform)
        {
            return Err(Error::invalid_argument(format!(
                "episode_streaming source {source:?} is given two platforms"
            )));
        }
    }
    for character_entry in &input.characters {
        if character_entry.character_page_provider.trim().is_empty()
            || character_entry.character_page_address.trim().is_empty()
            || character_entry.role.trim().is_empty()
        {
            return Err(Error::invalid_argument(
                "episode_character provider, address, and role must not be empty",
            ));
        }
    }
    Ok(())
}

/// Writes the episode's own columns, its new version and its merge metadata.
///
/// The first merge also sets `merged_at`.
///
/// # Errors
///
/// Returns a database error when a query fails.
async fn update_episode_row(
    tx: &mut Transaction<'_, Postgres>,
    input: &SaveEpisodeInput,
    next_version: i64,
    now: DateTime<Utc>,
) -> Result<(), Error> {
    sqlx::query!(
        r#"
        UPDATE episode
        SET
            episode_number = $2,
            season_number = $3,
            title = $4,
            title_japanese = $5,
            title_romaji = $6,
            synopsis = $7,
            description = $8,
            aired_at = $9,
            duration_seconds = $10,
            is_filler = $11,
            is_recap = $12,
            score = $13::float8::numeric(5, 2),
            version = $14,
            merged_record_hash = $15,
            hash_format = $16,
            merge_version = $17,
            merge_requested_at = NULL,
            merged_at = COALESCE(merged_at, $18),
            updated_at = $18
        WHERE id = $1
        "#,
        input.episode_id,
        input.episode_number,
        input.season_number,
        (!input.title.trim().is_empty()).then_some(input.title.as_str()),
        input.title_japanese,
        input.title_romaji,
        input.synopsis,
        input.description,
        input.aired_at,
        input.duration_seconds,
        input.is_filler,
        input.is_recap,
        input.score,
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

/// Replaces an episode's localized titles.
///
/// # Errors
///
/// Returns a database error when a query fails.
async fn replace_episode_titles(
    tx: &mut Transaction<'_, Postgres>,
    episode_id: Uuid,
    titles: &[EpisodeTitleInput],
) -> Result<(), Error> {
    sqlx::query!(
        "DELETE FROM episode_title WHERE episode_id = $1",
        episode_id
    )
    .execute(&mut **tx)
    .await?;

    let deduplicated_titles: BTreeMap<&str, &str> = titles
        .iter()
        .map(|entry| (entry.language.as_str(), entry.title.as_str()))
        .collect();
    if deduplicated_titles.is_empty() {
        return Ok(());
    }

    let languages: Vec<String> = deduplicated_titles
        .keys()
        .copied()
        .map(str::to_owned)
        .collect();
    let title_values: Vec<String> = deduplicated_titles
        .values()
        .copied()
        .map(str::to_owned)
        .collect();
    sqlx::query!(
        "INSERT INTO episode_title (episode_id, language, title)
         SELECT $1, language, title
         FROM UNNEST($2::text[], $3::text[]) AS entry(language, title)",
        episode_id,
        &languages,
        &title_values
    )
    .execute(&mut **tx)
    .await?;
    Ok(())
}

/// Replaces an episode's images.
///
/// # Errors
///
/// Returns a database error when a query fails.
async fn replace_episode_images(
    tx: &mut Transaction<'_, Postgres>,
    episode_id: Uuid,
    images: &[String],
) -> Result<(), Error> {
    sqlx::query!(
        "DELETE FROM episode_image WHERE episode_id = $1",
        episode_id
    )
    .execute(&mut **tx)
    .await?;

    let deduplicated_images: BTreeSet<&str> = images.iter().map(String::as_str).collect();
    if deduplicated_images.is_empty() {
        return Ok(());
    }

    let urls: Vec<String> = deduplicated_images.into_iter().map(str::to_owned).collect();
    sqlx::query!(
        "INSERT INTO episode_image (episode_id, source)
         SELECT $1, source
         FROM UNNEST($2::text[]) AS entry(source)",
        episode_id,
        &urls
    )
    .execute(&mut **tx)
    .await?;
    Ok(())
}

/// Adds the language codes and streaming URLs an episode's save references, before its row is locked.
///
/// A work's save writes the same tables in the same order, language codes and then
/// external sources, before it locks any entity row. Two saves at once, such as a work's
/// and one of its episodes', therefore cannot each hold a new shared row the other waits for.
///
/// # Errors
///
/// Returns a database error when an insert or upsert fails.
async fn upsert_episode_shared_rows(
    tx: &mut Transaction<'_, Postgres>,
    input: &SaveEpisodeInput,
) -> Result<(), Error> {
    crate::language_code::insert_language_codes(
        tx,
        input.titles.iter().map(|entry| entry.language.as_str()),
    )
    .await?;
    let url_details: BTreeMap<&str, (&str, Option<&str>)> = input
        .streaming
        .iter()
        .map(|entry| (entry.source.as_str(), (entry.platform.as_str(), None)))
        .collect();
    crate::anime_schedule_and_stats::upsert_external_sources(tx, url_details).await
}

/// Replaces an episode's streaming links; the save has already stored each URL's platform.
///
/// # Errors
///
/// Returns a database error when a query fails.
async fn replace_episode_streaming(
    tx: &mut Transaction<'_, Postgres>,
    episode_id: Uuid,
    streaming: &[EpisodeStreamingInput],
) -> Result<(), Error> {
    let sources: Vec<String> = streaming
        .iter()
        .map(|entry| entry.source.as_str())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(str::to_owned)
        .collect();

    sqlx::query!(
        "DELETE FROM episode_streaming WHERE episode_id = $1",
        episode_id
    )
    .execute(&mut **tx)
    .await?;
    if sources.is_empty() {
        return Ok(());
    }
    sqlx::query!(
        "INSERT INTO episode_streaming (episode_id, source)
         SELECT $1, source FROM UNNEST($2::text[]) AS entry(source)",
        episode_id,
        &sources
    )
    .execute(&mut **tx)
    .await?;
    Ok(())
}

/// Replaces the characters that appear in an episode.
///
/// # Errors
///
/// Returns a database error when a query fails.
async fn replace_episode_characters(
    tx: &mut Transaction<'_, Postgres>,
    episode_id: Uuid,
    characters: &[EpisodeCharacterInput],
) -> Result<(), Error> {
    sqlx::query!(
        "DELETE FROM episode_character WHERE episode_id = $1",
        episode_id
    )
    .execute(&mut **tx)
    .await?;

    let deduplicated_characters: BTreeMap<(&str, &str), &str> = characters
        .iter()
        .map(|entry| {
            (
                (
                    entry.character_page_provider.as_str(),
                    entry.character_page_address.as_str(),
                ),
                entry.role.as_str(),
            )
        })
        .collect();
    if deduplicated_characters.is_empty() {
        return Ok(());
    }

    let providers: Vec<String> = deduplicated_characters
        .keys()
        .map(|(provider, _)| (*provider).to_owned())
        .collect();
    let addresses: Vec<String> = deduplicated_characters
        .keys()
        .map(|(_, address)| (*address).to_owned())
        .collect();
    let roles: Vec<String> = deduplicated_characters
        .values()
        .copied()
        .map(str::to_owned)
        .collect();
    sqlx::query!(
        "INSERT INTO episode_character (
             episode_id, character_page_provider, character_page_address, role
         )
         SELECT $1, provider, address, role
         FROM UNNEST($2::text[], $3::text[], $4::text[]) AS entry(provider, address, role)",
        episode_id,
        &providers,
        &addresses,
        &roles
    )
    .execute(&mut **tx)
    .await?;
    Ok(())
}

/// Writes `notification.episode.aired` once when `aired_at` has passed and `aired_notified_at` is unset.
///
/// # Examples
///
/// ```no_run
/// # use sqlx::PgPool;
/// # use uuid::Uuid;
/// # use postgres_db::{mark_episode_aired, MarkEpisodeAiredOutcome};
/// # async fn example(pool: &PgPool, episode_id: Uuid) -> Result<(), postgres_db::Error> {
/// // The episode aired an hour ago, and no notification was written yet.
/// let first_call = mark_episode_aired(pool, episode_id, None).await?;
/// assert_eq!(first_call, MarkEpisodeAiredOutcome { has_notified: true });
///
/// // The notification is written once, however often episode tracking asks.
/// let repeated_call = mark_episode_aired(pool, episode_id, None).await?;
/// assert_eq!(repeated_call, MarkEpisodeAiredOutcome { has_notified: false });
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns [`Error::is_not_found`] when `episode_id` (or its parent work) does
/// not exist or is deleted, or a database error when the transaction fails.
pub async fn mark_episode_aired(
    pool: &PgPool,
    episode_id: Uuid,
    traceparent: Option<&str>,
) -> Result<MarkEpisodeAiredOutcome, Error> {
    let mut tx = pool.begin().await?;
    let now = read_transaction_time(&mut tx).await?;

    let notified_row = sqlx::query!(
        r#"
        UPDATE episode e
        SET aired_notified_at = $2
        FROM anime a
        WHERE e.id = $1
          AND e.deleted_at IS NULL
          AND a.id = e.anime_id
          AND a.deleted_at IS NULL
          AND e.aired_at IS NOT NULL
          AND e.aired_at <= $2
          AND e.aired_notified_at IS NULL
        RETURNING e.anime_id, a.title AS anime_title, e.episode_number, e.aired_at, e.version
        "#,
        episode_id,
        now,
    )
    .fetch_optional(&mut *tx)
    .await?;

    let Some(row) = notified_row else {
        let exists = sqlx::query_scalar!(
            r#"
            SELECT EXISTS(
                SELECT 1
                FROM episode e
                JOIN anime a ON a.id = e.anime_id
                WHERE e.id = $1
                  AND e.deleted_at IS NULL
                  AND a.deleted_at IS NULL
            )
            "#,
            episode_id
        )
        .fetch_one(&mut *tx)
        .await?
        .unwrap_or(false);
        if !exists {
            return Err(Error::not_found());
        }
        tx.commit().await?;
        return Ok(MarkEpisodeAiredOutcome {
            has_notified: false,
        });
    };

    let event_id = Uuid::now_v7();
    let notification = ProtoEpisodeAiredNotification {
        event_id: event_id.to_string(),
        saved_at: Some(to_proto_timestamp(now)),
        anime_id: row.anime_id.to_string(),
        anime_title: row.anime_title.unwrap_or_default(),
        episode_id: episode_id.to_string(),
        episode_number: row.episode_number,
        aired_at: row.aired_at.map(to_proto_timestamp),
    };
    let payload = notification.encode_to_vec();

    sqlx::query!(
        r#"
        INSERT INTO outbox (id, subject, entity_id, version, payload, created_at, traceparent)
        VALUES ($1, 'notification.episode.aired', $2, $3, $4, $5, $6)
        "#,
        event_id,
        episode_id,
        row.version,
        payload,
        now,
        traceparent,
    )
    .execute(&mut *tx)
    .await?;

    tx.commit().await?;
    Ok(MarkEpisodeAiredOutcome { has_notified: true })
}

/// Soft-deletes an episode, removes its page links and graph edges, and writes `episode.deleted`.
///
/// # Examples
///
/// ```no_run
/// # use sqlx::PgPool;
/// # use uuid::Uuid;
/// # use postgres_db::delete_episode;
/// # async fn example(pool: &PgPool, episode_id: Uuid) -> Result<(), postgres_db::Error> {
/// // The episode is at version 1; the delete raises it once more for `episode.deleted`.
/// let new_version = delete_episode(pool, episode_id, 1, None).await?;
/// assert_eq!(new_version, 2);
///
/// // A deleted episode cannot be deleted again.
/// let repeated_delete = delete_episode(pool, episode_id, new_version, None).await;
/// assert!(repeated_delete.is_err_and(|error| error.is_not_found()));
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns [`Error::is_not_found`] when `episode_id` does not exist or is
/// already deleted, [`Error::is_version_conflict`] when `expected_version`
/// does not match, or a database error when persistence fails.
pub async fn delete_episode(
    pool: &PgPool,
    episode_id: Uuid,
    expected_version: i64,
    traceparent: Option<&str>,
) -> Result<i64, Error> {
    let mut tx = pool.begin().await?;
    let existing = sqlx::query!(
        r#"
        SELECT anime_id, version
        FROM episode
        WHERE id = $1 AND deleted_at IS NULL
        FOR UPDATE
        "#,
        episode_id
    )
    .fetch_optional(&mut *tx)
    .await?
    .ok_or_else(Error::not_found)?;

    if existing.version != expected_version {
        return Err(Error::version_conflict(format!(
            "expected version {}, stored version {}",
            expected_version, existing.version
        )));
    }

    let next_version = existing.version + 1;
    let now = read_transaction_time(&mut tx).await?;

    sqlx::query!(
        r#"
        UPDATE episode
        SET deleted_at = $2,
            version = $3,
            updated_at = $2
        WHERE id = $1
        "#,
        episode_id,
        now,
        next_version,
    )
    .execute(&mut *tx)
    .await?;

    sqlx::query!(
        "DELETE FROM episode_source WHERE episode_id = $1",
        episode_id
    )
    .execute(&mut *tx)
    .await?;

    sqlx::query!(
        "DELETE FROM graph_edge WHERE from_id = $1 OR to_id = $1",
        episode_id
    )
    .execute(&mut *tx)
    .await?;

    insert_catalogue_outbox_event(
        &mut tx,
        "episode.deleted",
        episode_id,
        next_version,
        Some(existing.anime_id),
        now,
        traceparent,
    )
    .await?;

    tx.commit().await?;
    Ok(next_version)
}

/// Checks the episodes a work save lists before any write.
///
/// # Errors
///
/// Returns an invalid-argument error for a negative episode number, a page key that is
/// not an episode page, or a negative list entry duration.
pub(crate) fn validate_anime_episode_references(
    episodes: &[EpisodeReferenceInput],
) -> Result<(), Error> {
    for episode in episodes {
        if episode.episode_number < 0 {
            return Err(Error::invalid_argument("episode_number must be >= 0"));
        }
        for page_key in &episode.page_keys {
            if page_key.entity_type() != EntityType::Episode {
                return Err(Error::invalid_argument(
                    "episode page_keys must have EntityType::Episode",
                ));
            }
        }
        for entry in &episode.list_entries {
            if entry.duration_seconds.is_some_and(|seconds| seconds < 0) {
                return Err(Error::invalid_argument(
                    "episode_list_entry duration_seconds must be >= 0",
                ));
            }
        }
    }
    Ok(())
}

/// Creates the episodes a work newly lists and updates those it already has.
///
/// - A new episode gets its row, page links and list entries, with `merged_at` empty.
/// - An existing episode whose page links or list entries changed gets a new version and a
///   merge request.
/// - An existing episode whose list entries changed is also made due at once, since the sweep
///   skips episodes of finished works.
///
/// # Errors
///
/// Returns an invalid-argument error when an episode belongs to another work or is deleted,
/// or when a page already belongs to another episode, a unique violation when a new episode
/// takes a number an undeleted episode holds, or a database error when a query fails.
pub(crate) async fn sync_anime_episodes(
    tx: &mut Transaction<'_, Postgres>,
    anime_id: Uuid,
    episodes: &[EpisodeReferenceInput],
) -> Result<SyncAnimeEpisodesOutcome, Error> {
    let mut sorted_episodes: Vec<&EpisodeReferenceInput> = episodes.iter().collect();
    sorted_episodes.sort_by_key(|episode| episode.episode_id);

    let mut created_episode_ids = Vec::new();
    let mut updated_episode_ids = Vec::new();

    for episode in sorted_episodes {
        let existing = sqlx::query!(
            "SELECT anime_id, deleted_at FROM episode WHERE id = $1 FOR UPDATE",
            episode.episode_id
        )
        .fetch_optional(&mut **tx)
        .await?;

        match existing {
            None => {
                insert_new_episode_from_anime(tx, anime_id, episode).await?;
                created_episode_ids.push(episode.episode_id);
            }
            Some(row) => {
                if row.deleted_at.is_some() || row.anime_id != anime_id {
                    return Err(Error::invalid_argument(format!(
                        "episode_id {} cannot be updated by anime {}",
                        episode.episode_id, anime_id
                    )));
                }
                let pages_changed =
                    link_episode_pages(tx, episode.episode_id, &episode.page_keys).await?;
                let entries_changed =
                    replace_list_entries_if_changed(tx, episode.episode_id, &episode.list_entries)
                        .await?;
                if pages_changed || entries_changed {
                    sqlx::query!(
                        r#"
                        UPDATE episode
                        SET version = version + 1,
                            merge_requested_at = COALESCE(merge_requested_at, now()),
                            next_refresh_at = CASE WHEN $2 THEN NULL ELSE next_refresh_at END,
                            updated_at = now()
                        WHERE id = $1
                        "#,
                        episode.episode_id,
                        entries_changed,
                    )
                    .execute(&mut **tx)
                    .await?;
                    if entries_changed {
                        updated_episode_ids.push(episode.episode_id);
                    }
                }
            }
        }
    }

    Ok(SyncAnimeEpisodesOutcome {
        created_episode_ids,
        updated_episode_ids,
    })
}

/// Creates an episode listed by its work, with its page links and list entries.
///
/// # Errors
///
/// Returns a unique violation when the work already has an undeleted episode with the same
/// number, or a database error when a query fails.
async fn insert_new_episode_from_anime(
    tx: &mut Transaction<'_, Postgres>,
    anime_id: Uuid,
    episode: &EpisodeReferenceInput,
) -> Result<(), Error> {
    let stored_title = (!episode.title.trim().is_empty()).then(|| episode.title.clone());
    sqlx::query!(
        r#"
        INSERT INTO episode (id, anime_id, episode_number, title)
        VALUES ($1, $2, $3, $4)
        "#,
        episode.episode_id,
        anime_id,
        episode.episode_number,
        stored_title,
    )
    .execute(&mut **tx)
    .await?;

    link_episode_pages(tx, episode.episode_id, &episode.page_keys).await?;
    insert_episode_list_entries(tx, episode.episode_id, &episode.list_entries).await?;
    Ok(())
}

/// Links each page key to the episode, replacing an older page of the same provider.
///
/// The result tells whether any link was added.
///
/// # Errors
///
/// Returns an invalid-argument error when a page already belongs to another episode, or a
/// database error when a query fails.
async fn link_episode_pages(
    tx: &mut Transaction<'_, Postgres>,
    episode_id: Uuid,
    page_keys: &[PageKey],
) -> Result<bool, Error> {
    let mut sorted_keys: Vec<&PageKey> = page_keys.iter().collect();
    sorted_keys.sort_by(|left, right| {
        (left.provider().as_str(), left.address().as_str())
            .cmp(&(right.provider().as_str(), right.address().as_str()))
    });
    sorted_keys.dedup();

    let mut pages_changed = false;
    for page_key in sorted_keys {
        sqlx::query!(
            "SELECT pg_advisory_xact_lock(hashtextextended($1 || ':episode:' || $2, 0))",
            page_key.provider().as_str(),
            page_key.address().as_str(),
        )
        .execute(&mut **tx)
        .await?;

        let existing_owner = sqlx::query_scalar!(
            r#"
            SELECT episode_id
            FROM episode_source
            WHERE provider = $1 AND entity_type = 'episode' AND address = $2
            FOR UPDATE
            "#,
            page_key.provider().as_str(),
            page_key.address().as_str(),
        )
        .fetch_optional(&mut **tx)
        .await?;

        if let Some(owner_episode_id) = existing_owner {
            if owner_episode_id != episode_id {
                return Err(Error::invalid_argument(format!(
                    "episode page {}:episode:{} already belongs to episode {owner_episode_id}",
                    page_key.provider(),
                    page_key.address()
                )));
            }
            continue;
        }

        // Replace any previous address for this (episode_id, provider) so correcting
        // an episode's page preserves its UUID without violating UNIQUE(episode_id, provider).
        sqlx::query!(
            "DELETE FROM episode_source WHERE episode_id = $1 AND provider = $2",
            episode_id,
            page_key.provider().as_str(),
        )
        .execute(&mut **tx)
        .await?;

        sqlx::query!(
            r#"
            INSERT INTO provider_page (provider, entity_type, address)
            VALUES ($1, 'episode', $2)
            ON CONFLICT (provider, entity_type, address) DO NOTHING
            "#,
            page_key.provider().as_str(),
            page_key.address().as_str(),
        )
        .execute(&mut **tx)
        .await?;

        sqlx::query!(
            r#"
            INSERT INTO episode_source (episode_id, provider, entity_type, address)
            VALUES ($1, $2, 'episode', $3)
            "#,
            episode_id,
            page_key.provider().as_str(),
            page_key.address().as_str(),
        )
        .execute(&mut **tx)
        .await?;
        pages_changed = true;
    }
    Ok(pages_changed)
}

/// Replaces an episode's list entries when they differ from the stored ones.
///
/// The result tells whether the entries changed.
///
/// # Errors
///
/// Returns a database error when a query fails.
async fn replace_list_entries_if_changed(
    tx: &mut Transaction<'_, Postgres>,
    episode_id: Uuid,
    incoming_entries: &[EpisodeListEntryInput],
) -> Result<bool, Error> {
    let existing_entries = load_episode_list_entries(tx, episode_id).await?;
    let normalized_incoming = normalize_list_entries(incoming_entries);
    if existing_entries == normalized_incoming {
        return Ok(false);
    }

    sqlx::query!(
        "DELETE FROM episode_list_entry WHERE episode_id = $1",
        episode_id
    )
    .execute(&mut **tx)
    .await?;
    insert_episode_list_entries(tx, episode_id, &normalized_incoming).await?;
    Ok(true)
}

/// Keeps one list entry per provider, with times cut to PostgreSQL's microsecond precision.
///
/// Stored and incoming entries then compare equal when nothing changed.
fn normalize_list_entries(entries: &[EpisodeListEntryInput]) -> Vec<EpisodeListEntryInput> {
    let mut deduplicated_entries: BTreeMap<&'static str, EpisodeListEntryInput> = BTreeMap::new();
    for entry in entries {
        let mut normalized_entry = entry.clone();
        normalized_entry.aired_at = normalized_entry.aired_at.map(truncate_to_micros);
        deduplicated_entries.insert(entry.provider.as_str(), normalized_entry);
    }
    deduplicated_entries.into_values().collect()
}

/// Cuts a timestamp to the microsecond precision PostgreSQL stores.
fn truncate_to_micros(timestamp: DateTime<Utc>) -> DateTime<Utc> {
    let truncated_nanos = (timestamp.nanosecond() / 1_000) * 1_000;
    timestamp
        .with_nanosecond(truncated_nanos)
        .unwrap_or(timestamp)
}

/// Inserts an episode's list entries, one per provider.
///
/// # Errors
///
/// Returns a database error when a query fails.
async fn insert_episode_list_entries(
    tx: &mut Transaction<'_, Postgres>,
    episode_id: Uuid,
    entries: &[EpisodeListEntryInput],
) -> Result<(), Error> {
    let normalized = normalize_list_entries(entries);
    if normalized.is_empty() {
        return Ok(());
    }

    let providers: Vec<String> = normalized
        .iter()
        .map(|entry| entry.provider.as_str().to_owned())
        .collect();
    let titles: Vec<Option<String>> = normalized.iter().map(|entry| entry.title.clone()).collect();
    let aired_ats: Vec<Option<DateTime<Utc>>> =
        normalized.iter().map(|entry| entry.aired_at).collect();
    let durations: Vec<Option<i32>> = normalized
        .iter()
        .map(|entry| entry.duration_seconds)
        .collect();
    let fillers: Vec<Option<bool>> = normalized.iter().map(|entry| entry.is_filler).collect();
    let recaps: Vec<Option<bool>> = normalized.iter().map(|entry| entry.is_recap).collect();

    sqlx::query!(
        r#"
        INSERT INTO episode_list_entry (
            episode_id, provider, title, aired_at, duration_seconds, is_filler, is_recap
        )
        SELECT $1, provider, title, aired_at, duration_seconds, is_filler, is_recap
        FROM UNNEST(
            $2::text[],
            $3::text[],
            $4::timestamptz[],
            $5::int4[],
            $6::bool[],
            $7::bool[]
        ) AS entry(provider, title, aired_at, duration_seconds, is_filler, is_recap)
        "#,
        episode_id,
        &providers,
        &titles as &[Option<String>],
        &aired_ats as &[Option<DateTime<Utc>>],
        &durations as &[Option<i32>],
        &fillers as &[Option<bool>],
        &recaps as &[Option<bool>],
    )
    .execute(&mut **tx)
    .await?;
    Ok(())
}

/// Loads the undeleted episodes of `anime_id` with their page keys and list entries.
///
/// Episodes come in episode-number order, and each episode's page keys and list entries
/// in provider order, so two loads of the same rows give the same list.
///
/// # Errors
///
/// Returns a database error when a query fails, or an invalid-argument error when a
/// stored provider or address no longer passes validation.
pub(crate) async fn load_known_episodes(
    tx: &mut Transaction<'_, Postgres>,
    anime_id: Uuid,
) -> Result<Vec<EpisodeReferenceInput>, Error> {
    let episode_rows = sqlx::query!(
        r#"
        SELECT id, episode_number, title
        FROM episode
        WHERE anime_id = $1 AND deleted_at IS NULL
        ORDER BY episode_number, id
        "#,
        anime_id
    )
    .fetch_all(&mut **tx)
    .await?;
    let mut page_keys_by_episode = load_page_keys_by_episode(tx, anime_id).await?;
    let mut list_entries_by_episode = load_list_entries_by_episode(tx, anime_id).await?;

    Ok(episode_rows
        .into_iter()
        .map(|episode_row| EpisodeReferenceInput {
            episode_id: episode_row.id,
            episode_number: episode_row.episode_number,
            title: episode_row.title.unwrap_or_default(),
            page_keys: page_keys_by_episode
                .remove(&episode_row.id)
                .unwrap_or_default(),
            list_entries: list_entries_by_episode
                .remove(&episode_row.id)
                .unwrap_or_default(),
        })
        .collect())
}

/// Loads the page keys of every undeleted episode of `anime_id`, grouped by episode.
///
/// # Errors
///
/// Returns a database error when the query fails, or an invalid-argument error when a
/// stored provider or address no longer passes validation.
async fn load_page_keys_by_episode(
    tx: &mut Transaction<'_, Postgres>,
    anime_id: Uuid,
) -> Result<HashMap<Uuid, Vec<PageKey>>, Error> {
    let page_rows = sqlx::query!(
        r#"
        SELECT source.episode_id, source.provider, source.address
        FROM episode_source source
        JOIN episode e ON e.id = source.episode_id
        WHERE e.anime_id = $1 AND e.deleted_at IS NULL
        ORDER BY source.episode_id, source.provider, source.address
        "#,
        anime_id
    )
    .fetch_all(&mut **tx)
    .await?;

    let mut page_keys_by_episode: HashMap<Uuid, Vec<PageKey>> = HashMap::new();
    for page_row in page_rows {
        let page_key = PageKey::new(
            page_row.provider.parse()?,
            EntityType::Episode,
            page_row.address,
        )?;
        page_keys_by_episode
            .entry(page_row.episode_id)
            .or_default()
            .push(page_key);
    }
    Ok(page_keys_by_episode)
}

/// Loads the list entries of every undeleted episode of `anime_id`, grouped by episode.
///
/// # Errors
///
/// Returns a database error when the query fails, or an invalid-argument error when a
/// stored provider is not a known provider.
async fn load_list_entries_by_episode(
    tx: &mut Transaction<'_, Postgres>,
    anime_id: Uuid,
) -> Result<HashMap<Uuid, Vec<EpisodeListEntryInput>>, Error> {
    let entry_rows = sqlx::query!(
        r#"
        SELECT
            ele.episode_id,
            ele.provider,
            ele.title,
            ele.aired_at,
            ele.duration_seconds,
            ele.is_filler,
            ele.is_recap
        FROM episode_list_entry ele
        JOIN episode e ON e.id = ele.episode_id
        WHERE e.anime_id = $1 AND e.deleted_at IS NULL
        ORDER BY ele.episode_id, ele.provider
        "#,
        anime_id
    )
    .fetch_all(&mut **tx)
    .await?;

    let mut list_entries_by_episode: HashMap<Uuid, Vec<EpisodeListEntryInput>> = HashMap::new();
    for entry_row in entry_rows {
        list_entries_by_episode
            .entry(entry_row.episode_id)
            .or_default()
            .push(EpisodeListEntryInput {
                provider: entry_row.provider.parse()?,
                title: entry_row.title,
                aired_at: entry_row.aired_at,
                duration_seconds: entry_row.duration_seconds,
                is_filler: entry_row.is_filler,
                is_recap: entry_row.is_recap,
            });
    }
    Ok(list_entries_by_episode)
}

#[cfg(test)]
mod tests {
    use chrono::TimeDelta;
    use sqlx::PgPool;
    use uuid::Uuid;

    use super::*;
    use crate::provider_page::{FetchOutcome, PageObservation, record_page_observation};

    fn sample_save_episode_input(episode_id: Uuid, expected_version: i64) -> SaveEpisodeInput {
        SaveEpisodeInput {
            episode_id,
            expected_version,
            episode_number: 1,
            season_number: Some(1),
            title: "Asteroid Blues".to_owned(),
            title_japanese: Some("アステロイド・ブルース".to_owned()),
            title_romaji: Some("Asteroid Blues".to_owned()),
            synopsis: Some("Spike and Jet hunt Asimov Solensan.".to_owned()),
            description: Some("Extended episode notes.".to_owned()),
            aired_at: Some(Utc::now() - TimeDelta::days(30)),
            duration_seconds: Some(1440),
            is_filler: false,
            is_recap: false,
            score: Some(8.75),
            titles: vec![
                EpisodeTitleInput {
                    language: "ja".to_owned(),
                    title: "アステロイド・ブルース".to_owned(),
                },
                EpisodeTitleInput {
                    language: "en".to_owned(),
                    title: "Asteroid Blues".to_owned(),
                },
            ],
            images: vec![
                "https://cdn.example.com/ep1-a.jpg".to_owned(),
                "https://cdn.example.com/ep1-a.jpg".to_owned(),
                "https://cdn.example.com/ep1-b.jpg".to_owned(),
            ],
            streaming: vec![EpisodeStreamingInput {
                platform: "crunchyroll".to_owned(),
                source: "https://crunchyroll.com/watch/1".to_owned(),
            }],
            characters: vec![EpisodeCharacterInput {
                character_page_provider: "mal".to_owned(),
                character_page_address: "1".to_owned(),
                role: "MAIN".to_owned(),
            }],
            merged_record_hash: "hash-v1".to_owned(),
            hash_format: EPISODE_HASH_FORMAT,
            merge_version: 1,
            traceparent: Some("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01".to_owned()),
            page_sources: Vec::new(),
        }
    }

    async fn seed_anime_and_episode(
        pool: &PgPool,
        aired_at: Option<DateTime<Utc>>,
    ) -> (Uuid, Uuid) {
        let anime_id = Uuid::now_v7();
        let episode_id = Uuid::now_v7();

        sqlx::query!(
            r#"
            INSERT INTO anime (id, title, type)
            VALUES ($1, 'Cowboy Bebop', 'TV')
            "#,
            anime_id
        )
        .execute(pool)
        .await
        .expect("insert seed anime");

        sqlx::query!(
            r#"
            INSERT INTO episode (id, anime_id, episode_number, title, aired_at)
            VALUES ($1, $2, 1, 'Asteroid Blues', $3)
            "#,
            episode_id,
            anime_id,
            aired_at
        )
        .execute(pool)
        .await
        .expect("insert seed episode");

        (anime_id, episode_id)
    }

    #[test]
    fn validate_save_episode_input_rejects_invalid_fields() {
        let id = Uuid::now_v7();

        let mut neg_ep = sample_save_episode_input(id, 0);
        neg_ep.episode_number = -1;
        assert!(
            validate_save_episode_input(&neg_ep)
                .unwrap_err()
                .is_invalid_argument()
        );

        let mut zero_season = sample_save_episode_input(id, 0);
        zero_season.season_number = Some(0);
        assert!(
            validate_save_episode_input(&zero_season)
                .unwrap_err()
                .is_invalid_argument()
        );

        let mut neg_duration = sample_save_episode_input(id, 0);
        neg_duration.duration_seconds = Some(-5);
        assert!(
            validate_save_episode_input(&neg_duration)
                .unwrap_err()
                .is_invalid_argument()
        );

        let mut highest_stored_score = sample_save_episode_input(id, 0);
        highest_stored_score.score = Some(999.994);
        assert!(validate_save_episode_input(&highest_stored_score).is_ok());

        for bad_score in [-0.1, 999.995, 999.996, 1000.0, f64::NAN, f64::INFINITY] {
            let mut bad = sample_save_episode_input(id, 0);
            bad.score = Some(bad_score);
            assert!(
                validate_save_episode_input(&bad)
                    .unwrap_err()
                    .is_invalid_argument()
            );
        }

        let mut empty_hash = sample_save_episode_input(id, 0);
        empty_hash.merged_record_hash.clear();
        assert!(
            validate_save_episode_input(&empty_hash)
                .unwrap_err()
                .is_invalid_argument()
        );

        let mut neg_merge_ver = sample_save_episode_input(id, 0);
        neg_merge_ver.merge_version = -1;
        assert!(
            validate_save_episode_input(&neg_merge_ver)
                .unwrap_err()
                .is_invalid_argument()
        );

        let mut bad_title = sample_save_episode_input(id, 0);
        bad_title.titles.push(EpisodeTitleInput {
            language: String::new(),
            title: "Test".to_owned(),
        });
        assert!(
            validate_save_episode_input(&bad_title)
                .unwrap_err()
                .is_invalid_argument()
        );

        let mut bad_image = sample_save_episode_input(id, 0);
        bad_image.images.push("  ".to_owned());
        assert!(
            validate_save_episode_input(&bad_image)
                .unwrap_err()
                .is_invalid_argument()
        );

        let mut bad_stream = sample_save_episode_input(id, 0);
        bad_stream.streaming.push(EpisodeStreamingInput {
            platform: "netflix".to_owned(),
            source: String::new(),
        });
        assert!(
            validate_save_episode_input(&bad_stream)
                .unwrap_err()
                .is_invalid_argument()
        );

        let mut bad_char = sample_save_episode_input(id, 0);
        bad_char.characters.push(EpisodeCharacterInput {
            character_page_provider: "mal".to_owned(),
            character_page_address: String::new(),
            role: "MAIN".to_owned(),
        });
        assert!(
            validate_save_episode_input(&bad_char)
                .unwrap_err()
                .is_invalid_argument()
        );
    }

    #[test]
    fn validate_save_episode_input_conflicting_streaming_links_return_invalid_argument() {
        let id = Uuid::now_v7();
        let mut repeated_stream = sample_save_episode_input(id, 0);
        repeated_stream.streaming.push(EpisodeStreamingInput {
            platform: "crunchyroll".to_owned(),
            source: "https://crunchyroll.com/watch/1".to_owned(),
        });
        assert!(
            validate_save_episode_input(&repeated_stream).is_ok(),
            "the same platform and source twice is one link"
        );

        let mut two_sources_one_platform = sample_save_episode_input(id, 0);
        two_sources_one_platform
            .streaming
            .push(EpisodeStreamingInput {
                platform: "crunchyroll".to_owned(),
                source: "https://crunchyroll.com/watch/other".to_owned(),
            });
        assert!(
            validate_save_episode_input(&two_sources_one_platform)
                .unwrap_err()
                .is_invalid_argument()
        );

        let mut two_platforms_one_source = sample_save_episode_input(id, 0);
        two_platforms_one_source
            .streaming
            .push(EpisodeStreamingInput {
                platform: "funimation".to_owned(),
                source: "https://crunchyroll.com/watch/1".to_owned(),
            });
        assert!(
            validate_save_episode_input(&two_platforms_one_source)
                .unwrap_err()
                .is_invalid_argument()
        );
    }

    #[test]
    fn validate_anime_episode_references_enforces_invariants() {
        let valid_page = PageKey::new(Provider::Mal, EntityType::Episode, "21:1").unwrap();
        let wrong_kind_page = PageKey::new(Provider::Mal, EntityType::Anime, "1").unwrap();

        let neg_ep = EpisodeReferenceInput {
            episode_id: Uuid::now_v7(),
            episode_number: -1,
            title: "Ep -1".to_owned(),
            page_keys: vec![valid_page.clone()],
            list_entries: Vec::new(),
        };
        assert!(
            validate_anime_episode_references(&[neg_ep])
                .unwrap_err()
                .is_invalid_argument()
        );

        let wrong_kind = EpisodeReferenceInput {
            episode_id: Uuid::now_v7(),
            episode_number: 0,
            title: "Special".to_owned(),
            page_keys: vec![wrong_kind_page],
            list_entries: Vec::new(),
        };
        assert!(
            validate_anime_episode_references(&[wrong_kind])
                .unwrap_err()
                .is_invalid_argument()
        );

        let neg_entry_duration = EpisodeReferenceInput {
            episode_id: Uuid::now_v7(),
            episode_number: 1,
            title: "Ep 1".to_owned(),
            page_keys: vec![valid_page],
            list_entries: vec![EpisodeListEntryInput {
                provider: Provider::Mal,
                title: None,
                aired_at: None,
                duration_seconds: Some(-10),
                is_filler: None,
                is_recap: None,
            }],
        };
        assert!(
            validate_anime_episode_references(&[neg_entry_duration])
                .unwrap_err()
                .is_invalid_argument()
        );
    }

    #[sqlx::test]
    async fn load_episode_returns_not_found_for_missing_or_deleted(pool: PgPool) {
        let missing_id = Uuid::now_v7();
        assert!(
            load_episode(&pool, missing_id)
                .await
                .unwrap_err()
                .is_not_found()
        );

        let (_, episode_id) = seed_anime_and_episode(&pool, None).await;
        delete_episode(&pool, episode_id, 1, None).await.unwrap();
        assert!(
            load_episode(&pool, episode_id)
                .await
                .unwrap_err()
                .is_not_found()
        );
    }

    #[sqlx::test]
    async fn load_episode_hydrates_pages_list_entries_and_corrections(pool: PgPool) {
        let (anime_id, episode_id) = seed_anime_and_episode(&pool, None).await;
        let episode_page_key = PageKey::new(Provider::Mal, EntityType::Episode, "21:1").unwrap();
        let fetched_at = Utc::now();

        record_page_observation(
            &pool,
            PageObservation {
                page_key: episode_page_key.clone(),
                observed_at: fetched_at,
                outcome: FetchOutcome::Success {
                    responses: vec![PageResponse {
                        request_name: "episode_detail".to_owned(),
                        content_type: "application/json".to_owned(),
                        body: br#"{"mal_id":1}"#.to_vec(),
                    }],
                },
            },
        )
        .await
        .unwrap();

        sqlx::query!(
            "INSERT INTO episode_source (episode_id, provider, entity_type, address)
             VALUES ($1, 'mal', 'episode', '21:1')",
            episode_id
        )
        .execute(&pool)
        .await
        .unwrap();

        sqlx::query!(
            "INSERT INTO episode_list_entry (episode_id, provider, title, duration_seconds, is_filler, is_recap)
             VALUES ($1, 'mal', 'Asteroid Blues', 1440, false, false)",
            episode_id
        )
        .execute(&pool)
        .await
        .unwrap();

        sqlx::query!(
            "INSERT INTO episode_field_correction (episode_id, field_path, value_message, value, approved_by)
             VALUES ($1, 'title', 'google.protobuf.StringValue', $2, 'mod_1')",
            episode_id,
            b"\x0a\x0eAsteroid Blues" as &[u8],
        )
        .execute(&pool)
        .await
        .unwrap();

        let loaded = load_episode(&pool, episode_id).await.unwrap();
        assert_eq!(loaded.anime_id, anime_id);
        assert_eq!(loaded.version, 1);
        assert_eq!(loaded.pages.len(), 1);
        assert_eq!(loaded.pages[0].page_key, episode_page_key);
        assert_eq!(loaded.pages[0].responses[0].body, br#"{"mal_id":1}"#);
        assert_eq!(loaded.list_entries.len(), 1);
        assert_eq!(loaded.list_entries[0].provider, Provider::Mal);
        assert_eq!(loaded.corrections.len(), 1);
        assert_eq!(loaded.corrections[0].field_path, "title");
    }

    #[sqlx::test]
    async fn save_episode_handles_create_noop_update_and_version_conflict(pool: PgPool) {
        let (_, episode_id) = seed_anime_and_episode(&pool, None).await;
        let first_merge_input = sample_save_episode_input(episode_id, 1);

        let first_merge_outcome = save_episode(&pool, &first_merge_input).await.unwrap();
        assert!(first_merge_outcome.has_changed);
        assert_eq!(first_merge_outcome.version, 2);

        let subjects: Vec<String> = sqlx::query_scalar!(
            "SELECT subject FROM outbox WHERE entity_id = $1 ORDER BY created_at",
            episode_id
        )
        .fetch_all(&pool)
        .await
        .unwrap();
        assert_eq!(subjects, vec!["episode.created".to_owned()]);

        let image_count: i64 = sqlx::query_scalar!(
            "SELECT COUNT(*) FROM episode_image WHERE episode_id = $1",
            episode_id
        )
        .fetch_one(&pool)
        .await
        .unwrap()
        .unwrap_or(0);
        assert_eq!(image_count, 2);

        sqlx::query!(
            "UPDATE episode SET merge_requested_at = now() WHERE id = $1",
            episode_id
        )
        .execute(&pool)
        .await
        .unwrap();

        let mut unchanged_merge_input = first_merge_input.clone();
        unchanged_merge_input.expected_version = 2;
        let unchanged_outcome = save_episode(&pool, &unchanged_merge_input).await.unwrap();
        assert!(!unchanged_outcome.has_changed);
        assert_eq!(unchanged_outcome.version, 2);

        let merge_requested_at = sqlx::query_scalar!(
            "SELECT merge_requested_at FROM episode WHERE id = $1",
            episode_id
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(merge_requested_at.is_none());

        assert!(
            save_episode(&pool, &first_merge_input)
                .await
                .unwrap_err()
                .is_version_conflict()
        );

        let mut changed_merge_input = unchanged_merge_input;
        changed_merge_input.merged_record_hash = "hash-v2".to_owned();
        changed_merge_input.title = "Asteroid Blues (Remastered)".to_owned();
        let changed_merge_outcome = save_episode(&pool, &changed_merge_input).await.unwrap();
        assert!(changed_merge_outcome.has_changed);
        assert_eq!(changed_merge_outcome.version, 3);

        let subjects_after: Vec<String> = sqlx::query_scalar!(
            "SELECT subject FROM outbox WHERE entity_id = $1 ORDER BY created_at",
            episode_id
        )
        .fetch_all(&pool)
        .await
        .unwrap();
        assert_eq!(
            subjects_after,
            vec!["episode.created".to_owned(), "episode.updated".to_owned()]
        );
    }

    #[sqlx::test]
    async fn mark_episode_aired_notifies_once_and_ignores_future_or_unset(pool: PgPool) {
        let (_, future_episode_id) =
            seed_anime_and_episode(&pool, Some(Utc::now() + TimeDelta::hours(2))).await;
        let future_episode_outcome = mark_episode_aired(&pool, future_episode_id, None)
            .await
            .unwrap();
        assert!(!future_episode_outcome.has_notified);

        let (_, unaired_episode_id) = seed_anime_and_episode(&pool, None).await;
        let unaired_episode_outcome = mark_episode_aired(&pool, unaired_episode_id, None)
            .await
            .unwrap();
        assert!(!unaired_episode_outcome.has_notified);

        let (anime_id, aired_episode_id) =
            seed_anime_and_episode(&pool, Some(Utc::now() - TimeDelta::minutes(5))).await;
        let first = mark_episode_aired(&pool, aired_episode_id, Some("00-trace-01"))
            .await
            .unwrap();
        assert!(first.has_notified);

        let second = mark_episode_aired(&pool, aired_episode_id, None)
            .await
            .unwrap();
        assert!(!second.has_notified);

        let outbox_rows = sqlx::query!(
            "SELECT subject, payload, traceparent FROM outbox WHERE entity_id = $1",
            aired_episode_id
        )
        .fetch_all(&pool)
        .await
        .unwrap();
        assert_eq!(outbox_rows.len(), 1);
        assert_eq!(outbox_rows[0].subject, "notification.episode.aired");
        assert_eq!(outbox_rows[0].traceparent.as_deref(), Some("00-trace-01"));

        let decoded =
            ProtoEpisodeAiredNotification::decode(outbox_rows[0].payload.as_slice()).unwrap();
        assert_eq!(decoded.anime_id, anime_id.to_string());
        assert_eq!(decoded.anime_title, "Cowboy Bebop");
        assert_eq!(decoded.episode_id, aired_episode_id.to_string());
        assert_eq!(decoded.episode_number, 1);
    }

    #[sqlx::test]
    async fn delete_episode_soft_deletes_cleans_links_and_writes_outbox(pool: PgPool) {
        let (_, episode_id) = seed_anime_and_episode(&pool, None).await;

        sqlx::query!(
            "INSERT INTO provider_page (provider, entity_type, address) VALUES ('mal', 'episode', '21:10')"
        )
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query!(
            "INSERT INTO episode_source (episode_id, provider, entity_type, address)
             VALUES ($1, 'mal', 'episode', '21:10')",
            episode_id
        )
        .execute(&pool)
        .await
        .unwrap();

        assert!(
            delete_episode(&pool, episode_id, 99, None)
                .await
                .unwrap_err()
                .is_version_conflict()
        );

        let deleted_version = delete_episode(&pool, episode_id, 1, None).await.unwrap();
        assert_eq!(deleted_version, 2);

        assert!(
            delete_episode(&pool, episode_id, 2, None)
                .await
                .unwrap_err()
                .is_not_found()
        );

        let link_count: i64 = sqlx::query_scalar!(
            "SELECT COUNT(*) FROM episode_source WHERE episode_id = $1",
            episode_id
        )
        .fetch_one(&pool)
        .await
        .unwrap()
        .unwrap_or(0);
        assert_eq!(link_count, 0);

        let subject = sqlx::query_scalar!(
            "SELECT subject FROM outbox WHERE entity_id = $1",
            episode_id
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(subject, "episode.deleted");
    }

    #[sqlx::test]
    async fn sync_anime_episodes_handles_create_idempotent_noop_page_swap_and_conflicts(
        pool: PgPool,
    ) {
        let (anime_id, _) = seed_anime_and_episode(&pool, None).await;
        let listed_episode_id = Uuid::now_v7();
        let aired_with_nanos = Utc::now();

        let listed_episode_reference = EpisodeReferenceInput {
            episode_id: listed_episode_id,
            episode_number: 2,
            title: "Stray Dog Strut".to_owned(),
            page_keys: vec![PageKey::new(Provider::Mal, EntityType::Episode, "21:2").unwrap()],
            list_entries: vec![EpisodeListEntryInput {
                provider: Provider::Mal,
                title: Some("Stray Dog Strut".to_owned()),
                aired_at: Some(aired_with_nanos),
                duration_seconds: Some(1440),
                is_filler: Some(false),
                is_recap: Some(false),
            }],
        };

        let mut tx = pool.begin().await.unwrap();
        let created_outcome = sync_anime_episodes(
            &mut tx,
            anime_id,
            std::slice::from_ref(&listed_episode_reference),
        )
        .await
        .unwrap();
        tx.commit().await.unwrap();
        assert_eq!(created_outcome.created_episode_ids, vec![listed_episode_id]);
        assert_eq!(created_outcome.updated_episode_ids, Vec::<Uuid>::new());

        let mut tx = pool.begin().await.unwrap();
        let unchanged_outcome = sync_anime_episodes(
            &mut tx,
            anime_id,
            std::slice::from_ref(&listed_episode_reference),
        )
        .await
        .unwrap();
        tx.commit().await.unwrap();
        assert_eq!(unchanged_outcome.created_episode_ids, Vec::<Uuid>::new());
        assert_eq!(unchanged_outcome.updated_episode_ids, Vec::<Uuid>::new());

        let mut moved_page_reference = listed_episode_reference.clone();
        moved_page_reference.page_keys =
            vec![PageKey::new(Provider::Mal, EntityType::Episode, "21:200").unwrap()];
        let mut tx = pool.begin().await.unwrap();
        let moved_page_outcome = sync_anime_episodes(
            &mut tx,
            anime_id,
            std::slice::from_ref(&moved_page_reference),
        )
        .await
        .unwrap();
        tx.commit().await.unwrap();
        assert_eq!(moved_page_outcome.updated_episode_ids, Vec::<Uuid>::new());
        let swapped_episode = sqlx::query!(
            "SELECT version, merge_requested_at FROM episode WHERE id = $1",
            listed_episode_id
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(swapped_episode.version, 2);
        assert!(swapped_episode.merge_requested_at.is_some());

        let linked_addresses: Vec<String> = sqlx::query_scalar!(
            "SELECT address FROM episode_source WHERE episode_id = $1",
            listed_episode_id
        )
        .fetch_all(&pool)
        .await
        .unwrap();
        assert_eq!(linked_addresses, vec!["21:200".to_owned()]);

        let other_episode_id = Uuid::now_v7();
        let conflicting_page_ref = EpisodeReferenceInput {
            episode_id: other_episode_id,
            episode_number: 3,
            title: "Honky Tonk Women".to_owned(),
            page_keys: vec![PageKey::new(Provider::Mal, EntityType::Episode, "21:200").unwrap()],
            list_entries: Vec::new(),
        };
        let mut tx = pool.begin().await.unwrap();
        let page_owner_error = sync_anime_episodes(&mut tx, anime_id, &[conflicting_page_ref])
            .await
            .unwrap_err();
        assert!(page_owner_error.is_invalid_argument());
        tx.rollback().await.unwrap();

        let other_anime_id = Uuid::now_v7();
        sqlx::query!(
            "INSERT INTO anime (id, title, type) VALUES ($1, 'Other Anime', 'TV')",
            other_anime_id
        )
        .execute(&pool)
        .await
        .unwrap();

        let mut tx = pool.begin().await.unwrap();
        let other_anime_error =
            sync_anime_episodes(&mut tx, other_anime_id, &[moved_page_reference])
                .await
                .unwrap_err();
        assert!(other_anime_error.is_invalid_argument());
        tx.rollback().await.unwrap();
    }
}
