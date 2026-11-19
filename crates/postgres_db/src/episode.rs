//! Episode loading, version-checked saving, aired notifications, and work-list sync.

// Rust guideline compliant 2026-10-09

use std::collections::{BTreeMap, BTreeSet, HashMap};

use chrono::{DateTime, Utc};
use prost::Message as _;
use sqlx::{PgPool, Postgres, Transaction};
use tracing::Level;
use uuid::Uuid;

use crate::anime::{
    FieldCorrectionRecord, LoadedSourcePage, ProtoTimestamp, decompress_body,
    insert_catalogue_outbox_event, parse_page_kind, to_proto_timestamp,
};
use crate::provider_page::validate_page_address;
use crate::{Error, PageKeyInput, PageKind, PageResponseInput};

/// Current canonical hash format version stored in `episode.hash_format`.
pub const EPISODE_HASH_FORMAT: i16 = 1;

/// One provider's facts about an episode, sliced out of a work's page.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpisodeListEntryInput {
    /// Provider name (for example `"mal"` or `"anilist"`).
    pub provider: String,
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
    pub page_keys: Vec<PageKeyInput>,
    /// Per-provider episode list entries from the work's pages.
    pub list_entries: Vec<EpisodeListEntryInput>,
}

/// Localized episode title stored in `episode_title`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpisodeTitleInput {
    /// Language code or name (primary key component).
    pub language: String,
    /// Localized title text.
    pub title: String,
}

/// Episode streaming link stored in `episode_streaming`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpisodeStreamingInput {
    /// Streaming platform identifier (primary key component).
    pub platform: String,
    /// Direct episode URL on `platform`.
    pub url: String,
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
    /// Primary display title.
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

/// Loads an episode's version, linked pages, list entries, and approved corrections.
///
/// # Errors
///
/// Returns [`Error::is_not_found`] when `episode_id` does not exist or is
/// deleted, or a database error when any query or decompression fails.
pub async fn load_episode(pool: &PgPool, episode_id: Uuid) -> Result<LoadEpisodeOutcome, Error> {
    let mut tx = pool.begin().await?;
    sqlx::query!("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        .execute(&mut *tx)
        .await?;

    let episode_row = sqlx::query!(
        r#"
        SELECT version, anime_id
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
        pages,
        corrections,
        list_entries,
    })
}

async fn load_episode_pages(
    tx: &mut Transaction<'_, Postgres>,
    episode_id: Uuid,
) -> Result<Vec<LoadedSourcePage>, Error> {
    let linked_rows = sqlx::query!(
        r#"
        SELECT
            epp.provider,
            epp.page_kind,
            epp.address,
            pp.fetched_at,
            pp.is_dead,
            pp.missing_since
        FROM episode_provider_page epp
        JOIN provider_page pp
          ON pp.provider = epp.provider
         AND pp.page_kind = epp.page_kind
         AND pp.address = epp.address
        WHERE epp.episode_id = $1
        ORDER BY epp.provider, epp.address
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
        let page_kind = parse_page_kind(&row.page_kind)?;
        let page_key = PageKeyInput {
            provider: row.provider.clone(),
            page_kind,
            address: row.address.clone(),
        };
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

async fn load_episode_responses_by_page(
    tx: &mut Transaction<'_, Postgres>,
    episode_id: Uuid,
) -> Result<HashMap<(String, String), Vec<PageResponseInput>>, Error> {
    let response_rows = sqlx::query!(
        r#"
        SELECT
            ppr.provider,
            ppr.address,
            ppr.request_name,
            ppr.content_type,
            ppr.body
        FROM episode_provider_page epp
        JOIN provider_page_response ppr
          ON ppr.provider = epp.provider
         AND ppr.page_kind = epp.page_kind
         AND ppr.address = epp.address
        WHERE epp.episode_id = $1
        ORDER BY ppr.provider, ppr.address, ppr.request_name
        "#,
        episode_id
    )
    .fetch_all(&mut **tx)
    .await?;

    let mut responses_by_page: HashMap<(String, String), Vec<PageResponseInput>> = HashMap::new();
    for row in response_rows {
        let body = decompress_body(&row.body)?;
        responses_by_page
            .entry((row.provider, row.address))
            .or_default()
            .push(PageResponseInput {
                request_name: row.request_name,
                content_type: row.content_type,
                body,
            });
    }
    Ok(responses_by_page)
}

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

    Ok(rows
        .into_iter()
        .map(|row| EpisodeListEntryInput {
            provider: row.provider,
            title: row.title,
            aired_at: row.aired_at,
            duration_seconds: row.duration_seconds,
            is_filler: row.is_filler,
            is_recap: row.is_recap,
        })
        .collect())
}

/// Validates and persists a merged episode when its canonical hash changed.
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
            SET merge_requested_at = NULL
            WHERE id = $1 AND merge_requested_at IS NOT NULL
            "#,
            input.episode_id
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
    let now = Utc::now();
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
        Level::INFO,
        episode_id = %input.episode_id,
        version = next_version,
        subject,
        "episode.saved"
    );

    Ok(SaveEpisodeOutcome {
        has_changed: true,
        version: next_version,
    })
}

fn validate_save_episode_input(input: &SaveEpisodeInput) -> Result<(), Error> {
    if input.title.trim().is_empty() {
        return Err(Error::invalid_argument("episode title must not be empty"));
    }
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
        && (!score.is_finite() || score < 0.0)
    {
        return Err(Error::invalid_argument(
            "episode score must be finite and >= 0",
        ));
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
    for streaming_entry in &input.streaming {
        if streaming_entry.platform.trim().is_empty() || streaming_entry.url.trim().is_empty() {
            return Err(Error::invalid_argument(
                "episode_streaming platform and url must not be empty",
            ));
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
        input.title,
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
        "INSERT INTO episode_image (episode_id, url)
         SELECT $1, url
         FROM UNNEST($2::text[]) AS entry(url)",
        episode_id,
        &urls
    )
    .execute(&mut **tx)
    .await?;
    Ok(())
}

async fn replace_episode_streaming(
    tx: &mut Transaction<'_, Postgres>,
    episode_id: Uuid,
    streaming: &[EpisodeStreamingInput],
) -> Result<(), Error> {
    sqlx::query!(
        "DELETE FROM episode_streaming WHERE episode_id = $1",
        episode_id
    )
    .execute(&mut **tx)
    .await?;

    let deduplicated_streaming: BTreeMap<&str, &str> = streaming
        .iter()
        .map(|entry| (entry.platform.as_str(), entry.url.as_str()))
        .collect();
    if deduplicated_streaming.is_empty() {
        return Ok(());
    }

    let platforms: Vec<String> = deduplicated_streaming
        .keys()
        .copied()
        .map(str::to_owned)
        .collect();
    let urls: Vec<String> = deduplicated_streaming
        .values()
        .copied()
        .map(str::to_owned)
        .collect();
    sqlx::query!(
        "INSERT INTO episode_streaming (episode_id, platform, url)
         SELECT $1, platform, url
         FROM UNNEST($2::text[], $3::text[]) AS entry(platform, url)",
        episode_id,
        &platforms,
        &urls
    )
    .execute(&mut **tx)
    .await?;
    Ok(())
}

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
/// # Errors
///
/// Returns [`Error::is_not_found`] when `episode_id` does not exist or is
/// deleted, or a database error when the transaction fails.
pub async fn mark_episode_aired(
    pool: &PgPool,
    episode_id: Uuid,
    traceparent: Option<&str>,
) -> Result<MarkEpisodeAiredOutcome, Error> {
    let mut tx = pool.begin().await?;
    let now = Utc::now();

    let notified_row = sqlx::query!(
        r#"
        UPDATE episode
        SET aired_notified_at = $2
        WHERE id = $1
          AND deleted_at IS NULL
          AND aired_at IS NOT NULL
          AND aired_at <= $2
          AND aired_notified_at IS NULL
        RETURNING anime_id, episode_number, aired_at, version
        "#,
        episode_id,
        now,
    )
    .fetch_optional(&mut *tx)
    .await?;

    let Some(row) = notified_row else {
        let exists = sqlx::query_scalar!(
            "SELECT EXISTS(SELECT 1 FROM episode WHERE id = $1 AND deleted_at IS NULL)",
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

    let anime_title = sqlx::query_scalar!(
        "SELECT COALESCE(title, '') FROM anime WHERE id = $1",
        row.anime_id
    )
    .fetch_one(&mut *tx)
    .await?
    .unwrap_or_default();

    let event_id = Uuid::now_v7();
    let notification = ProtoEpisodeAiredNotification {
        event_id: event_id.to_string(),
        saved_at: Some(to_proto_timestamp(now)),
        anime_id: row.anime_id.to_string(),
        anime_title,
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
    let now = Utc::now();

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
        "DELETE FROM episode_provider_page WHERE episode_id = $1",
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

pub(crate) fn validate_work_episode_references(
    episodes: &[EpisodeReferenceInput],
) -> Result<(), Error> {
    for episode in episodes {
        if episode.episode_number < 0 {
            return Err(Error::invalid_argument("episode_number must be >= 0"));
        }
        for page_key in &episode.page_keys {
            if page_key.page_kind != PageKind::Episode {
                return Err(Error::invalid_argument(
                    "episode page_keys must have PageKind::Episode",
                ));
            }
            validate_page_address(&page_key.provider, &page_key.address)?;
        }
        for entry in &episode.list_entries {
            if entry.provider.trim().is_empty() {
                return Err(Error::invalid_argument(
                    "episode_list_entry provider must not be empty",
                ));
            }
            if entry.duration_seconds.is_some_and(|seconds| seconds < 0) {
                return Err(Error::invalid_argument(
                    "episode_list_entry duration_seconds must be >= 0",
                ));
            }
        }
    }
    Ok(())
}

pub(crate) async fn sync_work_episodes(
    tx: &mut Transaction<'_, Postgres>,
    anime_id: Uuid,
    episodes: &[EpisodeReferenceInput],
) -> Result<(Vec<Uuid>, Vec<Uuid>), Error> {
    let mut created_episode_ids = Vec::new();
    let mut changed_episode_ids = Vec::new();

    for episode in episodes {
        let existing = sqlx::query!(
            "SELECT anime_id, deleted_at FROM episode WHERE id = $1 FOR UPDATE",
            episode.episode_id
        )
        .fetch_optional(&mut **tx)
        .await?;

        match existing {
            None => {
                insert_new_episode_from_work(tx, anime_id, episode).await?;
                created_episode_ids.push(episode.episode_id);
            }
            Some(row) => {
                if row.deleted_at.is_some() || row.anime_id != anime_id {
                    return Err(Error::invalid_argument(format!(
                        "episode_id {} cannot be updated by anime {}",
                        episode.episode_id, anime_id
                    )));
                }
                link_episode_pages(tx, episode.episode_id, &episode.page_keys).await?;
                if replace_list_entries_if_changed(tx, episode.episode_id, &episode.list_entries)
                    .await?
                {
                    sqlx::query!(
                        r#"
                        UPDATE episode
                        SET next_refresh_at = NULL,
                            updated_at = now()
                        WHERE id = $1
                        "#,
                        episode.episode_id
                    )
                    .execute(&mut **tx)
                    .await?;
                    changed_episode_ids.push(episode.episode_id);
                }
            }
        }
    }

    Ok((created_episode_ids, changed_episode_ids))
}

async fn insert_new_episode_from_work(
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

async fn link_episode_pages(
    tx: &mut Transaction<'_, Postgres>,
    episode_id: Uuid,
    page_keys: &[PageKeyInput],
) -> Result<(), Error> {
    for page_key in page_keys {
        sqlx::query!(
            r#"
            INSERT INTO provider_page (provider, page_kind, address)
            VALUES ($1, 'episode', $2)
            ON CONFLICT (provider, page_kind, address) DO NOTHING
            "#,
            page_key.provider,
            page_key.address,
        )
        .execute(&mut **tx)
        .await?;

        sqlx::query!(
            r#"
            INSERT INTO episode_provider_page (episode_id, provider, page_kind, address)
            VALUES ($1, $2, 'episode', $3)
            ON CONFLICT (provider, page_kind, address) DO NOTHING
            "#,
            episode_id,
            page_key.provider,
            page_key.address,
        )
        .execute(&mut **tx)
        .await?;
    }
    Ok(())
}

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

fn normalize_list_entries(entries: &[EpisodeListEntryInput]) -> Vec<EpisodeListEntryInput> {
    let mut deduplicated_entries: BTreeMap<&str, &EpisodeListEntryInput> = BTreeMap::new();
    for entry in entries {
        deduplicated_entries.insert(entry.provider.as_str(), entry);
    }
    deduplicated_entries.into_values().cloned().collect()
}

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
        .map(|entry| entry.provider.clone())
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
