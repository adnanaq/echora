//! Provider page storage, response compression, and dead-page tracking.

// Rust guideline compliant 2026-10-09

use std::io::Write as _;

use chrono::{DateTime, Duration, Utc};
use flate2::Compression;
use flate2::write::GzEncoder;
use prost::Message as _;
use sha2::{Digest as _, Sha256};
use sqlx::{PgPool, Postgres, Transaction};
use tracing::Level;

use crate::Error;

/// Kind of provider page stored in PostgreSQL.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageKind {
    /// Work/anime page.
    Anime,
    /// Episode page.
    Episode,
    /// Character page.
    Character,
}

impl PageKind {
    /// Returns the lookup table key for this page kind.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Anime => "anime",
            Self::Episode => "episode",
            Self::Character => "character",
        }
    }

    const fn as_proto_i32(self) -> i32 {
        match self {
            Self::Anime => 1,
            Self::Episode => 2,
            Self::Character => 3,
        }
    }
}

#[derive(PartialEq, prost::Message)]
struct ProtoPageKey {
    #[prost(string, tag = "1")]
    provider: String,
    #[prost(int32, tag = "2")]
    page_kind: i32,
    #[prost(string, tag = "3")]
    address: String,
}

/// Encodes `page_key` as a `shared_proto.v1.PageKey` protobuf payload.
#[must_use]
pub fn encode_page_key_proposal(page_key: &PageKeyInput) -> Vec<u8> {
    ProtoPageKey {
        provider: page_key.provider.clone(),
        page_kind: page_key.page_kind.as_proto_i32(),
        address: page_key.address.clone(),
    }
    .encode_to_vec()
}

/// Identifies a provider page by provider name, page kind, and address.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageKeyInput {
    /// Provider key (for example `"mal"`, `"anilist"`, or `"kitsu"`).
    pub provider: String,
    /// Kind of entity described by the page.
    pub page_kind: PageKind,
    /// Provider-specific page identifier or slug.
    pub address: String,
}

/// One raw provider response belonging to a fetched page.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageResponseInput {
    /// Logical request name (for example `"main"` or `"characters"`).
    pub request_name: String,
    /// MIME content type returned by the provider.
    pub content_type: String,
    /// Uncompressed response body bytes as received from the provider.
    pub body: Vec<u8>,
}

/// Input payload for storing a fetched provider page or fetch outcome.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StorePageInput {
    /// Provider page key.
    pub page_key: PageKeyInput,
    /// Timestamp when the fetch occurred.
    pub fetched_at: DateTime<Utc>,
    /// Whether the page is confirmed dead.
    pub is_dead: bool,
    /// Uncompressed responses received for the page; empty on failure or not-found.
    pub responses: Vec<PageResponseInput>,
    /// Whether the provider returned its own not-found response.
    pub is_not_found: bool,
    /// Whether repeated not-found responses require review before setting `is_dead`.
    pub dead_page_review: bool,
}

/// Outcome returned after storing a provider page.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StorePageOutcome {
    /// `true` when stored responses or dead status changed.
    pub has_changed: bool,
}

#[derive(Debug)]
struct ExistingPageRow {
    fetched_at: Option<DateTime<Utc>>,
    is_dead: bool,
    missing_since: Option<DateTime<Utc>>,
    missing_last_at: Option<DateTime<Utc>>,
    missing_answers: i32,
    responses_hash: Option<String>,
    fetch_failed_at: Option<DateTime<Utc>>,
}

fn is_stale_observation(
    existing_page: Option<&ExistingPageRow>,
    fetched_at: DateTime<Utc>,
) -> bool {
    existing_page
        .and_then(|row| {
            [
                row.fetched_at,
                row.missing_since,
                row.missing_last_at,
                row.fetch_failed_at,
            ]
            .into_iter()
            .flatten()
            .max()
        })
        .is_some_and(|latest| fetched_at.timestamp_micros() <= latest.timestamp_micros())
}

/// Validates that `address` matches `provider`'s expected identifier format.
///
/// # Errors
///
/// Returns an invalid-argument error when `provider` is unknown or `address`
/// fails that provider's address pattern.
pub fn validate_page_address(provider: &str, address: &str) -> Result<(), Error> {
    if address.is_empty() {
        return Err(Error::invalid_argument("page address must not be empty"));
    }

    match provider {
        "mal" | "anilist" | "anidb" | "anisearch" | "ann" | "livechart" | "simkl"
        | "animecountdown" | "syoboi" => {
            if address.bytes().all(|byte| byte.is_ascii_digit()) {
                Ok(())
            } else {
                Err(Error::invalid_argument(format!(
                    "provider {provider} requires a numeric address, got {address:?}"
                )))
            }
        }
        "kitsu" | "anime_planet" | "animeschedule" => {
            if address
                .bytes()
                .any(|byte| matches!(byte, b'/' | b'?' | b'#' | b' ' | b'\t' | b'\n' | b'\r'))
            {
                Err(Error::invalid_argument(format!(
                    "provider {provider} address contains invalid path or whitespace characters: {address:?}"
                )))
            } else {
                Ok(())
            }
        }
        _ => Err(Error::invalid_argument(format!(
            "unknown provider {provider:?}"
        ))),
    }
}

/// Computes a deterministic SHA-256 hex digest over uncompressed `responses`.
#[must_use]
pub fn compute_responses_hash(responses: &[PageResponseInput]) -> String {
    let mut ordered: Vec<&PageResponseInput> = responses.iter().collect();
    ordered.sort_by(|left, right| left.request_name.cmp(&right.request_name));

    let mut hasher = Sha256::new();
    for response in ordered {
        hasher.update((response.request_name.len() as u64).to_be_bytes());
        hasher.update(response.request_name.as_bytes());
        hasher.update((response.content_type.len() as u64).to_be_bytes());
        hasher.update(response.content_type.as_bytes());
        hasher.update((response.body.len() as u64).to_be_bytes());
        hasher.update(&response.body);
    }

    let digest = hasher.finalize();
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut hex, "{byte:02x}");
    }
    hex
}

fn compress_body(body: &[u8]) -> Result<Vec<u8>, Error> {
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder
        .write_all(body)
        .map_err(|err| Error::invalid_argument(format!("gzip compression failed: {err}")))?;
    encoder
        .finish()
        .map_err(|err| Error::invalid_argument(format!("gzip finalization failed: {err}")))
}

/// Stores a fetched provider page, fetch failure, or not-found outcome in PostgreSQL.
///
/// # Errors
///
/// Returns an invalid-argument error on bad page addresses or empty response
/// metadata, or a database error when the transaction fails.
pub async fn store_page(pool: &PgPool, input: StorePageInput) -> Result<StorePageOutcome, Error> {
    validate_page_address(&input.page_key.provider, &input.page_key.address)?;
    for response in &input.responses {
        if response.request_name.is_empty() || response.content_type.is_empty() {
            return Err(Error::invalid_argument(
                "page response request_name and content_type must not be empty",
            ));
        }
    }

    let mut tx = pool.begin().await?;
    let existing_page = lock_existing_page(&mut tx, &input.page_key).await?;

    let outcome = if input.is_not_found {
        handle_not_found(&mut tx, &input, existing_page.as_ref()).await?
    } else if input.responses.is_empty() && !input.is_dead {
        handle_fetch_failure(&mut tx, &input, existing_page.as_ref()).await?
    } else {
        handle_success(&mut tx, &input, existing_page.as_ref()).await?
    };

    if outcome.has_changed {
        notify_linked_source_works(&mut tx, &input.page_key).await?;
    }

    tx.commit().await?;
    Ok(outcome)
}

async fn notify_linked_source_works(
    tx: &mut Transaction<'_, Postgres>,
    page_key: &PageKeyInput,
) -> Result<(), Error> {
    sqlx::query!(
        r#"
        UPDATE anime
        SET version = version + 1,
            merge_requested_at = COALESCE(merge_requested_at, now())
        WHERE id IN (
            SELECT anime_id
            FROM anime_provider_page
            WHERE provider = $1 AND page_kind = $2 AND address = $3 AND role = 'source'
            ORDER BY anime_id
        )
          AND deleted_at IS NULL
        "#,
        page_key.provider,
        page_key.page_kind.as_str(),
        page_key.address,
    )
    .execute(&mut **tx)
    .await?;

    Ok(())
}

async fn lock_existing_page(
    tx: &mut Transaction<'_, Postgres>,
    page_key: &PageKeyInput,
) -> Result<Option<ExistingPageRow>, Error> {
    // Serialize creation as well as updates when no row exists yet.
    sqlx::query!(
        "SELECT pg_advisory_xact_lock(hashtextextended($1 || ':' || $2 || ':' || $3, 0))",
        page_key.provider,
        page_key.page_kind.as_str(),
        page_key.address,
    )
    .execute(&mut **tx)
    .await?;

    let row = sqlx::query_as!(
        ExistingPageRow,
        "SELECT fetched_at, is_dead, missing_since, missing_last_at, missing_answers, responses_hash, fetch_failed_at
         FROM provider_page
         WHERE provider = $1 AND page_kind = $2 AND address = $3
         FOR UPDATE",
        page_key.provider,
        page_key.page_kind.as_str(),
        page_key.address,
    )
    .fetch_optional(&mut **tx)
    .await?;

    Ok(row)
}

async fn handle_not_found(
    tx: &mut Transaction<'_, Postgres>,
    input: &StorePageInput,
    existing_page: Option<&ExistingPageRow>,
) -> Result<StorePageOutcome, Error> {
    if is_stale_observation(existing_page, input.fetched_at) {
        return Ok(StorePageOutcome { has_changed: false });
    }

    let missing_since = existing_page
        .and_then(|row| row.missing_since)
        .unwrap_or(input.fetched_at);
    let next_answers = existing_page.map_or(1, |row| row.missing_answers.saturating_add(1));
    let elapsed = input.fetched_at.signed_duration_since(missing_since);
    let reached_dead_threshold = next_answers >= 2 && elapsed >= Duration::days(1);

    let next_is_dead = if reached_dead_threshold && !input.dead_page_review {
        true
    } else {
        existing_page.is_some_and(|row| row.is_dead) || input.is_dead
    };

    let page_key = &input.page_key;
    sqlx::query!(
        "INSERT INTO provider_page (
             provider, page_kind, address, is_dead, missing_since, missing_last_at, missing_answers
         )
         VALUES ($1, $2, $3, $4, $5, $6, $7)
         ON CONFLICT (provider, page_kind, address) DO UPDATE SET
             is_dead = EXCLUDED.is_dead,
             missing_since = EXCLUDED.missing_since,
             missing_last_at = EXCLUDED.missing_last_at,
             missing_answers = EXCLUDED.missing_answers",
        page_key.provider,
        page_key.page_kind.as_str(),
        page_key.address,
        next_is_dead,
        missing_since,
        input.fetched_at,
        next_answers,
    )
    .execute(&mut **tx)
    .await?;

    if reached_dead_threshold && input.dead_page_review && !next_is_dead {
        enqueue_dead_page_approval(tx, page_key).await?;
    }

    let was_dead = existing_page.is_some_and(|row| row.is_dead);
    Ok(StorePageOutcome {
        has_changed: next_is_dead != was_dead,
    })
}

async fn enqueue_dead_page_approval(
    tx: &mut Transaction<'_, Postgres>,
    page_key: &PageKeyInput,
) -> Result<(), Error> {
    let proposal = encode_page_key_proposal(page_key);
    let already_open = sqlx::query_scalar!(
        "SELECT EXISTS(
             SELECT 1 FROM approval_request
             WHERE kind = 'dead_page'
               AND status = 'open'
               AND proposal_message = 'shared_proto.v1.PageKey'
               AND proposal = $1
         )",
        proposal.as_slice(),
    )
    .fetch_one(&mut **tx)
    .await?
    .unwrap_or(false);

    if !already_open {
        sqlx::query!(
            "INSERT INTO approval_request (
                 kind, proposal_message, proposal, status, created_by
             )
             VALUES ('dead_page', 'shared_proto.v1.PageKey', $1, 'open', 'postgres_service')",
            proposal.as_slice(),
        )
        .execute(&mut **tx)
        .await?;
    }
    Ok(())
}

async fn close_open_dead_page_approval(
    tx: &mut Transaction<'_, Postgres>,
    page_key: &PageKeyInput,
) -> Result<(), Error> {
    let proposal = encode_page_key_proposal(page_key);
    sqlx::query!(
        "UPDATE approval_request
         SET status = 'rejected', decided_by = 'postgres_service', decided_at = now()
         WHERE kind = 'dead_page'
           AND status = 'open'
           AND proposal_message = 'shared_proto.v1.PageKey'
           AND proposal = $1",
        proposal.as_slice(),
    )
    .execute(&mut **tx)
    .await?;
    Ok(())
}

async fn handle_fetch_failure(
    tx: &mut Transaction<'_, Postgres>,
    input: &StorePageInput,
    existing_page: Option<&ExistingPageRow>,
) -> Result<StorePageOutcome, Error> {
    if is_stale_observation(existing_page, input.fetched_at) {
        return Ok(StorePageOutcome { has_changed: false });
    }

    let page_key = &input.page_key;
    sqlx::query!(
        "INSERT INTO provider_page (
             provider, page_kind, address, fetch_failed_at, fetch_failures
         )
         VALUES ($1, $2, $3, $4, 1)
         ON CONFLICT (provider, page_kind, address) DO UPDATE SET
             fetch_failed_at = EXCLUDED.fetch_failed_at,
             fetch_failures = provider_page.fetch_failures + 1",
        page_key.provider,
        page_key.page_kind.as_str(),
        page_key.address,
        input.fetched_at,
    )
    .execute(&mut **tx)
    .await?;

    Ok(StorePageOutcome { has_changed: false })
}

async fn handle_success(
    tx: &mut Transaction<'_, Postgres>,
    input: &StorePageInput,
    existing_page: Option<&ExistingPageRow>,
) -> Result<StorePageOutcome, Error> {
    if is_stale_observation(existing_page, input.fetched_at) {
        return Ok(StorePageOutcome { has_changed: false });
    }

    let new_hash = compute_responses_hash(&input.responses);
    let responses_changed =
        existing_page.and_then(|row| row.responses_hash.as_deref()) != Some(new_hash.as_str());
    let dead_changed = existing_page.is_some_and(|row| row.is_dead) != input.is_dead;

    let page_key = &input.page_key;
    sqlx::query!(
        "INSERT INTO provider_page (
             provider, page_kind, address, fetched_at, is_dead,
             missing_since, missing_last_at, missing_answers, responses_hash,
             fetch_failed_at, fetch_failures
         )
         VALUES ($1, $2, $3, $4, $5, NULL, NULL, 0, $6, NULL, 0)
         ON CONFLICT (provider, page_kind, address) DO UPDATE SET
             fetched_at = EXCLUDED.fetched_at,
             is_dead = EXCLUDED.is_dead,
             missing_since = NULL,
             missing_last_at = NULL,
             missing_answers = 0,
             responses_hash = EXCLUDED.responses_hash,
             fetch_failed_at = NULL,
             fetch_failures = 0",
        page_key.provider,
        page_key.page_kind.as_str(),
        page_key.address,
        input.fetched_at,
        input.is_dead,
        new_hash,
    )
    .execute(&mut **tx)
    .await?;

    if existing_page.is_some_and(|row| row.missing_since.is_some()) {
        close_open_dead_page_approval(tx, page_key).await?;
    }

    if responses_changed {
        sqlx::query!(
            "DELETE FROM provider_page_response
             WHERE provider = $1 AND page_kind = $2 AND address = $3",
            page_key.provider,
            page_key.page_kind.as_str(),
            page_key.address,
        )
        .execute(&mut **tx)
        .await?;

        for response in &input.responses {
            let compressed_body = compress_body(&response.body)?;
            sqlx::query!(
                "INSERT INTO provider_page_response (
                     provider, page_kind, address, request_name, content_type, body
                 )
                 VALUES ($1, $2, $3, $4, $5, $6)",
                page_key.provider,
                page_key.page_kind.as_str(),
                page_key.address,
                response.request_name,
                response.content_type,
                compressed_body,
            )
            .execute(&mut **tx)
            .await?;
        }
    }

    tracing::event!(
        name: "postgres_db.provider_page.stored",
        Level::DEBUG,
        provider = %page_key.provider,
        page_kind = page_key.page_kind.as_str(),
        address = %page_key.address,
        has_changed = responses_changed || dead_changed,
        "stored provider page",
    );

    Ok(StorePageOutcome {
        has_changed: responses_changed || dead_changed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_page_address_checks_numeric_and_slug_providers() {
        assert!(validate_page_address("mal", "21").is_ok());
        assert!(validate_page_address("anidb", "12345").is_ok());
        assert!(validate_page_address("mal", "21extra").is_err());
        assert!(validate_page_address("kitsu", "cowboy-bebop").is_ok());
        assert!(validate_page_address("kitsu", "bad/slug").is_err());
        assert!(validate_page_address("unknown_provider", "21").is_err());
    }

    #[test]
    fn compute_responses_hash_is_order_independent() {
        let first = PageResponseInput {
            request_name: "main".to_owned(),
            content_type: "application/json".to_owned(),
            body: b"{}".to_vec(),
        };
        let second = PageResponseInput {
            request_name: "characters".to_owned(),
            content_type: "application/json".to_owned(),
            body: b"[]".to_vec(),
        };

        let hash_a = compute_responses_hash(&[first.clone(), second.clone()]);
        let hash_b = compute_responses_hash(&[second, first]);
        assert_eq!(hash_a, hash_b);
        assert_eq!(hash_a.len(), 64);
    }

    #[test]
    fn encode_page_key_proposal_matches_protobuf_wire_format() {
        let encoded_proposal = encode_page_key_proposal(&PageKeyInput {
            provider: "mal".to_owned(),
            page_kind: PageKind::Anime,
            address: "21".to_owned(),
        });
        let decoded_proposal = ProtoPageKey::decode(encoded_proposal.as_slice());
        assert_eq!(
            decoded_proposal,
            Ok(ProtoPageKey {
                provider: "mal".to_owned(),
                page_kind: 1,
                address: "21".to_owned(),
            })
        );
    }

    #[test]
    fn is_stale_observation_checks_all_observation_timestamps() {
        let base_timestamp = Utc::now();
        let existing_page_row = ExistingPageRow {
            fetched_at: Some(base_timestamp),
            is_dead: false,
            missing_since: Some(base_timestamp + Duration::hours(2)),
            missing_last_at: Some(base_timestamp + Duration::hours(4)),
            missing_answers: 1,
            responses_hash: None,
            fetch_failed_at: Some(base_timestamp + Duration::hours(1)),
        };

        assert!(is_stale_observation(
            Some(&existing_page_row),
            base_timestamp + Duration::hours(1),
        ));
        assert!(is_stale_observation(
            Some(&existing_page_row),
            base_timestamp + Duration::hours(4),
        ));
        assert!(!is_stale_observation(
            Some(&existing_page_row),
            base_timestamp + Duration::hours(5),
        ));
        assert!(!is_stale_observation(None, base_timestamp));
    }
}
