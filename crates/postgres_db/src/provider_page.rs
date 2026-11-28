//! Provider page storage, response compression, and dead-page tracking.

use std::collections::{BTreeMap, HashSet};
use std::fmt::Write as _;
use std::io::Write as _;

use chrono::{DateTime, TimeDelta, Utc};
use echora_domain::{EntityType, PageKey};
use echora_protos::shared_proto::v1::PageKey as ProtoPageKey;
use flate2::Compression;
use flate2::write::GzEncoder;
use prost::Message as _;
use sha2::{Digest as _, Sha256};
use sqlx::{PgPool, Postgres, Transaction};
use tracing::Level;

use url::Url;
use uuid::Uuid;

use crate::Error;

/// Minimum consecutive 404 observations required before a page is considered dead.
const MIN_NOT_FOUND_OBSERVATIONS: i32 = 2;

/// Minimum elapsed time since the first 404 before a page is considered dead.
const MIN_MISSING_DURATION: TimeDelta = TimeDelta::days(1);

/// Furthest a fetch time may lie ahead of the database's clock.
///
/// A fetch stamped later than this counts as the newest answer and would block every
/// real fetch of the page after it, so it is refused. Five minutes covers the clock
/// drift between a fetching worker and the database without letting a wrong clock
/// through.
const MAX_FETCH_CLOCK_AHEAD: TimeDelta = TimeDelta::minutes(5);

/// A provider page with its URL.
///
/// A merge chooses the URL of each page its entity holds; a save that only names a page
/// passes the URL its merge reported, which the page keeps until its holder chooses one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageSource {
    /// The page's provider, entity type and address.
    pub page_key: PageKey,
    /// The page's URL, a provider-reported one with its slug when known, or `None` when no
    /// URL is known, such as a known episode's page whose URL is not stored.
    pub source: Option<Url>,
    /// The title the page's provider gives it (a character's name for a character page), or
    /// `None` when unknown.
    pub title: Option<String>,
}

/// Records the URL and title of each page an entity holds as a source, where they changed.
///
/// A page's URL and title depend only on the page, so they are stored once on
/// `provider_page`. The entity that holds a page as its source chooses them and replaces what
/// is stored; a save that only names the page fills them while they are empty, through
/// [`record_referenced_page_sources`]. A URL or title given as `None` keeps the stored one,
/// since the merge did not learn it this time. A page given here that the entity does not
/// hold as a source is left unchanged. The owned page rows
/// are locked in key order before the update, so two saves of one entity cannot deadlock.
///
/// Call it before the entity's own row is locked: every writer locks pages first, as
/// `StorePage` does.
///
/// # Errors
///
/// Returns an invalid-argument error when a page key is for another kind of entity or is
/// given twice, or a database error when the update fails.
pub(crate) async fn record_page_sources(
    tx: &mut Transaction<'_, Postgres>,
    entity_type: EntityType,
    entity_id: Uuid,
    page_sources: &[PageSource],
) -> Result<(), Error> {
    if page_sources.is_empty() {
        return Ok(());
    }
    let mut seen_page_keys = HashSet::with_capacity(page_sources.len());
    for page_source in page_sources {
        let page_key = &page_source.page_key;
        let page_label = format!(
            "{}:{}:{}",
            page_key.provider(),
            page_key.entity_type(),
            page_key.address()
        );
        if page_key.entity_type() != entity_type {
            return Err(Error::invalid_argument(format!(
                "page source {page_label} is not a {entity_type} page"
            )));
        }
        if !seen_page_keys.insert(page_key) {
            return Err(Error::invalid_argument(format!(
                "page source {page_label} is given twice"
            )));
        }
    }

    let providers: Vec<String> = page_sources
        .iter()
        .map(|page_source| page_source.page_key.provider().as_str().to_owned())
        .collect();
    let addresses: Vec<String> = page_sources
        .iter()
        .map(|page_source| page_source.page_key.address().as_str().to_owned())
        .collect();
    let sources: Vec<Option<&str>> = page_sources
        .iter()
        .map(|page_source| page_source.source.as_ref().map(Url::as_str))
        .collect();
    let titles: Vec<Option<&str>> = page_sources
        .iter()
        .map(|page_source| page_source.title.as_deref())
        .collect();

    sqlx::query!(
        r#"
        WITH given AS (
            SELECT given.provider, $2::text AS entity_type, given.address, given.source,
                   given.title
            FROM unnest($3::text[], $4::text[], $5::text[], $6::text[])
                 AS given(provider, address, source, title)
        ),
        owned AS (
            SELECT page.provider, page.entity_type, page.address
            FROM provider_page page
            JOIN given USING (provider, entity_type, address)
            WHERE ($2 = 'anime' AND EXISTS (
                      SELECT 1 FROM anime_source link
                      WHERE link.anime_id = $1
                        AND link.provider = page.provider
                        AND link.entity_type = page.entity_type
                        AND link.address = page.address))
               OR ($2 = 'episode' AND EXISTS (
                      SELECT 1 FROM episode_source link
                      WHERE link.episode_id = $1
                        AND link.provider = page.provider
                        AND link.entity_type = page.entity_type
                        AND link.address = page.address))
               OR ($2 = 'character' AND EXISTS (
                      SELECT 1 FROM character_source link
                      WHERE link.character_id = $1
                        AND link.provider = page.provider
                        AND link.entity_type = page.entity_type
                        AND link.address = page.address))
            ORDER BY page.provider, page.entity_type, page.address
            FOR NO KEY UPDATE OF page
        )
        UPDATE provider_page page
        SET source = coalesce(given.source, page.source),
            title = coalesce(given.title, page.title)
        FROM given
        JOIN owned USING (provider, entity_type, address)
        WHERE page.provider = given.provider
          AND page.entity_type = given.entity_type
          AND page.address = given.address
          AND (page.source IS DISTINCT FROM coalesce(given.source, page.source)
               OR page.title IS DISTINCT FROM coalesce(given.title, page.title))
        "#,
        entity_id,
        entity_type.as_str(),
        &providers,
        &addresses,
        &sources as &[Option<&str>],
        &titles as &[Option<&str>]
    )
    .execute(&mut **tx)
    .await?;
    Ok(())
}

/// The URL and title a save gives for one page it names.
#[derive(Debug, Default)]
pub(crate) struct NamedPageDetails<'a> {
    /// The longest URL given for the page.
    pub(crate) source: Option<&'a str>,
    /// The first title given for the page.
    pub(crate) title: Option<&'a str>,
}

/// Collects the URL and title given for each named page, in key order.
///
/// Several entries can name one page through different spellings of its URL. The longest
/// keeps the provider's slug, the same choice the merges make for an entity's own pages, and
/// the first title given is kept. The sorted keys give every writer one row order. A page
/// given with neither is left out. Provider pages are keyed by their [`PageKey`], and manga
/// pages by their provider and external ID.
pub(crate) fn collect_named_page_details<'a, K: Ord>(
    named_pages: impl IntoIterator<Item = (K, NamedPageDetails<'a>)>,
) -> BTreeMap<K, NamedPageDetails<'a>> {
    let mut details_by_page: BTreeMap<K, NamedPageDetails<'a>> = BTreeMap::new();
    for (page_key, named_details) in named_pages {
        if named_details.source.is_none() && named_details.title.is_none() {
            continue;
        }
        let page_details = details_by_page.entry(page_key).or_default();
        if named_details.source.map(str::len) > page_details.source.map(str::len) {
            page_details.source = named_details.source;
        }
        page_details.title = page_details.title.or(named_details.title);
    }
    details_by_page
}

/// Records the URL and title of each named page, where none is stored.
///
/// A relation, an appearance or an episode's character names a page by its key, and the
/// merge reports the URL it came from. Storing that URL lets a read show the link before
/// any entity holds the page. The entity that later holds the page replaces it with the URL
/// its own merge chooses, through [`record_page_sources`].
///
/// A missing page row is added with `ON CONFLICT DO NOTHING`, which does not lock a row that
/// already exists. An empty URL or title is filled only on rows locked with `SKIP LOCKED`: a row
/// another save holds is skipped, since that save is the page's holder or another save
/// filling the same URL. The save therefore never waits on a page row another save holds.
///
/// # Errors
///
/// Returns a database error when the insert or the update fails.
pub(crate) async fn record_referenced_page_sources<'a>(
    tx: &mut Transaction<'_, Postgres>,
    named_page_sources: impl IntoIterator<Item = &'a PageSource>,
) -> Result<(), Error> {
    let details_by_page =
        collect_named_page_details(named_page_sources.into_iter().map(|page_source| {
            (
                &page_source.page_key,
                NamedPageDetails {
                    source: page_source.source.as_ref().map(Url::as_str),
                    title: page_source.title.as_deref(),
                },
            )
        }));
    if details_by_page.is_empty() {
        return Ok(());
    }
    let mut providers = Vec::with_capacity(details_by_page.len());
    let mut entity_types = Vec::with_capacity(details_by_page.len());
    let mut addresses = Vec::with_capacity(details_by_page.len());
    let mut sources = Vec::with_capacity(details_by_page.len());
    let mut titles = Vec::with_capacity(details_by_page.len());
    for (page_key, page_details) in details_by_page {
        providers.push(page_key.provider().as_str());
        entity_types.push(page_key.entity_type().as_str());
        addresses.push(page_key.address().as_str());
        sources.push(page_details.source);
        titles.push(page_details.title);
    }

    sqlx::query!(
        r#"
        WITH named AS (
            SELECT named.provider, named.entity_type, named.address, named.source,
                   named.title
            FROM unnest($1::text[], $2::text[], $3::text[], $4::text[], $5::text[])
                 AS named(provider, entity_type, address, source, title)
        ),
        added AS (
            INSERT INTO provider_page (provider, entity_type, address, source, title)
            SELECT provider, entity_type, address, source, title
            FROM named
            ON CONFLICT (provider, entity_type, address) DO NOTHING
        ),
        unnamed AS (
            SELECT page.provider, page.entity_type, page.address
            FROM provider_page page
            JOIN named USING (provider, entity_type, address)
            WHERE (page.source IS NULL AND named.source IS NOT NULL)
               OR (page.title IS NULL AND named.title IS NOT NULL)
            ORDER BY page.provider, page.entity_type, page.address
            FOR NO KEY UPDATE OF page SKIP LOCKED
        )
        UPDATE provider_page page
        SET source = coalesce(page.source, named.source),
            title = coalesce(page.title, named.title)
        FROM named
        JOIN unnamed USING (provider, entity_type, address)
        WHERE page.provider = named.provider
          AND page.entity_type = named.entity_type
          AND page.address = named.address
        "#,
        &providers as &[&str],
        &entity_types as &[&str],
        &addresses as &[&str],
        &sources as &[Option<&str>],
        &titles as &[Option<&str>]
    )
    .execute(&mut **tx)
    .await?;
    Ok(())
}

/// Encodes `page_key` as a `shared_proto.v1.PageKey` protobuf payload.
///
/// # Examples
///
/// ```
/// use echora_domain::{EntityType, PageKey, Provider};
/// use postgres_db::encode_page_key_proto;
///
/// let key = PageKey::new(Provider::Mal, EntityType::Anime, "21")?;
/// let bytes = encode_page_key_proto(&key);
/// // provider "mal" (field 1), entity type ANIME = 1 (field 2), address "21" (field 3)
/// assert_eq!(bytes, b"\x0a\x03mal\x10\x01\x1a\x0221");
/// # Ok::<(), echora_domain::KeyError>(())
/// ```
#[must_use]
pub fn encode_page_key_proto(page_key: &PageKey) -> Vec<u8> {
    ProtoPageKey::from(page_key).encode_to_vec()
}

/// One raw provider response belonging to a fetched page.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageResponse {
    /// Logical request name (for example `"main"` or `"characters"`).
    pub request_name: String,
    /// MIME content type returned by the provider.
    pub content_type: String,
    /// Uncompressed response body bytes received from the provider.
    pub body: Vec<u8>,
}

/// Outcome of attempting to fetch a provider page.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FetchOutcome {
    /// Page fetched successfully with one or more responses.
    Success {
        /// Uncompressed responses received for the page.
        responses: Vec<PageResponse>,
    },
    /// Provider returned an explicit not-found response.
    NotFound {
        /// Whether reaching the dead threshold requires human review.
        ///
        /// When `true`, an approval request is enqueued instead of automatically
        /// setting `is_dead = true`.
        require_review: bool,
    },
    /// Fetch failed due to a transient error.
    Failed,
}

/// Observed result of scraping a provider page.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageObservation {
    /// Target provider page key.
    pub page_key: PageKey,
    /// Timestamp when the fetch occurred.
    pub observed_at: DateTime<Utc>,
    /// Outcome recorded during the fetch.
    pub outcome: FetchOutcome,
}

/// Result returned after recording a page observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecordOutcome {
    /// Whether stored response content or dead status changed.
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

#[derive(Debug)]
struct PreparedResponse {
    request_name: String,
    content_type: String,
    compressed_body: Vec<u8>,
}

#[derive(Debug)]
enum PreparedOutcome {
    Success {
        responses_hash: String,
        responses: Vec<PreparedResponse>,
    },
    NotFound {
        require_review: bool,
    },
    Failed,
}

/// Reports whether `observed_at` is no newer than any answer already stored for the page.
///
/// A retried or late fetch then never replaces newer data.
fn is_stale_observation(
    existing_page: Option<&ExistingPageRow>,
    observed_at: DateTime<Utc>,
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
        .is_some_and(|latest| observed_at.timestamp_micros() <= latest.timestamp_micros())
}

/// Computes a deterministic SHA-256 hex digest over uncompressed `responses`.
///
/// Responses are sorted by `request_name` before hashing so input slice order
/// does not affect the resulting digest.
///
/// # Examples
///
/// ```
/// # use postgres_db::{compute_responses_hash, PageResponse};
/// let main_response = PageResponse {
///     request_name: "main".to_owned(),
///     content_type: "application/json".to_owned(),
///     body: b"{}".to_vec(),
/// };
/// let characters_response = PageResponse {
///     request_name: "characters".to_owned(),
///     content_type: "application/json".to_owned(),
///     body: b"[]".to_vec(),
/// };
///
/// assert_eq!(
///     compute_responses_hash(&[main_response.clone()]),
///     "10adfb9736ff0c549610edd040821b900d0f192a44dac409b4b0c0cd2115e0f3"
/// );
/// // The responses are sorted by request name first, so their order does not matter.
/// assert_eq!(
///     compute_responses_hash(&[main_response.clone(), characters_response.clone()]),
///     compute_responses_hash(&[characters_response, main_response]),
/// );
/// ```
#[must_use]
pub fn compute_responses_hash(responses: &[PageResponse]) -> String {
    let mut ordered: Vec<&PageResponse> = responses.iter().collect();
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
        let _ = write!(&mut hex, "{byte:02x}");
    }
    hex
}

/// Compresses a response body with gzip for storage.
///
/// # Panics
///
/// Panics only on a bug in the gzip encoder, since writing into memory cannot fail.
#[expect(
    clippy::expect_used,
    reason = "a gzip encoder writing into a Vec has no I/O that can fail"
)]
fn compress_body(body: &[u8]) -> Vec<u8> {
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder
        .write_all(body)
        .expect("gzip encoder writing into memory cannot fail");
    encoder
        .finish()
        .expect("gzip encoder finishing into memory cannot fail")
}

/// Checks a fetch outcome and compresses its responses before the transaction opens.
///
/// # Errors
///
/// Returns an invalid-argument error for a successful fetch with no responses, a blank
/// request name or content type, or a repeated request name.
fn prepare_outcome(outcome: FetchOutcome) -> Result<PreparedOutcome, Error> {
    match outcome {
        FetchOutcome::Success { responses } => {
            if responses.is_empty() {
                return Err(Error::invalid_argument(
                    "successful fetch outcome must contain at least one response",
                ));
            }

            let mut seen_names = HashSet::with_capacity(responses.len());
            for response in &responses {
                if response.request_name.is_empty() || response.content_type.is_empty() {
                    return Err(Error::invalid_argument(
                        "page response request_name and content_type must not be empty",
                    ));
                }
                if !seen_names.insert(response.request_name.as_str()) {
                    return Err(Error::invalid_argument(format!(
                        "duplicate page response request_name: {:?}",
                        response.request_name
                    )));
                }
            }

            let responses_hash = compute_responses_hash(&responses);
            let prepared_responses = responses
                .into_iter()
                .map(|response| PreparedResponse {
                    compressed_body: compress_body(&response.body),
                    request_name: response.request_name,
                    content_type: response.content_type,
                })
                .collect();

            Ok(PreparedOutcome::Success {
                responses_hash,
                responses: prepared_responses,
            })
        }
        FetchOutcome::NotFound { require_review } => {
            Ok(PreparedOutcome::NotFound { require_review })
        }
        FetchOutcome::Failed => Ok(PreparedOutcome::Failed),
    }
}

/// Records what one fetch of a provider page found.
///
/// - A fetch no newer than the page's stored answers changes nothing, so a retried or late
///   fetch never replaces newer data.
/// - A successful fetch stores the responses, and counts as a change only when their hash
///   differs from the stored one.
/// - A "not found" answer keeps the stored responses. Two answers a day apart open a
///   `dead_page` review, or mark the page dead when the provider needs no review.
/// - A failed fetch counts one more failure since the last success.
///
/// A change raises the version of every entity built from the page and requests its merge,
/// so a merge that loaded before the change starts again.
///
/// # Examples
///
/// ```no_run
/// # use chrono::Utc;
/// # use sqlx::PgPool;
/// # use echora_domain::{EntityType, PageKey, Provider};
/// # use postgres_db::{FetchOutcome, PageObservation, PageResponse, record_page_observation};
/// # async fn example(pool: &PgPool) -> Result<(), postgres_db::Error> {
/// let page_key = PageKey::new(Provider::Mal, EntityType::Anime, "21")?;
///
/// // A fetch that failed every attempt only counts the failure: nothing changed.
/// let failed_fetch = PageObservation {
///     page_key: page_key.clone(),
///     observed_at: Utc::now(),
///     outcome: FetchOutcome::Failed,
/// };
/// assert!(!record_page_observation(pool, failed_fetch).await?.has_changed);
///
/// // The first successful fetch stores the page's responses, which is a change.
/// let first_success = PageObservation {
///     page_key: page_key.clone(),
///     observed_at: Utc::now(),
///     outcome: FetchOutcome::Success {
///         responses: vec![PageResponse {
///             request_name: "main".to_owned(),
///             content_type: "application/json".to_owned(),
///             body: br#"{"id":21}"#.to_vec(),
///         }],
///     },
/// };
/// assert!(record_page_observation(pool, first_success).await?.has_changed);
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns an invalid-argument error for a fetch time more than five minutes ahead of the
/// database's clock, for a successful fetch with no responses, a blank request name or
/// content type, or a repeated request name, and a database error when a query fails.
pub async fn record_page_observation(
    pool: &PgPool,
    observation: PageObservation,
) -> Result<RecordOutcome, Error> {
    let prepared_outcome = prepare_outcome(observation.outcome)?;

    let mut tx = pool.begin().await?;
    let transaction_time = crate::read_transaction_time(&mut tx).await?;
    if observation.observed_at > transaction_time + MAX_FETCH_CLOCK_AHEAD {
        return Err(Error::invalid_argument(format!(
            "fetch time {} is later than the database clock {transaction_time}",
            observation.observed_at
        )));
    }
    let existing_page = lock_existing_page(&mut tx, &observation.page_key).await?;

    if is_stale_observation(existing_page.as_ref(), observation.observed_at) {
        return Ok(RecordOutcome { has_changed: false });
    }

    let record_outcome = match prepared_outcome {
        PreparedOutcome::NotFound { require_review } => {
            record_not_found(
                &mut tx,
                &observation.page_key,
                observation.observed_at,
                require_review,
                existing_page.as_ref(),
            )
            .await?
        }
        PreparedOutcome::Failed => {
            record_fetch_failure(&mut tx, &observation.page_key, observation.observed_at).await?
        }
        PreparedOutcome::Success {
            responses_hash,
            responses,
        } => {
            record_success(
                &mut tx,
                &observation.page_key,
                observation.observed_at,
                &responses_hash,
                responses,
                existing_page.as_ref(),
            )
            .await?
        }
    };

    if record_outcome.has_changed {
        queue_page_owners_for_merge(&mut tx, &observation.page_key).await?;
    }

    tx.commit().await?;
    Ok(record_outcome)
}

/// Raises the version of every entity built from `page_key` and requests its merge.
///
/// A merge that loaded before the page changed then fails its version check and starts again.
/// - A work page queues the work that holds it as a source page.
/// - An episode page queues the episode that owns it.
/// - A character page queues the character that owns it.
///
/// # Errors
///
/// Returns a database error when an update fails.
async fn queue_page_owners_for_merge(
    tx: &mut Transaction<'_, Postgres>,
    page_key: &PageKey,
) -> Result<(), Error> {
    let provider = page_key.provider().as_str();
    let entity_type = page_key.entity_type().as_str();
    let address = page_key.address().as_str();
    match page_key.entity_type() {
        EntityType::Anime => {
            sqlx::query!(
                r#"
                UPDATE anime
                SET version = version + 1,
                    merge_requested_at = COALESCE(merge_requested_at, now())
                WHERE id IN (
                    SELECT anime_id
                    FROM anime_source
                    WHERE provider = $1 AND entity_type = $2 AND address = $3
                )
                  AND deleted_at IS NULL
                "#,
                provider,
                entity_type,
                address,
            )
            .execute(&mut **tx)
            .await?;
        }
        EntityType::Episode => {
            sqlx::query!(
                r#"
                UPDATE episode
                SET version = version + 1,
                    merge_requested_at = COALESCE(merge_requested_at, now())
                WHERE id IN (
                    SELECT episode_id
                    FROM episode_source
                    WHERE provider = $1 AND entity_type = $2 AND address = $3
                )
                  AND deleted_at IS NULL
                "#,
                provider,
                entity_type,
                address,
            )
            .execute(&mut **tx)
            .await?;
        }
        EntityType::Character => {
            sqlx::query!(
                r#"
                UPDATE character
                SET version = version + 1,
                    merge_requested_at = COALESCE(merge_requested_at, now())
                WHERE id IN (
                    SELECT character_id
                    FROM character_source
                    WHERE provider = $1 AND entity_type = $2 AND address = $3
                )
                  AND deleted_at IS NULL
                "#,
                provider,
                entity_type,
                address,
            )
            .execute(&mut **tx)
            .await?;
        }
    }
    Ok(())
}

/// Locks a page key for this transaction and reads the stored page, if any.
///
/// The lock also covers a page not stored yet, so two first fetches cannot race.
///
/// # Errors
///
/// Returns a database error when a query fails.
async fn lock_existing_page(
    tx: &mut Transaction<'_, Postgres>,
    page_key: &PageKey,
) -> Result<Option<ExistingPageRow>, Error> {
    // Serialize creation as well as updates when no row exists yet.
    sqlx::query!(
        "SELECT pg_advisory_xact_lock(hashtextextended($1 || ':' || $2 || ':' || $3, 0))",
        page_key.provider().as_str(),
        page_key.entity_type().as_str(),
        page_key.address().as_str(),
    )
    .execute(&mut **tx)
    .await?;

    let row = sqlx::query_as!(
        ExistingPageRow,
        "SELECT fetched_at, is_dead, missing_since, missing_last_at, missing_answers, responses_hash, fetch_failed_at
         FROM provider_page
         WHERE provider = $1 AND entity_type = $2 AND address = $3
         FOR UPDATE",
        page_key.provider().as_str(),
        page_key.entity_type().as_str(),
        page_key.address().as_str(),
    )
    .fetch_optional(&mut **tx)
    .await?;

    Ok(row)
}

/// Records a provider's "not found" answer for a page and keeps its responses.
///
/// - Two answers at least a day apart reach the dead threshold.
/// - With review required, reaching the threshold opens a `dead_page` approval request.
/// - Without review, reaching the threshold marks the page dead.
///
/// # Errors
///
/// Returns a database error when a query fails.
async fn record_not_found(
    tx: &mut Transaction<'_, Postgres>,
    page_key: &PageKey,
    observed_at: DateTime<Utc>,
    require_review: bool,
    existing_page: Option<&ExistingPageRow>,
) -> Result<RecordOutcome, Error> {
    let missing_since = existing_page
        .and_then(|row| row.missing_since)
        .unwrap_or(observed_at);
    let not_found_count = existing_page.map_or(1, |row| row.missing_answers.saturating_add(1));
    let elapsed = observed_at.signed_duration_since(missing_since);
    let reached_dead_threshold =
        not_found_count >= MIN_NOT_FOUND_OBSERVATIONS && elapsed >= MIN_MISSING_DURATION;

    let was_dead = existing_page.is_some_and(|row| row.is_dead);
    let next_is_dead = if reached_dead_threshold && !require_review {
        true
    } else {
        was_dead
    };

    sqlx::query!(
        "INSERT INTO provider_page (
             provider, entity_type, address, is_dead, missing_since, missing_last_at, missing_answers
         )
         VALUES ($1, $2, $3, $4, $5, $6, $7)
         ON CONFLICT (provider, entity_type, address) DO UPDATE SET
             is_dead = EXCLUDED.is_dead,
             missing_since = EXCLUDED.missing_since,
             missing_last_at = EXCLUDED.missing_last_at,
             missing_answers = EXCLUDED.missing_answers",
        page_key.provider().as_str(),
        page_key.entity_type().as_str(),
        page_key.address().as_str(),
        next_is_dead,
        missing_since,
        observed_at,
        not_found_count,
    )
    .execute(&mut **tx)
    .await?;

    if reached_dead_threshold && require_review && !next_is_dead {
        enqueue_dead_page_approval(tx, page_key).await?;
    }

    Ok(RecordOutcome {
        has_changed: next_is_dead != was_dead,
    })
}

/// Opens a `dead_page` approval request for a page, unless one is already open.
///
/// # Errors
///
/// Returns a database error when a query fails.
async fn enqueue_dead_page_approval(
    tx: &mut Transaction<'_, Postgres>,
    page_key: &PageKey,
) -> Result<(), Error> {
    let proposal = encode_page_key_proto(page_key);
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

/// Closes a page's open `dead_page` approval request as rejected.
///
/// A successful fetch shows the page is not dead, so the request no longer applies.
///
/// # Errors
///
/// Returns a database error when a query fails.
async fn reject_open_dead_page_approval(
    tx: &mut Transaction<'_, Postgres>,
    page_key: &PageKey,
) -> Result<(), Error> {
    let proposal = encode_page_key_proto(page_key);
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

/// Records a fetch that failed every attempt, counting failures since the last success.
///
/// # Errors
///
/// Returns a database error when a query fails.
async fn record_fetch_failure(
    tx: &mut Transaction<'_, Postgres>,
    page_key: &PageKey,
    observed_at: DateTime<Utc>,
) -> Result<RecordOutcome, Error> {
    sqlx::query!(
        "INSERT INTO provider_page (
             provider, entity_type, address, fetch_failed_at, fetch_failures
         )
         VALUES ($1, $2, $3, $4, 1)
         ON CONFLICT (provider, entity_type, address) DO UPDATE SET
             fetch_failed_at = EXCLUDED.fetch_failed_at,
             fetch_failures = provider_page.fetch_failures + 1",
        page_key.provider().as_str(),
        page_key.entity_type().as_str(),
        page_key.address().as_str(),
        observed_at,
    )
    .execute(&mut **tx)
    .await?;

    Ok(RecordOutcome { has_changed: false })
}

/// Stores a successful fetch, replacing the responses when their hash changed.
///
/// The fetch clears the page's missing state and failure count, and rejects an open
/// `dead_page` request.
///
/// # Errors
///
/// Returns a database error when a query fails.
async fn record_success(
    tx: &mut Transaction<'_, Postgres>,
    page_key: &PageKey,
    observed_at: DateTime<Utc>,
    responses_hash: &str,
    responses: Vec<PreparedResponse>,
    existing_page: Option<&ExistingPageRow>,
) -> Result<RecordOutcome, Error> {
    let responses_changed =
        existing_page.and_then(|row| row.responses_hash.as_deref()) != Some(responses_hash);
    let was_dead = existing_page.is_some_and(|row| row.is_dead);
    let has_changed = responses_changed || was_dead;

    sqlx::query!(
        "INSERT INTO provider_page (
             provider, entity_type, address, fetched_at, is_dead,
             missing_since, missing_last_at, missing_answers, responses_hash,
             fetch_failed_at, fetch_failures
         )
         VALUES ($1, $2, $3, $4, FALSE, NULL, NULL, 0, $5, NULL, 0)
         ON CONFLICT (provider, entity_type, address) DO UPDATE SET
             fetched_at = EXCLUDED.fetched_at,
             is_dead = FALSE,
             missing_since = NULL,
             missing_last_at = NULL,
             missing_answers = 0,
             responses_hash = EXCLUDED.responses_hash,
             fetch_failed_at = NULL,
             fetch_failures = 0",
        page_key.provider().as_str(),
        page_key.entity_type().as_str(),
        page_key.address().as_str(),
        observed_at,
        responses_hash,
    )
    .execute(&mut **tx)
    .await?;

    if existing_page.is_some_and(|row| row.missing_since.is_some()) {
        reject_open_dead_page_approval(tx, page_key).await?;
    }

    if responses_changed {
        sqlx::query!(
            "DELETE FROM provider_page_response
             WHERE provider = $1 AND entity_type = $2 AND address = $3",
            page_key.provider().as_str(),
            page_key.entity_type().as_str(),
            page_key.address().as_str(),
        )
        .execute(&mut **tx)
        .await?;

        let mut request_names = Vec::with_capacity(responses.len());
        let mut content_types = Vec::with_capacity(responses.len());
        let mut compressed_bodies = Vec::with_capacity(responses.len());

        for response in responses {
            request_names.push(response.request_name);
            content_types.push(response.content_type);
            compressed_bodies.push(response.compressed_body);
        }

        sqlx::query!(
            "INSERT INTO provider_page_response (
                 provider, entity_type, address, request_name, content_type, body
             )
             SELECT $1, $2, $3, * FROM UNNEST($4::text[], $5::text[], $6::bytea[])",
            page_key.provider().as_str(),
            page_key.entity_type().as_str(),
            page_key.address().as_str(),
            &request_names,
            &content_types,
            &compressed_bodies,
        )
        .execute(&mut **tx)
        .await?;
    }

    tracing::event!(
        name: "postgres_db.provider_page.recorded",
        Level::DEBUG,
        page.provider = %page_key.provider(),
        page.entity_type = page_key.entity_type().as_str(),
        page.address = %page_key.address(),
        page.has_changed = has_changed,
        "recorded provider page {{page.provider}}:{{page.entity_type}}:{{page.address}}",
    );

    Ok(RecordOutcome { has_changed })
}

#[cfg(test)]
mod tests {
    use echora_domain::Provider;

    use super::*;

    #[test]
    fn collect_named_page_details_keeps_longest_url_first_title_and_skips_empty_pages() {
        let page_source =
            |provider, address: &str, source: Option<&str>, title: Option<&str>| PageSource {
                page_key: PageKey::new(provider, EntityType::Anime, address).unwrap(),
                source: source.map(|url| Url::parse(url).unwrap()),
                title: title.map(str::to_owned),
            };
        let named_page_sources = [
            page_source(
                Provider::Mal,
                "2",
                Some("https://myanimelist.net/anime/2"),
                Some("Sequel"),
            ),
            page_source(
                Provider::Mal,
                "2",
                Some("https://myanimelist.net/anime/2/Sequel"),
                Some("Sequel (other list)"),
            ),
            page_source(Provider::AniList, "2", None, Some("Sequel")),
            page_source(Provider::Mal, "3", None, None),
        ];

        let details_by_page =
            collect_named_page_details(named_page_sources.iter().map(|page_source| {
                (
                    &page_source.page_key,
                    NamedPageDetails {
                        source: page_source.source.as_ref().map(Url::as_str),
                        title: page_source.title.as_deref(),
                    },
                )
            }));

        assert_eq!(
            details_by_page
                .iter()
                .map(|(page_key, page_details)| (
                    page_key.provider(),
                    page_key.address().as_str(),
                    page_details.source,
                    page_details.title,
                ))
                .collect::<Vec<_>>(),
            vec![
                (
                    Provider::Mal,
                    "2",
                    Some("https://myanimelist.net/anime/2/Sequel"),
                    Some("Sequel"),
                ),
                (Provider::AniList, "2", None, Some("Sequel")),
            ]
        );
    }

    #[test]
    fn prepare_outcome_success_invalid_responses_returns_invalid_argument() {
        let main_response = PageResponse {
            request_name: "main".to_owned(),
            content_type: "application/json".to_owned(),
            body: b"{}".to_vec(),
        };
        let unnamed_response = PageResponse {
            request_name: String::new(),
            ..main_response.clone()
        };
        let untyped_response = PageResponse {
            content_type: String::new(),
            ..main_response.clone()
        };

        for invalid_responses in [
            Vec::new(),
            vec![unnamed_response],
            vec![untyped_response],
            vec![main_response.clone(), main_response],
        ] {
            let prepare_error = (prepare_outcome(FetchOutcome::Success {
                responses: invalid_responses,
            }))
            .expect_err("invalid responses must be refused");
            assert!(prepare_error.is_invalid_argument());
        }
    }

    #[test]
    fn compute_responses_hash_is_order_independent() {
        let first = PageResponse {
            request_name: "main".to_owned(),
            content_type: "application/json".to_owned(),
            body: b"{}".to_vec(),
        };
        let second = PageResponse {
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
    fn encode_page_key_proto_matches_protobuf_wire_format() {
        let page_key = PageKey::new(Provider::Mal, EntityType::Anime, "21").unwrap();
        let encoded_proposal = encode_page_key_proto(&page_key);
        let decoded_proposal = ProtoPageKey::decode(encoded_proposal.as_slice());
        assert_eq!(
            decoded_proposal,
            Ok(ProtoPageKey {
                provider: "mal".to_owned(),
                entity_type: 1,
                address: "21".to_owned(),
            })
        );

        for (entity_type, proto_number, entity_type_name) in [
            (EntityType::Character, 2, "character"),
            (EntityType::Episode, 3, "episode"),
        ] {
            let page_key = PageKey::new(Provider::Kitsu, entity_type, "cowboy-bebop").unwrap();
            let decoded_page_key =
                ProtoPageKey::decode(encode_page_key_proto(&page_key).as_slice()).unwrap();
            assert_eq!(decoded_page_key.entity_type, proto_number);
            assert_eq!(entity_type.to_string(), entity_type_name);
            assert_eq!(page_key.address().as_ref(), "cowboy-bebop");
        }
    }

    #[test]
    fn is_stale_observation_checks_all_observation_timestamps() {
        let base_timestamp = Utc::now();
        let existing_page_row = ExistingPageRow {
            fetched_at: Some(base_timestamp),
            is_dead: false,
            missing_since: Some(base_timestamp + TimeDelta::hours(2)),
            missing_last_at: Some(base_timestamp + TimeDelta::hours(4)),
            missing_answers: 1,
            responses_hash: None,
            fetch_failed_at: Some(base_timestamp + TimeDelta::hours(1)),
        };

        assert!(is_stale_observation(
            Some(&existing_page_row),
            base_timestamp + TimeDelta::hours(1),
        ));
        assert!(is_stale_observation(
            Some(&existing_page_row),
            base_timestamp + TimeDelta::hours(4),
        ));
        assert!(!is_stale_observation(
            Some(&existing_page_row),
            base_timestamp + TimeDelta::hours(5),
        ));
        assert!(!is_stale_observation(None, base_timestamp));
    }
}
