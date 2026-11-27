//! Provider page storage, response compression, and dead-page tracking.

use std::collections::{BTreeMap, HashSet};
use std::fmt::{self, Write as _};
use std::io::Write as _;
use std::str::FromStr;

use chrono::{DateTime, TimeDelta, Utc};
use flate2::Compression;
use flate2::write::GzEncoder;
use prost::Message as _;
use sha2::{Digest as _, Sha256};
use sqlx::error::BoxDynError;
use sqlx::postgres::{PgTypeInfo, PgValueRef};
use sqlx::{Decode, PgPool, Postgres, Transaction, Type};
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

/// Supported external anime metadata providers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Provider {
    /// `MyAnimeList` (`"mal"`).
    Mal,
    /// `AniList` (`"anilist"`).
    AniList,
    /// `AniDB` (`"anidb"`).
    AniDb,
    /// aniSearch (`"anisearch"`).
    AniSearch,
    /// Anime News Network (`"ann"`).
    Ann,
    /// LiveChart.me (`"livechart"`).
    LiveChart,
    /// SIMKL (`"simkl"`).
    Simkl,
    /// `AnimeCountdown` (`"animecountdown"`).
    AnimeCountdown,
    /// Syoboi Calendar (`"syoboi"`).
    Syoboi,
    /// Kitsu (`"kitsu"`).
    Kitsu,
    /// Anime-Planet (`"anime_planet"`).
    AnimePlanet,
    /// AnimeSchedule.net (`"animeschedule"`).
    AnimeSchedule,
}

impl Provider {
    /// Every provider, in the order the `provider` lookup table lists them.
    pub const ALL: [Self; 12] = [
        Self::Mal,
        Self::AniList,
        Self::Kitsu,
        Self::AniDb,
        Self::AnimePlanet,
        Self::AniSearch,
        Self::AnimeSchedule,
        Self::Syoboi,
        Self::Ann,
        Self::LiveChart,
        Self::Simkl,
        Self::AnimeCountdown,
    ];

    /// Returns the canonical database key for this provider.
    ///
    /// # Examples
    ///
    /// ```
    /// # use postgres_db::Provider;
    /// assert_eq!(Provider::Mal.as_str(), "mal");
    /// assert_eq!(Provider::AnimePlanet.as_str(), "anime_planet");
    /// ```
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Mal => "mal",
            Self::AniList => "anilist",
            Self::AniDb => "anidb",
            Self::AniSearch => "anisearch",
            Self::Ann => "ann",
            Self::LiveChart => "livechart",
            Self::Simkl => "simkl",
            Self::AnimeCountdown => "animecountdown",
            Self::Syoboi => "syoboi",
            Self::Kitsu => "kitsu",
            Self::AnimePlanet => "anime_planet",
            Self::AnimeSchedule => "animeschedule",
        }
    }

    /// Validates that `address` matches this provider's address format for `entity_type`.
    ///
    /// A MAL episode page has no ID of its own, so its address is the work's ID and the
    /// episode number joined by a colon, as in `21:1`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use postgres_db::{EntityType, Provider};
    /// assert!(Provider::Mal.validate_address(EntityType::Anime, "21").is_ok());
    /// assert!(Provider::Mal.validate_address(EntityType::Episode, "21:1").is_ok());
    /// assert!(Provider::Kitsu.validate_address(EntityType::Anime, "cowboy-bebop").is_ok());
    /// assert!(Provider::Mal.validate_address(EntityType::Anime, "not-numeric").is_err());
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an invalid-argument error when `address` is empty or violates
    /// the provider's numeric or slug constraints.
    pub fn validate_address(
        self,
        entity_type: EntityType,
        address: impl AsRef<str>,
    ) -> Result<(), Error> {
        let address = address.as_ref();
        if address.is_empty() {
            return Err(Error::invalid_argument("page address must not be empty"));
        }

        match self {
            Self::Mal if entity_type == EntityType::Episode => {
                let is_work_and_episode_number =
                    address
                        .split_once(':')
                        .is_some_and(|(work_address, episode_number)| {
                            is_numeric_address(work_address) && is_numeric_address(episode_number)
                        });
                if is_work_and_episode_number {
                    Ok(())
                } else {
                    Err(Error::invalid_argument(format!(
                        "provider mal requires an episode address of the form \"<work id>:<episode number>\", got {address:?}"
                    )))
                }
            }
            Self::Mal
            | Self::AniList
            | Self::AniDb
            | Self::AniSearch
            | Self::Ann
            | Self::LiveChart
            | Self::Simkl
            | Self::AnimeCountdown
            | Self::Syoboi => {
                if is_numeric_address(address) {
                    Ok(())
                } else {
                    Err(Error::invalid_argument(format!(
                        "provider {} requires a numeric address, got {address:?}",
                        self.as_str()
                    )))
                }
            }
            Self::Kitsu | Self::AnimePlanet | Self::AnimeSchedule => {
                if address
                    .bytes()
                    .any(|byte| matches!(byte, b'/' | b'?' | b'#' | b' ' | b'\t' | b'\n' | b'\r'))
                {
                    Err(Error::invalid_argument(format!(
                        "provider {} address contains invalid path or whitespace characters: {address:?}",
                        self.as_str()
                    )))
                } else {
                    Ok(())
                }
            }
        }
    }
}

/// Returns whether `address` is a non-empty run of ASCII digits.
fn is_numeric_address(address: &str) -> bool {
    !address.is_empty() && address.bytes().all(|byte| byte.is_ascii_digit())
}

impl AsRef<str> for Provider {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl fmt::Display for Provider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for Provider {
    type Err = Error;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "mal" => Ok(Self::Mal),
            "anilist" => Ok(Self::AniList),
            "anidb" => Ok(Self::AniDb),
            "anisearch" => Ok(Self::AniSearch),
            "ann" => Ok(Self::Ann),
            "livechart" => Ok(Self::LiveChart),
            "simkl" => Ok(Self::Simkl),
            "animecountdown" => Ok(Self::AnimeCountdown),
            "syoboi" => Ok(Self::Syoboi),
            "kitsu" => Ok(Self::Kitsu),
            "anime_planet" => Ok(Self::AnimePlanet),
            "animeschedule" => Ok(Self::AnimeSchedule),
            _ => Err(Error::invalid_argument(format!(
                "unknown provider {value:?}"
            ))),
        }
    }
}

/// Validated provider-specific page identifier or slug.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Address(String);

impl Address {
    /// Creates a validated page address for a `provider` page describing `entity_type`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use postgres_db::{Address, EntityType, Provider};
    /// let address = Address::new(Provider::Mal, EntityType::Anime, "21")?;
    /// assert_eq!(address.as_str(), "21");
    /// # Ok::<(), postgres_db::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an invalid-argument error when `address` does not match
    /// `provider`'s identifier rules for `entity_type`.
    pub fn new(
        provider: Provider,
        entity_type: EntityType,
        address: impl Into<String>,
    ) -> Result<Self, Error> {
        let address = address.into();
        provider.validate_address(entity_type, &address)?;
        Ok(Self(address))
    }

    /// Returns the address as the provider writes it.
    ///
    /// # Examples
    ///
    /// ```
    /// use postgres_db::{Address, EntityType, Provider};
    ///
    /// let address =
    ///     Address::new(Provider::Mal, EntityType::Anime, "21").expect("MAL addresses are numeric");
    /// assert_eq!(address.as_str(), "21");
    /// ```
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl AsRef<str> for Address {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for Address {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Type of catalogue entity, which is also what a provider page describes.
///
/// Matches `EntityType` in the canonical model and `shared_proto.v1.EntityType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum EntityType {
    /// A work (anime).
    Anime,
    /// A character.
    Character,
    /// An episode of a work.
    Episode,
}

impl EntityType {
    /// Returns the lookup table key for this entity type.
    ///
    /// # Examples
    ///
    /// ```
    /// # use postgres_db::EntityType;
    /// assert_eq!(EntityType::Anime.as_str(), "anime");
    /// ```
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Anime => "anime",
            Self::Character => "character",
            Self::Episode => "episode",
        }
    }

    /// Returns the `shared_proto.v1.EntityType` enum number of this entity type.
    const fn as_proto_i32(self) -> i32 {
        match self {
            Self::Anime => 1,
            Self::Character => 2,
            Self::Episode => 3,
        }
    }
}

impl fmt::Display for EntityType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for EntityType {
    type Err = Error;

    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        match raw {
            "anime" => Ok(Self::Anime),
            "character" => Ok(Self::Character),
            "episode" => Ok(Self::Episode),
            other => Err(Error::invalid_argument(format!(
                "unknown entity type: {other}"
            ))),
        }
    }
}

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

/// Records the URL of each page an entity holds as a source, where it changed.
///
/// A page's URL depends only on the page, so it is stored once on `provider_page`. The entity
/// that holds a page as its source chooses its URL and replaces whatever is stored; a save
/// that only names the page fills it while it is empty, through
/// [`record_referenced_page_sources`]. A page given here that the entity does not hold as a
/// source is left unchanged. The owned page rows
/// are locked in key order before the update, so two saves of one entity cannot deadlock.
///
/// Call it before the entity's own row is locked: every writer locks pages first, as
/// `StorePage` does.
///
/// # Errors
///
/// Returns an invalid-argument error when a URL is blank, a page key is for another kind of
/// entity, or a page key is given twice, or a database error when the update fails.
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
        if page_source.source.is_none() {
            return Err(Error::invalid_argument(format!(
                "page source URL for {page_label} must be given"
            )));
        }
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
    let sources: Vec<&str> = page_sources
        .iter()
        .filter_map(|page_source| page_source.source.as_ref().map(Url::as_str))
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
        SET source = given.source,
            title = coalesce(given.title, page.title)
        FROM given
        JOIN owned USING (provider, entity_type, address)
        WHERE page.provider = given.provider
          AND page.entity_type = given.entity_type
          AND page.address = given.address
          AND (page.source IS DISTINCT FROM given.source
               OR page.title IS DISTINCT FROM coalesce(given.title, page.title))
        "#,
        entity_id,
        entity_type.as_str(),
        &providers,
        &addresses,
        &sources as &[&str],
        &titles as &[Option<&str>]
    )
    .execute(&mut **tx)
    .await?;
    Ok(())
}

/// The URL and title a save gives for one page it names.
#[derive(Debug, Default)]
struct NamedPageDetails<'a> {
    /// The longest URL given for the page.
    source: Option<&'a str>,
    /// The first title given for the page.
    title: Option<&'a str>,
}

/// Collects the URL and title given for each named page, in page key order.
///
/// Several entries can name one page through different spellings of its URL. The longest
/// keeps the provider's slug, the same choice the merges make for an entity's own pages, and
/// the first title given is kept. The sorted keys give every writer one row order. A page
/// given with neither is left out.
fn collect_named_page_details<'a>(
    named_page_sources: impl IntoIterator<Item = &'a PageSource>,
) -> BTreeMap<&'a PageKey, NamedPageDetails<'a>> {
    let mut details_by_page: BTreeMap<&'a PageKey, NamedPageDetails<'a>> = BTreeMap::new();
    for page_source in named_page_sources {
        let source = page_source.source.as_ref().map(Url::as_str);
        let title = page_source.title.as_deref();
        if source.is_none() && title.is_none() {
            continue;
        }
        let page_details = details_by_page.entry(&page_source.page_key).or_default();
        if source.map(str::len) > page_details.source.map(str::len) {
            page_details.source = source;
        }
        page_details.title = page_details.title.or(title);
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
    let details_by_page = collect_named_page_details(named_page_sources);
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

/// Reads a [`Provider`] from the `text` slug its `provider` lookup row holds.
impl Type<Postgres> for Provider {
    fn type_info() -> PgTypeInfo {
        <&str as Type<Postgres>>::type_info()
    }

    fn compatible(column_type: &PgTypeInfo) -> bool {
        <&str as Type<Postgres>>::compatible(column_type)
    }
}

/// Reads a stored provider slug through [`Provider::from_str`].
impl<'row> Decode<'row, Postgres> for Provider {
    fn decode(value: PgValueRef<'row>) -> Result<Self, BoxDynError> {
        Ok(<&str as Decode<Postgres>>::decode(value)?.parse()?)
    }
}

/// Reads an [`EntityType`] from the `text` value its `entity_type` lookup row holds.
impl Type<Postgres> for EntityType {
    fn type_info() -> PgTypeInfo {
        <&str as Type<Postgres>>::type_info()
    }

    fn compatible(column_type: &PgTypeInfo) -> bool {
        <&str as Type<Postgres>>::compatible(column_type)
    }
}

/// Reads a stored entity type through [`EntityType::from_str`].
impl<'row> Decode<'row, Postgres> for EntityType {
    fn decode(value: PgValueRef<'row>) -> Result<Self, BoxDynError> {
        Ok(<&str as Decode<Postgres>>::decode(value)?.parse()?)
    }
}

#[derive(PartialEq, prost::Message)]
struct ProtoPageKey {
    #[prost(string, tag = "1")]
    provider: String,
    #[prost(int32, tag = "2")]
    entity_type: i32,
    #[prost(string, tag = "3")]
    address: String,
}

/// Encodes `page_key` as a `shared_proto.v1.PageKey` protobuf payload.
///
/// # Examples
///
/// ```
/// # use postgres_db::{encode_page_key_proto, PageKey, EntityType, Provider};
/// let key = PageKey::new(Provider::Mal, EntityType::Anime, "21")?;
/// let bytes = encode_page_key_proto(&key);
/// // provider "mal" (field 1), entity type ANIME = 1 (field 2), address "21" (field 3)
/// assert_eq!(bytes, b"\x0a\x03mal\x10\x01\x1a\x0221");
/// # Ok::<(), postgres_db::Error>(())
/// ```
#[must_use]
pub fn encode_page_key_proto(page_key: &PageKey) -> Vec<u8> {
    ProtoPageKey {
        provider: page_key.provider.as_str().to_owned(),
        entity_type: page_key.entity_type.as_proto_i32(),
        address: page_key.address.as_str().to_owned(),
    }
    .encode_to_vec()
}

/// Identifies a provider page by provider, entity type, and address.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PageKey {
    provider: Provider,
    entity_type: EntityType,
    address: Address,
}

impl PageKey {
    /// Creates a validated page key for `provider`, `entity_type`, and `address`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use postgres_db::{PageKey, EntityType, Provider};
    /// let key = PageKey::new(Provider::Kitsu, EntityType::Anime, "cowboy-bebop")?;
    /// assert_eq!(key.provider(), Provider::Kitsu);
    /// assert_eq!(key.address().as_str(), "cowboy-bebop");
    /// # Ok::<(), postgres_db::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an invalid-argument error when `address` fails `provider` validation.
    pub fn new(
        provider: Provider,
        entity_type: EntityType,
        address: impl Into<String>,
    ) -> Result<Self, Error> {
        let address = Address::new(provider, entity_type, address)?;
        Ok(Self {
            provider,
            entity_type,
            address,
        })
    }

    /// Returns the provider for this page key.
    #[must_use]
    pub const fn provider(&self) -> Provider {
        self.provider
    }

    /// Returns the entity type this page key's page describes.
    #[must_use]
    pub const fn entity_type(&self) -> EntityType {
        self.entity_type
    }

    /// Returns the validated address for this page key.
    #[must_use]
    pub const fn address(&self) -> &Address {
        &self.address
    }
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
/// # use postgres_db::{
/// #     record_page_observation, FetchOutcome, PageKey, EntityType, PageObservation, PageResponse,
/// #     Provider,
/// # };
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
    let provider = page_key.provider.as_str();
    let entity_type = page_key.entity_type.as_str();
    let address = page_key.address.as_str();
    match page_key.entity_type {
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
        page_key.provider.as_str(),
        page_key.entity_type.as_str(),
        page_key.address.as_str(),
    )
    .execute(&mut **tx)
    .await?;

    let row = sqlx::query_as!(
        ExistingPageRow,
        "SELECT fetched_at, is_dead, missing_since, missing_last_at, missing_answers, responses_hash, fetch_failed_at
         FROM provider_page
         WHERE provider = $1 AND entity_type = $2 AND address = $3
         FOR UPDATE",
        page_key.provider.as_str(),
        page_key.entity_type.as_str(),
        page_key.address.as_str(),
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
        page_key.provider.as_str(),
        page_key.entity_type.as_str(),
        page_key.address.as_str(),
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
        page_key.provider.as_str(),
        page_key.entity_type.as_str(),
        page_key.address.as_str(),
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
        page_key.provider.as_str(),
        page_key.entity_type.as_str(),
        page_key.address.as_str(),
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
            page_key.provider.as_str(),
            page_key.entity_type.as_str(),
            page_key.address.as_str(),
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
            page_key.provider.as_str(),
            page_key.entity_type.as_str(),
            page_key.address.as_str(),
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
        page.provider = %page_key.provider,
        page.entity_type = page_key.entity_type.as_str(),
        page.address = %page_key.address,
        page.has_changed = has_changed,
        "recorded provider page {{page.provider}}:{{page.entity_type}}:{{page.address}}",
    );

    Ok(RecordOutcome { has_changed })
}

#[cfg(test)]
mod tests {
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

        let details_by_page = collect_named_page_details(&named_page_sources);

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
    fn provider_validates_numeric_and_slug_address() {
        assert!(
            Provider::Mal
                .validate_address(EntityType::Anime, "21")
                .is_ok()
        );
        assert!(
            Provider::AniDb
                .validate_address(EntityType::Episode, "12345")
                .is_ok()
        );
        assert!(
            Provider::Mal
                .validate_address(EntityType::Anime, "21extra")
                .is_err()
        );
        assert!(
            Provider::Mal
                .validate_address(EntityType::Anime, "21:1")
                .is_err()
        );
        assert!(
            Provider::Mal
                .validate_address(EntityType::Episode, "21:1")
                .is_ok()
        );
        assert!(
            Provider::Mal
                .validate_address(EntityType::Episode, "21")
                .is_err()
        );
        assert!(
            Provider::Mal
                .validate_address(EntityType::Episode, "21:")
                .is_err()
        );
        assert!(
            Provider::Mal
                .validate_address(EntityType::Episode, ":1")
                .is_err()
        );
        assert!(
            Provider::Mal
                .validate_address(EntityType::Episode, "21:1:2")
                .is_err()
        );
        assert!(
            Provider::Kitsu
                .validate_address(EntityType::Anime, "cowboy-bebop")
                .is_ok()
        );
        assert!(
            Provider::Kitsu
                .validate_address(EntityType::Anime, "bad/slug")
                .is_err()
        );
        assert!(
            Provider::Kitsu
                .validate_address(EntityType::Anime, "")
                .is_err()
        );
        assert!("unknown_provider".parse::<Provider>().is_err());
    }

    #[test]
    fn provider_from_str_round_trips_every_lookup_value() {
        for provider_name in [
            "mal",
            "anilist",
            "anidb",
            "anisearch",
            "ann",
            "livechart",
            "simkl",
            "animecountdown",
            "syoboi",
            "kitsu",
            "anime_planet",
            "animeschedule",
        ] {
            let provider: Provider = provider_name.parse().unwrap();
            assert_eq!(provider.as_str(), provider_name);
            assert_eq!(provider.to_string(), provider_name);
            assert_eq!(provider.as_ref(), provider_name);
        }
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
    fn page_key_new_invalid_address_returns_invalid_argument() {
        assert!(PageKey::new(Provider::Mal, EntityType::Anime, "21").is_ok());
        assert!(PageKey::new(Provider::Mal, EntityType::Anime, "invalid").is_err());
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
