//! PostgreSQL connection pool and `sqlx` schema migrations for Echora.

mod anime;
mod anime_details;
mod anime_relations;
mod anime_schedule_and_stats;
mod episode;
mod error;
mod graph_projection;
mod language_code;
mod provider_page;

#[doc(inline)]
pub use anime::{
    ANIME_HASH_FORMAT, FieldCorrectionRecord, LoadAnimeOutcome, LoadedSourcePage, SaveAnimeInput,
    SaveAnimeOutcome, load_anime, save_anime,
};
#[doc(inline)]
pub use anime_details::{AnimeDetailsInput, AnimeImageInput, AnimeThemeInput, AnimeTitleInput};
#[doc(inline)]
pub use anime_relations::{
    AnimeCompanyInput, AnimeRelationClaimInput, AnimeRelationInput, AnimeRelationsAndGraphInput,
    AnimeSourceMaterialInput, CompanyXrefInput, SourceMaterialXrefInput, link_anime_source_page,
};
#[doc(inline)]
pub use anime_schedule_and_stats::{
    AnimeBroadcastInput, AnimeContextualRankInput, AnimeExternalSourceInput, AnimeHiatusInput,
    AnimeScheduleAndStatsInput, AnimeStatisticsInput, AnimeStreamingSourceInput,
    AnimeThemeSongInput, AnimeTrailerInput, EpisodeRangeInput,
};
use chrono::{DateTime, Utc};
#[doc(inline)]
pub use episode::{
    EPISODE_HASH_FORMAT, EpisodeCharacterInput, EpisodeListEntryInput, EpisodeReferenceInput,
    EpisodeStreamingInput, EpisodeTitleInput, LoadEpisodeOutcome, MarkEpisodeAiredOutcome,
    SaveEpisodeInput, SaveEpisodeOutcome, delete_episode, load_episode, mark_episode_aired,
    save_episode,
};
#[doc(inline)]
pub use error::Error;
#[doc(inline)]
pub use provider_page::{
    Address, EntityType, FetchOutcome, PageKey, PageObservation, PageResponse, Provider,
    compute_responses_hash, encode_page_key_proto, record_page_observation,
};
use sqlx::migrate::Migrator;
use sqlx::postgres::PgPoolOptions;
use sqlx::{PgPool, Postgres, Transaction};
use tracing::Level;

/// Embeds the `migrations/` folder at build time, so the binary carries its own schema.
static MIGRATOR: Migrator = sqlx::migrate!("./migrations");

/// Connects a PostgreSQL connection pool using `database_url` and `max_connections`.
///
/// # Examples
///
/// ```no_run
/// # async fn example() -> Result<(), postgres_db::Error> {
/// let pool = postgres_db::connect_pool("postgres://localhost/echora", 5).await?;
/// assert_eq!(pool.options().get_max_connections(), 5);
/// postgres_db::run_migrations(&pool).await?;
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns an error when `database_url` is invalid or the database cannot be
/// reached.
pub async fn connect_pool(database_url: &str, max_connections: u32) -> Result<PgPool, Error> {
    let pool = PgPoolOptions::new()
        .max_connections(max_connections)
        .connect(database_url)
        .await?;
    Ok(pool)
}

/// Applies all embedded `sqlx` migrations to `pool`.
///
/// Migrations already applied are skipped, so this is safe to run at every start.
///
/// # Examples
///
/// ```no_run
/// # async fn example(pool: &sqlx::PgPool) -> Result<(), postgres_db::Error> {
/// postgres_db::run_migrations(pool).await?;
/// let has_schema: bool =
///     sqlx::query_scalar("SELECT to_regclass('anime_source') IS NOT NULL")
///         .fetch_one(pool)
///         .await?;
/// assert!(has_schema, "every table exists once the migrations ran");
///
/// // A second run finds every migration applied and changes nothing.
/// postgres_db::run_migrations(pool).await?;
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns an error when applying any migration fails or recorded checksums
/// diverge from the database history.
pub async fn run_migrations(pool: &PgPool) -> Result<(), Error> {
    MIGRATOR.run(pool).await?;
    tracing::event!(
        name: "postgres_db.migrations.applied",
        Level::INFO,
        "applied embedded PostgreSQL migrations",
    );
    Ok(())
}

/// Reads the start time of `tx`, which PostgreSQL's `now()` returns throughout it.
///
/// Every timestamp one write sets comes from this clock, so the rows and events of
/// one save agree with each other and with `now()` defaults, whichever replica wrote them.
pub(crate) async fn read_transaction_time(
    tx: &mut Transaction<'_, Postgres>,
) -> Result<DateTime<Utc>, Error> {
    let transaction_time = sqlx::query_scalar!(r#"SELECT now() AS "transaction_time!""#)
        .fetch_one(&mut **tx)
        .await?;
    Ok(transaction_time)
}
