//! PostgreSQL connection pool and `sqlx` schema migrations for Echora.

// Rust guideline compliant 2026-10-09

mod anime;
mod anime_details;
mod error;
mod provider_page;

#[doc(inline)]
pub use anime::{
    ANIME_HASH_FORMAT, FieldCorrectionRecord, LoadAnimeOutcome, LoadedSourcePage, SaveAnimeInput,
    SaveAnimeOutcome, load_anime, save_anime,
};
#[doc(inline)]
pub use anime_details::{AnimeDetailsInput, AnimeImageInput, AnimeThemeInput, AnimeTitleInput};
#[doc(inline)]
pub use error::Error;
#[doc(inline)]
pub use provider_page::{
    PageKeyInput, PageKind, PageResponseInput, StorePageInput, StorePageOutcome,
    compute_responses_hash, encode_page_key_proposal, store_page, validate_page_address,
};
pub use sqlx::PgPool;
use sqlx::migrate::Migrator;
use sqlx::postgres::PgPoolOptions;
use tracing::Level;

static MIGRATOR: Migrator = sqlx::migrate!("./migrations");

/// Connects a PostgreSQL connection pool using `database_url` and `max_connections`.
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
