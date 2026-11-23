//! Builders and log capture shared by the integration tests.
//!
//! Each test binary compiles this module on its own and uses only part of it.

#![allow(dead_code, reason = "each test binary uses only some of these helpers")]
#![allow(
    clippy::expect_used,
    reason = "a failed setup step stops the test, as a test's own expect does"
)]

use std::io;
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use postgres_db::{
    ANIME_HASH_FORMAT, AnimeDetailsInput, AnimeRelationsAndGraphInput, AnimeScheduleAndStatsInput,
    EpisodeReferenceInput, SaveAnimeInput,
};
use sqlx::PgPool;
use uuid::Uuid;

/// Inserts a work that exists only as an ID, as the seed load creates it.
pub async fn insert_unmerged_anime(pool: &PgPool) -> Uuid {
    let anime_id = Uuid::now_v7();
    sqlx::query("INSERT INTO anime (id) VALUES ($1)")
        .bind(anime_id)
        .execute(pool)
        .await
        .expect("insert placeholder work");
    anime_id
}

/// Inserts a character that exists only as an ID and a name, as the character load creates it.
pub async fn insert_unmerged_character(pool: &PgPool, name: &str) -> Uuid {
    let character_id = Uuid::now_v7();
    sqlx::query("INSERT INTO character (id, name) VALUES ($1, $2)")
        .bind(character_id)
        .bind(name)
        .execute(pool)
        .await
        .expect("insert placeholder character");
    character_id
}

/// Links a character page to a character, storing the page first when it is new.
pub async fn link_character_page(pool: &PgPool, character_id: Uuid, provider: &str, address: &str) {
    sqlx::query(
        "INSERT INTO provider_page (provider, entity_type, address)
         VALUES ($1, 'character', $2)
         ON CONFLICT DO NOTHING",
    )
    .bind(provider)
    .bind(address)
    .execute(pool)
    .await
    .expect("store the character page");
    sqlx::query(
        "INSERT INTO character_source (character_id, provider, entity_type, address)
         VALUES ($1, $2, 'character', $3)",
    )
    .bind(character_id)
    .bind(provider)
    .bind(address)
    .execute(pool)
    .await
    .expect("link the character page");
}

/// Builds a valid work save with no detail rows, for tests to fill in what they check.
pub fn build_anime_save(
    anime_id: Uuid,
    expected_version: i64,
    merged_record_hash: &str,
    episodes: Vec<EpisodeReferenceInput>,
) -> SaveAnimeInput {
    SaveAnimeInput {
        anime_id,
        expected_version,
        title: "Placeholder Work".to_owned(),
        title_english: None,
        title_japanese: None,
        anime_type: "TV".to_owned(),
        status: "FINISHED".to_owned(),
        rating: "PG-13 - Teens 13 or older".to_owned(),
        season: None,
        source_material: None,
        year: Some(2020),
        month: Some(1),
        episode_count: 12,
        duration_seconds: Some(1440),
        is_nsfw: Some(false),
        country_of_origin: Some("JP".to_owned()),
        synopsis: None,
        background: None,
        aired_from: None,
        aired_to: None,
        score_weighted: None,
        score_mean: None,
        score_median: None,
        merged_record_hash: merged_record_hash.to_owned(),
        hash_format: ANIME_HASH_FORMAT,
        merge_version: 1,
        traceparent: None,
        details: AnimeDetailsInput::default(),
        schedule_and_stats: AnimeScheduleAndStatsInput::default(),
        relations_and_graph: AnimeRelationsAndGraphInput::default(),
        episodes,
    }
}

/// Collects every log line the test binary writes, from every thread.
///
/// The subscriber is global, since a thread-scoped one misses events when another test
/// thread caches a call site as unwatched. Tests running in parallel share the lines, so
/// each test checks only lines naming its own entity ID or page address.
#[derive(Debug, Clone, Default)]
pub struct CapturedLogs {
    lines: Arc<Mutex<Vec<u8>>>,
}

impl CapturedLogs {
    /// Returns the binary's log capture, installing the global subscriber on first use.
    pub fn install_global() -> &'static Self {
        static GLOBAL_LOGS: OnceLock<CapturedLogs> = OnceLock::new();
        GLOBAL_LOGS.get_or_init(|| {
            let captured_logs = Self::default();
            let writer_logs = captured_logs.clone();
            let subscriber = tracing_subscriber::fmt()
                .with_max_level(tracing::Level::TRACE)
                .with_ansi(false)
                .with_writer(move || writer_logs.clone())
                .finish();
            tracing::subscriber::set_global_default(subscriber)
                .unwrap_or_else(|error| panic!("only this module sets the subscriber: {error}"));
            captured_logs
        })
    }

    /// Returns every captured line that contains `marker`.
    pub fn read_lines_with(&self, marker: &str) -> String {
        let lines = self.lines.lock().unwrap_or_else(PoisonError::into_inner);
        String::from_utf8_lossy(&lines)
            .lines()
            .filter(|line| line.contains(marker))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

impl io::Write for CapturedLogs {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.lines
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
