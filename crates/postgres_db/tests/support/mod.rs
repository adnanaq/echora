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

use std::collections::BTreeMap;

use postgres_db::{
    ANIME_HASH_FORMAT, AnimeDetailsInput, AnimeRelationsAndGraphInput, AnimeScheduleAndStatsInput,
    EPISODE_HASH_FORMAT, EntityType, EpisodeReferenceInput, PageKey, PageSource, Provider,
    SaveAnimeInput, SaveCharacterInput, SaveEpisodeInput,
};
use sqlx::PgPool;
use url::Url;
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

/// Links MAL work pages, created when missing, to a work as its source pages.
pub async fn link_mal_anime_pages(pool: &PgPool, anime_id: Uuid, source_addresses: &[&str]) {
    for address in source_addresses {
        sqlx::query(
            "INSERT INTO provider_page (provider, entity_type, address)
             VALUES ('mal', 'anime', $1) ON CONFLICT DO NOTHING",
        )
        .bind(address)
        .execute(pool)
        .await
        .expect("insert page");
        sqlx::query(
            "INSERT INTO anime_source (anime_id, provider, entity_type, address)
             VALUES ($1, 'mal', 'anime', $2)",
        )
        .bind(anime_id)
        .bind(address)
        .execute(pool)
        .await
        .expect("link source page");
    }
}

/// Builds the URL a merge chose for one MAL page.
pub fn build_mal_page_source(entity_type: EntityType, address: &str, source: &str) -> PageSource {
    PageSource {
        page_key: PageKey::new(Provider::Mal, entity_type, address).expect("valid page key"),
        source: Some(Url::parse(source).expect("valid URL")),
        title: None,
    }
}

/// Builds a valid episode save with no detail rows, for tests to fill in what they check.
pub fn build_episode_save(episode_id: Uuid, episode_number: i32, title: &str) -> SaveEpisodeInput {
    SaveEpisodeInput {
        episode_id,
        expected_version: 1,
        episode_number,
        season_number: None,
        title: title.to_owned(),
        title_japanese: None,
        title_romaji: None,
        synopsis: None,
        description: None,
        aired_at: None,
        duration_seconds: None,
        is_filler: false,
        is_recap: false,
        score: None,
        titles: Vec::new(),
        images: Vec::new(),
        streaming: Vec::new(),
        characters: Vec::new(),
        merged_record_hash: format!("{title}-v1"),
        hash_format: EPISODE_HASH_FORMAT,
        merge_version: 1,
        traceparent: None,
        page_sources: Vec::new(),
    }
}

/// Builds a valid character save with no detail rows, for tests to fill in what they check.
pub fn build_character_save(character_id: Uuid, name: &str) -> SaveCharacterInput {
    SaveCharacterInput {
        character_id,
        expected_version: 1,
        name: name.to_owned(),
        name_native: None,
        description: None,
        favorites: None,
        name_variations: Vec::new(),
        nicknames: Vec::new(),
        traits: Vec::new(),
        images: Vec::new(),
        attributes: BTreeMap::new(),
        spoilers: BTreeMap::new(),
        animeography: Vec::new(),
        mangaography: Vec::new(),
        merged_record_hash: format!("{name}-v1"),
        hash_format: 1,
        merge_version: 1,
        traceparent: None,
        page_sources: Vec::new(),
    }
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
        page_sources: Vec::new(),
        partial_page_sources: Vec::new(),
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
