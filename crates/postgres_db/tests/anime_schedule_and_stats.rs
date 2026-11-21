//! Integration tests for work songs, streaming and statistics against PostgreSQL.

mod support;

use postgres_db::{
    AnimeScheduleAndStatsInput, AnimeStatisticsInput, AnimeStreamingInput, AnimeThemeSongInput,
    save_anime,
};
use sqlx::PgPool;

use crate::support::{build_work_save, insert_unmerged_work};

async fn count_rows(pool: &PgPool, table: &str) -> i64 {
    sqlx::query_scalar(&format!("SELECT COUNT(*) FROM {table}"))
        .fetch_one(pool)
        .await
        .unwrap_or(-1)
}

#[sqlx::test(migrations = "./migrations")]
async fn save_anime_entries_without_child_lists_store_no_child_rows(pool: PgPool) {
    let anime_id = insert_unmerged_work(&pool).await;
    let mut work_save = build_work_save(anime_id, 1, "work-v1", Vec::new());
    work_save.schedule_and_stats = AnimeScheduleAndStatsInput {
        theme_songs: vec![AnimeThemeSongInput {
            kind: "opening".to_owned(),
            position: 0,
            title: "Opening".to_owned(),
            artist: None,
            episode_ranges: Vec::new(),
        }],
        streaming_entries: vec![AnimeStreamingInput {
            platform: "Crunchyroll".to_owned(),
            url: "https://example.com/watch".to_owned(),
            region: None,
            is_free: None,
            is_premium_required: None,
            has_dub: None,
            subtitle_languages: Vec::new(),
        }],
        statistics: vec![AnimeStatisticsInput {
            provider: "mal".to_owned(),
            score: Some(8.5),
            scored_by: None,
            members: None,
            favorites: None,
            rank: None,
            popularity: None,
            contextual_ranks: Vec::new(),
        }],
        ..AnimeScheduleAndStatsInput::default()
    };

    save_anime(&pool, &work_save).await.expect("the work saves");

    for (table, expected_rows) in [
        ("anime_theme_song", 1),
        ("theme_song_episode_range", 0),
        ("anime_streaming", 1),
        ("anime_streaming_subtitle_language", 0),
        ("anime_statistics", 1),
        ("anime_contextual_rank", 0),
    ] {
        assert_eq!(count_rows(&pool, table).await, expected_rows, "{table}");
    }
}
