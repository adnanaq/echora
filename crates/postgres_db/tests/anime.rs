//! Integration tests for work loads against PostgreSQL.

mod support;

use echora_domain::{EntityType, PageKey, Provider};
use postgres_db::{
    EpisodeListEntryInput, EpisodeReferenceInput, PageSource, delete_episode, load_anime,
    save_anime,
};
use sqlx::PgPool;
use uuid::Uuid;

use crate::support::{build_anime_save, insert_unmerged_anime};

#[sqlx::test(migrations = "./migrations")]
async fn load_anime_returns_undeleted_known_episodes(pool: PgPool) {
    let anime_id = insert_unmerged_anime(&pool).await;
    let listed_episode = EpisodeReferenceInput {
        episode_id: Uuid::now_v7(),
        episode_number: 1,
        title: "Placeholder episode 1".to_owned(),
        page_sources: vec![PageSource {
            page_key: PageKey::new(Provider::Mal, EntityType::Episode, "21:1").expect("valid key"),
            source: None,
            title: None,
        }],
        list_entries: vec![EpisodeListEntryInput {
            provider: Provider::Mal,
            title: Some("Placeholder episode 1".to_owned()),
            aired_at: None,
            duration_seconds: Some(1440),
            is_filler: Some(false),
            is_recap: Some(false),
        }],
    };
    let removed_episode = EpisodeReferenceInput {
        episode_id: Uuid::now_v7(),
        episode_number: 2,
        title: "Placeholder episode 2".to_owned(),
        page_sources: Vec::new(),
        list_entries: Vec::new(),
    };
    save_anime(
        &pool,
        &build_anime_save(
            anime_id,
            1,
            "work-v1",
            vec![listed_episode.clone(), removed_episode.clone()],
        ),
    )
    .await
    .expect("work save creates both episodes");
    delete_episode(&pool, removed_episode.episode_id, 1, None)
        .await
        .expect("delete episode 2");

    let loaded_anime = load_anime(&pool, anime_id).await.expect("load the work");

    assert_eq!(loaded_anime.known_episodes, vec![listed_episode]);
}

#[sqlx::test(migrations = "./migrations")]
async fn load_anime_never_fetched_source_page_returns_no_source_pages(pool: PgPool) {
    let anime_id = insert_unmerged_anime(&pool).await;
    sqlx::query(
        "INSERT INTO provider_page (provider, entity_type, address) VALUES ('mal', 'anime', '21')",
    )
    .execute(&pool)
    .await
    .expect("insert never-fetched page");
    sqlx::query(
        "INSERT INTO anime_source (anime_id, provider, entity_type, address)
         VALUES ($1, 'mal', 'anime', '21')",
    )
    .bind(anime_id)
    .execute(&pool)
    .await
    .expect("link the page");

    let loaded_anime = load_anime(&pool, anime_id).await.expect("load the work");

    assert_eq!(loaded_anime.source_pages, Vec::new());
}

#[sqlx::test(migrations = "./migrations")]
async fn load_anime_stored_invalid_episode_address_returns_invalid_argument(pool: PgPool) {
    let anime_id = Uuid::now_v7();
    let episode_id = Uuid::now_v7();
    sqlx::query("INSERT INTO anime (id) VALUES ($1)")
        .bind(anime_id)
        .execute(&pool)
        .await
        .expect("insert placeholder work");
    sqlx::query("INSERT INTO episode (id, anime_id, episode_number) VALUES ($1, $2, 1)")
        .bind(episode_id)
        .bind(anime_id)
        .execute(&pool)
        .await
        .expect("insert placeholder episode");
    sqlx::query(
        "INSERT INTO provider_page (provider, entity_type, address) VALUES ('mal', 'episode', 'not-numeric')",
    )
    .execute(&pool)
    .await
    .expect("insert page with an invalid address");
    sqlx::query(
        "INSERT INTO episode_source (episode_id, provider, entity_type, address)
         VALUES ($1, 'mal', 'episode', 'not-numeric')",
    )
    .bind(episode_id)
    .execute(&pool)
    .await
    .expect("link the page");

    let load_error = load_anime(&pool, anime_id)
        .await
        .expect_err("an invalid stored address must be refused");

    assert!(load_error.is_invalid_argument());
}

#[sqlx::test(migrations = "./migrations")]
async fn save_anime_timestamps_share_transaction_time(pool: PgPool) {
    let anime_id = insert_unmerged_anime(&pool).await;
    let new_episode = EpisodeReferenceInput {
        episode_id: Uuid::now_v7(),
        episode_number: 1,
        title: "Placeholder episode 1".to_owned(),
        page_sources: Vec::new(),
        list_entries: Vec::new(),
    };
    save_anime(
        &pool,
        &build_anime_save(anime_id, 1, "work-v1", vec![new_episode.clone()]),
    )
    .await
    .expect("the work saves");

    let (anime_updated_at, anime_merged_at, episode_created_at, event_created_at): (
        chrono::DateTime<chrono::Utc>,
        Option<chrono::DateTime<chrono::Utc>>,
        chrono::DateTime<chrono::Utc>,
        chrono::DateTime<chrono::Utc>,
    ) = sqlx::query_as(
        "SELECT a.updated_at, a.merged_at, e.created_at, o.created_at
         FROM anime a
         JOIN episode e ON e.id = $2
         JOIN outbox o ON o.entity_id = a.id AND o.subject = 'anime.created'
         WHERE a.id = $1",
    )
    .bind(anime_id)
    .bind(new_episode.episode_id)
    .fetch_one(&pool)
    .await
    .expect("read the save's timestamps");

    assert_eq!(anime_merged_at, Some(anime_updated_at));
    assert_eq!(episode_created_at, anime_updated_at);
    assert_eq!(event_created_at, anime_updated_at);
}

#[sqlx::test(migrations = "./migrations")]
async fn save_anime_unchanged_hash_from_newer_merge_records_merge_version(pool: PgPool) {
    let anime_id = insert_unmerged_anime(&pool).await;
    save_anime(&pool, &build_anime_save(anime_id, 1, "work-v1", Vec::new()))
        .await
        .expect("the first merge saves");
    let mut newer_merge_save = build_anime_save(anime_id, 2, "work-v1", Vec::new());
    newer_merge_save.merge_version = 2;

    let save_outcome = save_anime(&pool, &newer_merge_save)
        .await
        .expect("the unchanged save succeeds");
    let (stored_version, stored_merge_version, event_count): (i64, Option<i32>, i64) =
        sqlx::query_as(
            "SELECT a.version, a.merge_version,
                    (SELECT COUNT(*) FROM outbox o WHERE o.entity_id = a.id)
             FROM anime a WHERE a.id = $1",
        )
        .bind(anime_id)
        .fetch_one(&pool)
        .await
        .expect("read the work");

    assert!(!save_outcome.has_changed);
    assert_eq!(stored_version, 2);
    assert_eq!(stored_merge_version, Some(2));
    assert_eq!(event_count, 1);
}
