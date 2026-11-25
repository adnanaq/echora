//! Integration tests for scheduling, due queries, page listing, and refresh recording.

#![allow(
    clippy::expect_used,
    reason = "a failed setup step stops the test, as a test's own expect does"
)]

mod support;

use chrono::{Duration, Utc};
use postgres_db::{
    DueEntitiesQuery, EntityType, PageKey, Provider, RefreshOutcome, list_due_anime,
    list_due_characters, list_due_episodes, list_due_pages, list_entity_pages,
    list_entity_versions, list_requested_merges, list_score_anime, list_upcoming_episodes,
    record_refresh, save_anime,
};
use sqlx::PgPool;
use uuid::Uuid;

use crate::support::{
    build_anime_save, insert_unmerged_anime, insert_unmerged_character, link_character_page,
};

#[allow(
    clippy::too_many_arguments,
    reason = "test fixture inserts all provider_page columns"
)]
async fn insert_page(
    pool: &PgPool,
    provider: &str,
    entity_type: &str,
    address: &str,
    fetched_at: Option<chrono::DateTime<Utc>>,
    is_dead: bool,
    fetch_failed_at: Option<chrono::DateTime<Utc>>,
    fetch_failures: i32,
) {
    sqlx::query(
        "INSERT INTO provider_page (provider, entity_type, address, fetched_at, responses_hash, is_dead, fetch_failed_at, fetch_failures)
         VALUES ($1, $2, $3, $4, CASE WHEN $4 IS NOT NULL THEN 'hash' ELSE NULL END, $5, $6, $7)
         ON CONFLICT (provider, entity_type, address) DO UPDATE
         SET fetched_at = EXCLUDED.fetched_at,
             responses_hash = EXCLUDED.responses_hash,
             is_dead = EXCLUDED.is_dead,
             fetch_failed_at = EXCLUDED.fetch_failed_at,
             fetch_failures = EXCLUDED.fetch_failures",
    )
    .bind(provider)
    .bind(entity_type)
    .bind(address)
    .bind(fetched_at)
    .bind(is_dead)
    .bind(fetch_failed_at)
    .bind(fetch_failures)
    .execute(pool)
    .await
    .expect("insert provider_page");
}

#[sqlx::test(migrations = "./migrations")]
async fn list_entity_pages_and_record_refresh_cover_all_entities_and_outcomes(pool: PgPool) {
    let now = Utc::now();
    let anime_id = insert_unmerged_anime(&pool).await;
    insert_page(
        &pool,
        "mal",
        "anime",
        "100",
        Some(now - Duration::days(2)),
        false,
        None,
        0,
    )
    .await;
    insert_page(
        &pool,
        "anidb",
        "anime",
        "200",
        None,
        false,
        Some(now - Duration::minutes(10)),
        2,
    )
    .await;
    insert_page(
        &pool,
        "anilist",
        "anime",
        "300",
        None,
        true,
        Some(now - Duration::days(3)),
        3,
    )
    .await;
    insert_page(
        &pool,
        "kitsu",
        "anime",
        "400",
        None,
        false,
        Some(now - Duration::days(1)),
        1,
    )
    .await;

    sqlx::query(
        "INSERT INTO anime_source (anime_id, provider, entity_type, address)
         VALUES ($1, 'mal', 'anime', '100'), ($1, 'anidb', 'anime', '200'), ($1, 'anilist', 'anime', '300')",
    )
    .bind(anime_id)
    .execute(&pool)
    .await
    .expect("link anime sources");
    sqlx::query(
        "INSERT INTO anime_partial_source (anime_id, provider, entity_type, address)
         VALUES ($1, 'kitsu', 'anime', '400')",
    )
    .bind(anime_id)
    .execute(&pool)
    .await
    .expect("link anime partial source");

    let anime_pages = list_entity_pages(&pool, EntityType::Anime, anime_id)
        .await
        .expect("list anime pages");
    assert_eq!(anime_pages.len(), 4);
    assert_eq!(
        anime_pages.iter().filter(|p| !p.is_partial_source).count(),
        3
    );
    assert_eq!(
        anime_pages.iter().filter(|p| p.is_partial_source).count(),
        1
    );

    // Episode pages
    let episode_id = Uuid::now_v7();
    sqlx::query("INSERT INTO episode (id, anime_id, episode_number) VALUES ($1, $2, 1)")
        .bind(episode_id)
        .bind(anime_id)
        .execute(&pool)
        .await
        .expect("insert episode");
    insert_page(&pool, "mal", "episode", "101", None, false, None, 0).await;
    sqlx::query("INSERT INTO episode_source (episode_id, provider, entity_type, address) VALUES ($1, 'mal', 'episode', '101')")
        .bind(episode_id)
        .execute(&pool)
        .await
        .expect("link episode page");
    assert_eq!(
        list_entity_pages(&pool, EntityType::Episode, episode_id)
            .await
            .expect("list episode pages")
            .len(),
        1
    );

    // Character pages
    let character_id = insert_unmerged_character(&pool, "Spike").await;
    link_character_page(&pool, character_id, "mal", "102").await;
    assert_eq!(
        list_entity_pages(&pool, EntityType::Character, character_id)
            .await
            .expect("list character pages")
            .len(),
        1
    );

    // Missing entity -> NotFound
    assert!(
        list_entity_pages(&pool, EntityType::Anime, Uuid::now_v7())
            .await
            .expect_err("missing entity")
            .is_not_found()
    );

    // RecordRefresh: PagesFailed picks earliest failed non-dead source page (2^2 = 4h after fetch_failed_at)
    record_refresh(
        &pool,
        EntityType::Anime,
        anime_id,
        RefreshOutcome::PagesFailed,
        Duration::days(7),
    )
    .await
    .expect("record pages failed");
    let next_after_pages_failed: chrono::DateTime<Utc> =
        sqlx::query_scalar("SELECT next_refresh_at FROM anime WHERE id = $1")
            .bind(anime_id)
            .fetch_one(&pool)
            .await
            .expect("read next_refresh_at");
    assert!(next_after_pages_failed < now + Duration::hours(5));

    // RecordRefresh: Succeeded sets +refresh_after and resets failures
    record_refresh(
        &pool,
        EntityType::Anime,
        anime_id,
        RefreshOutcome::Succeeded,
        Duration::days(7),
    )
    .await
    .expect("record succeeded");
    let next_after_success: chrono::DateTime<Utc> =
        sqlx::query_scalar("SELECT next_refresh_at FROM anime WHERE id = $1")
            .bind(anime_id)
            .fetch_one(&pool)
            .await
            .expect("read next_refresh_at");
    assert!(next_after_success > now + Duration::days(6));

    // RecordRefresh: Failed increments refresh_failures and sets 2^failures backoff
    record_refresh(
        &pool,
        EntityType::Episode,
        episode_id,
        RefreshOutcome::Failed,
        Duration::days(7),
    )
    .await
    .expect("record failed");
    let (failures, next_after_fail): (i32, chrono::DateTime<Utc>) =
        sqlx::query_as("SELECT refresh_failures, next_refresh_at FROM episode WHERE id = $1")
            .bind(episode_id)
            .fetch_one(&pool)
            .await
            .expect("read episode failures");
    assert_eq!(failures, 1);
    assert!(next_after_fail > now + Duration::hours(1));
    record_refresh(
        &pool,
        EntityType::Episode,
        episode_id,
        RefreshOutcome::PagesFailed,
        Duration::days(7),
    )
    .await
    .expect("record episode pages failed");
    record_refresh(
        &pool,
        EntityType::Character,
        character_id,
        RefreshOutcome::PagesFailed,
        Duration::days(7),
    )
    .await
    .expect("record character pages failed");
    assert!(
        "invalid"
            .parse::<EntityType>()
            .expect_err("unknown entity type")
            .is_invalid_argument()
    );

    // Negative duration -> InvalidArgument; missing entity -> NotFound
    assert!(
        record_refresh(
            &pool,
            EntityType::Anime,
            anime_id,
            RefreshOutcome::Succeeded,
            Duration::seconds(-1),
        )
        .await
        .expect_err("negative duration")
        .is_invalid_argument()
    );
    assert!(
        record_refresh(
            &pool,
            EntityType::Character,
            Uuid::now_v7(),
            RefreshOutcome::Succeeded,
            Duration::days(7),
        )
        .await
        .expect_err("missing entity")
        .is_not_found()
    );
}

#[sqlx::test(migrations = "./migrations")]
async fn list_due_entities_covers_paging_age_status_and_merge_version(pool: PgPool) {
    let now = Utc::now();
    let ongoing_recent = insert_unmerged_anime(&pool).await;
    let old_but_stale_merge = insert_unmerged_anime(&pool).await;
    let old_current_merge = insert_unmerged_anime(&pool).await;
    let finished_recent = insert_unmerged_anime(&pool).await;

    sqlx::query("UPDATE anime SET status = 'ONGOING', year = 2025, merge_version = 2, next_refresh_at = NULL WHERE id = $1")
        .bind(ongoing_recent)
        .execute(&pool)
        .await
        .expect("update ongoing_recent");
    sqlx::query("UPDATE anime SET status = 'FINISHED', year = 1980, merge_version = 1, next_refresh_at = $2 WHERE id = $1")
        .bind(old_but_stale_merge)
        .bind(now + Duration::days(30))
        .execute(&pool)
        .await
        .expect("update old_but_stale_merge");
    sqlx::query("UPDATE anime SET status = 'FINISHED', year = 1980, merge_version = 2, next_refresh_at = $2 WHERE id = $1")
        .bind(old_current_merge)
        .bind(now - Duration::days(1))
        .execute(&pool)
        .await
        .expect("update old_current_merge");
    sqlx::query("UPDATE anime SET status = 'FINISHED', year = 2025, merge_version = 2, next_refresh_at = $2 WHERE id = $1")
        .bind(finished_recent)
        .bind(now - Duration::hours(1))
        .execute(&pool)
        .await
        .expect("update finished_recent");

    // Page 1 (size 2)
    let page1 = list_due_anime(
        &pool,
        &DueEntitiesQuery {
            due_before: now,
            older_after_years: 30,
            page_size: 2,
            after_next_refresh_at: None,
            after_id: None,
            merge_version: 2,
        },
    )
    .await
    .expect("list_due_anime page 1");
    assert_eq!(page1.len(), 2);
    assert_eq!(page1[0].entity_id, ongoing_recent);
    assert_eq!(page1[1].entity_id, finished_recent);

    // Page 2 using keyset cursor
    let page2 = list_due_anime(
        &pool,
        &DueEntitiesQuery {
            due_before: now,
            older_after_years: 30,
            page_size: 2,
            after_next_refresh_at: page1[1].next_refresh_at,
            after_id: Some(page1[1].entity_id),
            merge_version: 2,
        },
    )
    .await
    .expect("list_due_anime page 2");
    assert_eq!(page2.len(), 1);
    assert_eq!(page2[0].entity_id, old_but_stale_merge);

    // Episodes: only ongoing/upcoming works
    let ep_ongoing = Uuid::now_v7();
    let ep_finished = Uuid::now_v7();
    sqlx::query("INSERT INTO episode (id, anime_id, episode_number, merge_version) VALUES ($1, $2, 1, 2), ($3, $4, 1, 2)")
        .bind(ep_ongoing)
        .bind(ongoing_recent)
        .bind(ep_finished)
        .bind(finished_recent)
        .execute(&pool)
        .await
        .expect("insert episodes");
    let due_eps = list_due_episodes(
        &pool,
        &DueEntitiesQuery {
            due_before: now,
            older_after_years: 30,
            page_size: 10,
            after_next_refresh_at: None,
            after_id: None,
            merge_version: 2,
        },
    )
    .await
    .expect("list_due_episodes");
    assert_eq!(due_eps.len(), 1);
    assert_eq!(due_eps[0].entity_id, ep_ongoing);

    // Characters: included when at least one undeleted work is ongoing/upcoming and not all works are older
    let char_ongoing = insert_unmerged_character(&pool, "Ongoing Hero").await;
    let char_finished_only = insert_unmerged_character(&pool, "Finished Hero").await;
    insert_page(&pool, "mal", "anime", "ongoing_a", None, false, None, 0).await;
    insert_page(&pool, "mal", "anime", "finished_a", None, false, None, 0).await;
    sqlx::query("INSERT INTO anime_source (anime_id, provider, entity_type, address) VALUES ($1, 'mal', 'anime', 'ongoing_a'), ($2, 'mal', 'anime', 'finished_a')")
        .bind(ongoing_recent)
        .bind(finished_recent)
        .execute(&pool)
        .await
        .expect("link anime sources");
    sqlx::query("INSERT INTO character_animeography (character_id, anime_page_provider, anime_page_address, role) VALUES ($1, 'mal', 'ongoing_a', 'MAIN'), ($2, 'mal', 'finished_a', 'MAIN')")
        .bind(char_ongoing)
        .bind(char_finished_only)
        .execute(&pool)
        .await
        .expect("link animeography");
    let due_chars = list_due_characters(
        &pool,
        &DueEntitiesQuery {
            due_before: now,
            older_after_years: 30,
            page_size: 10,
            after_next_refresh_at: None,
            after_id: None,
            merge_version: 2,
        },
    )
    .await
    .expect("list_due_characters");
    assert_eq!(due_chars.len(), 1);
    assert_eq!(due_chars[0].entity_id, char_ongoing);

    // Invalid parameters
    assert!(
        list_due_anime(
            &pool,
            &DueEntitiesQuery {
                due_before: now,
                older_after_years: 30,
                page_size: 0,
                after_next_refresh_at: None,
                after_id: None,
                merge_version: 1,
            },
        )
        .await
        .expect_err("zero page_size")
        .is_invalid_argument()
    );
    assert!(
        list_due_anime(
            &pool,
            &DueEntitiesQuery {
                due_before: now,
                older_after_years: -1,
                page_size: 10,
                after_next_refresh_at: None,
                after_id: None,
                merge_version: 1,
            },
        )
        .await
        .expect_err("negative older_after_years")
        .is_invalid_argument()
    );
}

#[sqlx::test(migrations = "./migrations")]
async fn list_upcoming_score_requested_versions_and_due_pages_cover_all_rules(pool: PgPool) {
    let now = Utc::now();
    let ongoing_id = insert_unmerged_anime(&pool).await;
    let finished_recent_id = insert_unmerged_anime(&pool).await;
    let finished_old_id = insert_unmerged_anime(&pool).await;

    sqlx::query("UPDATE anime SET status = 'ONGOING', title = 'Ongoing', type = 'TV', merged_at = $2, merge_requested_at = $2 WHERE id = $1")
        .bind(ongoing_id)
        .bind(now)
        .execute(&pool)
        .await
        .expect("update ongoing");
    sqlx::query("UPDATE anime SET status = 'FINISHED', title = 'Recent', type = 'TV', aired_to = $2, merged_at = $3 WHERE id = $1")
        .bind(finished_recent_id)
        .bind(now - Duration::days(10))
        .bind(now)
        .execute(&pool)
        .await
        .expect("update finished_recent");
    sqlx::query("UPDATE anime SET status = 'FINISHED', title = 'Old', type = 'TV', aired_to = $2 WHERE id = $1")
        .bind(finished_old_id)
        .bind(now - Duration::days(120))
        .execute(&pool)
        .await
        .expect("update finished_old");

    // Upcoming episodes: one existing row in 24h lookback, one notified (excluded), one broadcast-only next episode
    let ep1 = Uuid::now_v7();
    let ep_notified = Uuid::now_v7();
    sqlx::query(
        "INSERT INTO episode (id, anime_id, episode_number, aired_at, aired_notified_at)
         VALUES ($1, $2, 1, $3, NULL), ($4, $2, 3, $3, $3)",
    )
    .bind(ep1)
    .bind(ongoing_id)
    .bind(now - Duration::hours(2))
    .bind(ep_notified)
    .execute(&pool)
    .await
    .expect("insert ep1 and notified ep3");
    sqlx::query("INSERT INTO anime_broadcast (anime_id, next_episode_at, next_episode_number) VALUES ($1, $2, 2)")
        .bind(ongoing_id)
        .bind(now + Duration::days(2))
        .execute(&pool)
        .await
        .expect("insert broadcast");
    let upcoming = list_upcoming_episodes(&pool, 7)
        .await
        .expect("list_upcoming_episodes");
    assert_eq!(upcoming.len(), 2);
    assert_eq!(upcoming[0].episode_id, Some(ep1));
    assert_eq!(upcoming[1].episode_id, None);
    assert_eq!(upcoming[1].episode_number, 2);

    // Score anime: ongoing + recently finished
    let score_ids = list_score_anime(&pool, 90, 10, None)
        .await
        .expect("list_score_anime");
    assert_eq!(score_ids.len(), 2);
    assert!(score_ids.contains(&ongoing_id));
    assert!(score_ids.contains(&finished_recent_id));

    // Requested merges
    let requested = list_requested_merges(&pool, EntityType::Anime, 10)
        .await
        .expect("list_requested_merges");
    assert_eq!(requested, vec![ongoing_id]);

    // Entity versions (visible only -> ongoing_id and finished_recent_id, NOT finished_old_id)
    let versions = list_entity_versions(&pool, EntityType::Anime, 10, None)
        .await
        .expect("list_entity_versions");
    assert_eq!(versions.len(), 2);
    assert_eq!(
        list_entity_versions(&pool, EntityType::Episode, 10, None)
            .await
            .expect("list episode versions")
            .len(),
        0
    );
    assert_eq!(
        list_entity_versions(&pool, EntityType::Character, 10, None)
            .await
            .expect("list character versions")
            .len(),
        0
    );

    // Due pages for budgeted provider (anidb): null-fetched first, expired backoff included, skips dead and active backoff
    insert_page(&pool, "anidb", "anime", "901", None, false, None, 0).await;
    insert_page(
        &pool,
        "anidb",
        "anime",
        "902",
        Some(now - Duration::days(10)),
        false,
        Some(now - Duration::hours(5)),
        1,
    )
    .await;
    insert_page(&pool, "anidb", "anime", "903", None, true, None, 0).await;
    insert_page(
        &pool,
        "anidb",
        "anime",
        "904",
        Some(now - Duration::days(10)),
        false,
        Some(now - Duration::minutes(5)),
        1,
    )
    .await;
    sqlx::query("INSERT INTO anime_source (anime_id, provider, entity_type, address) VALUES ($1, 'anidb', 'anime', '901'), ($2, 'anidb', 'anime', '902')")
        .bind(ongoing_id)
        .bind(finished_recent_id)
        .execute(&pool)
        .await
        .expect("link due pages");
    let due_pages = list_due_pages(&pool, Provider::AniDb, 10)
        .await
        .expect("list_due_pages");
    assert_eq!(due_pages.len(), 2);
    assert_eq!(
        due_pages[0].page_key,
        PageKey::new(Provider::AniDb, EntityType::Anime, "901").expect("key")
    );
    assert_eq!(
        due_pages[1].page_key,
        PageKey::new(Provider::AniDb, EntityType::Anime, "902").expect("key")
    );

    // Validation checks
    assert!(
        list_upcoming_episodes(&pool, -1)
            .await
            .expect_err("neg window")
            .is_invalid_argument()
    );
    assert!(
        list_score_anime(&pool, -1, 10, None)
            .await
            .expect_err("neg lookback")
            .is_invalid_argument()
    );
    assert!(
        list_score_anime(&pool, 90, 0, None)
            .await
            .expect_err("zero page_size")
            .is_invalid_argument()
    );
    assert!(
        list_requested_merges(&pool, EntityType::Anime, 0)
            .await
            .expect_err("zero limit")
            .is_invalid_argument()
    );
    assert!(
        list_entity_versions(&pool, EntityType::Anime, 0, None)
            .await
            .expect_err("zero page_size")
            .is_invalid_argument()
    );
    assert!(
        list_due_pages(&pool, Provider::AniDb, 0)
            .await
            .expect_err("zero page_size")
            .is_invalid_argument()
    );

    // Ensure SaveAnime keeps compile-checked support helper happy
    let saved = save_anime(
        &pool,
        &build_anime_save(finished_old_id, 1, "h1", Vec::new()),
    )
    .await
    .expect("save anime");
    assert!(saved.has_changed);
}
