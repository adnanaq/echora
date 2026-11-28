//! Integration tests for scheduling, due queries, page listing, and refresh recording.

#![expect(
    clippy::expect_used,
    reason = "a failed setup step stops the test, as a test's own expect does"
)]

mod support;

use chrono::{Duration, Utc};
use echora_domain::{EntityType, PageKey, Provider};
use postgres_db::{
    DueEntitiesQuery, MAX_OLDER_AFTER_YEARS, MAX_WINDOW_DAYS, RefreshOutcome, list_due_anime,
    list_due_characters, list_due_episodes, list_due_pages, list_entity_pages,
    list_entity_versions, list_requested_merges, list_score_anime, list_upcoming_episodes,
    record_refresh, save_anime,
};
use sqlx::PgPool;
use uuid::Uuid;

use crate::support::{
    build_anime_save, insert_unmerged_anime, insert_unmerged_character, link_character_page,
};

#[expect(
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
    assert_eq!(anime_pages.merge_version, None);
    assert_eq!(anime_pages.pages.len(), 4);
    assert_eq!(
        anime_pages
            .pages
            .iter()
            .filter(|page| !page.is_partial_source)
            .count(),
        3
    );
    assert_eq!(
        anime_pages
            .pages
            .iter()
            .filter(|page| page.is_partial_source)
            .count(),
        1
    );

    // Episode pages
    let episode_id = Uuid::now_v7();
    sqlx::query(
        "INSERT INTO episode (id, anime_id, episode_number, merge_version) VALUES ($1, $2, 1, 3)",
    )
    .bind(episode_id)
    .bind(anime_id)
    .execute(&pool)
    .await
    .expect("insert episode");
    insert_page(&pool, "mal", "episode", "21:101", None, false, None, 0).await;
    sqlx::query("INSERT INTO episode_source (episode_id, provider, entity_type, address) VALUES ($1, 'mal', 'episode', '21:101')")
        .bind(episode_id)
        .execute(&pool)
        .await
        .expect("link episode page");
    let episode_pages = list_entity_pages(&pool, EntityType::Episode, episode_id)
        .await
        .expect("list episode pages");
    assert_eq!(episode_pages.pages.len(), 1);
    assert_eq!(episode_pages.merge_version, Some(3));

    // Character pages
    let character_id = insert_unmerged_character(&pool, "Spike").await;
    link_character_page(&pool, character_id, "mal", "102").await;
    let character_pages = list_entity_pages(&pool, EntityType::Character, character_id)
        .await
        .expect("list character pages");
    assert_eq!(character_pages.pages.len(), 1);
    assert_eq!(character_pages.merge_version, None);

    // Missing entity -> NotFound
    assert!(
        list_entity_pages(&pool, EntityType::Anime, Uuid::now_v7())
            .await
            .expect_err("missing entity")
            .is_not_found()
    );

    insert_page(
        &pool,
        "anilist",
        "anime",
        "not-a-number",
        None,
        false,
        None,
        0,
    )
    .await;
    sqlx::query(
        "INSERT INTO anime_partial_source (anime_id, provider, entity_type, address) \
         VALUES ($1, 'anilist', 'anime', 'not-a-number')",
    )
    .bind(anime_id)
    .execute(&pool)
    .await
    .expect("link malformed page");
    let filtered_pages = list_entity_pages(&pool, EntityType::Anime, anime_id)
        .await
        .expect("malformed page key must be skipped rather than failing the work");
    assert_eq!(filtered_pages.pages.len(), 4);
    sqlx::query(
        "DELETE FROM anime_partial_source WHERE anime_id = $1 AND provider = 'anilist' AND address = 'not-a-number'",
    )
    .bind(anime_id)
    .execute(&pool)
    .await
    .expect("unlink malformed page");

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
    assert!(next_after_pages_failed > now + Duration::hours(3));
    record_refresh(
        &pool,
        EntityType::Anime,
        anime_id,
        RefreshOutcome::PagesFailed,
        Duration::hours(1),
    )
    .await
    .expect("record pages failed with short refresh_after");
    let capped_next_refresh_at: chrono::DateTime<Utc> =
        sqlx::query_scalar("SELECT next_refresh_at FROM anime WHERE id = $1")
            .bind(anime_id)
            .fetch_one(&pool)
            .await
            .expect("read capped next_refresh_at");
    assert!(capped_next_refresh_at > now + Duration::minutes(50));
    assert!(capped_next_refresh_at <= Utc::now() + Duration::hours(1));

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
        postgres_db::Error::from(
            "invalid"
                .parse::<EntityType>()
                .expect_err("unknown entity type")
        )
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

    let old_ongoing = insert_unmerged_anime(&pool).await;
    sqlx::query(
        "UPDATE anime SET status = 'ONGOING', year = 1969, merge_version = 2 WHERE id = $1",
    )
    .bind(old_ongoing)
    .execute(&pool)
    .await
    .expect("update old_ongoing");
    let char_ongoing = insert_unmerged_character(&pool, "Ongoing Hero").await;
    let char_finished_only = insert_unmerged_character(&pool, "Finished Hero").await;
    let char_old_ongoing_and_finished_recent =
        insert_unmerged_character(&pool, "Long Running Hero").await;
    insert_page(&pool, "mal", "anime", "ongoing_a", None, false, None, 0).await;
    insert_page(&pool, "mal", "anime", "finished_a", None, false, None, 0).await;
    insert_page(&pool, "mal", "anime", "old_ongoing_a", None, false, None, 0).await;
    sqlx::query("INSERT INTO anime_source (anime_id, provider, entity_type, address) VALUES ($1, 'mal', 'anime', 'ongoing_a'), ($2, 'mal', 'anime', 'finished_a'), ($3, 'mal', 'anime', 'old_ongoing_a')")
        .bind(ongoing_recent)
        .bind(finished_recent)
        .bind(old_ongoing)
        .execute(&pool)
        .await
        .expect("link anime sources");
    sqlx::query("INSERT INTO character_animeography (character_id, anime_page_provider, anime_page_address, role) VALUES ($1, 'mal', 'ongoing_a', 'MAIN'), ($2, 'mal', 'finished_a', 'MAIN'), ($3, 'mal', 'old_ongoing_a', 'MAIN'), ($3, 'mal', 'finished_a', 'SUPPORTING')")
        .bind(char_ongoing)
        .bind(char_finished_only)
        .bind(char_old_ongoing_and_finished_recent)
        .execute(&pool)
        .await
        .expect("link animeography");
    sqlx::query("UPDATE character SET merge_version = 2 WHERE id = ANY($1)")
        .bind(vec![
            char_ongoing,
            char_finished_only,
            char_old_ongoing_and_finished_recent,
        ])
        .execute(&pool)
        .await
        .expect("set current character merge version");
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
    sqlx::query("UPDATE character SET merge_version = 1 WHERE id = $1")
        .bind(char_finished_only)
        .execute(&pool)
        .await
        .expect("set older character merge version");
    let due_with_older_merge = list_due_characters(
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
    .expect("list_due_characters with older merge version");
    let mut due_character_ids: Vec<Uuid> = due_with_older_merge
        .iter()
        .map(|due_character| due_character.entity_id)
        .collect();
    due_character_ids.sort();
    let mut expected_character_ids = vec![char_ongoing, char_finished_only];
    expected_character_ids.sort();
    assert_eq!(due_character_ids, expected_character_ids);

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
    for older_after_years in [MAX_OLDER_AFTER_YEARS + 1, i32::MAX] {
        assert!(
            list_due_characters(
                &pool,
                &DueEntitiesQuery {
                    due_before: now,
                    older_after_years,
                    page_size: 10,
                    after_next_refresh_at: None,
                    after_id: None,
                    merge_version: 1,
                },
            )
            .await
            .expect_err("older_after_years above bound")
            .is_invalid_argument()
        );
    }
    for window_days in [MAX_WINDOW_DAYS + 1, i32::MAX] {
        assert!(
            list_upcoming_episodes(&pool, window_days)
                .await
                .expect_err("window_days above bound")
                .is_invalid_argument()
        );
        assert!(
            list_score_anime(&pool, window_days, 10, None)
                .await
                .expect_err("lookback_days above bound")
                .is_invalid_argument()
        );
    }
    assert!(list_upcoming_episodes(&pool, MAX_WINDOW_DAYS).await.is_ok());
    assert!(
        list_score_anime(&pool, MAX_WINDOW_DAYS, 10, None)
            .await
            .is_ok()
    );
}

#[sqlx::test(migrations = "./migrations")]
async fn list_due_anime_paging_through_tied_refresh_times_returns_each_work_once(pool: PgPool) {
    let now = Utc::now();
    let tied_refresh_at = now - Duration::hours(2);
    let mut never_refreshed = Vec::new();
    let mut refreshed_at_same_time = Vec::new();
    for _ in 0..3 {
        never_refreshed.push(insert_unmerged_anime(&pool).await);
        refreshed_at_same_time.push(insert_unmerged_anime(&pool).await);
    }
    sqlx::query("UPDATE anime SET status = 'ONGOING', year = 2025, merge_version = 2, next_refresh_at = NULL WHERE id = ANY($1)")
        .bind(&never_refreshed)
        .execute(&pool)
        .await
        .expect("update never_refreshed");
    sqlx::query("UPDATE anime SET status = 'ONGOING', year = 2025, merge_version = 2, next_refresh_at = $2 WHERE id = ANY($1)")
        .bind(&refreshed_at_same_time)
        .bind(tied_refresh_at)
        .execute(&pool)
        .await
        .expect("update refreshed_at_same_time");

    let mut paged_ids = Vec::new();
    let mut query = DueEntitiesQuery {
        due_before: now,
        older_after_years: 30,
        page_size: 1,
        after_next_refresh_at: None,
        after_id: None,
        merge_version: 2,
    };
    for _ in 0..10 {
        let due_page = list_due_anime(&pool, &query)
            .await
            .expect("list_due_anime page");
        let Some(last_row) = due_page.last() else {
            break;
        };
        query.after_next_refresh_at = last_row.next_refresh_at;
        query.after_id = Some(last_row.entity_id);
        paged_ids.extend(due_page.iter().map(|row| row.entity_id));
    }

    never_refreshed.sort();
    refreshed_at_same_time.sort();
    let expected_ids: Vec<Uuid> = never_refreshed
        .into_iter()
        .chain(refreshed_at_same_time)
        .collect();
    assert_eq!(paged_ids, expected_ids);
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
        "INSERT INTO episode (id, anime_id, episode_number, aired_at, aired_notified_at, merged_at)
         VALUES ($1, $2, 1, $3, NULL, $3), ($4, $2, 3, $3, $3, $3)",
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
        2
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

async fn merge_test_anime(pool: &PgPool, anime_id: Uuid, status: &str) {
    sqlx::query(
        "UPDATE anime SET title = 'Work', type = 'TV', status = $2, merged_at = now()
         WHERE id = $1",
    )
    .bind(anime_id)
    .bind(status)
    .execute(pool)
    .await
    .expect("merge work");
}

async fn insert_test_episode(
    pool: &PgPool,
    anime_id: Uuid,
    episode_number: i32,
    aired_at: Option<chrono::DateTime<Utc>>,
    is_merged: bool,
) -> Uuid {
    let episode_id = Uuid::now_v7();
    sqlx::query(
        "INSERT INTO episode (id, anime_id, episode_number, aired_at, merged_at)
         VALUES ($1, $2, $3, $4, CASE WHEN $5 THEN now() END)",
    )
    .bind(episode_id)
    .bind(anime_id)
    .bind(episode_number)
    .bind(aired_at)
    .bind(is_merged)
    .execute(pool)
    .await
    .expect("insert episode");
    episode_id
}

async fn insert_test_broadcast(
    pool: &PgPool,
    anime_id: Uuid,
    next_episode_number: i32,
    next_episode_at: chrono::DateTime<Utc>,
) {
    sqlx::query(
        "INSERT INTO anime_broadcast (anime_id, next_episode_at, next_episode_number)
         VALUES ($1, $2, $3)",
    )
    .bind(anime_id)
    .bind(next_episode_at)
    .bind(next_episode_number)
    .execute(pool)
    .await
    .expect("insert broadcast");
}

async fn read_anime_refresh_state(pool: &PgPool, anime_id: Uuid) -> (chrono::DateTime<Utc>, i32) {
    sqlx::query_as("SELECT next_refresh_at, refresh_failures FROM anime WHERE id = $1")
        .bind(anime_id)
        .fetch_one(pool)
        .await
        .expect("read refresh state")
}

async fn read_transaction_now(pool: &PgPool) -> chrono::DateTime<Utc> {
    sqlx::query_scalar("SELECT now()")
        .fetch_one(pool)
        .await
        .expect("read now")
}

fn assert_close(actual: chrono::DateTime<Utc>, expected: chrono::DateTime<Utc>) {
    assert!(
        (actual - expected).num_seconds().abs() <= 5,
        "{actual} is not within five seconds of {expected}"
    );
}

#[sqlx::test(migrations = "./migrations")]
async fn record_refresh_pages_failed_without_failed_page_uses_refresh_interval(pool: PgPool) {
    let anime_id = insert_unmerged_anime(&pool).await;
    let now = read_transaction_now(&pool).await;
    insert_page(&pool, "mal", "anime", "1", Some(now), false, None, 0).await;
    insert_page(&pool, "kitsu", "anime", "2", None, false, Some(now), 1).await;
    insert_page(&pool, "anilist", "anime", "3", None, true, Some(now), 1).await;
    sqlx::query(
        "INSERT INTO anime_source (anime_id, provider, entity_type, address)
         VALUES ($1, 'mal', 'anime', '1'), ($1, 'anilist', 'anime', '3')",
    )
    .bind(anime_id)
    .execute(&pool)
    .await
    .expect("link sources");
    sqlx::query(
        "INSERT INTO anime_partial_source (anime_id, provider, entity_type, address)
         VALUES ($1, 'kitsu', 'anime', '2')",
    )
    .bind(anime_id)
    .execute(&pool)
    .await
    .expect("link partial source");

    record_refresh(
        &pool,
        EntityType::Anime,
        anime_id,
        RefreshOutcome::PagesFailed,
        Duration::days(7),
    )
    .await
    .expect("record pages failed");
    let (next_refresh_at, refresh_failures) = read_anime_refresh_state(&pool, anime_id).await;
    assert_close(next_refresh_at, now + Duration::days(7));
    assert_eq!(refresh_failures, 0);
}

#[sqlx::test(migrations = "./migrations")]
async fn record_refresh_pages_failed_takes_earliest_retry_capped_at_week(pool: PgPool) {
    let anime_id = insert_unmerged_anime(&pool).await;
    let now = read_transaction_now(&pool).await;
    let long_failed_at = now - Duration::days(1);
    let recent_failed_at = now - Duration::hours(1);
    insert_page(
        &pool,
        "mal",
        "anime",
        "1",
        None,
        false,
        Some(long_failed_at),
        30,
    )
    .await;
    insert_page(
        &pool,
        "kitsu",
        "anime",
        "2",
        None,
        false,
        Some(recent_failed_at),
        2,
    )
    .await;
    sqlx::query(
        "INSERT INTO anime_source (anime_id, provider, entity_type, address)
         VALUES ($1, 'mal', 'anime', '1'), ($1, 'kitsu', 'anime', '2')",
    )
    .bind(anime_id)
    .execute(&pool)
    .await
    .expect("link sources");

    record_refresh(
        &pool,
        EntityType::Anime,
        anime_id,
        RefreshOutcome::PagesFailed,
        Duration::days(7),
    )
    .await
    .expect("record pages failed");
    let (next_refresh_at, _) = read_anime_refresh_state(&pool, anime_id).await;
    assert_close(next_refresh_at, recent_failed_at + Duration::hours(4));

    sqlx::query("DELETE FROM anime_source WHERE address = '2'")
        .execute(&pool)
        .await
        .expect("unlink recent page");
    record_refresh(
        &pool,
        EntityType::Anime,
        anime_id,
        RefreshOutcome::PagesFailed,
        Duration::days(7),
    )
    .await
    .expect("record pages failed");
    let (capped_refresh_at, _) = read_anime_refresh_state(&pool, anime_id).await;
    assert_close(capped_refresh_at, long_failed_at + Duration::hours(168));
}

#[sqlx::test(migrations = "./migrations")]
async fn record_refresh_failed_doubles_backoff_up_to_week_and_success_resets(pool: PgPool) {
    let anime_id = insert_unmerged_anime(&pool).await;
    let mut expected_backoff_hours = Vec::new();
    let mut recorded_backoff_hours = Vec::new();
    for refresh_failures in 1..=9 {
        record_refresh(
            &pool,
            EntityType::Anime,
            anime_id,
            RefreshOutcome::Failed,
            Duration::days(7),
        )
        .await
        .expect("record failed");
        let now = read_transaction_now(&pool).await;
        let (next_refresh_at, stored_failures) = read_anime_refresh_state(&pool, anime_id).await;
        assert_eq!(stored_failures, refresh_failures);
        expected_backoff_hours.push((1_i64 << refresh_failures).min(168));
        recorded_backoff_hours.push((next_refresh_at - now + Duration::minutes(1)).num_hours());
    }
    assert_eq!(recorded_backoff_hours, expected_backoff_hours);

    sqlx::query("UPDATE anime SET refresh_failures = 2147483646 WHERE id = $1")
        .bind(anime_id)
        .execute(&pool)
        .await
        .expect("set huge failure count");
    record_refresh(
        &pool,
        EntityType::Anime,
        anime_id,
        RefreshOutcome::Failed,
        Duration::days(7),
    )
    .await
    .expect("record failed at huge count");
    let now = read_transaction_now(&pool).await;
    let (capped_refresh_at, _) = read_anime_refresh_state(&pool, anime_id).await;
    assert_close(capped_refresh_at, now + Duration::hours(168));

    record_refresh(
        &pool,
        EntityType::Anime,
        anime_id,
        RefreshOutcome::Succeeded,
        Duration::days(3),
    )
    .await
    .expect("record succeeded");
    let now = read_transaction_now(&pool).await;
    let (next_refresh_at, refresh_failures) = read_anime_refresh_state(&pool, anime_id).await;
    assert_close(next_refresh_at, now + Duration::days(3));
    assert_eq!(refresh_failures, 0);
}

#[sqlx::test(migrations = "./migrations")]
async fn record_refresh_interval_postgresql_cannot_hold_returns_invalid_argument(pool: PgPool) {
    let anime_id = insert_unmerged_anime(&pool).await;
    for refresh_after in [Duration::nanoseconds(1), Duration::MAX] {
        let error = record_refresh(
            &pool,
            EntityType::Anime,
            anime_id,
            RefreshOutcome::Succeeded,
            refresh_after,
        )
        .await
        .expect_err("unrepresentable interval");
        assert!(error.is_invalid_argument(), "{error}");
    }
    record_refresh(
        &pool,
        EntityType::Anime,
        anime_id,
        RefreshOutcome::Succeeded,
        Duration::zero(),
    )
    .await
    .expect("zero interval is due now");
}

#[sqlx::test(migrations = "./migrations")]
async fn list_upcoming_episodes_broadcast_branch_follows_merged_air_dates(pool: PgPool) {
    let now = read_transaction_now(&pool).await;
    let in_two_days = now + Duration::days(2);
    let pending_work = insert_unmerged_anime(&pool).await;
    let dated_work = insert_unmerged_anime(&pool).await;
    let deleted_episode_work = insert_unmerged_anime(&pool).await;
    let unmerged_work = insert_unmerged_anime(&pool).await;
    let finished_work = insert_unmerged_anime(&pool).await;
    let late_work = insert_unmerged_anime(&pool).await;
    for (anime_id, status) in [
        (pending_work, "ONGOING"),
        (dated_work, "UPCOMING"),
        (deleted_episode_work, "ONGOING"),
        (finished_work, "FINISHED"),
        (late_work, "ONGOING"),
    ] {
        merge_test_anime(&pool, anime_id, status).await;
    }
    sqlx::query("UPDATE anime SET status = 'ONGOING' WHERE id = $1")
        .bind(unmerged_work)
        .execute(&pool)
        .await
        .expect("set unmerged status");

    let pending_episode = insert_test_episode(&pool, pending_work, 5, None, false).await;
    let dated_episode =
        insert_test_episode(&pool, dated_work, 2, Some(now + Duration::days(1)), true).await;
    let deleted_episode = insert_test_episode(&pool, deleted_episode_work, 7, None, true).await;
    sqlx::query("UPDATE episode SET deleted_at = now() WHERE id = $1")
        .bind(deleted_episode)
        .execute(&pool)
        .await
        .expect("delete episode");
    insert_test_episode(&pool, unmerged_work, 1, Some(now + Duration::days(1)), true).await;
    insert_test_episode(&pool, finished_work, 1, Some(now + Duration::days(1)), true).await;
    insert_test_episode(&pool, late_work, 1, Some(now + Duration::days(30)), true).await;
    insert_test_episode(&pool, late_work, 2, Some(now - Duration::hours(30)), true).await;

    insert_test_broadcast(&pool, pending_work, 5, in_two_days).await;
    insert_test_broadcast(&pool, dated_work, 2, in_two_days).await;
    insert_test_broadcast(&pool, deleted_episode_work, 7, in_two_days).await;
    insert_test_broadcast(&pool, unmerged_work, 2, in_two_days).await;
    insert_test_broadcast(&pool, finished_work, 2, in_two_days).await;
    insert_test_broadcast(&pool, late_work, 3, now + Duration::days(9)).await;

    let upcoming: Vec<(Uuid, Option<Uuid>, i32)> = list_upcoming_episodes(&pool, 7)
        .await
        .expect("list upcoming episodes")
        .into_iter()
        .map(|upcoming_episode| {
            (
                upcoming_episode.anime_id,
                upcoming_episode.episode_id,
                upcoming_episode.episode_number,
            )
        })
        .collect();
    let mut expected_two_day_rows = vec![
        (pending_work, Some(pending_episode), 5),
        (deleted_episode_work, None, 7),
    ];
    expected_two_day_rows.sort();
    let mut expected = vec![(dated_work, Some(dated_episode), 2)];
    expected.extend(expected_two_day_rows);
    assert_eq!(upcoming, expected);
}

#[sqlx::test(migrations = "./migrations")]
async fn list_due_episodes_older_merge_version_returns_episode_of_finished_work(pool: PgPool) {
    let finished_work = insert_unmerged_anime(&pool).await;
    merge_test_anime(&pool, finished_work, "FINISHED").await;
    sqlx::query("UPDATE anime SET year = 1990 WHERE id = $1")
        .bind(finished_work)
        .execute(&pool)
        .await
        .expect("set old year");
    let current_episode = insert_test_episode(&pool, finished_work, 1, None, true).await;
    let stale_episode = insert_test_episode(&pool, finished_work, 2, None, true).await;
    sqlx::query(
        "UPDATE episode SET merge_version = CASE WHEN id = $1 THEN 1 ELSE 2 END,
                            next_refresh_at = now() + interval '30 days'
         WHERE anime_id = $2",
    )
    .bind(stale_episode)
    .bind(finished_work)
    .execute(&pool)
    .await
    .expect("set merge versions");

    let due_episodes = list_due_episodes(
        &pool,
        &DueEntitiesQuery {
            due_before: Utc::now(),
            older_after_years: 10,
            page_size: 10,
            after_next_refresh_at: None,
            after_id: None,
            merge_version: 2,
        },
    )
    .await
    .expect("list due episodes");
    let due_episode_ids: Vec<Uuid> = due_episodes
        .iter()
        .map(|due_episode| due_episode.entity_id)
        .collect();
    assert_eq!(due_episode_ids, vec![stale_episode]);
    assert!(!due_episode_ids.contains(&current_episode));
}

#[sqlx::test(migrations = "./migrations")]
async fn list_score_anime_skips_unmerged_and_undated_works_and_pages_by_id(pool: PgPool) {
    let now = read_transaction_now(&pool).await;
    let mut visible_ids = Vec::new();
    for _ in 0..3 {
        let anime_id = insert_unmerged_anime(&pool).await;
        merge_test_anime(&pool, anime_id, "ONGOING").await;
        visible_ids.push(anime_id);
    }
    let unmerged_id = insert_unmerged_anime(&pool).await;
    sqlx::query("UPDATE anime SET status = 'ONGOING' WHERE id = $1")
        .bind(unmerged_id)
        .execute(&pool)
        .await
        .expect("set unmerged status");
    let undated_id = insert_unmerged_anime(&pool).await;
    merge_test_anime(&pool, undated_id, "FINISHED").await;
    let deleted_id = insert_unmerged_anime(&pool).await;
    merge_test_anime(&pool, deleted_id, "ONGOING").await;
    sqlx::query("UPDATE anime SET deleted_at = $2 WHERE id = $1")
        .bind(deleted_id)
        .bind(now)
        .execute(&pool)
        .await
        .expect("delete work");
    visible_ids.sort();

    let first_page = list_score_anime(&pool, 90, 2, None)
        .await
        .expect("first page");
    let second_page = list_score_anime(&pool, 90, 2, first_page.last().copied())
        .await
        .expect("second page");
    assert_eq!(first_page, visible_ids[..2]);
    assert_eq!(second_page, visible_ids[2..]);
}

#[sqlx::test(migrations = "./migrations")]
async fn record_refresh_every_outcome_updates_each_entity_type(pool: PgPool) {
    let anime_id = insert_unmerged_anime(&pool).await;
    let episode_id = insert_test_episode(&pool, anime_id, 1, None, false).await;
    let character_id = insert_unmerged_character(&pool, "Spike").await;
    for (entity_type, entity_id, table) in [
        (EntityType::Anime, anime_id, "anime"),
        (EntityType::Episode, episode_id, "episode"),
        (EntityType::Character, character_id, "character"),
    ] {
        let mut recorded_states = Vec::new();
        for outcome in [
            RefreshOutcome::Failed,
            RefreshOutcome::PagesFailed,
            RefreshOutcome::Failed,
            RefreshOutcome::Succeeded,
        ] {
            record_refresh(&pool, entity_type, entity_id, outcome, Duration::days(7))
                .await
                .expect("record refresh");
            let refresh_state: (i32, bool, bool) = sqlx::query_as(&format!(
                "SELECT refresh_failures, refreshed_at IS NOT NULL, refresh_failed_at IS NOT NULL
                 FROM {table} WHERE id = $1"
            ))
            .bind(entity_id)
            .fetch_one(&pool)
            .await
            .expect("read refresh state");
            recorded_states.push(refresh_state);
        }
        assert_eq!(
            recorded_states,
            vec![
                (1, false, true),
                (0, true, true),
                (1, true, true),
                (0, true, true)
            ],
            "{table}"
        );
        for outcome in [
            RefreshOutcome::Succeeded,
            RefreshOutcome::PagesFailed,
            RefreshOutcome::Failed,
        ] {
            let error = record_refresh(
                &pool,
                entity_type,
                Uuid::now_v7(),
                outcome,
                Duration::days(7),
            )
            .await
            .expect_err("missing entity");
            assert!(error.is_not_found(), "{table}: {error}");
        }
    }
}
