//! Integration tests for provider page observations against PostgreSQL.

mod support;

use chrono::{Duration, Utc};
use postgres_db::{
    EntityType, FetchOutcome, PageKey, PageObservation, PageResponse, Provider,
    record_page_observation,
};
use sqlx::PgPool;
use uuid::Uuid;

use crate::support::{
    CapturedLogs, insert_unmerged_anime, insert_unmerged_character, link_character_page,
};

fn build_success(body: &[u8]) -> FetchOutcome {
    FetchOutcome::Success {
        responses: vec![PageResponse {
            request_name: "main".to_owned(),
            content_type: "application/json".to_owned(),
            body: body.to_vec(),
        }],
    }
}

async fn read_page_state(pool: &PgPool, page_key: &PageKey) -> (bool, i64) {
    let is_dead: bool = sqlx::query_scalar(
        "SELECT is_dead FROM provider_page WHERE provider = $1 AND entity_type = $2 AND address = $3",
    )
    .bind(page_key.provider().as_str())
    .bind(page_key.entity_type().as_str())
    .bind(page_key.address().as_str())
    .fetch_one(pool)
    .await
    .unwrap_or(false);
    let response_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM provider_page_response
         WHERE provider = $1 AND entity_type = $2 AND address = $3",
    )
    .bind(page_key.provider().as_str())
    .bind(page_key.entity_type().as_str())
    .bind(page_key.address().as_str())
    .fetch_one(pool)
    .await
    .unwrap_or(-1);
    (is_dead, response_count)
}

#[sqlx::test(migrations = "./migrations")]
async fn record_page_observation_not_found_keeps_responses_and_page_alive(pool: PgPool) {
    let page_key = PageKey::new(Provider::Mal, EntityType::Anime, "21").expect("valid page key");
    let first_fetch_at = Utc::now() - Duration::days(7);
    record_page_observation(
        &pool,
        PageObservation {
            page_key: page_key.clone(),
            observed_at: first_fetch_at,
            outcome: build_success(br#"{"id":21}"#),
        },
    )
    .await
    .expect("store the live page");

    for answer_offset in [Duration::minutes(5), Duration::days(2)] {
        record_page_observation(
            &pool,
            PageObservation {
                page_key: page_key.clone(),
                observed_at: first_fetch_at + answer_offset,
                outcome: FetchOutcome::NotFound {
                    require_review: true,
                },
            },
        )
        .await
        .expect("record the not-found answer");
    }

    assert_eq!(read_page_state(&pool, &page_key).await, (false, 1));
}

#[sqlx::test(migrations = "./migrations")]
async fn record_page_observation_changed_episode_page_queues_episode_merge(pool: PgPool) {
    let anime_id = Uuid::now_v7();
    let episode_id = Uuid::now_v7();
    let page_key = PageKey::new(Provider::Mal, EntityType::Episode, "7").expect("valid page key");
    sqlx::query("INSERT INTO anime (id) VALUES ($1)")
        .bind(anime_id)
        .execute(&pool)
        .await
        .expect("insert placeholder work");
    sqlx::query("INSERT INTO episode (id, anime_id, episode_number) VALUES ($1, $2, 7)")
        .bind(episode_id)
        .bind(anime_id)
        .execute(&pool)
        .await
        .expect("insert placeholder episode");
    sqlx::query(
        "INSERT INTO provider_page (provider, entity_type, address) VALUES ('mal', 'episode', '7')",
    )
    .execute(&pool)
    .await
    .expect("insert placeholder page");
    sqlx::query(
        "INSERT INTO episode_source (episode_id, provider, entity_type, address)
         VALUES ($1, 'mal', 'episode', '7')",
    )
    .bind(episode_id)
    .execute(&pool)
    .await
    .expect("link the page");

    record_page_observation(
        &pool,
        PageObservation {
            page_key: page_key.clone(),
            observed_at: Utc::now(),
            outcome: build_success(br#"{"episode":7}"#),
        },
    )
    .await
    .expect("store the episode page");

    let (episode_version, is_merge_requested): (i64, bool) =
        sqlx::query_as("SELECT version, merge_requested_at IS NOT NULL FROM episode WHERE id = $1")
            .bind(episode_id)
            .fetch_one(&pool)
            .await
            .expect("read the episode");
    assert_eq!((episode_version, is_merge_requested), (2, true));
    let anime_version: i64 = sqlx::query_scalar("SELECT version FROM anime WHERE id = $1")
        .bind(anime_id)
        .fetch_one(&pool)
        .await
        .expect("read the work");
    assert_eq!(anime_version, 1);
}

#[sqlx::test(migrations = "./migrations")]
async fn record_page_observation_logs_recorded_event_with_page_key(pool: PgPool) {
    let page_key = PageKey::new(Provider::Mal, EntityType::Anime, "23").expect("valid page key");
    let captured_logs = CapturedLogs::install_global();

    record_page_observation(
        &pool,
        PageObservation {
            page_key: page_key.clone(),
            observed_at: Utc::now(),
            outcome: build_success(br#"{"id":23}"#),
        },
    )
    .await
    .expect("record the page");

    let log_text = captured_logs.read_lines_with("page.address=23");
    assert!(log_text.contains("recorded provider page"), "{log_text}");
    assert!(log_text.contains("page.has_changed=true"), "{log_text}");
    assert!(log_text.contains("page.provider=mal"), "{log_text}");
    assert!(log_text.contains("page.address=23"), "{log_text}");
}

#[sqlx::test(migrations = "./migrations")]
async fn record_page_observation_fetch_time_ahead_of_database_clock_returns_invalid_argument(
    pool: PgPool,
) {
    let page_key = PageKey::new(Provider::Mal, EntityType::Anime, "24").expect("valid page key");

    let future_error = record_page_observation(
        &pool,
        PageObservation {
            page_key: page_key.clone(),
            observed_at: Utc::now() + Duration::minutes(30),
            outcome: build_success(br#"{"id":24}"#),
        },
    )
    .await
    .expect_err("a fetch time half an hour ahead must be refused");
    assert!(future_error.is_invalid_argument());
    let stored_pages: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM provider_page WHERE address = '24'")
            .fetch_one(&pool)
            .await
            .expect("count the stored pages");
    assert_eq!(stored_pages, 0);

    let slightly_ahead = record_page_observation(
        &pool,
        PageObservation {
            page_key: page_key.clone(),
            observed_at: Utc::now() + Duration::minutes(1),
            outcome: build_success(br#"{"id":24}"#),
        },
    )
    .await
    .expect("a fetch time within the clock allowance is stored");
    assert!(slightly_ahead.has_changed);
}

#[sqlx::test(migrations = "./migrations")]
async fn record_page_observation_changed_page_requests_merge_of_source_anime_only(pool: PgPool) {
    let page_key = PageKey::new(Provider::Mal, EntityType::Anime, "25").expect("valid page key");
    let source_anime_id = insert_unmerged_anime(&pool).await;
    let partial_anime_id = insert_unmerged_anime(&pool).await;
    record_page_observation(
        &pool,
        PageObservation {
            page_key: page_key.clone(),
            observed_at: Utc::now() - Duration::days(1),
            outcome: build_success(br#"{"id":25}"#),
        },
    )
    .await
    .expect("store the first fetch");
    for (anime_id, link_table) in [
        (source_anime_id, "anime_source"),
        (partial_anime_id, "anime_partial_source"),
    ] {
        sqlx::query(&format!(
            "INSERT INTO {link_table} (anime_id, provider, entity_type, address)
             VALUES ($1, 'mal', 'anime', '25')"
        ))
        .bind(anime_id)
        .execute(&pool)
        .await
        .expect("link the page");
    }

    let changed_page = record_page_observation(
        &pool,
        PageObservation {
            page_key,
            observed_at: Utc::now(),
            outcome: build_success(br#"{"id":25,"title":"changed"}"#),
        },
    )
    .await
    .expect("store the changed fetch");

    assert!(changed_page.has_changed);
    let anime_states: Vec<(i64, bool)> = sqlx::query_as(
        "SELECT version, merge_requested_at IS NOT NULL FROM anime WHERE id = ANY($1)
         ORDER BY array_position($1, id)",
    )
    .bind(vec![source_anime_id, partial_anime_id])
    .fetch_all(&pool)
    .await
    .expect("read the works");
    assert_eq!(anime_states, vec![(2, true), (1, false)]);
}

#[sqlx::test(migrations = "./migrations")]
async fn record_page_observation_changed_character_page_requests_merge_of_owning_character(
    pool: PgPool,
) {
    let page_key =
        PageKey::new(Provider::Mal, EntityType::Character, "40").expect("valid page key");
    let owning_character_id = insert_unmerged_character(&pool, "Monkey D. Luffy").await;
    let other_character_id = insert_unmerged_character(&pool, "Roronoa Zoro").await;
    record_page_observation(
        &pool,
        PageObservation {
            page_key: page_key.clone(),
            observed_at: Utc::now() - Duration::days(1),
            outcome: build_success(br#"{"id":40}"#),
        },
    )
    .await
    .expect("store the first fetch");
    link_character_page(&pool, owning_character_id, "mal", "40").await;
    link_character_page(&pool, other_character_id, "mal", "62").await;

    let unchanged_page = record_page_observation(
        &pool,
        PageObservation {
            page_key: page_key.clone(),
            observed_at: Utc::now() - Duration::hours(1),
            outcome: build_success(br#"{"id":40}"#),
        },
    )
    .await
    .expect("store the same fetch again");
    let changed_page = record_page_observation(
        &pool,
        PageObservation {
            page_key,
            observed_at: Utc::now(),
            outcome: build_success(br#"{"id":40,"name":"changed"}"#),
        },
    )
    .await
    .expect("store the changed fetch");

    assert!(!unchanged_page.has_changed);
    assert!(changed_page.has_changed);
    let character_states: Vec<(i64, bool)> = sqlx::query_as(
        "SELECT version, merge_requested_at IS NOT NULL FROM character WHERE id = ANY($1)
         ORDER BY array_position($1, id)",
    )
    .bind(vec![owning_character_id, other_character_id])
    .fetch_all(&pool)
    .await
    .expect("read the characters");
    assert_eq!(character_states, vec![(2, true), (1, false)]);
}
