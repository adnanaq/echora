//! Integration tests for provider page observations against PostgreSQL.

mod support;

use chrono::{Duration, Utc};
use postgres_db::{
    FetchOutcome, PageKey, PageKind, PageObservation, PageResponse, Provider,
    record_page_observation,
};
use sqlx::PgPool;
use uuid::Uuid;

use crate::support::CapturedLogs;

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
        "SELECT is_dead FROM provider_page WHERE provider = $1 AND page_kind = $2 AND address = $3",
    )
    .bind(page_key.provider().as_str())
    .bind(page_key.page_kind().as_str())
    .bind(page_key.address().as_str())
    .fetch_one(pool)
    .await
    .unwrap_or(false);
    let response_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM provider_page_response
         WHERE provider = $1 AND page_kind = $2 AND address = $3",
    )
    .bind(page_key.provider().as_str())
    .bind(page_key.page_kind().as_str())
    .bind(page_key.address().as_str())
    .fetch_one(pool)
    .await
    .unwrap_or(-1);
    (is_dead, response_count)
}

#[sqlx::test(migrations = "./migrations")]
async fn record_page_observation_not_found_keeps_responses_and_page_alive(pool: PgPool) {
    let page_key = PageKey::new(Provider::Mal, PageKind::Anime, "21").expect("valid page key");
    let first_fetch_at = Utc::now();
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
    let page_key = PageKey::new(Provider::Mal, PageKind::Episode, "7").expect("valid page key");
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
        "INSERT INTO provider_page (provider, page_kind, address) VALUES ('mal', 'episode', '7')",
    )
    .execute(&pool)
    .await
    .expect("insert placeholder page");
    sqlx::query(
        "INSERT INTO episode_provider_page (episode_id, provider, page_kind, address)
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
    let work_version: i64 = sqlx::query_scalar("SELECT version FROM anime WHERE id = $1")
        .bind(anime_id)
        .fetch_one(&pool)
        .await
        .expect("read the work");
    assert_eq!(work_version, 1);
}

#[sqlx::test(migrations = "./migrations")]
async fn record_page_observation_logs_recorded_event_with_page_key(pool: PgPool) {
    let page_key = PageKey::new(Provider::Mal, PageKind::Anime, "23").expect("valid page key");
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
