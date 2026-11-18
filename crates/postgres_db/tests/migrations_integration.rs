//! Integration tests verifying `postgres_db` migrations and `store_page`.

// Rust guideline compliant 2026-10-09

use std::env;

use chrono::{Duration, Utc};
use postgres_db::{
    Error, PageKeyInput, PageKind, PageResponseInput, StorePageInput, connect_pool, run_migrations,
    store_page,
};

#[tokio::test]
async fn lookup_and_name_tables_migrate_and_seed_expected_rows() -> Result<(), Error> {
    let Ok(database_url) = env::var("DATABASE_URL") else {
        return Ok(());
    };

    let pool = connect_pool(&database_url, 2).await?;
    run_migrations(&pool).await?;
    run_migrations(&pool).await?;

    let provider_count = sqlx::query_scalar!("SELECT COUNT(*) FROM provider")
        .fetch_one(&pool)
        .await?;
    assert_eq!(provider_count, Some(12));

    let extra_providers = sqlx::query_scalar!(
        "SELECT COUNT(*) FROM provider WHERE value IN ('ann', 'livechart', 'simkl', 'animecountdown')"
    )
    .fetch_one(&pool)
    .await?;
    assert_eq!(extra_providers, Some(4));

    let has_dead_page =
        sqlx::query_scalar!("SELECT EXISTS(SELECT 1 FROM approval_kind WHERE value = 'dead_page')")
            .fetch_one(&pool)
            .await?;
    assert_eq!(has_dead_page, Some(true));

    let graph_edge_count = sqlx::query_scalar!("SELECT COUNT(*) FROM graph_edge_type")
        .fetch_one(&pool)
        .await?;
    assert_eq!(graph_edge_count, Some(23));

    let shared_name_table_count = sqlx::query_scalar!(
        "SELECT COUNT(*) FROM information_schema.tables
         WHERE table_schema = 'public'
           AND table_name IN ('genre', 'tag', 'demographic', 'content_warning', 'trait')"
    )
    .fetch_one(&pool)
    .await?;
    assert_eq!(shared_name_table_count, Some(5));

    let Err(duplicate_sqlx_err) = sqlx::query("INSERT INTO anime_type VALUES ('TV')")
        .execute(&pool)
        .await
    else {
        panic!("duplicate lookup insert must fail");
    };
    let duplicate_err = Error::from(duplicate_sqlx_err);
    assert!(duplicate_err.is_unique_violation());

    Ok(())
}

#[tokio::test]
async fn connect_pool_rejects_invalid_database_url() {
    let Err(err) = connect_pool("not-a-postgres-url", 1).await else {
        panic!("invalid database URL must fail");
    };
    assert!(err.is_database());
}

#[tokio::test]
async fn store_page_updates_hashes_and_ignores_stale_fetches() -> Result<(), Error> {
    let Ok(database_url) = env::var("DATABASE_URL") else {
        return Ok(());
    };

    let pool = connect_pool(&database_url, 2).await?;
    run_migrations(&pool).await?;

    let test_address = "999999001";
    sqlx::query!(
        "DELETE FROM provider_page WHERE provider = 'mal' AND page_kind = 'anime' AND address = $1",
        test_address
    )
    .execute(&pool)
    .await?;

    let t0 = Utc::now();
    let page_key = PageKeyInput {
        provider: "mal".to_owned(),
        page_kind: PageKind::Anime,
        address: test_address.to_owned(),
    };
    let response = PageResponseInput {
        request_name: "main".to_owned(),
        content_type: "application/json".to_owned(),
        body: br#"{"id":999999001}"#.to_vec(),
    };

    let first = store_page(
        &pool,
        StorePageInput {
            page_key: page_key.clone(),
            fetched_at: t0,
            is_dead: false,
            responses: vec![response.clone()],
            is_not_found: false,
            dead_page_review: true,
        },
    )
    .await?;
    assert!(first.has_changed);

    let same = store_page(
        &pool,
        StorePageInput {
            page_key: page_key.clone(),
            fetched_at: t0 + Duration::minutes(5),
            is_dead: false,
            responses: vec![response.clone()],
            is_not_found: false,
            dead_page_review: true,
        },
    )
    .await?;
    assert!(!same.has_changed);

    let older = store_page(
        &pool,
        StorePageInput {
            page_key,
            fetched_at: t0 - Duration::minutes(5),
            is_dead: false,
            responses: vec![PageResponseInput {
                body: b"older".to_vec(),
                ..response
            }],
            is_not_found: false,
            dead_page_review: true,
        },
    )
    .await?;
    assert!(!older.has_changed);

    Ok(())
}

#[tokio::test]
async fn store_page_tracks_failures_and_dead_page_reviews() -> Result<(), Error> {
    let Ok(database_url) = env::var("DATABASE_URL") else {
        return Ok(());
    };

    let pool = connect_pool(&database_url, 2).await?;
    run_migrations(&pool).await?;

    let test_address = "999999002";
    let t0 = Utc::now();
    let page_key = PageKeyInput {
        provider: "mal".to_owned(),
        page_kind: PageKind::Anime,
        address: test_address.to_owned(),
    };
    let expected_proposal = postgres_db::encode_page_key_proposal(&page_key);

    sqlx::query!(
        "DELETE FROM provider_page WHERE provider = 'mal' AND page_kind = 'anime' AND address = $1",
        test_address
    )
    .execute(&pool)
    .await?;
    sqlx::query!(
        "DELETE FROM approval_request WHERE kind = 'dead_page' AND proposal = $1",
        expected_proposal.as_slice()
    )
    .execute(&pool)
    .await?;

    let failed = store_page(
        &pool,
        StorePageInput {
            page_key: page_key.clone(),
            fetched_at: t0,
            is_dead: false,
            responses: Vec::new(),
            is_not_found: false,
            dead_page_review: true,
        },
    )
    .await?;
    assert!(!failed.has_changed);

    let failures = sqlx::query_scalar!(
        "SELECT fetch_failures FROM provider_page WHERE provider = 'mal' AND page_kind = 'anime' AND address = $1",
        test_address
    )
    .fetch_one(&pool)
    .await?;
    assert_eq!(failures, 1);

    store_page(
        &pool,
        StorePageInput {
            page_key: page_key.clone(),
            fetched_at: t0 + Duration::minutes(20),
            is_dead: false,
            responses: Vec::new(),
            is_not_found: true,
            dead_page_review: true,
        },
    )
    .await?;
    store_page(
        &pool,
        StorePageInput {
            page_key: page_key.clone(),
            fetched_at: t0 + Duration::days(2),
            is_dead: false,
            responses: Vec::new(),
            is_not_found: true,
            dead_page_review: true,
        },
    )
    .await?;

    let approval_count = sqlx::query_scalar!(
        "SELECT COUNT(*) FROM approval_request WHERE kind = 'dead_page' AND status = 'open' AND proposal = $1",
        expected_proposal.as_slice()
    )
    .fetch_one(&pool)
    .await?;
    assert_eq!(approval_count, Some(1));

    let direct = store_page(
        &pool,
        StorePageInput {
            page_key: page_key.clone(),
            fetched_at: t0 + Duration::days(3),
            is_dead: false,
            responses: Vec::new(),
            is_not_found: true,
            dead_page_review: false,
        },
    )
    .await?;
    assert!(direct.has_changed);

    let is_dead = sqlx::query_scalar!(
        "SELECT is_dead FROM provider_page WHERE provider = 'mal' AND page_kind = 'anime' AND address = $1",
        test_address
    )
    .fetch_one(&pool)
    .await?;
    assert!(is_dead);

    Ok(())
}

#[tokio::test]
async fn store_page_closes_open_dead_page_approval_on_recovery() -> Result<(), Error> {
    let Ok(database_url) = env::var("DATABASE_URL") else {
        return Ok(());
    };

    let pool = connect_pool(&database_url, 2).await?;
    run_migrations(&pool).await?;

    let test_address = "999999004";
    let t0 = Utc::now();
    let page_key = PageKeyInput {
        provider: "mal".to_owned(),
        page_kind: PageKind::Anime,
        address: test_address.to_owned(),
    };
    let expected_proposal = postgres_db::encode_page_key_proposal(&page_key);

    sqlx::query!(
        "DELETE FROM provider_page WHERE provider = 'mal' AND page_kind = 'anime' AND address = $1",
        test_address
    )
    .execute(&pool)
    .await?;
    sqlx::query!(
        "DELETE FROM approval_request WHERE kind = 'dead_page' AND proposal = $1",
        expected_proposal.as_slice()
    )
    .execute(&pool)
    .await?;

    store_page(
        &pool,
        StorePageInput {
            page_key: page_key.clone(),
            fetched_at: t0,
            is_dead: false,
            responses: Vec::new(),
            is_not_found: true,
            dead_page_review: true,
        },
    )
    .await?;
    store_page(
        &pool,
        StorePageInput {
            page_key: page_key.clone(),
            fetched_at: t0 + Duration::days(2),
            is_dead: false,
            responses: Vec::new(),
            is_not_found: true,
            dead_page_review: true,
        },
    )
    .await?;

    let recovered = store_page(
        &pool,
        StorePageInput {
            page_key,
            fetched_at: t0 + Duration::days(3),
            is_dead: false,
            responses: vec![PageResponseInput {
                request_name: "main".to_owned(),
                content_type: "application/json".to_owned(),
                body: br#"{"id":999999004}"#.to_vec(),
            }],
            is_not_found: false,
            dead_page_review: true,
        },
    )
    .await?;
    assert!(recovered.has_changed);

    let open_approvals_after_recovery = sqlx::query_scalar!(
        "SELECT COUNT(*) FROM approval_request WHERE kind = 'dead_page' AND status = 'open' AND proposal = $1",
        expected_proposal.as_slice()
    )
    .fetch_one(&pool)
    .await?;
    assert_eq!(open_approvals_after_recovery, Some(0));

    Ok(())
}

#[tokio::test]
async fn store_page_ignores_stale_success_after_not_found() -> Result<(), Error> {
    let Ok(database_url) = env::var("DATABASE_URL") else {
        return Ok(());
    };

    let pool = connect_pool(&database_url, 2).await?;
    run_migrations(&pool).await?;

    let test_address = "999999003";
    sqlx::query!(
        "DELETE FROM provider_page WHERE provider = 'mal' AND page_kind = 'anime' AND address = $1",
        test_address
    )
    .execute(&pool)
    .await?;

    let t0 = Utc::now();
    let page_key = PageKeyInput {
        provider: "mal".to_owned(),
        page_kind: PageKind::Anime,
        address: test_address.to_owned(),
    };

    store_page(
        &pool,
        StorePageInput {
            page_key: page_key.clone(),
            fetched_at: t0,
            is_dead: false,
            responses: Vec::new(),
            is_not_found: true,
            dead_page_review: false,
        },
    )
    .await?;

    store_page(
        &pool,
        StorePageInput {
            page_key: page_key.clone(),
            fetched_at: t0 + Duration::days(2),
            is_dead: false,
            responses: Vec::new(),
            is_not_found: true,
            dead_page_review: false,
        },
    )
    .await?;

    // Replaying the T0+2d not-found must not increment missing_answers again.
    store_page(
        &pool,
        StorePageInput {
            page_key: page_key.clone(),
            fetched_at: t0 + Duration::days(2),
            is_dead: false,
            responses: Vec::new(),
            is_not_found: true,
            dead_page_review: false,
        },
    )
    .await?;

    // A delayed success between T0 and T0+2d must not revive the dead page.
    let stale_fetch_outcome = store_page(
        &pool,
        StorePageInput {
            page_key,
            fetched_at: t0 + Duration::days(1),
            is_dead: false,
            responses: vec![PageResponseInput {
                request_name: "main".to_owned(),
                content_type: "application/json".to_owned(),
                body: br#"{"id":999999003}"#.to_vec(),
            }],
            is_not_found: false,
            dead_page_review: false,
        },
    )
    .await?;
    assert!(!stale_fetch_outcome.has_changed);

    let missing_answers = sqlx::query_scalar!(
        "SELECT missing_answers FROM provider_page WHERE provider = 'mal' AND page_kind = 'anime' AND address = $1",
        test_address
    )
    .fetch_one(&pool)
    .await?;
    assert_eq!(missing_answers, 2);

    Ok(())
}
