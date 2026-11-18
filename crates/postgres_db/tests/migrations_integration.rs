//! Integration tests verifying `postgres_db` migrations and `store_page`.

// Rust guideline compliant 2026-10-09

use std::env;

use chrono::{Duration, Utc};
use postgres_db::{
    ANIME_HASH_FORMAT_V1, Error, PageKeyInput, PageKind, PageResponseInput, SaveAnimeInput,
    StorePageInput, connect_pool, load_anime, run_migrations, save_anime, store_page,
};
use uuid::Uuid;

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

#[tokio::test]
#[expect(
    clippy::too_many_lines,
    reason = "End-to-end load, save, visibility, conflict, and outbox test"
)]
async fn load_and_save_anime_enforce_versions_hashes_visibility_and_outbox() -> Result<(), Error> {
    let Ok(database_url) = env::var("DATABASE_URL") else {
        return Ok(());
    };

    let pool = connect_pool(&database_url, 2).await?;
    run_migrations(&pool).await?;

    let anime_id = Uuid::now_v7();
    let source_address = "999999010";
    let partial_address = "999999011";

    store_page(
        &pool,
        StorePageInput {
            page_key: PageKeyInput {
                provider: "mal".to_owned(),
                page_kind: PageKind::Anime,
                address: source_address.to_owned(),
            },
            fetched_at: Utc::now(),
            is_dead: false,
            responses: vec![PageResponseInput {
                request_name: "main".to_owned(),
                content_type: "application/json".to_owned(),
                body: br#"{"mal_id":999999010}"#.to_vec(),
            }],
            is_not_found: false,
            dead_page_review: false,
        },
    )
    .await?;

    store_page(
        &pool,
        StorePageInput {
            page_key: PageKeyInput {
                provider: "anilist".to_owned(),
                page_kind: PageKind::Anime,
                address: partial_address.to_owned(),
            },
            fetched_at: Utc::now(),
            is_dead: false,
            responses: Vec::new(),
            is_not_found: false,
            dead_page_review: false,
        },
    )
    .await?;

    sqlx::query!(
        "INSERT INTO anime (id, merge_requested_at) VALUES ($1, now())",
        anime_id
    )
    .execute(&pool)
    .await?;

    // Unmerged works must not appear in anime_visible.
    let visible_before = sqlx::query_scalar!(
        "SELECT EXISTS(SELECT 1 FROM anime_visible WHERE id = $1)",
        anime_id
    )
    .fetch_one(&pool)
    .await?;
    assert_eq!(visible_before, Some(false));

    sqlx::query!(
        "INSERT INTO anime_provider_page (anime_id, provider, page_kind, address, role)
         VALUES ($1, 'mal', 'anime', $2, 'source'),
                ($1, 'anilist', 'anime', $3, 'partial_source')",
        anime_id,
        source_address,
        partial_address,
    )
    .execute(&pool)
    .await?;

    sqlx::query!(
        "INSERT INTO anime_field_correction (anime_id, field_path, value_message, value, approved_by)
         VALUES ($1, 'episode_count', 'type.googleapis.com/google.protobuf.Int32Value', $2, 'reviewer')",
        anime_id,
        &b"\x08\x4a"[..],
    )
    .execute(&pool)
    .await?;

    let loaded = load_anime(&pool, anime_id).await?;
    assert_eq!(loaded.version, 1);
    assert_eq!(loaded.source_pages.len(), 1);
    assert_eq!(
        loaded.source_pages[0].responses[0].body,
        br#"{"mal_id":999999010}"#
    );
    assert_eq!(loaded.partial_source_pages.len(), 1);
    assert_eq!(loaded.partial_source_pages[0].address, partial_address);
    assert_eq!(loaded.corrections.len(), 1);
    assert_eq!(loaded.corrections[0].field_path, "episode_count");

    let first_input = SaveAnimeInput {
        anime_id,
        expected_version: 1,
        title: "Monster".to_owned(),
        title_english: Some("Monster".to_owned()),
        title_japanese: Some("モンスター".to_owned()),
        anime_type: "TV".to_owned(),
        status: "ONGOING".to_owned(),
        rating: "R - 17+ (violence & profanity)".to_owned(),
        season: Some("SPRING".to_owned()),
        source_material: Some("MANGA".to_owned()),
        year: Some(2004),
        month: Some(4),
        episode_count: 74,
        duration_seconds: Some(1440),
        is_nsfw: Some(false),
        country_of_origin: Some("JP".to_owned()),
        synopsis: None,
        background: None,
        aired_from: None,
        aired_to: None,
        score_weighted: Some(8.88),
        score_mean: Some(8.80),
        score_median: Some(9.00),
        merged_record_hash: "hash-v1".to_owned(),
        hash_format: ANIME_HASH_FORMAT_V1,
        merge_version: 1,
        traceparent: Some("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01".to_owned()),
    };

    let first_save = save_anime(&pool, &first_input).await?;
    assert!(first_save.has_changed);
    assert_eq!(first_save.version, 2);

    // Merged work is now visible and merge_requested_at is cleared.
    let visible_after = sqlx::query_scalar!(
        "SELECT EXISTS(SELECT 1 FROM anime_visible WHERE id = $1)",
        anime_id
    )
    .fetch_one(&pool)
    .await?;
    assert_eq!(visible_after, Some(true));

    let merge_requested = sqlx::query_scalar!(
        "SELECT merge_requested_at FROM anime WHERE id = $1",
        anime_id
    )
    .fetch_one(&pool)
    .await?;
    assert!(merge_requested.is_none());

    // First save emits anime.created.
    let created_subjects: Vec<String> = sqlx::query_scalar!(
        "SELECT subject FROM outbox WHERE entity_id = $1 ORDER BY created_at",
        anime_id
    )
    .fetch_all(&pool)
    .await?;
    assert_eq!(created_subjects, vec!["anime.created"]);

    // Unchanged hash writes nothing and emits nothing.
    let noop_input = SaveAnimeInput {
        expected_version: 2,
        ..first_input.clone()
    };
    let noop_save = save_anime(&pool, &noop_input).await?;
    assert!(!noop_save.has_changed);
    assert_eq!(noop_save.version, 2);

    // Stale expected_version returns version conflict (ABORTED).
    let Err(conflict_err) = save_anime(&pool, &first_input).await else {
        panic!("stale expected_version must fail");
    };
    assert!(conflict_err.is_version_conflict());

    // Status change writes both anime.updated and notification.anime.status.changed.
    let status_change_input = SaveAnimeInput {
        expected_version: 2,
        status: "FINISHED".to_owned(),
        merged_record_hash: "hash-v2".to_owned(),
        ..first_input
    };
    let second_save = save_anime(&pool, &status_change_input).await?;
    assert!(second_save.has_changed);
    assert_eq!(second_save.version, 3);

    let all_subjects: Vec<String> = sqlx::query_scalar!(
        "SELECT subject FROM outbox WHERE entity_id = $1 ORDER BY id",
        anime_id
    )
    .fetch_all(&pool)
    .await?;
    assert_eq!(
        all_subjects,
        vec![
            "anime.created",
            "anime.updated",
            "notification.anime.status.changed"
        ]
    );

    sqlx::query!("DELETE FROM outbox WHERE entity_id = $1", anime_id)
        .execute(&pool)
        .await?;
    sqlx::query!("DELETE FROM anime WHERE id = $1", anime_id)
        .execute(&pool)
        .await?;
    sqlx::query!(
        "DELETE FROM provider_page WHERE provider = 'mal' AND page_kind = 'anime' AND address = $1",
        source_address
    )
    .execute(&pool)
    .await?;
    sqlx::query!(
        "DELETE FROM provider_page WHERE provider = 'anilist' AND page_kind = 'anime' AND address = $1",
        partial_address
    )
    .execute(&pool)
    .await?;

    Ok(())
}
