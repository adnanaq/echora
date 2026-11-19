//! Integration tests verifying `postgres_db` migrations and `store_page`.

// Rust guideline compliant 2026-10-09

use std::env;

use chrono::{Duration, NaiveDate, NaiveTime, Utc};
use postgres_db::{
    ANIME_HASH_FORMAT, AnimeBroadcastInput, AnimeCompanyInput, AnimeContextualRankInput,
    AnimeDetailsInput, AnimeExternalLinkInput, AnimeHiatusInput, AnimeImageInput,
    AnimeRelationClaimInput, AnimeRelationInput, AnimeRelationsAndGraphInput,
    AnimeScheduleAndStatsInput, AnimeSourceMaterialInput, AnimeStatisticsInput,
    AnimeStreamingInput, AnimeThemeInput, AnimeThemeSongInput, AnimeTitleInput, AnimeTrailerInput,
    CompanyXrefInput, EpisodeRangeInput, Error, PageKeyInput, PageKind, PageResponseInput,
    SaveAnimeInput, SourceMaterialXrefInput, StorePageInput, connect_pool, link_anime_source_page,
    load_anime, run_migrations, save_anime, store_page,
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

    let loaded_anime = load_anime(&pool, anime_id).await?;
    assert_eq!(loaded_anime.version, 1);
    assert_eq!(loaded_anime.source_pages.len(), 1);
    assert_eq!(
        loaded_anime.source_pages[0].responses[0].body,
        br#"{"mal_id":999999010}"#
    );
    assert_eq!(loaded_anime.partial_source_pages.len(), 1);
    assert_eq!(
        loaded_anime.partial_source_pages[0].address,
        partial_address
    );
    assert_eq!(loaded_anime.corrections.len(), 1);
    assert_eq!(loaded_anime.corrections[0].field_path, "episode_count");

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
        duration: Some(1440),
        nsfw: Some(false),
        country_of_origin: Some("JP".to_owned()),
        synopsis: None,
        background: None,
        aired_from: None,
        aired_to: None,
        score_weighted: Some(8.88),
        score_mean: Some(8.80),
        score_median: Some(9.00),
        merged_record_hash: "hash-v1".to_owned(),
        hash_format: ANIME_HASH_FORMAT,
        merge_version: 1,
        traceparent: Some("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01".to_owned()),
        details: AnimeDetailsInput::default(),
        schedule_and_stats: AnimeScheduleAndStatsInput::default(),
        relations_and_graph: AnimeRelationsAndGraphInput::default(),
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

    // Unchanged hash clears merge_requested_at, keeps version, and emits nothing.
    sqlx::query!(
        "UPDATE anime SET merge_requested_at = now() WHERE id = $1",
        anime_id
    )
    .execute(&pool)
    .await?;

    let noop_input = SaveAnimeInput {
        expected_version: 2,
        ..first_input.clone()
    };
    let noop_save = save_anime(&pool, &noop_input).await?;
    assert!(!noop_save.has_changed);
    assert_eq!(noop_save.version, 2);

    let merge_requested_after_noop = sqlx::query_scalar!(
        "SELECT merge_requested_at FROM anime WHERE id = $1",
        anime_id
    )
    .fetch_one(&pool)
    .await?;
    assert!(merge_requested_after_noop.is_none());

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

#[tokio::test]
#[expect(
    clippy::too_many_lines,
    reason = "End-to-end concurrent store_page and stale save test"
)]
async fn store_page_bumps_linked_anime_version_and_blocks_stale_save() -> Result<(), Error> {
    let Ok(database_url) = env::var("DATABASE_URL") else {
        return Ok(());
    };

    let pool = connect_pool(&database_url, 2).await?;
    run_migrations(&pool).await?;

    let anime_id = Uuid::now_v7();
    let page_address = "999999012";
    let page_key = PageKeyInput {
        provider: "mal".to_owned(),
        page_kind: PageKind::Anime,
        address: page_address.to_owned(),
    };
    let t0 = Utc::now();

    store_page(
        &pool,
        StorePageInput {
            page_key: page_key.clone(),
            fetched_at: t0,
            is_dead: false,
            responses: vec![PageResponseInput {
                request_name: "main".to_owned(),
                content_type: "application/json".to_owned(),
                body: br#"{"mal_id":999999012,"title":"V1"}"#.to_vec(),
            }],
            is_not_found: false,
            dead_page_review: false,
        },
    )
    .await?;

    sqlx::query!("INSERT INTO anime (id) VALUES ($1)", anime_id)
        .execute(&pool)
        .await?;
    sqlx::query!(
        "INSERT INTO anime_provider_page (anime_id, provider, page_kind, address, role)
         VALUES ($1, 'mal', 'anime', $2, 'source')",
        anime_id,
        page_address
    )
    .execute(&pool)
    .await?;

    let loaded_anime = load_anime(&pool, anime_id).await?;
    assert_eq!(loaded_anime.version, 1);

    // A newer page store commits while the caller is merging V1.
    let updated_page = store_page(
        &pool,
        StorePageInput {
            page_key,
            fetched_at: t0 + Duration::minutes(5),
            is_dead: false,
            responses: vec![PageResponseInput {
                request_name: "main".to_owned(),
                content_type: "application/json".to_owned(),
                body: br#"{"mal_id":999999012,"title":"V2"}"#.to_vec(),
            }],
            is_not_found: false,
            dead_page_review: false,
        },
    )
    .await?;
    assert!(updated_page.has_changed);

    let (bumped_version, merge_requested_at): (i64, Option<chrono::DateTime<Utc>>) =
        sqlx::query_as("SELECT version, merge_requested_at FROM anime WHERE id = $1")
            .bind(anime_id)
            .fetch_one(&pool)
            .await?;
    assert_eq!(bumped_version, 2);
    assert!(merge_requested_at.is_some());

    // The delayed save from V1 (expected_version = 1) must fail with version conflict.
    let Err(stale_err) = save_anime(
        &pool,
        &SaveAnimeInput {
            anime_id,
            expected_version: loaded_anime.version,
            title: "V1".to_owned(),
            title_english: None,
            title_japanese: None,
            anime_type: "TV".to_owned(),
            status: "FINISHED".to_owned(),
            rating: "UNKNOWN".to_owned(),
            season: None,
            year: None,
            month: None,
            episode_count: 1,
            duration: None,
            synopsis: None,
            background: None,
            source_material: None,
            country_of_origin: None,
            nsfw: Some(false),
            aired_from: None,
            aired_to: None,
            score_weighted: None,
            score_mean: None,
            score_median: None,
            merged_record_hash: "hash-v1".to_owned(),
            hash_format: 1,
            merge_version: 1,
            traceparent: None,
            details: AnimeDetailsInput::default(),
            schedule_and_stats: AnimeScheduleAndStatsInput::default(),
            relations_and_graph: AnimeRelationsAndGraphInput::default(),
        },
    )
    .await
    else {
        panic!("stale save after concurrent store_page must fail");
    };
    assert!(stale_err.is_version_conflict());

    sqlx::query!("DELETE FROM anime WHERE id = $1", anime_id)
        .execute(&pool)
        .await?;
    sqlx::query!(
        "DELETE FROM provider_page WHERE provider = 'mal' AND page_kind = 'anime' AND address = $1",
        page_address
    )
    .execute(&pool)
    .await?;

    Ok(())
}

#[tokio::test]
#[expect(
    clippy::too_many_lines,
    reason = "End-to-end ECHO-100 test covering all eight detail tables and replacements"
)]
async fn save_anime_replaces_names_classification_themes_and_images() -> Result<(), Error> {
    let Ok(database_url) = env::var("DATABASE_URL") else {
        return Ok(());
    };

    let pool = connect_pool(&database_url, 2).await?;
    run_migrations(&pool).await?;

    let anime_id = Uuid::now_v7();
    // Seed a pre-ECHO-100 row with hash_format = 1 and the same hash to verify
    // bumping ANIME_HASH_FORMAT triggers a one-time detail backfill.
    sqlx::query!(
        "INSERT INTO anime (id, title, type, status, merged_at, merged_record_hash, hash_format)
         VALUES ($1, 'Monster', 'TV', 'FINISHED', now(), 'details-v1', 1)",
        anime_id
    )
    .execute(&pool)
    .await?;

    let first_input = SaveAnimeInput {
        anime_id,
        expected_version: 1,
        title: "Monster".to_owned(),
        title_english: None,
        title_japanese: None,
        anime_type: "TV".to_owned(),
        status: "FINISHED".to_owned(),
        rating: "R - 17+ (violence & profanity)".to_owned(),
        season: None,
        source_material: None,
        year: Some(2004),
        month: Some(4),
        episode_count: 74,
        duration: Some(1440),
        nsfw: Some(false),
        country_of_origin: Some("JP".to_owned()),
        synopsis: None,
        background: None,
        aired_from: None,
        aired_to: None,
        score_weighted: None,
        score_mean: None,
        score_median: None,
        merged_record_hash: "details-v1".to_owned(),
        hash_format: ANIME_HASH_FORMAT,
        merge_version: 1,
        traceparent: None,
        details: AnimeDetailsInput {
            titles: vec![
                AnimeTitleInput {
                    language: "de".to_owned(),
                    title: "Monster DE".to_owned(),
                },
                AnimeTitleInput {
                    language: "fr".to_owned(),
                    title: "Monster FR".to_owned(),
                },
            ],
            synonyms: vec!["MONSTER".to_owned(), "MONSTER".to_owned()],
            genres: vec![
                "Thriller".to_owned(),
                "Drama".to_owned(),
                "Drama".to_owned(),
            ],
            tags: vec!["Crime".to_owned()],
            demographics: vec!["Seinen".to_owned()],
            content_warnings: vec!["Violence".to_owned()],
            themes: vec![AnimeThemeInput {
                name: "Psychological".to_owned(),
                description: Some("Cat and mouse".to_owned()),
            }],
            images: vec![
                AnimeImageInput {
                    image_type: "cover".to_owned(),
                    url: "https://example.com/cover.jpg".to_owned(),
                },
                AnimeImageInput {
                    image_type: "poster".to_owned(),
                    url: "https://example.com/poster.jpg".to_owned(),
                },
            ],
        },
        schedule_and_stats: AnimeScheduleAndStatsInput::default(),
        relations_and_graph: AnimeRelationsAndGraphInput::default(),
    };

    let first_save = save_anime(&pool, &first_input).await?;
    assert!(first_save.has_changed);
    assert_eq!(first_save.version, 2);

    let genre_matches: Vec<Uuid> = sqlx::query_scalar!(
        "SELECT anime_id FROM anime_genre WHERE genre = 'Drama' AND anime_id = $1",
        anime_id
    )
    .fetch_all(&pool)
    .await?;
    assert_eq!(genre_matches, vec![anime_id]);

    let tag_matches: Vec<Uuid> = sqlx::query_scalar!(
        "SELECT anime_id FROM anime_tag WHERE tag = 'Crime' AND anime_id = $1",
        anime_id
    )
    .fetch_all(&pool)
    .await?;
    assert_eq!(tag_matches, vec![anime_id]);

    let demographic_matches: Vec<Uuid> = sqlx::query_scalar!(
        "SELECT anime_id FROM anime_demographic WHERE demographic = 'Seinen' AND anime_id = $1",
        anime_id
    )
    .fetch_all(&pool)
    .await?;
    assert_eq!(demographic_matches, vec![anime_id]);

    let warning_matches: Vec<Uuid> = sqlx::query_scalar!(
        "SELECT anime_id FROM anime_content_warning WHERE content_warning = 'Violence' AND anime_id = $1",
        anime_id
    )
    .fetch_all(&pool)
    .await?;
    assert_eq!(warning_matches, vec![anime_id]);

    // Second save replaces all eight detail tables with updated rows.
    let second_input = SaveAnimeInput {
        expected_version: 2,
        merged_record_hash: "details-v2".to_owned(),
        details: AnimeDetailsInput {
            titles: vec![AnimeTitleInput {
                language: "es".to_owned(),
                title: "Monster ES".to_owned(),
            }],
            synonyms: vec!["Dr. Tenma".to_owned()],
            genres: vec!["Mystery".to_owned()],
            tags: vec!["Police".to_owned()],
            demographics: Vec::new(),
            content_warnings: Vec::new(),
            themes: vec![AnimeThemeInput {
                name: "Medical".to_owned(),
                description: None,
            }],
            images: vec![AnimeImageInput {
                image_type: "banner".to_owned(),
                url: "https://example.com/banner.jpg".to_owned(),
            }],
        },
        ..first_input
    };

    let second_save = save_anime(&pool, &second_input).await?;
    assert!(second_save.has_changed);
    assert_eq!(second_save.version, 3);

    let title_langs: Vec<String> = sqlx::query_scalar!(
        "SELECT language FROM anime_title WHERE anime_id = $1 ORDER BY language",
        anime_id
    )
    .fetch_all(&pool)
    .await?;
    assert_eq!(title_langs, vec!["es"]);

    let synonyms: Vec<String> = sqlx::query_scalar!(
        "SELECT synonym FROM anime_synonym WHERE anime_id = $1 ORDER BY synonym",
        anime_id
    )
    .fetch_all(&pool)
    .await?;
    assert_eq!(synonyms, vec!["Dr. Tenma"]);

    let genres: Vec<String> = sqlx::query_scalar!(
        "SELECT genre FROM anime_genre WHERE anime_id = $1 ORDER BY genre",
        anime_id
    )
    .fetch_all(&pool)
    .await?;
    assert_eq!(genres, vec!["Mystery"]);

    let tags: Vec<String> = sqlx::query_scalar!(
        "SELECT tag FROM anime_tag WHERE anime_id = $1 ORDER BY tag",
        anime_id
    )
    .fetch_all(&pool)
    .await?;
    assert_eq!(tags, vec!["Police"]);

    let demographics_count: i64 = sqlx::query_scalar!(
        "SELECT COUNT(*) FROM anime_demographic WHERE anime_id = $1",
        anime_id
    )
    .fetch_one(&pool)
    .await?
    .unwrap_or_default();
    assert_eq!(demographics_count, 0);

    let warnings_count: i64 = sqlx::query_scalar!(
        "SELECT COUNT(*) FROM anime_content_warning WHERE anime_id = $1",
        anime_id
    )
    .fetch_one(&pool)
    .await?
    .unwrap_or_default();
    assert_eq!(warnings_count, 0);

    let theme_names: Vec<String> = sqlx::query_scalar!(
        "SELECT name FROM anime_theme WHERE anime_id = $1 ORDER BY name",
        anime_id
    )
    .fetch_all(&pool)
    .await?;
    assert_eq!(theme_names, vec!["Medical"]);

    let image_types: Vec<String> = sqlx::query_scalar!(
        "SELECT image_type FROM anime_image WHERE anime_id = $1 ORDER BY image_type",
        anime_id
    )
    .fetch_all(&pool)
    .await?;
    assert_eq!(image_types, vec!["banner"]);

    sqlx::query!("DELETE FROM outbox WHERE entity_id = $1", anime_id)
        .execute(&pool)
        .await?;
    sqlx::query!("DELETE FROM anime WHERE id = $1", anime_id)
        .execute(&pool)
        .await?;

    Ok(())
}

#[tokio::test]
#[expect(
    clippy::too_many_lines,
    reason = "End-to-end ECHO-101 test covering songs, links, streaming, stats, broadcast, and hiatus history"
)]
async fn save_anime_stores_songs_links_streaming_stats_broadcast_and_hiatus() -> Result<(), Error> {
    let Ok(database_url) = env::var("DATABASE_URL") else {
        return Ok(());
    };

    let pool = connect_pool(&database_url, 2).await?;
    run_migrations(&pool).await?;

    let anime_id = Uuid::now_v7();
    sqlx::query!("INSERT INTO anime (id) VALUES ($1)", anime_id)
        .execute(&pool)
        .await?;

    let Some(hiatus_start) = NaiveDate::from_ymd_opt(2024, 5, 1) else {
        panic!("valid hiatus_start date");
    };
    let Some(hiatus_end) = NaiveDate::from_ymd_opt(2024, 5, 15) else {
        panic!("valid hiatus_end date");
    };
    let Some(jp_time) = NaiveTime::from_hms_opt(0, 30, 0) else {
        panic!("valid jp_time");
    };

    let first_input = SaveAnimeInput {
        anime_id,
        expected_version: 1,
        title: "Monster".to_owned(),
        title_english: None,
        title_japanese: None,
        anime_type: "TV".to_owned(),
        status: "ONGOING".to_owned(),
        rating: "R - 17+ (violence & profanity)".to_owned(),
        season: Some("SPRING".to_owned()),
        source_material: Some("MANGA".to_owned()),
        year: Some(2004),
        month: Some(4),
        episode_count: 74,
        duration: Some(1440),
        nsfw: Some(false),
        country_of_origin: Some("JP".to_owned()),
        synopsis: None,
        background: None,
        aired_from: None,
        aired_to: None,
        score_weighted: None,
        score_mean: None,
        score_median: None,
        merged_record_hash: "extended-v1".to_owned(),
        hash_format: ANIME_HASH_FORMAT,
        merge_version: 1,
        traceparent: None,
        details: AnimeDetailsInput::default(),
        schedule_and_stats: AnimeScheduleAndStatsInput {
            theme_songs: vec![AnimeThemeSongInput {
                kind: "opening".to_owned(),
                position: 0,
                title: "Grain".to_owned(),
                artist: Some("Kuniaki Haishima".to_owned()),
                episode_ranges: vec![EpisodeRangeInput {
                    range_start: 1,
                    range_end: Some(74),
                }],
            }],
            trailers: vec![AnimeTrailerInput {
                url: "https://youtube.com/watch?v=1".to_owned(),
                title: Some("PV 1".to_owned()),
                thumbnail_url: None,
            }],
            external_links: vec![AnimeExternalLinkInput {
                url: "https://official.example.com".to_owned(),
                platform: "Official Site".to_owned(),
                label: Some("JP".to_owned()),
                language: Some("ja".to_owned()),
            }],
            streaming_entries: vec![AnimeStreamingInput {
                platform: "Netflix".to_owned(),
                url: "https://netflix.com/title/1".to_owned(),
                region: Some("US".to_owned()),
                is_free: Some(false),
                is_premium_required: Some(true),
                has_dub: Some(true),
                subtitle_languages: vec!["en".to_owned(), "es".to_owned()],
            }],
            statistics: vec![AnimeStatisticsInput {
                provider: "mal".to_owned(),
                score: Some(8.88),
                scored_by: Some(400_000),
                members: Some(1_100_000),
                favorites: Some(50_000),
                rank: Some(25),
                popularity: Some(120),
                contextual_ranks: vec![AnimeContextualRankInput {
                    context: "highest rated".to_owned(),
                    rank: 1,
                    is_all_time: Some(true),
                    format: Some("TV".to_owned()),
                    season: Some("SPRING".to_owned()),
                    year: Some(2004),
                }],
            }],
            broadcast: Some(AnimeBroadcastInput {
                weekday: Some(3),
                timezone: Some("Asia/Tokyo".to_owned()),
                jp_time: Some(jp_time),
                sub_time: None,
                dub_time: None,
                sub_delay_days: Some(0),
                dub_delay_days: None,
                premiere_jp: None,
                premiere_sub: None,
                premiere_dub: None,
                next_episode_at: None,
                next_episode_number: Some(2),
            }),
            hiatus: Some(AnimeHiatusInput {
                reason: Some("Golden Week".to_owned()),
                hiatus_from: Some(hiatus_start),
                hiatus_until: None,
            }),
        },
        relations_and_graph: AnimeRelationsAndGraphInput::default(),
    };

    let first_save = save_anime(&pool, &first_input).await?;
    assert!(first_save.has_changed);
    assert_eq!(first_save.version, 2);

    // Second save with identical hiatus snapshot must NOT add a second hiatus row.
    let second_input = SaveAnimeInput {
        expected_version: 2,
        merged_record_hash: "extended-v2".to_owned(),
        ..first_input.clone()
    };
    let second_save = save_anime(&pool, &second_input).await?;
    assert_eq!(second_save.version, 3);

    let hiatus_count_after_same: i64 = sqlx::query_scalar!(
        "SELECT COUNT(*) FROM anime_hiatus WHERE anime_id = $1",
        anime_id
    )
    .fetch_one(&pool)
    .await?
    .unwrap_or_default();
    assert_eq!(hiatus_count_after_same, 1);

    // Third save with a changed hiatus snapshot appends a second history row.
    let mut third_schedule = first_input.schedule_and_stats.clone();
    third_schedule.hiatus = Some(AnimeHiatusInput {
        reason: Some("Golden Week".to_owned()),
        hiatus_from: Some(hiatus_start),
        hiatus_until: Some(hiatus_end),
    });
    third_schedule.broadcast = None;
    let third_input = SaveAnimeInput {
        expected_version: 3,
        merged_record_hash: "extended-v3".to_owned(),
        schedule_and_stats: third_schedule,
        ..first_input
    };
    let third_save = save_anime(&pool, &third_input).await?;
    assert_eq!(third_save.version, 4);

    let hiatus_count_after_change: i64 = sqlx::query_scalar!(
        "SELECT COUNT(*) FROM anime_hiatus WHERE anime_id = $1",
        anime_id
    )
    .fetch_one(&pool)
    .await?
    .unwrap_or_default();
    assert_eq!(hiatus_count_after_change, 2);

    let broadcast_count: i64 = sqlx::query_scalar!(
        "SELECT COUNT(*) FROM anime_broadcast WHERE anime_id = $1",
        anime_id
    )
    .fetch_one(&pool)
    .await?
    .unwrap_or_default();
    assert_eq!(broadcast_count, 0);

    let subtitle_langs: Vec<String> = sqlx::query_scalar!(
        "SELECT s.language
         FROM anime_streaming_subtitle_language s
         JOIN anime_streaming st ON st.id = s.streaming_id
         WHERE st.anime_id = $1
         ORDER BY s.language",
        anime_id
    )
    .fetch_all(&pool)
    .await?;
    assert_eq!(subtitle_langs, vec!["en", "es"]);

    let range_end: Option<i32> = sqlx::query_scalar!(
        "SELECT r.range_end
         FROM theme_song_episode_range r
         JOIN anime_theme_song ts ON ts.id = r.theme_song_id
         WHERE ts.anime_id = $1",
        anime_id
    )
    .fetch_one(&pool)
    .await?;
    assert_eq!(range_end, Some(74));

    let contextual_rank: i32 = sqlx::query_scalar!(
        "SELECT rank FROM anime_contextual_rank WHERE anime_id = $1 AND provider = 'mal'",
        anime_id
    )
    .fetch_one(&pool)
    .await?;
    assert_eq!(contextual_rank, 1);

    sqlx::query!("DELETE FROM outbox WHERE entity_id = $1", anime_id)
        .execute(&pool)
        .await?;
    sqlx::query!("DELETE FROM anime WHERE id = $1", anime_id)
        .execute(&pool)
        .await?;

    Ok(())
}

#[tokio::test]
#[expect(
    clippy::too_many_lines,
    reason = "End-to-end ECHO-102 test covering relations, pending resolution, claims, source material, companies, and graph_edge"
)]
async fn save_anime_and_link_page_persist_relations_xrefs_and_graph_edges() -> Result<(), Error> {
    let Ok(database_url) = env::var("DATABASE_URL") else {
        return Ok(());
    };

    let pool = connect_pool(&database_url, 2).await?;
    run_migrations(&pool).await?;

    let anime_a = Uuid::now_v7();
    let anime_b = Uuid::now_v7();
    let target_address = "999999020";
    let manga_id_ext = "999999021";

    store_page(
        &pool,
        StorePageInput {
            page_key: PageKeyInput {
                provider: "mal".to_owned(),
                page_kind: PageKind::Anime,
                address: target_address.to_owned(),
            },
            fetched_at: Utc::now(),
            is_dead: false,
            responses: vec![PageResponseInput {
                request_name: "main".to_owned(),
                content_type: "application/json".to_owned(),
                body: br#"{"mal_id":999999020}"#.to_vec(),
            }],
            is_not_found: false,
            dead_page_review: false,
        },
    )
    .await?;

    sqlx::query!("INSERT INTO anime (id) VALUES ($1), ($2)", anime_a, anime_b)
        .execute(&pool)
        .await?;

    // Save anime_a with an unresolved relation to mal:anime:999999020, claims,
    // source material, and a company with two roles.
    let first_work_input = SaveAnimeInput {
        anime_id: anime_a,
        expected_version: 1,
        title: "Work A".to_owned(),
        title_english: None,
        title_japanese: None,
        anime_type: "TV".to_owned(),
        status: "FINISHED".to_owned(),
        rating: "UNKNOWN".to_owned(),
        season: None,
        source_material: Some("MANGA".to_owned()),
        year: Some(2004),
        month: Some(4),
        episode_count: 12,
        duration: None,
        nsfw: Some(false),
        country_of_origin: Some("JP".to_owned()),
        synopsis: None,
        background: None,
        aired_from: None,
        aired_to: None,
        score_weighted: None,
        score_mean: None,
        score_median: None,
        merged_record_hash: "rel-a-v1".to_owned(),
        hash_format: ANIME_HASH_FORMAT,
        merge_version: 1,
        traceparent: None,
        details: AnimeDetailsInput::default(),
        schedule_and_stats: AnimeScheduleAndStatsInput::default(),
        relations_and_graph: AnimeRelationsAndGraphInput {
            relations: vec![AnimeRelationInput {
                relation_type: "SPIN_OFF".to_owned(),
                to_page_provider: "mal".to_owned(),
                to_page_address: target_address.to_owned(),
            }],
            relation_claims: vec![AnimeRelationClaimInput {
                provider: "mal".to_owned(),
                relation_type: "SPIN_OFF".to_owned(),
                target_page: PageKeyInput {
                    provider: "mal".to_owned(),
                    page_kind: PageKind::Anime,
                    address: target_address.to_owned(),
                },
            }],
            source_materials: vec![AnimeSourceMaterialInput {
                relation_type: "ADAPTATION".to_owned(),
                title: "Source Manga".to_owned(),
                title_english: None,
                title_japanese: None,
                source_material_type: "MANGA".to_owned(),
                status: Some("FINISHED".to_owned()),
                score: Some(8.5),
                chapters: Some(100),
                volumes: Some(10),
                images: vec!["https://example.com/manga.jpg".to_owned()],
                xrefs: vec![SourceMaterialXrefInput {
                    provider: "mal".to_owned(),
                    external_id: manga_id_ext.to_owned(),
                    url: format!("https://myanimelist.net/manga/{manga_id_ext}"),
                }],
            }],
            companies: vec![AnimeCompanyInput {
                name: "Madhouse".to_owned(),
                description: None,
                roles: vec!["STUDIO".to_owned(), "PRODUCER".to_owned()],
                xrefs: vec![CompanyXrefInput {
                    provider: "mal".to_owned(),
                    url: "https://myanimelist.net/anime/producer/999999020/Madhouse".to_owned(),
                }],
            }],
        },
    };
    save_anime(&pool, &first_work_input).await?;

    // Unresolved relation has to_anime_id IS NULL until target page is linked.
    let unresolved_target: Option<Uuid> = sqlx::query_scalar!(
        "SELECT to_anime_id FROM anime_relation WHERE from_anime_id = $1",
        anime_a
    )
    .fetch_one(&pool)
    .await?;
    assert!(unresolved_target.is_none());

    // Linking mal:anime:999999020 to anime_b resolves the pending relation and
    // populates both directions in graph_edge (SPIN_OFF + PARENT_STORY).
    let resolved_count = link_anime_source_page(
        &pool,
        anime_b,
        &PageKeyInput {
            provider: "mal".to_owned(),
            page_kind: PageKind::Anime,
            address: target_address.to_owned(),
        },
    )
    .await?;
    assert_eq!(resolved_count, 1);

    let ab_edge: String = sqlx::query_scalar!(
        "SELECT edge_type FROM graph_edge WHERE from_id = $1 AND to_id = $2",
        anime_a,
        anime_b
    )
    .fetch_one(&pool)
    .await?;
    assert_eq!(ab_edge, "SPIN_OFF");

    let ba_edge: String = sqlx::query_scalar!(
        "SELECT edge_type FROM graph_edge WHERE from_id = $1 AND to_id = $2",
        anime_b,
        anime_a
    )
    .fetch_one(&pool)
    .await?;
    assert_eq!(ba_edge, "PARENT_STORY");

    // Company reverse edges use each role as edge_type (2 rows for 2 roles).
    let company_reverse_edges: Vec<String> = sqlx::query_scalar!(
        "SELECT edge_type
         FROM graph_edge
         WHERE from_type = 'company' AND to_id = $1
         ORDER BY edge_type",
        anime_a
    )
    .fetch_all(&pool)
    .await?;
    assert_eq!(company_reverse_edges, vec!["PRODUCER", "STUDIO"]);

    // Source material edges exist in both directions.
    let source_edge_count: i64 = sqlx::query_scalar!(
        "SELECT COUNT(*) FROM graph_edge WHERE from_id = $1 AND to_type = 'source_material' AND edge_type = 'ADAPTED_FROM'",
        anime_a
    )
    .fetch_one(&pool)
    .await?
    .unwrap_or_default();
    assert_eq!(source_edge_count, 1);

    // Save anime_b sharing the same company xref and verify company row is reused.
    let second_work_input = SaveAnimeInput {
        anime_id: anime_b,
        expected_version: 1,
        title: "Work B".to_owned(),
        merged_record_hash: "rel-b-v1".to_owned(),
        relations_and_graph: AnimeRelationsAndGraphInput {
            relations: Vec::new(),
            relation_claims: Vec::new(),
            source_materials: Vec::new(),
            companies: vec![AnimeCompanyInput {
                name: "Madhouse".to_owned(),
                description: None,
                roles: vec!["STUDIO".to_owned()],
                xrefs: vec![CompanyXrefInput {
                    provider: "mal".to_owned(),
                    url: "https://myanimelist.net/anime/producer/999999020/Madhouse".to_owned(),
                }],
            }],
        },
        ..first_work_input
    };
    save_anime(&pool, &second_work_input).await?;

    let matching_companies: i64 = sqlx::query_scalar!(
        "SELECT COUNT(*) FROM company_xref WHERE provider = 'mal' AND url = 'https://myanimelist.net/anime/producer/999999020/Madhouse'"
    )
    .fetch_one(&pool)
    .await?
    .unwrap_or_default();
    assert_eq!(matching_companies, 1);

    sqlx::query!(
        "DELETE FROM graph_edge WHERE from_id IN ($1, $2) OR to_id IN ($1, $2)",
        anime_a,
        anime_b
    )
    .execute(&pool)
    .await?;
    sqlx::query!(
        "DELETE FROM outbox WHERE entity_id IN ($1, $2)",
        anime_a,
        anime_b
    )
    .execute(&pool)
    .await?;
    sqlx::query!("DELETE FROM anime WHERE id IN ($1, $2)", anime_a, anime_b)
        .execute(&pool)
        .await?;
    sqlx::query!(
        "DELETE FROM company WHERE id IN (
             SELECT company_id FROM company_xref WHERE provider = 'mal' AND url = 'https://myanimelist.net/anime/producer/999999020/Madhouse'
         )"
    )
    .execute(&pool)
    .await?;
    sqlx::query!(
        "DELETE FROM source_material WHERE id IN (
             SELECT source_material_id FROM source_material_xref WHERE provider = 'mal' AND external_id = $1
         )",
        manga_id_ext
    )
    .execute(&pool)
    .await?;
    sqlx::query!(
        "DELETE FROM provider_page WHERE provider = 'mal' AND page_kind = 'anime' AND address = $1",
        target_address
    )
    .execute(&pool)
    .await?;

    Ok(())
}
