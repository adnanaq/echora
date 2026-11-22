//! Integration tests verifying `postgres_db` migrations, page observations, and saves.

use std::env;

use chrono::{DateTime, Duration, NaiveDate, NaiveTime, Utc};
use postgres_db::{
    ANIME_HASH_FORMAT, AnimeBroadcastInput, AnimeCompanyInput, AnimeContextualRankInput,
    AnimeDetailsInput, AnimeExternalSourceInput, AnimeHiatusInput, AnimeImageInput,
    AnimeRelationClaimInput, AnimeRelationInput, AnimeRelationsAndGraphInput,
    AnimeScheduleAndStatsInput, AnimeSourceMaterialInput, AnimeStatisticsInput,
    AnimeStreamingSourceInput, AnimeThemeInput, AnimeThemeSongInput, AnimeTitleInput,
    AnimeTrailerInput, CompanyXrefInput, EntityType, EpisodeRangeInput, Error, FetchOutcome,
    PageKey, PageObservation, PageResponse, Provider, SaveAnimeInput, SourceMaterialXrefInput,
    connect_pool, link_anime_source_page, load_anime, record_page_observation, run_migrations,
    save_anime,
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

    let duplicate_sqlx_err = sqlx::query("INSERT INTO anime_type VALUES ('TV')")
        .execute(&pool)
        .await
        .expect_err("duplicate lookup insert must fail");
    let duplicate_err = Error::from(duplicate_sqlx_err);
    assert!(duplicate_err.is_unique_violation());

    Ok(())
}

#[tokio::test]
async fn connect_pool_rejects_invalid_database_url() {
    let err =
        (connect_pool("not-a-postgres-url", 1).await).expect_err("invalid database URL must fail");
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
        "DELETE FROM provider_page WHERE provider = 'mal' AND entity_type = 'anime' AND address = $1",
        test_address
    )
    .execute(&pool)
    .await?;

    // Fetch times may not lie ahead of the database clock, so each timeline starts a week back.
    let t0 = Utc::now() - Duration::days(7);
    let page_key = PageKey::new(Provider::Mal, EntityType::Anime, test_address)?;
    let response = PageResponse {
        request_name: "main".to_owned(),
        content_type: "application/json".to_owned(),
        body: br#"{"id":999999001}"#.to_vec(),
    };

    let first = record_page_observation(
        &pool,
        PageObservation {
            page_key: page_key.clone(),
            observed_at: t0,
            outcome: FetchOutcome::Success {
                responses: vec![response.clone()],
            },
        },
    )
    .await?;
    assert!(first.has_changed);

    let same = record_page_observation(
        &pool,
        PageObservation {
            page_key: page_key.clone(),
            observed_at: t0 + Duration::minutes(5),
            outcome: FetchOutcome::Success {
                responses: vec![response.clone()],
            },
        },
    )
    .await?;
    assert!(!same.has_changed);

    let older = record_page_observation(
        &pool,
        PageObservation {
            page_key,
            observed_at: t0 - Duration::minutes(5),
            outcome: FetchOutcome::Success {
                responses: vec![PageResponse {
                    body: b"older".to_vec(),
                    ..response
                }],
            },
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
    // Fetch times may not lie ahead of the database clock, so each timeline starts a week back.
    let t0 = Utc::now() - Duration::days(7);
    let page_key = PageKey::new(Provider::Mal, EntityType::Anime, test_address)?;
    let expected_proposal = postgres_db::encode_page_key_proto(&page_key);

    sqlx::query!(
        "DELETE FROM provider_page WHERE provider = 'mal' AND entity_type = 'anime' AND address = $1",
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

    let failed = record_page_observation(
        &pool,
        PageObservation {
            page_key: page_key.clone(),
            observed_at: t0,
            outcome: FetchOutcome::Failed,
        },
    )
    .await?;
    assert!(!failed.has_changed);

    let failures = sqlx::query_scalar!(
        "SELECT fetch_failures FROM provider_page WHERE provider = 'mal' AND entity_type = 'anime' AND address = $1",
        test_address
    )
    .fetch_one(&pool)
    .await?;
    assert_eq!(failures, 1);

    record_page_observation(
        &pool,
        PageObservation {
            page_key: page_key.clone(),
            observed_at: t0 + Duration::minutes(20),
            outcome: FetchOutcome::NotFound {
                require_review: true,
            },
        },
    )
    .await?;
    record_page_observation(
        &pool,
        PageObservation {
            page_key: page_key.clone(),
            observed_at: t0 + Duration::days(2),
            outcome: FetchOutcome::NotFound {
                require_review: true,
            },
        },
    )
    .await?;
    record_page_observation(
        &pool,
        PageObservation {
            page_key: page_key.clone(),
            observed_at: t0 + Duration::days(2) + Duration::hours(1),
            outcome: FetchOutcome::NotFound {
                require_review: true,
            },
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

    let direct = record_page_observation(
        &pool,
        PageObservation {
            page_key: page_key.clone(),
            observed_at: t0 + Duration::days(3),
            outcome: FetchOutcome::NotFound {
                require_review: false,
            },
        },
    )
    .await?;
    assert!(direct.has_changed);

    let is_dead = sqlx::query_scalar!(
        "SELECT is_dead FROM provider_page WHERE provider = 'mal' AND entity_type = 'anime' AND address = $1",
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
    // Fetch times may not lie ahead of the database clock, so each timeline starts a week back.
    let t0 = Utc::now() - Duration::days(7);
    let page_key = PageKey::new(Provider::Mal, EntityType::Anime, test_address)?;
    let expected_proposal = postgres_db::encode_page_key_proto(&page_key);

    sqlx::query!(
        "DELETE FROM provider_page WHERE provider = 'mal' AND entity_type = 'anime' AND address = $1",
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

    record_page_observation(
        &pool,
        PageObservation {
            page_key: page_key.clone(),
            observed_at: t0,
            outcome: FetchOutcome::NotFound {
                require_review: true,
            },
        },
    )
    .await?;
    record_page_observation(
        &pool,
        PageObservation {
            page_key: page_key.clone(),
            observed_at: t0 + Duration::days(2),
            outcome: FetchOutcome::NotFound {
                require_review: true,
            },
        },
    )
    .await?;

    let recovered = record_page_observation(
        &pool,
        PageObservation {
            page_key,
            observed_at: t0 + Duration::days(3),
            outcome: FetchOutcome::Success {
                responses: vec![PageResponse {
                    request_name: "main".to_owned(),
                    content_type: "application/json".to_owned(),
                    body: br#"{"id":999999004}"#.to_vec(),
                }],
            },
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
        "DELETE FROM provider_page WHERE provider = 'mal' AND entity_type = 'anime' AND address = $1",
        test_address
    )
    .execute(&pool)
    .await?;

    // Fetch times may not lie ahead of the database clock, so each timeline starts a week back.
    let t0 = Utc::now() - Duration::days(7);
    let page_key = PageKey::new(Provider::Mal, EntityType::Anime, test_address)?;

    record_page_observation(
        &pool,
        PageObservation {
            page_key: page_key.clone(),
            observed_at: t0,
            outcome: FetchOutcome::NotFound {
                require_review: false,
            },
        },
    )
    .await?;

    record_page_observation(
        &pool,
        PageObservation {
            page_key: page_key.clone(),
            observed_at: t0 + Duration::days(2),
            outcome: FetchOutcome::NotFound {
                require_review: false,
            },
        },
    )
    .await?;

    record_page_observation(
        &pool,
        PageObservation {
            page_key: page_key.clone(),
            observed_at: t0 + Duration::days(2),
            outcome: FetchOutcome::NotFound {
                require_review: false,
            },
        },
    )
    .await?;

    let stale_fetch_outcome = record_page_observation(
        &pool,
        PageObservation {
            page_key,
            observed_at: t0 + Duration::days(1),
            outcome: FetchOutcome::Success {
                responses: vec![PageResponse {
                    request_name: "main".to_owned(),
                    content_type: "application/json".to_owned(),
                    body: br#"{"id":999999003}"#.to_vec(),
                }],
            },
        },
    )
    .await?;
    assert!(!stale_fetch_outcome.has_changed);

    let missing_answers = sqlx::query_scalar!(
        "SELECT missing_answers FROM provider_page WHERE provider = 'mal' AND entity_type = 'anime' AND address = $1",
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

    record_page_observation(
        &pool,
        PageObservation {
            page_key: PageKey::new(Provider::Mal, EntityType::Anime, source_address)?,
            observed_at: Utc::now(),
            outcome: FetchOutcome::Success {
                responses: vec![PageResponse {
                    request_name: "main".to_owned(),
                    content_type: "application/json".to_owned(),
                    body: br#"{"mal_id":999999010}"#.to_vec(),
                }],
            },
        },
    )
    .await?;

    record_page_observation(
        &pool,
        PageObservation {
            page_key: PageKey::new(Provider::AniList, EntityType::Anime, partial_address)?,
            observed_at: Utc::now(),
            outcome: FetchOutcome::Failed,
        },
    )
    .await?;

    sqlx::query!(
        "INSERT INTO anime (id, merge_requested_at) VALUES ($1, now())",
        anime_id
    )
    .execute(&pool)
    .await?;

    let visible_before = sqlx::query_scalar!(
        "SELECT EXISTS(SELECT 1 FROM anime_visible WHERE id = $1)",
        anime_id
    )
    .fetch_one(&pool)
    .await?;
    assert_eq!(visible_before, Some(false));

    sqlx::query!(
        "INSERT INTO anime_source (anime_id, provider, entity_type, address)
         VALUES ($1, 'mal', 'anime', $2)",
        anime_id,
        source_address,
    )
    .execute(&pool)
    .await?;
    sqlx::query!(
        "INSERT INTO anime_partial_source (anime_id, provider, entity_type, address)
         VALUES ($1, 'anilist', 'anime', $2)",
        anime_id,
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
        loaded_anime.partial_source_pages[0].address().as_str(),
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
        hash_format: ANIME_HASH_FORMAT,
        merge_version: 1,
        traceparent: Some("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01".to_owned()),
        details: AnimeDetailsInput::default(),
        schedule_and_stats: AnimeScheduleAndStatsInput::default(),
        relations_and_graph: AnimeRelationsAndGraphInput::default(),
        episodes: Vec::new(),
    };

    let first_save = save_anime(&pool, &first_input).await?;
    assert!(first_save.has_changed);
    assert_eq!(first_save.version, 2);

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

    let created_subjects: Vec<String> = sqlx::query_scalar!(
        "SELECT subject FROM outbox WHERE entity_id = $1 ORDER BY created_at",
        anime_id
    )
    .fetch_all(&pool)
    .await?;
    assert_eq!(created_subjects, vec!["anime.created"]);

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

    let conflict_err = save_anime(&pool, &first_input)
        .await
        .expect_err("stale expected_version must fail");
    assert!(conflict_err.is_version_conflict());

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
        "DELETE FROM provider_page WHERE provider = 'mal' AND entity_type = 'anime' AND address = $1",
        source_address
    )
    .execute(&pool)
    .await?;
    sqlx::query!(
        "DELETE FROM provider_page WHERE provider = 'anilist' AND entity_type = 'anime' AND address = $1",
        partial_address
    )
    .execute(&pool)
    .await?;

    Ok(())
}

#[tokio::test]
#[expect(
    clippy::too_many_lines,
    reason = "End-to-end concurrent page observation and stale save test"
)]
async fn store_page_bumps_linked_anime_version_and_blocks_stale_save() -> Result<(), Error> {
    let Ok(database_url) = env::var("DATABASE_URL") else {
        return Ok(());
    };

    let pool = connect_pool(&database_url, 2).await?;
    run_migrations(&pool).await?;

    let anime_id = Uuid::now_v7();
    let page_address = "999999012";
    let page_key = PageKey::new(Provider::Mal, EntityType::Anime, page_address)?;
    // Fetch times may not lie ahead of the database clock, so each timeline starts a week back.
    let t0 = Utc::now() - Duration::days(7);

    record_page_observation(
        &pool,
        PageObservation {
            page_key: page_key.clone(),
            observed_at: t0,
            outcome: FetchOutcome::Success {
                responses: vec![PageResponse {
                    request_name: "main".to_owned(),
                    content_type: "application/json".to_owned(),
                    body: br#"{"mal_id":999999012,"title":"V1"}"#.to_vec(),
                }],
            },
        },
    )
    .await?;

    sqlx::query!("INSERT INTO anime (id) VALUES ($1)", anime_id)
        .execute(&pool)
        .await?;
    sqlx::query!(
        "INSERT INTO anime_source (anime_id, provider, entity_type, address)
         VALUES ($1, 'mal', 'anime', $2)",
        anime_id,
        page_address
    )
    .execute(&pool)
    .await?;

    let loaded_anime = load_anime(&pool, anime_id).await?;
    assert_eq!(loaded_anime.version, 1);

    let updated_page = record_page_observation(
        &pool,
        PageObservation {
            page_key,
            observed_at: t0 + Duration::minutes(5),
            outcome: FetchOutcome::Success {
                responses: vec![PageResponse {
                    request_name: "main".to_owned(),
                    content_type: "application/json".to_owned(),
                    body: br#"{"mal_id":999999012,"title":"V2"}"#.to_vec(),
                }],
            },
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

    let stale_err = save_anime(
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
            duration_seconds: None,
            synopsis: None,
            background: None,
            source_material: None,
            country_of_origin: None,
            is_nsfw: Some(false),
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
            episodes: Vec::new(),
        },
    )
    .await
    .expect_err("stale save after concurrent store_page must fail");
    assert!(stale_err.is_version_conflict());

    sqlx::query!("DELETE FROM anime WHERE id = $1", anime_id)
        .execute(&pool)
        .await?;
    sqlx::query!(
        "DELETE FROM provider_page WHERE provider = 'mal' AND entity_type = 'anime' AND address = $1",
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
        episodes: Vec::new(),
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
        "SELECT theme FROM anime_theme WHERE anime_id = $1 ORDER BY theme",
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

    let hiatus_start = NaiveDate::from_ymd_opt(2024, 5, 1).expect("valid hiatus_start date");
    let hiatus_end = NaiveDate::from_ymd_opt(2024, 5, 15).expect("valid hiatus_end date");
    let jp_time = NaiveTime::from_hms_opt(0, 30, 0).expect("valid jp_time");

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
        merged_record_hash: "extended-v1".to_owned(),
        hash_format: ANIME_HASH_FORMAT,
        merge_version: 1,
        traceparent: None,
        details: AnimeDetailsInput::default(),
        schedule_and_stats: AnimeScheduleAndStatsInput {
            opening_themes: vec![AnimeThemeSongInput {
                position: 0,
                title: "Grain".to_owned(),
                artist: Some("Kuniaki Haishima".to_owned()),
                episode_ranges: vec![EpisodeRangeInput {
                    range_start: 1,
                    range_end: Some(74),
                }],
            }],
            ending_themes: Vec::new(),
            trailers: vec![AnimeTrailerInput {
                source: "https://youtube.com/watch?v=1".to_owned(),
                title: Some("PV 1".to_owned()),
                thumbnail_source: None,
            }],
            external_sources: vec![AnimeExternalSourceInput {
                source: "https://official.example.com".to_owned(),
                platform: "Official Site".to_owned(),
                label: Some("JP".to_owned()),
                language: Some("ja".to_owned()),
            }],
            streaming_sources: vec![AnimeStreamingSourceInput {
                platform: "Netflix".to_owned(),
                source: "https://netflix.com/title/1".to_owned(),
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
        episodes: Vec::new(),
    };

    let first_save = save_anime(&pool, &first_input).await?;
    assert!(first_save.has_changed);
    assert_eq!(first_save.version, 2);

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
         FROM anime_streaming_source_subtitle_language s
         JOIN anime_streaming_source st ON st.id = s.streaming_source_id
         WHERE st.anime_id = $1
         ORDER BY s.language",
        anime_id
    )
    .fetch_all(&pool)
    .await?;
    assert_eq!(subtitle_langs, vec!["en", "es"]);

    let range_end: Option<i32> = sqlx::query_scalar!(
        "SELECT r.range_end
         FROM anime_opening_theme_episode r
         JOIN anime_opening_theme ts ON ts.id = r.opening_theme_id
         WHERE ts.anime_id = $1",
        anime_id
    )
    .fetch_one(&pool)
    .await?;
    assert_eq!(range_end, Some(74));

    let contextual_rank: i32 = sqlx::query_scalar!(
        "SELECT rank FROM anime_statistics_contextual_rank WHERE anime_id = $1 AND provider = 'mal'",
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
    let unique_num = anime_a.as_u128() % 1_000_000_000;
    let target_address = format!("9{unique_num:09}");
    let manga_id_ext = format!("8{unique_num:09}");
    let company_url = format!("https://myanimelist.net/anime/producer/{target_address}/Madhouse");

    record_page_observation(
        &pool,
        PageObservation {
            page_key: PageKey::new(Provider::Mal, EntityType::Anime, target_address.clone())?,
            observed_at: Utc::now(),
            outcome: FetchOutcome::Success {
                responses: vec![PageResponse {
                    request_name: "main".to_owned(),
                    content_type: "application/json".to_owned(),
                    body: br#"{"mal_id":999999020}"#.to_vec(),
                }],
            },
        },
    )
    .await?;

    sqlx::query!("INSERT INTO anime (id) VALUES ($1), ($2)", anime_a, anime_b)
        .execute(&pool)
        .await?;

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
        duration_seconds: None,
        is_nsfw: Some(false),
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
                to_page_address: target_address.clone(),
            }],
            relation_claims: vec![AnimeRelationClaimInput {
                provider: Provider::Mal,
                relation_type: "SPIN_OFF".to_owned(),
                target_page: PageKey::new(
                    Provider::Mal,
                    EntityType::Anime,
                    target_address.clone(),
                )?,
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
                    external_id: manga_id_ext.clone(),
                    url: format!("https://myanimelist.net/manga/{manga_id_ext}"),
                }],
            }],
            companies: vec![AnimeCompanyInput {
                name: "Madhouse".to_owned(),
                description: None,
                roles: vec!["STUDIO".to_owned(), "PRODUCER".to_owned()],
                xrefs: vec![CompanyXrefInput {
                    provider: "mal".to_owned(),
                    url: company_url.clone(),
                }],
            }],
        },
        episodes: Vec::new(),
    };
    save_anime(&pool, &first_work_input).await?;

    let target_before_link: Option<Uuid> = sqlx::query_scalar!(
        "SELECT target.anime_id
         FROM anime_related_anime relation
         JOIN anime_source target
           ON target.provider = relation.to_page_provider
          AND target.address = relation.to_page_address
         WHERE relation.from_anime_id = $1",
        anime_a
    )
    .fetch_optional(&pool)
    .await?;
    assert!(target_before_link.is_none());

    let resolved_count = link_anime_source_page(
        &pool,
        anime_b,
        &PageKey::new(Provider::Mal, EntityType::Anime, target_address.clone())?,
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

    let company_reverse_edges: Vec<String> = sqlx::query_scalar!(
        "SELECT edge_type
         FROM graph_edge
         WHERE from_node_type = 'company' AND to_id = $1
         ORDER BY edge_type",
        anime_a
    )
    .fetch_all(&pool)
    .await?;
    assert_eq!(company_reverse_edges, vec!["PRODUCER", "STUDIO"]);

    let source_edge_count: i64 = sqlx::query_scalar!(
        "SELECT COUNT(*) FROM graph_edge WHERE from_id = $1 AND to_node_type = 'source_material' AND edge_type = 'ADAPTED_FROM'",
        anime_a
    )
    .fetch_one(&pool)
    .await?
    .unwrap_or_default();
    assert_eq!(source_edge_count, 1);

    let second_work_input = SaveAnimeInput {
        anime_id: anime_b,
        expected_version: 2,
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
                    url: company_url.clone(),
                }],
            }],
        },
        ..first_work_input
    };
    save_anime(&pool, &second_work_input).await?;

    let matching_companies: i64 = sqlx::query_scalar!(
        "SELECT COUNT(*) FROM company_xref WHERE provider = 'mal' AND source = $1",
        company_url
    )
    .fetch_one(&pool)
    .await?
    .unwrap_or_default();
    assert_eq!(matching_companies, 1);

    let first_work_company_id: Uuid = sqlx::query_scalar!(
        "SELECT company_id FROM anime_company WHERE anime_id = $1 AND role = 'STUDIO'",
        anime_a
    )
    .fetch_one(&pool)
    .await?;
    let second_work_company_id: Uuid = sqlx::query_scalar!(
        "SELECT company_id FROM anime_company WHERE anime_id = $1 AND role = 'STUDIO'",
        anime_b
    )
    .fetch_one(&pool)
    .await?;
    assert_eq!(first_work_company_id, second_work_company_id);

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
             SELECT company_id FROM company_xref WHERE provider = 'mal' AND source = $1
         )",
        company_url
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
        "DELETE FROM provider_page WHERE provider = 'mal' AND entity_type = 'anime' AND address = $1",
        target_address
    )
    .execute(&pool)
    .await?;

    Ok(())
}

#[expect(
    clippy::too_many_lines,
    reason = "End-to-end lifecycle test exercises SaveAnime episode sync, LoadEpisode, SaveEpisode, MarkEpisodeAired, and delete_episode together"
)]
#[tokio::test]
async fn save_anime_and_episode_lifecycle_load_save_mark_aired_and_delete()
-> Result<(), Box<dyn std::error::Error>> {
    use postgres_db::{
        EpisodeCharacterInput, EpisodeListEntryInput, EpisodeReferenceInput, EpisodeStreamingInput,
        EpisodeTitleInput, SaveEpisodeInput, delete_episode, load_episode, mark_episode_aired,
        save_episode,
    };

    let Ok(database_url) = env::var("DATABASE_URL") else {
        return Ok(());
    };
    let pool = connect_pool(&database_url, 2).await?;
    run_migrations(&pool).await?;

    let anime_id = Uuid::now_v7();
    let episode_id = Uuid::now_v7();
    let suffix = anime_id.as_u128() % 1_000_000_000;
    let episode_address = format!("7{suffix:09}");

    sqlx::query!("INSERT INTO anime (id) VALUES ($1)", anime_id)
        .execute(&pool)
        .await?;

    let episode_page_key =
        PageKey::new(Provider::Mal, EntityType::Episode, episode_address.clone())?;
    let initial_aired_at =
        DateTime::parse_from_rfc3339("2020-01-01T00:00:00Z")?.with_timezone(&Utc);

    let first_anime_save = SaveAnimeInput {
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
        merged_record_hash: "ep-work-v1".to_owned(),
        hash_format: ANIME_HASH_FORMAT,
        merge_version: 1,
        traceparent: None,
        details: AnimeDetailsInput::default(),
        schedule_and_stats: AnimeScheduleAndStatsInput::default(),
        relations_and_graph: AnimeRelationsAndGraphInput::default(),
        episodes: vec![EpisodeReferenceInput {
            episode_id,
            episode_number: 1,
            title: "Herr Dr. Tenma".to_owned(),
            page_keys: vec![episode_page_key.clone()],
            list_entries: vec![EpisodeListEntryInput {
                provider: Provider::Mal,
                title: Some("Herr Dr. Tenma".to_owned()),
                aired_at: Some(initial_aired_at),
                duration_seconds: Some(1440),
                is_filler: Some(false),
                is_recap: Some(false),
            }],
        }],
    };
    let created_outcome = save_anime(&pool, &first_anime_save).await?;
    assert_eq!(created_outcome.created_episode_ids, vec![episode_id]);
    assert_eq!(created_outcome.updated_episode_ids, Vec::<Uuid>::new());

    let visible_before_merge: Option<Uuid> = sqlx::query_scalar!(
        r#"SELECT id AS "id!" FROM episode_visible WHERE id = $1"#,
        episode_id
    )
    .fetch_optional(&pool)
    .await?;
    assert!(visible_before_merge.is_none());

    sqlx::query!(
        "UPDATE episode SET next_refresh_at = now() + interval '1 day' WHERE id = $1",
        episode_id
    )
    .execute(&pool)
    .await?;

    let second_anime_save = SaveAnimeInput {
        expected_version: 2,
        merged_record_hash: "ep-work-v2".to_owned(),
        episodes: vec![EpisodeReferenceInput {
            episode_id,
            episode_number: 1,
            title: "Herr Dr. Tenma".to_owned(),
            page_keys: vec![episode_page_key.clone()],
            list_entries: vec![EpisodeListEntryInput {
                provider: Provider::Mal,
                title: Some("Herr Dr. Tenma (Revised)".to_owned()),
                aired_at: Some(initial_aired_at),
                duration_seconds: Some(1445),
                is_filler: Some(false),
                is_recap: Some(false),
            }],
        }],
        ..first_anime_save
    };
    let changed_outcome = save_anime(&pool, &second_anime_save).await?;
    assert_eq!(changed_outcome.created_episode_ids, Vec::<Uuid>::new());
    assert_eq!(changed_outcome.updated_episode_ids, vec![episode_id]);

    let next_refresh_after_change: Option<DateTime<Utc>> = sqlx::query_scalar!(
        "SELECT next_refresh_at FROM episode WHERE id = $1",
        episode_id
    )
    .fetch_one(&pool)
    .await?;
    assert!(next_refresh_after_change.is_none());

    record_page_observation(
        &pool,
        PageObservation {
            page_key: episode_page_key.clone(),
            observed_at: Utc::now(),
            outcome: FetchOutcome::Success {
                responses: vec![PageResponse {
                    request_name: "episode".to_owned(),
                    content_type: "application/json".to_owned(),
                    body: br#"{"title":"Herr Dr. Tenma"}"#.to_vec(),
                }],
            },
        },
    )
    .await?;

    let loaded_episode = load_episode(&pool, episode_id).await?;
    assert_eq!(loaded_episode.version, 3);
    assert_eq!(loaded_episode.anime_id, anime_id);
    assert_eq!(loaded_episode.pages.len(), 1);
    assert_eq!(loaded_episode.list_entries.len(), 1);
    assert_eq!(
        loaded_episode.list_entries[0].title.as_deref(),
        Some("Herr Dr. Tenma (Revised)")
    );

    let episode_input = SaveEpisodeInput {
        episode_id,
        expected_version: 3,
        episode_number: 1,
        season_number: Some(1),
        title: "Herr Dr. Tenma".to_owned(),
        title_japanese: Some("テンマ先生".to_owned()),
        title_romaji: None,
        synopsis: Some("Dr. Tenma saves Johan.".to_owned()),
        description: None,
        aired_at: Some(initial_aired_at),
        duration_seconds: Some(1445),
        is_filler: false,
        is_recap: false,
        score: Some(8.75),
        titles: vec![EpisodeTitleInput {
            language: "en".to_owned(),
            title: "Herr Dr. Tenma".to_owned(),
        }],
        images: vec!["https://example.com/ep1.jpg".to_owned()],
        streaming: vec![EpisodeStreamingInput {
            platform: "crunchyroll".to_owned(),
            source: "https://crunchyroll.com/watch/1".to_owned(),
        }],
        characters: vec![EpisodeCharacterInput {
            character_page_provider: "mal".to_owned(),
            character_page_address: "42".to_owned(),
            role: "MAIN".to_owned(),
        }],
        merged_record_hash: "ep-hash-1".to_owned(),
        hash_format: postgres_db::EPISODE_HASH_FORMAT,
        merge_version: 1,
        traceparent: None,
    };

    let overflow_score_result = save_episode(
        &pool,
        &SaveEpisodeInput {
            score: Some(1000.0),
            ..episode_input.clone()
        },
    )
    .await;
    assert!(matches!(overflow_score_result, Err(err) if err.is_invalid_argument()));

    let stale_result = save_episode(
        &pool,
        &SaveEpisodeInput {
            expected_version: 2,
            ..episode_input.clone()
        },
    )
    .await;
    assert!(matches!(stale_result, Err(err) if err.is_version_conflict()));

    let first_ep_save = save_episode(&pool, &episode_input).await?;
    assert!(first_ep_save.has_changed);
    assert_eq!(first_ep_save.version, 4);

    let visible_after_merge: Option<Uuid> = sqlx::query_scalar!(
        r#"SELECT id AS "id!" FROM episode_visible WHERE id = $1"#,
        episode_id
    )
    .fetch_optional(&pool)
    .await?;
    assert_eq!(visible_after_merge, Some(episode_id));

    let unchanged_ep_save = save_episode(
        &pool,
        &SaveEpisodeInput {
            expected_version: 4,
            ..episode_input.clone()
        },
    )
    .await?;
    assert!(!unchanged_ep_save.has_changed);
    assert_eq!(unchanged_ep_save.version, 4);

    let updated_ep_save = save_episode(
        &pool,
        &SaveEpisodeInput {
            expected_version: 4,
            merged_record_hash: "ep-hash-2".to_owned(),
            synopsis: Some("Updated synopsis".to_owned()),
            ..episode_input
        },
    )
    .await?;
    assert!(updated_ep_save.has_changed);
    assert_eq!(updated_ep_save.version, 5);

    let first_aired = mark_episode_aired(&pool, episode_id, None).await?;
    assert!(first_aired.has_notified);
    let second_aired = mark_episode_aired(&pool, episode_id, None).await?;
    assert!(!second_aired.has_notified);

    let outbox_subjects: Vec<String> = sqlx::query_scalar!(
        "SELECT subject FROM outbox WHERE entity_id = $1 ORDER BY created_at, id",
        episode_id
    )
    .fetch_all(&pool)
    .await?;
    assert_eq!(
        outbox_subjects,
        vec![
            "episode.created".to_owned(),
            "episode.updated".to_owned(),
            "notification.episode.aired".to_owned(),
        ]
    );

    let deleted_version = delete_episode(&pool, episode_id, 5, None).await?;
    assert_eq!(deleted_version, 6);

    let remaining_page_links: i64 = sqlx::query_scalar!(
        "SELECT COUNT(*) FROM episode_source WHERE episode_id = $1",
        episode_id
    )
    .fetch_one(&pool)
    .await?
    .unwrap_or_default();
    assert_eq!(remaining_page_links, 0);

    let replacement_episode_id = Uuid::now_v7();
    sqlx::query!(
        "INSERT INTO episode (id, anime_id, episode_number, title) VALUES ($1, $2, 1, 'Replacement')",
        replacement_episode_id,
        anime_id
    )
    .execute(&pool)
    .await?;

    sqlx::query!(
        "DELETE FROM outbox WHERE entity_id IN ($1, $2)",
        anime_id,
        episode_id
    )
    .execute(&pool)
    .await?;
    sqlx::query!("DELETE FROM anime WHERE id = $1", anime_id)
        .execute(&pool)
        .await?;
    sqlx::query!(
        "DELETE FROM provider_page WHERE provider = 'mal' AND entity_type = 'episode' AND address = $1",
        episode_address
    )
    .execute(&pool)
    .await?;

    Ok(())
}

#[expect(
    clippy::too_many_lines,
    reason = "Integration test exercises timestamp precision, page replacement, and ownership conflict together"
)]
#[tokio::test]
async fn sync_work_episodes_validates_page_ownership_replacement_and_timestamp_precision()
-> Result<(), Box<dyn std::error::Error>> {
    use chrono::Timelike as _;
    use postgres_db::{EpisodeListEntryInput, EpisodeReferenceInput};

    let Ok(database_url) = env::var("DATABASE_URL") else {
        return Ok(());
    };
    let pool = connect_pool(&database_url, 2).await?;
    run_migrations(&pool).await?;

    let anime_id = Uuid::now_v7();
    let first_episode_id = Uuid::now_v7();
    let second_episode_id = Uuid::now_v7();
    let suffix = anime_id.as_u128() % 1_000_000_000;
    let first_address = format!("6{suffix:09}");
    let second_address = format!("5{suffix:09}");

    sqlx::query!("INSERT INTO anime (id) VALUES ($1)", anime_id)
        .execute(&pool)
        .await?;

    let aired_with_nanos = DateTime::parse_from_rfc3339("2021-02-03T04:05:06Z")?
        .with_timezone(&Utc)
        .with_nanosecond(123_456_789);

    let base_save = SaveAnimeInput {
        anime_id,
        expected_version: 1,
        title: "Precision Work".to_owned(),
        title_english: None,
        title_japanese: None,
        anime_type: "TV".to_owned(),
        status: "FINISHED".to_owned(),
        rating: "PG-13 - Teens 13 or older".to_owned(),
        season: None,
        source_material: None,
        year: Some(2021),
        month: Some(2),
        episode_count: 1,
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
        merged_record_hash: "prec-v1".to_owned(),
        hash_format: ANIME_HASH_FORMAT,
        merge_version: 1,
        traceparent: None,
        details: AnimeDetailsInput::default(),
        schedule_and_stats: AnimeScheduleAndStatsInput::default(),
        relations_and_graph: AnimeRelationsAndGraphInput::default(),
        episodes: vec![EpisodeReferenceInput {
            episode_id: first_episode_id,
            episode_number: 1,
            title: "Ep 1".to_owned(),
            page_keys: vec![PageKey::new(
                Provider::Mal,
                EntityType::Episode,
                first_address.clone(),
            )?],
            list_entries: vec![EpisodeListEntryInput {
                provider: Provider::Mal,
                title: Some("Ep 1".to_owned()),
                aired_at: aired_with_nanos,
                duration_seconds: Some(1440),
                is_filler: Some(false),
                is_recap: Some(false),
            }],
        }],
    };
    save_anime(&pool, &base_save).await?;

    let replace_page_save = SaveAnimeInput {
        expected_version: 2,
        merged_record_hash: "prec-v2".to_owned(),
        episodes: vec![EpisodeReferenceInput {
            episode_id: first_episode_id,
            episode_number: 1,
            title: "Ep 1".to_owned(),
            page_keys: vec![PageKey::new(
                Provider::Mal,
                EntityType::Episode,
                second_address.clone(),
            )?],
            list_entries: vec![EpisodeListEntryInput {
                provider: Provider::Mal,
                title: Some("Ep 1".to_owned()),
                aired_at: aired_with_nanos,
                duration_seconds: Some(1440),
                is_filler: Some(false),
                is_recap: Some(false),
            }],
        }],
        ..base_save.clone()
    };
    let replace_outcome = save_anime(&pool, &replace_page_save).await?;
    assert_eq!(replace_outcome.updated_episode_ids, Vec::<Uuid>::new());

    let replaced_episode_state = sqlx::query!(
        "SELECT version, merge_requested_at FROM episode WHERE id = $1",
        first_episode_id
    )
    .fetch_one(&pool)
    .await?;
    assert_eq!(replaced_episode_state.version, 2);
    assert!(replaced_episode_state.merge_requested_at.is_some());

    let conflict_save = SaveAnimeInput {
        expected_version: 3,
        merged_record_hash: "prec-v3".to_owned(),
        episodes: vec![EpisodeReferenceInput {
            episode_id: second_episode_id,
            episode_number: 2,
            title: "Ep 2".to_owned(),
            page_keys: vec![PageKey::new(
                Provider::Mal,
                EntityType::Episode,
                second_address.clone(),
            )?],
            list_entries: Vec::new(),
        }],
        ..base_save
    };
    let conflict_result = save_anime(&pool, &conflict_save).await;
    assert!(matches!(conflict_result, Err(err) if err.is_invalid_argument()));

    sqlx::query!("DELETE FROM outbox WHERE entity_id = $1", anime_id)
        .execute(&pool)
        .await?;
    sqlx::query!("DELETE FROM anime WHERE id = $1", anime_id)
        .execute(&pool)
        .await?;
    sqlx::query!(
        "DELETE FROM provider_page WHERE provider = 'mal' AND entity_type = 'episode' AND address IN ($1, $2)",
        first_address,
        second_address
    )
    .execute(&pool)
    .await?;

    Ok(())
}

/// PostgreSQL silently truncates identifiers to 63 bytes, so a name at that length was cut.
#[sqlx::test(migrations = "./migrations")]
async fn migrations_every_identifier_fits_without_truncation(
    pool: sqlx::PgPool,
) -> Result<(), Error> {
    let truncated_names: Vec<String> = sqlx::query_scalar!(
        r#"SELECT relname::text AS "name!" FROM pg_class
           WHERE relnamespace = 'public'::regnamespace AND octet_length(relname::text) >= 63
           UNION ALL
           SELECT conname::text FROM pg_constraint
           WHERE connamespace = 'public'::regnamespace AND octet_length(conname::text) >= 63"#
    )
    .fetch_all(&pool)
    .await?;
    assert_eq!(truncated_names, Vec::<String>::new());
    Ok(())
}
