//! Integration tests for provider page observations against PostgreSQL.

#![expect(
    clippy::expect_used,
    reason = "a failed setup step stops the test, as a test's own expect does"
)]

mod support;

use chrono::{Duration, Utc};
use postgres_db::{
    AnimeRelationInput, AnimeRelationsAndGraphInput, EntityType, FetchOutcome, PageKey,
    PageObservation, PageResponse, PageSource, Provider, record_page_observation, save_anime,
};
use sqlx::PgPool;
use url::Url;
use uuid::Uuid;

use crate::support::{
    CapturedLogs, build_anime_save, build_mal_page_source, insert_unmerged_anime,
    insert_unmerged_character, link_character_page, link_mal_anime_pages,
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
    let page_key =
        PageKey::new(Provider::Mal, EntityType::Episode, "21:7").expect("valid page key");
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
        "INSERT INTO provider_page (provider, entity_type, address) VALUES ('mal', 'episode', '21:7')",
    )
    .execute(&pool)
    .await
    .expect("insert placeholder page");
    sqlx::query(
        "INSERT INTO episode_source (episode_id, provider, entity_type, address)
         VALUES ($1, 'mal', 'episode', '21:7')",
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
async fn record_page_observation_changed_or_first_fetched_character_page_requests_merge_of_owning_character(
    pool: PgPool,
) {
    let page_key =
        PageKey::new(Provider::Mal, EntityType::Character, "40").expect("valid page key");
    let owning_character_id = insert_unmerged_character(&pool, "Monkey D. Luffy").await;
    let other_character_id = insert_unmerged_character(&pool, "Roronoa Zoro").await;
    let unfetched_page_owner_id = insert_unmerged_character(&pool, "Nami").await;
    link_character_page(&pool, unfetched_page_owner_id, "mal", "723").await;
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

    let first_fetched_page = record_page_observation(
        &pool,
        PageObservation {
            page_key: PageKey::new(Provider::Mal, EntityType::Character, "723")
                .expect("valid page key"),
            observed_at: Utc::now(),
            outcome: build_success(br#"{"id":723}"#),
        },
    )
    .await
    .expect("store the first fetch of the linked page");

    assert!(!unchanged_page.has_changed);
    assert!(changed_page.has_changed);
    assert!(first_fetched_page.has_changed);
    let character_states: Vec<(i64, bool)> = sqlx::query_as(
        "SELECT version, merge_requested_at IS NOT NULL FROM character WHERE id = ANY($1)
         ORDER BY array_position($1, id)",
    )
    .bind(vec![
        owning_character_id,
        other_character_id,
        unfetched_page_owner_id,
    ])
    .fetch_all(&pool)
    .await
    .expect("read the characters");
    assert_eq!(character_states, vec![(2, true), (1, false), (2, true)]);
}

async fn read_page_source(pool: &PgPool, address: &str) -> Option<String> {
    sqlx::query_scalar(
        "SELECT source FROM provider_page
         WHERE provider = 'mal' AND entity_type = 'anime' AND address = $1",
    )
    .bind(address)
    .fetch_one(pool)
    .await
    .expect("read page source")
}

async fn read_page_title(pool: &PgPool, address: &str) -> Option<String> {
    sqlx::query_scalar(
        "SELECT title FROM provider_page
         WHERE provider = 'mal' AND entity_type = 'anime' AND address = $1",
    )
    .bind(address)
    .fetch_one(pool)
    .await
    .expect("read page title")
}

async fn has_page_row(pool: &PgPool, address: &str) -> bool {
    sqlx::query_scalar(
        "SELECT EXISTS (
             SELECT 1 FROM provider_page
             WHERE provider = 'mal' AND entity_type = 'anime' AND address = $1
         )",
    )
    .bind(address)
    .fetch_one(pool)
    .await
    .expect("check page row")
}

fn build_relation_to(
    relation_type: &str,
    address: &str,
    source: Option<&str>,
    title: Option<&str>,
) -> AnimeRelationInput {
    AnimeRelationInput {
        relation_type: relation_type.to_owned(),
        target_page_source: PageSource {
            page_key: PageKey::new(Provider::Mal, EntityType::Anime, address)
                .expect("valid page key"),
            source: source.map(|url| Url::parse(url).expect("valid URL")),
            title: title.map(str::to_owned),
        },
    }
}

fn build_relations(relations: Vec<AnimeRelationInput>) -> AnimeRelationsAndGraphInput {
    AnimeRelationsAndGraphInput {
        relations,
        ..AnimeRelationsAndGraphInput::default()
    }
}

#[sqlx::test(migrations = "./migrations")]
async fn save_anime_page_sources_refuse_duplicate_and_other_entity_pages(pool: PgPool) {
    let anime_id = insert_unmerged_anime(&pool).await;
    link_mal_anime_pages(&pool, anime_id, &["1"]).await;
    let refused_page_sources = [
        vec![
            build_mal_page_source(EntityType::Anime, "1", "https://myanimelist.net/anime/1"),
            build_mal_page_source(
                EntityType::Anime,
                "1",
                "https://myanimelist.net/anime/1/Slug",
            ),
        ],
        vec![build_mal_page_source(
            EntityType::Episode,
            "1:1",
            "https://myanimelist.net/anime/1/episode/1",
        )],
    ];
    for page_sources in refused_page_sources {
        let mut save_input = build_anime_save(anime_id, 1, "v1", Vec::new());
        save_input.page_sources = page_sources;
        let error = save_anime(&pool, &save_input)
            .await
            .expect_err("invalid page sources must be refused");
        assert!(error.is_invalid_argument(), "{error}");
    }
    assert_eq!(read_page_source(&pool, "1").await, None);
    let anime_version: i64 = sqlx::query_scalar("SELECT version FROM anime WHERE id = $1")
        .bind(anime_id)
        .fetch_one(&pool)
        .await
        .expect("read version");
    assert_eq!(anime_version, 1);
}

#[sqlx::test(migrations = "./migrations")]
async fn save_anime_page_sources_write_owned_pages_on_unchanged_hash(pool: PgPool) {
    let anime_id = insert_unmerged_anime(&pool).await;
    let other_anime_id = insert_unmerged_anime(&pool).await;
    link_mal_anime_pages(&pool, anime_id, &["1"]).await;
    link_mal_anime_pages(&pool, other_anime_id, &["2"]).await;

    let first_save = build_anime_save(anime_id, 1, "same-hash", Vec::new());
    let first_outcome = save_anime(&pool, &first_save).await.expect("first save");
    assert_eq!(read_page_source(&pool, "1").await, None);

    let mut repeated_save =
        build_anime_save(anime_id, first_outcome.version, "same-hash", Vec::new());
    repeated_save.page_sources = vec![
        build_mal_page_source(
            EntityType::Anime,
            "1",
            "https://myanimelist.net/anime/1/Slug",
        ),
        build_mal_page_source(EntityType::Anime, "2", "https://example.org/not-owned"),
        build_mal_page_source(EntityType::Anime, "3", "https://example.org/no-page"),
    ];
    let repeated_outcome = save_anime(&pool, &repeated_save)
        .await
        .expect("repeated save");
    assert!(!repeated_outcome.has_changed);
    assert_eq!(repeated_outcome.version, first_outcome.version);
    assert_eq!(
        read_page_source(&pool, "1").await.as_deref(),
        Some("https://myanimelist.net/anime/1/Slug")
    );
    assert_eq!(read_page_source(&pool, "2").await, None);

    let mut renamed_save =
        build_anime_save(anime_id, repeated_outcome.version, "same-hash", Vec::new());
    renamed_save.page_sources = vec![build_mal_page_source(
        EntityType::Anime,
        "1",
        "https://myanimelist.net/anime/1/New_Slug",
    )];
    save_anime(&pool, &renamed_save)
        .await
        .expect("renamed save");
    assert_eq!(
        read_page_source(&pool, "1").await.as_deref(),
        Some("https://myanimelist.net/anime/1/New_Slug")
    );
}

#[sqlx::test(migrations = "./migrations")]
async fn provider_and_entity_type_decode_from_text_and_refuse_unknown_values(pool: PgPool) {
    let (provider, entity_type): (Provider, EntityType) =
        sqlx::query_as("SELECT 'anime_planet'::text, 'character'::text")
            .fetch_one(&pool)
            .await
            .expect("decode known values");
    assert_eq!(
        (provider, entity_type),
        (Provider::AnimePlanet, EntityType::Character)
    );

    let unknown_provider = sqlx::query_scalar::<_, Provider>("SELECT 'myanimelist'::text")
        .fetch_one(&pool)
        .await
        .expect_err("unknown provider slug");
    assert!(
        matches!(unknown_provider, sqlx::Error::ColumnDecode { .. }),
        "{unknown_provider}"
    );
    let unknown_entity_type = sqlx::query_scalar::<_, EntityType>("SELECT 'manga'::text")
        .fetch_one(&pool)
        .await
        .expect_err("unknown entity type");
    assert!(
        matches!(unknown_entity_type, sqlx::Error::ColumnDecode { .. }),
        "{unknown_entity_type}"
    );
    let mismatched_column = sqlx::query_scalar::<_, Provider>("SELECT 1::int4")
        .fetch_one(&pool)
        .await
        .expect_err("integer column");
    assert!(
        matches!(mismatched_column, sqlx::Error::ColumnDecode { .. }),
        "{mismatched_column}"
    );
    let mismatched_entity_column = sqlx::query_scalar::<_, EntityType>("SELECT 1::int4")
        .fetch_one(&pool)
        .await
        .expect_err("integer column");
    assert!(
        matches!(mismatched_entity_column, sqlx::Error::ColumnDecode { .. }),
        "{mismatched_entity_column}"
    );
}

#[sqlx::test(migrations = "./migrations")]
async fn save_anime_relation_url_fills_unheld_page_with_longest_url(pool: PgPool) {
    let anime_id = insert_unmerged_anime(&pool).await;
    link_mal_anime_pages(&pool, anime_id, &["1"]).await;
    let mut save_input = build_anime_save(anime_id, 1, "v1", Vec::new());
    save_input.relations_and_graph = build_relations(vec![
        build_relation_to(
            "SEQUEL",
            "2",
            Some("https://myanimelist.net/anime/2"),
            Some("Sequel"),
        ),
        build_relation_to(
            "SIDE_STORY",
            "2",
            Some("https://myanimelist.net/anime/2/Sequel"),
            Some("Sequel (side story list)"),
        ),
        build_relation_to("PREQUEL", "3", None, None),
    ]);

    save_anime(&pool, &save_input).await.expect("save work");

    assert_eq!(
        read_page_source(&pool, "2").await.as_deref(),
        Some("https://myanimelist.net/anime/2/Sequel")
    );
    assert_eq!(read_page_title(&pool, "2").await.as_deref(), Some("Sequel"));
    assert!(!has_page_row(&pool, "3").await);
}

#[sqlx::test(migrations = "./migrations")]
async fn save_anime_relation_url_yields_to_holder_url(pool: PgPool) {
    let naming_anime_id = insert_unmerged_anime(&pool).await;
    let holding_anime_id = insert_unmerged_anime(&pool).await;
    link_mal_anime_pages(&pool, naming_anime_id, &["1"]).await;
    link_mal_anime_pages(&pool, holding_anime_id, &["2"]).await;

    let mut naming_save = build_anime_save(naming_anime_id, 1, "naming-v1", Vec::new());
    naming_save.relations_and_graph = build_relations(vec![build_relation_to(
        "SEQUEL",
        "2",
        Some("https://myanimelist.net/anime/2"),
        Some("Named Sequel"),
    )]);
    let naming_outcome = save_anime(&pool, &naming_save).await.expect("naming save");
    assert_eq!(
        read_page_source(&pool, "2").await.as_deref(),
        Some("https://myanimelist.net/anime/2")
    );
    assert_eq!(
        read_page_title(&pool, "2").await.as_deref(),
        Some("Named Sequel")
    );

    let mut holding_save = build_anime_save(holding_anime_id, 1, "holding-v1", Vec::new());
    holding_save.page_sources = vec![build_mal_page_source(
        EntityType::Anime,
        "2",
        "https://myanimelist.net/anime/2/Sequel",
    )];
    holding_save.page_sources[0].title = Some("Held Sequel".to_owned());
    let holding_outcome = save_anime(&pool, &holding_save)
        .await
        .expect("holding save");
    assert_eq!(
        read_page_source(&pool, "2").await.as_deref(),
        Some("https://myanimelist.net/anime/2/Sequel")
    );
    assert_eq!(
        read_page_title(&pool, "2").await.as_deref(),
        Some("Held Sequel")
    );

    let mut renaming_save = build_anime_save(
        naming_anime_id,
        naming_outcome.version,
        "naming-v2",
        Vec::new(),
    );
    renaming_save.relations_and_graph = build_relations(vec![build_relation_to(
        "SEQUEL",
        "2",
        Some("https://myanimelist.net/anime/2/Longer_Other_Slug"),
        Some("Renamed Sequel"),
    )]);
    save_anime(&pool, &renaming_save)
        .await
        .expect("renaming save");
    assert_eq!(
        read_page_source(&pool, "2").await.as_deref(),
        Some("https://myanimelist.net/anime/2/Sequel")
    );
    assert_eq!(
        read_page_title(&pool, "2").await.as_deref(),
        Some("Held Sequel")
    );

    let mut retitling_save = build_anime_save(
        holding_anime_id,
        holding_outcome.version,
        "holding-v2",
        Vec::new(),
    );
    retitling_save.page_sources = vec![PageSource {
        page_key: PageKey::new(Provider::Mal, EntityType::Anime, "2").expect("valid page key"),
        source: None,
        title: Some("Sequel Retitled".to_owned()),
    }];
    save_anime(&pool, &retitling_save)
        .await
        .expect("retitling save without URL");
    assert_eq!(
        read_page_source(&pool, "2").await.as_deref(),
        Some("https://myanimelist.net/anime/2/Sequel")
    );
    assert_eq!(
        read_page_title(&pool, "2").await.as_deref(),
        Some("Sequel Retitled")
    );
}

#[sqlx::test(migrations = "./migrations")]
async fn save_anime_relation_url_skips_page_row_locked_by_other_transaction(pool: PgPool) {
    let anime_id = insert_unmerged_anime(&pool).await;
    link_mal_anime_pages(&pool, anime_id, &["1"]).await;
    sqlx::query(
        "INSERT INTO provider_page (provider, entity_type, address) VALUES ('mal', 'anime', '2')",
    )
    .execute(&pool)
    .await
    .expect("insert unheld page");
    let mut locking_tx = pool.begin().await.expect("begin locking transaction");
    sqlx::query(
        "SELECT 1 FROM provider_page
         WHERE provider = 'mal' AND entity_type = 'anime' AND address = '2'
         FOR NO KEY UPDATE",
    )
    .execute(&mut *locking_tx)
    .await
    .expect("lock page row");
    let mut save_input = build_anime_save(anime_id, 1, "v1", Vec::new());
    save_input.relations_and_graph = build_relations(vec![build_relation_to(
        "SEQUEL",
        "2",
        Some("https://myanimelist.net/anime/2"),
        Some("Sequel"),
    )]);

    tokio::time::timeout(
        std::time::Duration::from_secs(10),
        save_anime(&pool, &save_input),
    )
    .await
    .expect("save does not wait for the locked page row")
    .expect("save work");

    assert_eq!(read_page_source(&pool, "2").await, None);
    assert_eq!(read_page_title(&pool, "2").await, None);
    locking_tx.rollback().await.expect("release lock");
}
