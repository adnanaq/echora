//! Integration tests for episode loads, saves and aired notifications against PostgreSQL.

#![expect(
    clippy::expect_used,
    reason = "a failed setup step stops the test, as a test's own expect does"
)]

mod support;

use echora_domain::{EntityType, PageKey, Provider};
use postgres_db::{
    AnimeTitleInput, EPISODE_HASH_FORMAT, EpisodeCharacterInput, EpisodeListEntryInput,
    EpisodeReferenceInput, EpisodeStreamingInput, EpisodeTitleInput, PageSource, SaveEpisodeInput,
    delete_episode, load_anime, load_episode, mark_episode_aired, save_anime, save_episode,
};
use sqlx::PgPool;
use url::Url;
use uuid::Uuid;

use crate::support::{CapturedLogs, build_anime_save, insert_unmerged_anime};

fn build_episode_reference(episode_id: Uuid, episode_number: i32) -> EpisodeReferenceInput {
    EpisodeReferenceInput {
        episode_id,
        episode_number,
        title: format!("Placeholder episode {episode_number}"),
        page_sources: Vec::new(),
        list_entries: Vec::new(),
    }
}

fn build_episode_save(
    episode_id: Uuid,
    expected_version: i64,
    merged_record_hash: &str,
) -> SaveEpisodeInput {
    SaveEpisodeInput {
        episode_id,
        expected_version,
        episode_number: 1,
        season_number: None,
        title: "Placeholder episode 1".to_owned(),
        title_japanese: None,
        title_romaji: None,
        synopsis: None,
        description: None,
        aired_at: None,
        duration_seconds: None,
        is_filler: false,
        is_recap: false,
        score: None,
        titles: Vec::new(),
        images: Vec::new(),
        streaming: Vec::new(),
        characters: Vec::new(),
        merged_record_hash: merged_record_hash.to_owned(),
        hash_format: EPISODE_HASH_FORMAT,
        merge_version: 1,
        traceparent: None,
        page_sources: Vec::new(),
    }
}

async fn create_listed_episode(
    pool: &PgPool,
    episode_reference: EpisodeReferenceInput,
) -> Result<Uuid, postgres_db::Error> {
    let anime_id = insert_unmerged_anime(pool).await;
    let episode_id = episode_reference.episode_id;
    save_anime(
        pool,
        &build_anime_save(anime_id, 1, "work-v1", vec![episode_reference]),
    )
    .await?;
    Ok(episode_id)
}

async fn count_episode_rows(pool: &PgPool, table: &str, episode_id: Uuid) -> i64 {
    sqlx::query_scalar(&format!(
        "SELECT COUNT(*) FROM {table} WHERE episode_id = $1"
    ))
    .bind(episode_id)
    .fetch_one(pool)
    .await
    .unwrap_or(-1)
}

#[sqlx::test(migrations = "./migrations")]
async fn save_anime_new_episode_id_for_listed_number_returns_unique_violation(pool: PgPool) {
    let anime_id = insert_unmerged_anime(&pool).await;
    let first_episode_id = Uuid::now_v7();
    save_anime(
        &pool,
        &build_anime_save(
            anime_id,
            1,
            "work-v1",
            vec![build_episode_reference(first_episode_id, 5)],
        ),
    )
    .await
    .expect("first save creates episode 5");

    let renamed_episode_save = build_anime_save(
        anime_id,
        2,
        "work-v2",
        vec![build_episode_reference(Uuid::now_v7(), 5)],
    );
    let save_error = save_anime(&pool, &renamed_episode_save)
        .await
        .expect_err("second episode 5 must be refused");

    assert!(save_error.is_unique_violation());
    let stored_version: i64 = sqlx::query_scalar("SELECT version FROM anime WHERE id = $1")
        .bind(anime_id)
        .fetch_one(&pool)
        .await
        .expect("read work version");
    assert_eq!(stored_version, 2);
}

#[sqlx::test(migrations = "./migrations")]
async fn save_episode_score_rounding_to_one_thousand_returns_invalid_argument(pool: PgPool) {
    let episode_id = create_listed_episode(&pool, build_episode_reference(Uuid::now_v7(), 1))
        .await
        .expect("work save creates the episode");
    let mut episode_save = build_episode_save(episode_id, 1, "episode-v1");
    episode_save.score = Some(999.996);

    let save_error = save_episode(&pool, &episode_save)
        .await
        .expect_err("score that rounds to 1000.00 must be refused");

    assert!(save_error.is_invalid_argument());
}

#[sqlx::test(migrations = "./migrations")]
async fn load_episode_never_fetched_page_returns_no_pages(pool: PgPool) {
    let mut episode_reference = build_episode_reference(Uuid::now_v7(), 1);
    episode_reference.page_sources = vec![PageSource {
        page_key: PageKey::new(Provider::Mal, EntityType::Episode, "21:1").expect("valid page key"),
        source: None,
        title: None,
    }];
    let episode_id = create_listed_episode(&pool, episode_reference)
        .await
        .expect("work save creates the episode");

    let loaded_episode = load_episode(&pool, episode_id)
        .await
        .expect("load the new episode");

    assert_eq!(loaded_episode.pages, Vec::new());
    assert_eq!(
        count_episode_rows(&pool, "episode_source", episode_id).await,
        1
    );
}

#[sqlx::test(migrations = "./migrations")]
async fn save_anime_episode_pages_from_two_providers_links_both(pool: PgPool) {
    let mut episode_reference = build_episode_reference(Uuid::now_v7(), 1);
    episode_reference.page_sources = vec![
        PageSource {
            page_key: PageKey::new(Provider::Mal, EntityType::Episode, "21:1")
                .expect("valid page key"),
            source: None,
            title: None,
        },
        PageSource {
            page_key: PageKey::new(Provider::AniDb, EntityType::Episode, "1")
                .expect("valid page key"),
            source: None,
            title: None,
        },
    ];

    let episode_id = create_listed_episode(&pool, episode_reference)
        .await
        .expect("work save creates the episode");

    let linked_providers: Vec<String> = sqlx::query_scalar(
        "SELECT provider FROM episode_source WHERE episode_id = $1 ORDER BY provider",
    )
    .bind(episode_id)
    .fetch_all(&pool)
    .await
    .expect("read linked pages");
    assert_eq!(linked_providers, vec!["anidb".to_owned(), "mal".to_owned()]);
}

#[sqlx::test(migrations = "./migrations")]
async fn save_episode_empty_lists_removes_previous_rows(pool: PgPool) {
    let episode_id = create_listed_episode(&pool, build_episode_reference(Uuid::now_v7(), 1))
        .await
        .expect("work save creates the episode");
    let mut filled_save = build_episode_save(episode_id, 1, "episode-v1");
    filled_save.titles = vec![EpisodeTitleInput {
        language: "en".to_owned(),
        title: "Placeholder".to_owned(),
    }];
    filled_save.images = vec!["https://example.com/placeholder.jpg".to_owned()];
    filled_save.streaming = vec![EpisodeStreamingInput {
        platform: "crunchyroll".to_owned(),
        source: "https://example.com/watch/1".to_owned(),
    }];
    filled_save.characters = vec![EpisodeCharacterInput {
        character_page_source: PageSource {
            page_key: PageKey::new(Provider::Mal, EntityType::Character, "1")
                .expect("valid page key"),
            source: Some(Url::parse("https://myanimelist.net/character/1").expect("valid URL")),
            title: None,
        },
        role: "MAIN".to_owned(),
    }];
    save_episode(&pool, &filled_save)
        .await
        .expect("first merge stores the lists");
    let stored_platform: String = sqlx::query_scalar(
        "SELECT platform FROM external_source WHERE source = 'https://example.com/watch/1'",
    )
    .fetch_one(&pool)
    .await
    .expect("the streaming URL is stored once with its platform");
    assert_eq!(stored_platform, "crunchyroll");
    let stored_languages: Vec<String> = sqlx::query_scalar("SELECT code FROM language")
        .fetch_all(&pool)
        .await
        .expect("read the language codes");
    assert_eq!(stored_languages, vec!["en".to_owned()]);

    save_episode(&pool, &build_episode_save(episode_id, 2, "episode-v2"))
        .await
        .expect("second merge clears the lists");

    for table in [
        "episode_title",
        "episode_image",
        "episode_streaming",
        "episode_character",
    ] {
        assert_eq!(
            count_episode_rows(&pool, table, episode_id).await,
            0,
            "{table}"
        );
    }
}

#[sqlx::test(migrations = "./migrations")]
async fn mark_episode_aired_missing_episode_returns_not_found(pool: PgPool) {
    let mark_error = mark_episode_aired(&pool, Uuid::now_v7(), None)
        .await
        .expect_err("missing episode must be refused");

    assert!(mark_error.is_not_found());
}

#[sqlx::test(migrations = "./migrations")]
async fn mark_episode_aired_deleted_anime_returns_not_found(pool: PgPool) {
    let episode_id = create_listed_episode(&pool, build_episode_reference(Uuid::now_v7(), 1))
        .await
        .expect("work save creates the episode");
    sqlx::query(
        "UPDATE anime SET deleted_at = now()
         WHERE id = (SELECT anime_id FROM episode WHERE id = $1)",
    )
    .bind(episode_id)
    .execute(&pool)
    .await
    .expect("delete the work");

    let mark_error = mark_episode_aired(&pool, episode_id, None)
        .await
        .expect_err("episode of a deleted work must be refused");

    assert!(mark_error.is_not_found());
}

#[sqlx::test(migrations = "./migrations")]
async fn save_episode_first_merge_logs_saved_event(pool: PgPool) {
    let episode_id = create_listed_episode(&pool, build_episode_reference(Uuid::now_v7(), 1))
        .await
        .expect("work save creates the episode");
    let captured_logs = CapturedLogs::install_global();

    save_episode(&pool, &build_episode_save(episode_id, 1, "episode-v1"))
        .await
        .expect("first merge saves the episode");

    let log_text = captured_logs.read_lines_with(&format!("episode.id={episode_id}"));
    assert!(log_text.contains("saved episode"), "{log_text}");
    assert!(log_text.contains("episode.version=2"), "{log_text}");
    assert!(
        log_text.contains("outbox.subject=\"episode.created\""),
        "{log_text}"
    );
}

#[sqlx::test(migrations = "./migrations")]
async fn save_episode_missing_title_stores_none_until_merge_finds_one(pool: PgPool) {
    let episode_id = create_listed_episode(&pool, build_episode_reference(Uuid::now_v7(), 1))
        .await
        .expect("work save creates the episode");
    let mut untitled_save = build_episode_save(episode_id, 1, "episode-v1");
    untitled_save.title = "  ".to_owned();

    save_episode(&pool, &untitled_save)
        .await
        .expect("an untitled episode saves");
    let untitled_title: Option<String> =
        sqlx::query_scalar("SELECT title FROM episode WHERE id = $1")
            .bind(episode_id)
            .fetch_one(&pool)
            .await
            .expect("read the title");
    save_episode(&pool, &build_episode_save(episode_id, 2, "episode-v2"))
        .await
        .expect("a later merge adds the title");
    let later_title: Option<String> = sqlx::query_scalar("SELECT title FROM episode WHERE id = $1")
        .bind(episode_id)
        .fetch_one(&pool)
        .await
        .expect("read the title");

    assert_eq!(untitled_title, None);
    assert_eq!(later_title.as_deref(), Some("Placeholder episode 1"));
}

#[sqlx::test(migrations = "./migrations")]
async fn save_episode_timestamps_share_transaction_time(pool: PgPool) {
    let episode_id = create_listed_episode(&pool, build_episode_reference(Uuid::now_v7(), 1))
        .await
        .expect("the work creates the episode");
    save_episode(&pool, &build_episode_save(episode_id, 1, "episode-v1"))
        .await
        .expect("the episode saves");

    let (episode_updated_at, episode_merged_at, event_created_at): (
        chrono::DateTime<chrono::Utc>,
        Option<chrono::DateTime<chrono::Utc>>,
        chrono::DateTime<chrono::Utc>,
    ) = sqlx::query_as(
        "SELECT e.updated_at, e.merged_at, o.created_at
         FROM episode e
         JOIN outbox o ON o.entity_id = e.id AND o.subject = 'episode.created'
         WHERE e.id = $1",
    )
    .bind(episode_id)
    .fetch_one(&pool)
    .await
    .expect("read the save's timestamps");

    assert_eq!(episode_merged_at, Some(episode_updated_at));
    assert_eq!(event_created_at, episode_updated_at);
}

#[sqlx::test(migrations = "./migrations")]
async fn save_episode_unchanged_hash_from_newer_merge_records_merge_version(pool: PgPool) {
    let episode_id = create_listed_episode(&pool, build_episode_reference(Uuid::now_v7(), 1))
        .await
        .expect("the work creates the episode");
    save_episode(&pool, &build_episode_save(episode_id, 1, "episode-v1"))
        .await
        .expect("the first merge saves");
    let mut newer_merge_save = build_episode_save(episode_id, 2, "episode-v1");
    newer_merge_save.merge_version = 2;

    let save_outcome = save_episode(&pool, &newer_merge_save)
        .await
        .expect("the unchanged save succeeds");
    let (stored_version, stored_merge_version, event_count): (i64, Option<i32>, i64) =
        sqlx::query_as(
            "SELECT e.version, e.merge_version,
                    (SELECT COUNT(*) FROM outbox o WHERE o.entity_id = e.id)
             FROM episode e WHERE e.id = $1",
        )
        .bind(episode_id)
        .fetch_one(&pool)
        .await
        .expect("read the episode");

    assert!(!save_outcome.has_changed);
    assert_eq!(stored_version, 2);
    assert_eq!(stored_merge_version, Some(2));
    assert_eq!(event_count, 1);
}

#[sqlx::test(migrations = "./migrations")]
async fn save_episode_number_other_than_listed_number_returns_invalid_argument(pool: PgPool) {
    let anime_id = insert_unmerged_anime(&pool).await;
    let first_episode_id = Uuid::now_v7();
    let second_episode_id = Uuid::now_v7();
    save_anime(
        &pool,
        &build_anime_save(
            anime_id,
            1,
            "work-v1",
            vec![
                build_episode_reference(first_episode_id, 1),
                build_episode_reference(second_episode_id, 2),
            ],
        ),
    )
    .await
    .expect("the work creates both episodes");

    for other_number in [1, 0, 3] {
        let mut renumbered_save = build_episode_save(second_episode_id, 1, "episode-v1");
        renumbered_save.episode_number = other_number;

        let save_error = save_episode(&pool, &renumbered_save)
            .await
            .expect_err("the work's list owns the number");

        assert!(save_error.is_invalid_argument(), "number {other_number}");
    }
    let stored_number: i32 = sqlx::query_scalar("SELECT episode_number FROM episode WHERE id = $1")
        .bind(second_episode_id)
        .fetch_one(&pool)
        .await
        .expect("read the episode number");
    assert_eq!(stored_number, 2);
}

#[sqlx::test(migrations = "./migrations")]
async fn save_episode_deleted_episode_returns_not_found(pool: PgPool) {
    let episode_id = create_listed_episode(&pool, build_episode_reference(Uuid::now_v7(), 1))
        .await
        .expect("the work creates the episode");
    delete_episode(&pool, episode_id, 1, None)
        .await
        .expect("delete the episode");

    let save_error = save_episode(&pool, &build_episode_save(episode_id, 2, "episode-v1"))
        .await
        .expect_err("a deleted episode cannot be saved");

    assert!(save_error.is_not_found());
}

#[sqlx::test(migrations = "./migrations")]
async fn save_anime_and_save_episode_sharing_new_language_at_once_both_succeed(pool: PgPool) {
    for attempt in 0..20 {
        let new_language = format!("x-shared{attempt}");
        let anime_id = insert_unmerged_anime(&pool).await;
        let episode_id = Uuid::now_v7();
        save_anime(
            &pool,
            &build_anime_save(
                anime_id,
                1,
                "work-v1",
                vec![build_episode_reference(episode_id, 1)],
            ),
        )
        .await
        .unwrap_or_else(|error| panic!("attempt {attempt}: first work save: {error}"));

        let mut changed_reference = build_episode_reference(episode_id, 1);
        changed_reference.list_entries = vec![EpisodeListEntryInput {
            provider: Provider::Mal,
            title: Some(format!("Listed title {attempt}")),
            aired_at: None,
            duration_seconds: None,
            is_filler: None,
            is_recap: None,
        }];
        let mut anime_save = build_anime_save(anime_id, 2, "work-v2", vec![changed_reference]);
        anime_save.details.titles = vec![AnimeTitleInput {
            language: new_language.clone(),
            title: "Work title".to_owned(),
        }];
        let mut episode_save = build_episode_save(episode_id, 1, "episode-v1");
        episode_save.titles = vec![EpisodeTitleInput {
            language: new_language,
            title: "Episode title".to_owned(),
        }];

        let (anime_outcome, episode_outcome) = tokio::join!(
            save_anime(&pool, &anime_save),
            save_episode(&pool, &episode_save)
        );

        anime_outcome.unwrap_or_else(|error| panic!("attempt {attempt}: work save: {error}"));
        if let Err(episode_error) = episode_outcome {
            assert!(
                episode_error.is_version_conflict(),
                "attempt {attempt}: episode save failed with {episode_error}"
            );
        }
    }
}

async fn read_episode_page_source(pool: &PgPool, address: &str) -> Option<String> {
    sqlx::query_scalar(
        "SELECT source FROM provider_page
         WHERE provider = 'mal' AND entity_type = 'episode' AND address = $1",
    )
    .bind(address)
    .fetch_one(pool)
    .await
    .expect("read the episode page URL")
}

#[sqlx::test(migrations = "./migrations")]
async fn save_anime_listed_episode_page_url_waits_for_episode_url(pool: PgPool) {
    let anime_id = insert_unmerged_anime(&pool).await;
    let episode_id = Uuid::now_v7();
    let listed_page = |source: &str| PageSource {
        page_key: PageKey::new(Provider::Mal, EntityType::Episode, "21:1").expect("valid page key"),
        source: Some(Url::parse(source).expect("valid URL")),
        title: None,
    };
    let mut episode_reference = build_episode_reference(episode_id, 1);
    episode_reference.page_sources = vec![listed_page(
        "https://myanimelist.net/anime/21/One_Piece/episode/1",
    )];
    let first_work_save = save_anime(
        &pool,
        &build_anime_save(anime_id, 1, "work-v1", vec![episode_reference.clone()]),
    )
    .await
    .expect("the work lists the episode");

    assert_eq!(
        read_episode_page_source(&pool, "21:1").await.as_deref(),
        Some("https://myanimelist.net/anime/21/One_Piece/episode/1")
    );
    let known_episodes = load_anime(&pool, anime_id)
        .await
        .expect("load the work")
        .known_episodes;
    assert_eq!(
        known_episodes[0].page_sources,
        episode_reference.page_sources
    );

    let mut episode_save = build_episode_save(episode_id, 1, "episode-v1");
    episode_save.page_sources = vec![listed_page(
        "https://myanimelist.net/anime/21/One_Piece_Renamed/episode/1",
    )];
    save_episode(&pool, &episode_save)
        .await
        .expect("the episode's merge chooses its URL");
    episode_reference.page_sources = vec![listed_page(
        "https://myanimelist.net/anime/21/Old_List_Slug/episode/1",
    )];
    save_anime(
        &pool,
        &build_anime_save(
            anime_id,
            first_work_save.version,
            "work-v2",
            vec![episode_reference],
        ),
    )
    .await
    .expect("the work lists the episode again");

    assert_eq!(
        read_episode_page_source(&pool, "21:1").await.as_deref(),
        Some("https://myanimelist.net/anime/21/One_Piece_Renamed/episode/1")
    );
}
