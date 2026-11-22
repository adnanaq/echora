//! Integration tests for work songs, streaming and statistics against PostgreSQL.

mod support;

use postgres_db::{
    AnimeDetailsInput, AnimeExternalSourceInput, AnimeScheduleAndStatsInput, AnimeStatisticsInput,
    AnimeStreamingSourceInput, AnimeThemeInput, AnimeThemeSongInput, AnimeTitleInput,
    AnimeTrailerInput, EpisodeRangeInput, SaveAnimeInput, save_anime,
};
use sqlx::PgPool;
use uuid::Uuid;

use crate::support::{build_work_save, insert_unmerged_work};

async fn count_rows(pool: &PgPool, table: &str) -> i64 {
    sqlx::query_scalar(&format!("SELECT COUNT(*) FROM {table}"))
        .fetch_one(pool)
        .await
        .unwrap_or(-1)
}

#[sqlx::test(migrations = "./migrations")]
async fn save_anime_entries_without_child_lists_store_no_child_rows(pool: PgPool) {
    let anime_id = insert_unmerged_work(&pool).await;
    let mut work_save = build_work_save(anime_id, 1, "work-v1", Vec::new());
    work_save.schedule_and_stats = AnimeScheduleAndStatsInput {
        opening_themes: vec![AnimeThemeSongInput {
            position: 0,
            title: "Opening".to_owned(),
            artist: None,
            episode_ranges: Vec::new(),
        }],
        streaming_sources: vec![AnimeStreamingSourceInput {
            platform: "Crunchyroll".to_owned(),
            source: "https://example.com/watch".to_owned(),
            region: None,
            is_free: None,
            is_premium_required: None,
            has_dub: None,
            subtitle_languages: Vec::new(),
        }],
        statistics: vec![AnimeStatisticsInput {
            provider: "mal".to_owned(),
            score: Some(8.5),
            scored_by: None,
            members: None,
            favorites: None,
            rank: None,
            popularity: None,
            contextual_ranks: Vec::new(),
        }],
        ..AnimeScheduleAndStatsInput::default()
    };

    save_anime(&pool, &work_save).await.expect("the work saves");

    for (table, expected_rows) in [
        ("anime_opening_theme", 1),
        ("anime_opening_theme_episode", 0),
        ("anime_streaming_source", 1),
        ("anime_streaming_source_subtitle_language", 0),
        ("anime_statistics", 1),
        ("anime_statistics_contextual_rank", 0),
    ] {
        assert_eq!(count_rows(&pool, table).await, expected_rows, "{table}");
    }
}

const SHARED_LINK: &str = "https://official.example.com";
const SHARED_TRAILER: &str = "https://youtube.com/watch?v=shared";
const SHARED_STREAM: &str = "https://crunchyroll.com/series/shared";

fn build_shared_details_save(
    anime_id: Uuid,
    expected_version: i64,
    merged_record_hash: &str,
    link_label: &str,
) -> SaveAnimeInput {
    let mut work_save = build_work_save(anime_id, expected_version, merged_record_hash, Vec::new());
    work_save.details = AnimeDetailsInput {
        titles: vec![AnimeTitleInput {
            language: "de".to_owned(),
            title: "Ein Stück".to_owned(),
        }],
        themes: vec![AnimeThemeInput {
            name: "Time Travel".to_owned(),
            description: Some("Characters move through time.".to_owned()),
        }],
        ..AnimeDetailsInput::default()
    };
    work_save.schedule_and_stats = AnimeScheduleAndStatsInput {
        trailers: vec![AnimeTrailerInput {
            source: SHARED_TRAILER.to_owned(),
            title: Some("Franchise PV".to_owned()),
            thumbnail_source: Some("https://img.example.com/pv.jpg".to_owned()),
        }],
        external_sources: vec![AnimeExternalSourceInput {
            source: SHARED_LINK.to_owned(),
            platform: "Official Site".to_owned(),
            label: Some(link_label.to_owned()),
            language: Some("ja".to_owned()),
        }],
        streaming_sources: vec![AnimeStreamingSourceInput {
            platform: "Crunchyroll".to_owned(),
            source: SHARED_STREAM.to_owned(),
            region: None,
            is_free: None,
            is_premium_required: None,
            has_dub: None,
            subtitle_languages: vec!["en".to_owned(), "pt-BR".to_owned()],
        }],
        ..AnimeScheduleAndStatsInput::default()
    };
    work_save
}

async fn read_text_rows(pool: &PgPool, query: &str) -> Result<Vec<String>, sqlx::Error> {
    sqlx::query_scalar(query).fetch_all(pool).await
}

#[sqlx::test(migrations = "./migrations")]
async fn save_anime_details_shared_by_two_works_stored_once_with_labels_per_work(pool: PgPool) {
    let first_anime_id = insert_unmerged_work(&pool).await;
    let second_anime_id = insert_unmerged_work(&pool).await;

    save_anime(
        &pool,
        &build_shared_details_save(first_anime_id, 1, "first-v1", "Official"),
    )
    .await
    .expect("the first work saves");
    save_anime(
        &pool,
        &build_shared_details_save(second_anime_id, 1, "second-v1", "Site"),
    )
    .await
    .expect("the second work saves");

    assert_eq!(
        read_text_rows(&pool, "SELECT source || ' ' || platform || ' ' || coalesce(language, '-') FROM external_source ORDER BY source")
            .await
            .expect("read the rows"),
        vec![
            format!("{SHARED_STREAM} Crunchyroll -"),
            format!("{SHARED_LINK} Official Site ja"),
        ]
    );
    assert_eq!(
        read_text_rows(
            &pool,
            "SELECT label FROM anime_external_source ORDER BY label"
        )
        .await
        .expect("read the rows"),
        vec!["Official", "Site"]
    );
    assert_eq!(
        read_text_rows(&pool, "SELECT source FROM trailer")
            .await
            .expect("read the rows"),
        vec![SHARED_TRAILER]
    );
    assert_eq!(count_rows(&pool, "anime_trailer").await, 2);
    assert_eq!(
        read_text_rows(&pool, "SELECT name FROM theme")
            .await
            .expect("read the rows"),
        vec!["Time Travel"]
    );
    assert_eq!(count_rows(&pool, "anime_theme").await, 2);
    assert_eq!(
        read_text_rows(&pool, "SELECT code FROM language ORDER BY code")
            .await
            .expect("read the rows"),
        vec!["de", "en", "ja", "pt-BR"]
    );
    assert_eq!(
        read_text_rows(&pool, "SELECT code FROM country")
            .await
            .expect("read the rows"),
        vec!["JP"]
    );
}

#[sqlx::test(migrations = "./migrations")]
async fn save_anime_later_save_without_shared_detail_keeps_stored_value(pool: PgPool) {
    let first_anime_id = insert_unmerged_work(&pool).await;
    let second_anime_id = insert_unmerged_work(&pool).await;
    save_anime(
        &pool,
        &build_shared_details_save(first_anime_id, 1, "first-v1", "Official"),
    )
    .await
    .expect("the first work saves");

    let mut sparse_save = build_shared_details_save(second_anime_id, 1, "second-v1", "Site");
    sparse_save.details.themes[0].description = None;
    sparse_save.schedule_and_stats.trailers[0].title = None;
    sparse_save.schedule_and_stats.trailers[0].thumbnail_source = None;
    sparse_save.schedule_and_stats.external_sources[0].language = None;
    sparse_save.schedule_and_stats.external_sources[0].platform = "Official".to_owned();
    save_anime(&pool, &sparse_save)
        .await
        .expect("the second work saves without the shared details");

    assert_eq!(
        read_text_rows(&pool, "SELECT description FROM theme")
            .await
            .expect("read the rows"),
        vec!["Characters move through time."]
    );
    assert_eq!(
        read_text_rows(
            &pool,
            "SELECT title || ' ' || thumbnail_source FROM trailer"
        )
        .await
        .expect("read the rows"),
        vec!["Franchise PV https://img.example.com/pv.jpg"]
    );
    assert_eq!(
        read_text_rows(&pool, &format!("SELECT platform || ' ' || language FROM external_source WHERE source = '{SHARED_LINK}'"))
            .await
            .expect("read the rows"),
        vec!["Official ja"]
    );
}

#[sqlx::test(migrations = "./migrations")]
async fn save_anime_later_save_with_new_shared_detail_replaces_stored_value(pool: PgPool) {
    let first_anime_id = insert_unmerged_work(&pool).await;
    let second_anime_id = insert_unmerged_work(&pool).await;
    save_anime(
        &pool,
        &build_shared_details_save(first_anime_id, 1, "first-v1", "Official"),
    )
    .await
    .expect("the first work saves");

    let mut corrected_save = build_shared_details_save(second_anime_id, 1, "second-v1", "Site");
    corrected_save.details.themes[0].description = Some("Stories of time loops.".to_owned());
    corrected_save.schedule_and_stats.trailers[0].title = Some("Franchise PV 2".to_owned());
    corrected_save.schedule_and_stats.external_sources[0].language = Some("en".to_owned());
    corrected_save.schedule_and_stats.external_sources[0].platform = "Official".to_owned();
    save_anime(&pool, &corrected_save)
        .await
        .expect("the second work saves with new shared details");

    assert_eq!(
        read_text_rows(&pool, "SELECT description FROM theme")
            .await
            .expect("read the rows"),
        vec!["Stories of time loops."]
    );
    assert_eq!(
        read_text_rows(&pool, "SELECT title FROM trailer")
            .await
            .expect("read the rows"),
        vec!["Franchise PV 2"]
    );
    assert_eq!(
        read_text_rows(&pool, &format!("SELECT platform || ' ' || language FROM external_source WHERE source = '{SHARED_LINK}'"))
            .await
            .expect("read the rows"),
        vec!["Official en"]
    );
}

#[sqlx::test(migrations = "./migrations")]
async fn save_anime_one_url_given_two_platforms_returns_invalid_argument(pool: PgPool) {
    let anime_id = insert_unmerged_work(&pool).await;
    let mut conflicting_save = build_shared_details_save(anime_id, 1, "conflict-v1", "Official");
    conflicting_save.schedule_and_stats.streaming_sources[0].source = SHARED_LINK.to_owned();

    let save_error = save_anime(&pool, &conflicting_save)
        .await
        .expect_err("one URL with two platforms must be refused");

    assert!(save_error.is_invalid_argument());
    assert_eq!(count_rows(&pool, "external_source").await, 0);
}

#[sqlx::test(migrations = "./migrations")]
async fn save_anime_openings_and_endings_stored_in_their_own_tables(pool: PgPool) {
    let anime_id = insert_unmerged_work(&pool).await;
    let build_song = |position: i32, title: &str, range_end: Option<i32>| AnimeThemeSongInput {
        position,
        title: title.to_owned(),
        artist: None,
        episode_ranges: range_end
            .map(|last_episode| EpisodeRangeInput {
                range_start: 1,
                range_end: Some(last_episode),
            })
            .into_iter()
            .collect(),
    };
    let mut work_save = build_work_save(anime_id, 1, "songs-v1", Vec::new());
    work_save.schedule_and_stats = AnimeScheduleAndStatsInput {
        opening_themes: vec![
            build_song(0, "We Are!", Some(115)),
            build_song(1, "Believe", None),
        ],
        ending_themes: vec![
            build_song(0, "Memories", Some(30)),
            build_song(1, "RUN! RUN! RUN!", None),
        ],
        ..AnimeScheduleAndStatsInput::default()
    };

    save_anime(&pool, &work_save).await.expect("the work saves");

    assert_eq!(
        read_text_rows(
            &pool,
            "SELECT title FROM anime_opening_theme ORDER BY position"
        )
        .await
        .expect("read the rows"),
        vec!["We Are!", "Believe"]
    );
    assert_eq!(
        read_text_rows(
            &pool,
            "SELECT title FROM anime_ending_theme ORDER BY position"
        )
        .await
        .expect("read the rows"),
        vec!["Memories", "RUN! RUN! RUN!"]
    );
    assert_eq!(
        read_text_rows(&pool, "SELECT song.title FROM anime_opening_theme_episode episode JOIN anime_opening_theme song ON song.id = episode.opening_theme_id")
            .await
            .expect("read the rows"),
        vec!["We Are!"]
    );
    assert_eq!(
        read_text_rows(&pool, "SELECT song.title FROM anime_ending_theme_episode episode JOIN anime_ending_theme song ON song.id = episode.ending_theme_id")
            .await
            .expect("read the rows"),
        vec!["Memories"]
    );
}

fn build_many_shared_links_save(anime_id: Uuid, reverse_order: bool) -> SaveAnimeInput {
    let mut work_save = build_work_save(anime_id, 1, "many-links-v1", Vec::new());
    let mut link_numbers: Vec<u32> = (0..40).collect();
    if reverse_order {
        link_numbers.reverse();
    }
    work_save.details.themes = link_numbers
        .iter()
        .map(|link_number| AnimeThemeInput {
            name: format!("Theme {link_number}"),
            description: Some(format!("Description {link_number}")),
        })
        .collect();
    work_save.schedule_and_stats = AnimeScheduleAndStatsInput {
        trailers: link_numbers
            .iter()
            .map(|link_number| AnimeTrailerInput {
                source: format!("https://youtube.com/watch?v={link_number}"),
                title: Some(format!("PV {link_number}")),
                thumbnail_source: None,
            })
            .collect(),
        external_sources: link_numbers
            .iter()
            .map(|link_number| AnimeExternalSourceInput {
                source: format!("https://example.com/link/{link_number}"),
                platform: "Wiki".to_owned(),
                label: None,
                language: Some(if link_number % 2 == 0 { "ja" } else { "en" }.to_owned()),
            })
            .collect(),
        ..AnimeScheduleAndStatsInput::default()
    };
    work_save
}

#[sqlx::test(migrations = "./migrations")]
async fn save_anime_two_works_sharing_rows_saved_at_once_both_succeed(pool: PgPool) {
    for attempt in 0..5 {
        let first_anime_id = insert_unmerged_work(&pool).await;
        let second_anime_id = insert_unmerged_work(&pool).await;
        let first_save = build_many_shared_links_save(first_anime_id, false);
        let second_save = build_many_shared_links_save(second_anime_id, true);

        let (first_outcome, second_outcome) = tokio::join!(
            save_anime(&pool, &first_save),
            save_anime(&pool, &second_save)
        );

        first_outcome.unwrap_or_else(|error| panic!("attempt {attempt}: first save: {error}"));
        second_outcome.unwrap_or_else(|error| panic!("attempt {attempt}: second save: {error}"));
    }
    assert_eq!(count_rows(&pool, "external_source").await, 40);
    assert_eq!(count_rows(&pool, "trailer").await, 40);
    assert_eq!(count_rows(&pool, "theme").await, 40);
    assert_eq!(count_rows(&pool, "anime_external_source").await, 400);
}

fn build_crossed_languages_save(
    anime_id: Uuid,
    title_language: &str,
    link_language: &str,
) -> SaveAnimeInput {
    let mut work_save = build_work_save(anime_id, 1, "languages-v1", Vec::new());
    work_save.details.titles = vec![AnimeTitleInput {
        language: title_language.to_owned(),
        title: format!("Title in {title_language}"),
    }];
    work_save.schedule_and_stats.external_sources = vec![AnimeExternalSourceInput {
        source: format!("https://example.com/{anime_id}"),
        platform: "Official Site".to_owned(),
        label: None,
        language: Some(link_language.to_owned()),
    }];
    work_save
}

#[sqlx::test(migrations = "./migrations")]
async fn save_anime_new_languages_in_crossed_order_saved_at_once_both_succeed(pool: PgPool) {
    for attempt in 0..20 {
        let first_language = format!("x-first{attempt}");
        let second_language = format!("x-second{attempt}");
        let first_anime_id = insert_unmerged_work(&pool).await;
        let second_anime_id = insert_unmerged_work(&pool).await;
        let first_save =
            build_crossed_languages_save(first_anime_id, &first_language, &second_language);
        let second_save =
            build_crossed_languages_save(second_anime_id, &second_language, &first_language);

        let (first_outcome, second_outcome) = tokio::join!(
            save_anime(&pool, &first_save),
            save_anime(&pool, &second_save)
        );

        first_outcome.unwrap_or_else(|error| panic!("attempt {attempt}: first save: {error}"));
        second_outcome.unwrap_or_else(|error| panic!("attempt {attempt}: second save: {error}"));
    }
}
