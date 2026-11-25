//! Integration tests for visible entity reads (`read_visible_*` and `list_visible_episodes`).

#![allow(
    clippy::expect_used,
    reason = "a failed setup step stops the test, as a test's own expect does"
)]

mod support;

use chrono::{Duration, NaiveDate, NaiveTime, Utc};
use postgres_db::{
    ANIME_HASH_FORMAT, AnimeBroadcastInput, AnimeCompanyInput, AnimeContextualRankInput,
    AnimeDetailsInput, AnimeExternalSourceInput, AnimeHiatusInput, AnimeImageInput,
    AnimeRelationInput, AnimeRelationsAndGraphInput, AnimeScheduleAndStatsInput,
    AnimeSourceMaterialInput, AnimeStatisticsInput, AnimeStreamingSourceInput, AnimeThemeInput,
    AnimeThemeSongInput, AnimeTitleInput, AnimeTrailerInput, CharacterAppearanceInput,
    CharacterMangaographyInput, CompanyXrefInput, EPISODE_HASH_FORMAT, EpisodeCharacterInput,
    EpisodeRangeInput, EpisodeReferenceInput, EpisodeStreamingInput, EpisodeTitleInput, Provider,
    SaveAnimeInput, SaveCharacterInput, SaveEpisodeInput, SourceMaterialXrefInput,
    list_visible_episodes, read_visible_anime, read_visible_character, read_visible_episode,
    save_anime, save_character, save_episode,
};
use sqlx::PgPool;
use uuid::Uuid;

use crate::support::{insert_unmerged_anime, insert_unmerged_character, link_character_page};

#[sqlx::test(migrations = "./migrations")]
async fn read_visible_entities_return_not_found_when_unmerged_or_deleted_and_hydrate_all_fields(
    pool: PgPool,
) {
    let now = Utc::now();
    let anime_id = insert_unmerged_anime(&pool).await;
    let target_anime_id = insert_unmerged_anime(&pool).await;
    assert!(
        read_visible_anime(&pool, anime_id)
            .await
            .expect_err("unmerged anime must not be visible")
            .is_not_found()
    );

    sqlx::query(
        "INSERT INTO provider_page (provider, entity_type, address)
         VALUES ('mal', 'anime', '1'), ('anilist', 'anime', '9'), ('mal', 'anime', '2')",
    )
    .execute(&pool)
    .await
    .expect("insert pages");
    sqlx::query(
        "INSERT INTO anime_source (anime_id, provider, entity_type, address)
         VALUES ($1, 'mal', 'anime', '1'), ($2, 'mal', 'anime', '2')",
    )
    .bind(anime_id)
    .bind(target_anime_id)
    .execute(&pool)
    .await
    .expect("link sources");
    sqlx::query(
        "INSERT INTO anime_partial_source (anime_id, provider, entity_type, address)
         VALUES ($1, 'anilist', 'anime', '9')",
    )
    .bind(anime_id)
    .execute(&pool)
    .await
    .expect("link partial source");

    save_anime(
        &pool,
        &SaveAnimeInput {
            anime_id: target_anime_id,
            expected_version: 1,
            title: "Sequel Target".to_owned(),
            title_english: None,
            title_japanese: None,
            anime_type: "TV".to_owned(),
            status: "UPCOMING".to_owned(),
            rating: "UNKNOWN".to_owned(),
            season: None,
            source_material: None,
            year: Some(2026),
            month: None,
            episode_count: 12,
            duration_seconds: None,
            is_nsfw: Some(false),
            country_of_origin: None,
            synopsis: None,
            background: None,
            aired_from: None,
            aired_to: None,
            score_weighted: None,
            score_mean: None,
            score_median: None,
            details: AnimeDetailsInput::default(),
            schedule_and_stats: AnimeScheduleAndStatsInput::default(),
            relations_and_graph: AnimeRelationsAndGraphInput::default(),
            episodes: Vec::new(),
            merged_record_hash: "target-v1".to_owned(),
            hash_format: ANIME_HASH_FORMAT,
            merge_version: 1,
            traceparent: None,
        },
    )
    .await
    .expect("save target anime");

    let ep_id = Uuid::now_v7();
    let save_input = SaveAnimeInput {
        anime_id,
        expected_version: 1,
        title: "Cowboy Bebop".to_owned(),
        title_english: Some("Cowboy Bebop".to_owned()),
        title_japanese: Some("カウボーイビバップ".to_owned()),
        anime_type: "TV".to_owned(),
        status: "FINISHED".to_owned(),
        rating: "PG-13 - Teens 13 or older".to_owned(),
        season: Some("SPRING".to_owned()),
        source_material: Some("ORIGINAL".to_owned()),
        year: Some(1998),
        month: Some(4),
        episode_count: 26,
        duration_seconds: Some(1440),
        is_nsfw: Some(false),
        country_of_origin: Some("JP".to_owned()),
        synopsis: Some("Space bounty hunters.".to_owned()),
        background: Some("Classic Sunrise series.".to_owned()),
        aired_from: Some(now - Duration::days(100)),
        aired_to: Some(now - Duration::days(10)),
        score_weighted: Some(8.75),
        score_mean: Some(8.70),
        score_median: Some(9.0),
        details: AnimeDetailsInput {
            titles: vec![AnimeTitleInput {
                language: "en".to_owned(),
                title: "Cowboy Bebop".to_owned(),
            }],
            synonyms: vec!["Bebop".to_owned()],
            genres: vec!["Action".to_owned()],
            tags: vec!["Space".to_owned()],
            demographics: vec!["Shounen".to_owned()],
            content_warnings: vec!["Violence".to_owned()],
            themes: vec![AnimeThemeInput {
                name: "Space".to_owned(),
                description: Some("Outer space setting".to_owned()),
            }],
            images: vec![
                AnimeImageInput {
                    image_type: "cover".to_owned(),
                    url: "https://cdn.example/cover.jpg".to_owned(),
                },
                AnimeImageInput {
                    image_type: "poster".to_owned(),
                    url: "https://cdn.example/poster.jpg".to_owned(),
                },
                AnimeImageInput {
                    image_type: "banner".to_owned(),
                    url: "https://cdn.example/banner.jpg".to_owned(),
                },
            ],
        },
        schedule_and_stats: AnimeScheduleAndStatsInput {
            opening_themes: vec![AnimeThemeSongInput {
                position: 0,
                title: "Tank!".to_owned(),
                artist: Some("The Seatbelts".to_owned()),
                episode_ranges: vec![EpisodeRangeInput {
                    range_start: 1,
                    range_end: Some(26),
                }],
            }],
            ending_themes: vec![AnimeThemeSongInput {
                position: 0,
                title: "The Real Folk Blues".to_owned(),
                artist: None,
                episode_ranges: vec![EpisodeRangeInput {
                    range_start: 1,
                    range_end: None,
                }],
            }],
            trailers: vec![AnimeTrailerInput {
                source: "https://youtube.com/watch?v=1".to_owned(),
                title: Some("PV".to_owned()),
                thumbnail_source: None,
            }],
            external_sources: vec![AnimeExternalSourceInput {
                source: "https://cowboy-bebop.net".to_owned(),
                platform: "Official".to_owned(),
                label: Some("Official Site".to_owned()),
                language: Some("ja".to_owned()),
            }],
            streaming_sources: vec![AnimeStreamingSourceInput {
                platform: "Crunchyroll".to_owned(),
                source: "https://crunchyroll.com/cb".to_owned(),
                region: Some("US".to_owned()),
                is_free: Some(false),
                is_premium_required: Some(true),
                has_dub: Some(true),
                subtitle_languages: vec!["en".to_owned()],
            }],
            statistics: vec![AnimeStatisticsInput {
                provider: "mal".to_owned(),
                score: Some(8.75),
                scored_by: Some(1000),
                members: Some(5000),
                favorites: Some(500),
                rank: Some(40),
                popularity: Some(45),
                contextual_ranks: vec![AnimeContextualRankInput {
                    context: "SPRING 1998".to_owned(),
                    rank: 1,
                    is_all_time: Some(false),
                    format: Some("TV".to_owned()),
                    season: Some("SPRING".to_owned()),
                    year: Some(1998),
                }],
            }],
            broadcast: Some(AnimeBroadcastInput {
                weekday: Some(5),
                timezone: Some("Asia/Tokyo".to_owned()),
                jp_time: NaiveTime::from_hms_opt(1, 0, 0),
                sub_time: None,
                dub_time: None,
                sub_delay_days: None,
                dub_delay_days: None,
                premiere_jp: None,
                premiere_sub: None,
                premiere_dub: None,
                next_episode_at: None,
                next_episode_number: None,
            }),
            hiatus: Some(AnimeHiatusInput {
                reason: Some("Recap week".to_owned()),
                hiatus_from: NaiveDate::from_ymd_opt(1998, 6, 1),
                hiatus_until: NaiveDate::from_ymd_opt(1998, 6, 8),
            }),
        },
        relations_and_graph: AnimeRelationsAndGraphInput {
            relations: vec![AnimeRelationInput {
                relation_type: "SEQUEL".to_owned(),
                to_page_provider: Provider::Mal.to_string(),
                to_page_address: "2".to_owned(),
            }],
            relation_claims: Vec::new(),
            source_materials: vec![AnimeSourceMaterialInput {
                relation_type: "ADAPTATION".to_owned(),
                source_material_type: "MANGA".to_owned(),
                title: "Cowboy Bebop Manga".to_owned(),
                title_english: None,
                title_japanese: None,
                status: None,
                score: None,
                chapters: None,
                volumes: None,
                images: Vec::new(),
                xrefs: vec![SourceMaterialXrefInput {
                    provider: "mal".to_owned(),
                    external_id: "100".to_owned(),
                    url: "https://myanimelist.net/manga/100".to_owned(),
                }],
            }],
            companies: vec![AnimeCompanyInput {
                name: "Sunrise".to_owned(),
                description: None,
                roles: vec!["STUDIO".to_owned()],
                xrefs: vec![CompanyXrefInput {
                    provider: "mal".to_owned(),
                    url: "https://myanimelist.net/anime/producer/14".to_owned(),
                }],
            }],
        },
        episodes: vec![EpisodeReferenceInput {
            episode_id: ep_id,
            episode_number: 1,
            title: "Asteroid Blues".to_owned(),
            page_keys: Vec::new(),
            list_entries: Vec::new(),
        }],
        merged_record_hash: "cb-v1".to_owned(),
        hash_format: ANIME_HASH_FORMAT,
        merge_version: 1,
        traceparent: None,
    };
    save_anime(&pool, &save_input).await.expect("save anime");

    let visible_anime = read_visible_anime(&pool, anime_id)
        .await
        .expect("read visible anime");
    assert_eq!(visible_anime.title, "Cowboy Bebop");
    assert_eq!(visible_anime.source_pages.len(), 1);
    assert_eq!(visible_anime.partial_source_pages.len(), 1);
    assert_eq!(visible_anime.images.len(), 3);
    assert_eq!(visible_anime.opening_themes.len(), 1);
    assert_eq!(visible_anime.ending_themes.len(), 1);
    assert_eq!(
        visible_anime.streaming_sources[0].subtitle_languages,
        vec!["en"]
    );
    assert_eq!(visible_anime.statistics[0].contextual_ranks.len(), 1);
    assert!(visible_anime.broadcast.is_some());
    assert!(visible_anime.hiatus.is_some());
    assert_eq!(
        visible_anime.related_anime[0].target_anime_id,
        Some(target_anime_id)
    );
    assert_eq!(visible_anime.related_source_material.len(), 1);
    assert_eq!(visible_anime.companies[0].name, "Sunrise");

    // Episode unmerged -> NotFound, then merged -> visible
    assert!(
        read_visible_episode(&pool, ep_id)
            .await
            .expect_err("unmerged episode")
            .is_not_found()
    );
    save_episode(
        &pool,
        &SaveEpisodeInput {
            episode_id: ep_id,
            expected_version: 1,
            episode_number: 1,
            season_number: Some(1),
            title: "Asteroid Blues".to_owned(),
            title_japanese: Some("アステロイド・ブルース".to_owned()),
            title_romaji: None,
            synopsis: Some("Spike chases Asimov.".to_owned()),
            description: None,
            aired_at: Some(now - Duration::days(99)),
            duration_seconds: Some(1440),
            is_filler: false,
            is_recap: false,
            score: Some(8.9),
            titles: vec![EpisodeTitleInput {
                language: "en".to_owned(),
                title: "Asteroid Blues".to_owned(),
            }],
            images: vec!["https://cdn.example/ep1.jpg".to_owned()],
            streaming: vec![EpisodeStreamingInput {
                platform: "Crunchyroll".to_owned(),
                source: "https://crunchyroll.com/cb".to_owned(),
            }],
            characters: vec![EpisodeCharacterInput {
                character_page_provider: "mal".to_owned(),
                character_page_address: "1000".to_owned(),
                role: "MAIN".to_owned(),
            }],
            merged_record_hash: "ep1-v1".to_owned(),
            hash_format: EPISODE_HASH_FORMAT,
            merge_version: 1,
            traceparent: None,
        },
    )
    .await
    .expect("save episode");
    let visible_ep = read_visible_episode(&pool, ep_id)
        .await
        .expect("read visible episode");
    assert_eq!(visible_ep.title, "Asteroid Blues");
    assert_eq!(visible_ep.characters.len(), 1);
    assert_eq!(
        list_visible_episodes(&pool, anime_id)
            .await
            .expect("list visible episodes")
            .len(),
        1
    );

    // Character unmerged -> NotFound, then merged -> visible
    let character_id = insert_unmerged_character(&pool, "Spike Spiegel").await;
    link_character_page(&pool, character_id, "mal", "1000").await;
    assert!(
        read_visible_character(&pool, character_id)
            .await
            .expect_err("unmerged character")
            .is_not_found()
    );
    save_character(
        &pool,
        &SaveCharacterInput {
            character_id,
            expected_version: 1,
            name: "Spike Spiegel".to_owned(),
            name_native: Some("スパイク・スピーゲル".to_owned()),
            description: Some("Bounty hunter.".to_owned()),
            favorites: Some(10000),
            name_variations: vec!["Spike".to_owned()],
            nicknames: vec!["Swimming Bird".to_owned()],
            traits: vec!["Cool".to_owned()],
            images: vec!["https://cdn.example/spike.jpg".to_owned()],
            attributes: std::collections::BTreeMap::from([(
                "height".to_owned(),
                "185 cm".to_owned(),
            )]),
            spoilers: std::collections::BTreeMap::from([("ending".to_owned(), "Bang.".to_owned())]),
            animeography: vec![CharacterAppearanceInput {
                anime_page_provider: Provider::Mal.to_string(),
                anime_page_address: "1".to_owned(),
                role: "MAIN".to_owned(),
            }],
            mangaography: vec![CharacterMangaographyInput {
                source_material_provider: Provider::Mal.to_string(),
                source_material_external_id: "100".to_owned(),
                role: "MAIN".to_owned(),
            }],
            merged_record_hash: "char-v1".to_owned(),
            hash_format: 1,
            merge_version: 1,
            traceparent: None,
        },
    )
    .await
    .expect("save character");

    let visible_char = read_visible_character(&pool, character_id)
        .await
        .expect("read visible character");
    assert_eq!(visible_char.name, "Spike Spiegel");
    assert_eq!(visible_char.anime_ids, vec![(anime_id, "MAIN".to_owned())]);
    assert_eq!(
        visible_char.attributes.get("height").map(String::as_str),
        Some("185 cm")
    );
    assert_eq!(
        visible_char.spoilers.get("ending").map(String::as_str),
        Some("Bang.")
    );
}
