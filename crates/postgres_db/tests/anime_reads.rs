//! Integration tests for visible work reads (`read_visible_anime`).

#![allow(
    clippy::expect_used,
    reason = "a failed setup step stops the test, as a test's own expect does"
)]

mod support;

use chrono::{Duration, NaiveDate, NaiveTime, Utc};
use postgres_db::{
    AnimeBroadcastInput, AnimeCompanyInput, AnimeContextualRankInput, AnimeDetailsInput,
    AnimeExternalSourceInput, AnimeHiatusInput, AnimeImageInput, AnimeRelationInput,
    AnimeRelationsAndGraphInput, AnimeScheduleAndStatsInput, AnimeSourceMaterialInput,
    AnimeStatisticsInput, AnimeStreamingSourceInput, AnimeThemeInput, AnimeThemeSongInput,
    AnimeTitleInput, AnimeTrailerInput, CompanyXrefInput, EntityType, EpisodeRangeInput, PageKey,
    PageSource, Provider, RelatedAnimeSummary, SourceMaterialXrefInput, VisibleAnimeRelation,
    read_visible_anime, save_anime,
};
use sqlx::PgPool;
use uuid::Uuid;

use crate::support::{
    build_anime_save, build_mal_page_source, insert_unmerged_anime, link_mal_anime_pages,
};

fn build_opening_themes() -> Vec<AnimeThemeSongInput> {
    vec![
        AnimeThemeSongInput {
            position: 0,
            title: "Tank!".to_owned(),
            artist: Some("The Seatbelts".to_owned()),
            episode_ranges: vec![
                EpisodeRangeInput {
                    range_start: 1,
                    range_end: Some(12),
                },
                EpisodeRangeInput {
                    range_start: 14,
                    range_end: Some(26),
                },
            ],
        },
        AnimeThemeSongInput {
            position: 1,
            title: "Tank! (TV edit)".to_owned(),
            artist: None,
            episode_ranges: Vec::new(),
        },
    ]
}

fn build_full_schedule_and_stats() -> AnimeScheduleAndStatsInput {
    AnimeScheduleAndStatsInput {
        opening_themes: build_opening_themes(),
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
        streaming_sources: vec![
            AnimeStreamingSourceInput {
                platform: "Crunchyroll".to_owned(),
                source: "https://crunchyroll.com/cb".to_owned(),
                region: Some("US".to_owned()),
                is_free: Some(false),
                is_premium_required: Some(true),
                has_dub: Some(true),
                subtitle_languages: vec!["de".to_owned(), "en".to_owned()],
            },
            AnimeStreamingSourceInput {
                platform: "Netflix".to_owned(),
                source: "https://netflix.com/cb".to_owned(),
                region: None,
                is_free: None,
                is_premium_required: None,
                has_dub: None,
                subtitle_languages: Vec::new(),
            },
        ],
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
    }
}

#[sqlx::test(migrations = "./migrations")]
async fn read_visible_anime_returns_every_saved_field(pool: PgPool) {
    let now = Utc::now();
    let anime_id = insert_unmerged_anime(&pool).await;
    link_mal_anime_pages(&pool, anime_id, &["1"]).await;

    let details = AnimeDetailsInput {
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
                image_type: "banner".to_owned(),
                url: "https://cdn.example/banner.jpg".to_owned(),
            },
            AnimeImageInput {
                image_type: "cover".to_owned(),
                url: "https://cdn.example/cover.jpg".to_owned(),
            },
        ],
    };
    let source_material = AnimeSourceMaterialInput {
        relation_type: "ADAPTATION".to_owned(),
        source_material_type: "4-KOMA".to_owned(),
        title: "Cowboy Bebop Manga".to_owned(),
        title_english: Some("Cowboy Bebop (Manga)".to_owned()),
        title_japanese: None,
        status: Some("FINISHED".to_owned()),
        score: Some(7.5),
        chapters: Some(20),
        volumes: Some(3),
        images: vec!["https://cdn.example/manga.jpg".to_owned()],
        xrefs: vec![
            SourceMaterialXrefInput {
                provider: "anilist".to_owned(),
                external_id: "30100".to_owned(),
                url: "https://anilist.co/manga/30100".to_owned(),
            },
            SourceMaterialXrefInput {
                provider: "mal".to_owned(),
                external_id: "100".to_owned(),
                url: "https://myanimelist.net/manga/100".to_owned(),
            },
        ],
    };
    let company = AnimeCompanyInput {
        name: "Sunrise".to_owned(),
        description: Some("Studio".to_owned()),
        roles: vec!["PRODUCER".to_owned(), "STUDIO".to_owned()],
        xrefs: vec![CompanyXrefInput {
            provider: "mal".to_owned(),
            url: "https://myanimelist.net/anime/producer/14".to_owned(),
        }],
    };
    let mut save_input = build_anime_save(anime_id, 1, "cb-v1", Vec::new());
    save_input.title = "Cowboy Bebop".to_owned();
    save_input.title_english = Some("Cowboy Bebop".to_owned());
    save_input.anime_type = "TV SPECIAL".to_owned();
    save_input.status = "FINISHED".to_owned();
    save_input.rating = "PG-13 - Teens 13 or older".to_owned();
    save_input.season = Some("SPRING".to_owned());
    save_input.source_material = Some("ORIGINAL".to_owned());
    save_input.year = Some(1998);
    save_input.month = Some(4);
    save_input.episode_count = 26;
    save_input.aired_from = Some(now - Duration::days(100));
    save_input.score_weighted = Some(8.75);
    save_input.details = details.clone();
    save_input.schedule_and_stats = build_full_schedule_and_stats();
    save_input.relations_and_graph = AnimeRelationsAndGraphInput {
        source_materials: vec![source_material.clone()],
        companies: vec![company.clone()],
        ..AnimeRelationsAndGraphInput::default()
    };
    save_input.page_sources = vec![build_mal_page_source(
        EntityType::Anime,
        "1",
        "https://myanimelist.net/anime/1/Cowboy_Bebop",
    )];
    save_anime(&pool, &save_input).await.expect("save anime");

    let visible_anime = read_visible_anime(&pool, anime_id)
        .await
        .expect("read visible anime");
    assert_eq!(visible_anime.anime_type, "TV SPECIAL");
    assert_eq!(visible_anime.rating, "PG-13 - Teens 13 or older");
    assert_eq!(visible_anime.score_weighted, Some(8.75));
    assert_eq!(visible_anime.titles, details.titles);
    assert_eq!(visible_anime.themes, details.themes);
    assert_eq!(visible_anime.images, details.images);
    assert_eq!(
        visible_anime.sources,
        vec!["https://myanimelist.net/anime/1/Cowboy_Bebop"]
    );
    assert_eq!(visible_anime.partial_sources, Vec::<String>::new());
    let saved_schedule = build_full_schedule_and_stats();
    assert_eq!(visible_anime.opening_themes, saved_schedule.opening_themes);
    assert_eq!(visible_anime.ending_themes, saved_schedule.ending_themes);
    assert_eq!(visible_anime.trailers, saved_schedule.trailers);
    assert_eq!(
        visible_anime.external_sources,
        saved_schedule.external_sources
    );
    assert_eq!(
        visible_anime.streaming_sources,
        saved_schedule.streaming_sources
    );
    assert_eq!(visible_anime.statistics, saved_schedule.statistics);
    assert_eq!(visible_anime.broadcast, saved_schedule.broadcast);
    assert_eq!(visible_anime.hiatus, saved_schedule.hiatus);
    assert_eq!(visible_anime.related_source_material, vec![source_material]);
    assert_eq!(visible_anime.companies, vec![company]);
}

#[sqlx::test(migrations = "./migrations")]
async fn read_visible_anime_unmerged_or_deleted_returns_not_found(pool: PgPool) {
    let anime_id = insert_unmerged_anime(&pool).await;
    assert!(
        read_visible_anime(&pool, anime_id)
            .await
            .expect_err("unmerged work")
            .is_not_found()
    );
    save_anime(&pool, &build_anime_save(anime_id, 1, "v1", Vec::new()))
        .await
        .expect("save anime");
    sqlx::query("UPDATE anime SET deleted_at = now() WHERE id = $1")
        .bind(anime_id)
        .execute(&pool)
        .await
        .expect("delete work");
    assert!(
        read_visible_anime(&pool, anime_id)
            .await
            .expect_err("deleted work")
            .is_not_found()
    );
    assert!(
        read_visible_anime(&pool, Uuid::now_v7())
            .await
            .expect_err("missing work")
            .is_not_found()
    );
}

#[sqlx::test(migrations = "./migrations")]
async fn read_visible_anime_relations_resolve_only_other_visible_works(pool: PgPool) {
    let anime_id = insert_unmerged_anime(&pool).await;
    let sequel_id = insert_unmerged_anime(&pool).await;
    let unmerged_id = insert_unmerged_anime(&pool).await;
    link_mal_anime_pages(&pool, anime_id, &["1"]).await;
    link_mal_anime_pages(&pool, sequel_id, &["2"]).await;
    link_mal_anime_pages(&pool, unmerged_id, &["3"]).await;
    sqlx::query(
        "INSERT INTO provider_page (provider, entity_type, address)
         VALUES ('mal', 'anime', '4'), ('mal', 'anime', '5'), ('anilist', 'anime', '22')",
    )
    .execute(&pool)
    .await
    .expect("insert unowned pages");
    sqlx::query(
        "INSERT INTO anime_source (anime_id, provider, entity_type, address)
         VALUES ($1, 'anilist', 'anime', '22')",
    )
    .bind(sequel_id)
    .execute(&pool)
    .await
    .expect("link second sequel page");

    let mut sequel_save = build_anime_save(sequel_id, 1, "sequel-v1", Vec::new());
    sequel_save.title = "Sequel".to_owned();
    sequel_save.anime_type = "MOVIE".to_owned();
    sequel_save.status = "UPCOMING".to_owned();
    sequel_save.year = Some(2027);
    sequel_save.episode_count = 1;
    sequel_save.page_sources = vec![
        build_mal_page_source(
            EntityType::Anime,
            "2",
            "https://myanimelist.net/anime/2/Sequel",
        ),
        PageSource {
            page_key: PageKey::new(Provider::AniList, EntityType::Anime, "22")
                .expect("valid page key"),
            source: "https://anilist.co/anime/22".to_owned(),
        },
    ];
    save_anime(&pool, &sequel_save).await.expect("save sequel");

    let relation = |relation_type: &str, address: &str| AnimeRelationInput {
        relation_type: relation_type.to_owned(),
        to_page_provider: "mal".to_owned(),
        to_page_address: address.to_owned(),
    };
    let anilist_relation = AnimeRelationInput {
        relation_type: "SEQUEL".to_owned(),
        to_page_provider: "anilist".to_owned(),
        to_page_address: "22".to_owned(),
    };
    let mut save_input = build_anime_save(anime_id, 1, "work-v1", Vec::new());
    save_input.relations_and_graph = AnimeRelationsAndGraphInput {
        relations: vec![
            relation("SEQUEL", "2"),
            anilist_relation,
            relation("SIDE_STORY", "3"),
            relation("SPIN_OFF", "4"),
            relation("SPIN_OFF", "5"),
            relation("SUMMARY", "1"),
        ],
        ..AnimeRelationsAndGraphInput::default()
    };
    save_anime(&pool, &save_input).await.expect("save work");

    let visible_anime = read_visible_anime(&pool, anime_id)
        .await
        .expect("read work");
    assert_eq!(
        visible_anime.related_anime,
        vec![
            VisibleAnimeRelation {
                relation_type: "SEQUEL".to_owned(),
                target_page_sources: vec![
                    "https://anilist.co/anime/22".to_owned(),
                    "https://myanimelist.net/anime/2/Sequel".to_owned(),
                ],
                target_anime: Some(RelatedAnimeSummary {
                    anime_id: sequel_id,
                    title: "Sequel".to_owned(),
                    anime_type: "MOVIE".to_owned(),
                    status: "UPCOMING".to_owned(),
                    year: Some(2027),
                    episode_count: 1,
                }),
            },
            VisibleAnimeRelation {
                relation_type: "SIDE_STORY".to_owned(),
                target_page_sources: Vec::new(),
                target_anime: None,
            },
            VisibleAnimeRelation {
                relation_type: "SPIN_OFF".to_owned(),
                target_page_sources: Vec::new(),
                target_anime: None,
            },
            VisibleAnimeRelation {
                relation_type: "SPIN_OFF".to_owned(),
                target_page_sources: Vec::new(),
                target_anime: None,
            },
            VisibleAnimeRelation {
                relation_type: "SUMMARY".to_owned(),
                target_page_sources: Vec::new(),
                target_anime: None,
            },
        ]
    );
}

#[sqlx::test(migrations = "./migrations")]
async fn read_visible_anime_partial_source_url_comes_from_owner_save(pool: PgPool) {
    let owner_id = insert_unmerged_anime(&pool).await;
    let part_id = insert_unmerged_anime(&pool).await;
    link_mal_anime_pages(&pool, owner_id, &["10"]).await;
    link_mal_anime_pages(&pool, part_id, &["11"]).await;
    sqlx::query(
        "INSERT INTO anime_partial_source (anime_id, provider, entity_type, address)
         VALUES ($1, 'mal', 'anime', '10')",
    )
    .bind(part_id)
    .execute(&pool)
    .await
    .expect("link partial source");

    let mut part_save = build_anime_save(part_id, 1, "part-v1", Vec::new());
    part_save.page_sources = vec![
        build_mal_page_source(
            EntityType::Anime,
            "11",
            "https://myanimelist.net/anime/11/Part",
        ),
        build_mal_page_source(EntityType::Anime, "10", "https://example.org/not-the-owner"),
    ];
    save_anime(&pool, &part_save).await.expect("save part");
    let part_before_owner = read_visible_anime(&pool, part_id).await.expect("read part");
    assert_eq!(
        part_before_owner.sources,
        vec!["https://myanimelist.net/anime/11/Part"]
    );
    assert_eq!(part_before_owner.partial_sources, Vec::<String>::new());

    let mut owner_save = build_anime_save(owner_id, 1, "owner-v1", Vec::new());
    owner_save.page_sources = vec![build_mal_page_source(
        EntityType::Anime,
        "10",
        "https://myanimelist.net/anime/10/Whole_Work",
    )];
    save_anime(&pool, &owner_save).await.expect("save owner");
    let part_after_owner = read_visible_anime(&pool, part_id).await.expect("read part");
    assert_eq!(
        part_after_owner.partial_sources,
        vec!["https://myanimelist.net/anime/10/Whole_Work"]
    );
}
