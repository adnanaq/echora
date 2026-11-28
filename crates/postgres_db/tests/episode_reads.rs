//! Integration tests for visible episode reads (`read_visible_episode`, `list_visible_episodes`).

mod support;

use echora_domain::{EntityType, PageKey, Provider};
use postgres_db::{
    EpisodeCharacterInput, EpisodeReferenceInput, EpisodeStreamingInput, EpisodeTitleInput,
    PageSource, VisibleEpisodeCharacter, list_visible_episodes, read_visible_episode, save_anime,
    save_character, save_episode,
};
use sqlx::PgPool;
use url::Url;
use uuid::Uuid;

use crate::support::{
    build_anime_save, build_character_save, build_episode_save, build_mal_page_source,
    insert_unmerged_anime, insert_unmerged_character, link_character_page,
};

#[sqlx::test(migrations = "./migrations")]
async fn read_visible_episode_returns_saved_fields_with_platform_and_character(pool: PgPool) {
    let episode_id = Uuid::now_v7();
    let anime_id = insert_unmerged_anime(&pool).await;
    save_anime(
        &pool,
        &build_anime_save(
            anime_id,
            1,
            "work-v1",
            vec![EpisodeReferenceInput {
                episode_id,
                episode_number: 1,
                title: "Asteroid Blues".to_owned(),
                page_sources: vec![PageSource {
                    page_key: PageKey::new(Provider::Mal, EntityType::Episode, "1:1").expect("key"),
                    source: None,
                    title: None,
                }],
                list_entries: Vec::new(),
            }],
        ),
    )
    .await
    .expect("save work");
    assert!(
        read_visible_episode(&pool, episode_id)
            .await
            .expect_err("unmerged episode")
            .is_not_found()
    );

    let character_id = insert_unmerged_character(&pool, "Spike Spiegel").await;
    link_character_page(&pool, character_id, "mal", "1").await;
    link_character_page(&pool, character_id, "anilist", "1").await;
    let mut character_save = build_character_save(character_id, "Spike Spiegel");
    character_save.page_sources = vec![
        build_mal_page_source(
            EntityType::Character,
            "1",
            "https://myanimelist.net/character/1/Spike_Spiegel",
        ),
        PageSource {
            page_key: PageKey::new(Provider::AniList, EntityType::Character, "1")
                .expect("valid page key"),
            source: Some(Url::parse("https://anilist.co/character/1").expect("valid URL")),
            title: None,
        },
    ];
    save_character(&pool, &character_save)
        .await
        .expect("save character");

    let mut episode_save = build_episode_save(episode_id, 1, "Asteroid Blues");
    episode_save.titles = vec![EpisodeTitleInput {
        language: "en".to_owned(),
        title: "Asteroid Blues".to_owned(),
    }];
    episode_save.images = vec!["https://cdn.example/ep1.jpg".to_owned()];
    episode_save.streaming = vec![EpisodeStreamingInput {
        platform: "Crunchyroll".to_owned(),
        source: "https://crunchyroll.com/cb/1".to_owned(),
    }];
    episode_save.characters = vec![
        EpisodeCharacterInput {
            character_page_source: PageSource {
                page_key: PageKey::new(Provider::Mal, EntityType::Character, "1")
                    .expect("valid page key"),
                source: Some(Url::parse("https://myanimelist.net/character/1").expect("valid URL")),
                title: Some("Spike".to_owned()),
            },
            role: "MAIN".to_owned(),
        },
        EpisodeCharacterInput {
            character_page_source: PageSource {
                page_key: PageKey::new(Provider::AniList, EntityType::Character, "1")
                    .expect("valid page key"),
                source: Some(Url::parse("https://anilist.co/character/1").expect("valid URL")),
                title: None,
            },
            role: "SUPPORTING".to_owned(),
        },
        EpisodeCharacterInput {
            character_page_source: PageSource {
                page_key: PageKey::new(Provider::Mal, EntityType::Character, "2")
                    .expect("valid page key"),
                source: Some(Url::parse("https://myanimelist.net/character/2").expect("valid URL")),
                title: Some("Jet Black".to_owned()),
            },
            role: "SUPPORTING".to_owned(),
        },
    ];
    episode_save.page_sources = vec![build_mal_page_source(
        EntityType::Episode,
        "1:1",
        "https://myanimelist.net/anime/1/Cowboy_Bebop/episode/1",
    )];
    save_episode(&pool, &episode_save)
        .await
        .expect("save episode");

    let visible_episode = read_visible_episode(&pool, episode_id)
        .await
        .expect("read episode");
    assert_eq!(visible_episode.titles, episode_save.titles);
    assert_eq!(visible_episode.images, episode_save.images);
    assert_eq!(visible_episode.streaming, episode_save.streaming);
    assert_eq!(
        visible_episode.sources,
        vec!["https://myanimelist.net/anime/1/Cowboy_Bebop/episode/1"]
    );
    assert_eq!(
        visible_episode.characters,
        vec![
            VisibleEpisodeCharacter {
                name: Some("Spike Spiegel".to_owned()),
                role: "MAIN".to_owned(),
                sources: vec![
                    "https://anilist.co/character/1".to_owned(),
                    "https://myanimelist.net/character/1/Spike_Spiegel".to_owned(),
                ],
            },
            VisibleEpisodeCharacter {
                name: Some("Jet Black".to_owned()),
                role: "SUPPORTING".to_owned(),
                sources: vec!["https://myanimelist.net/character/2".to_owned()],
            },
        ]
    );
}

#[sqlx::test(migrations = "./migrations")]
async fn list_visible_episodes_keeps_child_rows_with_their_episode(pool: PgPool) {
    let anime_id = insert_unmerged_anime(&pool).await;
    let episode_ids = [Uuid::now_v7(), Uuid::now_v7(), Uuid::now_v7()];
    let references = episode_ids
        .iter()
        .zip(1..)
        .map(|(episode_id, episode_number)| EpisodeReferenceInput {
            episode_id: *episode_id,
            episode_number,
            title: format!("Episode {episode_number}"),
            page_sources: Vec::new(),
            list_entries: Vec::new(),
        })
        .collect();
    save_anime(&pool, &build_anime_save(anime_id, 1, "work-v1", references))
        .await
        .expect("save work");
    for (episode_id, episode_number) in episode_ids.iter().zip(1..).take(2) {
        let mut episode_save = build_episode_save(
            *episode_id,
            episode_number,
            &format!("Episode {episode_number}"),
        );
        episode_save.images = vec![format!("https://cdn.example/ep{episode_number}.jpg")];
        save_episode(&pool, &episode_save)
            .await
            .expect("save episode");
    }

    let visible_episodes = list_visible_episodes(&pool, anime_id)
        .await
        .expect("list episodes");
    let episode_images: Vec<(i32, Vec<String>)> = visible_episodes
        .iter()
        .map(|episode| (episode.episode_number, episode.images.clone()))
        .collect();
    assert_eq!(
        episode_images,
        vec![
            (1, vec!["https://cdn.example/ep1.jpg".to_owned()]),
            (2, vec!["https://cdn.example/ep2.jpg".to_owned()]),
        ]
    );
    assert_eq!(
        list_visible_episodes(&pool, Uuid::now_v7())
            .await
            .expect("list for missing work"),
        Vec::new()
    );
}

#[sqlx::test(migrations = "./migrations")]
async fn read_and_list_visible_episodes_omit_dead_source_pages(pool: PgPool) {
    let episode_id = Uuid::now_v7();
    let anime_id = insert_unmerged_anime(&pool).await;
    save_anime(
        &pool,
        &build_anime_save(
            anime_id,
            1,
            "work-dead-ep-v1",
            vec![EpisodeReferenceInput {
                episode_id,
                episode_number: 1,
                title: "Asteroid Blues".to_owned(),
                page_sources: vec![PageSource {
                    page_key: PageKey::new(Provider::Mal, EntityType::Episode, "1:1").expect("key"),
                    source: None,
                    title: None,
                }],
                list_entries: Vec::new(),
            }],
        ),
    )
    .await
    .expect("save work");
    sqlx::query(
        "INSERT INTO provider_page (provider, entity_type, address, source, is_dead)
         VALUES ('anidb', 'episode', '999', 'https://anidb.net/episode/999', true)",
    )
    .execute(&pool)
    .await
    .expect("insert dead episode page");
    sqlx::query(
        "INSERT INTO episode_source (episode_id, provider, entity_type, address)
         VALUES ($1, 'anidb', 'episode', '999')",
    )
    .bind(episode_id)
    .execute(&pool)
    .await
    .expect("link dead episode page");

    let mut episode_save = build_episode_save(episode_id, 1, "Asteroid Blues");
    episode_save.page_sources = vec![build_mal_page_source(
        EntityType::Episode,
        "1:1",
        "https://myanimelist.net/anime/1/Cowboy_Bebop/episode/1",
    )];
    save_episode(&pool, &episode_save)
        .await
        .expect("save episode");

    let visible_episode = read_visible_episode(&pool, episode_id)
        .await
        .expect("read episode");
    assert_eq!(
        visible_episode.sources,
        vec!["https://myanimelist.net/anime/1/Cowboy_Bebop/episode/1"]
    );
    let listed = list_visible_episodes(&pool, anime_id)
        .await
        .expect("list episodes");
    assert_eq!(listed.len(), 1);
    assert_eq!(
        listed[0].sources,
        vec!["https://myanimelist.net/anime/1/Cowboy_Bebop/episode/1"]
    );
}
