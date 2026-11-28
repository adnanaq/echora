//! Integration tests for visible character reads (`read_visible_character`).

mod support;

use std::collections::BTreeMap;

use echora_domain::{EntityType, PageKey, Provider};
use postgres_db::{
    AnimeSourceMaterialInput, CharacterAnime, CharacterAppearanceInput, CharacterMangaographyInput,
    PageSource, SourceMaterialXrefInput, VisibleOgraphyEntry, read_visible_character, save_anime,
    save_character,
};
use sqlx::PgPool;
use url::Url;

use crate::support::{
    build_anime_save, build_character_save, build_mal_page_source, insert_unmerged_anime,
    insert_unmerged_character, link_character_page, link_mal_anime_pages,
};

#[sqlx::test(migrations = "./migrations")]
async fn read_visible_character_lists_only_visible_works_with_titles(pool: PgPool) {
    let visible_anime_id = insert_unmerged_anime(&pool).await;
    let unmerged_anime_id = insert_unmerged_anime(&pool).await;
    link_mal_anime_pages(&pool, visible_anime_id, &["1"]).await;
    link_mal_anime_pages(&pool, unmerged_anime_id, &["2"]).await;
    let mut work_save = build_anime_save(visible_anime_id, 1, "work-v1", Vec::new());
    work_save.title = "Cowboy Bebop".to_owned();
    sqlx::query("INSERT INTO provider_page (provider, entity_type, address) VALUES ('anilist', 'anime', '1')")
        .execute(&pool)
        .await
        .expect("insert anilist page");
    sqlx::query(
        "INSERT INTO anime_source (anime_id, provider, entity_type, address)
         VALUES ($1, 'anilist', 'anime', '1')",
    )
    .bind(visible_anime_id)
    .execute(&pool)
    .await
    .expect("link anilist page");
    work_save.page_sources = vec![
        build_mal_page_source(
            EntityType::Anime,
            "1",
            "https://myanimelist.net/anime/1/Cowboy_Bebop",
        ),
        PageSource {
            page_key: PageKey::new(Provider::AniList, EntityType::Anime, "1")
                .expect("valid page key"),
            source: Some(Url::parse("https://anilist.co/anime/1").expect("valid URL")),
            title: None,
        },
    ];
    work_save.relations_and_graph.source_materials = vec![AnimeSourceMaterialInput {
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
        xrefs: vec![
            SourceMaterialXrefInput {
                provider: "anilist".to_owned(),
                external_id: "30100".to_owned(),
                url: "https://anilist.co/manga/30100".to_owned(),
                title: None,
            },
            SourceMaterialXrefInput {
                provider: "mal".to_owned(),
                external_id: "100".to_owned(),
                url: "https://myanimelist.net/manga/100".to_owned(),
                title: None,
            },
        ],
    }];
    save_anime(&pool, &work_save).await.expect("save work");

    let character_id = insert_unmerged_character(&pool, "Spike Spiegel").await;
    link_character_page(&pool, character_id, "mal", "1").await;
    assert!(
        read_visible_character(&pool, character_id)
            .await
            .expect_err("unmerged character")
            .is_not_found()
    );
    let appearance = |address: &str, role: &str| CharacterAppearanceInput {
        anime_page_source: PageSource {
            page_key: PageKey::new(Provider::Mal, EntityType::Anime, address)
                .expect("valid page key"),
            source: Some(
                Url::parse(&format!("https://myanimelist.net/anime/{address}")).expect("valid URL"),
            ),
            title: Some(format!("Anime {address}")),
        },
        role: role.to_owned(),
    };
    let mut character_save = build_character_save(character_id, "Spike Spiegel");
    character_save.name_native = Some("スパイク・スピーゲル".to_owned());
    character_save.description = Some("Bounty hunter.".to_owned());
    character_save.favorites = Some(10_000);
    character_save.name_variations = vec!["Spike".to_owned()];
    character_save.nicknames = vec!["Swimming Bird".to_owned()];
    character_save.traits = vec!["Cool".to_owned()];
    character_save.images = vec!["https://cdn.example/spike.jpg".to_owned()];
    character_save.attributes = BTreeMap::from([
        ("age".to_owned(), "27".to_owned()),
        ("height".to_owned(), "185 cm".to_owned()),
    ]);
    character_save.spoilers = BTreeMap::from([("ending".to_owned(), "Bang.".to_owned())]);
    character_save.page_sources = vec![build_mal_page_source(
        EntityType::Character,
        "1",
        "https://myanimelist.net/character/1/Spike_Spiegel",
    )];
    character_save.animeography = vec![
        appearance("1", "MAIN"),
        appearance("2", "SUPPORTING"),
        CharacterAppearanceInput {
            anime_page_source: PageSource {
                page_key: PageKey::new(Provider::AniList, EntityType::Anime, "1")
                    .expect("valid page key"),
                source: Some(Url::parse("https://anilist.co/anime/1").expect("valid URL")),
                title: None,
            },
            role: "SUPPORTING".to_owned(),
        },
    ];
    character_save.mangaography = vec![
        CharacterMangaographyInput {
            role: "MAIN".to_owned(),
            xrefs: vec![SourceMaterialXrefInput {
                provider: "mal".to_owned(),
                external_id: "100".to_owned(),
                url: "https://myanimelist.net/manga/100".to_owned(),
                title: Some("Manga 100".to_owned()),
            }],
        },
        CharacterMangaographyInput {
            role: "BACKGROUND".to_owned(),
            xrefs: vec![SourceMaterialXrefInput {
                provider: "mal".to_owned(),
                external_id: "999".to_owned(),
                url: "https://myanimelist.net/manga/999/Shooting_Star_Bebop".to_owned(),
                title: Some("Shooting Star Bebop".to_owned()),
            }],
        },
        CharacterMangaographyInput {
            role: "SUPPORTING".to_owned(),
            xrefs: vec![SourceMaterialXrefInput {
                provider: "anilist".to_owned(),
                external_id: "30100".to_owned(),
                url: "https://anilist.co/manga/30100".to_owned(),
                title: None,
            }],
        },
    ];
    save_character(&pool, &character_save)
        .await
        .expect("save character");

    let visible_character = read_visible_character(&pool, character_id)
        .await
        .expect("read character");
    assert_eq!(visible_character.name_native, character_save.name_native);
    assert_eq!(visible_character.description, character_save.description);
    assert_eq!(visible_character.favorites, character_save.favorites);
    assert_eq!(
        visible_character.name_variations,
        character_save.name_variations
    );
    assert_eq!(visible_character.nicknames, character_save.nicknames);
    assert_eq!(visible_character.traits, character_save.traits);
    assert_eq!(visible_character.images, character_save.images);
    assert_eq!(visible_character.attributes, character_save.attributes);
    assert_eq!(visible_character.spoilers, character_save.spoilers);
    assert_eq!(
        visible_character.sources,
        vec!["https://myanimelist.net/character/1/Spike_Spiegel"]
    );
    assert_eq!(
        visible_character.anime,
        vec![CharacterAnime {
            anime_id: visible_anime_id,
            role: "MAIN".to_owned(),
        }]
    );
    assert_eq!(
        visible_character.animeography,
        vec![
            VisibleOgraphyEntry {
                title: Some("Cowboy Bebop".to_owned()),
                role: "MAIN".to_owned(),
                sources: vec![
                    "https://anilist.co/anime/1".to_owned(),
                    "https://myanimelist.net/anime/1/Cowboy_Bebop".to_owned(),
                ],
            },
            VisibleOgraphyEntry {
                title: Some("Anime 2".to_owned()),
                role: "SUPPORTING".to_owned(),
                sources: vec!["https://myanimelist.net/anime/2".to_owned()],
            },
        ]
    );
    assert_eq!(
        visible_character.mangaography,
        vec![
            VisibleOgraphyEntry {
                title: Some("Cowboy Bebop Manga".to_owned()),
                role: "MAIN".to_owned(),
                sources: vec![
                    "https://anilist.co/manga/30100".to_owned(),
                    "https://myanimelist.net/manga/100".to_owned(),
                ],
            },
            VisibleOgraphyEntry {
                title: Some("Shooting Star Bebop".to_owned()),
                role: "BACKGROUND".to_owned(),
                sources: vec!["https://myanimelist.net/manga/999/Shooting_Star_Bebop".to_owned()],
            },
        ]
    );
}

#[sqlx::test(migrations = "./migrations")]
async fn read_visible_character_omits_dead_source_pages_from_sources(pool: PgPool) {
    let character_id = insert_unmerged_character(&pool, "Spike Spiegel").await;
    link_character_page(&pool, character_id, "mal", "1").await;
    sqlx::query(
        "INSERT INTO provider_page (provider, entity_type, address, source, is_dead)
         VALUES ('anidb', 'character', '474', 'https://anidb.net/character/474', true)",
    )
    .execute(&pool)
    .await
    .expect("insert dead character page");
    sqlx::query(
        "INSERT INTO character_source (character_id, provider, entity_type, address)
         VALUES ($1, 'anidb', 'character', '474')",
    )
    .bind(character_id)
    .execute(&pool)
    .await
    .expect("link dead character page");

    let mut character_save = build_character_save(character_id, "Spike Spiegel");
    character_save.page_sources = vec![build_mal_page_source(
        EntityType::Character,
        "1",
        "https://myanimelist.net/character/1/Spike_Spiegel",
    )];
    save_character(&pool, &character_save)
        .await
        .expect("save character");

    let visible_character = read_visible_character(&pool, character_id)
        .await
        .expect("read character");
    assert_eq!(
        visible_character.sources,
        vec!["https://myanimelist.net/character/1/Spike_Spiegel"]
    );
}
