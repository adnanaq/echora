//! Integration tests for character loads, saves and the works a character appears in.

mod support;

use std::collections::BTreeMap;

use chrono::{Duration, Utc};
use postgres_db::{
    CHARACTER_HASH_FORMAT, CharacterAnime, CharacterAppearanceInput, CharacterMangaographyInput,
    EntityType, FetchOutcome, PageKey, PageObservation, PageResponse, Provider, SaveCharacterInput,
    SaveCharacterOutcome, SourceMaterialXrefInput, link_anime_source_page, list_character_anime,
    load_character, record_page_observation, save_character,
};
use sqlx::PgPool;
use uuid::Uuid;

use crate::support::{insert_unmerged_anime, insert_unmerged_character, link_character_page};

fn build_character_save(
    character_id: Uuid,
    expected_version: i64,
    merged_record_hash: &str,
) -> SaveCharacterInput {
    SaveCharacterInput {
        character_id,
        expected_version,
        name: "Monkey D. Luffy".to_owned(),
        name_native: None,
        description: None,
        favorites: None,
        name_variations: Vec::new(),
        nicknames: Vec::new(),
        traits: Vec::new(),
        images: Vec::new(),
        attributes: BTreeMap::new(),
        spoilers: BTreeMap::new(),
        animeography: Vec::new(),
        mangaography: Vec::new(),
        merged_record_hash: merged_record_hash.to_owned(),
        hash_format: CHARACTER_HASH_FORMAT,
        merge_version: 1,
        traceparent: None,
    }
}

fn build_appearance(provider: &str, address: &str, role: &str) -> CharacterAppearanceInput {
    CharacterAppearanceInput {
        anime_page_provider: provider.to_owned(),
        anime_page_address: address.to_owned(),
        role: role.to_owned(),
    }
}

async fn insert_anime_with_source_page(
    pool: &PgPool,
    provider: Provider,
    address: &str,
) -> Result<Uuid, postgres_db::Error> {
    let anime_id = insert_unmerged_anime(pool).await;
    let page_key = PageKey::new(provider, EntityType::Anime, address)?;
    link_anime_source_page(pool, anime_id, &page_key).await?;
    Ok(anime_id)
}

async fn read_character_state(
    pool: &PgPool,
    character_id: Uuid,
) -> Result<(i64, bool, bool), sqlx::Error> {
    sqlx::query_as(
        "SELECT version, merged_at IS NOT NULL, merge_requested_at IS NOT NULL
         FROM character WHERE id = $1",
    )
    .bind(character_id)
    .fetch_one(pool)
    .await
}

async fn read_outbox_subjects(
    pool: &PgPool,
    character_id: Uuid,
) -> Result<Vec<(String, i64)>, sqlx::Error> {
    sqlx::query_as("SELECT subject, version FROM outbox WHERE entity_id = $1 ORDER BY id")
        .bind(character_id)
        .fetch_all(pool)
        .await
}

async fn read_character_edges(
    pool: &PgPool,
    character_id: Uuid,
) -> Result<Vec<(Uuid, Uuid, String)>, sqlx::Error> {
    sqlx::query_as(
        "SELECT from_id, to_id, edge_type FROM graph_edge
         WHERE from_id = $1 OR to_id = $1
         ORDER BY edge_type, from_id, to_id",
    )
    .bind(character_id)
    .fetch_all(pool)
    .await
}

#[sqlx::test(migrations = "./migrations")]
async fn load_character_returns_version_fetched_pages_and_corrections(pool: PgPool) {
    let character_id = insert_unmerged_character(&pool, "Monkey D. Luffy").await;
    let fetched_page =
        PageKey::new(Provider::Mal, EntityType::Character, "40").expect("valid page key");
    record_page_observation(
        &pool,
        PageObservation {
            page_key: fetched_page.clone(),
            observed_at: Utc::now() - Duration::hours(1),
            outcome: FetchOutcome::Success {
                responses: vec![PageResponse {
                    request_name: "main".to_owned(),
                    content_type: "text/html".to_owned(),
                    body: b"<h1>Luffy</h1>".to_vec(),
                }],
            },
        },
    )
    .await
    .expect("store the fetched page");
    link_character_page(&pool, character_id, "mal", "40").await;
    link_character_page(&pool, character_id, "anilist", "40").await;
    sqlx::query(
        "INSERT INTO character_field_correction
             (character_id, field_path, value_message, value, approved_by)
         VALUES ($1, 'name', 'google.protobuf.StringValue', '\\x0a024c75', 'reviewer')",
    )
    .bind(character_id)
    .execute(&pool)
    .await
    .expect("store a correction");

    let loaded = load_character(&pool, character_id)
        .await
        .expect("load the character");

    assert_eq!(loaded.version, 1);
    assert_eq!(
        loaded
            .pages
            .iter()
            .map(|page| (page.page_key.clone(), page.responses.len()))
            .collect::<Vec<_>>(),
        vec![(fetched_page, 1)]
    );
    assert_eq!(
        loaded.pages[0].responses[0].body,
        b"<h1>Luffy</h1>".to_vec()
    );
    assert_eq!(
        loaded
            .corrections
            .iter()
            .map(|correction| (correction.field_path.as_str(), correction.value.as_slice()))
            .collect::<Vec<_>>(),
        vec![("name", b"\x0a\x02Lu".as_slice())]
    );
}

#[sqlx::test(migrations = "./migrations")]
async fn load_character_missing_or_deleted_character_returns_not_found(pool: PgPool) {
    let deleted_character_id = insert_unmerged_character(&pool, "Deleted").await;
    sqlx::query("UPDATE character SET deleted_at = now() WHERE id = $1")
        .bind(deleted_character_id)
        .execute(&pool)
        .await
        .expect("delete the character");

    let missing_error = load_character(&pool, Uuid::now_v7())
        .await
        .expect_err("an unknown character is not found");
    let deleted_error = load_character(&pool, deleted_character_id)
        .await
        .expect_err("a deleted character is not found");

    assert!(missing_error.is_not_found());
    assert!(deleted_error.is_not_found());
}

#[sqlx::test(migrations = "./migrations")]
async fn save_character_first_save_stores_every_field_and_sends_created(pool: PgPool) {
    let anime_id = insert_anime_with_source_page(&pool, Provider::Mal, "21")
        .await
        .expect("store the work with its source page");
    let character_id = insert_unmerged_character(&pool, "Luffy").await;
    let mut input = build_character_save(character_id, 1, "first-hash");
    input.name_native = Some("モンキー・D・ルフィ".to_owned());
    input.description = Some("Captain of the Straw Hats".to_owned());
    input.favorites = Some(150_000);
    input.name_variations = vec!["Luffy".to_owned(), "Luffy".to_owned()];
    input.nicknames = vec!["Straw Hat".to_owned()];
    input.traits = vec!["Rubber".to_owned(), "Captain".to_owned()];
    input.images = vec!["https://cdn.example/luffy.jpg".to_owned()];
    input.attributes = BTreeMap::from([("Age".to_owned(), "19".to_owned())]);
    input.spoilers = BTreeMap::from([
        ("description".to_owned(), "Son of Dragon".to_owned()),
        ("Age".to_owned(), "20".to_owned()),
    ]);
    input.animeography = vec![build_appearance("mal", "21", "MAIN")];

    let saved = save_character(&pool, &input)
        .await
        .expect("save the character");

    assert_eq!(
        saved,
        SaveCharacterOutcome {
            has_changed: true,
            version: 2,
        }
    );
    let stored_row: (
        String,
        Option<String>,
        Option<String>,
        Option<i32>,
        i16,
        i32,
    ) = sqlx::query_as(
        "SELECT name, name_native, description, favorites, hash_format, merge_version
             FROM character WHERE id = $1",
    )
    .bind(character_id)
    .fetch_one(&pool)
    .await
    .expect("read the character row");
    assert_eq!(
        stored_row,
        (
            "Monkey D. Luffy".to_owned(),
            Some("モンキー・D・ルフィ".to_owned()),
            Some("Captain of the Straw Hats".to_owned()),
            Some(150_000),
            CHARACTER_HASH_FORMAT,
            1,
        )
    );
    let stored_details: (Vec<String>, Vec<String>, Vec<String>, Vec<String>) = sqlx::query_as(
        "SELECT
             ARRAY(SELECT name_variation FROM character_name_variation WHERE character_id = $1 ORDER BY 1),
             ARRAY(SELECT nickname FROM character_nickname WHERE character_id = $1 ORDER BY 1),
             ARRAY(SELECT trait FROM character_trait WHERE character_id = $1 ORDER BY 1),
             ARRAY(SELECT source FROM character_image WHERE character_id = $1 ORDER BY 1)",
    )
    .bind(character_id)
    .fetch_one(&pool)
    .await
    .expect("read the character's lists");
    assert_eq!(
        stored_details,
        (
            vec!["Luffy".to_owned()],
            vec!["Straw Hat".to_owned()],
            vec!["Captain".to_owned(), "Rubber".to_owned()],
            vec!["https://cdn.example/luffy.jpg".to_owned()],
        )
    );
    let stored_maps: (Vec<String>, Vec<String>) = sqlx::query_as(
        "SELECT
             ARRAY(SELECT attribute_name || '=' || attribute_value FROM character_attribute
                   WHERE character_id = $1 ORDER BY 1),
             ARRAY(SELECT spoiler_name || '=' || spoiler_value FROM character_spoiler
                   WHERE character_id = $1 ORDER BY 1)",
    )
    .bind(character_id)
    .fetch_one(&pool)
    .await
    .expect("read the character's maps");
    assert_eq!(
        stored_maps,
        (
            vec!["Age=19".to_owned()],
            vec!["Age=20".to_owned(), "description=Son of Dragon".to_owned()],
        )
    );
    assert_eq!(
        read_character_state(&pool, character_id)
            .await
            .expect("read the state"),
        (2, true, false)
    );
    assert_eq!(
        read_outbox_subjects(&pool, character_id)
            .await
            .expect("read the outbox"),
        vec![("character.created".to_owned(), 2)]
    );
    assert_eq!(
        list_character_anime(&pool, character_id)
            .await
            .expect("list the works"),
        vec![CharacterAnime {
            anime_id,
            role: "MAIN".to_owned(),
        }]
    );
    assert_eq!(
        read_character_edges(&pool, character_id)
            .await
            .expect("read the edges"),
        vec![
            (character_id, anime_id, "APPEARS_IN".to_owned()),
            (anime_id, character_id, "FEATURES".to_owned()),
        ]
    );
}

#[sqlx::test(migrations = "./migrations")]
async fn save_character_unchanged_hash_writes_nothing_and_clears_merge_request(pool: PgPool) {
    let character_id = insert_unmerged_character(&pool, "Luffy").await;
    let first_input = build_character_save(character_id, 1, "same-hash");
    save_character(&pool, &first_input)
        .await
        .expect("first save");
    sqlx::query("UPDATE character SET merge_requested_at = now() WHERE id = $1")
        .bind(character_id)
        .execute(&pool)
        .await
        .expect("request a merge");

    let mut repeated_input = build_character_save(character_id, 2, "same-hash");
    repeated_input.name = "Changed name with the same hash".to_owned();
    repeated_input.merge_version = 2;
    let repeated = save_character(&pool, &repeated_input)
        .await
        .expect("repeated save");

    assert_eq!(
        repeated,
        SaveCharacterOutcome {
            has_changed: false,
            version: 2,
        }
    );
    let stored: (String, i32) =
        sqlx::query_as("SELECT name, merge_version FROM character WHERE id = $1")
            .bind(character_id)
            .fetch_one(&pool)
            .await
            .expect("read the character");
    assert_eq!(stored, ("Monkey D. Luffy".to_owned(), 2));
    assert_eq!(
        read_character_state(&pool, character_id)
            .await
            .expect("read the state"),
        (2, true, false)
    );
    assert_eq!(
        read_outbox_subjects(&pool, character_id)
            .await
            .expect("read the outbox"),
        vec![("character.created".to_owned(), 2)]
    );
}

#[sqlx::test(migrations = "./migrations")]
async fn save_character_changed_save_replaces_details_and_sends_updated(pool: PgPool) {
    let first_anime_id = insert_anime_with_source_page(&pool, Provider::Mal, "21")
        .await
        .expect("store the work with its source page");
    let second_anime_id = insert_anime_with_source_page(&pool, Provider::Mal, "459")
        .await
        .expect("store the work with its source page");
    let character_id = insert_unmerged_character(&pool, "Luffy").await;
    let mut first_input = build_character_save(character_id, 1, "first-hash");
    first_input.traits = vec!["Rubber".to_owned()];
    first_input.nicknames = vec!["Straw Hat".to_owned()];
    first_input.animeography = vec![build_appearance("mal", "21", "MAIN")];
    save_character(&pool, &first_input)
        .await
        .expect("first save");

    let mut second_input = build_character_save(character_id, 2, "second-hash");
    second_input.traits = vec!["Captain".to_owned()];
    second_input.animeography = vec![build_appearance("mal", "459", "SUPPORTING")];
    let second = save_character(&pool, &second_input)
        .await
        .expect("second save");

    assert_eq!(
        second,
        SaveCharacterOutcome {
            has_changed: true,
            version: 3,
        }
    );
    let stored_lists: (Vec<String>, Vec<String>) = sqlx::query_as(
        "SELECT
             ARRAY(SELECT trait FROM character_trait WHERE character_id = $1),
             ARRAY(SELECT nickname FROM character_nickname WHERE character_id = $1)",
    )
    .bind(character_id)
    .fetch_one(&pool)
    .await
    .expect("read the lists");
    assert_eq!(stored_lists, (vec!["Captain".to_owned()], Vec::new()));
    assert_eq!(
        list_character_anime(&pool, character_id)
            .await
            .expect("list the works"),
        vec![CharacterAnime {
            anime_id: second_anime_id,
            role: "SUPPORTING".to_owned(),
        }]
    );
    assert_eq!(
        read_character_edges(&pool, character_id)
            .await
            .expect("read the edges"),
        vec![
            (character_id, second_anime_id, "APPEARS_IN".to_owned()),
            (second_anime_id, character_id, "FEATURES".to_owned()),
        ]
    );
    assert_ne!(first_anime_id, second_anime_id);
    assert_eq!(
        read_outbox_subjects(&pool, character_id)
            .await
            .expect("read the outbox"),
        vec![
            ("character.created".to_owned(), 2),
            ("character.updated".to_owned(), 3),
        ]
    );
}

#[sqlx::test(migrations = "./migrations")]
async fn save_character_stale_version_missing_or_deleted_character_is_refused(pool: PgPool) {
    let character_id = insert_unmerged_character(&pool, "Luffy").await;
    let deleted_character_id = insert_unmerged_character(&pool, "Deleted").await;
    sqlx::query("UPDATE character SET deleted_at = now() WHERE id = $1")
        .bind(deleted_character_id)
        .execute(&pool)
        .await
        .expect("delete the character");

    let stale_error = save_character(&pool, &build_character_save(character_id, 5, "hash"))
        .await
        .expect_err("a stale version is refused");
    let missing_error = save_character(&pool, &build_character_save(Uuid::now_v7(), 1, "hash"))
        .await
        .expect_err("an unknown character is refused");
    let deleted_error = save_character(
        &pool,
        &build_character_save(deleted_character_id, 1, "hash"),
    )
    .await
    .expect_err("a deleted character is refused");

    assert!(stale_error.is_version_conflict());
    assert!(missing_error.is_not_found());
    assert!(deleted_error.is_not_found());
    assert_eq!(
        read_character_state(&pool, character_id)
            .await
            .expect("read the state"),
        (1, false, false)
    );
}

#[sqlx::test(migrations = "./migrations")]
async fn save_character_invalid_fields_return_invalid_argument(pool: PgPool) {
    let character_id = insert_unmerged_character(&pool, "Luffy").await;
    let valid_input = build_character_save(character_id, 1, "hash");
    let invalid_inputs: Vec<(&str, SaveCharacterInput)> = vec![
        (
            "blank name",
            SaveCharacterInput {
                name: " ".to_owned(),
                ..valid_input.clone()
            },
        ),
        (
            "negative favorites",
            SaveCharacterInput {
                favorites: Some(-1),
                ..valid_input.clone()
            },
        ),
        (
            "empty hash",
            SaveCharacterInput {
                merged_record_hash: String::new(),
                ..valid_input.clone()
            },
        ),
        (
            "negative merge version",
            SaveCharacterInput {
                merge_version: -1,
                ..valid_input.clone()
            },
        ),
        (
            "blank nickname",
            SaveCharacterInput {
                nicknames: vec![String::new()],
                ..valid_input.clone()
            },
        ),
        (
            "blank trait",
            SaveCharacterInput {
                traits: vec![" ".to_owned()],
                ..valid_input.clone()
            },
        ),
        (
            "blank attribute value",
            SaveCharacterInput {
                attributes: BTreeMap::from([("Age".to_owned(), " ".to_owned())]),
                ..valid_input.clone()
            },
        ),
        (
            "blank spoiler name",
            SaveCharacterInput {
                spoilers: BTreeMap::from([(String::new(), "hidden".to_owned())]),
                ..valid_input.clone()
            },
        ),
        (
            "blank appearance address",
            SaveCharacterInput {
                animeography: vec![build_appearance("mal", " ", "MAIN")],
                ..valid_input.clone()
            },
        ),
        (
            "blank appearance role",
            SaveCharacterInput {
                animeography: vec![build_appearance("mal", "21", "")],
                ..valid_input.clone()
            },
        ),
        (
            "blank mangaography role",
            SaveCharacterInput {
                mangaography: vec![CharacterMangaographyInput {
                    xrefs: Vec::new(),
                    role: String::new(),
                }],
                ..valid_input.clone()
            },
        ),
        (
            "blank mangaography external id",
            SaveCharacterInput {
                mangaography: vec![CharacterMangaographyInput {
                    xrefs: vec![SourceMaterialXrefInput {
                        provider: "mal".to_owned(),
                        external_id: String::new(),
                        url: String::new(),
                    }],
                    role: "MAIN".to_owned(),
                }],
                ..valid_input.clone()
            },
        ),
    ];

    for (case_name, invalid_input) in invalid_inputs {
        let save_error = save_character(&pool, &invalid_input)
            .await
            .expect_err(case_name);
        assert!(
            save_error.is_invalid_argument(),
            "{case_name}: {save_error}"
        );
    }
    let nul_byte_error = save_character(
        &pool,
        &SaveCharacterInput {
            description: Some("before\0after".to_owned()),
            ..valid_input.clone()
        },
    )
    .await
    .expect_err("text PostgreSQL cannot store is refused");
    assert!(nul_byte_error.is_invalid_argument(), "{nul_byte_error}");
    let unknown_role_error = save_character(
        &pool,
        &SaveCharacterInput {
            animeography: vec![build_appearance("mal", "21", "HERO")],
            ..valid_input
        },
    )
    .await
    .expect_err("an unknown role is refused by its lookup table");
    assert!(unknown_role_error.is_foreign_key_violation());
    assert_eq!(
        read_character_state(&pool, character_id)
            .await
            .expect("read the state"),
        (1, false, false)
    );
}

#[sqlx::test(migrations = "./migrations")]
async fn save_character_two_pages_of_one_anime_show_strongest_role_and_one_edge_pair(pool: PgPool) {
    let anime_id = insert_anime_with_source_page(&pool, Provider::Mal, "21")
        .await
        .expect("store the work with its source page");
    let anilist_page =
        PageKey::new(Provider::AniList, EntityType::Anime, "21").expect("valid page key");
    link_anime_source_page(&pool, anime_id, &anilist_page)
        .await
        .expect("link the work's AniList page");
    let character_id = insert_unmerged_character(&pool, "Luffy").await;
    let mut input = build_character_save(character_id, 1, "hash");
    input.animeography = vec![
        build_appearance("anilist", "21", "SUPPORTING"),
        build_appearance("mal", "21", "BACKGROUND"),
        build_appearance("mal", "21", "MAIN"),
        build_appearance("mal", "21", "UNKNOWN"),
    ];

    save_character(&pool, &input)
        .await
        .expect("save the character");

    let stored_appearances: Vec<(String, String)> = sqlx::query_as(
        "SELECT anime_page_provider, role FROM character_animeography
         WHERE character_id = $1 ORDER BY anime_page_provider",
    )
    .bind(character_id)
    .fetch_all(&pool)
    .await
    .expect("read the appearances");
    assert_eq!(
        stored_appearances,
        vec![
            ("anilist".to_owned(), "SUPPORTING".to_owned()),
            ("mal".to_owned(), "MAIN".to_owned()),
        ]
    );
    assert_eq!(
        list_character_anime(&pool, character_id)
            .await
            .expect("list the works"),
        vec![CharacterAnime {
            anime_id,
            role: "MAIN".to_owned(),
        }]
    );
    assert_eq!(
        read_character_edges(&pool, character_id)
            .await
            .expect("read the edges")
            .len(),
        2
    );
}

#[sqlx::test(migrations = "./migrations")]
async fn list_character_anime_orders_by_role_then_id_and_skips_deleted_anime(pool: PgPool) {
    let main_anime_id = insert_anime_with_source_page(&pool, Provider::Mal, "21")
        .await
        .expect("store the work with its source page");
    let background_anime_id = insert_anime_with_source_page(&pool, Provider::Mal, "22")
        .await
        .expect("store the work with its source page");
    let deleted_anime_id = insert_anime_with_source_page(&pool, Provider::Mal, "23")
        .await
        .expect("store the work with its source page");
    sqlx::query("UPDATE anime SET deleted_at = now() WHERE id = $1")
        .bind(deleted_anime_id)
        .execute(&pool)
        .await
        .expect("delete the work");
    let character_id = insert_unmerged_character(&pool, "Luffy").await;
    let mut input = build_character_save(character_id, 1, "hash");
    input.animeography = vec![
        build_appearance("mal", "22", "BACKGROUND"),
        build_appearance("mal", "23", "MAIN"),
        build_appearance("mal", "21", "MAIN"),
        build_appearance("mal", "999", "MAIN"),
    ];

    save_character(&pool, &input)
        .await
        .expect("save the character");

    assert_eq!(
        list_character_anime(&pool, character_id)
            .await
            .expect("list the works"),
        vec![
            CharacterAnime {
                anime_id: main_anime_id,
                role: "MAIN".to_owned(),
            },
            CharacterAnime {
                anime_id: background_anime_id,
                role: "BACKGROUND".to_owned(),
            },
        ]
    );
    assert_eq!(
        read_character_edges(&pool, character_id)
            .await
            .expect("read the edges")
            .len(),
        4
    );
}

#[sqlx::test(migrations = "./migrations")]
async fn link_anime_source_page_named_by_characters_updates_them_and_their_edges(pool: PgPool) {
    let merged_character_id = insert_unmerged_character(&pool, "Luffy").await;
    let mut merged_input = build_character_save(merged_character_id, 1, "hash");
    merged_input.animeography = vec![build_appearance("mal", "52991", "MAIN")];
    save_character(&pool, &merged_input)
        .await
        .expect("save the merged character");
    let unmerged_character_id = insert_unmerged_character(&pool, "Zoro").await;
    sqlx::query(
        "INSERT INTO character_animeography
             (character_id, anime_page_provider, anime_page_address, role)
         VALUES ($1, 'mal', '52991', 'SUPPORTING')",
    )
    .bind(unmerged_character_id)
    .execute(&pool)
    .await
    .expect("store the unmerged character's appearance");
    assert_eq!(
        list_character_anime(&pool, merged_character_id)
            .await
            .expect("list before the link"),
        Vec::new()
    );

    let anime_id = insert_anime_with_source_page(&pool, Provider::Mal, "52991")
        .await
        .expect("store the work with its source page");

    assert_eq!(
        list_character_anime(&pool, merged_character_id)
            .await
            .expect("list after the link"),
        vec![CharacterAnime {
            anime_id,
            role: "MAIN".to_owned(),
        }]
    );
    assert_eq!(
        read_character_edges(&pool, merged_character_id)
            .await
            .expect("read the merged character's edges"),
        vec![
            (merged_character_id, anime_id, "APPEARS_IN".to_owned()),
            (anime_id, merged_character_id, "FEATURES".to_owned()),
        ]
    );
    assert_eq!(
        read_character_edges(&pool, unmerged_character_id)
            .await
            .expect("read the unmerged character's edges")
            .len(),
        2
    );
    assert_eq!(
        read_character_state(&pool, merged_character_id)
            .await
            .expect("read the merged state"),
        (3, true, false)
    );
    assert_eq!(
        read_character_state(&pool, unmerged_character_id)
            .await
            .expect("read the unmerged state"),
        (2, false, false)
    );
    assert_eq!(
        read_outbox_subjects(&pool, merged_character_id)
            .await
            .expect("read the merged outbox"),
        vec![
            ("character.created".to_owned(), 2),
            ("character.updated".to_owned(), 3),
        ]
    );
    assert_eq!(
        read_outbox_subjects(&pool, unmerged_character_id)
            .await
            .expect("read the unmerged outbox"),
        Vec::new()
    );
}

#[sqlx::test(migrations = "./migrations")]
async fn save_character_mangaography_keeps_matched_source_material_with_strongest_role(
    pool: PgPool,
) {
    let source_material_id = Uuid::now_v7();
    sqlx::query("INSERT INTO source_material (id, title, type) VALUES ($1, 'One Piece', 'MANGA')")
        .bind(source_material_id)
        .execute(&pool)
        .await
        .expect("store the source material");
    sqlx::query(
        "INSERT INTO source_material_xref (provider, external_id, source_material_id, source)
         VALUES ('mal', '13', $1, 'https://myanimelist.net/manga/13'),
                ('anilist', '30013', $1, 'https://anilist.co/manga/30013')",
    )
    .bind(source_material_id)
    .execute(&pool)
    .await
    .expect("store the cross-references");
    let character_id = insert_unmerged_character(&pool, "Luffy").await;
    let build_xref = |provider: &str, external_id: &str| SourceMaterialXrefInput {
        provider: provider.to_owned(),
        external_id: external_id.to_owned(),
        url: String::new(),
    };
    let mut input = build_character_save(character_id, 1, "hash");
    input.mangaography = vec![
        CharacterMangaographyInput {
            xrefs: vec![build_xref("anilist", "30013")],
            role: "SUPPORTING".to_owned(),
        },
        CharacterMangaographyInput {
            xrefs: vec![build_xref("mal", "13")],
            role: "MAIN".to_owned(),
        },
        CharacterMangaographyInput {
            xrefs: vec![build_xref("mal", "999999")],
            role: "MAIN".to_owned(),
        },
    ];

    save_character(&pool, &input)
        .await
        .expect("save the character");

    let stored_mangaography: Vec<(Uuid, String)> = sqlx::query_as(
        "SELECT source_material_id, role FROM character_mangaography WHERE character_id = $1",
    )
    .bind(character_id)
    .fetch_all(&pool)
    .await
    .expect("read the mangaography");
    assert_eq!(
        stored_mangaography,
        vec![(source_material_id, "MAIN".to_owned())]
    );
}

#[sqlx::test(migrations = "./migrations")]
async fn save_character_and_link_page_at_once_both_succeed(pool: PgPool) {
    for attempt in 0..20 {
        let page_address = format!("70{attempt}");
        let character_id = insert_unmerged_character(&pool, "Luffy").await;
        let anime_id = insert_unmerged_anime(&pool).await;
        let page_key =
            PageKey::new(Provider::Mal, EntityType::Anime, &page_address).expect("valid page key");
        let mut input = build_character_save(character_id, 1, "hash");
        input.animeography = vec![build_appearance("mal", &page_address, "MAIN")];

        let (save_outcome, link_outcome) = tokio::join!(
            save_character(&pool, &input),
            link_anime_source_page(&pool, anime_id, &page_key),
        );

        save_outcome.expect("the character save succeeds");
        link_outcome.expect("the page link succeeds");
        assert_eq!(
            list_character_anime(&pool, character_id)
                .await
                .expect("list the works"),
            vec![CharacterAnime {
                anime_id,
                role: "MAIN".to_owned(),
            }],
            "attempt {attempt}"
        );
        assert_eq!(
            read_character_edges(&pool, character_id)
                .await
                .expect("read the edges")
                .len(),
            2,
            "attempt {attempt}"
        );
    }
}

#[sqlx::test(migrations = "./migrations")]
async fn save_character_two_characters_adding_crossed_new_traits_at_once_both_succeed(
    pool: PgPool,
) {
    for attempt in 0..20 {
        let first_character_id = insert_unmerged_character(&pool, "Luffy").await;
        let second_character_id = insert_unmerged_character(&pool, "Zoro").await;
        let shared_traits: Vec<String> = (0..30)
            .map(|trait_number| format!("trait-{attempt}-{trait_number}"))
            .collect();
        let mut first_input = build_character_save(first_character_id, 1, "hash");
        first_input.traits = shared_traits.clone();
        let mut second_input = build_character_save(second_character_id, 1, "hash");
        second_input.traits = shared_traits.into_iter().rev().collect();

        let (first_outcome, second_outcome) = tokio::join!(
            save_character(&pool, &first_input),
            save_character(&pool, &second_input),
        );

        first_outcome.expect("the first save succeeds");
        second_outcome.expect("the second save succeeds");
    }
}

#[sqlx::test(migrations = "./migrations")]
async fn save_character_named_page_shared_by_other_save_does_not_wait(pool: PgPool) {
    let character_id = insert_unmerged_character(&pool, "Luffy").await;
    let mut input = build_character_save(character_id, 1, "hash");
    input.animeography = vec![build_appearance("mal", "21", "MAIN")];
    let mut other_save = pool.begin().await.expect("begin the other save");
    sqlx::query("SELECT pg_advisory_xact_lock_shared(hashtextextended('mal:anime:21', 0))")
        .execute(&mut *other_save)
        .await
        .expect("hold the page as another save does");

    let saved = tokio::time::timeout(
        std::time::Duration::from_secs(2),
        save_character(&pool, &input),
    )
    .await;

    other_save.rollback().await.expect("end the other save");
    assert_eq!(
        saved
            .expect("the save does not wait for another save of the page")
            .map(|outcome| outcome.has_changed)
            .ok(),
        Some(true)
    );
}

#[sqlx::test(migrations = "./migrations")]
async fn save_character_named_page_being_linked_waits_for_link(pool: PgPool) {
    let character_id = insert_unmerged_character(&pool, "Luffy").await;
    let mut input = build_character_save(character_id, 1, "hash");
    input.animeography = vec![build_appearance("mal", "21", "MAIN")];
    let mut page_link = pool.begin().await.expect("begin the page link");
    sqlx::query("SELECT pg_advisory_xact_lock(hashtextextended('mal:anime:21', 0))")
        .execute(&mut *page_link)
        .await
        .expect("hold the page as a link does");

    let save_task = tokio::spawn({
        let pool = pool.clone();
        async move { save_character(&pool, &input).await }
    });
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    let was_waiting = !save_task.is_finished();
    page_link.rollback().await.expect("end the page link");
    let saved = save_task
        .await
        .expect("join the save")
        .expect("the save finishes after the link");

    assert!(
        was_waiting,
        "the save must wait while the page is being linked"
    );
    assert!(saved.has_changed);
}
