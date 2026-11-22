//! Integration tests for work relations, source material and companies against PostgreSQL.

mod support;

use postgres_db::{
    AnimeCompanyInput, AnimeRelationInput, AnimeRelationsAndGraphInput, AnimeSourceMaterialInput,
    CompanyXrefInput, EntityType, PageKey, Provider, SourceMaterialXrefInput,
    link_anime_source_page, save_anime,
};
use sqlx::PgPool;
use uuid::Uuid;

use crate::support::{build_anime_save, insert_unmerged_anime};

fn build_source_material(title: &str, xrefs: &[(&str, &str)]) -> AnimeSourceMaterialInput {
    AnimeSourceMaterialInput {
        relation_type: "ADAPTATION".to_owned(),
        title: title.to_owned(),
        title_english: None,
        title_japanese: None,
        source_material_type: "MANGA".to_owned(),
        status: None,
        score: None,
        chapters: None,
        volumes: None,
        images: Vec::new(),
        xrefs: xrefs
            .iter()
            .map(|(provider, external_id)| SourceMaterialXrefInput {
                provider: (*provider).to_owned(),
                external_id: (*external_id).to_owned(),
                url: format!("https://example.com/{provider}/{external_id}"),
            })
            .collect(),
    }
}

fn build_company(name: &str, xref_urls: &[(&str, &str)]) -> AnimeCompanyInput {
    AnimeCompanyInput {
        name: name.to_owned(),
        description: None,
        roles: vec!["STUDIO".to_owned()],
        xrefs: xref_urls
            .iter()
            .map(|(provider, url)| CompanyXrefInput {
                provider: (*provider).to_owned(),
                url: (*url).to_owned(),
            })
            .collect(),
    }
}

async fn save_anime_with_relations(
    pool: &PgPool,
    anime_id: Uuid,
    expected_version: i64,
    merged_record_hash: &str,
    relations_and_graph: AnimeRelationsAndGraphInput,
) -> Result<(), postgres_db::Error> {
    let mut anime_save =
        build_anime_save(anime_id, expected_version, merged_record_hash, Vec::new());
    anime_save.relations_and_graph = relations_and_graph;
    save_anime(pool, &anime_save).await.map(|_| ())
}

async fn count_anime_edges(pool: &PgPool, first_id: Uuid, second_id: Uuid) -> i64 {
    sqlx::query_scalar(
        "SELECT COUNT(*) FROM graph_edge
         WHERE (from_id = $1 AND to_id = $2) OR (from_id = $2 AND to_id = $1)",
    )
    .bind(first_id)
    .bind(second_id)
    .fetch_one(pool)
    .await
    .unwrap_or(-1)
}

#[sqlx::test(migrations = "./migrations")]
async fn link_anime_source_page_invalid_link_returns_error(pool: PgPool) {
    let owning_anime_id = insert_unmerged_anime(&pool).await;
    let other_anime_id = insert_unmerged_anime(&pool).await;
    let anime_page = PageKey::new(Provider::Mal, EntityType::Anime, "21").expect("valid page key");
    let episode_page =
        PageKey::new(Provider::Mal, EntityType::Episode, "21").expect("valid page key");
    link_anime_source_page(&pool, owning_anime_id, &anime_page)
        .await
        .expect("first link succeeds");

    let episode_error = link_anime_source_page(&pool, owning_anime_id, &episode_page)
        .await
        .expect_err("an episode page cannot be a work's source");
    let missing_error = link_anime_source_page(&pool, Uuid::now_v7(), &anime_page)
        .await
        .expect_err("a missing work cannot gain a page");
    let taken_error = link_anime_source_page(&pool, other_anime_id, &anime_page)
        .await
        .expect_err("a page has at most one source work");

    assert!(episode_error.is_invalid_argument());
    assert!(missing_error.is_not_found());
    assert!(taken_error.is_invalid_argument());
}

#[sqlx::test(migrations = "./migrations")]
async fn link_anime_source_page_repeated_link_resolves_nothing(pool: PgPool) {
    let anime_id = insert_unmerged_anime(&pool).await;
    let anime_page = PageKey::new(Provider::Mal, EntityType::Anime, "21").expect("valid page key");
    link_anime_source_page(&pool, anime_id, &anime_page)
        .await
        .expect("first link succeeds");

    let resolved_count = link_anime_source_page(&pool, anime_id, &anime_page)
        .await
        .expect("repeated link succeeds");
    let stored_version: i64 = sqlx::query_scalar("SELECT version FROM anime WHERE id = $1")
        .bind(anime_id)
        .fetch_one(&pool)
        .await
        .expect("read the version");

    assert_eq!(resolved_count, 0);
    assert_eq!(stored_version, 2);
}

#[sqlx::test(migrations = "./migrations")]
async fn save_anime_relation_to_stored_anime_resolves_then_clears_graph_edges(pool: PgPool) {
    let sequel_anime_id = insert_unmerged_anime(&pool).await;
    let prequel_anime_id = insert_unmerged_anime(&pool).await;
    link_anime_source_page(
        &pool,
        sequel_anime_id,
        &PageKey::new(Provider::Mal, EntityType::Anime, "2").expect("valid page key"),
    )
    .await
    .expect("link the sequel's page");
    let sequel_relation = AnimeRelationsAndGraphInput {
        relations: vec![AnimeRelationInput {
            relation_type: "SEQUEL".to_owned(),
            to_page_provider: "mal".to_owned(),
            to_page_address: "2".to_owned(),
        }],
        ..AnimeRelationsAndGraphInput::default()
    };

    save_anime_with_relations(&pool, prequel_anime_id, 1, "work-v1", sequel_relation)
        .await
        .expect("save the relation");
    let edges_after_relation = count_anime_edges(&pool, prequel_anime_id, sequel_anime_id).await;
    save_anime_with_relations(
        &pool,
        prequel_anime_id,
        2,
        "work-v2",
        AnimeRelationsAndGraphInput::default(),
    )
    .await
    .expect("save without the relation");
    let edges_after_removal = count_anime_edges(&pool, prequel_anime_id, sequel_anime_id).await;

    assert_eq!(edges_after_relation, 2);
    assert_eq!(edges_after_removal, 0);
}

#[sqlx::test(migrations = "./migrations")]
async fn save_anime_shared_xrefs_reuse_stored_source_material_and_company(pool: PgPool) {
    let first_anime_id = insert_unmerged_anime(&pool).await;
    let second_anime_id = insert_unmerged_anime(&pool).await;
    let first_relations = AnimeRelationsAndGraphInput {
        source_materials: vec![
            build_source_material("Second Manga", &[("mal", "20")]),
            build_source_material("First Manga", &[("mal", "13"), ("anilist", "7")]),
            build_source_material("First Manga", &[("mal", "13")]),
        ],
        companies: vec![
            build_company(
                "Studio B",
                &[("mal", "https://myanimelist.net/anime/producer/2")],
            ),
            build_company(
                "Studio A",
                &[
                    ("mal", "https://myanimelist.net/anime/producer/1"),
                    ("anilist", "https://anilist.co/studio/1"),
                ],
            ),
            build_company(
                "Studio A",
                &[("mal", "https://myanimelist.net/anime/producer/1")],
            ),
        ],
        ..AnimeRelationsAndGraphInput::default()
    };
    let second_relations = AnimeRelationsAndGraphInput {
        source_materials: vec![build_source_material(
            "First Manga Renamed",
            &[("mal", "13")],
        )],
        companies: vec![build_company(
            "Studio A Renamed",
            &[("mal", "https://myanimelist.net/anime/producer/1")],
        )],
        ..AnimeRelationsAndGraphInput::default()
    };

    save_anime_with_relations(&pool, first_anime_id, 1, "work-v1", first_relations)
        .await
        .expect("first work saves");
    save_anime_with_relations(&pool, second_anime_id, 1, "work-v1", second_relations)
        .await
        .expect("second work saves");

    let source_material_titles: Vec<String> =
        sqlx::query_scalar("SELECT title FROM source_material ORDER BY title")
            .fetch_all(&pool)
            .await
            .expect("read source material");
    let company_names: Vec<String> = sqlx::query_scalar("SELECT name FROM company ORDER BY name")
        .fetch_all(&pool)
        .await
        .expect("read companies");
    assert_eq!(
        source_material_titles,
        vec!["First Manga Renamed".to_owned(), "Second Manga".to_owned()]
    );
    let first_anime_links: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM anime_related_source_material WHERE anime_id = $1",
    )
    .bind(first_anime_id)
    .fetch_one(&pool)
    .await
    .expect("read source material links");
    assert_eq!(first_anime_links, 2);
    assert_eq!(
        company_names,
        vec!["Studio A Renamed".to_owned(), "Studio B".to_owned()]
    );
}

#[sqlx::test(migrations = "./migrations")]
async fn save_anime_xrefs_naming_two_stored_rows_returns_invalid_argument(pool: PgPool) {
    let first_anime_id = insert_unmerged_anime(&pool).await;
    let merging_anime_id = insert_unmerged_anime(&pool).await;
    let separate_relations = AnimeRelationsAndGraphInput {
        source_materials: vec![
            build_source_material("Manga One", &[("mal", "13")]),
            build_source_material("Manga Two", &[("anilist", "7")]),
        ],
        companies: vec![
            build_company(
                "Studio One",
                &[("mal", "https://myanimelist.net/anime/producer/1")],
            ),
            build_company("Studio Two", &[("anilist", "https://anilist.co/studio/1")]),
        ],
        ..AnimeRelationsAndGraphInput::default()
    };
    save_anime_with_relations(&pool, first_anime_id, 1, "work-v1", separate_relations)
        .await
        .expect("separate rows save");
    let merged_source_material = AnimeRelationsAndGraphInput {
        source_materials: vec![build_source_material(
            "Merged Manga",
            &[("mal", "13"), ("anilist", "7")],
        )],
        ..AnimeRelationsAndGraphInput::default()
    };
    let merged_company = AnimeRelationsAndGraphInput {
        companies: vec![build_company(
            "Merged Studio",
            &[
                ("mal", "https://myanimelist.net/anime/producer/1"),
                ("anilist", "https://anilist.co/studio/1"),
            ],
        )],
        ..AnimeRelationsAndGraphInput::default()
    };

    let source_material_error = save_anime_with_relations(
        &pool,
        merging_anime_id,
        1,
        "work-v1",
        merged_source_material,
    )
    .await
    .expect_err("xrefs of two stored source material rows must be refused");
    let company_error =
        save_anime_with_relations(&pool, merging_anime_id, 1, "work-v1", merged_company)
            .await
            .expect_err("xrefs of two stored company rows must be refused");

    assert!(source_material_error.is_invalid_argument());
    assert!(company_error.is_invalid_argument());
}

async fn link_partial_source_page(
    pool: &PgPool,
    anime_id: Uuid,
    page_key: &PageKey,
) -> Result<(), sqlx::Error> {
    sqlx::query(
        "INSERT INTO provider_page (provider, entity_type, address) VALUES ($1, 'anime', $2)
         ON CONFLICT DO NOTHING",
    )
    .bind(page_key.provider().as_str())
    .bind(page_key.address().as_str())
    .execute(pool)
    .await?;
    sqlx::query(
        "INSERT INTO anime_partial_source (anime_id, provider, entity_type, address)
         VALUES ($1, $2, 'anime', $3)",
    )
    .bind(anime_id)
    .bind(page_key.provider().as_str())
    .bind(page_key.address().as_str())
    .execute(pool)
    .await?;
    Ok(())
}

async fn read_page_roles(
    pool: &PgPool,
    page_key: &PageKey,
) -> Result<Vec<(Uuid, String)>, sqlx::Error> {
    sqlx::query_as(
        "SELECT anime_id, 'partial_source' AS role FROM anime_partial_source
         WHERE provider = $1 AND address = $2
         UNION ALL
         SELECT anime_id, 'source' AS role FROM anime_source
         WHERE provider = $1 AND address = $2
         ORDER BY role, anime_id",
    )
    .bind(page_key.provider().as_str())
    .bind(page_key.address().as_str())
    .fetch_all(pool)
    .await
}

#[sqlx::test(migrations = "./migrations")]
async fn link_anime_source_page_partial_source_links_respect_roles(pool: PgPool) {
    let linking_anime_id = insert_unmerged_anime(&pool).await;
    let larger_anime_id = insert_unmerged_anime(&pool).await;
    let shared_page = PageKey::new(Provider::Mal, EntityType::Anime, "21").expect("valid page key");
    link_partial_source_page(&pool, linking_anime_id, &shared_page)
        .await
        .expect("link the linking work's partial page");
    link_partial_source_page(&pool, larger_anime_id, &shared_page)
        .await
        .expect("link the larger work's partial page");

    link_anime_source_page(&pool, linking_anime_id, &shared_page)
        .await
        .expect("the work's own partial page becomes its source");
    let third_anime_id = insert_unmerged_anime(&pool).await;
    let taken_error = link_anime_source_page(&pool, third_anime_id, &shared_page)
        .await
        .expect_err("the page already has a source work");
    let second_mal_page =
        PageKey::new(Provider::Mal, EntityType::Anime, "22").expect("valid page key");
    link_partial_source_page(&pool, linking_anime_id, &second_mal_page)
        .await
        .expect("link a second MAL page as a partial source");
    let second_source_error = link_anime_source_page(&pool, linking_anime_id, &second_mal_page)
        .await
        .expect_err("a work holds one source page per provider");
    let linking_anime_version: i64 = sqlx::query_scalar("SELECT version FROM anime WHERE id = $1")
        .bind(linking_anime_id)
        .fetch_one(&pool)
        .await
        .expect("read the version");

    assert_eq!(
        read_page_roles(&pool, &shared_page)
            .await
            .expect("read the page's links"),
        vec![
            (larger_anime_id, "partial_source".to_owned()),
            (linking_anime_id, "source".to_owned()),
        ]
    );
    assert!(taken_error.is_invalid_argument());
    assert!(second_source_error.is_unique_violation());
    assert_eq!(
        read_page_roles(&pool, &second_mal_page)
            .await
            .expect("read the second page's links"),
        vec![(linking_anime_id, "partial_source".to_owned())]
    );
    assert_eq!(linking_anime_version, 2);
}

fn build_relation_to(page_address: &str) -> AnimeRelationInput {
    AnimeRelationInput {
        relation_type: "SEQUEL".to_owned(),
        to_page_provider: "mal".to_owned(),
        to_page_address: page_address.to_owned(),
    }
}

async fn read_anime_state(pool: &PgPool, anime_id: Uuid) -> Result<(i64, bool), sqlx::Error> {
    sqlx::query_as("SELECT version, merge_requested_at IS NOT NULL FROM anime WHERE id = $1")
        .bind(anime_id)
        .fetch_one(pool)
        .await
}

async fn count_update_events(pool: &PgPool, anime_id: Uuid) -> Result<i64, sqlx::Error> {
    sqlx::query_scalar(
        "SELECT COUNT(*) FROM outbox WHERE entity_id = $1 AND subject = 'anime.updated'",
    )
    .bind(anime_id)
    .fetch_one(pool)
    .await
}

#[sqlx::test(migrations = "./migrations")]
async fn link_anime_source_page_relating_anime_get_version_merge_request_and_event(pool: PgPool) {
    let merged_relating_id = insert_unmerged_anime(&pool).await;
    let unmerged_relating_id = insert_unmerged_anime(&pool).await;
    let linked_anime_id = insert_unmerged_anime(&pool).await;
    let relations = AnimeRelationsAndGraphInput {
        relations: vec![build_relation_to("700")],
        ..AnimeRelationsAndGraphInput::default()
    };
    save_anime_with_relations(
        &pool,
        merged_relating_id,
        1,
        "relating-v1",
        relations.clone(),
    )
    .await
    .expect("the merged relating work saves");
    sqlx::query(
        "INSERT INTO anime_related_anime (from_anime_id, relation_type, to_page_provider, to_page_address)
         VALUES ($1, 'PREQUEL', 'mal', '700')",
    )
    .bind(unmerged_relating_id)
    .execute(&pool)
    .await
    .expect("the unmerged work relates to the page");
    let merged_state_before = read_anime_state(&pool, merged_relating_id)
        .await
        .expect("read the work\'s version");
    let unmerged_state_before = read_anime_state(&pool, unmerged_relating_id)
        .await
        .expect("read the work\'s version");
    let merged_events_before = count_update_events(&pool, merged_relating_id)
        .await
        .expect("count the work\'s update events");

    let relating_anime_count = link_anime_source_page(
        &pool,
        linked_anime_id,
        &PageKey::new(Provider::Mal, EntityType::Anime, "700").expect("valid page key"),
    )
    .await
    .expect("the page links");

    assert_eq!(relating_anime_count, 2);
    assert_eq!(
        read_anime_state(&pool, merged_relating_id)
            .await
            .expect("read the work\'s version"),
        (merged_state_before.0 + 1, true)
    );
    assert_eq!(
        read_anime_state(&pool, unmerged_relating_id)
            .await
            .expect("read the work\'s version"),
        (unmerged_state_before.0 + 1, true)
    );
    assert_eq!(
        count_update_events(&pool, merged_relating_id)
            .await
            .expect("count the work\'s update events"),
        merged_events_before + 1
    );
    assert_eq!(
        count_update_events(&pool, unmerged_relating_id)
            .await
            .expect("count the work\'s update events"),
        0
    );
    assert_eq!(
        count_anime_edges(&pool, merged_relating_id, linked_anime_id).await,
        2
    );
    assert_eq!(
        count_anime_edges(&pool, unmerged_relating_id, linked_anime_id).await,
        2
    );
}

#[sqlx::test(migrations = "./migrations")]
async fn save_anime_relation_after_target_page_moved_follows_page_and_clears_old_edges(
    pool: PgPool,
) {
    let relating_anime_id = insert_unmerged_anime(&pool).await;
    let first_owner_id = insert_unmerged_anime(&pool).await;
    let second_owner_id = insert_unmerged_anime(&pool).await;
    let moved_page = PageKey::new(Provider::Mal, EntityType::Anime, "701").expect("valid page key");
    link_anime_source_page(&pool, first_owner_id, &moved_page)
        .await
        .expect("the first owner holds the page");
    let relations = AnimeRelationsAndGraphInput {
        relations: vec![build_relation_to("701")],
        ..AnimeRelationsAndGraphInput::default()
    };
    save_anime_with_relations(
        &pool,
        relating_anime_id,
        1,
        "relating-v1",
        relations.clone(),
    )
    .await
    .expect("the relating work saves");
    assert_eq!(
        count_anime_edges(&pool, relating_anime_id, first_owner_id).await,
        2
    );

    sqlx::query("UPDATE anime_source SET anime_id = $1 WHERE provider = 'mal' AND address = '701'")
        .bind(second_owner_id)
        .execute(&pool)
        .await
        .expect("move the page to the second owner");
    let relating_version = read_anime_state(&pool, relating_anime_id)
        .await
        .expect("read the work\'s version")
        .0;
    save_anime_with_relations(
        &pool,
        relating_anime_id,
        relating_version,
        "relating-v2",
        relations,
    )
    .await
    .expect("the relating work saves again");

    assert_eq!(
        count_anime_edges(&pool, relating_anime_id, second_owner_id).await,
        2
    );
    assert_eq!(
        count_anime_edges(&pool, relating_anime_id, first_owner_id).await,
        0
    );
}

#[sqlx::test(migrations = "./migrations")]
async fn save_anime_relation_to_own_source_page_reaches_no_anime(pool: PgPool) {
    let anime_id = insert_unmerged_anime(&pool).await;
    link_anime_source_page(
        &pool,
        anime_id,
        &PageKey::new(Provider::Mal, EntityType::Anime, "702").expect("valid page key"),
    )
    .await
    .expect("the work holds its own page");
    let version = read_anime_state(&pool, anime_id)
        .await
        .expect("read the work\'s version")
        .0;
    let relations = AnimeRelationsAndGraphInput {
        relations: vec![build_relation_to("702")],
        ..AnimeRelationsAndGraphInput::default()
    };

    save_anime_with_relations(&pool, anime_id, version, "self-v1", relations)
        .await
        .expect("a relation to the work's own page saves");

    assert_eq!(count_anime_edges(&pool, anime_id, anime_id).await, 0);
}

#[sqlx::test(migrations = "./migrations")]
async fn link_anime_source_page_while_relating_anime_saves_both_succeed(pool: PgPool) {
    for attempt in 0..20 {
        let page_address = format!("{}", 800 + attempt);
        let relating_anime_id = insert_unmerged_anime(&pool).await;
        let linked_anime_id = insert_unmerged_anime(&pool).await;
        let relations = AnimeRelationsAndGraphInput {
            relations: vec![build_relation_to(&page_address)],
            ..AnimeRelationsAndGraphInput::default()
        };
        save_anime_with_relations(
            &pool,
            relating_anime_id,
            1,
            "relating-v1",
            relations.clone(),
        )
        .await
        .unwrap_or_else(|error| panic!("attempt {attempt}: first save: {error}"));
        let page_key = PageKey::new(Provider::Mal, EntityType::Anime, page_address.clone())
            .expect("valid page key");

        let (link_outcome, save_outcome) = tokio::join!(
            link_anime_source_page(&pool, linked_anime_id, &page_key),
            save_anime_with_relations(&pool, relating_anime_id, 2, "relating-v2", relations),
        );

        link_outcome.unwrap_or_else(|error| panic!("attempt {attempt}: link: {error}"));
        if let Err(save_error) = save_outcome {
            assert!(
                save_error.is_version_conflict(),
                "attempt {attempt}: save failed with {save_error}"
            );
        }
        assert_eq!(
            count_anime_edges(&pool, relating_anime_id, linked_anime_id).await,
            2
        );
    }
}
