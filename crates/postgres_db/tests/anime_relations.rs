//! Integration tests for work relations, source material and companies against PostgreSQL.

mod support;

use postgres_db::{
    AnimeCompanyInput, AnimeRelationInput, AnimeRelationsAndGraphInput, AnimeSourceMaterialInput,
    CompanyXrefInput, PageKey, PageKind, Provider, SourceMaterialXrefInput, link_anime_source_page,
    save_anime,
};
use sqlx::PgPool;
use uuid::Uuid;

use crate::support::{build_work_save, insert_unmerged_work};

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

async fn save_work_with_relations(
    pool: &PgPool,
    anime_id: Uuid,
    expected_version: i64,
    merged_record_hash: &str,
    relations_and_graph: AnimeRelationsAndGraphInput,
) -> Result<(), postgres_db::Error> {
    let mut work_save = build_work_save(anime_id, expected_version, merged_record_hash, Vec::new());
    work_save.relations_and_graph = relations_and_graph;
    save_anime(pool, &work_save).await.map(|_| ())
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
    let owning_anime_id = insert_unmerged_work(&pool).await;
    let other_anime_id = insert_unmerged_work(&pool).await;
    let work_page = PageKey::new(Provider::Mal, PageKind::Anime, "21").expect("valid page key");
    let episode_page =
        PageKey::new(Provider::Mal, PageKind::Episode, "21").expect("valid page key");
    link_anime_source_page(&pool, owning_anime_id, &work_page)
        .await
        .expect("first link succeeds");

    let episode_error = link_anime_source_page(&pool, owning_anime_id, &episode_page)
        .await
        .expect_err("an episode page cannot be a work's source");
    let missing_error = link_anime_source_page(&pool, Uuid::now_v7(), &work_page)
        .await
        .expect_err("a missing work cannot gain a page");
    let taken_error = link_anime_source_page(&pool, other_anime_id, &work_page)
        .await
        .expect_err("a page has at most one source work");

    assert!(episode_error.is_invalid_argument());
    assert!(missing_error.is_not_found());
    assert!(taken_error.is_invalid_argument());
}

#[sqlx::test(migrations = "./migrations")]
async fn link_anime_source_page_repeated_link_resolves_nothing(pool: PgPool) {
    let anime_id = insert_unmerged_work(&pool).await;
    let work_page = PageKey::new(Provider::Mal, PageKind::Anime, "21").expect("valid page key");
    link_anime_source_page(&pool, anime_id, &work_page)
        .await
        .expect("first link succeeds");

    let resolved_count = link_anime_source_page(&pool, anime_id, &work_page)
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
async fn save_anime_relation_to_stored_work_resolves_then_clears_graph_edges(pool: PgPool) {
    let sequel_anime_id = insert_unmerged_work(&pool).await;
    let prequel_anime_id = insert_unmerged_work(&pool).await;
    link_anime_source_page(
        &pool,
        sequel_anime_id,
        &PageKey::new(Provider::Mal, PageKind::Anime, "2").expect("valid page key"),
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

    save_work_with_relations(&pool, prequel_anime_id, 1, "work-v1", sequel_relation)
        .await
        .expect("save the relation");
    let edges_after_relation = count_anime_edges(&pool, prequel_anime_id, sequel_anime_id).await;
    save_work_with_relations(
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
    let first_anime_id = insert_unmerged_work(&pool).await;
    let second_anime_id = insert_unmerged_work(&pool).await;
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

    save_work_with_relations(&pool, first_anime_id, 1, "work-v1", first_relations)
        .await
        .expect("first work saves");
    save_work_with_relations(&pool, second_anime_id, 1, "work-v1", second_relations)
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
    let first_work_links: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM anime_source_material_relation WHERE anime_id = $1",
    )
    .bind(first_anime_id)
    .fetch_one(&pool)
    .await
    .expect("read source material links");
    assert_eq!(first_work_links, 2);
    assert_eq!(
        company_names,
        vec!["Studio A Renamed".to_owned(), "Studio B".to_owned()]
    );
}

#[sqlx::test(migrations = "./migrations")]
async fn save_anime_xrefs_naming_two_stored_rows_returns_invalid_argument(pool: PgPool) {
    let first_anime_id = insert_unmerged_work(&pool).await;
    let merging_anime_id = insert_unmerged_work(&pool).await;
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
    save_work_with_relations(&pool, first_anime_id, 1, "work-v1", separate_relations)
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

    let source_material_error = save_work_with_relations(
        &pool,
        merging_anime_id,
        1,
        "work-v1",
        merged_source_material,
    )
    .await
    .expect_err("xrefs of two stored source material rows must be refused");
    let company_error =
        save_work_with_relations(&pool, merging_anime_id, 1, "work-v1", merged_company)
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
        "INSERT INTO provider_page (provider, page_kind, address) VALUES ($1, 'anime', $2)
         ON CONFLICT DO NOTHING",
    )
    .bind(page_key.provider().as_str())
    .bind(page_key.address().as_str())
    .execute(pool)
    .await?;
    sqlx::query(
        "INSERT INTO anime_provider_page (anime_id, provider, page_kind, address, role)
         VALUES ($1, $2, 'anime', $3, 'partial_source')",
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
        "SELECT anime_id, role FROM anime_provider_page
         WHERE provider = $1 AND address = $2 ORDER BY role, anime_id",
    )
    .bind(page_key.provider().as_str())
    .bind(page_key.address().as_str())
    .fetch_all(pool)
    .await
}

#[sqlx::test(migrations = "./migrations")]
async fn link_anime_source_page_partial_source_links_respect_roles(pool: PgPool) {
    let linking_work_id = insert_unmerged_work(&pool).await;
    let larger_work_id = insert_unmerged_work(&pool).await;
    let shared_page = PageKey::new(Provider::Mal, PageKind::Anime, "21").expect("valid page key");
    link_partial_source_page(&pool, linking_work_id, &shared_page)
        .await
        .expect("link the linking work's partial page");
    link_partial_source_page(&pool, larger_work_id, &shared_page)
        .await
        .expect("link the larger work's partial page");

    link_anime_source_page(&pool, linking_work_id, &shared_page)
        .await
        .expect("the work's own partial page becomes its source");
    let third_work_id = insert_unmerged_work(&pool).await;
    let taken_error = link_anime_source_page(&pool, third_work_id, &shared_page)
        .await
        .expect_err("the page already has a source work");
    let linking_work_version: i64 = sqlx::query_scalar("SELECT version FROM anime WHERE id = $1")
        .bind(linking_work_id)
        .fetch_one(&pool)
        .await
        .expect("read the version");

    assert_eq!(
        read_page_roles(&pool, &shared_page)
            .await
            .expect("read the page's links"),
        vec![
            (larger_work_id, "partial_source".to_owned()),
            (linking_work_id, "source".to_owned()),
        ]
    );
    assert!(taken_error.is_invalid_argument());
    assert_eq!(linking_work_version, 2);
}
