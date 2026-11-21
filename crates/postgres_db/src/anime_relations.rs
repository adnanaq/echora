//! Persistence helpers for work relations, claims, source material, and companies.

use std::collections::{BTreeMap, BTreeSet};

use sqlx::{PgPool, Postgres, Transaction};
use uuid::Uuid;

use crate::error::Error;
use crate::graph_projection::{
    invert_anime_relation_type, recompute_anime_pair_graph_edges, replace_company_graph_edges,
    replace_source_material_graph_edges,
};
use crate::provider_page::{PageKey, PageKind, Provider};

/// Directed anime-to-anime relation stored in `anime_relation`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnimeRelationInput {
    /// Canonical `anime_relation_type` lookup value.
    pub relation_type: String,
    /// Target provider name (for example `"mal"`).
    pub to_page_provider: String,
    /// Target provider page address (for example `"21"`).
    pub to_page_address: String,
}

/// Provider-specific relation provenance stored in `anime_relation_claim`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnimeRelationClaimInput {
    /// Provider making the claim.
    pub provider: Provider,
    /// Canonical `anime_relation_type` lookup value claimed by `provider`.
    pub relation_type: String,
    /// Target provider page key.
    pub target_page: PageKey,
}

/// External cross-reference identifying a `source_material` row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceMaterialXrefInput {
    /// Canonical `provider` lookup value.
    pub provider: String,
    /// Provider-local identifier for the source material.
    pub external_id: String,
    /// Canonical provider URL for the source material.
    pub url: String,
}

/// Related source material entry and its link to the anime.
#[derive(Debug, Clone, PartialEq)]
pub struct AnimeSourceMaterialInput {
    /// Canonical `source_material_relation_type` lookup value.
    pub relation_type: String,
    /// Primary title.
    pub title: String,
    /// English title, if known.
    pub title_english: Option<String>,
    /// Japanese title, if known.
    pub title_japanese: Option<String>,
    /// Canonical `source_material_type` lookup value.
    pub source_material_type: String,
    /// Publication status (`anime_status` lookup value), if known.
    pub status: Option<String>,
    /// Score (`0.0..=10.0`), if known.
    pub score: Option<f64>,
    /// Chapter count (`>= 0`), if known.
    pub chapters: Option<i32>,
    /// Volume count (`>= 0`), if known.
    pub volumes: Option<i32>,
    /// Image URLs stored in `source_material_image`.
    pub images: Vec<String>,
    /// Cross-references used to match or create the `source_material` row.
    pub xrefs: Vec<SourceMaterialXrefInput>,
}

/// External cross-reference identifying a `company` row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompanyXrefInput {
    /// Canonical `provider` lookup value.
    pub provider: String,
    /// Canonical company page URL on `provider`.
    pub url: String,
}

/// Company entry and its roles on an anime.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnimeCompanyInput {
    /// Company display name.
    pub name: String,
    /// Optional company description.
    pub description: Option<String>,
    /// Canonical `company_role` lookup values held on this work.
    pub roles: Vec<String>,
    /// Cross-references used to match or create the `company` row.
    pub xrefs: Vec<CompanyXrefInput>,
}

/// Relations, claims, source material, and companies saved with a work.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AnimeRelationsAndGraphInput {
    /// Directed anime-to-anime relations.
    pub relations: Vec<AnimeRelationInput>,
    /// Per-provider relation claims.
    pub relation_claims: Vec<AnimeRelationClaimInput>,
    /// Linked source materials.
    pub source_materials: Vec<AnimeSourceMaterialInput>,
    /// Linked companies.
    pub companies: Vec<AnimeCompanyInput>,
}

/// Validates relations, provider claims, source materials, and companies.
///
/// # Errors
///
/// Returns an invalid-argument error when relation types, scores, counts, or
/// cross-references are invalid.
pub(crate) fn validate_relations_and_graph(
    input: &AnimeRelationsAndGraphInput,
) -> Result<(), Error> {
    for relation in &input.relations {
        invert_anime_relation_type(&relation.relation_type)?;
        if relation.to_page_provider.trim().is_empty() || relation.to_page_address.trim().is_empty()
        {
            return Err(Error::invalid_argument(
                "anime_relation target provider and address must be non-empty",
            ));
        }
    }

    for claim in &input.relation_claims {
        invert_anime_relation_type(&claim.relation_type)?;
        if claim.target_page.page_kind() != PageKind::Anime {
            return Err(Error::invalid_argument(
                "anime_relation_claim requires PageKind::Anime",
            ));
        }
    }

    validate_source_materials(&input.source_materials)?;
    validate_companies(&input.companies)?;
    Ok(())
}

/// Checks the related source material of a work save.
///
/// # Errors
///
/// Returns an invalid-argument error for a blank title, type or relation type, a score outside
/// 0 to 10, a negative count, or no valid cross-reference.
fn validate_source_materials(source_materials: &[AnimeSourceMaterialInput]) -> Result<(), Error> {
    for material in source_materials {
        if material.title.trim().is_empty()
            || material.source_material_type.trim().is_empty()
            || material.relation_type.trim().is_empty()
        {
            return Err(Error::invalid_argument(
                "source_material requires non-empty title, type, and relation_type",
            ));
        }
        if let Some(score) = material.score
            && (!(0.0..=10.0).contains(&score) || score.is_nan())
        {
            return Err(Error::invalid_argument(
                "source_material score must be between 0 and 10",
            ));
        }
        if [material.chapters, material.volumes]
            .into_iter()
            .flatten()
            .any(|count| count < 0)
        {
            return Err(Error::invalid_argument(
                "source_material chapters and volumes must be >= 0",
            ));
        }
        if material.xrefs.is_empty()
            || material.xrefs.iter().any(|xref| {
                xref.provider.trim().is_empty()
                    || xref.external_id.trim().is_empty()
                    || xref.url.trim().is_empty()
            })
        {
            return Err(Error::invalid_argument(
                "source_material requires at least one valid cross-reference",
            ));
        }
    }
    Ok(())
}

/// Checks the companies of a work save.
///
/// # Errors
///
/// Returns an invalid-argument error for a blank name, no role, or no valid cross-reference.
fn validate_companies(companies: &[AnimeCompanyInput]) -> Result<(), Error> {
    for company in companies {
        if company.name.trim().is_empty() || company.roles.is_empty() {
            return Err(Error::invalid_argument(
                "company requires non-empty name and at least one role",
            ));
        }
        if company.xrefs.is_empty()
            || company
                .xrefs
                .iter()
                .any(|xref| xref.provider.trim().is_empty() || xref.url.trim().is_empty())
        {
            return Err(Error::invalid_argument(
                "company requires at least one valid cross-reference",
            ));
        }
    }
    Ok(())
}

/// Links a source page to a work and resolves pending relations.
///
/// A page that was never fetched is linked too, and becomes due for its first fetch.
/// Pending `anime_relation` rows targeting `page_key` are resolved, and the affected
/// `graph_edge` pairs are recomputed.
///
/// # Examples
///
/// ```no_run
/// # use sqlx::PgPool;
/// # use uuid::Uuid;
/// # use postgres_db::{link_anime_source_page, PageKey, PageKind, Provider};
/// # async fn example(pool: &PgPool, anime_id: Uuid) -> Result<(), postgres_db::Error> {
/// let key = PageKey::new(Provider::Mal, PageKind::Anime, "21")?;
/// let resolved = link_anime_source_page(pool, anime_id, &key).await?;
/// assert_eq!(resolved, 0);
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns [`Error::invalid_argument`] if `page_key.page_kind` is not
/// [`PageKind::Anime`], or a database error when any query fails.
pub async fn link_anime_source_page(
    pool: &PgPool,
    anime_id: Uuid,
    page_key: &PageKey,
) -> Result<u64, Error> {
    if page_key.page_kind() != PageKind::Anime {
        return Err(Error::invalid_argument(
            "link_anime_source_page requires PageKind::Anime",
        ));
    }

    let mut tx = pool.begin().await?;
    let resolved_count = link_anime_source_page_in_tx(&mut tx, anime_id, page_key).await?;
    tx.commit().await?;
    Ok(resolved_count)
}

/// Links `page_key` to a work as a source page and resolves relations pointing at it.
///
/// A page the work already holds as a partial source becomes its source. A new link
/// raises the work's version. The result counts the relations it resolved.
///
/// # Errors
///
/// Returns a not-found error for a missing or deleted work, an invalid-argument error when
/// the page is already another work's source page, or a database error when a query fails.
pub(crate) async fn link_anime_source_page_in_tx(
    tx: &mut Transaction<'_, Postgres>,
    anime_id: Uuid,
    page_key: &PageKey,
) -> Result<u64, Error> {
    let anime_exists = sqlx::query_scalar!(
        "SELECT 1 FROM anime WHERE id = $1 AND deleted_at IS NULL FOR NO KEY UPDATE",
        anime_id,
    )
    .fetch_optional(&mut **tx)
    .await?
    .is_some();
    if !anime_exists {
        return Err(Error::not_found());
    }

    // Serialize page linking with replace_anime_relations target resolution.
    sqlx::query!(
        "SELECT pg_advisory_xact_lock(hashtextextended($1 || ':anime:' || $2, 0))",
        page_key.provider().as_str(),
        page_key.address().as_str(),
    )
    .execute(&mut **tx)
    .await?;

    sqlx::query!(
        "INSERT INTO provider_page (provider, page_kind, address)
         VALUES ($1, $2, $3)
         ON CONFLICT (provider, page_kind, address) DO NOTHING",
        page_key.provider().as_str(),
        page_key.page_kind().as_str(),
        page_key.address().as_str(),
    )
    .execute(&mut **tx)
    .await?;

    let source_owner_id: Option<Uuid> = sqlx::query_scalar!(
        "SELECT anime_id
         FROM anime_provider_page
         WHERE provider = $1 AND page_kind = $2 AND address = $3 AND role = 'source'",
        page_key.provider().as_str(),
        page_key.page_kind().as_str(),
        page_key.address().as_str(),
    )
    .fetch_optional(&mut **tx)
    .await?;
    match source_owner_id {
        Some(owner_id) if owner_id != anime_id => {
            return Err(Error::invalid_argument(format!(
                "source page {}:{} is already linked to another work",
                page_key.provider(),
                page_key.address()
            )));
        }
        Some(_) => {}
        None => {
            sqlx::query!(
                "INSERT INTO anime_provider_page (anime_id, provider, page_kind, address, role)
                 VALUES ($1, $2, $3, $4, 'source')
                 ON CONFLICT (anime_id, provider, page_kind, address)
                 DO UPDATE SET role = 'source'",
                anime_id,
                page_key.provider().as_str(),
                page_key.page_kind().as_str(),
                page_key.address().as_str(),
            )
            .execute(&mut **tx)
            .await?;
            sqlx::query!(
                "UPDATE anime
                 SET version = version + 1,
                     merge_requested_at = COALESCE(merge_requested_at, now())
                 WHERE id = $1",
                anime_id,
            )
            .execute(&mut **tx)
            .await?;
        }
    }

    let resolved_sources: Vec<Uuid> = sqlx::query_scalar!(
        "UPDATE anime_relation
         SET to_anime_id = $3
         WHERE to_anime_id IS NULL
           AND to_page_provider = $1
           AND to_page_address = $2
           AND from_anime_id <> $3
         RETURNING from_anime_id",
        page_key.provider().as_str(),
        page_key.address().as_str(),
        anime_id,
    )
    .fetch_all(&mut **tx)
    .await?;

    let resolved_count = u64::try_from(resolved_sources.len()).unwrap_or(u64::MAX);
    let affected_pairs: BTreeSet<(Uuid, Uuid)> = resolved_sources
        .into_iter()
        .map(|from_anime_id| (from_anime_id, anime_id))
        .collect();
    if !affected_pairs.is_empty() {
        recompute_anime_pair_graph_edges(tx, affected_pairs).await?;
    }

    Ok(resolved_count)
}

/// Replaces a work's relations, claims, source material and companies, with their graph edges.
///
/// # Errors
///
/// Returns an invalid-argument error for an unknown relation type, or a database error
/// when a query fails.
pub(crate) async fn replace_relations_and_graph(
    tx: &mut Transaction<'_, Postgres>,
    anime_id: Uuid,
    input: &AnimeRelationsAndGraphInput,
) -> Result<(), Error> {
    let affected_pairs = replace_anime_relations(tx, anime_id, &input.relations).await?;
    replace_anime_relation_claims(tx, anime_id, &input.relation_claims).await?;
    recompute_anime_pair_graph_edges(tx, affected_pairs).await?;

    let linked_source_ids =
        replace_anime_source_materials(tx, anime_id, &input.source_materials).await?;
    replace_source_material_graph_edges(tx, anime_id, &linked_source_ids).await?;

    let company_roles = replace_anime_companies(tx, anime_id, &input.companies).await?;
    replace_company_graph_edges(tx, anime_id, &company_roles).await?;

    Ok(())
}

/// Replaces the relations a work asserts and resolves the targets already stored.
///
/// The result holds every work pair whose graph edges must be recomputed.
///
/// # Errors
///
/// Returns a database error when a query fails.
async fn replace_anime_relations(
    tx: &mut Transaction<'_, Postgres>,
    anime_id: Uuid,
    relations: &[AnimeRelationInput],
) -> Result<BTreeSet<(Uuid, Uuid)>, Error> {
    let deduplicated_relations: BTreeSet<(&str, &str, &str)> = relations
        .iter()
        .map(|relation| {
            (
                relation.relation_type.as_str(),
                relation.to_page_provider.as_str(),
                relation.to_page_address.as_str(),
            )
        })
        .collect();

    // Lock target page keys in sorted order before modifying relations so
    // concurrent link_anime_source_page calls cannot race between delete/lookup/insert.
    let target_pages: BTreeSet<(&str, &str)> = deduplicated_relations
        .iter()
        .map(|(_, provider, address)| (*provider, *address))
        .collect();
    for (provider, address) in target_pages {
        sqlx::query!(
            "SELECT pg_advisory_xact_lock(hashtextextended($1 || ':anime:' || $2, 0))",
            provider,
            address,
        )
        .execute(&mut **tx)
        .await?;
    }

    let deleted_targets: Vec<Option<Uuid>> = sqlx::query_scalar!(
        "DELETE FROM anime_relation
         WHERE from_anime_id = $1
         RETURNING to_anime_id",
        anime_id,
    )
    .fetch_all(&mut **tx)
    .await?;

    let mut affected_pairs: BTreeSet<(Uuid, Uuid)> = deleted_targets
        .into_iter()
        .flatten()
        .map(|target_id| (anime_id, target_id))
        .collect();

    for (relation_type, to_page_provider, to_page_address) in deduplicated_relations {
        let resolved_target = sqlx::query_scalar!(
            "SELECT anime_id
             FROM anime_provider_page
             WHERE provider = $1
               AND page_kind = 'anime'
               AND address = $2
               AND role = 'source'
               AND anime_id <> $3",
            to_page_provider,
            to_page_address,
            anime_id,
        )
        .fetch_optional(&mut **tx)
        .await?;

        if let Some(target_id) = resolved_target {
            affected_pairs.insert((anime_id, target_id));
        }

        sqlx::query!(
            "INSERT INTO anime_relation (
                 from_anime_id, relation_type, to_anime_id, to_page_provider, to_page_address
             )
             VALUES ($1, $2, $3, $4, $5)",
            anime_id,
            relation_type,
            resolved_target,
            to_page_provider,
            to_page_address,
        )
        .execute(&mut **tx)
        .await?;
    }

    Ok(affected_pairs)
}

/// Replaces each provider's own relation claims for a work.
///
/// # Errors
///
/// Returns a database error when a query fails.
async fn replace_anime_relation_claims(
    tx: &mut Transaction<'_, Postgres>,
    anime_id: Uuid,
    claims: &[AnimeRelationClaimInput],
) -> Result<(), Error> {
    sqlx::query!(
        "DELETE FROM anime_relation_claim WHERE from_anime_id = $1",
        anime_id
    )
    .execute(&mut **tx)
    .await?;

    let deduplicated_relation_claims: BTreeSet<(&str, &str, &str, &str)> = claims
        .iter()
        .map(|claim| {
            (
                claim.provider.as_str(),
                claim.relation_type.as_str(),
                claim.target_page.provider().as_str(),
                claim.target_page.address().as_str(),
            )
        })
        .collect();
    if deduplicated_relation_claims.is_empty() {
        return Ok(());
    }

    let providers: Vec<String> = deduplicated_relation_claims
        .iter()
        .map(|(p, _, _, _)| (*p).to_owned())
        .collect();
    let relation_types: Vec<String> = deduplicated_relation_claims
        .iter()
        .map(|(_, r, _, _)| (*r).to_owned())
        .collect();
    let target_providers: Vec<String> = deduplicated_relation_claims
        .iter()
        .map(|(_, _, tp, _)| (*tp).to_owned())
        .collect();
    let target_addresses: Vec<String> = deduplicated_relation_claims
        .into_iter()
        .map(|(_, _, _, ta)| ta.to_owned())
        .collect();

    sqlx::query!(
        "INSERT INTO anime_relation_claim (
             from_anime_id, provider, relation_type, to_page_provider, to_page_address
         )
         SELECT $1, provider, relation_type, to_page_provider, to_page_address
         FROM UNNEST($2::text[], $3::text[], $4::text[], $5::text[])
           AS entry(provider, relation_type, to_page_provider, to_page_address)",
        anime_id,
        &providers,
        &relation_types,
        &target_providers,
        &target_addresses,
    )
    .execute(&mut **tx)
    .await?;

    Ok(())
}

/// Replaces a work's source material links, creating or updating the shared rows.
///
/// # Errors
///
/// Returns a database error when a query fails.
async fn replace_anime_source_materials(
    tx: &mut Transaction<'_, Postgres>,
    anime_id: Uuid,
    materials: &[AnimeSourceMaterialInput],
) -> Result<Vec<Uuid>, Error> {
    sqlx::query!(
        "DELETE FROM anime_source_material_relation WHERE anime_id = $1",
        anime_id
    )
    .execute(&mut **tx)
    .await?;

    // Lock all source-material xref keys in sorted order before touching shared
    // rows so concurrent saves cannot deadlock or create duplicate rows.
    let mut xref_locks: BTreeSet<(&str, &str)> = BTreeSet::new();
    for material in materials {
        for xref in &material.xrefs {
            xref_locks.insert((xref.provider.as_str(), xref.external_id.as_str()));
        }
    }
    for (provider, external_id) in xref_locks {
        sqlx::query!(
            "SELECT pg_advisory_xact_lock(hashtextextended('source_material_xref:' || $1 || ':' || $2, 0))",
            provider,
            external_id,
        )
        .execute(&mut **tx)
        .await?;
    }

    let mut ordered_materials: Vec<&AnimeSourceMaterialInput> = materials.iter().collect();
    ordered_materials.sort_by_key(|material| {
        material
            .xrefs
            .iter()
            .map(|xref| (xref.provider.as_str(), xref.external_id.as_str()))
            .min()
    });

    // Resolve canonical IDs for every material first, rejecting conflicting owners.
    let mut resolved_materials: Vec<(Option<Uuid>, &AnimeSourceMaterialInput)> =
        Vec::with_capacity(ordered_materials.len());
    let mut existing_ids: BTreeSet<Uuid> = BTreeSet::new();
    for material in ordered_materials {
        let existing_id = resolve_existing_source_material_id(tx, &material.xrefs).await?;
        if let Some(id) = existing_id {
            existing_ids.insert(id);
        }
        resolved_materials.push((existing_id, material));
    }

    // Lock existing source_material rows in ascending UUID order to prevent deadlocks
    // when two concurrent saves touch the same entities via different xref aliases.
    if !existing_ids.is_empty() {
        let locked_ids: Vec<Uuid> = existing_ids.into_iter().collect();
        sqlx::query_scalar!(
            "SELECT id FROM source_material WHERE id = ANY($1) ORDER BY id FOR NO KEY UPDATE",
            &locked_ids,
        )
        .fetch_all(&mut **tx)
        .await?;
    }

    let mut linked_source_ids = Vec::new();
    let mut seen_links: BTreeSet<(Uuid, String)> = BTreeSet::new();

    for (existing_id, material) in resolved_materials {
        // A material listed twice in one save finds the row its first entry just created.
        let existing_id = match existing_id {
            Some(found_id) => Some(found_id),
            None => resolve_existing_source_material_id(tx, &material.xrefs).await?,
        };
        let source_material_id = upsert_source_material_by_xref(tx, existing_id, material).await?;
        linked_source_ids.push(source_material_id);
        if seen_links.insert((source_material_id, material.relation_type.clone())) {
            sqlx::query!(
                "INSERT INTO anime_source_material_relation (
                     anime_id, source_material_id, relation_type
                 )
                 VALUES ($1, $2, $3)",
                anime_id,
                source_material_id,
                material.relation_type,
            )
            .execute(&mut **tx)
            .await?;
        }
    }

    Ok(linked_source_ids)
}

/// Finds the stored source material that any of `xrefs` already identifies.
///
/// # Errors
///
/// Returns a database error when a query fails.
async fn resolve_existing_source_material_id(
    tx: &mut Transaction<'_, Postgres>,
    xrefs: &[SourceMaterialXrefInput],
) -> Result<Option<Uuid>, Error> {
    let mut resolved_ids: BTreeSet<Uuid> = BTreeSet::new();
    for xref in xrefs {
        if let Some(found_id) = sqlx::query_scalar!(
            "SELECT source_material_id
             FROM source_material_xref
             WHERE provider = $1 AND external_id = $2",
            xref.provider,
            xref.external_id,
        )
        .fetch_optional(&mut **tx)
        .await?
        {
            resolved_ids.insert(found_id);
        }
    }
    if resolved_ids.len() > 1 {
        return Err(Error::invalid_argument(
            "source_material xrefs resolve to conflicting source_material rows",
        ));
    }
    Ok(resolved_ids.into_iter().next())
}

/// Creates or updates a source material row and its cross-references.
///
/// # Errors
///
/// Returns a database error when a query fails.
async fn upsert_source_material_by_xref(
    tx: &mut Transaction<'_, Postgres>,
    existing_id: Option<Uuid>,
    material: &AnimeSourceMaterialInput,
) -> Result<Uuid, Error> {
    let mut sorted_xrefs: Vec<&SourceMaterialXrefInput> = material.xrefs.iter().collect();
    sorted_xrefs.sort_by(|left, right| {
        (&left.provider, &left.external_id).cmp(&(&right.provider, &right.external_id))
    });

    let source_material_id = if let Some(found_id) = existing_id {
        sqlx::query!(
            "UPDATE source_material
             SET title = $2,
                 title_english = $3,
                 title_japanese = $4,
                 type = $5,
                 status = $6,
                 score = ROUND($7::float8::numeric, 2),
                 chapters = $8,
                 volumes = $9,
                 updated_at = now()
             WHERE id = $1",
            found_id,
            material.title,
            material.title_english,
            material.title_japanese,
            material.source_material_type,
            material.status,
            material.score,
            material.chapters,
            material.volumes,
        )
        .execute(&mut **tx)
        .await?;
        found_id
    } else {
        sqlx::query_scalar!(
            "INSERT INTO source_material (
                 title, title_english, title_japanese, type, status, score, chapters, volumes
             )
             VALUES ($1, $2, $3, $4, $5, ROUND($6::float8::numeric, 2), $7, $8)
             RETURNING id",
            material.title,
            material.title_english,
            material.title_japanese,
            material.source_material_type,
            material.status,
            material.score,
            material.chapters,
            material.volumes,
        )
        .fetch_one(&mut **tx)
        .await?
    };

    let mut canonical_id = source_material_id;
    for xref in sorted_xrefs {
        canonical_id = sqlx::query_scalar!(
            "INSERT INTO source_material_xref (provider, external_id, source_material_id, source)
             VALUES ($1, $2, $3, $4)
             ON CONFLICT (provider, external_id) DO UPDATE SET source = EXCLUDED.source
             RETURNING source_material_id",
            xref.provider,
            xref.external_id,
            canonical_id,
            xref.url,
        )
        .fetch_one(&mut **tx)
        .await?;
    }

    replace_source_material_images(tx, canonical_id, &material.images).await?;
    Ok(canonical_id)
}

/// Replaces a source material's images.
///
/// # Errors
///
/// Returns a database error when a query fails.
async fn replace_source_material_images(
    tx: &mut Transaction<'_, Postgres>,
    source_material_id: Uuid,
    images: &[String],
) -> Result<(), Error> {
    sqlx::query!(
        "DELETE FROM source_material_image WHERE source_material_id = $1",
        source_material_id,
    )
    .execute(&mut **tx)
    .await?;

    let unique_urls: Vec<String> = images
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(str::to_owned)
        .collect();
    if unique_urls.is_empty() {
        return Ok(());
    }

    sqlx::query!(
        "INSERT INTO source_material_image (source_material_id, source)
         SELECT $1, source
         FROM UNNEST($2::text[]) AS entry(source)",
        source_material_id,
        &unique_urls,
    )
    .execute(&mut **tx)
    .await?;
    Ok(())
}

/// Replaces a work's company links, creating or updating the shared company rows.
///
/// The result holds each linked company with its role, for the graph edges.
///
/// # Errors
///
/// Returns a database error when a query fails.
async fn replace_anime_companies(
    tx: &mut Transaction<'_, Postgres>,
    anime_id: Uuid,
    companies: &[AnimeCompanyInput],
) -> Result<Vec<(Uuid, String)>, Error> {
    sqlx::query!("DELETE FROM anime_company WHERE anime_id = $1", anime_id)
        .execute(&mut **tx)
        .await?;

    // Lock all company xref keys in sorted order before touching shared rows so
    // concurrent saves cannot deadlock or create duplicate company rows.
    let mut xref_locks: BTreeSet<(&str, &str)> = BTreeSet::new();
    for company in companies {
        for xref in &company.xrefs {
            xref_locks.insert((xref.provider.as_str(), xref.url.as_str()));
        }
    }
    for (provider, url) in xref_locks {
        sqlx::query!(
            "SELECT pg_advisory_xact_lock(hashtextextended('company_xref:' || $1 || ':' || $2, 0))",
            provider,
            url,
        )
        .execute(&mut **tx)
        .await?;
    }

    let mut ordered_companies: Vec<&AnimeCompanyInput> = companies.iter().collect();
    ordered_companies.sort_by_key(|company| {
        company
            .xrefs
            .iter()
            .map(|xref| (xref.provider.as_str(), xref.url.as_str()))
            .min()
    });

    // Resolve canonical company IDs first, rejecting conflicting owners.
    let mut resolved_companies: Vec<(Option<Uuid>, &AnimeCompanyInput)> =
        Vec::with_capacity(ordered_companies.len());
    let mut existing_ids: BTreeSet<Uuid> = BTreeSet::new();
    for company in ordered_companies {
        let existing_id = resolve_existing_company_id(tx, &company.xrefs).await?;
        if let Some(id) = existing_id {
            existing_ids.insert(id);
        }
        resolved_companies.push((existing_id, company));
    }

    // Lock existing company rows in ascending UUID order before updating metadata.
    if !existing_ids.is_empty() {
        let locked_ids: Vec<Uuid> = existing_ids.into_iter().collect();
        sqlx::query_scalar!(
            "SELECT id FROM company WHERE id = ANY($1) ORDER BY id FOR NO KEY UPDATE",
            &locked_ids,
        )
        .fetch_all(&mut **tx)
        .await?;
    }

    let mut unique_links: BTreeSet<(Uuid, String)> = BTreeSet::new();
    for (existing_id, company) in resolved_companies {
        // A company listed twice in one save finds the row its first entry just created.
        let existing_id = match existing_id {
            Some(found_id) => Some(found_id),
            None => resolve_existing_company_id(tx, &company.xrefs).await?,
        };
        let company_id = upsert_company_by_xref(tx, existing_id, company).await?;
        for role in &company.roles {
            unique_links.insert((company_id, role.clone()));
        }
    }

    let company_roles: Vec<(Uuid, String)> = unique_links.into_iter().collect();
    if company_roles.is_empty() {
        return Ok(company_roles);
    }

    let company_ids: Vec<Uuid> = company_roles.iter().map(|(id, _)| *id).collect();
    let roles: Vec<String> = company_roles.iter().map(|(_, role)| role.clone()).collect();

    sqlx::query!(
        "INSERT INTO anime_company (anime_id, company_id, role)
         SELECT $1, company_id, role
         FROM UNNEST($2::uuid[], $3::text[]) AS entry(company_id, role)",
        anime_id,
        &company_ids,
        &roles,
    )
    .execute(&mut **tx)
    .await?;

    Ok(company_roles)
}

/// Finds the stored company that any of `xrefs` already identifies.
///
/// # Errors
///
/// Returns a database error when a query fails.
async fn resolve_existing_company_id(
    tx: &mut Transaction<'_, Postgres>,
    xrefs: &[CompanyXrefInput],
) -> Result<Option<Uuid>, Error> {
    let mut resolved_ids: BTreeSet<Uuid> = BTreeSet::new();
    for xref in xrefs {
        if let Some(found_id) = sqlx::query_scalar!(
            "SELECT company_id FROM company_xref WHERE provider = $1 AND source = $2",
            xref.provider,
            xref.url,
        )
        .fetch_optional(&mut **tx)
        .await?
        {
            resolved_ids.insert(found_id);
        }
    }
    if resolved_ids.len() > 1 {
        return Err(Error::invalid_argument(
            "company xrefs resolve to conflicting company rows",
        ));
    }
    Ok(resolved_ids.into_iter().next())
}

/// Creates or updates a company row and its cross-references.
///
/// # Errors
///
/// Returns a database error when a query fails.
async fn upsert_company_by_xref(
    tx: &mut Transaction<'_, Postgres>,
    existing_id: Option<Uuid>,
    company: &AnimeCompanyInput,
) -> Result<Uuid, Error> {
    let mut sorted_xrefs: Vec<&CompanyXrefInput> = company.xrefs.iter().collect();
    sorted_xrefs
        .sort_by(|left, right| (&left.provider, &left.url).cmp(&(&right.provider, &right.url)));

    let company_id = if let Some(found_id) = existing_id {
        sqlx::query!(
            "UPDATE company SET name = $2, description = $3 WHERE id = $1",
            found_id,
            company.name,
            company.description,
        )
        .execute(&mut **tx)
        .await?;
        found_id
    } else {
        sqlx::query_scalar!(
            "INSERT INTO company (name, description) VALUES ($1, $2) RETURNING id",
            company.name,
            company.description,
        )
        .fetch_one(&mut **tx)
        .await?
    };

    let mut xrefs_by_provider: BTreeMap<&str, &str> = BTreeMap::new();
    for xref in sorted_xrefs {
        xrefs_by_provider.insert(xref.provider.as_str(), xref.url.as_str());
    }

    let mut canonical_id = company_id;
    for (provider, url) in xrefs_by_provider {
        canonical_id = sqlx::query_scalar!(
            "INSERT INTO company_xref (provider, source, company_id)
             VALUES ($1, $2, $3)
             ON CONFLICT (provider, source) DO UPDATE SET source = EXCLUDED.source
             RETURNING company_id",
            provider,
            url,
            canonical_id,
        )
        .fetch_one(&mut **tx)
        .await?;
    }

    Ok(canonical_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_valid_relations() -> AnimeRelationsAndGraphInput {
        AnimeRelationsAndGraphInput {
            relations: vec![AnimeRelationInput {
                relation_type: "SEQUEL".to_owned(),
                to_page_provider: "mal".to_owned(),
                to_page_address: "2".to_owned(),
            }],
            relation_claims: vec![AnimeRelationClaimInput {
                provider: Provider::Mal,
                relation_type: "SEQUEL".to_owned(),
                target_page: PageKey::new(Provider::Mal, PageKind::Anime, "2").unwrap(),
            }],
            source_materials: vec![AnimeSourceMaterialInput {
                relation_type: "ADAPTATION".to_owned(),
                title: "Source Manga".to_owned(),
                title_english: None,
                title_japanese: None,
                source_material_type: "MANGA".to_owned(),
                status: None,
                score: Some(8.0),
                chapters: Some(10),
                volumes: Some(2),
                images: Vec::new(),
                xrefs: vec![SourceMaterialXrefInput {
                    provider: "mal".to_owned(),
                    external_id: "13".to_owned(),
                    url: "https://myanimelist.net/manga/13".to_owned(),
                }],
            }],
            companies: vec![AnimeCompanyInput {
                name: "Studio X".to_owned(),
                description: None,
                roles: vec!["STUDIO".to_owned()],
                xrefs: vec![CompanyXrefInput {
                    provider: "mal".to_owned(),
                    url: "https://myanimelist.net/anime/producer/1".to_owned(),
                }],
            }],
        }
    }

    #[test]
    fn validate_relations_and_graph_broken_rule_returns_invalid_argument() {
        let valid_relations = build_valid_relations();
        assert!(validate_relations_and_graph(&valid_relations).is_ok());

        let breaking_changes: Vec<fn(&mut AnimeRelationsAndGraphInput)> = vec![
            |input| input.relations[0].relation_type = "NOT_A_TYPE".to_owned(),
            |input| input.relations[0].to_page_provider = " ".to_owned(),
            |input| input.relations[0].to_page_address = String::new(),
            |input| input.relation_claims[0].relation_type = "NOT_A_TYPE".to_owned(),
            |input| {
                input.relation_claims[0].target_page =
                    PageKey::new(Provider::Mal, PageKind::Episode, "2").unwrap();
            },
            |input| input.source_materials[0].title = " ".to_owned(),
            |input| input.source_materials[0].source_material_type = String::new(),
            |input| input.source_materials[0].relation_type = String::new(),
            |input| input.source_materials[0].score = Some(10.5),
            |input| input.source_materials[0].score = Some(f64::NAN),
            |input| input.source_materials[0].chapters = Some(-1),
            |input| input.source_materials[0].volumes = Some(-1),
            |input| input.source_materials[0].xrefs.clear(),
            |input| input.source_materials[0].xrefs[0].provider = String::new(),
            |input| input.source_materials[0].xrefs[0].external_id = " ".to_owned(),
            |input| input.source_materials[0].xrefs[0].url = String::new(),
            |input| input.companies[0].name = String::new(),
            |input| input.companies[0].roles.clear(),
            |input| input.companies[0].xrefs.clear(),
            |input| input.companies[0].xrefs[0].provider = " ".to_owned(),
            |input| input.companies[0].xrefs[0].url = String::new(),
        ];
        for break_rule in breaking_changes {
            let mut broken_relations = valid_relations.clone();
            break_rule(&mut broken_relations);

            let validation_error = validate_relations_and_graph(&broken_relations)
                .expect_err("a broken rule must be refused");

            assert!(validation_error.is_invalid_argument());
        }
    }
}
