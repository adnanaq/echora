//! Bidirectional `graph_edge` projection and deadlock-free pairwise advisory locks.

use std::collections::BTreeSet;

use sqlx::{Postgres, Transaction};
use uuid::Uuid;

use crate::error::Error;

/// 64-bit FNV-1a offset basis used for deterministic cross-process advisory lock keys.
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;

/// 64-bit FNV-1a prime multiplier.
const FNV_PRIME: u64 = 0x0100_0000_01b3;

/// Returns the inverse `graph_edge_type` for a directed anime relation.
///
/// # Errors
///
/// Returns [`Error::invalid_argument`] when `relation_type` is not a recognized
/// `anime_relation_type` lookup value.
pub(crate) fn invert_anime_relation_type(
    relation_type: &str,
) -> Result<Option<&'static str>, Error> {
    match relation_type {
        "SEQUEL" => Ok(Some("PREQUEL")),
        "PREQUEL" => Ok(Some("SEQUEL")),
        "SPIN_OFF" | "SIDE_STORY" => Ok(Some("PARENT_STORY")),
        "PARENT_STORY" => Ok(Some("SIDE_STORY")),
        "SUMMARY" => Ok(Some("FULL_STORY")),
        "FULL_STORY" => Ok(Some("SUMMARY")),
        "SAME_FRANCHISE" => Ok(Some("SAME_FRANCHISE")),
        "SAME_SETTING" => Ok(Some("SAME_SETTING")),
        "ALTERNATIVE_VERSION" => Ok(Some("ALTERNATIVE_VERSION")),
        "ALTERNATIVE_SETTING" => Ok(Some("ALTERNATIVE_SETTING")),
        "CROSSOVER" => Ok(Some("CROSSOVER")),
        "ADAPTATION" | "CHARACTER" | "OTHER" => Ok(None),
        unknown => Err(Error::invalid_argument(format!(
            "unknown anime_relation_type {unknown:?}"
        ))),
    }
}

/// Computes a deterministic 64-bit advisory lock key for an unordered work pair.
#[must_use]
pub(crate) fn pair_advisory_lock_key(first_id: Uuid, second_id: Uuid) -> i64 {
    let (left_id, right_id) = if first_id <= second_id {
        (first_id, second_id)
    } else {
        (second_id, first_id)
    };
    let mut hash = FNV_OFFSET_BASIS;
    for byte in left_id.as_bytes().iter().chain(right_id.as_bytes().iter()) {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    i64::from_ne_bytes(hash.to_ne_bytes())
}

/// Recomputes bidirectional `graph_edge` rows for the given work pairs.
///
/// Acquires pairwise advisory locks in ascending hash order to prevent deadlocks.
///
/// # Errors
///
/// Returns an invalid-argument error on unknown relation types or a database
/// error when any query fails.
pub(crate) async fn recompute_anime_pair_graph_edges(
    tx: &mut Transaction<'_, Postgres>,
    affected_pairs: BTreeSet<(Uuid, Uuid)>,
) -> Result<(), Error> {
    let mut locked_pairs: Vec<(i64, Uuid, Uuid)> = affected_pairs
        .into_iter()
        .filter(|(first_id, second_id)| first_id != second_id)
        .map(|(first_id, second_id)| {
            let (left_id, right_id) = if first_id <= second_id {
                (first_id, second_id)
            } else {
                (second_id, first_id)
            };
            (pair_advisory_lock_key(left_id, right_id), left_id, right_id)
        })
        .collect();

    // Acquire advisory locks in ascending order of hashed keys so concurrent
    // saves of related works cannot deadlock.
    locked_pairs.sort_unstable();
    locked_pairs.dedup();

    for (lock_key, _, _) in &locked_pairs {
        sqlx::query!("SELECT pg_advisory_xact_lock($1)", lock_key)
            .execute(&mut **tx)
            .await?;
    }

    for (_, left_id, right_id) in locked_pairs {
        recompute_single_anime_pair(tx, left_id, right_id).await?;
    }

    Ok(())
}

/// Recomputes the graph edges between two works from both works' relations.
///
/// # Errors
///
/// Returns an invalid-argument error for an unknown relation type, or a database error
/// when a query fails.
async fn recompute_single_anime_pair(
    tx: &mut Transaction<'_, Postgres>,
    left_id: Uuid,
    right_id: Uuid,
) -> Result<(), Error> {
    sqlx::query!(
        "DELETE FROM graph_edge
         WHERE from_node_type = 'anime'
           AND to_node_type = 'anime'
           AND ((from_id = $1 AND to_id = $2) OR (from_id = $2 AND to_id = $1))",
        left_id,
        right_id,
    )
    .execute(&mut **tx)
    .await?;

    // A relation reaches the work that holds its target page as a source page.
    let stored_relations = sqlx::query!(
        r#"SELECT relation.from_anime_id, target.anime_id AS to_anime_id, relation.relation_type
         FROM anime_related_anime relation
         JOIN anime_source target
           ON target.provider = relation.to_page_provider
          AND target.entity_type = 'anime'
          AND target.address = relation.to_page_address
         WHERE (relation.from_anime_id = $1 AND target.anime_id = $2)
            OR (relation.from_anime_id = $2 AND target.anime_id = $1)"#,
        left_id,
        right_id,
    )
    .fetch_all(&mut **tx)
    .await?;

    let mut edges: BTreeSet<(Uuid, Uuid, String)> = BTreeSet::new();
    for row in stored_relations {
        let target_id = row.to_anime_id;
        let inverse_type = invert_anime_relation_type(&row.relation_type)?;
        edges.insert((row.from_anime_id, target_id, row.relation_type));
        if let Some(reverse_edge_type) = inverse_type {
            edges.insert((target_id, row.from_anime_id, reverse_edge_type.to_owned()));
        }
    }

    if edges.is_empty() {
        return Ok(());
    }

    let from_ids: Vec<Uuid> = edges.iter().map(|(from_id, _, _)| *from_id).collect();
    let to_ids: Vec<Uuid> = edges.iter().map(|(_, to_id, _)| *to_id).collect();
    let edge_types: Vec<String> = edges
        .into_iter()
        .map(|(_, _, edge_type)| edge_type)
        .collect();

    sqlx::query!(
        "INSERT INTO graph_edge (from_id, from_node_type, to_id, to_node_type, edge_type)
         SELECT from_id, 'anime', to_id, 'anime', edge_type
         FROM UNNEST($1::uuid[], $2::uuid[], $3::text[]) AS entry(from_id, to_id, edge_type)
         ON CONFLICT DO NOTHING",
        &from_ids,
        &to_ids,
        &edge_types,
    )
    .execute(&mut **tx)
    .await?;

    Ok(())
}

/// Replaces bidirectional `graph_edge` rows linking `anime_id` to source materials.
///
/// # Errors
///
/// Returns a database error when any delete or batch insert fails.
pub(crate) async fn replace_source_material_graph_edges(
    tx: &mut Transaction<'_, Postgres>,
    anime_id: Uuid,
    source_material_ids: &[Uuid],
) -> Result<(), Error> {
    sqlx::query!(
        "DELETE FROM graph_edge
         WHERE (from_id = $1 AND from_node_type = 'anime' AND to_node_type = 'source_material')
            OR (to_id = $1 AND to_node_type = 'anime' AND from_node_type = 'source_material')",
        anime_id,
    )
    .execute(&mut **tx)
    .await?;

    let unique_ids: Vec<Uuid> = source_material_ids
        .iter()
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    if unique_ids.is_empty() {
        return Ok(());
    }

    sqlx::query!(
        "INSERT INTO graph_edge (from_id, from_node_type, to_id, to_node_type, edge_type)
         SELECT $1, 'anime', source_id, 'source_material', 'ADAPTED_FROM'
         FROM UNNEST($2::uuid[]) AS entry(source_id)
         ON CONFLICT DO NOTHING",
        anime_id,
        &unique_ids,
    )
    .execute(&mut **tx)
    .await?;

    sqlx::query!(
        "INSERT INTO graph_edge (from_id, from_node_type, to_id, to_node_type, edge_type)
         SELECT source_id, 'source_material', $1, 'anime', 'WAS_ADAPTED_TO'
         FROM UNNEST($2::uuid[]) AS entry(source_id)
         ON CONFLICT DO NOTHING",
        anime_id,
        &unique_ids,
    )
    .execute(&mut **tx)
    .await?;

    Ok(())
}

/// Replaces bidirectional `graph_edge` rows linking `anime_id` to companies.
///
/// # Errors
///
/// Returns a database error when any delete or batch insert fails.
pub(crate) async fn replace_company_graph_edges(
    tx: &mut Transaction<'_, Postgres>,
    anime_id: Uuid,
    company_roles: &[(Uuid, String)],
) -> Result<(), Error> {
    sqlx::query!(
        "DELETE FROM graph_edge
         WHERE (from_id = $1 AND from_node_type = 'anime' AND to_node_type = 'company')
            OR (to_id = $1 AND to_node_type = 'anime' AND from_node_type = 'company')",
        anime_id,
    )
    .execute(&mut **tx)
    .await?;

    let unique_links: BTreeSet<(Uuid, &str)> = company_roles
        .iter()
        .map(|(company_id, role)| (*company_id, role.as_str()))
        .collect();
    if unique_links.is_empty() {
        return Ok(());
    }

    let company_ids: Vec<Uuid> = unique_links.iter().map(|(id, _)| *id).collect();
    let roles: Vec<String> = unique_links
        .into_iter()
        .map(|(_, role)| role.to_owned())
        .collect();

    sqlx::query!(
        "INSERT INTO graph_edge (from_id, from_node_type, to_id, to_node_type, edge_type)
         SELECT $1, 'anime', company_id, 'company', role
         FROM UNNEST($2::uuid[], $3::text[]) AS entry(company_id, role)
         ON CONFLICT DO NOTHING",
        anime_id,
        &company_ids,
        &roles,
    )
    .execute(&mut **tx)
    .await?;

    sqlx::query!(
        "INSERT INTO graph_edge (from_id, from_node_type, to_id, to_node_type, edge_type)
         SELECT company_id, 'company', $1, 'anime', role
         FROM UNNEST($2::uuid[], $3::text[]) AS entry(company_id, role)
         ON CONFLICT DO NOTHING",
        anime_id,
        &company_ids,
        &roles,
    )
    .execute(&mut **tx)
    .await?;

    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    use sqlx::PgPool;

    #[test]
    fn invert_anime_relation_type_matches_design_rules() {
        assert_eq!(
            invert_anime_relation_type("SEQUEL").ok(),
            Some(Some("PREQUEL"))
        );
        assert_eq!(
            invert_anime_relation_type("PREQUEL").ok(),
            Some(Some("SEQUEL"))
        );
        assert_eq!(
            invert_anime_relation_type("SPIN_OFF").ok(),
            Some(Some("PARENT_STORY"))
        );
        assert_eq!(
            invert_anime_relation_type("SIDE_STORY").ok(),
            Some(Some("PARENT_STORY"))
        );
        assert_eq!(
            invert_anime_relation_type("PARENT_STORY").ok(),
            Some(Some("SIDE_STORY"))
        );
        assert_eq!(
            invert_anime_relation_type("SUMMARY").ok(),
            Some(Some("FULL_STORY"))
        );
        assert_eq!(
            invert_anime_relation_type("FULL_STORY").ok(),
            Some(Some("SUMMARY"))
        );
        assert_eq!(
            invert_anime_relation_type("SAME_FRANCHISE").ok(),
            Some(Some("SAME_FRANCHISE"))
        );
        assert_eq!(
            invert_anime_relation_type("SAME_SETTING").ok(),
            Some(Some("SAME_SETTING"))
        );
        assert_eq!(
            invert_anime_relation_type("ALTERNATIVE_VERSION").ok(),
            Some(Some("ALTERNATIVE_VERSION"))
        );
        assert_eq!(
            invert_anime_relation_type("ALTERNATIVE_SETTING").ok(),
            Some(Some("ALTERNATIVE_SETTING"))
        );
        assert_eq!(
            invert_anime_relation_type("CROSSOVER").ok(),
            Some(Some("CROSSOVER"))
        );
        assert_eq!(invert_anime_relation_type("ADAPTATION").ok(), Some(None));
        assert_eq!(invert_anime_relation_type("CHARACTER").ok(), Some(None));
        assert_eq!(invert_anime_relation_type("OTHER").ok(), Some(None));
        assert!(invert_anime_relation_type("INVALID").is_err());
    }

    #[test]
    fn pair_advisory_lock_key_is_symmetric() {
        let first_id = Uuid::from_u128(10);
        let second_id = Uuid::from_u128(20);
        assert_eq!(
            pair_advisory_lock_key(first_id, second_id),
            pair_advisory_lock_key(second_id, first_id)
        );
        assert_ne!(
            pair_advisory_lock_key(first_id, second_id),
            pair_advisory_lock_key(first_id, Uuid::from_u128(30))
        );
    }

    #[sqlx::test]
    async fn graph_projection_recomputes_all_bidirectional_edge_types(pool: PgPool) {
        let anime_a = Uuid::now_v7();
        let anime_b = Uuid::now_v7();
        let unrelated_anime = Uuid::now_v7();
        for (id, title) in [
            (anime_a, "Work A"),
            (anime_b, "Work B"),
            (unrelated_anime, "Unrelated Work"),
        ] {
            sqlx::query!(
                "INSERT INTO anime (id, title, type, status, rating)
                 VALUES ($1, $2, 'TV', 'FINISHED', 'PG-13 - Teens 13 or older')",
                id,
                title,
            )
            .execute(&pool)
            .await
            .unwrap();
        }

        sqlx::query!(
            "INSERT INTO provider_page (provider, entity_type, address) VALUES ('mal', 'anime', '2')"
        )
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query!(
            "INSERT INTO anime_source (anime_id, provider, entity_type, address)
             VALUES ($1, 'mal', 'anime', '2')",
            anime_b,
        )
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query!(
            "INSERT INTO anime_related_anime (from_anime_id, relation_type, to_page_provider, to_page_address)
             VALUES ($1, 'SEQUEL', 'mal', '2'),
                    ($1, 'ADAPTATION', 'mal', '2')",
            anime_a,
        )
        .execute(&pool)
        .await
        .unwrap();

        let mut pairs = BTreeSet::new();
        pairs.insert((anime_a, anime_a));
        pairs.insert((anime_b, anime_a));
        pairs.insert((anime_a, unrelated_anime));

        let mut tx = pool.begin().await.unwrap();
        recompute_anime_pair_graph_edges(&mut tx, pairs)
            .await
            .unwrap();
        tx.commit().await.unwrap();

        let anime_edges: Vec<(Uuid, Uuid, String)> = sqlx::query!(
            "SELECT from_id, to_id, edge_type
             FROM graph_edge
             WHERE from_node_type = 'anime' AND to_node_type = 'anime'
             ORDER BY edge_type"
        )
        .fetch_all(&pool)
        .await
        .unwrap()
        .into_iter()
        .map(|r| (r.from_id, r.to_id, r.edge_type))
        .collect();
        assert_eq!(
            anime_edges,
            vec![
                (anime_a, anime_b, "ADAPTATION".to_owned()),
                (anime_b, anime_a, "PREQUEL".to_owned()),
                (anime_a, anime_b, "SEQUEL".to_owned()),
            ]
        );

        let source_id = sqlx::query_scalar!(
            "INSERT INTO source_material (title, type) VALUES ('Manga A', 'MANGA') RETURNING id"
        )
        .fetch_one(&pool)
        .await
        .unwrap();

        let mut tx = pool.begin().await.unwrap();
        replace_source_material_graph_edges(&mut tx, anime_a, &[source_id, source_id])
            .await
            .unwrap();
        tx.commit().await.unwrap();

        let sm_edges: Vec<String> = sqlx::query_scalar!(
            "SELECT edge_type
             FROM graph_edge
             WHERE from_node_type IN ('anime', 'source_material')
               AND to_node_type IN ('anime', 'source_material')
               AND (from_node_type = 'source_material' OR to_node_type = 'source_material')
             ORDER BY edge_type"
        )
        .fetch_all(&pool)
        .await
        .unwrap();
        assert_eq!(
            sm_edges,
            vec!["ADAPTED_FROM".to_owned(), "WAS_ADAPTED_TO".to_owned()]
        );

        let company_id =
            sqlx::query_scalar!("INSERT INTO company (name) VALUES ('Studio X') RETURNING id")
                .fetch_one(&pool)
                .await
                .unwrap();

        let mut tx = pool.begin().await.unwrap();
        replace_company_graph_edges(&mut tx, anime_a, &[(company_id, "STUDIO".to_owned())])
            .await
            .unwrap();
        tx.commit().await.unwrap();

        let company_edges = sqlx::query_scalar!(
            "SELECT COUNT(*)
             FROM graph_edge
             WHERE (from_id = $1 AND to_id = $2) OR (from_id = $2 AND to_id = $1)",
            anime_a,
            company_id,
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(company_edges, Some(2));
    }
}
