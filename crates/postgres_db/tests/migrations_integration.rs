//! Integration tests verifying `postgres_db` migrations against PostgreSQL 18.

// Rust guideline compliant 2026-10-09

use std::env;

use postgres_db::{Error, connect_pool, run_migrations};

#[tokio::test]
async fn lookup_and_name_tables_migrate_and_seed_expected_rows() -> Result<(), Error> {
    let Ok(database_url) = env::var("DATABASE_URL") else {
        return Ok(());
    };

    let pool = connect_pool(&database_url, 2).await?;
    run_migrations(&pool).await?;
    run_migrations(&pool).await?;

    let provider_count = sqlx::query_scalar!("SELECT COUNT(*) FROM provider")
        .fetch_one(&pool)
        .await?;
    assert_eq!(provider_count, Some(12));

    let extra_providers = sqlx::query_scalar!(
        "SELECT COUNT(*) FROM provider WHERE value IN ('ann', 'livechart', 'simkl', 'animecountdown')"
    )
    .fetch_one(&pool)
    .await?;
    assert_eq!(extra_providers, Some(4));

    let has_dead_page =
        sqlx::query_scalar!("SELECT EXISTS(SELECT 1 FROM approval_kind WHERE value = 'dead_page')")
            .fetch_one(&pool)
            .await?;
    assert_eq!(has_dead_page, Some(true));

    let graph_edge_count = sqlx::query_scalar!("SELECT COUNT(*) FROM graph_edge_type")
        .fetch_one(&pool)
        .await?;
    assert_eq!(graph_edge_count, Some(23));

    let shared_name_table_count = sqlx::query_scalar!(
        "SELECT COUNT(*) FROM information_schema.tables
         WHERE table_schema = 'public'
           AND table_name IN ('genre', 'tag', 'demographic', 'content_warning', 'trait')"
    )
    .fetch_one(&pool)
    .await?;
    assert_eq!(shared_name_table_count, Some(5));

    let Err(duplicate_sqlx_err) = sqlx::query("INSERT INTO anime_type VALUES ('TV')")
        .execute(&pool)
        .await
    else {
        panic!("duplicate lookup insert must fail");
    };
    let duplicate_err = Error::from(duplicate_sqlx_err);
    assert!(duplicate_err.is_unique_violation());

    Ok(())
}

#[tokio::test]
async fn connect_pool_rejects_invalid_database_url() {
    let Err(err) = connect_pool("not-a-postgres-url", 1).await else {
        panic!("invalid database URL must fail");
    };
    assert!(err.is_database());
}
