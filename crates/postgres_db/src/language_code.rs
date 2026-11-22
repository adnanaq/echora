//! The shared `language` and `country` code tables every code column references.

use std::collections::BTreeSet;

use sqlx::{Postgres, Transaction};

use crate::Error;

/// Adds the language codes not yet stored to the shared `language` table.
///
/// The codes are written in sorted order, so two saves adding the same codes take
/// their row locks in the same order and cannot deadlock. The canonical model has
/// already standardized every code, so nothing here checks the BCP 47 form.
///
/// # Errors
///
/// Returns a database error when the insert fails.
pub(crate) async fn insert_language_codes(
    tx: &mut Transaction<'_, Postgres>,
    language_codes: impl IntoIterator<Item = &str>,
) -> Result<(), Error> {
    let sorted_codes: Vec<String> = language_codes
        .into_iter()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(str::to_owned)
        .collect();
    if sorted_codes.is_empty() {
        return Ok(());
    }
    sqlx::query!(
        "INSERT INTO language (code) SELECT * FROM UNNEST($1::text[]) ON CONFLICT DO NOTHING",
        &sorted_codes
    )
    .execute(&mut **tx)
    .await?;
    Ok(())
}

/// Adds a country code not yet stored to the shared `country` table.
///
/// # Errors
///
/// Returns a database error when the insert fails, including the table's check that
/// the code is two upper-case letters.
pub(crate) async fn insert_country_code(
    tx: &mut Transaction<'_, Postgres>,
    country_code: impl AsRef<str>,
) -> Result<(), Error> {
    sqlx::query!(
        "INSERT INTO country (code) VALUES ($1) ON CONFLICT (code) DO NOTHING",
        country_code.as_ref()
    )
    .execute(&mut **tx)
    .await?;
    Ok(())
}
