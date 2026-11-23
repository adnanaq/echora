//! Character loading, version-checked saving, and the works a character appears in.
//!
//! A character's own pages list the works it appears in and its role in each. A save stores
//! each such work by its page in `character_animeography`, and the work is found by joining
//! that page to `anime_source`, the way a relation finds its target. A work stored after the
//! character, or a page that moves between works, therefore needs no change to the
//! character's rows: linking the page reaches the character through
//! [`notify_characters_naming_anime_page`].
//!
//! The graph keeps one `FEATURES` edge from each work to the character and one `APPEARS_IN`
//! edge back, recomputed from that join whenever either side changes.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use chrono::{DateTime, Utc};
use sqlx::{PgPool, Postgres, Transaction};
use tracing::Level;
use uuid::Uuid;

use crate::anime::{
    FieldCorrectionRecord, LoadedSourcePage, decompress_body, insert_catalogue_outbox_event,
    parse_entity_type,
};
use crate::anime_relations::lock_anime_pages_shared;
use crate::{Error, PageKey, PageResponse, Provider};

/// Current canonical hash format version stored in `character.hash_format`.
pub const CHARACTER_HASH_FORMAT: i16 = 1;

/// Subject of the notice for a character's first merge.
const CHARACTER_CREATED_SUBJECT: &str = "character.created";

/// Subject of the notice for a later change to a character.
const CHARACTER_UPDATED_SUBJECT: &str = "character.updated";

/// A work a character's pages name, by the work's page, with the character's role there.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CharacterAppearanceInput {
    /// Provider of the work's page.
    pub anime_page_provider: String,
    /// Address of the work's page on `anime_page_provider`.
    pub anime_page_address: String,
    /// Canonical `character_role` lookup value (for example `"MAIN"`).
    pub role: String,
}

/// A manga or novel a character's pages name, by its provider identifier, with the role there.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CharacterMangaographyInput {
    /// Provider of the manga or novel page.
    pub source_material_provider: String,
    /// Identifier of the manga or novel on `source_material_provider`.
    pub source_material_external_id: String,
    /// Canonical `character_role` lookup value (for example `"MAIN"`).
    pub role: String,
}

/// One anime a character appears in, with the strongest role its pages give there.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CharacterAnime {
    /// The work's ID.
    pub anime_id: Uuid,
    /// Canonical `character_role` lookup value (for example `"MAIN"`).
    pub role: String,
}

/// Merge inputs returned by [`load_character`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadCharacterOutcome {
    /// Current optimistic-locking version of the character.
    pub version: i64,
    /// Linked character pages that were fetched, with their raw responses.
    pub pages: Vec<LoadedSourcePage>,
    /// Approved per-field corrections to apply after merging.
    pub corrections: Vec<FieldCorrectionRecord>,
}

/// Canonical character fields and merge metadata passed to [`save_character`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SaveCharacterInput {
    /// UUID of the existing character row to update.
    pub character_id: Uuid,
    /// Optimistic concurrency version expected in PostgreSQL.
    pub expected_version: i64,
    /// Display name.
    pub name: String,
    /// Name in the character's native script, if known.
    pub name_native: Option<String>,
    /// Description, if known; its spoiler is the `description` entry of `spoilers`.
    pub description: Option<String>,
    /// Favorites count (`>= 0`), if known.
    pub favorites: Option<i32>,
    /// Other spellings of the name.
    pub name_variations: Vec<String>,
    /// Nicknames.
    pub nicknames: Vec<String>,
    /// Trait names, stored once in the shared `trait` table.
    pub traits: Vec<String>,
    /// Image URLs.
    pub images: Vec<String>,
    /// Biographical attributes by name (for example `"Age"`).
    pub attributes: BTreeMap<String, String>,
    /// Spoiler values by the field they hide, the description's under `"description"`.
    pub spoilers: BTreeMap<String, String>,
    /// The works the character's pages name, each by the work's page.
    pub animeography: Vec<CharacterAppearanceInput>,
    /// The manga and novels the character's pages name.
    pub mangaography: Vec<CharacterMangaographyInput>,
    /// Canonical hash of the merged character record.
    pub merged_record_hash: String,
    /// Format identifier of `merged_record_hash`.
    pub hash_format: i16,
    /// Version of the mapping and merge code that produced this save.
    pub merge_version: i32,
    /// Optional W3C `traceparent` header propagated into outbox rows.
    pub traceparent: Option<String>,
}

/// Result returned by [`save_character`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SaveCharacterOutcome {
    /// Whether the canonical hash differed and database state was updated.
    pub has_changed: bool,
    /// Character version after the save attempt.
    pub version: i64,
}

/// Loads a character's version, fetched pages, and approved corrections for merging.
///
/// # Examples
///
/// ```no_run
/// # use sqlx::PgPool;
/// # use uuid::Uuid;
/// # use postgres_db::load_character;
/// # async fn example(pool: &PgPool, character_id: Uuid) -> Result<(), postgres_db::Error> {
/// // A character the character load just created, before its own first merge.
/// let outcome = load_character(pool, character_id).await?;
/// assert_eq!(outcome.version, 1);
/// assert_eq!(outcome.pages, Vec::new(), "its pages are linked but not fetched yet");
/// assert_eq!(outcome.corrections, Vec::new());
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns [`Error::is_not_found`] when `character_id` does not exist or is deleted, or a
/// database error when a query or a body's decompression fails.
pub async fn load_character(
    pool: &PgPool,
    character_id: Uuid,
) -> Result<LoadCharacterOutcome, Error> {
    let mut tx = pool.begin().await?;
    sqlx::query!("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        .execute(&mut *tx)
        .await?;

    let version = sqlx::query_scalar!(
        "SELECT version FROM character WHERE id = $1 AND deleted_at IS NULL",
        character_id
    )
    .fetch_optional(&mut *tx)
    .await?
    .ok_or_else(Error::not_found)?;

    let pages = load_character_pages(&mut tx, character_id).await?;
    let corrections = load_character_corrections(&mut tx, character_id).await?;
    tx.commit().await?;

    Ok(LoadCharacterOutcome {
        version,
        pages,
        corrections,
    })
}

/// Loads a character's fetched pages with their responses.
///
/// A linked page that was never fetched is left out, since the merge has nothing to read.
///
/// # Errors
///
/// Returns a database error when a query fails or a body cannot be decompressed, or an
/// invalid-argument error when a stored page key no longer passes validation.
async fn load_character_pages(
    tx: &mut Transaction<'_, Postgres>,
    character_id: Uuid,
) -> Result<Vec<LoadedSourcePage>, Error> {
    let linked_rows = sqlx::query!(
        r#"
        SELECT
            source.provider,
            source.entity_type,
            source.address,
            pp.fetched_at,
            pp.is_dead,
            pp.missing_since
        FROM character_source source
        JOIN provider_page pp
          ON pp.provider = source.provider
         AND pp.entity_type = source.entity_type
         AND pp.address = source.address
        WHERE source.character_id = $1
        ORDER BY source.provider, source.address
        "#,
        character_id
    )
    .fetch_all(&mut **tx)
    .await?;

    let mut responses_by_page = load_character_responses_by_page(tx, character_id).await?;
    let mut pages = Vec::with_capacity(linked_rows.len());
    for row in linked_rows {
        let Some(fetched_at) = row.fetched_at else {
            continue;
        };
        let provider: Provider = row.provider.parse()?;
        let entity_type = parse_entity_type(&row.entity_type)?;
        let page_key = PageKey::new(provider, entity_type, &row.address)?;
        let responses = responses_by_page
            .remove(&(row.provider, row.address))
            .unwrap_or_default();
        pages.push(LoadedSourcePage {
            page_key,
            fetched_at,
            is_dead: row.is_dead,
            is_not_found: row.missing_since.is_some(),
            responses,
        });
    }
    Ok(pages)
}

/// Loads the decompressed responses of a character's pages, grouped by page.
///
/// # Errors
///
/// Returns a database error when the query fails or a stored body cannot be decompressed.
async fn load_character_responses_by_page(
    tx: &mut Transaction<'_, Postgres>,
    character_id: Uuid,
) -> Result<HashMap<(String, String), Vec<PageResponse>>, Error> {
    let response_rows = sqlx::query!(
        r#"
        SELECT
            ppr.provider,
            ppr.address,
            ppr.request_name,
            ppr.content_type,
            ppr.body
        FROM character_source source
        JOIN provider_page_response ppr
          ON ppr.provider = source.provider
         AND ppr.entity_type = source.entity_type
         AND ppr.address = source.address
        WHERE source.character_id = $1
        ORDER BY ppr.provider, ppr.address, ppr.request_name
        "#,
        character_id
    )
    .fetch_all(&mut **tx)
    .await?;

    let mut responses_by_page: HashMap<(String, String), Vec<PageResponse>> = HashMap::new();
    for row in response_rows {
        let body = decompress_body(&row.body)?;
        responses_by_page
            .entry((row.provider, row.address))
            .or_default()
            .push(PageResponse {
                request_name: row.request_name,
                content_type: row.content_type,
                body,
            });
    }
    Ok(responses_by_page)
}

/// Loads a character's approved field corrections in field-path order.
///
/// # Errors
///
/// Returns a database error when the query fails.
async fn load_character_corrections(
    tx: &mut Transaction<'_, Postgres>,
    character_id: Uuid,
) -> Result<Vec<FieldCorrectionRecord>, Error> {
    let rows = sqlx::query!(
        r#"
        SELECT field_path, value_message, value, approved_by, approved_at
        FROM character_field_correction
        WHERE character_id = $1
        ORDER BY field_path
        "#,
        character_id
    )
    .fetch_all(&mut **tx)
    .await?;

    Ok(rows
        .into_iter()
        .map(|row| FieldCorrectionRecord {
            field_path: row.field_path,
            value_message: row.value_message,
            value: row.value,
            approved_by: row.approved_by,
            approved_at: row.approved_at,
        })
        .collect())
}

/// Validates and persists a merged character when its canonical hash changed.
///
/// The save stores every field, replaces the works the character appears in, recomputes
/// the character's graph edges, and writes `character.created` on the first merge or
/// `character.updated` after it, all in one transaction. An unchanged hash clears
/// `merge_requested_at` and records `merge_version`, and writes nothing else.
///
/// It takes the locks in the order every save does: a shared lock on each work page it
/// names, so it waits for a link of the page but not for other saves, then the shared
/// `trait` rows, then the character's own row.
///
/// # Examples
///
/// ```no_run
/// # use std::collections::BTreeMap;
/// # use sqlx::PgPool;
/// # use uuid::Uuid;
/// # use postgres_db::{
/// #     CHARACTER_HASH_FORMAT, CharacterAppearanceInput, SaveCharacterInput,
/// #     SaveCharacterOutcome, save_character,
/// # };
/// # async fn example(pool: &PgPool, character_id: Uuid) -> Result<(), postgres_db::Error> {
/// let input = SaveCharacterInput {
///     character_id,
///     expected_version: 1,
///     name: "Monkey D. Luffy".to_owned(),
///     name_native: Some("モンキー・D・ルフィ".to_owned()),
///     description: None,
///     favorites: Some(150_000),
///     name_variations: vec!["Luffy".to_owned()],
///     nicknames: vec!["Straw Hat".to_owned()],
///     traits: vec!["Rubber".to_owned()],
///     images: Vec::new(),
///     attributes: BTreeMap::from([("Age".to_owned(), "19".to_owned())]),
///     spoilers: BTreeMap::new(),
///     animeography: vec![CharacterAppearanceInput {
///         anime_page_provider: "mal".to_owned(),
///         anime_page_address: "21".to_owned(),
///         role: "MAIN".to_owned(),
///     }],
///     mangaography: Vec::new(),
///     merged_record_hash: "blake3-hash-value".to_owned(),
///     hash_format: CHARACTER_HASH_FORMAT,
///     merge_version: 1,
///     traceparent: None,
/// };
/// let first_save = save_character(pool, &input).await?;
/// assert_eq!(first_save, SaveCharacterOutcome { has_changed: true, version: 2 });
///
/// // The same merged record saved again writes nothing and keeps the version.
/// let repeated_input = SaveCharacterInput { expected_version: 2, ..input };
/// let repeated_save = save_character(pool, &repeated_input).await?;
/// assert_eq!(repeated_save, SaveCharacterOutcome { has_changed: false, version: 2 });
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns [`Error::is_invalid_argument`] for an empty or out-of-range field,
/// [`Error::is_not_found`] when `character_id` does not exist or is deleted,
/// [`Error::is_version_conflict`] when `expected_version` does not match, or a database
/// error when persistence fails, such as a foreign-key error for an unknown role.
pub async fn save_character(
    pool: &PgPool,
    input: &SaveCharacterInput,
) -> Result<SaveCharacterOutcome, Error> {
    validate_save_character_input(input)?;

    let mut tx = pool.begin().await?;
    let anime_pages: BTreeSet<(&str, &str)> = input
        .animeography
        .iter()
        .map(|appearance| {
            (
                appearance.anime_page_provider.as_str(),
                appearance.anime_page_address.as_str(),
            )
        })
        .collect();
    lock_anime_pages_shared(&mut tx, &anime_pages).await?;
    insert_traits(&mut tx, &input.traits).await?;

    let existing = sqlx::query!(
        r#"
        SELECT version, merged_record_hash, hash_format, merged_at
        FROM character
        WHERE id = $1 AND deleted_at IS NULL
        FOR UPDATE
        "#,
        input.character_id
    )
    .fetch_optional(&mut *tx)
    .await?
    .ok_or_else(Error::not_found)?;

    if existing.version != input.expected_version {
        return Err(Error::version_conflict(format!(
            "expected version {}, stored version {}",
            input.expected_version, existing.version
        )));
    }

    if existing.merged_record_hash.as_deref() == Some(input.merged_record_hash.as_str())
        && existing.hash_format == Some(input.hash_format)
    {
        sqlx::query!(
            r#"
            UPDATE character
            SET merge_requested_at = NULL,
                merge_version = $2
            WHERE id = $1
              AND (merge_requested_at IS NOT NULL OR merge_version IS DISTINCT FROM $2)
            "#,
            input.character_id,
            input.merge_version
        )
        .execute(&mut *tx)
        .await?;
        tx.commit().await?;
        return Ok(SaveCharacterOutcome {
            has_changed: false,
            version: existing.version,
        });
    }

    let next_version = existing.version + 1;
    let now = crate::read_transaction_time(&mut tx).await?;
    update_character_row(&mut tx, input, next_version, now).await?;
    replace_character_names(&mut tx, input).await?;
    replace_character_traits_and_images(&mut tx, input).await?;
    replace_character_attributes_and_spoilers(&mut tx, input).await?;
    replace_character_animeography(&mut tx, input.character_id, &input.animeography).await?;
    replace_character_mangaography(&mut tx, input.character_id, &input.mangaography).await?;
    recompute_character_graph_edges(&mut tx, input.character_id).await?;

    let subject = if existing.merged_at.is_none() {
        CHARACTER_CREATED_SUBJECT
    } else {
        CHARACTER_UPDATED_SUBJECT
    };
    insert_catalogue_outbox_event(
        &mut tx,
        subject,
        input.character_id,
        next_version,
        None,
        now,
        input.traceparent.as_deref(),
    )
    .await?;
    tx.commit().await?;

    tracing::event!(
        name: "postgres_db.character.saved",
        Level::INFO,
        character.id = %input.character_id,
        character.version = next_version,
        outbox.subject = subject,
        "saved character {{character.id}} at version {{character.version}} with {{outbox.subject}}",
    );
    Ok(SaveCharacterOutcome {
        has_changed: true,
        version: next_version,
    })
}

/// Checks a character save's fields and child lists before any write.
///
/// # Errors
///
/// Returns an invalid-argument error naming the first field that is out of range or empty.
fn validate_save_character_input(input: &SaveCharacterInput) -> Result<(), Error> {
    if input.name.trim().is_empty() {
        return Err(Error::invalid_argument("character name must not be empty"));
    }
    if input.favorites.is_some_and(|favorites| favorites < 0) {
        return Err(Error::invalid_argument("favorites must be >= 0"));
    }
    if input.merged_record_hash.is_empty() {
        return Err(Error::invalid_argument(
            "merged_record_hash must not be empty",
        ));
    }
    if input.merge_version < 0 {
        return Err(Error::invalid_argument("merge_version must be >= 0"));
    }
    for (field_name, values) in [
        ("name_variations", &input.name_variations),
        ("nicknames", &input.nicknames),
        ("traits", &input.traits),
        ("images", &input.images),
    ] {
        if values.iter().any(|value| value.trim().is_empty()) {
            return Err(Error::invalid_argument(format!(
                "character {field_name} must not hold an empty value"
            )));
        }
    }
    for (field_name, entries) in [
        ("attributes", &input.attributes),
        ("spoilers", &input.spoilers),
    ] {
        if entries
            .iter()
            .any(|(name, value)| name.trim().is_empty() || value.trim().is_empty())
        {
            return Err(Error::invalid_argument(format!(
                "character {field_name} names and values must not be empty"
            )));
        }
    }
    if input.animeography.iter().any(|appearance| {
        appearance.anime_page_provider.trim().is_empty()
            || appearance.anime_page_address.trim().is_empty()
            || appearance.role.trim().is_empty()
    }) {
        return Err(Error::invalid_argument(
            "character_animeography provider, address and role must not be empty",
        ));
    }
    if input.mangaography.iter().any(|entry| {
        entry.source_material_provider.trim().is_empty()
            || entry.source_material_external_id.trim().is_empty()
            || entry.role.trim().is_empty()
    }) {
        return Err(Error::invalid_argument(
            "character_mangaography provider, external ID and role must not be empty",
        ));
    }
    Ok(())
}

/// Adds the trait names a character's save references to the shared `trait` table.
///
/// The names are inserted in sorted order, so two saves adding new traits at once take
/// their row locks in the same order.
///
/// # Errors
///
/// Returns a database error when the insert fails.
async fn insert_traits(tx: &mut Transaction<'_, Postgres>, traits: &[String]) -> Result<(), Error> {
    let trait_names: Vec<String> = traits
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(str::to_owned)
        .collect();
    if trait_names.is_empty() {
        return Ok(());
    }
    sqlx::query!(
        "INSERT INTO trait (name)
         SELECT name FROM UNNEST($1::text[]) WITH ORDINALITY AS entry(name, position)
         ORDER BY position
         ON CONFLICT (name) DO NOTHING",
        &trait_names
    )
    .execute(&mut **tx)
    .await?;
    Ok(())
}

/// Writes the character's own columns, its new version and its merge metadata.
///
/// The first merge also sets `merged_at`.
///
/// # Errors
///
/// Returns a database error when the update fails.
async fn update_character_row(
    tx: &mut Transaction<'_, Postgres>,
    input: &SaveCharacterInput,
    next_version: i64,
    now: DateTime<Utc>,
) -> Result<(), Error> {
    sqlx::query!(
        r#"
        UPDATE character
        SET
            name = $2,
            name_native = $3,
            description = $4,
            favorites = $5,
            version = $6,
            merged_record_hash = $7,
            hash_format = $8,
            merge_version = $9,
            merge_requested_at = NULL,
            merged_at = COALESCE(merged_at, $10),
            updated_at = $10
        WHERE id = $1
        "#,
        input.character_id,
        input.name,
        input.name_native,
        input.description,
        input.favorites,
        next_version,
        input.merged_record_hash,
        input.hash_format,
        input.merge_version,
        now,
    )
    .execute(&mut **tx)
    .await?;
    Ok(())
}

/// Replaces a character's name variations and nicknames.
///
/// # Errors
///
/// Returns a database error when a query fails.
async fn replace_character_names(
    tx: &mut Transaction<'_, Postgres>,
    input: &SaveCharacterInput,
) -> Result<(), Error> {
    let name_variations = sort_unique(&input.name_variations);
    let nicknames = sort_unique(&input.nicknames);
    sqlx::query!(
        "DELETE FROM character_name_variation WHERE character_id = $1",
        input.character_id
    )
    .execute(&mut **tx)
    .await?;
    sqlx::query!(
        "INSERT INTO character_name_variation (character_id, name_variation)
         SELECT $1, name_variation FROM UNNEST($2::text[]) AS entry(name_variation)",
        input.character_id,
        &name_variations
    )
    .execute(&mut **tx)
    .await?;
    sqlx::query!(
        "DELETE FROM character_nickname WHERE character_id = $1",
        input.character_id
    )
    .execute(&mut **tx)
    .await?;
    sqlx::query!(
        "INSERT INTO character_nickname (character_id, nickname)
         SELECT $1, nickname FROM UNNEST($2::text[]) AS entry(nickname)",
        input.character_id,
        &nicknames
    )
    .execute(&mut **tx)
    .await?;
    Ok(())
}

/// Replaces a character's traits and images.
///
/// # Errors
///
/// Returns a database error when a query fails.
async fn replace_character_traits_and_images(
    tx: &mut Transaction<'_, Postgres>,
    input: &SaveCharacterInput,
) -> Result<(), Error> {
    let traits = sort_unique(&input.traits);
    let images = sort_unique(&input.images);
    sqlx::query!(
        "DELETE FROM character_trait WHERE character_id = $1",
        input.character_id
    )
    .execute(&mut **tx)
    .await?;
    sqlx::query!(
        "INSERT INTO character_trait (character_id, trait)
         SELECT $1, trait FROM UNNEST($2::text[]) AS entry(trait)",
        input.character_id,
        &traits
    )
    .execute(&mut **tx)
    .await?;
    sqlx::query!(
        "DELETE FROM character_image WHERE character_id = $1",
        input.character_id
    )
    .execute(&mut **tx)
    .await?;
    sqlx::query!(
        "INSERT INTO character_image (character_id, source)
         SELECT $1, source FROM UNNEST($2::text[]) AS entry(source)",
        input.character_id,
        &images
    )
    .execute(&mut **tx)
    .await?;
    Ok(())
}

/// Replaces a character's attributes and spoilers, the description's spoiler included.
///
/// # Errors
///
/// Returns a database error when a query fails.
async fn replace_character_attributes_and_spoilers(
    tx: &mut Transaction<'_, Postgres>,
    input: &SaveCharacterInput,
) -> Result<(), Error> {
    let (attribute_names, attribute_values): (Vec<String>, Vec<String>) =
        input.attributes.clone().into_iter().unzip();
    let (spoiler_names, spoiler_values): (Vec<String>, Vec<String>) =
        input.spoilers.clone().into_iter().unzip();
    sqlx::query!(
        "DELETE FROM character_attribute WHERE character_id = $1",
        input.character_id
    )
    .execute(&mut **tx)
    .await?;
    sqlx::query!(
        "INSERT INTO character_attribute (character_id, attribute_name, attribute_value)
         SELECT $1, attribute_name, attribute_value
         FROM UNNEST($2::text[], $3::text[]) AS entry(attribute_name, attribute_value)",
        input.character_id,
        &attribute_names,
        &attribute_values
    )
    .execute(&mut **tx)
    .await?;
    sqlx::query!(
        "DELETE FROM character_spoiler WHERE character_id = $1",
        input.character_id
    )
    .execute(&mut **tx)
    .await?;
    sqlx::query!(
        "INSERT INTO character_spoiler (character_id, spoiler_name, spoiler_value)
         SELECT $1, spoiler_name, spoiler_value
         FROM UNNEST($2::text[], $3::text[]) AS entry(spoiler_name, spoiler_value)",
        input.character_id,
        &spoiler_names,
        &spoiler_values
    )
    .execute(&mut **tx)
    .await?;
    Ok(())
}

/// Replaces the works a character appears in, each held by the work's page.
///
/// A page given twice keeps its strongest role, by the `character_role` rank.
///
/// # Errors
///
/// Returns a database error when a query fails, such as a foreign-key error for an
/// unknown role.
async fn replace_character_animeography(
    tx: &mut Transaction<'_, Postgres>,
    character_id: Uuid,
    animeography: &[CharacterAppearanceInput],
) -> Result<(), Error> {
    let providers: Vec<&str> = animeography
        .iter()
        .map(|appearance| appearance.anime_page_provider.as_str())
        .collect();
    let addresses: Vec<&str> = animeography
        .iter()
        .map(|appearance| appearance.anime_page_address.as_str())
        .collect();
    let roles: Vec<&str> = animeography
        .iter()
        .map(|appearance| appearance.role.as_str())
        .collect();

    sqlx::query!(
        "DELETE FROM character_animeography WHERE character_id = $1",
        character_id
    )
    .execute(&mut **tx)
    .await?;
    sqlx::query!(
        "INSERT INTO character_animeography (
             character_id, anime_page_provider, anime_page_address, role
         )
         SELECT DISTINCT ON (entry.provider, entry.address)
                $1, entry.provider, entry.address, entry.role
         FROM UNNEST($2::text[], $3::text[], $4::text[]) AS entry(provider, address, role)
         LEFT JOIN character_role role_order ON role_order.value = entry.role
         ORDER BY entry.provider, entry.address, role_order.rank",
        character_id,
        &providers as &[&str],
        &addresses as &[&str],
        &roles as &[&str]
    )
    .execute(&mut **tx)
    .await?;
    Ok(())
}

/// Replaces the manga and novels a character appears in, each held by its provider identifier.
///
/// The source material is found by joining the identifier to `source_material_xref`, so
/// source material stored after the character still reaches it. An identifier given twice
/// keeps its strongest role, by the `character_role` rank.
///
/// # Errors
///
/// Returns a database error when a query fails, such as a foreign-key error for an
/// unknown role.
async fn replace_character_mangaography(
    tx: &mut Transaction<'_, Postgres>,
    character_id: Uuid,
    mangaography: &[CharacterMangaographyInput],
) -> Result<(), Error> {
    let providers: Vec<&str> = mangaography
        .iter()
        .map(|entry| entry.source_material_provider.as_str())
        .collect();
    let external_ids: Vec<&str> = mangaography
        .iter()
        .map(|entry| entry.source_material_external_id.as_str())
        .collect();
    let roles: Vec<&str> = mangaography
        .iter()
        .map(|entry| entry.role.as_str())
        .collect();

    sqlx::query!(
        "DELETE FROM character_mangaography WHERE character_id = $1",
        character_id
    )
    .execute(&mut **tx)
    .await?;
    sqlx::query!(
        "INSERT INTO character_mangaography (
             character_id, source_material_provider, source_material_external_id, role
         )
         SELECT DISTINCT ON (entry.provider, entry.external_id)
                $1, entry.provider, entry.external_id, entry.role
         FROM UNNEST($2::text[], $3::text[], $4::text[]) AS entry(provider, external_id, role)
         LEFT JOIN character_role role_order ON role_order.value = entry.role
         ORDER BY entry.provider, entry.external_id, role_order.rank",
        character_id,
        &providers as &[&str],
        &external_ids as &[&str],
        &roles as &[&str]
    )
    .execute(&mut **tx)
    .await?;
    Ok(())
}

/// Recomputes a character's `FEATURES` and `APPEARS_IN` edges from the works it reaches.
///
/// A work is reached when one of the character's `character_animeography` pages is that
/// undeleted work's source page. Each work gets one edge each way, however many of its
/// pages the character names.
///
/// # Errors
///
/// Returns a database error when a query fails.
pub(crate) async fn recompute_character_graph_edges(
    tx: &mut Transaction<'_, Postgres>,
    character_id: Uuid,
) -> Result<(), Error> {
    sqlx::query!(
        r#"
        DELETE FROM graph_edge
        WHERE (from_id = $1 AND from_node_type = 'character' AND edge_type = 'APPEARS_IN')
           OR (to_id = $1 AND to_node_type = 'character' AND edge_type = 'FEATURES')
        "#,
        character_id
    )
    .execute(&mut **tx)
    .await?;
    sqlx::query!(
        r#"
        -- reached_anime holds several anime: every anime the character reaches, once each.
        WITH reached_anime AS (
            SELECT DISTINCT anime_page.anime_id
            FROM character_animeography appearance
            JOIN anime_source anime_page
              ON anime_page.provider = appearance.anime_page_provider
             AND anime_page.address = appearance.anime_page_address
            JOIN anime ON anime.id = anime_page.anime_id AND anime.deleted_at IS NULL
            WHERE appearance.character_id = $1
        )
        INSERT INTO graph_edge (from_id, from_node_type, to_id, to_node_type, edge_type)
        SELECT anime_id, 'anime', $1, 'character', 'FEATURES' FROM reached_anime
        UNION ALL
        SELECT $1, 'character', anime_id, 'anime', 'APPEARS_IN' FROM reached_anime
        "#,
        character_id
    )
    .execute(&mut **tx)
    .await?;
    Ok(())
}

/// Updates every character whose appearances name a work page that just changed owner.
///
/// Each such character gets a new version and its graph edges recomputed, in ascending ID
/// order so two page links at once lock the rows in the same order. A merged, undeleted
/// character also gets `character.updated`, since the works it reaches changed. The caller
/// holds the page's lock. The result counts the characters.
///
/// # Errors
///
/// Returns a database error when a query fails.
pub(crate) async fn notify_characters_naming_anime_page(
    tx: &mut Transaction<'_, Postgres>,
    page_key: &PageKey,
) -> Result<u64, Error> {
    // An UPDATE locks rows in no set order, so the rows are locked in ID order first.
    let character_ids: Vec<Uuid> = sqlx::query_scalar!(
        "SELECT id
         FROM character
         WHERE id IN (
             SELECT character_id
             FROM character_animeography
             WHERE anime_page_provider = $1 AND anime_page_address = $2
         )
           AND deleted_at IS NULL
         ORDER BY id
         FOR NO KEY UPDATE",
        page_key.provider().as_str(),
        page_key.address().as_str(),
    )
    .fetch_all(&mut **tx)
    .await?;
    if character_ids.is_empty() {
        return Ok(0);
    }

    let updated_characters = sqlx::query!(
        r#"
        UPDATE character
        SET version = version + 1
        WHERE id = ANY($1)
        RETURNING id, version, merged_at IS NOT NULL AS "is_merged!"
        "#,
        &character_ids,
    )
    .fetch_all(&mut **tx)
    .await?;

    let transaction_time = crate::read_transaction_time(tx).await?;
    for updated_character in &updated_characters {
        recompute_character_graph_edges(tx, updated_character.id).await?;
        if updated_character.is_merged {
            insert_catalogue_outbox_event(
                tx,
                CHARACTER_UPDATED_SUBJECT,
                updated_character.id,
                updated_character.version,
                None,
                transaction_time,
                None,
            )
            .await?;
        }
    }
    Ok(u64::try_from(updated_characters.len()).unwrap_or(u64::MAX))
}

/// Lists the works a character appears in, with the strongest role its pages give in each.
///
/// The result holds several anime, one [`CharacterAnime`] per anime the character reaches.
///
/// A work is listed when one of the character's pages for it is that undeleted work's
/// source page. The works are in role order, then by ID: `MAIN`, then `SUPPORTING`,
/// `BACKGROUND` and `UNKNOWN`.
///
/// # Examples
///
/// ```no_run
/// # use sqlx::PgPool;
/// # use uuid::Uuid;
/// # use postgres_db::{CharacterAnime, list_character_anime};
/// # async fn example(
/// #     pool: &PgPool,
/// #     character_id: Uuid,
/// #     anime_id: Uuid,
/// # ) -> Result<(), postgres_db::Error> {
/// // The character's MAL and AniList pages both name the stored work, as MAIN and SUPPORTING.
/// let works = list_character_anime(pool, character_id).await?;
/// assert_eq!(works, vec![CharacterAnime { anime_id, role: "MAIN".to_owned() }]);
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns a database error when the query fails.
pub async fn list_character_anime(
    pool: &PgPool,
    character_id: Uuid,
) -> Result<Vec<CharacterAnime>, Error> {
    let rows = sqlx::query!(
        r#"
        SELECT
            anime_page.anime_id AS "anime_id!",
            (array_agg(appearance.role ORDER BY role_order.rank))[1] AS "role!"
        FROM character_animeography appearance
        JOIN character_role role_order ON role_order.value = appearance.role
        JOIN anime_source anime_page
          ON anime_page.provider = appearance.anime_page_provider
         AND anime_page.address = appearance.anime_page_address
        JOIN anime ON anime.id = anime_page.anime_id AND anime.deleted_at IS NULL
        WHERE appearance.character_id = $1
        GROUP BY anime_page.anime_id
        ORDER BY min(role_order.rank), anime_page.anime_id
        "#,
        character_id
    )
    .fetch_all(pool)
    .await?;
    Ok(rows
        .into_iter()
        .map(|row| CharacterAnime {
            anime_id: row.anime_id,
            role: row.role,
        })
        .collect())
}

/// Returns `values` sorted with repeats removed, ready for an `UNNEST` insert.
fn sort_unique(values: &[String]) -> Vec<String> {
    values
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(str::to_owned)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sort_unique_returns_sorted_values_once() {
        let values = vec!["b".to_owned(), "a".to_owned(), "b".to_owned()];
        assert_eq!(sort_unique(&values), vec!["a".to_owned(), "b".to_owned()]);
    }
}
