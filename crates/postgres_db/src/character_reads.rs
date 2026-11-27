//! Read queries for merged, undeleted characters (`character_visible`) and their child tables.

use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
use sqlx::{PgPool, Postgres, Transaction};
use uuid::Uuid;

use crate::{CharacterAnime, Error, begin_read_snapshot};

/// One work, manga or novel in a character's animeography or mangaography.
///
/// The entries that name pages of one visible work, or identifiers of one stored source
/// material, form one entry. An entry no stored work or source material matches stays on its own.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VisibleOgraphyEntry {
    /// Title of the visible work or the stored source material, when one is found.
    pub title: Option<String>,
    /// The character's most prominent role lookup text among the grouped entries.
    pub role: String,
    /// URLs of the pages the grouped entries name, where stored.
    pub sources: Vec<String>,
}

/// Visible character record returned by [`read_visible_character`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VisibleCharacterRecord {
    /// Character UUID.
    pub character_id: Uuid,
    /// Optimistic concurrency version.
    pub version: i64,
    /// Scheduled next refresh timestamp, or `None` when immediately due.
    pub next_refresh_at: Option<DateTime<Utc>>,
    /// Primary character name.
    pub name: String,
    /// Native name.
    pub name_native: Option<String>,
    /// Character description.
    pub description: Option<String>,
    /// Favorites count.
    pub favorites: Option<i32>,
    /// Name variations.
    pub name_variations: Vec<String>,
    /// Nicknames.
    pub nicknames: Vec<String>,
    /// Traits.
    pub traits: Vec<String>,
    /// Images.
    pub images: Vec<String>,
    /// Typed attributes map.
    pub attributes: BTreeMap<String, String>,
    /// Spoiler attributes map.
    pub spoilers: BTreeMap<String, String>,
    /// URLs of the character's source pages, where a merge chose one.
    pub sources: Vec<String>,
    /// Visible works the character appears in, each with the character's most prominent role.
    pub anime: Vec<CharacterAnime>,
    /// Stored animeography entries.
    pub animeography: Vec<VisibleOgraphyEntry>,
    /// Stored mangaography entries.
    pub mangaography: Vec<VisibleOgraphyEntry>,
}

/// Reads one merged, undeleted character from `character_visible`.
///
/// Every query runs in one read-only snapshot. The works the character appears in are the
/// visible works holding the pages its animeography names.
///
/// # Examples
///
/// ```no_run
/// # use uuid::Uuid;
/// # async fn example(pool: &sqlx::PgPool, character_id: Uuid) -> Result<(), postgres_db::Error> {
/// let character = postgres_db::read_visible_character(pool, character_id).await?;
/// assert_eq!(character.character_id, character_id);
/// assert!(character.version >= 1);
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns [`Error::not_found`] when the character does not exist, is deleted,
/// or has not been merged yet, or a database error if querying PostgreSQL fails.
pub async fn read_visible_character(
    pool: &PgPool,
    character_id: Uuid,
) -> Result<VisibleCharacterRecord, Error> {
    let mut tx = begin_read_snapshot(pool).await?;
    let character_row = sqlx::query!(
        r#"
        SELECT id AS "id!", version AS "version!", next_refresh_at, name AS "name!",
               name_native, description, favorites
        FROM character_visible
        WHERE id = $1
        "#,
        character_id
    )
    .fetch_optional(&mut *tx)
    .await?
    .ok_or_else(Error::not_found)?;
    let profile = read_character_profile(&mut tx, character_id).await?;
    let sources = sqlx::query_scalar!(
        r#"
        SELECT page.source AS "source!"
        FROM character_source link
        JOIN provider_page page USING (provider, entity_type, address)
        WHERE link.character_id = $1 AND page.source IS NOT NULL
        ORDER BY page.source
        "#,
        character_id
    )
    .fetch_all(&mut *tx)
    .await?;
    let anime = read_visible_character_anime(&mut tx, character_id).await?;
    let animeography = read_character_animeography(&mut tx, character_id).await?;
    let mangaography = read_character_mangaography(&mut tx, character_id).await?;
    tx.commit().await?;

    Ok(VisibleCharacterRecord {
        character_id: character_row.id,
        version: character_row.version,
        next_refresh_at: character_row.next_refresh_at,
        name: character_row.name,
        name_native: character_row.name_native,
        description: character_row.description,
        favorites: character_row.favorites,
        name_variations: profile.name_variations,
        nicknames: profile.nicknames,
        traits: profile.traits,
        images: profile.images,
        attributes: profile.attributes,
        spoilers: profile.spoilers,
        sources,
        anime,
        animeography,
        mangaography,
    })
}

/// Names, traits, images, attributes and spoilers of one character.
struct CharacterProfile {
    name_variations: Vec<String>,
    nicknames: Vec<String>,
    traits: Vec<String>,
    images: Vec<String>,
    attributes: BTreeMap<String, String>,
    spoilers: BTreeMap<String, String>,
}

/// Reads a character's names, traits, images, attributes and spoilers.
async fn read_character_profile(
    tx: &mut Transaction<'_, Postgres>,
    character_id: Uuid,
) -> Result<CharacterProfile, Error> {
    let name_variations = sqlx::query_scalar!(
        "SELECT name_variation FROM character_name_variation
         WHERE character_id = $1 ORDER BY name_variation",
        character_id
    )
    .fetch_all(&mut **tx)
    .await?;
    let nicknames = sqlx::query_scalar!(
        "SELECT nickname FROM character_nickname WHERE character_id = $1 ORDER BY nickname",
        character_id
    )
    .fetch_all(&mut **tx)
    .await?;
    let traits = sqlx::query_scalar!(
        "SELECT trait FROM character_trait WHERE character_id = $1 ORDER BY trait",
        character_id
    )
    .fetch_all(&mut **tx)
    .await?;
    let images = sqlx::query_scalar!(
        "SELECT source FROM character_image WHERE character_id = $1 ORDER BY source",
        character_id
    )
    .fetch_all(&mut **tx)
    .await?;
    let attributes = sqlx::query!(
        "SELECT attribute_name, attribute_value FROM character_attribute
         WHERE character_id = $1 ORDER BY attribute_name",
        character_id
    )
    .fetch_all(&mut **tx)
    .await?
    .into_iter()
    .map(|attribute| (attribute.attribute_name, attribute.attribute_value))
    .collect();
    let spoilers = sqlx::query!(
        "SELECT spoiler_name, spoiler_value FROM character_spoiler
         WHERE character_id = $1 ORDER BY spoiler_name",
        character_id
    )
    .fetch_all(&mut **tx)
    .await?
    .into_iter()
    .map(|spoiler| (spoiler.spoiler_name, spoiler.spoiler_value))
    .collect();
    Ok(CharacterProfile {
        name_variations,
        nicknames,
        traits,
        images,
        attributes,
        spoilers,
    })
}

/// Reads the visible works a character appears in, each with its most prominent role.
async fn read_visible_character_anime(
    tx: &mut Transaction<'_, Postgres>,
    character_id: Uuid,
) -> Result<Vec<CharacterAnime>, Error> {
    let anime = sqlx::query_as!(
        CharacterAnime,
        r#"
        SELECT anime_page.anime_id AS "anime_id!",
               (array_agg(appearance.role ORDER BY role_order.rank))[1] AS "role!"
        FROM character_animeography appearance
        JOIN character_role role_order ON role_order.value = appearance.role
        JOIN anime_source anime_page
          ON anime_page.provider = appearance.anime_page_provider
         AND anime_page.entity_type = 'anime'
         AND anime_page.address = appearance.anime_page_address
        JOIN anime_visible anime ON anime.id = anime_page.anime_id
        WHERE appearance.character_id = $1
        GROUP BY anime_page.anime_id
        ORDER BY min(role_order.rank), anime_page.anime_id
        "#,
        character_id
    )
    .fetch_all(&mut **tx)
    .await?;
    Ok(anime)
}

/// Reads a character's animeography, one entry per visible work, with its title and page URLs.
async fn read_character_animeography(
    tx: &mut Transaction<'_, Postgres>,
    character_id: Uuid,
) -> Result<Vec<VisibleOgraphyEntry>, Error> {
    let animeography = sqlx::query_as!(
        VisibleOgraphyEntry,
        r#"
        SELECT anime.title AS "title?",
               (array_agg(appearance.role ORDER BY role_order.rank))[1] AS "role!",
               coalesce(
                   array_agg(page.source ORDER BY page.source) FILTER (WHERE page.source IS NOT NULL),
                   '{}'
               ) AS "sources!"
        FROM character_animeography appearance
        JOIN character_role role_order ON role_order.value = appearance.role
        LEFT JOIN provider_page page
          ON page.provider = appearance.anime_page_provider
         AND page.entity_type = 'anime'
         AND page.address = appearance.anime_page_address
        LEFT JOIN anime_source anime_page
          ON anime_page.provider = appearance.anime_page_provider
         AND anime_page.entity_type = 'anime'
         AND anime_page.address = appearance.anime_page_address
        LEFT JOIN anime_visible anime ON anime.id = anime_page.anime_id
        WHERE appearance.character_id = $1
        GROUP BY anime.id, anime.title,
                 CASE WHEN anime.id IS NULL
                      THEN appearance.anime_page_provider || ':' || appearance.anime_page_address END
        ORDER BY min(role_order.rank), anime.title NULLS LAST,
                 min(appearance.anime_page_provider || ':' || appearance.anime_page_address)
        "#,
        character_id
    )
    .fetch_all(&mut **tx)
    .await?;
    Ok(animeography)
}

/// Reads a character's mangaography, one entry per stored source material, with its title and
/// the URLs of the identifiers the character names.
async fn read_character_mangaography(
    tx: &mut Transaction<'_, Postgres>,
    character_id: Uuid,
) -> Result<Vec<VisibleOgraphyEntry>, Error> {
    let mangaography = sqlx::query_as!(
        VisibleOgraphyEntry,
        r#"
        SELECT material.title AS "title?",
               (array_agg(appearance.role ORDER BY role_order.rank))[1] AS "role!",
               coalesce(
                   array_agg(xref.source ORDER BY xref.source) FILTER (WHERE xref.source IS NOT NULL),
                   '{}'
               ) AS "sources!"
        FROM character_mangaography appearance
        JOIN character_role role_order ON role_order.value = appearance.role
        LEFT JOIN source_material_xref xref
          ON xref.provider = appearance.source_material_provider
         AND xref.external_id = appearance.source_material_external_id
        LEFT JOIN source_material material ON material.id = xref.source_material_id
        WHERE appearance.character_id = $1
        GROUP BY material.id, material.title,
                 CASE WHEN material.id IS NULL
                      THEN appearance.source_material_provider || ':'
                           || appearance.source_material_external_id END
        ORDER BY min(role_order.rank), material.title NULLS LAST,
                 min(appearance.source_material_provider || ':'
                     || appearance.source_material_external_id)
        "#,
        character_id
    )
    .fetch_all(&mut **tx)
    .await?;
    Ok(mangaography)
}
