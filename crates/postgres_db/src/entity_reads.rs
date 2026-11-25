//! Read queries for merged, undeleted episodes (`episode_visible`) and characters (`character_visible`).

use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
use sqlx::{PgPool, Row};
use uuid::Uuid;

use crate::{
    EntityType, EpisodeCharacterInput, EpisodeStreamingInput, EpisodeTitleInput, Error, PageKey,
    list_character_anime,
};

/// Visible episode record returned by [`read_visible_episode`] and [`list_visible_episodes`].
#[derive(Debug, Clone, PartialEq)]
pub struct VisibleEpisodeRecord {
    /// Episode UUID.
    pub episode_id: Uuid,
    /// Parent work UUID.
    pub anime_id: Uuid,
    /// Optimistic concurrency version.
    pub version: i64,
    /// Scheduled next refresh timestamp, or `None` when immediately due.
    pub next_refresh_at: Option<DateTime<Utc>>,
    /// Episode number.
    pub episode_number: i32,
    /// Season number, if any.
    pub season_number: Option<i32>,
    /// Primary episode title.
    pub title: String,
    /// Japanese episode title.
    pub title_japanese: Option<String>,
    /// Romaji episode title.
    pub title_romaji: Option<String>,
    /// Episode synopsis.
    pub synopsis: Option<String>,
    /// Episode description.
    pub description: Option<String>,
    /// Air timestamp.
    pub aired_at: Option<DateTime<Utc>>,
    /// Duration in seconds.
    pub duration_seconds: Option<i32>,
    /// Filler flag.
    pub is_filler: bool,
    /// Recap flag.
    pub is_recap: bool,
    /// Episode score.
    pub score: Option<f64>,
    /// Localized titles.
    pub titles: Vec<EpisodeTitleInput>,
    /// Episode images.
    pub images: Vec<String>,
    /// Streaming sources.
    pub streaming: Vec<EpisodeStreamingInput>,
    /// Episode source page keys.
    pub source_pages: Vec<PageKey>,
    /// Episode character appearances.
    pub characters: Vec<EpisodeCharacterInput>,
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
    /// Character source page keys.
    pub source_pages: Vec<PageKey>,
    /// Distinct undeleted anime IDs and roles derived via `character_animeography`.
    pub anime_ids: Vec<(Uuid, String)>,
    /// Stored animeography entries (`provider`, `address`, `role`).
    pub animeography: Vec<(String, String, String)>,
    /// Stored mangaography entries (`provider`, `external_id`, `role`).
    pub mangaography: Vec<(String, String, String)>,
}

/// Reads a single merged, undeleted episode from `episode_visible`.
///
/// # Examples
///
/// ```no_run
/// # use uuid::Uuid;
/// # async fn example(pool: &sqlx::PgPool, episode_id: Uuid) -> Result<(), postgres_db::Error> {
/// let episode = postgres_db::read_visible_episode(pool, episode_id).await?;
/// assert_eq!(episode.episode_id, episode_id);
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns [`Error::not_found`] when the episode does not exist, is deleted,
/// or has not been merged yet, or a database error if querying PostgreSQL fails.
pub async fn read_visible_episode(
    pool: &PgPool,
    episode_id: Uuid,
) -> Result<VisibleEpisodeRecord, Error> {
    let row = sqlx::query(
        "SELECT id, anime_id, version, next_refresh_at, episode_number, season_number,
                coalesce(title, '') AS title, title_japanese, title_romaji,
                synopsis, description, aired_at, duration_seconds, is_filler, is_recap,
                score::float8 AS score
         FROM episode_visible
         WHERE id = $1",
    )
    .bind(episode_id)
    .fetch_optional(pool)
    .await?
    .ok_or_else(Error::not_found)?;

    hydrate_episode_record(pool, row).await
}

/// Lists all merged, undeleted episodes of a work in episode-number order.
///
/// # Examples
///
/// ```no_run
/// # use uuid::Uuid;
/// # async fn example(pool: &sqlx::PgPool, anime_id: Uuid) -> Result<(), postgres_db::Error> {
/// let episodes = postgres_db::list_visible_episodes(pool, anime_id).await?;
/// assert!(episodes.windows(2).all(|pair| pair[0].episode_number <= pair[1].episode_number));
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns a database error if querying PostgreSQL fails.
pub async fn list_visible_episodes(
    pool: &PgPool,
    anime_id: Uuid,
) -> Result<Vec<VisibleEpisodeRecord>, Error> {
    let rows = sqlx::query(
        "SELECT e.id, e.anime_id, e.version, e.next_refresh_at, e.episode_number, e.season_number,
                coalesce(e.title, '') AS title, e.title_japanese, e.title_romaji,
                e.synopsis, e.description, e.aired_at, e.duration_seconds, e.is_filler, e.is_recap,
                e.score::float8 AS score
         FROM episode_visible e
         JOIN anime_visible a ON a.id = e.anime_id
         WHERE e.anime_id = $1
         ORDER BY e.episode_number ASC, e.id ASC",
    )
    .bind(anime_id)
    .fetch_all(pool)
    .await?;

    let mut episodes = Vec::with_capacity(rows.len());
    for row in rows {
        episodes.push(hydrate_episode_record(pool, row).await?);
    }
    Ok(episodes)
}

/// Hydrates child tables for one `episode_visible` row.
async fn hydrate_episode_record(
    pool: &PgPool,
    row: sqlx::postgres::PgRow,
) -> Result<VisibleEpisodeRecord, Error> {
    let episode_id: Uuid = row.try_get("id")?;

    let title_rows: Vec<(String, String)> = sqlx::query_as(
        "SELECT language, title FROM episode_title WHERE episode_id = $1 ORDER BY language",
    )
    .bind(episode_id)
    .fetch_all(pool)
    .await?;
    let titles = title_rows
        .into_iter()
        .map(|(language, title)| EpisodeTitleInput { language, title })
        .collect();

    let images: Vec<String> = sqlx::query_scalar(
        "SELECT source FROM episode_image WHERE episode_id = $1 ORDER BY source",
    )
    .bind(episode_id)
    .fetch_all(pool)
    .await?;

    let streaming_rows: Vec<String> = sqlx::query_scalar(
        "SELECT source FROM episode_streaming WHERE episode_id = $1 ORDER BY source",
    )
    .bind(episode_id)
    .fetch_all(pool)
    .await?;
    let streaming = streaming_rows
        .into_iter()
        .map(|source| EpisodeStreamingInput {
            platform: source.clone(),
            source,
        })
        .collect();

    let source_rows: Vec<(String, String)> = sqlx::query_as(
        "SELECT provider, address FROM episode_source WHERE episode_id = $1 ORDER BY provider, address",
    )
    .bind(episode_id)
    .fetch_all(pool)
    .await?;
    let source_pages = source_rows
        .into_iter()
        .map(|(provider, address)| PageKey::new(provider.parse()?, EntityType::Episode, address))
        .collect::<Result<Vec<_>, _>>()?;

    let char_rows: Vec<(String, String, String)> = sqlx::query_as(
        "SELECT character_page_provider, character_page_address, role
         FROM episode_character
         WHERE episode_id = $1
         ORDER BY character_page_provider, character_page_address",
    )
    .bind(episode_id)
    .fetch_all(pool)
    .await?;
    let characters = char_rows
        .into_iter()
        .map(
            |(character_page_provider, character_page_address, role)| EpisodeCharacterInput {
                character_page_provider,
                character_page_address,
                role,
            },
        )
        .collect();

    Ok(VisibleEpisodeRecord {
        episode_id,
        anime_id: row.try_get("anime_id")?,
        version: row.try_get("version")?,
        next_refresh_at: row.try_get("next_refresh_at")?,
        episode_number: row.try_get("episode_number")?,
        season_number: row.try_get("season_number")?,
        title: row.try_get("title")?,
        title_japanese: row.try_get("title_japanese")?,
        title_romaji: row.try_get("title_romaji")?,
        synopsis: row.try_get("synopsis")?,
        description: row.try_get("description")?,
        aired_at: row.try_get("aired_at")?,
        duration_seconds: row.try_get("duration_seconds")?,
        is_filler: row.try_get("is_filler")?,
        is_recap: row.try_get("is_recap")?,
        score: row.try_get("score")?,
        titles,
        images,
        streaming,
        source_pages,
        characters,
    })
}

/// Reads a single merged, undeleted character from `character_visible`.
///
/// # Examples
///
/// ```no_run
/// # use uuid::Uuid;
/// # async fn example(pool: &sqlx::PgPool, character_id: Uuid) -> Result<(), postgres_db::Error> {
/// let character = postgres_db::read_visible_character(pool, character_id).await?;
/// assert_eq!(character.character_id, character_id);
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
    let row = sqlx::query(
        "SELECT id, version, next_refresh_at, name, name_native, description, favorites
         FROM character_visible
         WHERE id = $1",
    )
    .bind(character_id)
    .fetch_optional(pool)
    .await?
    .ok_or_else(Error::not_found)?;

    let (name_variations, nicknames, traits, images, attributes, spoilers) =
        read_character_attributes(pool, character_id).await?;

    let source_rows: Vec<(String, String)> = sqlx::query_as(
        "SELECT provider, address FROM character_source WHERE character_id = $1 ORDER BY provider, address",
    )
    .bind(character_id)
    .fetch_all(pool)
    .await?;
    let source_pages = source_rows
        .into_iter()
        .map(|(provider, address)| PageKey::new(provider.parse()?, EntityType::Character, address))
        .collect::<Result<Vec<_>, _>>()?;

    let anime_links = list_character_anime(pool, character_id).await?;
    let anime_ids = anime_links
        .into_iter()
        .map(|entry| (entry.anime_id, entry.role))
        .collect();

    let animeography: Vec<(String, String, String)> = sqlx::query_as(
        "SELECT anime_page_provider, anime_page_address, role
         FROM character_animeography WHERE character_id = $1
         ORDER BY anime_page_provider, anime_page_address",
    )
    .bind(character_id)
    .fetch_all(pool)
    .await?;

    let mangaography: Vec<(String, String, String)> = sqlx::query_as(
        "SELECT source_material_provider, source_material_external_id, role
         FROM character_mangaography WHERE character_id = $1
         ORDER BY source_material_provider, source_material_external_id",
    )
    .bind(character_id)
    .fetch_all(pool)
    .await?;

    Ok(VisibleCharacterRecord {
        character_id: row.try_get("id")?,
        version: row.try_get("version")?,
        next_refresh_at: row.try_get("next_refresh_at")?,
        name: row.try_get("name")?,
        name_native: row.try_get("name_native")?,
        description: row.try_get("description")?,
        favorites: row.try_get("favorites")?,
        name_variations,
        nicknames,
        traits,
        images,
        attributes,
        spoilers,
        source_pages,
        anime_ids,
        animeography,
        mangaography,
    })
}

/// Reads scalar lists and key-value attribute/spoiler maps for a character.
async fn read_character_attributes(
    pool: &PgPool,
    character_id: Uuid,
) -> Result<
    (
        Vec<String>,
        Vec<String>,
        Vec<String>,
        Vec<String>,
        BTreeMap<String, String>,
        BTreeMap<String, String>,
    ),
    Error,
> {
    let name_variations: Vec<String> = sqlx::query_scalar(
        "SELECT name_variation FROM character_name_variation WHERE character_id = $1 ORDER BY name_variation",
    )
    .bind(character_id)
    .fetch_all(pool)
    .await?;
    let nicknames: Vec<String> = sqlx::query_scalar(
        "SELECT nickname FROM character_nickname WHERE character_id = $1 ORDER BY nickname",
    )
    .bind(character_id)
    .fetch_all(pool)
    .await?;
    let traits: Vec<String> = sqlx::query_scalar(
        "SELECT trait FROM character_trait WHERE character_id = $1 ORDER BY trait",
    )
    .bind(character_id)
    .fetch_all(pool)
    .await?;
    let images: Vec<String> = sqlx::query_scalar(
        "SELECT source FROM character_image WHERE character_id = $1 ORDER BY source",
    )
    .bind(character_id)
    .fetch_all(pool)
    .await?;
    let attr_rows: Vec<(String, String)> = sqlx::query_as(
        "SELECT attribute_name, attribute_value
         FROM character_attribute WHERE character_id = $1 ORDER BY attribute_name",
    )
    .bind(character_id)
    .fetch_all(pool)
    .await?;
    let spoiler_rows: Vec<(String, String)> = sqlx::query_as(
        "SELECT spoiler_name, spoiler_value
         FROM character_spoiler WHERE character_id = $1 ORDER BY spoiler_name",
    )
    .bind(character_id)
    .fetch_all(pool)
    .await?;

    Ok((
        name_variations,
        nicknames,
        traits,
        images,
        attr_rows.into_iter().collect(),
        spoiler_rows.into_iter().collect(),
    ))
}
