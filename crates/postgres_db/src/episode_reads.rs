//! Read queries for merged, undeleted episodes (`episode_visible`) and their child tables.

use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
use sqlx::{PgPool, Postgres, Transaction};
use uuid::Uuid;

use crate::{EpisodeStreamingInput, EpisodeTitleInput, Error, begin_read_snapshot};

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
    /// Primary episode title, empty when no provider gave one.
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
    /// Streaming links with their platforms.
    pub streaming: Vec<EpisodeStreamingInput>,
    /// URLs of the episode's source pages, where a merge chose one.
    pub sources: Vec<String>,
    /// Characters appearing in the episode.
    pub characters: Vec<VisibleEpisodeCharacter>,
}

/// A character appearing in an episode, found through the character pages the episode names.
///
/// The pages of one visible character form one entry. A page no visible character holds stays
/// an entry of its own.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VisibleEpisodeCharacter {
    /// Name of the visible character holding the pages, or otherwise the name the character's
    /// page gives, when one is stored.
    pub name: Option<String>,
    /// The most prominent character role lookup text among the grouped pages.
    pub role: String,
    /// URLs of the character pages, where a merge of each page's owner chose one.
    pub sources: Vec<String>,
}

/// One `episode_visible` row, before its child rows are attached.
struct EpisodeRow {
    id: Uuid,
    anime_id: Uuid,
    version: i64,
    next_refresh_at: Option<DateTime<Utc>>,
    episode_number: i32,
    season_number: Option<i32>,
    title: String,
    title_japanese: Option<String>,
    title_romaji: Option<String>,
    synopsis: Option<String>,
    description: Option<String>,
    aired_at: Option<DateTime<Utc>>,
    duration_seconds: Option<i32>,
    is_filler: bool,
    is_recap: bool,
    score: Option<f64>,
}

/// Child rows of the episodes one read returns, keyed by episode ID.
#[derive(Default)]
struct EpisodeDetails {
    titles: BTreeMap<Uuid, Vec<EpisodeTitleInput>>,
    images: BTreeMap<Uuid, Vec<String>>,
    streaming: BTreeMap<Uuid, Vec<EpisodeStreamingInput>>,
    sources: BTreeMap<Uuid, Vec<String>>,
    characters: BTreeMap<Uuid, Vec<VisibleEpisodeCharacter>>,
}

/// Reads one merged, undeleted episode from `episode_visible`.
///
/// Every query runs in one read-only snapshot.
///
/// # Examples
///
/// ```no_run
/// # use uuid::Uuid;
/// # async fn example(pool: &sqlx::PgPool, episode_id: Uuid) -> Result<(), postgres_db::Error> {
/// let episode = postgres_db::read_visible_episode(pool, episode_id).await?;
/// assert_eq!(episode.episode_id, episode_id);
/// assert!(episode.version >= 1);
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
    let mut tx = begin_read_snapshot(pool).await?;
    let episode_row = sqlx::query_as!(
        EpisodeRow,
        r#"
        SELECT id AS "id!", anime_id AS "anime_id!", version AS "version!", next_refresh_at,
               episode_number AS "episode_number!", season_number,
               coalesce(title, '') AS "title!", title_japanese, title_romaji,
               synopsis, description, aired_at, duration_seconds,
               is_filler AS "is_filler!", is_recap AS "is_recap!", score::float8 AS score
        FROM episode_visible
        WHERE id = $1
        "#,
        episode_id
    )
    .fetch_optional(&mut *tx)
    .await?
    .ok_or_else(Error::not_found)?;
    let mut details = read_episode_details(&mut tx, &[episode_row.id]).await?;
    tx.commit().await?;
    Ok(build_episode_record(episode_row, &mut details))
}

/// Lists every merged, undeleted episode of a visible work in episode-number order.
///
/// The child rows of all episodes are read with one query per table, in one snapshot.
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
    let mut tx = begin_read_snapshot(pool).await?;
    let episode_rows = sqlx::query_as!(
        EpisodeRow,
        r#"
        SELECT episode.id AS "id!", episode.anime_id AS "anime_id!",
               episode.version AS "version!", episode.next_refresh_at,
               episode.episode_number AS "episode_number!", episode.season_number,
               coalesce(episode.title, '') AS "title!", episode.title_japanese,
               episode.title_romaji, episode.synopsis, episode.description, episode.aired_at,
               episode.duration_seconds, episode.is_filler AS "is_filler!",
               episode.is_recap AS "is_recap!", episode.score::float8 AS score
        FROM episode_visible episode
        JOIN anime_visible anime ON anime.id = episode.anime_id
        WHERE episode.anime_id = $1
        ORDER BY episode.episode_number, episode.id
        "#,
        anime_id
    )
    .fetch_all(&mut *tx)
    .await?;
    let episode_ids: Vec<Uuid> = episode_rows
        .iter()
        .map(|episode_row| episode_row.id)
        .collect();
    let mut details = read_episode_details(&mut tx, &episode_ids).await?;
    tx.commit().await?;
    Ok(episode_rows
        .into_iter()
        .map(|episode_row| build_episode_record(episode_row, &mut details))
        .collect())
}

/// Attaches an episode's child rows to its row.
fn build_episode_record(
    episode_row: EpisodeRow,
    details: &mut EpisodeDetails,
) -> VisibleEpisodeRecord {
    let episode_id = episode_row.id;
    VisibleEpisodeRecord {
        episode_id,
        anime_id: episode_row.anime_id,
        version: episode_row.version,
        next_refresh_at: episode_row.next_refresh_at,
        episode_number: episode_row.episode_number,
        season_number: episode_row.season_number,
        title: episode_row.title,
        title_japanese: episode_row.title_japanese,
        title_romaji: episode_row.title_romaji,
        synopsis: episode_row.synopsis,
        description: episode_row.description,
        aired_at: episode_row.aired_at,
        duration_seconds: episode_row.duration_seconds,
        is_filler: episode_row.is_filler,
        is_recap: episode_row.is_recap,
        score: episode_row.score,
        titles: details.titles.remove(&episode_id).unwrap_or_default(),
        images: details.images.remove(&episode_id).unwrap_or_default(),
        streaming: details.streaming.remove(&episode_id).unwrap_or_default(),
        sources: details.sources.remove(&episode_id).unwrap_or_default(),
        characters: details.characters.remove(&episode_id).unwrap_or_default(),
    }
}

/// Reads the child rows of the given episodes, one query per table.
async fn read_episode_details(
    tx: &mut Transaction<'_, Postgres>,
    episode_ids: &[Uuid],
) -> Result<EpisodeDetails, Error> {
    let mut details = EpisodeDetails::default();
    if episode_ids.is_empty() {
        return Ok(details);
    }
    read_episode_titles_and_images(tx, episode_ids, &mut details).await?;
    read_episode_streaming(tx, episode_ids, &mut details).await?;
    read_episode_page_sources(tx, episode_ids, &mut details).await?;
    read_episode_characters(tx, episode_ids, &mut details).await?;
    Ok(details)
}

/// Reads the localized titles and images of the given episodes.
async fn read_episode_titles_and_images(
    tx: &mut Transaction<'_, Postgres>,
    episode_ids: &[Uuid],
    details: &mut EpisodeDetails,
) -> Result<(), Error> {
    let title_rows = sqlx::query!(
        "SELECT episode_id, language, title FROM episode_title
         WHERE episode_id = ANY($1) ORDER BY episode_id, language",
        episode_ids
    )
    .fetch_all(&mut **tx)
    .await?;
    for title_row in title_rows {
        details
            .titles
            .entry(title_row.episode_id)
            .or_default()
            .push(EpisodeTitleInput {
                language: title_row.language,
                title: title_row.title,
            });
    }

    let image_rows = sqlx::query!(
        "SELECT episode_id, source FROM episode_image
         WHERE episode_id = ANY($1) ORDER BY episode_id, source",
        episode_ids
    )
    .fetch_all(&mut **tx)
    .await?;
    for image_row in image_rows {
        details
            .images
            .entry(image_row.episode_id)
            .or_default()
            .push(image_row.source);
    }

    Ok(())
}

/// Reads the streaming links of the given episodes, each with its platform.
async fn read_episode_streaming(
    tx: &mut Transaction<'_, Postgres>,
    episode_ids: &[Uuid],
    details: &mut EpisodeDetails,
) -> Result<(), Error> {
    let streaming_rows = sqlx::query!(
        r#"
        SELECT streaming.episode_id, external_source.platform, streaming.source
        FROM episode_streaming streaming
        JOIN external_source ON external_source.source = streaming.source
        WHERE streaming.episode_id = ANY($1)
        ORDER BY streaming.episode_id, streaming.source
        "#,
        episode_ids
    )
    .fetch_all(&mut **tx)
    .await?;
    for streaming_row in streaming_rows {
        details
            .streaming
            .entry(streaming_row.episode_id)
            .or_default()
            .push(EpisodeStreamingInput {
                platform: streaming_row.platform,
                source: streaming_row.source,
            });
    }

    Ok(())
}

/// Reads the URLs of the given episodes' live source pages, where a merge chose one.
async fn read_episode_page_sources(
    tx: &mut Transaction<'_, Postgres>,
    episode_ids: &[Uuid],
    details: &mut EpisodeDetails,
) -> Result<(), Error> {
    let source_rows = sqlx::query!(
        r#"
        SELECT link.episode_id, page.source AS "source!"
        FROM episode_source link
        JOIN provider_page page USING (provider, entity_type, address)
        WHERE link.episode_id = ANY($1) AND NOT page.is_dead AND page.source IS NOT NULL
        ORDER BY link.episode_id, page.source
        "#,
        episode_ids
    )
    .fetch_all(&mut **tx)
    .await?;
    for source_row in source_rows {
        details
            .sources
            .entry(source_row.episode_id)
            .or_default()
            .push(source_row.source);
    }

    Ok(())
}

/// Reads the characters of the given episodes, each with the visible character holding its page.
///
/// A character page no visible character holds is a character of its own, named by its page's
/// title.
async fn read_episode_characters(
    tx: &mut Transaction<'_, Postgres>,
    episode_ids: &[Uuid],
    details: &mut EpisodeDetails,
) -> Result<(), Error> {
    let character_rows = sqlx::query!(
        r#"
        SELECT appearance.episode_id AS "episode_id!",
               coalesce(
                   character.name,
                   (array_agg(page.title ORDER BY appearance.character_page_provider,
                                                  appearance.character_page_address)
                        FILTER (WHERE page.title IS NOT NULL))[1]
               ) AS "name?",
               (array_agg(appearance.role ORDER BY role_order.rank))[1] AS "role!",
               coalesce(
                   array_agg(page.source ORDER BY page.source) FILTER (WHERE page.source IS NOT NULL),
                   '{}'
               ) AS "sources!"
        FROM episode_character appearance
        JOIN character_role role_order ON role_order.value = appearance.role
        LEFT JOIN provider_page page
          ON page.provider = appearance.character_page_provider
         AND page.entity_type = 'character'
         AND page.address = appearance.character_page_address
        LEFT JOIN character_source link
          ON link.provider = appearance.character_page_provider
         AND link.entity_type = 'character'
         AND link.address = appearance.character_page_address
        LEFT JOIN character_visible character ON character.id = link.character_id
        WHERE appearance.episode_id = ANY($1)
        GROUP BY appearance.episode_id, character.id, character.name,
                 CASE WHEN character.id IS NULL
                      THEN appearance.character_page_provider || ':'
                           || appearance.character_page_address END
        ORDER BY appearance.episode_id, min(role_order.rank), character.name NULLS LAST,
                 min(appearance.character_page_provider || ':' || appearance.character_page_address)
        "#,
        episode_ids
    )
    .fetch_all(&mut **tx)
    .await?;
    for character_row in character_rows {
        details
            .characters
            .entry(character_row.episode_id)
            .or_default()
            .push(VisibleEpisodeCharacter {
                name: character_row.name,
                role: character_row.role,
                sources: character_row.sources,
            });
    }
    Ok(())
}
