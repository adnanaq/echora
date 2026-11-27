//! Read queries for merged, undeleted works (`anime_visible`) and their child tables.

use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
use sqlx::{PgPool, Postgres, Transaction};
use uuid::Uuid;

use crate::{
    AnimeBroadcastInput, AnimeCompanyInput, AnimeContextualRankInput, AnimeExternalSourceInput,
    AnimeHiatusInput, AnimeImageInput, AnimeSourceMaterialInput, AnimeStatisticsInput,
    AnimeStreamingSourceInput, AnimeThemeInput, AnimeThemeSongInput, AnimeTitleInput,
    AnimeTrailerInput, CompanyXrefInput, EpisodeRangeInput, Error, SourceMaterialXrefInput,
    begin_read_snapshot,
};

/// A related work that a work's own pages list, with its target page URLs.
///
/// The relation rows that name pages of one visible work form one relation. A target page that
/// no visible work holds forms a relation of its own.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VisibleAnimeRelation {
    /// Relation type (`SEQUEL`, `PREQUEL`, and so on).
    pub relation_type: String,
    /// URLs of the target pages, where a merge of each page's owner chose one.
    pub target_page_sources: Vec<String>,
    /// The title a target page's provider gives, first in page key order, when one is stored.
    ///
    /// A relation whose target no visible work holds shows this title.
    pub target_page_title: Option<String>,
    /// The visible work that holds the target pages as sources, other than the work itself.
    pub target_anime: Option<RelatedAnimeSummary>,
}

/// The fields of a related work that a relation shows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelatedAnimeSummary {
    /// Work UUID.
    pub anime_id: Uuid,
    /// Primary title.
    pub title: String,
    /// Work format type lookup text (`TV`, `MOVIE`, and so on).
    pub anime_type: String,
    /// Airing status lookup text.
    pub status: String,
    /// Release year.
    pub year: Option<i16>,
    /// Total episode count.
    pub episode_count: i32,
}

/// Visible work record and all child details returned by [`read_visible_anime`].
#[derive(Debug, Clone, PartialEq)]
pub struct VisibleAnimeRecord {
    /// Work UUID.
    pub anime_id: Uuid,
    /// Optimistic concurrency version.
    pub version: i64,
    /// Scheduled next refresh timestamp, or `None` when immediately due.
    pub next_refresh_at: Option<DateTime<Utc>>,
    /// Primary title.
    pub title: String,
    /// English title.
    pub title_english: Option<String>,
    /// Japanese title.
    pub title_japanese: Option<String>,
    /// Work format type lookup text (`TV`, `TV SPECIAL`, and so on).
    pub anime_type: String,
    /// Airing status lookup text.
    pub status: String,
    /// Content rating lookup text.
    pub rating: String,
    /// Release season lookup text.
    pub season: Option<String>,
    /// Original source material lookup text.
    pub source_material: Option<String>,
    /// Release year.
    pub year: Option<i16>,
    /// Release month (`1..=12`).
    pub month: Option<i16>,
    /// Total episode count.
    pub episode_count: i32,
    /// Episode duration in seconds.
    pub duration_seconds: Option<i32>,
    /// NSFW flag.
    pub is_nsfw: Option<bool>,
    /// ISO 3166-1 alpha-2 country code.
    pub country_of_origin: Option<String>,
    /// Synopsis.
    pub synopsis: Option<String>,
    /// Editorial background text.
    pub background: Option<String>,
    /// First aired timestamp.
    pub aired_from: Option<DateTime<Utc>>,
    /// Last aired timestamp.
    pub aired_to: Option<DateTime<Utc>>,
    /// Weighted aggregate score.
    pub score_weighted: Option<f64>,
    /// Mean aggregate score.
    pub score_mean: Option<f64>,
    /// Median aggregate score.
    pub score_median: Option<f64>,
    /// Localized titles by language code.
    pub titles: Vec<AnimeTitleInput>,
    /// Alternate titles.
    pub synonyms: Vec<String>,
    /// Genres.
    pub genres: Vec<String>,
    /// Themes.
    pub themes: Vec<AnimeThemeInput>,
    /// Demographics.
    pub demographics: Vec<String>,
    /// Content warnings.
    pub content_warnings: Vec<String>,
    /// Descriptive tags.
    pub tags: Vec<String>,
    /// URLs of the pages merged into this work, where a merge chose one.
    pub sources: Vec<String>,
    /// URLs of the partial-source pages, where the merge of each page's owner chose one.
    pub partial_sources: Vec<String>,
    /// Cover, poster and banner image links.
    pub images: Vec<AnimeImageInput>,
    /// Opening theme songs.
    pub opening_themes: Vec<AnimeThemeSongInput>,
    /// Ending theme songs.
    pub ending_themes: Vec<AnimeThemeSongInput>,
    /// Trailers.
    pub trailers: Vec<AnimeTrailerInput>,
    /// External links.
    pub external_sources: Vec<AnimeExternalSourceInput>,
    /// Streaming sources.
    pub streaming_sources: Vec<AnimeStreamingSourceInput>,
    /// Per-provider statistics with their contextual ranks.
    pub statistics: Vec<AnimeStatisticsInput>,
    /// Broadcast schedule, if present.
    pub broadcast: Option<AnimeBroadcastInput>,
    /// Latest hiatus record, if present.
    pub hiatus: Option<AnimeHiatusInput>,
    /// Related works.
    pub related_anime: Vec<VisibleAnimeRelation>,
    /// Related source material, with its cross-references and images.
    pub related_source_material: Vec<AnimeSourceMaterialInput>,
    /// Companies with their roles and cross-references.
    pub companies: Vec<AnimeCompanyInput>,
}

/// Reads one merged, undeleted work and all of its detail rows.
///
/// Every query runs in one read-only snapshot, so the work's row and its detail rows always
/// come from the same committed save.
///
/// # Examples
///
/// ```no_run
/// # use uuid::Uuid;
/// # async fn example(pool: &sqlx::PgPool, anime_id: Uuid) -> Result<(), postgres_db::Error> {
/// let work = postgres_db::read_visible_anime(pool, anime_id).await?;
/// assert_eq!(work.anime_id, anime_id);
/// assert!(work.version >= 1);
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns [`Error::not_found`] when the work does not exist, is deleted, or has
/// not been merged yet, or a database error if querying PostgreSQL fails.
pub async fn read_visible_anime(
    pool: &PgPool,
    anime_id: Uuid,
) -> Result<VisibleAnimeRecord, Error> {
    let mut tx = begin_read_snapshot(pool).await?;
    let work = sqlx::query!(
        r#"
        SELECT id AS "id!", version AS "version!", next_refresh_at,
               title AS "title!", title_english, title_japanese,
               type AS "anime_type!", status AS "status!", rating AS "rating!", season,
               source_material, year, month, episode_count AS "episode_count!",
               duration_seconds, is_nsfw, country_of_origin, synopsis, background,
               aired_from, aired_to,
               score_weighted::float8 AS score_weighted,
               score_mean::float8 AS score_mean,
               score_median::float8 AS score_median
        FROM anime_visible
        WHERE id = $1
        "#,
        anime_id
    )
    .fetch_optional(&mut *tx)
    .await?
    .ok_or_else(Error::not_found)?;

    let classification = read_anime_classification(&mut tx, anime_id).await?;
    let (sources, partial_sources) = read_anime_page_sources(&mut tx, anime_id).await?;
    let opening_themes = read_opening_themes(&mut tx, anime_id).await?;
    let ending_themes = read_ending_themes(&mut tx, anime_id).await?;
    let links = read_anime_links(&mut tx, anime_id).await?;
    let statistics = read_anime_statistics(&mut tx, anime_id).await?;
    let (broadcast, hiatus) = read_anime_broadcast_and_hiatus(&mut tx, anime_id).await?;
    let related_anime = read_related_anime(&mut tx, anime_id).await?;
    let related_source_material = read_related_source_material(&mut tx, anime_id).await?;
    let companies = read_anime_companies(&mut tx, anime_id).await?;
    tx.commit().await?;

    Ok(VisibleAnimeRecord {
        anime_id: work.id,
        version: work.version,
        next_refresh_at: work.next_refresh_at,
        title: work.title,
        title_english: work.title_english,
        title_japanese: work.title_japanese,
        anime_type: work.anime_type,
        status: work.status,
        rating: work.rating,
        season: work.season,
        source_material: work.source_material,
        year: work.year,
        month: work.month,
        episode_count: work.episode_count,
        duration_seconds: work.duration_seconds,
        is_nsfw: work.is_nsfw,
        country_of_origin: work.country_of_origin,
        synopsis: work.synopsis,
        background: work.background,
        aired_from: work.aired_from,
        aired_to: work.aired_to,
        score_weighted: work.score_weighted,
        score_mean: work.score_mean,
        score_median: work.score_median,
        titles: classification.titles,
        synonyms: classification.synonyms,
        genres: classification.genres,
        themes: classification.themes,
        demographics: classification.demographics,
        content_warnings: classification.content_warnings,
        tags: classification.tags,
        sources,
        partial_sources,
        images: classification.images,
        opening_themes,
        ending_themes,
        trailers: links.trailers,
        external_sources: links.external_sources,
        streaming_sources: links.streaming_sources,
        statistics,
        broadcast,
        hiatus,
        related_anime,
        related_source_material,
        companies,
    })
}

/// Titles, classification lists and images of one work.
struct AnimeClassification {
    titles: Vec<AnimeTitleInput>,
    synonyms: Vec<String>,
    genres: Vec<String>,
    themes: Vec<AnimeThemeInput>,
    demographics: Vec<String>,
    content_warnings: Vec<String>,
    tags: Vec<String>,
    images: Vec<AnimeImageInput>,
}

/// Trailers, external links and streaming sources of one work.
struct AnimeLinks {
    trailers: Vec<AnimeTrailerInput>,
    external_sources: Vec<AnimeExternalSourceInput>,
    streaming_sources: Vec<AnimeStreamingSourceInput>,
}

/// Reads localized titles, classification lists and images of a work.
async fn read_anime_classification(
    tx: &mut Transaction<'_, Postgres>,
    anime_id: Uuid,
) -> Result<AnimeClassification, Error> {
    let titles = sqlx::query_as!(
        AnimeTitleInput,
        "SELECT language, title FROM anime_title WHERE anime_id = $1 ORDER BY language",
        anime_id
    )
    .fetch_all(&mut **tx)
    .await?;
    let synonyms = sqlx::query_scalar!(
        "SELECT synonym FROM anime_synonym WHERE anime_id = $1 ORDER BY synonym",
        anime_id
    )
    .fetch_all(&mut **tx)
    .await?;
    let genres = sqlx::query_scalar!(
        "SELECT genre FROM anime_genre WHERE anime_id = $1 ORDER BY genre",
        anime_id
    )
    .fetch_all(&mut **tx)
    .await?;
    let themes = sqlx::query_as!(
        AnimeThemeInput,
        r#"
        SELECT anime_theme.theme AS name, theme.description
        FROM anime_theme
        JOIN theme ON theme.name = anime_theme.theme
        WHERE anime_theme.anime_id = $1
        ORDER BY anime_theme.theme
        "#,
        anime_id
    )
    .fetch_all(&mut **tx)
    .await?;
    let demographics = sqlx::query_scalar!(
        "SELECT demographic FROM anime_demographic WHERE anime_id = $1 ORDER BY demographic",
        anime_id
    )
    .fetch_all(&mut **tx)
    .await?;
    let content_warnings = sqlx::query_scalar!(
        "SELECT content_warning FROM anime_content_warning
         WHERE anime_id = $1 ORDER BY content_warning",
        anime_id
    )
    .fetch_all(&mut **tx)
    .await?;
    let tags = sqlx::query_scalar!(
        "SELECT tag FROM anime_tag WHERE anime_id = $1 ORDER BY tag",
        anime_id
    )
    .fetch_all(&mut **tx)
    .await?;
    let images = sqlx::query_as!(
        AnimeImageInput,
        r#"SELECT image_type, source AS url FROM anime_image
           WHERE anime_id = $1 ORDER BY image_type, source"#,
        anime_id
    )
    .fetch_all(&mut **tx)
    .await?;
    Ok(AnimeClassification {
        titles,
        synonyms,
        genres,
        themes,
        demographics,
        content_warnings,
        tags,
        images,
    })
}

/// Reads the URLs of a work's source and partial-source pages.
///
/// A page whose URL no merge has chosen yet is left out until its owner's next save writes it.
async fn read_anime_page_sources(
    tx: &mut Transaction<'_, Postgres>,
    anime_id: Uuid,
) -> Result<(Vec<String>, Vec<String>), Error> {
    let page_sources = sqlx::query!(
        r#"
        SELECT page.source AS "source!", false AS "is_partial_source!"
        FROM anime_source link
        JOIN provider_page page USING (provider, entity_type, address)
        WHERE link.anime_id = $1 AND page.source IS NOT NULL
        UNION ALL
        SELECT page.source AS "source!", true AS "is_partial_source!"
        FROM anime_partial_source link
        JOIN provider_page page USING (provider, entity_type, address)
        WHERE link.anime_id = $1 AND page.source IS NOT NULL
        ORDER BY 1
        "#,
        anime_id
    )
    .fetch_all(&mut **tx)
    .await?;
    let (partial_page_sources, own_page_sources): (Vec<_>, Vec<_>) = page_sources
        .into_iter()
        .partition(|page_source| page_source.is_partial_source);
    Ok((
        own_page_sources
            .into_iter()
            .map(|page_source| page_source.source)
            .collect(),
        partial_page_sources
            .into_iter()
            .map(|page_source| page_source.source)
            .collect(),
    ))
}

/// One theme song row with its database ID, before its episode ranges are attached.
struct ThemeSongRow {
    id: i64,
    position: i32,
    title: String,
    artist: Option<String>,
}

/// One episode range of a theme song.
struct ThemeSongRangeRow {
    theme_song_id: i64,
    range_start: i32,
    range_end: Option<i32>,
}

/// Reads a work's opening themes with their episode ranges.
async fn read_opening_themes(
    tx: &mut Transaction<'_, Postgres>,
    anime_id: Uuid,
) -> Result<Vec<AnimeThemeSongInput>, Error> {
    let songs = sqlx::query_as!(
        ThemeSongRow,
        "SELECT id, position, title, artist FROM anime_opening_theme
         WHERE anime_id = $1 ORDER BY position",
        anime_id
    )
    .fetch_all(&mut **tx)
    .await?;
    let song_ids: Vec<i64> = songs.iter().map(|song| song.id).collect();
    let ranges = sqlx::query_as!(
        ThemeSongRangeRow,
        "SELECT opening_theme_id AS theme_song_id, range_start, range_end
         FROM anime_opening_theme_episode
         WHERE opening_theme_id = ANY($1) ORDER BY range_start",
        &song_ids
    )
    .fetch_all(&mut **tx)
    .await?;
    Ok(build_theme_songs(songs, ranges))
}

/// Reads a work's ending themes with their episode ranges.
async fn read_ending_themes(
    tx: &mut Transaction<'_, Postgres>,
    anime_id: Uuid,
) -> Result<Vec<AnimeThemeSongInput>, Error> {
    let songs = sqlx::query_as!(
        ThemeSongRow,
        "SELECT id, position, title, artist FROM anime_ending_theme
         WHERE anime_id = $1 ORDER BY position",
        anime_id
    )
    .fetch_all(&mut **tx)
    .await?;
    let song_ids: Vec<i64> = songs.iter().map(|song| song.id).collect();
    let ranges = sqlx::query_as!(
        ThemeSongRangeRow,
        "SELECT ending_theme_id AS theme_song_id, range_start, range_end
         FROM anime_ending_theme_episode
         WHERE ending_theme_id = ANY($1) ORDER BY range_start",
        &song_ids
    )
    .fetch_all(&mut **tx)
    .await?;
    Ok(build_theme_songs(songs, ranges))
}

/// Attaches each episode range to its theme song, keeping the songs' order.
fn build_theme_songs(
    songs: Vec<ThemeSongRow>,
    ranges: Vec<ThemeSongRangeRow>,
) -> Vec<AnimeThemeSongInput> {
    let mut ranges_by_song: BTreeMap<i64, Vec<EpisodeRangeInput>> = BTreeMap::new();
    for range in ranges {
        ranges_by_song
            .entry(range.theme_song_id)
            .or_default()
            .push(EpisodeRangeInput {
                range_start: range.range_start,
                range_end: range.range_end,
            });
    }
    songs
        .into_iter()
        .map(|song| AnimeThemeSongInput {
            position: song.position,
            title: song.title,
            artist: song.artist,
            episode_ranges: ranges_by_song.remove(&song.id).unwrap_or_default(),
        })
        .collect()
}

/// Reads trailers, external links and streaming sources of a work.
async fn read_anime_links(
    tx: &mut Transaction<'_, Postgres>,
    anime_id: Uuid,
) -> Result<AnimeLinks, Error> {
    let trailers = sqlx::query_as!(
        AnimeTrailerInput,
        r#"
        SELECT anime_trailer.source, trailer.title, trailer.thumbnail_source
        FROM anime_trailer
        JOIN trailer ON trailer.source = anime_trailer.source
        WHERE anime_trailer.anime_id = $1
        ORDER BY anime_trailer.source
        "#,
        anime_id
    )
    .fetch_all(&mut **tx)
    .await?;
    let external_sources = sqlx::query_as!(
        AnimeExternalSourceInput,
        r#"
        SELECT link.source, external_source.platform, link.label, external_source.language
        FROM anime_external_source link
        JOIN external_source ON external_source.source = link.source
        WHERE link.anime_id = $1
        ORDER BY link.source
        "#,
        anime_id
    )
    .fetch_all(&mut **tx)
    .await?;
    let streaming_rows = sqlx::query!(
        r#"
        SELECT streaming.id, external_source.platform, streaming.source, streaming.region,
               streaming.is_free, streaming.is_premium_required, streaming.has_dub,
               coalesce(
                   array_agg(subtitle.language ORDER BY subtitle.language)
                       FILTER (WHERE subtitle.language IS NOT NULL),
                   '{}'
               ) AS "subtitle_languages!"
        FROM anime_streaming_source streaming
        JOIN external_source ON external_source.source = streaming.source
        LEFT JOIN anime_streaming_source_subtitle_language subtitle
          ON subtitle.streaming_source_id = streaming.id
        WHERE streaming.anime_id = $1
        GROUP BY streaming.id, external_source.platform
        ORDER BY streaming.source, streaming.id
        "#,
        anime_id
    )
    .fetch_all(&mut **tx)
    .await?;
    let streaming_sources = streaming_rows
        .into_iter()
        .map(|streaming| AnimeStreamingSourceInput {
            platform: streaming.platform,
            source: streaming.source,
            region: streaming.region,
            is_free: streaming.is_free,
            is_premium_required: streaming.is_premium_required,
            has_dub: streaming.has_dub,
            subtitle_languages: streaming.subtitle_languages,
        })
        .collect();
    Ok(AnimeLinks {
        trailers,
        external_sources,
        streaming_sources,
    })
}

/// Reads per-provider statistics and contextual ranks of a work.
async fn read_anime_statistics(
    tx: &mut Transaction<'_, Postgres>,
    anime_id: Uuid,
) -> Result<Vec<AnimeStatisticsInput>, Error> {
    let rank_rows = sqlx::query!(
        "SELECT provider, context, rank, is_all_time, format, season, year
         FROM anime_statistics_contextual_rank
         WHERE anime_id = $1 ORDER BY provider, context",
        anime_id
    )
    .fetch_all(&mut **tx)
    .await?;
    let mut ranks_by_provider: BTreeMap<String, Vec<AnimeContextualRankInput>> = BTreeMap::new();
    for rank_row in rank_rows {
        ranks_by_provider
            .entry(rank_row.provider)
            .or_default()
            .push(AnimeContextualRankInput {
                context: rank_row.context,
                rank: rank_row.rank,
                is_all_time: rank_row.is_all_time,
                format: rank_row.format,
                season: rank_row.season,
                year: rank_row.year,
            });
    }

    let statistics_rows = sqlx::query!(
        "SELECT provider, score::float8 AS score, scored_by, members, favorites, rank, popularity
         FROM anime_statistics WHERE anime_id = $1 ORDER BY provider",
        anime_id
    )
    .fetch_all(&mut **tx)
    .await?;
    Ok(statistics_rows
        .into_iter()
        .map(|statistics_row| AnimeStatisticsInput {
            contextual_ranks: ranks_by_provider
                .remove(&statistics_row.provider)
                .unwrap_or_default(),
            provider: statistics_row.provider,
            score: statistics_row.score,
            scored_by: statistics_row.scored_by,
            members: statistics_row.members,
            favorites: statistics_row.favorites,
            rank: statistics_row.rank,
            popularity: statistics_row.popularity,
        })
        .collect())
}

/// Reads the broadcast schedule and the latest hiatus record of a work.
async fn read_anime_broadcast_and_hiatus(
    tx: &mut Transaction<'_, Postgres>,
    anime_id: Uuid,
) -> Result<(Option<AnimeBroadcastInput>, Option<AnimeHiatusInput>), Error> {
    let broadcast = sqlx::query_as!(
        AnimeBroadcastInput,
        "SELECT weekday, timezone, jp_time, sub_time, dub_time,
                sub_delay_days, dub_delay_days, premiere_jp, premiere_sub, premiere_dub,
                next_episode_at, next_episode_number
         FROM anime_broadcast WHERE anime_id = $1",
        anime_id
    )
    .fetch_optional(&mut **tx)
    .await?;
    let hiatus = sqlx::query_as!(
        AnimeHiatusInput,
        "SELECT reason, hiatus_from, hiatus_until
         FROM anime_hiatus WHERE anime_id = $1
         ORDER BY recorded_at DESC, id DESC LIMIT 1",
        anime_id
    )
    .fetch_optional(&mut **tx)
    .await?;
    Ok((broadcast, hiatus))
}

/// Reads the relations a work's own pages list, one per related visible work.
///
/// The target is whichever visible work holds a target page as a source when the read runs,
/// other than the work itself. Relation rows of one type that name pages of the same target form
/// one relation with every target page URL. A target page that no visible work holds stays a
/// relation of its own, shown by its page URL and title when they are stored.
async fn read_related_anime(
    tx: &mut Transaction<'_, Postgres>,
    anime_id: Uuid,
) -> Result<Vec<VisibleAnimeRelation>, Error> {
    let relation_rows = sqlx::query!(
        r#"
        SELECT relation.relation_type AS "relation_type!",
               target.id AS "target_anime_id?", target.title AS "target_title?",
               target.type AS "target_type?", target.status AS "target_status?",
               target.year AS "target_year?", target.episode_count AS "target_episode_count?",
               coalesce(
                   array_agg(target_page.source ORDER BY target_page.source)
                       FILTER (WHERE target_page.source IS NOT NULL),
                   '{}'
               ) AS "target_page_sources!",
               (array_agg(target_page.title
                          ORDER BY relation.to_page_provider, relation.to_page_address)
                    FILTER (WHERE target_page.title IS NOT NULL))[1] AS "target_page_title?"
        FROM anime_related_anime relation
        LEFT JOIN provider_page target_page
          ON target_page.provider = relation.to_page_provider
         AND target_page.entity_type = 'anime'
         AND target_page.address = relation.to_page_address
        LEFT JOIN anime_source target_link
          ON target_link.provider = relation.to_page_provider
         AND target_link.entity_type = 'anime'
         AND target_link.address = relation.to_page_address
         AND target_link.anime_id <> relation.from_anime_id
        LEFT JOIN anime_visible target ON target.id = target_link.anime_id
        WHERE relation.from_anime_id = $1
        GROUP BY relation.relation_type, target.id, target.title, target.type, target.status,
                 target.year, target.episode_count,
                 CASE WHEN target.id IS NULL
                      THEN relation.to_page_provider || ':' || relation.to_page_address END
        ORDER BY relation.relation_type, target.title NULLS LAST,
                 min(relation.to_page_provider || ':' || relation.to_page_address)
        "#,
        anime_id
    )
    .fetch_all(&mut **tx)
    .await?;
    Ok(relation_rows
        .into_iter()
        .map(|relation_row| {
            let target_anime = match (
                relation_row.target_anime_id,
                relation_row.target_title,
                relation_row.target_type,
                relation_row.target_status,
                relation_row.target_episode_count,
            ) {
                (
                    Some(target_anime_id),
                    Some(title),
                    Some(anime_type),
                    Some(status),
                    Some(count),
                ) => Some(RelatedAnimeSummary {
                    anime_id: target_anime_id,
                    title,
                    anime_type,
                    status,
                    year: relation_row.target_year,
                    episode_count: count,
                }),
                _ => None,
            };
            VisibleAnimeRelation {
                relation_type: relation_row.relation_type,
                target_page_sources: relation_row.target_page_sources,
                target_page_title: relation_row.target_page_title,
                target_anime,
            }
        })
        .collect())
}

/// Reads a work's related source material with its images and cross-references.
async fn read_related_source_material(
    tx: &mut Transaction<'_, Postgres>,
    anime_id: Uuid,
) -> Result<Vec<AnimeSourceMaterialInput>, Error> {
    let material_rows = sqlx::query!(
        r#"
        SELECT relation.relation_type, material.id, material.title, material.title_english,
               material.title_japanese, material.type AS source_material_type, material.status,
               material.score::float8 AS score, material.chapters, material.volumes,
               coalesce((SELECT array_agg(image.source ORDER BY image.source)
                         FROM source_material_image image
                         WHERE image.source_material_id = material.id), '{}') AS "images!",
               coalesce((SELECT array_agg(xref.provider ORDER BY xref.provider)
                         FROM source_material_xref xref
                         WHERE xref.source_material_id = material.id), '{}') AS "xref_providers!",
               coalesce((SELECT array_agg(xref.external_id ORDER BY xref.provider)
                         FROM source_material_xref xref
                         WHERE xref.source_material_id = material.id), '{}') AS "xref_external_ids!",
               coalesce((SELECT array_agg(xref.source ORDER BY xref.provider)
                         FROM source_material_xref xref
                         WHERE xref.source_material_id = material.id), '{}') AS "xref_sources!"
        FROM anime_related_source_material relation
        JOIN source_material material ON material.id = relation.source_material_id
        WHERE relation.anime_id = $1
        ORDER BY relation.relation_type, material.title, material.id
        "#,
        anime_id
    )
    .fetch_all(&mut **tx)
    .await?;
    Ok(material_rows
        .into_iter()
        .map(|material_row| AnimeSourceMaterialInput {
            relation_type: material_row.relation_type,
            title: material_row.title,
            title_english: material_row.title_english,
            title_japanese: material_row.title_japanese,
            source_material_type: material_row.source_material_type,
            status: material_row.status,
            score: material_row.score,
            chapters: material_row.chapters,
            volumes: material_row.volumes,
            images: material_row.images,
            xrefs: material_row
                .xref_providers
                .into_iter()
                .zip(material_row.xref_external_ids)
                .zip(material_row.xref_sources)
                .map(|((provider, external_id), url)| SourceMaterialXrefInput {
                    provider,
                    external_id,
                    url,
                    title: None,
                })
                .collect(),
        })
        .collect())
}

/// Reads a work's companies, each once with all of its roles and cross-references.
async fn read_anime_companies(
    tx: &mut Transaction<'_, Postgres>,
    anime_id: Uuid,
) -> Result<Vec<AnimeCompanyInput>, Error> {
    let company_rows = sqlx::query!(
        r#"
        SELECT company.name, company.description,
               array_agg(credit.role ORDER BY credit.role) AS "roles!",
               coalesce((SELECT array_agg(xref.provider ORDER BY xref.provider)
                         FROM company_xref xref
                         WHERE xref.company_id = company.id), '{}') AS "xref_providers!",
               coalesce((SELECT array_agg(xref.source ORDER BY xref.provider)
                         FROM company_xref xref
                         WHERE xref.company_id = company.id), '{}') AS "xref_sources!"
        FROM anime_company credit
        JOIN company ON company.id = credit.company_id
        WHERE credit.anime_id = $1
        GROUP BY company.id
        ORDER BY company.name, company.id
        "#,
        anime_id
    )
    .fetch_all(&mut **tx)
    .await?;
    Ok(company_rows
        .into_iter()
        .map(|company_row| AnimeCompanyInput {
            name: company_row.name,
            description: company_row.description,
            roles: company_row.roles,
            xrefs: company_row
                .xref_providers
                .into_iter()
                .zip(company_row.xref_sources)
                .map(|(provider, url)| CompanyXrefInput { provider, url })
                .collect(),
        })
        .collect())
}
