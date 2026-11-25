//! Read queries for merged, undeleted works (`anime_visible`) and their child tables.

use std::collections::BTreeMap;

use chrono::{DateTime, NaiveDate, NaiveTime, Utc};
use sqlx::{PgPool, Row};
use uuid::Uuid;

use crate::{
    AnimeBroadcastInput, AnimeContextualRankInput, AnimeExternalSourceInput, AnimeHiatusInput,
    AnimeImageInput, AnimeStatisticsInput, AnimeStreamingSourceInput, AnimeThemeInput,
    AnimeThemeSongInput, AnimeTitleInput, AnimeTrailerInput, EntityType, EpisodeRangeInput, Error,
    PageKey,
};

/// Related anime link projected for [`VisibleAnimeRecord`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VisibleAnimeRelation {
    /// Relation edge type (`SEQUEL`, `PREQUEL`, etc.).
    pub relation_type: String,
    /// Target work UUID when the target page is currently held in `anime_source`.
    pub target_anime_id: Option<Uuid>,
    /// Target work title when resolved to an undeleted work.
    pub target_title: Option<String>,
    /// Target provider slug.
    pub target_provider: String,
    /// Target provider address.
    pub target_address: String,
}

/// Related source material link projected for [`VisibleAnimeRecord`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VisibleSourceMaterialRelation {
    /// Relation edge type (`ADAPTATION`, `SOURCE`, etc.).
    pub relation_type: String,
    /// Source material type (`MANGA`, `LIGHT_NOVEL`, etc.).
    pub source_material_type: String,
    /// Source material title.
    pub title: String,
}

/// Company credit projected for [`VisibleAnimeRecord`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VisibleAnimeCompany {
    /// Company display name.
    pub name: String,
    /// Company role (`STUDIO`, `PRODUCER`, `LICENSOR`).
    pub role: String,
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
    /// Work format type (`TV`, `MOVIE`, etc.).
    pub anime_type: String,
    /// Airing status (`ONGOING`, `FINISHED`, `UPCOMING`, etc.).
    pub status: String,
    /// Content rating (`PG_13`, `R`, etc.).
    pub rating: String,
    /// Release season (`SPRING`, `SUMMER`, `FALL`, `WINTER`).
    pub season: Option<String>,
    /// Original source material (`MANGA`, `ORIGINAL`, etc.).
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
    /// AI-written synopsis.
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
    /// Source page keys merged into this work.
    pub source_pages: Vec<PageKey>,
    /// Partial-source page keys linked for reference.
    pub partial_source_pages: Vec<PageKey>,
    /// Cover/poster/banner image links.
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
    /// Per-provider statistics (with nested contextual ranks).
    pub statistics: Vec<AnimeStatisticsInput>,
    /// Broadcast schedule, if present.
    pub broadcast: Option<AnimeBroadcastInput>,
    /// Latest hiatus record, if present.
    pub hiatus: Option<AnimeHiatusInput>,
    /// Related works.
    pub related_anime: Vec<VisibleAnimeRelation>,
    /// Related source materials.
    pub related_source_material: Vec<VisibleSourceMaterialRelation>,
    /// Involved companies.
    pub companies: Vec<VisibleAnimeCompany>,
}

/// Reads a single merged, undeleted work and all of its detail rows from `anime_visible`.
///
/// # Examples
///
/// ```no_run
/// # use uuid::Uuid;
/// # async fn example(pool: &sqlx::PgPool, anime_id: Uuid) -> Result<(), postgres_db::Error> {
/// let work = postgres_db::read_visible_anime(pool, anime_id).await?;
/// assert_eq!(work.anime_id, anime_id);
/// assert!(work.version >= 1);
/// assert!(!work.title.is_empty());
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
    let base = sqlx::query(
        "SELECT id, version, next_refresh_at, title, title_english, title_japanese,
                type AS anime_type, status, rating, season, source_material, year, month,
                episode_count, duration_seconds, is_nsfw, country_of_origin, synopsis,
                background, aired_from, aired_to,
                score_weighted::float8 AS score_weighted,
                score_mean::float8 AS score_mean,
                score_median::float8 AS score_median
         FROM anime_visible
         WHERE id = $1",
    )
    .bind(anime_id)
    .fetch_optional(pool)
    .await?
    .ok_or_else(Error::not_found)?;

    let (titles, synonyms, genres, themes, demographics, content_warnings, tags, images) =
        read_anime_classification_and_media(pool, anime_id).await?;
    let (source_pages, partial_source_pages) = read_anime_page_links(pool, anime_id).await?;
    let (opening_themes, ending_themes) = read_anime_theme_songs(pool, anime_id).await?;
    let (trailers, external_sources, streaming_sources) = read_anime_media(pool, anime_id).await?;
    let statistics = read_anime_statistics(pool, anime_id).await?;
    let (broadcast, hiatus) = read_anime_broadcast_and_hiatus(pool, anime_id).await?;
    let (related_anime, related_source_material, companies) =
        read_anime_relations_and_companies(pool, anime_id).await?;

    Ok(VisibleAnimeRecord {
        anime_id: base.try_get("id")?,
        version: base.try_get("version")?,
        next_refresh_at: base.try_get("next_refresh_at")?,
        title: base.try_get("title")?,
        title_english: base.try_get("title_english")?,
        title_japanese: base.try_get("title_japanese")?,
        anime_type: base.try_get("anime_type")?,
        status: base.try_get("status")?,
        rating: base.try_get("rating")?,
        season: base.try_get("season")?,
        source_material: base.try_get("source_material")?,
        year: base.try_get("year")?,
        month: base.try_get("month")?,
        episode_count: base.try_get("episode_count")?,
        duration_seconds: base.try_get("duration_seconds")?,
        is_nsfw: base.try_get("is_nsfw")?,
        country_of_origin: base.try_get("country_of_origin")?,
        synopsis: base.try_get("synopsis")?,
        background: base.try_get("background")?,
        aired_from: base.try_get("aired_from")?,
        aired_to: base.try_get("aired_to")?,
        score_weighted: base.try_get("score_weighted")?,
        score_mean: base.try_get("score_mean")?,
        score_median: base.try_get("score_median")?,
        titles,
        synonyms,
        genres,
        themes,
        demographics,
        content_warnings,
        tags,
        source_pages,
        partial_source_pages,
        images,
        opening_themes,
        ending_themes,
        trailers,
        external_sources,
        streaming_sources,
        statistics,
        broadcast,
        hiatus,
        related_anime,
        related_source_material,
        companies,
    })
}

/// Reads scalar string list rows for a single anime child table.
async fn read_anime_string_list(
    pool: &PgPool,
    sql: &str,
    anime_id: Uuid,
) -> Result<Vec<String>, Error> {
    let values: Vec<String> = sqlx::query_scalar(sql)
        .bind(anime_id)
        .fetch_all(pool)
        .await?;
    Ok(values)
}

/// Reads localized titles, classification labels, and images for a work.
async fn read_anime_classification_and_media(
    pool: &PgPool,
    anime_id: Uuid,
) -> Result<
    (
        Vec<AnimeTitleInput>,
        Vec<String>,
        Vec<String>,
        Vec<AnimeThemeInput>,
        Vec<String>,
        Vec<String>,
        Vec<String>,
        Vec<AnimeImageInput>,
    ),
    Error,
> {
    let title_rows: Vec<(String, String)> = sqlx::query_as(
        "SELECT language, title FROM anime_title WHERE anime_id = $1 ORDER BY language",
    )
    .bind(anime_id)
    .fetch_all(pool)
    .await?;
    let titles = title_rows
        .into_iter()
        .map(|(language, title)| AnimeTitleInput { language, title })
        .collect();

    let synonyms = read_anime_string_list(
        pool,
        "SELECT synonym FROM anime_synonym WHERE anime_id = $1 ORDER BY synonym",
        anime_id,
    )
    .await?;
    let genres = read_anime_string_list(
        pool,
        "SELECT genre FROM anime_genre WHERE anime_id = $1 ORDER BY genre",
        anime_id,
    )
    .await?;
    let theme_rows: Vec<(String, Option<String>)> = sqlx::query_as(
        "SELECT at.theme, t.description
         FROM anime_theme at
         JOIN theme t ON t.name = at.theme
         WHERE at.anime_id = $1
         ORDER BY at.theme",
    )
    .bind(anime_id)
    .fetch_all(pool)
    .await?;
    let themes = theme_rows
        .into_iter()
        .map(|(name, description)| AnimeThemeInput { name, description })
        .collect();
    let demographics = read_anime_string_list(
        pool,
        "SELECT demographic FROM anime_demographic WHERE anime_id = $1 ORDER BY demographic",
        anime_id,
    )
    .await?;
    let content_warnings = read_anime_string_list(
        pool,
        "SELECT content_warning FROM anime_content_warning WHERE anime_id = $1 ORDER BY content_warning",
        anime_id,
    )
    .await?;
    let tags = read_anime_string_list(
        pool,
        "SELECT tag FROM anime_tag WHERE anime_id = $1 ORDER BY tag",
        anime_id,
    )
    .await?;

    let image_rows: Vec<(String, String)> = sqlx::query_as(
        "SELECT image_type, source FROM anime_image WHERE anime_id = $1 ORDER BY image_type, source",
    )
    .bind(anime_id)
    .fetch_all(pool)
    .await?;
    let images = image_rows
        .into_iter()
        .map(|(image_type, url)| AnimeImageInput { image_type, url })
        .collect();

    Ok((
        titles,
        synonyms,
        genres,
        themes,
        demographics,
        content_warnings,
        tags,
        images,
    ))
}

/// Reads source and partial-source page keys linked to a work.
async fn read_anime_page_links(
    pool: &PgPool,
    anime_id: Uuid,
) -> Result<(Vec<PageKey>, Vec<PageKey>), Error> {
    let source_rows: Vec<(String, String)> = sqlx::query_as(
        "SELECT provider, address FROM anime_source WHERE anime_id = $1 ORDER BY provider, address",
    )
    .bind(anime_id)
    .fetch_all(pool)
    .await?;
    let source_pages = source_rows
        .into_iter()
        .map(|(provider, address)| PageKey::new(provider.parse()?, EntityType::Anime, address))
        .collect::<Result<Vec<_>, _>>()?;

    let partial_rows: Vec<(String, String)> = sqlx::query_as(
        "SELECT provider, address FROM anime_partial_source WHERE anime_id = $1 ORDER BY provider, address",
    )
    .bind(anime_id)
    .fetch_all(pool)
    .await?;
    let partial_source_pages = partial_rows
        .into_iter()
        .map(|(provider, address)| PageKey::new(provider.parse()?, EntityType::Anime, address))
        .collect::<Result<Vec<_>, _>>()?;

    Ok((source_pages, partial_source_pages))
}

/// Reads opening and ending theme songs with their episode ranges.
async fn read_anime_theme_songs(
    pool: &PgPool,
    anime_id: Uuid,
) -> Result<(Vec<AnimeThemeSongInput>, Vec<AnimeThemeSongInput>), Error> {
    let openings = read_theme_kind(
        pool,
        anime_id,
        "SELECT id, position, title, artist FROM anime_opening_theme WHERE anime_id = $1 ORDER BY position",
        "SELECT opening_theme_id, range_start, range_end FROM anime_opening_theme_episode WHERE opening_theme_id = ANY($1) ORDER BY range_start",
    )
    .await?;
    let endings = read_theme_kind(
        pool,
        anime_id,
        "SELECT id, position, title, artist FROM anime_ending_theme WHERE anime_id = $1 ORDER BY position",
        "SELECT ending_theme_id, range_start, range_end FROM anime_ending_theme_episode WHERE ending_theme_id = ANY($1) ORDER BY range_start",
    )
    .await?;
    Ok((openings, endings))
}

/// Reads one theme song table and its episode range rows.
async fn read_theme_kind(
    pool: &PgPool,
    anime_id: Uuid,
    song_sql: &str,
    range_sql: &str,
) -> Result<Vec<AnimeThemeSongInput>, Error> {
    let songs: Vec<(i64, i32, String, Option<String>)> = sqlx::query_as(song_sql)
        .bind(anime_id)
        .fetch_all(pool)
        .await?;
    if songs.is_empty() {
        return Ok(Vec::new());
    }
    let song_ids: Vec<i64> = songs.iter().map(|(id, _, _, _)| *id).collect();
    let range_rows: Vec<(i64, i32, Option<i32>)> = sqlx::query_as(range_sql)
        .bind(&song_ids)
        .fetch_all(pool)
        .await?;

    let mut ranges_by_song: BTreeMap<i64, Vec<EpisodeRangeInput>> = BTreeMap::new();
    for (song_id, range_start, range_end) in range_rows {
        ranges_by_song
            .entry(song_id)
            .or_default()
            .push(EpisodeRangeInput {
                range_start,
                range_end,
            });
    }

    Ok(songs
        .into_iter()
        .map(|(song_id, position, title, artist)| AnimeThemeSongInput {
            position,
            title,
            artist,
            episode_ranges: ranges_by_song.remove(&song_id).unwrap_or_default(),
        })
        .collect())
}

/// Reads trailers, external links, and streaming sources for a work.
async fn read_anime_media(
    pool: &PgPool,
    anime_id: Uuid,
) -> Result<
    (
        Vec<AnimeTrailerInput>,
        Vec<AnimeExternalSourceInput>,
        Vec<AnimeStreamingSourceInput>,
    ),
    Error,
> {
    let trailer_rows: Vec<(String, Option<String>, Option<String>)> = sqlx::query_as(
        "SELECT at.source, t.title, t.thumbnail_source
         FROM anime_trailer at
         JOIN trailer t ON t.source = at.source
         WHERE at.anime_id = $1
         ORDER BY at.source",
    )
    .bind(anime_id)
    .fetch_all(pool)
    .await?;
    let trailers = trailer_rows
        .into_iter()
        .map(|(source, title, thumbnail_source)| AnimeTrailerInput {
            source,
            title,
            thumbnail_source,
        })
        .collect();

    let ext_rows: Vec<(String, String, Option<String>, Option<String>)> = sqlx::query_as(
        "SELECT aes.source, es.platform, aes.label, es.language
         FROM anime_external_source aes
         JOIN external_source es ON es.source = aes.source
         WHERE aes.anime_id = $1
         ORDER BY aes.source",
    )
    .bind(anime_id)
    .fetch_all(pool)
    .await?;
    let external_sources = ext_rows
        .into_iter()
        .map(
            |(source, platform, label, language)| AnimeExternalSourceInput {
                source,
                platform,
                label,
                language,
            },
        )
        .collect();

    let streaming_rows = sqlx::query(
        "SELECT ass.id, es.platform, ass.source, ass.region, ass.is_free,
                ass.is_premium_required, ass.has_dub
         FROM anime_streaming_source ass
         JOIN external_source es ON es.source = ass.source
         WHERE ass.anime_id = $1
         ORDER BY ass.source",
    )
    .bind(anime_id)
    .fetch_all(pool)
    .await?;
    let streaming_ids: Vec<i64> = streaming_rows
        .iter()
        .map(|row| row.try_get("id"))
        .collect::<Result<Vec<_>, _>>()?;
    let mut langs_by_stream: BTreeMap<i64, Vec<String>> = BTreeMap::new();
    if !streaming_ids.is_empty() {
        let lang_rows: Vec<(i64, String)> = sqlx::query_as(
            "SELECT streaming_source_id, language
             FROM anime_streaming_source_subtitle_language
             WHERE streaming_source_id = ANY($1)
             ORDER BY language",
        )
        .bind(&streaming_ids)
        .fetch_all(pool)
        .await?;
        for (stream_id, lang) in lang_rows {
            langs_by_stream.entry(stream_id).or_default().push(lang);
        }
    }
    let streaming_sources = streaming_rows
        .into_iter()
        .map(|row| {
            let id: i64 = row.try_get("id")?;
            Ok(AnimeStreamingSourceInput {
                platform: row.try_get("platform")?,
                source: row.try_get("source")?,
                region: row.try_get("region")?,
                is_free: row.try_get("is_free")?,
                is_premium_required: row.try_get("is_premium_required")?,
                has_dub: row.try_get("has_dub")?,
                subtitle_languages: langs_by_stream.remove(&id).unwrap_or_default(),
            })
        })
        .collect::<Result<Vec<_>, Error>>()?;

    Ok((trailers, external_sources, streaming_sources))
}

/// Reads per-provider statistics and contextual ranks for a work.
async fn read_anime_statistics(
    pool: &PgPool,
    anime_id: Uuid,
) -> Result<Vec<AnimeStatisticsInput>, Error> {
    let rank_rows = sqlx::query(
        "SELECT provider, context, rank, is_all_time, format, season, year
         FROM anime_statistics_contextual_rank WHERE anime_id = $1 ORDER BY provider, context",
    )
    .bind(anime_id)
    .fetch_all(pool)
    .await?;
    let mut ranks_by_provider: BTreeMap<String, Vec<AnimeContextualRankInput>> = BTreeMap::new();
    for row in rank_rows {
        let provider: String = row.try_get("provider")?;
        ranks_by_provider
            .entry(provider)
            .or_default()
            .push(AnimeContextualRankInput {
                context: row.try_get("context")?,
                rank: row.try_get("rank")?,
                is_all_time: row.try_get("is_all_time")?,
                format: row.try_get("format")?,
                season: row.try_get("season")?,
                year: row.try_get("year")?,
            });
    }

    let stat_rows = sqlx::query(
        "SELECT provider, score::float8 AS score, scored_by, members, favorites, rank, popularity
         FROM anime_statistics WHERE anime_id = $1 ORDER BY provider",
    )
    .bind(anime_id)
    .fetch_all(pool)
    .await?;
    stat_rows
        .into_iter()
        .map(|row| {
            let provider: String = row.try_get("provider")?;
            let contextual_ranks = ranks_by_provider.remove(&provider).unwrap_or_default();
            Ok(AnimeStatisticsInput {
                provider,
                score: row.try_get("score")?,
                scored_by: row.try_get("scored_by")?,
                members: row.try_get("members")?,
                favorites: row.try_get("favorites")?,
                rank: row.try_get("rank")?,
                popularity: row.try_get("popularity")?,
                contextual_ranks,
            })
        })
        .collect::<Result<Vec<_>, Error>>()
}

/// Reads broadcast schedule and latest hiatus snapshot for a work.
async fn read_anime_broadcast_and_hiatus(
    pool: &PgPool,
    anime_id: Uuid,
) -> Result<(Option<AnimeBroadcastInput>, Option<AnimeHiatusInput>), Error> {
    let broadcast_row = sqlx::query(
        "SELECT weekday, timezone, jp_time, sub_time, dub_time,
                sub_delay_days, dub_delay_days, premiere_jp, premiere_sub, premiere_dub,
                next_episode_at, next_episode_number
         FROM anime_broadcast WHERE anime_id = $1",
    )
    .bind(anime_id)
    .fetch_optional(pool)
    .await?;

    let broadcast = broadcast_row
        .map(|row| -> Result<AnimeBroadcastInput, Error> {
            Ok(AnimeBroadcastInput {
                weekday: row.try_get("weekday")?,
                timezone: row.try_get("timezone")?,
                jp_time: row.try_get::<Option<NaiveTime>, _>("jp_time")?,
                sub_time: row.try_get::<Option<NaiveTime>, _>("sub_time")?,
                dub_time: row.try_get::<Option<NaiveTime>, _>("dub_time")?,
                sub_delay_days: row.try_get("sub_delay_days")?,
                dub_delay_days: row.try_get("dub_delay_days")?,
                premiere_jp: row.try_get("premiere_jp")?,
                premiere_sub: row.try_get("premiere_sub")?,
                premiere_dub: row.try_get("premiere_dub")?,
                next_episode_at: row.try_get("next_episode_at")?,
                next_episode_number: row.try_get("next_episode_number")?,
            })
        })
        .transpose()?;

    let hiatus_row = sqlx::query(
        "SELECT reason, hiatus_from, hiatus_until
         FROM anime_hiatus WHERE anime_id = $1
         ORDER BY recorded_at DESC, id DESC LIMIT 1",
    )
    .bind(anime_id)
    .fetch_optional(pool)
    .await?;

    let hiatus = hiatus_row
        .map(|row| -> Result<AnimeHiatusInput, Error> {
            Ok(AnimeHiatusInput {
                reason: row.try_get("reason")?,
                hiatus_from: row.try_get::<Option<NaiveDate>, _>("hiatus_from")?,
                hiatus_until: row.try_get::<Option<NaiveDate>, _>("hiatus_until")?,
            })
        })
        .transpose()?;

    Ok((broadcast, hiatus))
}

/// Reads related anime, related source materials, and companies for a work.
async fn read_anime_relations_and_companies(
    pool: &PgPool,
    anime_id: Uuid,
) -> Result<
    (
        Vec<VisibleAnimeRelation>,
        Vec<VisibleSourceMaterialRelation>,
        Vec<VisibleAnimeCompany>,
    ),
    Error,
> {
    let rel_rows = sqlx::query(
        "SELECT r.relation_type, r.to_page_provider, r.to_page_address,
                target.id AS target_anime_id, target.title AS target_title
         FROM anime_related_anime r
         LEFT JOIN anime_source ap
           ON ap.provider = r.to_page_provider
          AND ap.entity_type = 'anime'
          AND ap.address = r.to_page_address
         LEFT JOIN anime target
           ON target.id = ap.anime_id AND target.deleted_at IS NULL
         WHERE r.from_anime_id = $1
         ORDER BY r.relation_type, r.to_page_provider, r.to_page_address",
    )
    .bind(anime_id)
    .fetch_all(pool)
    .await?;
    let related_anime = rel_rows
        .into_iter()
        .map(|row| {
            Ok(VisibleAnimeRelation {
                relation_type: row.try_get("relation_type")?,
                target_anime_id: row.try_get("target_anime_id")?,
                target_title: row.try_get("target_title")?,
                target_provider: row.try_get("to_page_provider")?,
                target_address: row.try_get("to_page_address")?,
            })
        })
        .collect::<Result<Vec<_>, Error>>()?;

    let sm_rows = sqlx::query(
        "SELECT r.relation_type, sm.type AS source_material_type, sm.title
         FROM anime_related_source_material r
         JOIN source_material sm ON sm.id = r.source_material_id
         WHERE r.anime_id = $1
         ORDER BY r.relation_type, sm.title",
    )
    .bind(anime_id)
    .fetch_all(pool)
    .await?;
    let related_source_material = sm_rows
        .into_iter()
        .map(|row| {
            Ok(VisibleSourceMaterialRelation {
                relation_type: row.try_get("relation_type")?,
                source_material_type: row.try_get("source_material_type")?,
                title: row.try_get("title")?,
            })
        })
        .collect::<Result<Vec<_>, Error>>()?;

    let company_rows = sqlx::query(
        "SELECT c.name, ac.role
         FROM anime_company ac
         JOIN company c ON c.id = ac.company_id
         WHERE ac.anime_id = $1
         ORDER BY ac.role, c.name",
    )
    .bind(anime_id)
    .fetch_all(pool)
    .await?;
    let companies = company_rows
        .into_iter()
        .map(|row| {
            Ok(VisibleAnimeCompany {
                name: row.try_get("name")?,
                role: row.try_get("role")?,
            })
        })
        .collect::<Result<Vec<_>, Error>>()?;

    Ok((related_anime, related_source_material, companies))
}
