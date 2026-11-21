//! Persistence helpers for theme songs, trailers, external links, streaming,
//! statistics, broadcast schedule, and hiatus history.

use std::collections::{BTreeMap, BTreeSet};

use chrono::{DateTime, NaiveDate, NaiveTime, Utc};
use sqlx::{Postgres, Transaction};
use uuid::Uuid;

use crate::anime::MIN_ANIME_YEAR;
use crate::error::Error;

/// Episode range (`[range_start, range_end]`) for a theme song.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpisodeRangeInput {
    /// First episode number (`>= 0`).
    pub range_start: i32,
    /// Last episode number (`>= range_start`), or `None` while still in use.
    pub range_end: Option<i32>,
}

/// Opening or ending theme song entry stored in `anime_theme_song`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnimeThemeSongInput {
    /// Lookup value in `theme_song_kind` (`"opening"` or `"ending"`).
    pub kind: String,
    /// Zero-based position within the work's openings or endings.
    pub position: i32,
    /// Song title.
    pub title: String,
    /// Performing artist, if known.
    pub artist: Option<String>,
    /// Episode ranges where this theme song is used.
    pub episode_ranges: Vec<EpisodeRangeInput>,
}

/// Trailer entry stored in `anime_trailer`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnimeTrailerInput {
    /// Trailer URL.
    pub url: String,
    /// Optional trailer title.
    pub title: Option<String>,
    /// Optional thumbnail URL.
    pub thumbnail_url: Option<String>,
}

/// External link stored in `anime_external_link`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnimeExternalLinkInput {
    /// Destination URL.
    pub url: String,
    /// Platform name.
    pub platform: String,
    /// Optional link label.
    pub label: Option<String>,
    /// Optional BCP 47 language code.
    pub language: Option<String>,
}

/// Streaming availability entry stored in `anime_streaming`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnimeStreamingInput {
    /// Streaming platform name.
    pub platform: String,
    /// Watch URL.
    pub url: String,
    /// Region code, if region-specific.
    pub region: Option<String>,
    /// Whether free tier viewing is available.
    pub is_free: Option<bool>,
    /// Whether a paid subscription is required.
    pub is_premium_required: Option<bool>,
    /// Whether a dub is available.
    pub has_dub: Option<bool>,
    /// Available subtitle languages.
    pub subtitle_languages: Vec<String>,
}

/// Contextual ranking entry stored in `anime_contextual_rank`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnimeContextualRankInput {
    /// Ranking context description.
    pub context: String,
    /// Positive rank (`> 0`).
    pub rank: i32,
    /// Whether the rank is all-time.
    pub is_all_time: Option<bool>,
    /// Optional format scope.
    pub format: Option<String>,
    /// Optional `anime_season` lookup value.
    pub season: Option<String>,
    /// Optional year (`>= 1900`).
    pub year: Option<i16>,
}

/// Per-provider statistics stored in `anime_statistics`.
#[derive(Debug, Clone, PartialEq)]
pub struct AnimeStatisticsInput {
    /// Canonical `provider` lookup value.
    pub provider: String,
    /// Provider score (`0.0..=10.0`).
    pub score: Option<f64>,
    /// Number of users who scored.
    pub scored_by: Option<i32>,
    /// Number of list members.
    pub members: Option<i32>,
    /// Number of favorites.
    pub favorites: Option<i32>,
    /// Overall rank (`> 0`).
    pub rank: Option<i32>,
    /// Popularity rank (`> 0`).
    pub popularity: Option<i32>,
    /// Contextual ranks reported by this provider.
    pub contextual_ranks: Vec<AnimeContextualRankInput>,
}

/// Broadcast schedule and premiere timestamps stored in `anime_broadcast`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnimeBroadcastInput {
    /// ISO weekday (`1..=7`), if known.
    pub weekday: Option<i16>,
    /// IANA timezone name (for example `"Asia/Tokyo"`).
    pub timezone: Option<String>,
    /// Japanese broadcast time of day.
    pub jp_time: Option<NaiveTime>,
    /// Subtitled release time of day.
    pub sub_time: Option<NaiveTime>,
    /// Dubbed release time of day.
    pub dub_time: Option<NaiveTime>,
    /// Subtitle delay in days.
    pub sub_delay_days: Option<i32>,
    /// Dub delay in days.
    pub dub_delay_days: Option<i32>,
    /// Japanese premiere timestamp.
    pub premiere_jp: Option<DateTime<Utc>>,
    /// Subtitled premiere timestamp.
    pub premiere_sub: Option<DateTime<Utc>>,
    /// Dubbed premiere timestamp.
    pub premiere_dub: Option<DateTime<Utc>>,
    /// Next scheduled episode timestamp.
    pub next_episode_at: Option<DateTime<Utc>>,
    /// Next scheduled episode number (`>= 0`).
    pub next_episode_number: Option<i32>,
}

/// Hiatus snapshot stored in `anime_hiatus` when it changes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnimeHiatusInput {
    /// Reason for the hiatus, if stated.
    pub reason: Option<String>,
    /// Hiatus start date, if known.
    pub hiatus_from: Option<NaiveDate>,
    /// Hiatus end date, or `None` while ongoing.
    pub hiatus_until: Option<NaiveDate>,
}

/// Extended work tables (songs, links, streaming, statistics, broadcast, hiatus).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AnimeScheduleAndStatsInput {
    /// Opening and ending theme songs.
    pub theme_songs: Vec<AnimeThemeSongInput>,
    /// Trailers.
    pub trailers: Vec<AnimeTrailerInput>,
    /// External links.
    pub external_links: Vec<AnimeExternalLinkInput>,
    /// Streaming availability entries.
    pub streaming_entries: Vec<AnimeStreamingInput>,
    /// Per-provider statistics and contextual ranks.
    pub statistics: Vec<AnimeStatisticsInput>,
    /// Broadcast slot and premiere dates.
    pub broadcast: Option<AnimeBroadcastInput>,
    /// Current hiatus snapshot, if any.
    pub hiatus: Option<AnimeHiatusInput>,
}

/// Checks a work's songs, links, streaming, statistics and broadcast before any write.
///
/// # Errors
///
/// Returns an invalid-argument error naming the first field that is out of range or empty.
pub(crate) fn validate_schedule_and_stats(input: &AnimeScheduleAndStatsInput) -> Result<(), Error> {
    validate_theme_songs(&input.theme_songs)?;
    validate_trailers_and_links(
        &input.trailers,
        &input.external_links,
        &input.streaming_entries,
    )?;
    validate_statistics(&input.statistics)?;
    if let Some(broadcast) = input.broadcast.as_ref() {
        validate_broadcast(broadcast)?;
    }
    Ok(())
}

/// Checks theme song titles, positions and episode ranges.
///
/// # Errors
///
/// Returns an invalid-argument error for an unknown song kind, a negative position, a blank
/// title, a negative episode range start, or a range that ends before it starts.
fn validate_theme_songs(theme_songs: &[AnimeThemeSongInput]) -> Result<(), Error> {
    for song in theme_songs {
        if !matches!(song.kind.as_str(), "opening" | "ending") {
            return Err(Error::invalid_argument(format!(
                "unknown theme_song_kind {:?}",
                song.kind
            )));
        }
        if song.position < 0 || song.title.trim().is_empty() {
            return Err(Error::invalid_argument(
                "theme song requires non-negative position and non-empty title",
            ));
        }
        for range in &song.episode_ranges {
            if range.range_start < 0 {
                return Err(Error::invalid_argument("range_start must be >= 0"));
            }
            if let Some(range_end) = range.range_end
                && range_end < range.range_start
            {
                return Err(Error::invalid_argument("range_end must be >= range_start"));
            }
        }
    }
    Ok(())
}

/// Checks trailer, external link and streaming URLs and platforms.
///
/// # Errors
///
/// Returns an invalid-argument error for a blank URL, platform or subtitle language.
fn validate_trailers_and_links(
    trailers: &[AnimeTrailerInput],
    external_links: &[AnimeExternalLinkInput],
    streaming_entries: &[AnimeStreamingInput],
) -> Result<(), Error> {
    if trailers.iter().any(|trailer| trailer.url.trim().is_empty()) {
        return Err(Error::invalid_argument("trailer url must be non-empty"));
    }
    if external_links
        .iter()
        .any(|link| link.url.trim().is_empty() || link.platform.trim().is_empty())
    {
        return Err(Error::invalid_argument(
            "external link url and platform must be non-empty",
        ));
    }
    for entry in streaming_entries {
        if entry.url.trim().is_empty() || entry.platform.trim().is_empty() {
            return Err(Error::invalid_argument(
                "streaming url and platform must be non-empty",
            ));
        }
        if entry
            .subtitle_languages
            .iter()
            .any(|language| language.trim().is_empty())
        {
            return Err(Error::invalid_argument(
                "streaming subtitle_languages must be non-empty",
            ));
        }
    }
    Ok(())
}

/// Checks each provider's scores, counts and ranks.
///
/// # Errors
///
/// Returns an invalid-argument error for a score outside 0 to 10, a negative count, a rank or
/// popularity below 1, or a contextual rank with no context, a rank below 1 or a year too early.
fn validate_statistics(statistics: &[AnimeStatisticsInput]) -> Result<(), Error> {
    for stats in statistics {
        if let Some(score) = stats.score
            && (!(0.0..=10.0).contains(&score) || score.is_nan())
        {
            return Err(Error::invalid_argument(
                "statistics score must be between 0 and 10",
            ));
        }
        if [stats.scored_by, stats.members, stats.favorites]
            .into_iter()
            .flatten()
            .any(|count| count < 0)
        {
            return Err(Error::invalid_argument("statistics counts must be >= 0"));
        }
        if [stats.rank, stats.popularity]
            .into_iter()
            .flatten()
            .any(|rank| rank <= 0)
        {
            return Err(Error::invalid_argument(
                "statistics rank and popularity must be > 0",
            ));
        }
        for contextual in &stats.contextual_ranks {
            if contextual.context.trim().is_empty() || contextual.rank <= 0 {
                return Err(Error::invalid_argument(
                    "contextual rank requires non-empty context and rank > 0",
                ));
            }
            if let Some(year) = contextual.year
                && year < MIN_ANIME_YEAR
            {
                return Err(Error::invalid_argument(format!(
                    "contextual rank year {year} must be >= {MIN_ANIME_YEAR}"
                )));
            }
        }
    }
    Ok(())
}

/// Checks the broadcast weekday and next episode number.
///
/// # Errors
///
/// Returns an invalid-argument error for a weekday outside 1 to 7 or a negative episode number.
fn validate_broadcast(broadcast: &AnimeBroadcastInput) -> Result<(), Error> {
    if let Some(weekday) = broadcast.weekday
        && !(1..=7).contains(&weekday)
    {
        return Err(Error::invalid_argument(
            "broadcast weekday must be between 1 and 7",
        ));
    }
    if let Some(next_episode) = broadcast.next_episode_number
        && next_episode < 0
    {
        return Err(Error::invalid_argument(
            "broadcast next_episode_number must be >= 0",
        ));
    }
    Ok(())
}

/// Replaces a work's songs, trailers, links, streaming, statistics and broadcast, and records its hiatus.
///
/// # Errors
///
/// Returns a database error when a query fails.
pub(crate) async fn replace_schedule_and_stats(
    tx: &mut Transaction<'_, Postgres>,
    anime_id: Uuid,
    input: &AnimeScheduleAndStatsInput,
) -> Result<(), Error> {
    replace_theme_songs(tx, anime_id, &input.theme_songs).await?;
    replace_trailers(tx, anime_id, &input.trailers).await?;
    replace_external_links(tx, anime_id, &input.external_links).await?;
    replace_streaming(tx, anime_id, &input.streaming_entries).await?;
    replace_statistics(tx, anime_id, &input.statistics).await?;
    upsert_or_delete_broadcast(tx, anime_id, input.broadcast.as_ref()).await?;
    record_hiatus_if_changed(tx, anime_id, input.hiatus.as_ref()).await?;
    Ok(())
}

/// Replaces a work's opening and ending songs with their episode ranges.
///
/// # Errors
///
/// Returns a database error when a query fails.
async fn replace_theme_songs(
    tx: &mut Transaction<'_, Postgres>,
    anime_id: Uuid,
    theme_songs: &[AnimeThemeSongInput],
) -> Result<(), Error> {
    sqlx::query!("DELETE FROM anime_theme_song WHERE anime_id = $1", anime_id)
        .execute(&mut **tx)
        .await?;

    let mut deduplicated_songs: BTreeMap<(&str, i32), &AnimeThemeSongInput> = BTreeMap::new();
    for song in theme_songs {
        deduplicated_songs.insert((song.kind.as_str(), song.position), song);
    }

    for song in deduplicated_songs.into_values() {
        let song_id = sqlx::query_scalar!(
            "INSERT INTO anime_theme_song (anime_id, kind, position, title, artist)
             VALUES ($1, $2, $3, $4, $5)
             RETURNING id",
            anime_id,
            song.kind,
            song.position,
            song.title,
            song.artist,
        )
        .fetch_one(&mut **tx)
        .await?;

        let range_map: BTreeMap<i32, Option<i32>> = song
            .episode_ranges
            .iter()
            .map(|range| (range.range_start, range.range_end))
            .collect();
        if range_map.is_empty() {
            continue;
        }
        let range_starts: Vec<i32> = range_map.keys().copied().collect();
        let range_ends: Vec<Option<i32>> = range_map.values().copied().collect();
        sqlx::query!(
            "INSERT INTO theme_song_episode_range (theme_song_id, range_start, range_end)
             SELECT $1, range_start, range_end
             FROM UNNEST($2::int4[], $3::int4[]) AS entry(range_start, range_end)",
            song_id,
            &range_starts,
            &range_ends as &[Option<i32>],
        )
        .execute(&mut **tx)
        .await?;
    }

    Ok(())
}

/// Replaces a work's trailers.
///
/// # Errors
///
/// Returns a database error when a query fails.
async fn replace_trailers(
    tx: &mut Transaction<'_, Postgres>,
    anime_id: Uuid,
    trailers: &[AnimeTrailerInput],
) -> Result<(), Error> {
    sqlx::query!("DELETE FROM anime_trailer WHERE anime_id = $1", anime_id)
        .execute(&mut **tx)
        .await?;

    let deduplicated_trailers: BTreeMap<&str, &AnimeTrailerInput> = trailers
        .iter()
        .map(|trailer| (trailer.url.as_str(), trailer))
        .collect();
    if deduplicated_trailers.is_empty() {
        return Ok(());
    }

    let urls: Vec<String> = deduplicated_trailers
        .keys()
        .copied()
        .map(str::to_owned)
        .collect();
    let titles: Vec<Option<String>> = deduplicated_trailers
        .values()
        .map(|trailer| trailer.title.clone())
        .collect();
    let thumbnails: Vec<Option<String>> = deduplicated_trailers
        .values()
        .map(|trailer| trailer.thumbnail_url.clone())
        .collect();
    sqlx::query!(
        "INSERT INTO anime_trailer (anime_id, source, title, thumbnail_url)
         SELECT $1, source, title, thumbnail_url
         FROM UNNEST($2::text[], $3::text[], $4::text[]) AS entry(source, title, thumbnail_url)",
        anime_id,
        &urls,
        &titles as &[Option<String>],
        &thumbnails as &[Option<String>],
    )
    .execute(&mut **tx)
    .await?;
    Ok(())
}

/// Replaces a work's external links.
///
/// # Errors
///
/// Returns a database error when a query fails.
async fn replace_external_links(
    tx: &mut Transaction<'_, Postgres>,
    anime_id: Uuid,
    links: &[AnimeExternalLinkInput],
) -> Result<(), Error> {
    sqlx::query!(
        "DELETE FROM anime_external_link WHERE anime_id = $1",
        anime_id
    )
    .execute(&mut **tx)
    .await?;

    let deduplicated_links: BTreeMap<&str, &AnimeExternalLinkInput> =
        links.iter().map(|link| (link.url.as_str(), link)).collect();
    if deduplicated_links.is_empty() {
        return Ok(());
    }

    let urls: Vec<String> = deduplicated_links
        .keys()
        .copied()
        .map(str::to_owned)
        .collect();
    let platforms: Vec<String> = deduplicated_links
        .values()
        .map(|link| link.platform.clone())
        .collect();
    let labels: Vec<Option<String>> = deduplicated_links
        .values()
        .map(|link| link.label.clone())
        .collect();
    let languages: Vec<Option<String>> = deduplicated_links
        .values()
        .map(|link| link.language.clone())
        .collect();
    sqlx::query!(
        "INSERT INTO anime_external_link (anime_id, source, platform, label, language)
         SELECT $1, source, platform, label, language
         FROM UNNEST($2::text[], $3::text[], $4::text[], $5::text[])
           AS entry(source, platform, label, language)",
        anime_id,
        &urls,
        &platforms,
        &labels as &[Option<String>],
        &languages as &[Option<String>],
    )
    .execute(&mut **tx)
    .await?;
    Ok(())
}

/// Replaces a work's streaming entries with their subtitle languages.
///
/// # Errors
///
/// Returns a database error when a query fails.
async fn replace_streaming(
    tx: &mut Transaction<'_, Postgres>,
    anime_id: Uuid,
    entries: &[AnimeStreamingInput],
) -> Result<(), Error> {
    sqlx::query!("DELETE FROM anime_streaming WHERE anime_id = $1", anime_id)
        .execute(&mut **tx)
        .await?;

    let mut deduplicated_streaming: BTreeMap<(&str, Option<&str>), &AnimeStreamingInput> =
        BTreeMap::new();
    for entry in entries {
        deduplicated_streaming.insert((entry.url.as_str(), entry.region.as_deref()), entry);
    }

    for entry in deduplicated_streaming.into_values() {
        let streaming_id = sqlx::query_scalar!(
            "INSERT INTO anime_streaming (
                 anime_id, platform, source, region, is_free, is_premium_required, has_dub
             )
             VALUES ($1, $2, $3, $4, $5, $6, $7)
             RETURNING id",
            anime_id,
            entry.platform,
            entry.url,
            entry.region,
            entry.is_free,
            entry.is_premium_required,
            entry.has_dub,
        )
        .fetch_one(&mut **tx)
        .await?;

        let languages: Vec<String> = entry
            .subtitle_languages
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .map(str::to_owned)
            .collect();
        if languages.is_empty() {
            continue;
        }

        sqlx::query!(
            "INSERT INTO anime_streaming_subtitle_language (streaming_id, language)
             SELECT $1, language
             FROM UNNEST($2::text[]) AS entry(language)",
            streaming_id,
            &languages,
        )
        .execute(&mut **tx)
        .await?;
    }

    Ok(())
}

/// Replaces each provider's statistics for a work, with their contextual ranks.
///
/// # Errors
///
/// Returns a database error when a query fails.
async fn replace_statistics(
    tx: &mut Transaction<'_, Postgres>,
    anime_id: Uuid,
    statistics: &[AnimeStatisticsInput],
) -> Result<(), Error> {
    sqlx::query!("DELETE FROM anime_statistics WHERE anime_id = $1", anime_id)
        .execute(&mut **tx)
        .await?;

    let deduplicated_statistics: BTreeMap<&str, &AnimeStatisticsInput> = statistics
        .iter()
        .map(|stats| (stats.provider.as_str(), stats))
        .collect();

    for stats in deduplicated_statistics.into_values() {
        sqlx::query!(
            "INSERT INTO anime_statistics (
                 anime_id, provider, score, scored_by, members, favorites, rank, popularity
             )
             VALUES ($1, $2, ROUND($3::float8::numeric, 2), $4, $5, $6, $7, $8)",
            anime_id,
            stats.provider,
            stats.score,
            stats.scored_by,
            stats.members,
            stats.favorites,
            stats.rank,
            stats.popularity,
        )
        .execute(&mut **tx)
        .await?;

        let rank_map: BTreeMap<&str, &AnimeContextualRankInput> = stats
            .contextual_ranks
            .iter()
            .map(|rank| (rank.context.as_str(), rank))
            .collect();
        if rank_map.is_empty() {
            continue;
        }

        let contexts: Vec<String> = rank_map.keys().copied().map(str::to_owned).collect();
        let ranks: Vec<i32> = rank_map.values().map(|item| item.rank).collect();
        let is_all_times: Vec<Option<bool>> =
            rank_map.values().map(|item| item.is_all_time).collect();
        let formats: Vec<Option<String>> =
            rank_map.values().map(|item| item.format.clone()).collect();
        let seasons: Vec<Option<String>> =
            rank_map.values().map(|item| item.season.clone()).collect();
        let years: Vec<Option<i16>> = rank_map.values().map(|item| item.year).collect();

        sqlx::query!(
            "INSERT INTO anime_contextual_rank (
                 anime_id, provider, context, rank, is_all_time, format, season, year
             )
             SELECT $1, $2, context, rank, is_all_time, format, season, year
             FROM UNNEST($3::text[], $4::int4[], $5::bool[], $6::text[], $7::text[], $8::int2[])
               AS entry(context, rank, is_all_time, format, season, year)",
            anime_id,
            stats.provider,
            &contexts,
            &ranks,
            &is_all_times as &[Option<bool>],
            &formats as &[Option<String>],
            &seasons as &[Option<String>],
            &years as &[Option<i16>],
        )
        .execute(&mut **tx)
        .await?;
    }

    Ok(())
}

/// Stores a work's broadcast slot, or deletes it when the merge has none.
///
/// # Errors
///
/// Returns a database error when a query fails.
async fn upsert_or_delete_broadcast(
    tx: &mut Transaction<'_, Postgres>,
    anime_id: Uuid,
    broadcast: Option<&AnimeBroadcastInput>,
) -> Result<(), Error> {
    let Some(broadcast_row) = broadcast else {
        sqlx::query!("DELETE FROM anime_broadcast WHERE anime_id = $1", anime_id)
            .execute(&mut **tx)
            .await?;
        return Ok(());
    };

    sqlx::query!(
        "INSERT INTO anime_broadcast (
             anime_id, weekday, timezone, jp_time, sub_time, dub_time,
             sub_delay_days, dub_delay_days, premiere_jp, premiere_sub, premiere_dub,
             next_episode_at, next_episode_number
         )
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
         ON CONFLICT (anime_id) DO UPDATE SET
             weekday = EXCLUDED.weekday,
             timezone = EXCLUDED.timezone,
             jp_time = EXCLUDED.jp_time,
             sub_time = EXCLUDED.sub_time,
             dub_time = EXCLUDED.dub_time,
             sub_delay_days = EXCLUDED.sub_delay_days,
             dub_delay_days = EXCLUDED.dub_delay_days,
             premiere_jp = EXCLUDED.premiere_jp,
             premiere_sub = EXCLUDED.premiere_sub,
             premiere_dub = EXCLUDED.premiere_dub,
             next_episode_at = EXCLUDED.next_episode_at,
             next_episode_number = EXCLUDED.next_episode_number",
        anime_id,
        broadcast_row.weekday,
        broadcast_row.timezone,
        broadcast_row.jp_time,
        broadcast_row.sub_time,
        broadcast_row.dub_time,
        broadcast_row.sub_delay_days,
        broadcast_row.dub_delay_days,
        broadcast_row.premiere_jp,
        broadcast_row.premiere_sub,
        broadcast_row.premiere_dub,
        broadcast_row.next_episode_at,
        broadcast_row.next_episode_number,
    )
    .execute(&mut **tx)
    .await?;

    Ok(())
}

/// Adds a hiatus history row when the work's hiatus differs from the latest one.
///
/// A work with no hiatus in the merge keeps its history unchanged.
///
/// # Errors
///
/// Returns a database error when a query fails.
async fn record_hiatus_if_changed(
    tx: &mut Transaction<'_, Postgres>,
    anime_id: Uuid,
    hiatus: Option<&AnimeHiatusInput>,
) -> Result<(), Error> {
    let Some(hiatus_snapshot) = hiatus else {
        return Ok(());
    };

    let latest_hiatus = sqlx::query!(
        "SELECT reason, hiatus_from, hiatus_until
         FROM anime_hiatus
         WHERE anime_id = $1
         ORDER BY recorded_at DESC, id DESC
         LIMIT 1",
        anime_id,
    )
    .fetch_optional(&mut **tx)
    .await?;

    let is_unchanged = latest_hiatus.is_some_and(|row| {
        row.reason == hiatus_snapshot.reason
            && row.hiatus_from == hiatus_snapshot.hiatus_from
            && row.hiatus_until == hiatus_snapshot.hiatus_until
    });
    if is_unchanged {
        return Ok(());
    }

    sqlx::query!(
        "INSERT INTO anime_hiatus (anime_id, reason, hiatus_from, hiatus_until)
         VALUES ($1, $2, $3, $4)",
        anime_id,
        hiatus_snapshot.reason,
        hiatus_snapshot.hiatus_from,
        hiatus_snapshot.hiatus_until,
    )
    .execute(&mut **tx)
    .await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_valid_schedule_and_stats() -> AnimeScheduleAndStatsInput {
        AnimeScheduleAndStatsInput {
            theme_songs: vec![AnimeThemeSongInput {
                kind: "opening".to_owned(),
                position: 0,
                title: "Opening".to_owned(),
                artist: None,
                episode_ranges: vec![EpisodeRangeInput {
                    range_start: 1,
                    range_end: Some(12),
                }],
            }],
            trailers: vec![AnimeTrailerInput {
                url: "https://example.com/trailer".to_owned(),
                title: None,
                thumbnail_url: None,
            }],
            external_links: vec![AnimeExternalLinkInput {
                url: "https://example.com".to_owned(),
                platform: "Official".to_owned(),
                label: None,
                language: None,
            }],
            streaming_entries: vec![AnimeStreamingInput {
                platform: "Crunchyroll".to_owned(),
                url: "https://example.com/watch".to_owned(),
                region: None,
                is_free: None,
                is_premium_required: None,
                has_dub: None,
                subtitle_languages: vec!["en".to_owned()],
            }],
            statistics: vec![AnimeStatisticsInput {
                provider: "mal".to_owned(),
                score: Some(8.5),
                scored_by: Some(10),
                members: Some(20),
                favorites: Some(3),
                rank: Some(1),
                popularity: Some(2),
                contextual_ranks: vec![AnimeContextualRankInput {
                    context: "highest rated all time".to_owned(),
                    rank: 1,
                    is_all_time: Some(true),
                    format: None,
                    season: None,
                    year: Some(2004),
                }],
            }],
            broadcast: Some(AnimeBroadcastInput {
                weekday: Some(7),
                timezone: None,
                jp_time: None,
                sub_time: None,
                dub_time: None,
                sub_delay_days: None,
                dub_delay_days: None,
                premiere_jp: None,
                premiere_sub: None,
                premiere_dub: None,
                next_episode_at: None,
                next_episode_number: Some(5),
            }),
            hiatus: None,
        }
    }

    #[test]
    fn validate_schedule_and_stats_broken_rule_returns_invalid_argument() {
        let valid_schedule = build_valid_schedule_and_stats();
        assert!(validate_schedule_and_stats(&valid_schedule).is_ok());

        let breaking_changes: Vec<fn(&mut AnimeScheduleAndStatsInput)> = vec![
            |input| input.theme_songs[0].kind = "insert".to_owned(),
            |input| input.theme_songs[0].position = -1,
            |input| input.theme_songs[0].title = " ".to_owned(),
            |input| input.theme_songs[0].episode_ranges[0].range_start = -1,
            |input| input.theme_songs[0].episode_ranges[0].range_end = Some(0),
            |input| input.trailers[0].url = String::new(),
            |input| input.external_links[0].url = " ".to_owned(),
            |input| input.external_links[0].platform = String::new(),
            |input| input.streaming_entries[0].url = String::new(),
            |input| input.streaming_entries[0].platform = " ".to_owned(),
            |input| input.streaming_entries[0].subtitle_languages = vec![String::new()],
            |input| input.statistics[0].score = Some(11.0),
            |input| input.statistics[0].score = Some(f64::NAN),
            |input| input.statistics[0].members = Some(-1),
            |input| input.statistics[0].rank = Some(0),
            |input| input.statistics[0].popularity = Some(-3),
            |input| input.statistics[0].contextual_ranks[0].context = " ".to_owned(),
            |input| input.statistics[0].contextual_ranks[0].rank = 0,
            |input| input.statistics[0].contextual_ranks[0].year = Some(1800),
            |input| {
                input.broadcast.as_mut().expect("valid broadcast").weekday = Some(8);
            },
            |input| {
                input
                    .broadcast
                    .as_mut()
                    .expect("valid broadcast")
                    .next_episode_number = Some(-1);
            },
        ];
        for break_rule in breaking_changes {
            let mut broken_schedule = valid_schedule.clone();
            break_rule(&mut broken_schedule);

            let validation_error = validate_schedule_and_stats(&broken_schedule)
                .expect_err("a broken rule must be refused");

            assert!(validation_error.is_invalid_argument());
        }
    }
}
