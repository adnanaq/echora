//! Protobuf-to-SQL mapping for theme songs, trailers, external links, streaming,
//! statistics, broadcast schedule, and hiatus snapshot.

// Rust guideline compliant 2026-10-09

use std::collections::BTreeMap;

use chrono::{DateTime, NaiveDate, NaiveTime, Utc};
use postgres_db::{
    AnimeBroadcastInput, AnimeContextualRankInput, AnimeExternalLinkInput, AnimeHiatusInput,
    AnimeScheduleAndStatsInput, AnimeStatisticsInput, AnimeStreamingInput, AnimeThemeSongInput,
    AnimeTrailerInput, EpisodeRangeInput,
};
use prost_types::Timestamp;
use tonic::Status;

use crate::service::shared_proto::v1::{
    AnimeHiatus, Broadcast, ContextualRank, EpisodeRange, ExternalLink, Statistics, StreamingEntry,
    ThemeSong, TrailerEntry,
};

#[expect(
    clippy::too_many_arguments,
    reason = "Maps the eight ECHO-101 proto fields into AnimeScheduleAndStatsInput"
)]
pub(crate) fn build_schedule_and_stats(
    opening_themes: Vec<ThemeSong>,
    ending_themes: Vec<ThemeSong>,
    trailers: Vec<TrailerEntry>,
    external_sources: Vec<ExternalLink>,
    streaming_sources: Vec<StreamingEntry>,
    statistics: BTreeMap<String, Statistics>,
    broadcast: Option<Broadcast>,
    hiatus: Option<AnimeHiatus>,
) -> Result<AnimeScheduleAndStatsInput, Status> {
    let mut theme_songs = map_theme_songs("opening", opening_themes)?;
    theme_songs.extend(map_theme_songs("ending", ending_themes)?);

    let mapped_trailers = map_trailers(trailers);
    let mapped_links = map_external_links(external_sources);
    let mapped_streaming = map_streaming(streaming_sources);
    let mapped_statistics = map_statistics(statistics)?;
    let mapped_broadcast = broadcast.map(map_broadcast).transpose()?;
    let mapped_hiatus = hiatus.map(map_hiatus).transpose()?;

    Ok(AnimeScheduleAndStatsInput {
        theme_songs,
        trailers: mapped_trailers,
        external_links: mapped_links,
        streaming_entries: mapped_streaming,
        statistics: mapped_statistics,
        broadcast: mapped_broadcast,
        hiatus: mapped_hiatus,
    })
}

fn map_theme_songs(kind: &str, songs: Vec<ThemeSong>) -> Result<Vec<AnimeThemeSongInput>, Status> {
    songs
        .into_iter()
        .enumerate()
        .map(
            |(
                index,
                ThemeSong {
                    title,
                    artist,
                    episodes,
                },
            )| {
                let position = i32::try_from(index)
                    .map_err(|_| Status::invalid_argument("theme_song position overflows i32"))?;
                let episode_ranges = episodes
                    .into_iter()
                    .map(|EpisodeRange { start, end }| EpisodeRangeInput {
                        range_start: start,
                        range_end: end,
                    })
                    .collect();
                Ok(AnimeThemeSongInput {
                    kind: kind.to_owned(),
                    position,
                    title,
                    artist,
                    episode_ranges,
                })
            },
        )
        .collect()
}

fn map_trailers(trailers: Vec<TrailerEntry>) -> Vec<AnimeTrailerInput> {
    trailers
        .into_iter()
        .filter_map(
            |TrailerEntry {
                 source,
                 title,
                 thumbnail,
             }| {
                let url = source.filter(|value| !value.trim().is_empty())?;
                Some(AnimeTrailerInput {
                    url,
                    title,
                    thumbnail_url: thumbnail,
                })
            },
        )
        .collect()
}

fn map_external_links(links: Vec<ExternalLink>) -> Vec<AnimeExternalLinkInput> {
    links
        .into_iter()
        .map(
            |ExternalLink {
                 platform,
                 source,
                 label,
                 language,
             }| AnimeExternalLinkInput {
                url: source,
                platform,
                label,
                language,
            },
        )
        .collect()
}

fn map_streaming(entries: Vec<StreamingEntry>) -> Vec<AnimeStreamingInput> {
    entries
        .into_iter()
        .map(
            |StreamingEntry {
                 platform,
                 source,
                 region,
                 free,
                 premium_required,
                 dub_available,
                 subtitle_languages,
             }| AnimeStreamingInput {
                platform,
                url: source,
                region,
                is_free: free,
                is_premium_required: premium_required,
                has_dub: dub_available,
                subtitle_languages,
            },
        )
        .collect()
}

fn map_statistics(
    statistics: BTreeMap<String, Statistics>,
) -> Result<Vec<AnimeStatisticsInput>, Status> {
    statistics
        .into_iter()
        .map(
            |(
                provider,
                Statistics {
                    score,
                    scored_by,
                    rank,
                    popularity,
                    members,
                    favorites,
                    contextual_ranks,
                },
            )| {
                let mapped_ranks = contextual_ranks
                    .into_iter()
                    .map(map_contextual_rank)
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(AnimeStatisticsInput {
                    provider,
                    score: score.map(f64::from),
                    scored_by: opt_i64_to_i32(scored_by, "statistics.scored_by")?,
                    members: opt_i64_to_i32(members, "statistics.members")?,
                    favorites: opt_i64_to_i32(favorites, "statistics.favorites")?,
                    rank: opt_i64_to_i32(rank, "statistics.rank")?,
                    popularity: opt_i64_to_i32(popularity, "statistics.popularity")?,
                    contextual_ranks: mapped_ranks,
                })
            },
        )
        .collect()
}

fn map_contextual_rank(rank_entry: ContextualRank) -> Result<AnimeContextualRankInput, Status> {
    let ContextualRank {
        rank,
        context,
        format,
        year,
        season,
        all_time,
    } = rank_entry;
    let rank_year = year
        .map(|value| {
            i16::try_from(value)
                .map_err(|_| Status::invalid_argument("contextual_rank.year overflows i16"))
        })
        .transpose()?;
    let rank_season = season
        .as_deref()
        .map(normalize_season)
        .transpose()?
        .map(str::to_owned);

    Ok(AnimeContextualRankInput {
        context,
        rank,
        is_all_time: all_time,
        format,
        season: rank_season,
        year: rank_year,
    })
}

fn normalize_season(season_text: &str) -> Result<&'static str, Status> {
    match season_text.trim().to_ascii_uppercase().as_str() {
        "SPRING" => Ok("SPRING"),
        "SUMMER" => Ok("SUMMER"),
        "FALL" | "AUTUMN" => Ok("FALL"),
        "WINTER" => Ok("WINTER"),
        _ => Err(Status::invalid_argument(format!(
            "unknown contextual_rank season {season_text:?}"
        ))),
    }
}

fn map_broadcast(broadcast: Broadcast) -> Result<AnimeBroadcastInput, Status> {
    let Broadcast {
        day,
        time,
        timezone,
        jp_time,
        sub_time,
        dub_time,
        sub_delay_days,
        dub_delay_days,
        premiere_jp,
        premiere_sub,
        premiere_dub,
        next_episode_at,
        next_episode_number,
    } = broadcast;

    Ok(AnimeBroadcastInput {
        weekday: day.as_deref().map(parse_iso_weekday).transpose()?,
        timezone,
        jp_time: jp_time
            .or(time)
            .as_deref()
            .map(parse_time_of_day)
            .transpose()?,
        sub_time: sub_time.as_deref().map(parse_time_of_day).transpose()?,
        dub_time: dub_time.as_deref().map(parse_time_of_day).transpose()?,
        sub_delay_days,
        dub_delay_days,
        premiere_jp: premiere_jp
            .map(|ts| timestamp_to_utc(ts, "broadcast.premiere_jp"))
            .transpose()?,
        premiere_sub: premiere_sub
            .map(|ts| timestamp_to_utc(ts, "broadcast.premiere_sub"))
            .transpose()?,
        premiere_dub: premiere_dub
            .map(|ts| timestamp_to_utc(ts, "broadcast.premiere_dub"))
            .transpose()?,
        next_episode_at: next_episode_at
            .map(|ts| timestamp_to_utc(ts, "broadcast.next_episode_at"))
            .transpose()?,
        next_episode_number,
    })
}

fn map_hiatus(hiatus: AnimeHiatus) -> Result<AnimeHiatusInput, Status> {
    let AnimeHiatus {
        reason,
        hiatus_from,
        hiatus_until,
    } = hiatus;

    Ok(AnimeHiatusInput {
        reason,
        hiatus_from: hiatus_from
            .as_deref()
            .map(|value| parse_iso_date(value, "hiatus.hiatus_from"))
            .transpose()?,
        hiatus_until: hiatus_until
            .as_deref()
            .map(|value| parse_iso_date(value, "hiatus.hiatus_until"))
            .transpose()?,
    })
}

fn parse_iso_weekday(day_text: &str) -> Result<i16, Status> {
    match day_text.trim().to_ascii_lowercase().as_str() {
        "monday" | "mondays" | "1" => Ok(1),
        "tuesday" | "tuesdays" | "2" => Ok(2),
        "wednesday" | "wednesdays" | "3" => Ok(3),
        "thursday" | "thursdays" | "4" => Ok(4),
        "friday" | "fridays" | "5" => Ok(5),
        "saturday" | "saturdays" | "6" => Ok(6),
        "sunday" | "sundays" | "7" => Ok(7),
        _ => Err(Status::invalid_argument(format!(
            "unknown broadcast day {day_text:?}"
        ))),
    }
}

fn parse_time_of_day(time_text: &str) -> Result<NaiveTime, Status> {
    let trimmed = time_text.trim();
    if let Ok(clock_time) = NaiveTime::parse_from_str(trimmed, "%H:%M:%S")
        .or_else(|_| NaiveTime::parse_from_str(trimmed, "%H:%M"))
    {
        return Ok(clock_time);
    }
    if let Ok(fixed_dt) = DateTime::parse_from_rfc3339(trimmed) {
        return Ok(fixed_dt.time());
    }
    Err(Status::invalid_argument(format!(
        "invalid time {time_text:?}"
    )))
}

fn parse_iso_date(date_text: &str, field_name: &str) -> Result<NaiveDate, Status> {
    NaiveDate::parse_from_str(date_text.trim(), "%Y-%m-%d").map_err(|err| {
        Status::invalid_argument(format!("invalid {field_name} date {date_text:?}: {err}"))
    })
}

fn opt_i64_to_i32(value: Option<i64>, field_name: &str) -> Result<Option<i32>, Status> {
    value
        .map(|number| {
            i32::try_from(number)
                .map_err(|_| Status::invalid_argument(format!("{field_name} overflows i32")))
        })
        .transpose()
}

fn timestamp_to_utc(ts: Timestamp, field_name: &str) -> Result<DateTime<Utc>, Status> {
    let nanos = u32::try_from(ts.nanos)
        .map_err(|_| Status::invalid_argument(format!("{field_name} has negative nanos")))?;
    DateTime::<Utc>::from_timestamp(ts.seconds, nanos)
        .ok_or_else(|| Status::invalid_argument(format!("{field_name} is out of range")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_schedule_and_stats_maps_all_extended_fields() {
        let mut stats = BTreeMap::new();
        stats.insert(
            "mal".to_owned(),
            Statistics {
                score: Some(8.75),
                scored_by: Some(12345),
                rank: Some(10),
                popularity: Some(50),
                members: Some(50000),
                favorites: Some(2000),
                contextual_ranks: vec![ContextualRank {
                    rank: 1,
                    context: "top".to_owned(),
                    format: Some("TV".to_owned()),
                    year: Some(2024),
                    season: Some("spring".to_owned()),
                    all_time: Some(false),
                }],
            },
        );

        let Ok(mapped) = build_schedule_and_stats(
            vec![ThemeSong {
                title: "OP1".to_owned(),
                artist: Some("Artist".to_owned()),
                episodes: vec![EpisodeRange {
                    start: 1,
                    end: Some(12),
                }],
            }],
            Vec::new(),
            vec![
                TrailerEntry {
                    source: Some("https://youtube.com/watch?v=1".to_owned()),
                    title: Some("PV".to_owned()),
                    thumbnail: None,
                },
                TrailerEntry {
                    source: None,
                    title: Some("Incomplete".to_owned()),
                    thumbnail: Some("https://example.com/thumb.jpg".to_owned()),
                },
            ],
            vec![ExternalLink {
                platform: "Official".to_owned(),
                source: "https://example.com".to_owned(),
                label: None,
                language: Some("ja".to_owned()),
            }],
            vec![StreamingEntry {
                platform: "Crunchyroll".to_owned(),
                source: "https://crunchyroll.com/1".to_owned(),
                region: Some("US".to_owned()),
                free: Some(false),
                premium_required: Some(true),
                dub_available: Some(true),
                subtitle_languages: vec!["en".to_owned()],
            }],
            stats,
            Some(Broadcast {
                day: Some("Wednesdays".to_owned()),
                time: Some("00:30".to_owned()),
                timezone: Some("Asia/Tokyo".to_owned()),
                jp_time: Some("2026-11-18T23:15:00+09:00".to_owned()),
                sub_time: Some("01:30:00".to_owned()),
                dub_time: Some("2026-11-18T15:30:00Z".to_owned()),
                sub_delay_days: Some(0),
                dub_delay_days: None,
                premiere_jp: None,
                premiere_sub: None,
                premiere_dub: None,
                next_episode_at: None,
                next_episode_number: Some(5),
            }),
            Some(AnimeHiatus {
                reason: Some("Break".to_owned()),
                hiatus_from: Some("2024-05-01".to_owned()),
                hiatus_until: None,
            }),
        ) else {
            panic!("valid extended fields should map");
        };

        assert_eq!(mapped.theme_songs.len(), 1);
        assert_eq!(mapped.theme_songs[0].kind, "opening");
        assert_eq!(mapped.trailers.len(), 1);
        assert_eq!(mapped.external_links.len(), 1);
        assert_eq!(mapped.streaming_entries.len(), 1);
        assert_eq!(
            mapped.statistics[0].contextual_ranks[0].season.as_deref(),
            Some("SPRING")
        );
        assert_eq!(mapped.broadcast.as_ref().and_then(|b| b.weekday), Some(3));
        assert!(mapped.hiatus.as_ref().and_then(|h| h.hiatus_from).is_some());
    }
}
