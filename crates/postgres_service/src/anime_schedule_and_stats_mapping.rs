//! Protobuf-to-SQL mapping for opening and ending themes, trailers, external sources,
//! streaming sources, statistics, broadcast schedule, and hiatus snapshot.

use std::collections::BTreeMap;

use chrono::{DateTime, NaiveDate, NaiveTime, Utc};
use postgres_db::{
    AnimeBroadcastInput, AnimeContextualRankInput, AnimeExternalSourceInput, AnimeHiatusInput,
    AnimeScheduleAndStatsInput, AnimeStatisticsInput, AnimeStreamingSourceInput,
    AnimeThemeSongInput, AnimeTrailerInput, EpisodeRangeInput,
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
/// Collects a work's songs, trailers, links, streaming, statistics, broadcast and hiatus.
///
/// # Errors
///
/// Returns `INVALID_ARGUMENT` for an invalid season, weekday, time, date, timestamp or count.
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
    let mapped_opening_themes = map_theme_songs(opening_themes);
    let mapped_ending_themes = map_theme_songs(ending_themes);
    let mapped_trailers = map_trailers(trailers);
    let mapped_external_sources = map_external_sources(external_sources);
    let mapped_streaming_sources = map_streaming_sources(streaming_sources);
    let mapped_statistics = map_statistics(statistics)?;
    let mapped_broadcast = broadcast.map(map_broadcast).transpose()?;
    let mapped_hiatus = hiatus.map(map_hiatus).transpose()?;

    Ok(AnimeScheduleAndStatsInput {
        opening_themes: mapped_opening_themes,
        ending_themes: mapped_ending_themes,
        trailers: mapped_trailers,
        external_sources: mapped_external_sources,
        streaming_sources: mapped_streaming_sources,
        statistics: mapped_statistics,
        broadcast: mapped_broadcast,
        hiatus: mapped_hiatus,
    })
}

/// Converts opening or ending songs into theme song inputs.
///
/// Each song's position is its place in the list, counted from 0.
fn map_theme_songs(songs: Vec<ThemeSong>) -> Vec<AnimeThemeSongInput> {
    songs
        .into_iter()
        .zip(0_i32..)
        .map(
            |(
                ThemeSong {
                    title,
                    artist,
                    episodes,
                },
                position,
            )| {
                let episode_ranges = episodes
                    .into_iter()
                    .map(
                        |EpisodeRange {
                             range_start,
                             range_end,
                         }| EpisodeRangeInput {
                            range_start,
                            range_end,
                        },
                    )
                    .collect();
                AnimeThemeSongInput {
                    position,
                    title,
                    artist,
                    episode_ranges,
                }
            },
        )
        .collect()
}

/// Converts trailer entries into trailer inputs.
fn map_trailers(trailers: Vec<TrailerEntry>) -> Vec<AnimeTrailerInput> {
    trailers
        .into_iter()
        .filter_map(
            |TrailerEntry {
                 source,
                 title,
                 thumbnail_source,
             }| {
                let source = source.filter(|value| !value.trim().is_empty())?;
                Some(AnimeTrailerInput {
                    source,
                    title,
                    thumbnail_source,
                })
            },
        )
        .collect()
}

/// Converts external links into external source inputs.
fn map_external_sources(links: Vec<ExternalLink>) -> Vec<AnimeExternalSourceInput> {
    links
        .into_iter()
        .map(
            |ExternalLink {
                 platform,
                 source,
                 label,
                 language,
             }| AnimeExternalSourceInput {
                source,
                platform,
                label,
                language,
            },
        )
        .collect()
}

/// Converts streaming entries into streaming source inputs with their subtitle languages.
fn map_streaming_sources(entries: Vec<StreamingEntry>) -> Vec<AnimeStreamingSourceInput> {
    entries
        .into_iter()
        .map(
            |StreamingEntry {
                 platform,
                 source,
                 region,
                 is_free,
                 is_premium_required,
                 has_dub,
                 subtitle_languages,
             }| AnimeStreamingSourceInput {
                platform,
                source,
                region,
                is_free,
                is_premium_required,
                has_dub,
                subtitle_languages,
            },
        )
        .collect()
}

/// Converts each provider's statistics into statistics inputs with contextual ranks.
///
/// # Errors
///
/// Returns `INVALID_ARGUMENT` for a count that does not fit in an `i32`, or a contextual rank
/// with a year that does not fit in an `i16` or an unknown season.
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

/// Converts one contextual rank into its input.
///
/// # Errors
///
/// Returns `INVALID_ARGUMENT` for a year that does not fit in an `i16`, or an unknown season.
fn map_contextual_rank(rank_entry: ContextualRank) -> Result<AnimeContextualRankInput, Status> {
    let ContextualRank {
        rank,
        context,
        format,
        year,
        season,
        is_all_time,
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
        is_all_time,
        format,
        season: rank_season,
        year: rank_year,
    })
}

/// Converts a season name into its lookup text.
///
/// # Errors
///
/// Returns `INVALID_ARGUMENT` for a name that is not a season.
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

/// Converts the broadcast slot into its input.
///
/// # Errors
///
/// Returns `INVALID_ARGUMENT` for an invalid weekday, time or timestamp.
fn map_broadcast(broadcast: Broadcast) -> Result<AnimeBroadcastInput, Status> {
    let Broadcast {
        day,
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
        weekday: day.map(validate_iso_weekday).transpose()?,
        timezone,
        jp_time: jp_time.as_deref().map(parse_time_of_day).transpose()?,
        sub_time: sub_time.as_deref().map(parse_time_of_day).transpose()?,
        dub_time: dub_time.as_deref().map(parse_time_of_day).transpose()?,
        sub_delay_days,
        dub_delay_days,
        premiere_jp: premiere_jp
            .map(|proto_timestamp| timestamp_to_utc(proto_timestamp, "broadcast.premiere_jp"))
            .transpose()?,
        premiere_sub: premiere_sub
            .map(|proto_timestamp| timestamp_to_utc(proto_timestamp, "broadcast.premiere_sub"))
            .transpose()?,
        premiere_dub: premiere_dub
            .map(|proto_timestamp| timestamp_to_utc(proto_timestamp, "broadcast.premiere_dub"))
            .transpose()?,
        next_episode_at: next_episode_at
            .map(|proto_timestamp| timestamp_to_utc(proto_timestamp, "broadcast.next_episode_at"))
            .transpose()?,
        next_episode_number,
    })
}

/// Converts the hiatus snapshot into its input.
///
/// # Errors
///
/// Returns `INVALID_ARGUMENT` for a date that is not ISO 8601.
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

/// Validates that a weekday integer is in the ISO range 1..=7.
///
/// # Errors
///
/// Returns `INVALID_ARGUMENT` for an integer outside 1..=7.
fn validate_iso_weekday(weekday: i32) -> Result<i16, Status> {
    i16::try_from(weekday)
        .ok()
        .filter(|value| (1..=7).contains(value))
        .ok_or_else(|| Status::invalid_argument(format!("unknown broadcast day {weekday}")))
}

/// Converts `HH:MM`, `HH:MM:SS` or RFC 3339 text into a time of day.
///
/// # Errors
///
/// Returns `INVALID_ARGUMENT` for text in none of those forms.
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

/// Converts `YYYY-MM-DD` text into a date.
///
/// # Errors
///
/// Returns `INVALID_ARGUMENT` naming `field_name` for text that is not a date.
fn parse_iso_date(date_text: &str, field_name: &str) -> Result<NaiveDate, Status> {
    NaiveDate::parse_from_str(date_text.trim(), "%Y-%m-%d").map_err(|err| {
        Status::invalid_argument(format!("invalid {field_name} date {date_text:?}: {err}"))
    })
}

/// Converts an optional 64-bit count into a 32-bit one.
///
/// # Errors
///
/// Returns `INVALID_ARGUMENT` naming `field_name` when the count does not fit.
fn opt_i64_to_i32(value: Option<i64>, field_name: &str) -> Result<Option<i32>, Status> {
    value
        .map(|number| {
            i32::try_from(number)
                .map_err(|_| Status::invalid_argument(format!("{field_name} overflows i32")))
        })
        .transpose()
}

/// Converts a protobuf `Timestamp` into a UTC timestamp.
///
/// # Errors
///
/// Returns `INVALID_ARGUMENT` naming `field_name` when the time is out of range.
fn timestamp_to_utc(proto_timestamp: Timestamp, field_name: &str) -> Result<DateTime<Utc>, Status> {
    let nanos = u32::try_from(proto_timestamp.nanos)
        .map_err(|_| Status::invalid_argument(format!("{field_name} has negative nanos")))?;
    DateTime::<Utc>::from_timestamp(proto_timestamp.seconds, nanos)
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
                    is_all_time: Some(false),
                }],
            },
        );

        let mapped = build_schedule_and_stats(
            vec![ThemeSong {
                title: "OP1".to_owned(),
                artist: Some("Artist".to_owned()),
                episodes: vec![EpisodeRange {
                    range_start: 1,
                    range_end: Some(12),
                }],
            }],
            Vec::new(),
            vec![
                TrailerEntry {
                    source: Some("https://youtube.com/watch?v=1".to_owned()),
                    title: Some("PV".to_owned()),
                    thumbnail_source: None,
                },
                TrailerEntry {
                    source: None,
                    title: Some("Incomplete".to_owned()),
                    thumbnail_source: Some("https://example.com/thumb.jpg".to_owned()),
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
                is_free: Some(false),
                is_premium_required: Some(true),
                has_dub: Some(true),
                subtitle_languages: vec!["en".to_owned()],
            }],
            stats,
            Some(Broadcast {
                day: Some(3),
                timezone: Some("Asia/Tokyo".to_owned()),
                jp_time: Some("2026-11-18T23:15:00+09:00".to_owned()),
                sub_time: Some("01:30:00".to_owned()),
                dub_time: Some("2026-11-18T15:30:00Z".to_owned()),
                sub_delay_days: Some(0),
                premiere_jp: Some(build_timestamp(1_700_000_000)),
                premiere_sub: Some(build_timestamp(1_700_003_600)),
                premiere_dub: Some(build_timestamp(1_700_007_200)),
                next_episode_at: Some(build_timestamp(1_700_604_800)),
                next_episode_number: Some(5),
                ..build_broadcast()
            }),
            Some(AnimeHiatus {
                reason: Some("Break".to_owned()),
                hiatus_from: Some("2024-05-01".to_owned()),
                hiatus_until: Some("2024-05-15".to_owned()),
            }),
        )
        .expect("valid extended fields should map");

        assert_eq!(mapped.opening_themes.len(), 1);
        assert_eq!(mapped.ending_themes, Vec::new());
        assert_eq!(mapped.trailers.len(), 1);
        assert_eq!(mapped.external_sources.len(), 1);
        assert_eq!(mapped.streaming_sources.len(), 1);
        assert_eq!(
            mapped.statistics[0].contextual_ranks[0].season.as_deref(),
            Some("SPRING")
        );
        let mapped_broadcast = mapped.broadcast.expect("broadcast maps");
        assert_eq!(mapped_broadcast.weekday, Some(3));
        assert_eq!(
            mapped_broadcast
                .next_episode_at
                .map(|aired_at| aired_at.timestamp()),
            Some(1_700_604_800)
        );
        assert!(mapped_broadcast.premiere_dub.is_some());
        let mapped_hiatus = mapped.hiatus.expect("hiatus maps");
        assert!(mapped_hiatus.hiatus_from.is_some());
        assert!(mapped_hiatus.hiatus_until.is_some());
    }

    fn build_timestamp(seconds: i64) -> Timestamp {
        Timestamp { seconds, nanos: 0 }
    }

    fn build_broadcast() -> Broadcast {
        Broadcast {
            day: Some(7),
            timezone: None,
            jp_time: Some("00:30".to_owned()),
            sub_time: None,
            dub_time: None,
            sub_delay_days: None,
            dub_delay_days: None,
            premiere_jp: None,
            premiere_sub: None,
            premiere_dub: None,
            next_episode_at: None,
            next_episode_number: None,
        }
    }

    fn build_statistics(rank_year: Option<i32>, rank_season: Option<&str>) -> Statistics {
        Statistics {
            score: None,
            scored_by: None,
            rank: None,
            popularity: None,
            members: None,
            favorites: None,
            contextual_ranks: vec![ContextualRank {
                rank: 1,
                context: "top".to_owned(),
                format: None,
                year: rank_year,
                season: rank_season.map(str::to_owned),
                is_all_time: None,
            }],
        }
    }

    fn map_schedule_parts(
        statistics: Option<Statistics>,
        broadcast: Option<Broadcast>,
        hiatus: Option<AnimeHiatus>,
    ) -> Result<AnimeScheduleAndStatsInput, Status> {
        build_schedule_and_stats(
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            statistics
                .map(|provider_statistics| {
                    BTreeMap::from([("mal".to_owned(), provider_statistics)])
                })
                .unwrap_or_default(),
            broadcast,
            hiatus,
        )
    }

    #[test]
    fn build_schedule_and_stats_invalid_field_returns_invalid_argument() {
        let negative_nanos = Timestamp {
            seconds: 0,
            nanos: -1,
        };
        let mut overflowing_count = build_statistics(None, None);
        overflowing_count.scored_by = Some(i64::MAX);
        let invalid_cases = vec![
            map_schedule_parts(Some(build_statistics(Some(100_000), None)), None, None),
            map_schedule_parts(Some(build_statistics(None, Some("monsoon"))), None, None),
            map_schedule_parts(Some(overflowing_count), None, None),
            map_schedule_parts(
                None,
                Some(Broadcast {
                    day: Some(8),
                    ..build_broadcast()
                }),
                None,
            ),
            map_schedule_parts(
                None,
                Some(Broadcast {
                    jp_time: Some("half past noon".to_owned()),
                    ..build_broadcast()
                }),
                None,
            ),
            map_schedule_parts(
                None,
                Some(Broadcast {
                    premiere_jp: Some(negative_nanos),
                    ..build_broadcast()
                }),
                None,
            ),
            map_schedule_parts(
                None,
                Some(Broadcast {
                    premiere_sub: Some(negative_nanos),
                    ..build_broadcast()
                }),
                None,
            ),
            map_schedule_parts(
                None,
                Some(Broadcast {
                    premiere_dub: Some(negative_nanos),
                    ..build_broadcast()
                }),
                None,
            ),
            map_schedule_parts(
                None,
                Some(Broadcast {
                    next_episode_at: Some(build_timestamp(i64::MAX)),
                    ..build_broadcast()
                }),
                None,
            ),
            map_schedule_parts(
                None,
                None,
                Some(AnimeHiatus {
                    reason: None,
                    hiatus_from: Some("2024-13-01".to_owned()),
                    hiatus_until: None,
                }),
            ),
            map_schedule_parts(
                None,
                None,
                Some(AnimeHiatus {
                    reason: None,
                    hiatus_from: None,
                    hiatus_until: Some("soon".to_owned()),
                }),
            ),
        ];

        for mapping_result in invalid_cases {
            let status = mapping_result.expect_err("an invalid field must be refused");
            assert_eq!(status.code(), tonic::Code::InvalidArgument);
        }
    }

    #[test]
    fn validate_iso_weekday_accepts_one_through_seven() {
        for iso_weekday in 1..=7 {
            assert_eq!(
                validate_iso_weekday(iso_weekday).expect("1..=7 is valid"),
                i16::try_from(iso_weekday).expect("fits in i16")
            );
        }
        assert!(validate_iso_weekday(0).is_err());
        assert!(validate_iso_weekday(8).is_err());
    }

    #[test]
    fn normalize_season_season_names_return_lookup_text() {
        for (season_name, lookup_text) in [
            ("spring", "SPRING"),
            ("Summer", "SUMMER"),
            ("fall", "FALL"),
            ("autumn", "FALL"),
            ("WINTER", "WINTER"),
        ] {
            assert_eq!(
                normalize_season(season_name).expect(season_name),
                lookup_text
            );
        }
    }
}
