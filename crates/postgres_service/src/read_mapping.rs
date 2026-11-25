//! SQL-to-protobuf mapping helpers for visible entity reads (`GetAnime`, `GetEpisode`, `GetCharacter`).

use std::collections::BTreeMap;

use postgres_db::{
    AnimeBroadcastInput, AnimeContextualRankInput, AnimeExternalSourceInput, AnimeHiatusInput,
    AnimeStatisticsInput, AnimeStreamingSourceInput, AnimeThemeSongInput, AnimeTrailerInput,
    PageKey, VisibleAnimeCompany, VisibleAnimeRecord, VisibleAnimeRelation, VisibleCharacterRecord,
    VisibleEpisodeRecord, VisibleSourceMaterialRelation,
};

use crate::service::postgres_service_proto::v1::{
    GetAnimeResponse, GetCharacterResponse, GetEpisodeResponse, ListEpisodesResponse,
};
use crate::service::shared_proto::v1::{
    AiredDates, Anime, AnimeHiatus, AnimeImages, AnimeRating, AnimeSeason, AnimeStatus, AnimeType,
    Broadcast, Character, CharacterRole, CompanyEntry, CompanyRole, ContextualRank,
    EntityType as ProtoEntityType, Episode, EpisodeCharacter, EpisodeRange, ExternalLink, Ography,
    RelatedAnime, RelatedAnimeList, RelatedSourceMaterial, RelatedSourceMaterialList,
    ScoreCalculations, SourceMaterialType, Statistics, StreamingEntry, ThemeEntry, ThemeSong,
    TrailerEntry,
};
use crate::service::to_proto_timestamp;

/// Converts a stored `VisibleAnimeRecord` into a [`GetAnimeResponse`].
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    reason = "proto float fields are f32 scores"
)]
pub(crate) fn to_get_anime_response(record: VisibleAnimeRecord) -> GetAnimeResponse {
    let anime_type =
        AnimeType::from_str_name(&record.anime_type).unwrap_or(AnimeType::Unspecified) as i32;
    let status =
        AnimeStatus::from_str_name(&record.status).unwrap_or(AnimeStatus::Unspecified) as i32;
    let rating = AnimeRating::from_str_name(&record.rating).map(|value| value as i32);
    let season = record
        .season
        .as_deref()
        .and_then(AnimeSeason::from_str_name)
        .map(|value| value as i32);
    let source_material = record
        .source_material
        .as_deref()
        .and_then(SourceMaterialType::from_str_name)
        .map(|value| value as i32);

    let titles = record
        .titles
        .into_iter()
        .map(|entry| (entry.language, entry.title))
        .collect();
    let themes = record
        .themes
        .into_iter()
        .map(|entry| ThemeEntry {
            name: entry.name,
            description: entry.description,
        })
        .collect();

    let images = map_anime_images(record.images);

    let related_anime = group_related_anime(record.related_anime);
    let related_source_material = group_related_source_material(record.related_source_material);
    let statistics = map_statistics(record.statistics);

    let aired_dates =
        (record.aired_from.is_some() || record.aired_to.is_some()).then(|| AiredDates {
            aired_from: record.aired_from.map(to_proto_timestamp),
            aired_to: record.aired_to.map(to_proto_timestamp),
        });

    let score = (record.score_weighted.is_some()
        || record.score_mean.is_some()
        || record.score_median.is_some())
    .then(|| ScoreCalculations {
        weighted: record.score_weighted.map(|val| val as f32),
        mean: record.score_mean.map(|val| val as f32),
        median: record.score_median.map(|val| val as f32),
    });

    let anime = Anime {
        id: record.anime_id.to_string(),
        entity_type: ProtoEntityType::Anime as i32,
        title: record.title,
        title_english: record.title_english,
        title_japanese: record.title_japanese,
        r#type: anime_type,
        status,
        rating,
        season,
        source_material,
        year: record.year.map(i32::from),
        month: record.month.map(i32::from),
        episode_count: record.episode_count,
        duration_seconds: record.duration_seconds,
        is_nsfw: record.is_nsfw,
        country_of_origin: record.country_of_origin,
        synopsis: record.synopsis,
        background: record.background,
        titles,
        synonyms: record.synonyms,
        genres: record.genres,
        themes,
        demographics: record.demographics,
        content_warnings: record.content_warnings,
        tags: record.tags,
        sources: format_page_sources(&record.source_pages),
        partial_sources: format_page_sources(&record.partial_source_pages),
        images: Some(images),
        opening_themes: map_theme_songs(record.opening_themes),
        ending_themes: map_theme_songs(record.ending_themes),
        trailers: map_trailers(record.trailers),
        external_sources: map_external_sources(record.external_sources),
        streaming_sources: map_streaming_sources(record.streaming_sources),
        statistics,
        broadcast: record.broadcast.map(map_broadcast),
        hiatus: record.hiatus.map(map_hiatus),
        aired_dates,
        score,
        related_anime,
        related_source_material,
        companies: map_companies(record.companies),
        production_staff: None,
    };

    GetAnimeResponse {
        anime: Some(anime),
        version: record.version,
        next_refresh_at: record.next_refresh_at.map(to_proto_timestamp),
    }
}

/// Converts a stored `VisibleEpisodeRecord` into a [`GetEpisodeResponse`].
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    reason = "proto float fields are f32 scores"
)]
pub(crate) fn to_get_episode_response(record: VisibleEpisodeRecord) -> GetEpisodeResponse {
    let titles = record
        .titles
        .into_iter()
        .map(|entry| (entry.language, entry.title))
        .collect();
    let streaming = record
        .streaming
        .into_iter()
        .map(|entry| (entry.platform, entry.source))
        .collect();
    let characters = record
        .characters
        .into_iter()
        .map(|entry| EpisodeCharacter {
            name: String::new(),
            role: CharacterRole::from_str_name(&entry.role).unwrap_or(CharacterRole::Unspecified)
                as i32,
            sources: vec![format!(
                "{}:{}",
                entry.character_page_provider, entry.character_page_address
            )],
            voice_actors: Vec::new(),
        })
        .collect();

    let episode = Episode {
        aired_at: record.aired_at.map(to_proto_timestamp),
        anime_id: Some(record.anime_id.to_string()),
        description: record.description,
        duration_seconds: record.duration_seconds,
        episode_number: record.episode_number,
        is_filler: record.is_filler,
        id: Some(record.episode_id.to_string()),
        entity_type: ProtoEntityType::Episode as i32,
        is_recap: record.is_recap,
        score: record.score.map(|val| val as f32),
        season_number: record.season_number,
        synopsis: record.synopsis,
        title: record.title,
        title_japanese: record.title_japanese,
        title_romaji: record.title_romaji,
        images: record.images,
        streaming,
        sources: format_page_sources(&record.source_pages),
        characters,
        staff: Vec::new(),
        titles,
    };

    GetEpisodeResponse {
        episode: Some(episode),
        version: record.version,
        next_refresh_at: record.next_refresh_at.map(to_proto_timestamp),
    }
}

/// Converts visible episode records into a [`ListEpisodesResponse`].
#[must_use]
pub(crate) fn to_list_episodes_response(
    records: Vec<VisibleEpisodeRecord>,
) -> ListEpisodesResponse {
    ListEpisodesResponse {
        episodes: records.into_iter().map(to_get_episode_response).collect(),
    }
}

/// Converts a stored `VisibleCharacterRecord` into a [`GetCharacterResponse`].
#[must_use]
pub(crate) fn to_get_character_response(record: VisibleCharacterRecord) -> GetCharacterResponse {
    let mut seen_roles = Vec::new();
    let mut anime_ids = Vec::new();
    for (anime_id, role_str) in record.anime_ids {
        anime_ids.push(anime_id.to_string());
        let role =
            CharacterRole::from_str_name(&role_str).unwrap_or(CharacterRole::Unspecified) as i32;
        if !seen_roles.contains(&role) {
            seen_roles.push(role);
        }
    }

    let animeography = record
        .animeography
        .into_iter()
        .map(|(provider, address, role)| Ography {
            title: String::new(),
            role: CharacterRole::from_str_name(&role).unwrap_or(CharacterRole::Unspecified) as i32,
            sources: vec![format!("{provider}:{address}")],
        })
        .collect();

    let mangaography = record
        .mangaography
        .into_iter()
        .map(|(provider, external_id, role)| Ography {
            title: String::new(),
            role: CharacterRole::from_str_name(&role).unwrap_or(CharacterRole::Unspecified) as i32,
            sources: vec![format!("{provider}:{external_id}")],
        })
        .collect();

    let character = Character {
        description: record.description,
        favorites: record.favorites.map(i64::from),
        id: Some(record.character_id.to_string()),
        entity_type: ProtoEntityType::Character as i32,
        name: record.name,
        name_native: record.name_native,
        roles: seen_roles,
        anime_ids,
        animeography,
        traits: record.traits,
        images: record.images,
        mangaography,
        name_variations: record.name_variations,
        nicknames: record.nicknames,
        sources: format_page_sources(&record.source_pages),
        voice_actors: Vec::new(),
        attributes: record.attributes,
        spoilers: record.spoilers,
    };

    GetCharacterResponse {
        character: Some(character),
        version: record.version,
        next_refresh_at: record.next_refresh_at.map(to_proto_timestamp),
    }
}

fn format_page_sources(pages: &[PageKey]) -> Vec<String> {
    pages
        .iter()
        .map(|page| format!("{}:{}", page.provider().as_str(), page.address().as_str()))
        .collect()
}

fn map_anime_images(entries: Vec<postgres_db::AnimeImageInput>) -> AnimeImages {
    let mut images = AnimeImages::default();
    for image in entries {
        match image.image_type.as_str() {
            "cover" => images.covers.push(image.url),
            "poster" => images.posters.push(image.url),
            "banner" => images.banners.push(image.url),
            _ => {}
        }
    }
    images
}

fn map_theme_songs(songs: Vec<AnimeThemeSongInput>) -> Vec<ThemeSong> {
    songs
        .into_iter()
        .map(|song| ThemeSong {
            title: song.title,
            artist: song.artist,
            episodes: song
                .episode_ranges
                .into_iter()
                .map(|range| EpisodeRange {
                    range_start: range.range_start,
                    range_end: range.range_end,
                })
                .collect(),
        })
        .collect()
}

fn map_trailers(trailers: Vec<AnimeTrailerInput>) -> Vec<TrailerEntry> {
    trailers
        .into_iter()
        .map(|trailer| TrailerEntry {
            title: trailer.title,
            thumbnail_source: trailer.thumbnail_source,
            source: Some(trailer.source),
        })
        .collect()
}

fn map_external_sources(sources: Vec<AnimeExternalSourceInput>) -> Vec<ExternalLink> {
    sources
        .into_iter()
        .map(|source| ExternalLink {
            platform: source.platform,
            source: source.source,
            label: source.label,
            language: source.language,
        })
        .collect()
}

fn map_streaming_sources(sources: Vec<AnimeStreamingSourceInput>) -> Vec<StreamingEntry> {
    sources
        .into_iter()
        .map(|source| StreamingEntry {
            platform: source.platform,
            source: source.source,
            region: source.region,
            is_free: source.is_free,
            is_premium_required: source.is_premium_required,
            has_dub: source.has_dub,
            subtitle_languages: source.subtitle_languages,
        })
        .collect()
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "proto float fields are f32 scores"
)]
fn map_statistics(stats: Vec<AnimeStatisticsInput>) -> BTreeMap<String, Statistics> {
    stats
        .into_iter()
        .map(|stat| {
            let contextual_ranks = stat
                .contextual_ranks
                .into_iter()
                .map(map_contextual_rank)
                .collect();
            (
                stat.provider,
                Statistics {
                    score: stat.score.map(|val| val as f32),
                    scored_by: stat.scored_by.map(i64::from),
                    members: stat.members.map(i64::from),
                    favorites: stat.favorites.map(i64::from),
                    rank: stat.rank.map(i64::from),
                    popularity: stat.popularity.map(i64::from),
                    contextual_ranks,
                },
            )
        })
        .collect()
}

fn map_contextual_rank(rank: AnimeContextualRankInput) -> ContextualRank {
    ContextualRank {
        context: rank.context,
        rank: rank.rank,
        is_all_time: rank.is_all_time,
        format: rank.format,
        season: rank.season,
        year: rank.year.map(i32::from),
    }
}

fn map_broadcast(broadcast: AnimeBroadcastInput) -> Broadcast {
    Broadcast {
        day: broadcast.weekday.map(i32::from),
        timezone: broadcast.timezone,
        jp_time: broadcast
            .jp_time
            .map(|time| time.format("%H:%M:%S").to_string()),
        sub_time: broadcast
            .sub_time
            .map(|time| time.format("%H:%M:%S").to_string()),
        dub_time: broadcast
            .dub_time
            .map(|time| time.format("%H:%M:%S").to_string()),
        sub_delay_days: broadcast.sub_delay_days,
        dub_delay_days: broadcast.dub_delay_days,
        premiere_jp: broadcast.premiere_jp.map(to_proto_timestamp),
        premiere_sub: broadcast.premiere_sub.map(to_proto_timestamp),
        premiere_dub: broadcast.premiere_dub.map(to_proto_timestamp),
        next_episode_at: broadcast.next_episode_at.map(to_proto_timestamp),
        next_episode_number: broadcast.next_episode_number,
    }
}

fn map_hiatus(hiatus: AnimeHiatusInput) -> AnimeHiatus {
    AnimeHiatus {
        reason: hiatus.reason,
        hiatus_from: hiatus
            .hiatus_from
            .map(|date| date.format("%Y-%m-%d").to_string()),
        hiatus_until: hiatus
            .hiatus_until
            .map(|date| date.format("%Y-%m-%d").to_string()),
    }
}

fn group_related_anime(relations: Vec<VisibleAnimeRelation>) -> BTreeMap<String, RelatedAnimeList> {
    let mut grouped: BTreeMap<String, RelatedAnimeList> = BTreeMap::new();
    for relation in relations {
        grouped
            .entry(relation.relation_type)
            .or_default()
            .items
            .push(RelatedAnime {
                title: relation.target_title.unwrap_or_default(),
                r#type: AnimeType::Unspecified as i32,
                sources: vec![format!(
                    "{}:{}",
                    relation.target_provider, relation.target_address
                )],
                status: None,
                year: None,
                score: None,
                images: Vec::new(),
                episode_count: None,
            });
    }
    grouped
}

fn group_related_source_material(
    relations: Vec<VisibleSourceMaterialRelation>,
) -> BTreeMap<String, RelatedSourceMaterialList> {
    let mut grouped: BTreeMap<String, RelatedSourceMaterialList> = BTreeMap::new();
    for relation in relations {
        let material_type = SourceMaterialType::from_str_name(&relation.source_material_type)
            .unwrap_or(SourceMaterialType::Unspecified) as i32;
        grouped
            .entry(relation.relation_type)
            .or_default()
            .items
            .push(RelatedSourceMaterial {
                title: relation.title,
                r#type: material_type,
                sources: Vec::new(),
                status: None,
                score: None,
                images: Vec::new(),
                chapters: None,
                volumes: None,
            });
    }
    grouped
}

fn map_companies(companies: Vec<VisibleAnimeCompany>) -> Vec<CompanyEntry> {
    let mut by_name: BTreeMap<String, Vec<i32>> = BTreeMap::new();
    for company in companies {
        let role =
            CompanyRole::from_str_name(&company.role).unwrap_or(CompanyRole::Unspecified) as i32;
        by_name.entry(company.name).or_default().push(role);
    }
    by_name
        .into_iter()
        .map(|(name, roles)| CompanyEntry {
            name,
            description: None,
            sources: Vec::new(),
            roles,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use chrono::{NaiveDate, NaiveTime, Utc};
    use postgres_db::{
        AnimeBroadcastInput, AnimeContextualRankInput, AnimeExternalSourceInput, AnimeHiatusInput,
        AnimeImageInput, AnimeStatisticsInput, AnimeStreamingSourceInput, AnimeThemeInput,
        AnimeThemeSongInput, AnimeTitleInput, AnimeTrailerInput, EntityType, EpisodeCharacterInput,
        EpisodeRangeInput, EpisodeStreamingInput, EpisodeTitleInput, PageKey, Provider,
        VisibleAnimeCompany, VisibleAnimeRecord, VisibleAnimeRelation, VisibleCharacterRecord,
        VisibleEpisodeRecord, VisibleSourceMaterialRelation,
    };
    use uuid::Uuid;

    use super::*;

    #[test]
    #[allow(
        clippy::too_many_lines,
        reason = "test exercises every field of the anime, episode, and character read mappers"
    )]
    fn read_mappers_convert_all_anime_episode_and_character_fields() {
        let now = Utc::now();
        let anime_id = Uuid::now_v7();
        let anime_record = VisibleAnimeRecord {
            anime_id,
            version: 3,
            next_refresh_at: Some(now),
            title: "Work".to_owned(),
            title_english: Some("Work EN".to_owned()),
            title_japanese: Some("Work JP".to_owned()),
            anime_type: "TV".to_owned(),
            status: "ONGOING".to_owned(),
            rating: "PG_13".to_owned(),
            season: Some("SPRING".to_owned()),
            source_material: Some("MANGA".to_owned()),
            year: Some(2025),
            month: Some(4),
            episode_count: 12,
            duration_seconds: Some(1440),
            is_nsfw: Some(false),
            country_of_origin: Some("JP".to_owned()),
            synopsis: Some("Synopsis".to_owned()),
            background: Some("Background".to_owned()),
            aired_from: Some(now),
            aired_to: None,
            score_weighted: Some(8.5),
            score_mean: Some(8.4),
            score_median: Some(8.6),
            titles: vec![AnimeTitleInput {
                language: "en".to_owned(),
                title: "Work EN".to_owned(),
            }],
            synonyms: vec!["W".to_owned()],
            genres: vec!["Action".to_owned()],
            themes: vec![AnimeThemeInput {
                name: "Space".to_owned(),
                description: Some("Space".to_owned()),
            }],
            demographics: vec!["Shounen".to_owned()],
            content_warnings: vec!["Violence".to_owned()],
            tags: vec!["Tag".to_owned()],
            source_pages: vec![
                PageKey::new(Provider::Mal, EntityType::Anime, "1").expect("valid key"),
            ],
            partial_source_pages: vec![
                PageKey::new(Provider::AniList, EntityType::Anime, "2").expect("valid key"),
            ],
            images: vec![
                AnimeImageInput {
                    image_type: "cover".to_owned(),
                    url: "https://ex/c.jpg".to_owned(),
                },
                AnimeImageInput {
                    image_type: "poster".to_owned(),
                    url: "https://ex/p.jpg".to_owned(),
                },
                AnimeImageInput {
                    image_type: "banner".to_owned(),
                    url: "https://ex/b.jpg".to_owned(),
                },
                AnimeImageInput {
                    image_type: "other".to_owned(),
                    url: "https://ex/o.jpg".to_owned(),
                },
            ],
            opening_themes: vec![AnimeThemeSongInput {
                position: 0,
                title: "OP".to_owned(),
                artist: Some("Artist".to_owned()),
                episode_ranges: vec![EpisodeRangeInput {
                    range_start: 1,
                    range_end: Some(12),
                }],
            }],
            ending_themes: Vec::new(),
            trailers: vec![AnimeTrailerInput {
                source: "https://yt/1".to_owned(),
                title: Some("PV".to_owned()),
                thumbnail_source: None,
            }],
            external_sources: vec![AnimeExternalSourceInput {
                source: "https://ex.com".to_owned(),
                platform: "Official".to_owned(),
                label: Some("Site".to_owned()),
                language: Some("en".to_owned()),
            }],
            streaming_sources: vec![AnimeStreamingSourceInput {
                platform: "CR".to_owned(),
                source: "https://cr.com".to_owned(),
                region: Some("US".to_owned()),
                is_free: Some(false),
                is_premium_required: Some(true),
                has_dub: Some(true),
                subtitle_languages: vec!["en".to_owned()],
            }],
            statistics: vec![AnimeStatisticsInput {
                provider: "mal".to_owned(),
                score: Some(8.5),
                scored_by: Some(100),
                members: Some(200),
                favorites: Some(50),
                rank: Some(10),
                popularity: Some(20),
                contextual_ranks: vec![AnimeContextualRankInput {
                    context: "Spring".to_owned(),
                    rank: 1,
                    is_all_time: Some(false),
                    format: Some("TV".to_owned()),
                    season: Some("SPRING".to_owned()),
                    year: Some(2025),
                }],
            }],
            broadcast: Some(AnimeBroadcastInput {
                weekday: Some(5),
                timezone: Some("Asia/Tokyo".to_owned()),
                jp_time: NaiveTime::from_hms_opt(12, 0, 0),
                sub_time: NaiveTime::from_hms_opt(13, 0, 0),
                dub_time: NaiveTime::from_hms_opt(14, 0, 0),
                sub_delay_days: Some(0),
                dub_delay_days: Some(7),
                premiere_jp: Some(now),
                premiere_sub: Some(now),
                premiere_dub: None,
                next_episode_at: Some(now),
                next_episode_number: Some(2),
            }),
            hiatus: Some(AnimeHiatusInput {
                reason: Some("Break".to_owned()),
                hiatus_from: NaiveDate::from_ymd_opt(2025, 5, 1),
                hiatus_until: NaiveDate::from_ymd_opt(2025, 5, 8),
            }),
            related_anime: vec![VisibleAnimeRelation {
                relation_type: "SEQUEL".to_owned(),
                target_anime_id: Some(Uuid::now_v7()),
                target_title: Some("Sequel".to_owned()),
                target_provider: "mal".to_owned(),
                target_address: "2".to_owned(),
            }],
            related_source_material: vec![VisibleSourceMaterialRelation {
                relation_type: "ADAPTATION".to_owned(),
                source_material_type: "MANGA".to_owned(),
                title: "Manga".to_owned(),
            }],
            companies: vec![VisibleAnimeCompany {
                name: "Studio".to_owned(),
                role: "STUDIO".to_owned(),
            }],
        };

        let anime_resp = to_get_anime_response(anime_record);
        let anime = anime_resp.anime.expect("anime");
        assert_eq!(anime.images.expect("images").covers.len(), 1);
        assert!(anime.broadcast.is_some());
        assert!(anime.hiatus.is_some());
        assert_eq!(anime.companies.len(), 1);

        let ep_resp = to_get_episode_response(VisibleEpisodeRecord {
            episode_id: Uuid::now_v7(),
            anime_id,
            version: 1,
            next_refresh_at: None,
            episode_number: 1,
            season_number: Some(1),
            title: "Ep 1".to_owned(),
            title_japanese: None,
            title_romaji: None,
            synopsis: None,
            description: None,
            aired_at: Some(now),
            duration_seconds: Some(1440),
            is_filler: false,
            is_recap: false,
            score: Some(8.0),
            titles: vec![EpisodeTitleInput {
                language: "en".to_owned(),
                title: "Ep 1".to_owned(),
            }],
            images: vec!["https://ex/ep1.jpg".to_owned()],
            streaming: vec![EpisodeStreamingInput {
                platform: "CR".to_owned(),
                source: "https://cr.com/ep1".to_owned(),
            }],
            source_pages: vec![PageKey::new(Provider::Mal, EntityType::Episode, "1").expect("key")],
            characters: vec![EpisodeCharacterInput {
                character_page_provider: "mal".to_owned(),
                character_page_address: "10".to_owned(),
                role: "MAIN".to_owned(),
            }],
        });
        assert_eq!(ep_resp.episode.expect("ep").characters.len(), 1);

        let char_resp = to_get_character_response(VisibleCharacterRecord {
            character_id: Uuid::now_v7(),
            version: 1,
            next_refresh_at: None,
            name: "Hero".to_owned(),
            name_native: None,
            description: None,
            favorites: Some(10),
            name_variations: Vec::new(),
            nicknames: Vec::new(),
            traits: Vec::new(),
            images: Vec::new(),
            attributes: BTreeMap::new(),
            spoilers: BTreeMap::new(),
            source_pages: vec![
                PageKey::new(Provider::Mal, EntityType::Character, "10").expect("key"),
            ],
            anime_ids: vec![
                (anime_id, "MAIN".to_owned()),
                (Uuid::now_v7(), "MAIN".to_owned()),
            ],
            animeography: vec![("mal".to_owned(), "1".to_owned(), "MAIN".to_owned())],
            mangaography: vec![("mal".to_owned(), "2".to_owned(), "MAIN".to_owned())],
        });
        let ch = char_resp.character.expect("char");
        assert_eq!(ch.roles.len(), 1);
        assert_eq!(ch.animeography.len(), 1);
        assert_eq!(ch.mangaography.len(), 1);

        let minimal_anime = to_get_anime_response(VisibleAnimeRecord {
            anime_id: Uuid::now_v7(),
            version: 1,
            next_refresh_at: None,
            title: "Minimal".to_owned(),
            title_english: None,
            title_japanese: None,
            anime_type: "MOVIE".to_owned(),
            status: "FINISHED".to_owned(),
            rating: "UNKNOWN".to_owned(),
            season: None,
            source_material: None,
            year: None,
            month: None,
            episode_count: 1,
            duration_seconds: None,
            is_nsfw: None,
            country_of_origin: None,
            synopsis: None,
            background: None,
            aired_from: None,
            aired_to: None,
            score_weighted: None,
            score_mean: None,
            score_median: None,
            titles: Vec::new(),
            synonyms: Vec::new(),
            genres: Vec::new(),
            themes: Vec::new(),
            demographics: Vec::new(),
            content_warnings: Vec::new(),
            tags: Vec::new(),
            source_pages: Vec::new(),
            partial_source_pages: Vec::new(),
            images: Vec::new(),
            opening_themes: Vec::new(),
            ending_themes: Vec::new(),
            trailers: Vec::new(),
            external_sources: Vec::new(),
            streaming_sources: Vec::new(),
            statistics: Vec::new(),
            broadcast: None,
            hiatus: None,
            related_anime: Vec::new(),
            related_source_material: Vec::new(),
            companies: Vec::new(),
        });
        let minimal_msg = minimal_anime.anime.expect("minimal anime");
        assert_eq!(minimal_msg.season, None);
        assert_eq!(minimal_msg.source_material, None);
    }
}
