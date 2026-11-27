//! SQL-to-protobuf mapping for visible entity reads (`GetAnime`, `GetEpisode`, `GetCharacter`).

use std::collections::BTreeMap;

use postgres_db::{
    AnimeBroadcastInput, AnimeCompanyInput, AnimeContextualRankInput, AnimeExternalSourceInput,
    AnimeHiatusInput, AnimeImageInput, AnimeSourceMaterialInput, AnimeStatisticsInput,
    AnimeStreamingSourceInput, AnimeThemeSongInput, AnimeTrailerInput, VisibleAnimeRecord,
    VisibleAnimeRelation, VisibleCharacterRecord, VisibleEpisodeCharacter, VisibleEpisodeRecord,
    VisibleOgraphyEntry,
};
use tonic::Status;

use crate::lookup_text::{
    ANIME_RATING_LOOKUP_TEXTS, ANIME_SEASON_LOOKUP_TEXTS, ANIME_STATUS_LOOKUP_TEXTS,
    ANIME_TYPE_LOOKUP_TEXTS, CHARACTER_ROLE_LOOKUP_TEXTS, COMPANY_ROLE_LOOKUP_TEXTS,
    SOURCE_MATERIAL_TYPE_LOOKUP_TEXTS, read_enum_number,
};
use crate::service::postgres_service_proto::v1::{
    GetAnimeResponse, GetCharacterResponse, GetEpisodeResponse, ListEpisodesResponse,
};
use crate::service::shared_proto::v1::{
    AiredDates, Anime, AnimeHiatus, AnimeImages, Broadcast, Character, CompanyEntry,
    ContextualRank, EntityType as ProtoEntityType, Episode, EpisodeCharacter, EpisodeRange,
    ExternalLink, Ography, RelatedAnime, RelatedAnimeList, RelatedSourceMaterial,
    RelatedSourceMaterialList, ScoreCalculations, Statistics, StreamingEntry, ThemeEntry,
    ThemeSong, TrailerEntry,
};
use crate::service::to_proto_timestamp;

/// Converts a stored work into a [`GetAnimeResponse`].
///
/// # Errors
///
/// Returns `INTERNAL` when a stored lookup value has no proto enum value.
#[expect(
    clippy::cast_possible_truncation,
    reason = "proto score fields are f32, and stored scores hold two decimals between 0 and 10"
)]
pub(crate) fn build_get_anime_response(
    record: VisibleAnimeRecord,
) -> Result<GetAnimeResponse, Status> {
    let anime_type = read_enum_number(ANIME_TYPE_LOOKUP_TEXTS, &record.anime_type, "anime.type")?;
    let status = read_enum_number(ANIME_STATUS_LOOKUP_TEXTS, &record.status, "anime.status")?;
    let rating = read_enum_number(ANIME_RATING_LOOKUP_TEXTS, &record.rating, "anime.rating")?;
    let season = record
        .season
        .as_deref()
        .map(|season| read_enum_number(ANIME_SEASON_LOOKUP_TEXTS, season, "anime.season"))
        .transpose()?;
    let source_material = record
        .source_material
        .as_deref()
        .map(|source_material| {
            read_enum_number(
                SOURCE_MATERIAL_TYPE_LOOKUP_TEXTS,
                source_material,
                "anime.source_material",
            )
        })
        .transpose()?;

    let aired_dates =
        (record.aired_from.is_some() || record.aired_to.is_some()).then(|| AiredDates {
            aired_from: record.aired_from.map(to_proto_timestamp),
            aired_to: record.aired_to.map(to_proto_timestamp),
        });
    let score = (record.score_weighted.is_some()
        || record.score_mean.is_some()
        || record.score_median.is_some())
    .then(|| ScoreCalculations {
        weighted: record.score_weighted.map(|score| score as f32),
        mean: record.score_mean.map(|score| score as f32),
        median: record.score_median.map(|score| score as f32),
    });

    let anime = Anime {
        id: record.anime_id.to_string(),
        entity_type: ProtoEntityType::Anime as i32,
        title: record.title,
        title_english: record.title_english,
        title_japanese: record.title_japanese,
        r#type: anime_type,
        status,
        rating: Some(rating),
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
        titles: record
            .titles
            .into_iter()
            .map(|title| (title.language, title.title))
            .collect(),
        synonyms: record.synonyms,
        genres: record.genres,
        themes: record
            .themes
            .into_iter()
            .map(|theme| ThemeEntry {
                name: theme.name,
                description: theme.description,
            })
            .collect(),
        demographics: record.demographics,
        content_warnings: record.content_warnings,
        tags: record.tags,
        sources: record.sources,
        partial_sources: record.partial_sources,
        images: Some(build_anime_images(record.images)),
        opening_themes: build_theme_songs(record.opening_themes),
        ending_themes: build_theme_songs(record.ending_themes),
        trailers: build_trailers(record.trailers),
        external_sources: build_external_links(record.external_sources),
        streaming_sources: build_streaming_entries(record.streaming_sources),
        statistics: build_statistics(record.statistics),
        broadcast: record.broadcast.map(build_broadcast),
        hiatus: record.hiatus.map(build_hiatus),
        aired_dates,
        score,
        related_anime: build_related_anime(record.related_anime)?,
        related_source_material: build_related_source_material(record.related_source_material)?,
        companies: build_company_entries(record.companies)?,
        production_staff: None,
    };

    Ok(GetAnimeResponse {
        anime: Some(anime),
        version: record.version,
        next_refresh_at: record.next_refresh_at.map(to_proto_timestamp),
    })
}

/// Converts a stored episode into a [`GetEpisodeResponse`].
///
/// # Errors
///
/// Returns `INTERNAL` when a stored character role has no proto enum value.
#[expect(
    clippy::cast_possible_truncation,
    reason = "proto score fields are f32, and stored scores hold two decimals between 0 and 10"
)]
pub(crate) fn build_get_episode_response(
    record: VisibleEpisodeRecord,
) -> Result<GetEpisodeResponse, Status> {
    let characters = record
        .characters
        .into_iter()
        .map(build_episode_character)
        .collect::<Result<Vec<_>, Status>>()?;

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
        score: record.score.map(|score| score as f32),
        season_number: record.season_number,
        synopsis: record.synopsis,
        title: record.title,
        title_japanese: record.title_japanese,
        title_romaji: record.title_romaji,
        images: record.images,
        streaming: record
            .streaming
            .into_iter()
            .map(|streaming| (streaming.platform, streaming.source))
            .collect(),
        sources: record.sources,
        characters,
        staff: Vec::new(),
        titles: record
            .titles
            .into_iter()
            .map(|title| (title.language, title.title))
            .collect(),
    };

    Ok(GetEpisodeResponse {
        episode: Some(episode),
        version: record.version,
        next_refresh_at: record.next_refresh_at.map(to_proto_timestamp),
    })
}

/// Converts a work's stored episodes into a [`ListEpisodesResponse`].
///
/// # Errors
///
/// Returns `INTERNAL` when a stored character role has no proto enum value.
pub(crate) fn build_list_episodes_response(
    records: Vec<VisibleEpisodeRecord>,
) -> Result<ListEpisodesResponse, Status> {
    Ok(ListEpisodesResponse {
        episodes: records
            .into_iter()
            .map(build_get_episode_response)
            .collect::<Result<Vec<_>, Status>>()?,
    })
}

/// Converts a stored character into a [`GetCharacterResponse`].
///
/// `roles` lists each distinct role once, in the order of the character's most prominent role
/// per work, and `anime_ids` lists the visible works in the same order.
///
/// # Errors
///
/// Returns `INTERNAL` when a stored character role has no proto enum value.
pub(crate) fn build_get_character_response(
    record: VisibleCharacterRecord,
) -> Result<GetCharacterResponse, Status> {
    let mut roles = Vec::new();
    let mut anime_ids = Vec::with_capacity(record.anime.len());
    for character_anime in record.anime {
        let role = read_character_role(&character_anime.role)?;
        if !roles.contains(&role) {
            roles.push(role);
        }
        anime_ids.push(character_anime.anime_id.to_string());
    }

    let character = Character {
        description: record.description,
        favorites: record.favorites.map(i64::from),
        id: Some(record.character_id.to_string()),
        entity_type: ProtoEntityType::Character as i32,
        name: record.name,
        name_native: record.name_native,
        roles,
        anime_ids,
        animeography: build_ography(record.animeography)?,
        traits: record.traits,
        images: record.images,
        mangaography: build_ography(record.mangaography)?,
        name_variations: record.name_variations,
        nicknames: record.nicknames,
        sources: record.sources,
        voice_actors: Vec::new(),
        attributes: record.attributes,
        spoilers: record.spoilers,
    };

    Ok(GetCharacterResponse {
        character: Some(character),
        version: record.version,
        next_refresh_at: record.next_refresh_at.map(to_proto_timestamp),
    })
}

/// Converts stored character role text into its proto enum number.
fn read_character_role(role: &str) -> Result<i32, Status> {
    read_enum_number(CHARACTER_ROLE_LOOKUP_TEXTS, role, "character role")
}

/// Converts an episode's character appearance into an [`EpisodeCharacter`].
///
/// The name is empty while no visible character holds the page the episode names.
fn build_episode_character(
    episode_character: VisibleEpisodeCharacter,
) -> Result<EpisodeCharacter, Status> {
    Ok(EpisodeCharacter {
        name: episode_character.name.unwrap_or_default(),
        role: read_character_role(&episode_character.role)?,
        sources: episode_character.sources,
        voice_actors: Vec::new(),
    })
}

/// Converts animeography or mangaography entries into [`Ography`] messages.
///
/// The title is empty while no visible work or stored source material matches the entry.
fn build_ography(ography_entries: Vec<VisibleOgraphyEntry>) -> Result<Vec<Ography>, Status> {
    ography_entries
        .into_iter()
        .map(|ography_entry| {
            Ok(Ography {
                title: ography_entry.title.unwrap_or_default(),
                role: read_character_role(&ography_entry.role)?,
                sources: ography_entry.sources,
            })
        })
        .collect()
}

/// Groups a work's images into cover, poster and banner lists.
///
/// The `image_type` lookup holds only these three values, so no other type can be stored.
fn build_anime_images(anime_images: Vec<AnimeImageInput>) -> AnimeImages {
    let mut images = AnimeImages::default();
    for anime_image in anime_images {
        match anime_image.image_type.as_str() {
            "cover" => images.covers.push(anime_image.url),
            "poster" => images.posters.push(anime_image.url),
            "banner" => images.banners.push(anime_image.url),
            _ => {}
        }
    }
    images
}

/// Converts stored opening or ending theme songs into protobuf [`ThemeSong`] entries.
fn build_theme_songs(songs: Vec<AnimeThemeSongInput>) -> Vec<ThemeSong> {
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

/// Converts stored trailer inputs into protobuf [`TrailerEntry`] messages.
fn build_trailers(trailers: Vec<AnimeTrailerInput>) -> Vec<TrailerEntry> {
    trailers
        .into_iter()
        .map(|trailer| TrailerEntry {
            title: trailer.title,
            thumbnail_source: trailer.thumbnail_source,
            source: Some(trailer.source),
        })
        .collect()
}

/// Converts stored external links into protobuf [`ExternalLink`] messages.
fn build_external_links(sources: Vec<AnimeExternalSourceInput>) -> Vec<ExternalLink> {
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

/// Converts stored streaming sources into protobuf [`StreamingEntry`] messages.
fn build_streaming_entries(sources: Vec<AnimeStreamingSourceInput>) -> Vec<StreamingEntry> {
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

/// Converts per-provider statistics and contextual ranks keyed by provider slug.
#[expect(
    clippy::cast_possible_truncation,
    reason = "proto score fields are f32, and stored scores hold two decimals between 0 and 10"
)]
fn build_statistics(stats: Vec<AnimeStatisticsInput>) -> BTreeMap<String, Statistics> {
    stats
        .into_iter()
        .map(|stat| {
            let contextual_ranks = stat
                .contextual_ranks
                .into_iter()
                .map(build_contextual_rank)
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

/// Converts a single contextual ranking entry into a protobuf [`ContextualRank`].
fn build_contextual_rank(rank: AnimeContextualRankInput) -> ContextualRank {
    ContextualRank {
        context: rank.context,
        rank: rank.rank,
        is_all_time: rank.is_all_time,
        format: rank.format,
        season: rank.season,
        year: rank.year.map(i32::from),
    }
}

/// Converts a stored broadcast schedule into a protobuf [`Broadcast`].
fn build_broadcast(broadcast: AnimeBroadcastInput) -> Broadcast {
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

/// Converts a stored hiatus snapshot into a protobuf [`AnimeHiatus`].
fn build_hiatus(hiatus: AnimeHiatusInput) -> AnimeHiatus {
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

/// Groups a work's relations by relation type.
///
/// A relation whose target pages no visible work holds keeps their page URLs and title, when
/// stored, with an unspecified type.
fn build_related_anime(
    relations: Vec<VisibleAnimeRelation>,
) -> Result<BTreeMap<String, RelatedAnimeList>, Status> {
    let mut related_anime: BTreeMap<String, RelatedAnimeList> = BTreeMap::new();
    for VisibleAnimeRelation {
        relation_type,
        target_page_sources,
        target_page_title,
        target_anime,
    } in relations
    {
        let related_entry = match target_anime {
            Some(target_anime) => RelatedAnime {
                title: target_anime.title,
                r#type: read_enum_number(
                    ANIME_TYPE_LOOKUP_TEXTS,
                    &target_anime.anime_type,
                    "related anime type",
                )?,
                sources: target_page_sources,
                status: Some(read_enum_number(
                    ANIME_STATUS_LOOKUP_TEXTS,
                    &target_anime.status,
                    "related anime status",
                )?),
                year: target_anime.year.map(i32::from),
                score: None,
                images: Vec::new(),
                episode_count: Some(target_anime.episode_count),
            },
            None => RelatedAnime {
                title: target_page_title.unwrap_or_default(),
                sources: target_page_sources,
                ..RelatedAnime::default()
            },
        };
        related_anime
            .entry(relation_type)
            .or_default()
            .items
            .push(related_entry);
    }
    Ok(related_anime)
}

/// Groups a work's related source material by relation type.
#[expect(
    clippy::cast_possible_truncation,
    reason = "proto score fields are f32, and stored scores hold two decimals between 0 and 10"
)]
fn build_related_source_material(
    source_materials: Vec<AnimeSourceMaterialInput>,
) -> Result<BTreeMap<String, RelatedSourceMaterialList>, Status> {
    let mut related_source_material: BTreeMap<String, RelatedSourceMaterialList> = BTreeMap::new();
    for source_material in source_materials {
        let related_entry = RelatedSourceMaterial {
            title: source_material.title,
            r#type: read_enum_number(
                SOURCE_MATERIAL_TYPE_LOOKUP_TEXTS,
                &source_material.source_material_type,
                "related source material type",
            )?,
            sources: source_material
                .xrefs
                .into_iter()
                .map(|xref| xref.url)
                .collect(),
            status: source_material
                .status
                .as_deref()
                .map(|status| {
                    read_enum_number(
                        ANIME_STATUS_LOOKUP_TEXTS,
                        status,
                        "related source material status",
                    )
                })
                .transpose()?,
            score: source_material.score.map(|score| score as f32),
            images: source_material.images,
            chapters: source_material.chapters,
            volumes: source_material.volumes,
        };
        related_source_material
            .entry(source_material.relation_type)
            .or_default()
            .items
            .push(related_entry);
    }
    Ok(related_source_material)
}

/// Converts a work's companies into [`CompanyEntry`] messages, one per company.
fn build_company_entries(companies: Vec<AnimeCompanyInput>) -> Result<Vec<CompanyEntry>, Status> {
    companies
        .into_iter()
        .map(|company| {
            Ok(CompanyEntry {
                name: company.name,
                description: company.description,
                sources: company.xrefs.into_iter().map(|xref| xref.url).collect(),
                roles: company
                    .roles
                    .iter()
                    .map(|role| read_enum_number(COMPANY_ROLE_LOOKUP_TEXTS, role, "company role"))
                    .collect::<Result<Vec<_>, Status>>()?,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use postgres_db::{
        CharacterAnime, CompanyXrefInput, EpisodeStreamingInput, RelatedAnimeSummary,
        SourceMaterialXrefInput,
    };
    use tonic::Code;
    use uuid::Uuid;

    use super::*;
    use crate::service::shared_proto::v1::{
        AnimeRating, AnimeSeason, AnimeStatus, AnimeType, CharacterRole, CompanyRole,
        SourceMaterialType,
    };

    fn build_anime_record(anime_type: &str, rating: &str) -> VisibleAnimeRecord {
        VisibleAnimeRecord {
            anime_id: Uuid::now_v7(),
            version: 3,
            next_refresh_at: None,
            title: "Cowboy Bebop".to_owned(),
            title_english: None,
            title_japanese: None,
            anime_type: anime_type.to_owned(),
            status: "FINISHED".to_owned(),
            rating: rating.to_owned(),
            season: Some("SPRING".to_owned()),
            source_material: Some("4-KOMA".to_owned()),
            year: Some(1998),
            month: Some(4),
            episode_count: 26,
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
            sources: vec!["https://myanimelist.net/anime/1/Cowboy_Bebop".to_owned()],
            partial_sources: Vec::new(),
            images: vec![
                AnimeImageInput {
                    image_type: "cover".to_owned(),
                    url: "https://cdn.example/cover.jpg".to_owned(),
                },
                AnimeImageInput {
                    image_type: "banner".to_owned(),
                    url: "https://cdn.example/banner.jpg".to_owned(),
                },
            ],
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
        }
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "one record covers every lookup field, relation, source material and company"
    )]
    fn build_get_anime_response_stored_lookup_text_returns_proto_values() {
        let mut record = build_anime_record("TV SPECIAL", "PG-13 - Teens 13 or older");
        let sequel_id = Uuid::now_v7();
        record.related_anime = vec![
            VisibleAnimeRelation {
                relation_type: "SEQUEL".to_owned(),
                target_page_sources: vec!["https://myanimelist.net/anime/2/Movie".to_owned()],
                target_page_title: None,
                target_anime: Some(RelatedAnimeSummary {
                    anime_id: sequel_id,
                    title: "Movie".to_owned(),
                    anime_type: "MOVIE".to_owned(),
                    status: "UPCOMING".to_owned(),
                    year: Some(2001),
                    episode_count: 1,
                }),
            },
            VisibleAnimeRelation {
                relation_type: "SEQUEL".to_owned(),
                target_page_sources: vec!["https://anilist.co/anime/3".to_owned()],
                target_page_title: Some("Third".to_owned()),
                target_anime: None,
            },
        ];
        record.related_source_material = vec![AnimeSourceMaterialInput {
            relation_type: "ADAPTATION".to_owned(),
            title: "Manga".to_owned(),
            title_english: None,
            title_japanese: None,
            source_material_type: "LIGHT NOVEL".to_owned(),
            status: Some("ONGOING".to_owned()),
            score: Some(7.5),
            chapters: Some(10),
            volumes: None,
            images: vec!["https://cdn.example/manga.jpg".to_owned()],
            xrefs: vec![SourceMaterialXrefInput {
                provider: "mal".to_owned(),
                external_id: "100".to_owned(),
                url: "https://myanimelist.net/manga/100".to_owned(),
                title: None,
            }],
        }];
        record.companies = vec![AnimeCompanyInput {
            name: "Sunrise".to_owned(),
            description: Some("Studio".to_owned()),
            roles: vec!["PRODUCER".to_owned(), "STUDIO".to_owned()],
            xrefs: vec![CompanyXrefInput {
                provider: "mal".to_owned(),
                url: "https://myanimelist.net/anime/producer/14".to_owned(),
            }],
        }];

        let anime = build_get_anime_response(record)
            .expect("map anime")
            .anime
            .expect("anime");
        assert_eq!(anime.r#type, AnimeType::TvSpecial as i32);
        assert_eq!(anime.status, AnimeStatus::Finished as i32);
        assert_eq!(anime.rating, Some(AnimeRating::Pg13 as i32));
        assert_eq!(anime.season, Some(AnimeSeason::Spring as i32));
        assert_eq!(
            anime.source_material,
            Some(SourceMaterialType::Koma4 as i32)
        );
        assert_eq!(
            anime.sources,
            vec!["https://myanimelist.net/anime/1/Cowboy_Bebop"]
        );
        let images = anime.images.expect("images");
        assert_eq!(images.covers, vec!["https://cdn.example/cover.jpg"]);
        assert_eq!(images.banners, vec!["https://cdn.example/banner.jpg"]);
        assert_eq!(images.posters, Vec::<String>::new());
        assert_eq!(
            anime.related_anime["SEQUEL"].items,
            vec![
                RelatedAnime {
                    title: "Movie".to_owned(),
                    r#type: AnimeType::Movie as i32,
                    sources: vec!["https://myanimelist.net/anime/2/Movie".to_owned()],
                    status: Some(AnimeStatus::Upcoming as i32),
                    year: Some(2001),
                    score: None,
                    images: Vec::new(),
                    episode_count: Some(1),
                },
                RelatedAnime {
                    title: "Third".to_owned(),
                    sources: vec!["https://anilist.co/anime/3".to_owned()],
                    ..RelatedAnime::default()
                },
            ]
        );
        assert_eq!(
            anime.related_source_material["ADAPTATION"].items,
            vec![RelatedSourceMaterial {
                title: "Manga".to_owned(),
                r#type: SourceMaterialType::LightNovel as i32,
                sources: vec!["https://myanimelist.net/manga/100".to_owned()],
                status: Some(AnimeStatus::Ongoing as i32),
                score: Some(7.5),
                images: vec!["https://cdn.example/manga.jpg".to_owned()],
                chapters: Some(10),
                volumes: None,
            }]
        );
        assert_eq!(
            anime.companies,
            vec![CompanyEntry {
                name: "Sunrise".to_owned(),
                description: Some("Studio".to_owned()),
                sources: vec!["https://myanimelist.net/anime/producer/14".to_owned()],
                roles: vec![CompanyRole::Producer as i32, CompanyRole::Studio as i32],
            }]
        );
    }

    #[test]
    fn build_get_anime_response_unknown_stored_text_returns_internal() {
        for record in [
            build_anime_record("ANIME_TYPE_TV", "UNKNOWN"),
            build_anime_record("TV", "PG_13"),
        ] {
            let status = build_get_anime_response(record).expect_err("unknown text");
            assert_eq!(status.code(), Code::Internal);
        }
    }

    #[test]
    fn build_get_episode_response_maps_characters_streaming_and_sources() {
        let episode = build_get_episode_response(VisibleEpisodeRecord {
            episode_id: Uuid::now_v7(),
            anime_id: Uuid::now_v7(),
            version: 1,
            next_refresh_at: None,
            episode_number: 1,
            season_number: None,
            title: "Asteroid Blues".to_owned(),
            title_japanese: None,
            title_romaji: None,
            synopsis: None,
            description: None,
            aired_at: None,
            duration_seconds: None,
            is_filler: false,
            is_recap: false,
            score: None,
            titles: Vec::new(),
            images: Vec::new(),
            streaming: vec![EpisodeStreamingInput {
                platform: "Crunchyroll".to_owned(),
                source: "https://crunchyroll.com/cb/1".to_owned(),
            }],
            sources: vec!["https://myanimelist.net/anime/1/Cowboy_Bebop/episode/1".to_owned()],
            characters: vec![
                VisibleEpisodeCharacter {
                    name: Some("Spike Spiegel".to_owned()),
                    role: "MAIN".to_owned(),
                    sources: vec!["https://myanimelist.net/character/1".to_owned()],
                },
                VisibleEpisodeCharacter {
                    name: None,
                    role: "BACKGROUND".to_owned(),
                    sources: Vec::new(),
                },
            ],
        })
        .expect("map episode")
        .episode
        .expect("episode");
        assert_eq!(
            episode.streaming,
            BTreeMap::from([(
                "Crunchyroll".to_owned(),
                "https://crunchyroll.com/cb/1".to_owned()
            )])
        );
        assert_eq!(
            episode.characters,
            vec![
                EpisodeCharacter {
                    name: "Spike Spiegel".to_owned(),
                    role: CharacterRole::Main as i32,
                    sources: vec!["https://myanimelist.net/character/1".to_owned()],
                    voice_actors: Vec::new(),
                },
                EpisodeCharacter {
                    name: String::new(),
                    role: CharacterRole::Background as i32,
                    sources: Vec::new(),
                    voice_actors: Vec::new(),
                },
            ]
        );
    }

    #[test]
    fn build_get_character_response_lists_each_role_once_in_prominence_order() {
        let first_anime_id = Uuid::now_v7();
        let second_anime_id = Uuid::now_v7();
        let third_anime_id = Uuid::now_v7();
        let character = build_get_character_response(VisibleCharacterRecord {
            character_id: Uuid::now_v7(),
            version: 1,
            next_refresh_at: None,
            name: "Spike Spiegel".to_owned(),
            name_native: None,
            description: None,
            favorites: Some(10),
            name_variations: Vec::new(),
            nicknames: Vec::new(),
            traits: Vec::new(),
            images: Vec::new(),
            attributes: BTreeMap::new(),
            spoilers: BTreeMap::new(),
            sources: vec!["https://myanimelist.net/character/1".to_owned()],
            anime: vec![
                CharacterAnime {
                    anime_id: first_anime_id,
                    role: "MAIN".to_owned(),
                },
                CharacterAnime {
                    anime_id: second_anime_id,
                    role: "MAIN".to_owned(),
                },
                CharacterAnime {
                    anime_id: third_anime_id,
                    role: "SUPPORTING".to_owned(),
                },
            ],
            animeography: vec![VisibleOgraphyEntry {
                title: Some("Cowboy Bebop".to_owned()),
                role: "MAIN".to_owned(),
                sources: vec!["https://myanimelist.net/anime/1".to_owned()],
            }],
            mangaography: vec![VisibleOgraphyEntry {
                title: None,
                role: "UNKNOWN".to_owned(),
                sources: Vec::new(),
            }],
        })
        .expect("map character")
        .character
        .expect("character");
        assert_eq!(
            character.roles,
            vec![CharacterRole::Main as i32, CharacterRole::Supporting as i32]
        );
        assert_eq!(
            character.anime_ids,
            vec![
                first_anime_id.to_string(),
                second_anime_id.to_string(),
                third_anime_id.to_string()
            ]
        );
        assert_eq!(
            character.animeography,
            vec![Ography {
                title: "Cowboy Bebop".to_owned(),
                role: CharacterRole::Main as i32,
                sources: vec!["https://myanimelist.net/anime/1".to_owned()],
            }]
        );
        assert_eq!(
            character.mangaography,
            vec![Ography {
                title: String::new(),
                role: CharacterRole::Unknown as i32,
                sources: Vec::new(),
            }]
        );
    }

    #[test]
    fn build_get_character_response_unknown_role_returns_internal() {
        let status = build_get_character_response(VisibleCharacterRecord {
            character_id: Uuid::now_v7(),
            version: 1,
            next_refresh_at: None,
            name: "Spike Spiegel".to_owned(),
            name_native: None,
            description: None,
            favorites: None,
            name_variations: Vec::new(),
            nicknames: Vec::new(),
            traits: Vec::new(),
            images: Vec::new(),
            attributes: BTreeMap::new(),
            spoilers: BTreeMap::new(),
            sources: Vec::new(),
            anime: vec![CharacterAnime {
                anime_id: Uuid::now_v7(),
                role: "HERO".to_owned(),
            }],
            animeography: Vec::new(),
            mangaography: Vec::new(),
        })
        .expect_err("unknown role");
        assert_eq!(status.code(), Code::Internal);
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "one record fills every detail list and compares each mapped message"
    )]
    fn build_get_anime_response_maps_every_detail_field() {
        let aired_from = chrono::Utc::now();
        let mut record = build_anime_record("TV", "UNKNOWN");
        record.aired_from = Some(aired_from);
        record.score_weighted = Some(8.5);
        record.score_median = Some(9.0);
        record.titles = vec![postgres_db::AnimeTitleInput {
            language: "ja".to_owned(),
            title: "カウボーイビバップ".to_owned(),
        }];
        record.themes = vec![postgres_db::AnimeThemeInput {
            name: "Space".to_owned(),
            description: Some("Outer space".to_owned()),
        }];
        record.images = vec![
            AnimeImageInput {
                image_type: "poster".to_owned(),
                url: "https://cdn.example/poster.jpg".to_owned(),
            },
            AnimeImageInput {
                image_type: "thumbnail".to_owned(),
                url: "https://cdn.example/thumbnail.jpg".to_owned(),
            },
        ];
        record.opening_themes = vec![AnimeThemeSongInput {
            position: 0,
            title: "Tank!".to_owned(),
            artist: Some("The Seatbelts".to_owned()),
            episode_ranges: vec![postgres_db::EpisodeRangeInput {
                range_start: 1,
                range_end: Some(26),
            }],
        }];
        record.trailers = vec![AnimeTrailerInput {
            source: "https://youtube.com/watch?v=1".to_owned(),
            title: Some("PV".to_owned()),
            thumbnail_source: Some("https://img.youtube.com/1.jpg".to_owned()),
        }];
        record.external_sources = vec![AnimeExternalSourceInput {
            source: "https://cowboy-bebop.net".to_owned(),
            platform: "Official".to_owned(),
            label: Some("Official Site".to_owned()),
            language: Some("ja".to_owned()),
        }];
        record.streaming_sources = vec![AnimeStreamingSourceInput {
            platform: "Crunchyroll".to_owned(),
            source: "https://crunchyroll.com/cb".to_owned(),
            region: Some("US".to_owned()),
            is_free: Some(false),
            is_premium_required: Some(true),
            has_dub: Some(true),
            subtitle_languages: vec!["en".to_owned()],
        }];
        record.statistics = vec![AnimeStatisticsInput {
            provider: "mal".to_owned(),
            score: Some(8.75),
            scored_by: Some(1000),
            members: Some(5000),
            favorites: Some(500),
            rank: Some(40),
            popularity: Some(45),
            contextual_ranks: vec![AnimeContextualRankInput {
                context: "SPRING 1998".to_owned(),
                rank: 1,
                is_all_time: Some(false),
                format: Some("TV".to_owned()),
                season: Some("SPRING".to_owned()),
                year: Some(1998),
            }],
        }];
        record.broadcast = Some(AnimeBroadcastInput {
            weekday: Some(5),
            timezone: Some("Asia/Tokyo".to_owned()),
            jp_time: chrono::NaiveTime::from_hms_opt(1, 30, 0),
            sub_time: chrono::NaiveTime::from_hms_opt(2, 0, 0),
            dub_time: chrono::NaiveTime::from_hms_opt(3, 15, 30),
            sub_delay_days: Some(0),
            dub_delay_days: Some(7),
            premiere_jp: Some(aired_from),
            premiere_sub: None,
            premiere_dub: None,
            next_episode_at: Some(aired_from),
            next_episode_number: Some(2),
        });
        record.hiatus = Some(AnimeHiatusInput {
            reason: Some("Recap week".to_owned()),
            hiatus_from: chrono::NaiveDate::from_ymd_opt(1998, 6, 1),
            hiatus_until: chrono::NaiveDate::from_ymd_opt(1998, 6, 8),
        });

        let anime = build_get_anime_response(record)
            .expect("map anime")
            .anime
            .expect("anime");
        assert_eq!(
            anime.aired_dates,
            Some(AiredDates {
                aired_from: Some(to_proto_timestamp(aired_from)),
                aired_to: None,
            })
        );
        assert_eq!(
            anime.score,
            Some(ScoreCalculations {
                weighted: Some(8.5),
                mean: None,
                median: Some(9.0),
            })
        );
        assert_eq!(
            anime.titles,
            BTreeMap::from([("ja".to_owned(), "カウボーイビバップ".to_owned())])
        );
        assert_eq!(
            anime.themes,
            vec![ThemeEntry {
                name: "Space".to_owned(),
                description: Some("Outer space".to_owned()),
            }]
        );
        let images = anime.images.expect("images");
        assert_eq!(images.posters, vec!["https://cdn.example/poster.jpg"]);
        assert_eq!(images.covers, Vec::<String>::new());
        assert_eq!(
            anime.opening_themes,
            vec![ThemeSong {
                title: "Tank!".to_owned(),
                artist: Some("The Seatbelts".to_owned()),
                episodes: vec![EpisodeRange {
                    range_start: 1,
                    range_end: Some(26),
                }],
            }]
        );
        assert_eq!(anime.ending_themes, Vec::new());
        assert_eq!(
            anime.trailers,
            vec![TrailerEntry {
                title: Some("PV".to_owned()),
                thumbnail_source: Some("https://img.youtube.com/1.jpg".to_owned()),
                source: Some("https://youtube.com/watch?v=1".to_owned()),
            }]
        );
        assert_eq!(
            anime.external_sources,
            vec![ExternalLink {
                platform: "Official".to_owned(),
                source: "https://cowboy-bebop.net".to_owned(),
                label: Some("Official Site".to_owned()),
                language: Some("ja".to_owned()),
            }]
        );
        assert_eq!(
            anime.streaming_sources,
            vec![StreamingEntry {
                platform: "Crunchyroll".to_owned(),
                source: "https://crunchyroll.com/cb".to_owned(),
                region: Some("US".to_owned()),
                is_free: Some(false),
                is_premium_required: Some(true),
                has_dub: Some(true),
                subtitle_languages: vec!["en".to_owned()],
            }]
        );
        assert_eq!(
            anime.statistics,
            BTreeMap::from([(
                "mal".to_owned(),
                Statistics {
                    score: Some(8.75),
                    scored_by: Some(1000),
                    members: Some(5000),
                    favorites: Some(500),
                    rank: Some(40),
                    popularity: Some(45),
                    contextual_ranks: vec![ContextualRank {
                        context: "SPRING 1998".to_owned(),
                        rank: 1,
                        is_all_time: Some(false),
                        format: Some("TV".to_owned()),
                        season: Some("SPRING".to_owned()),
                        year: Some(1998),
                    }],
                }
            )])
        );
        assert_eq!(
            anime.broadcast,
            Some(Broadcast {
                day: Some(5),
                timezone: Some("Asia/Tokyo".to_owned()),
                jp_time: Some("01:30:00".to_owned()),
                sub_time: Some("02:00:00".to_owned()),
                dub_time: Some("03:15:30".to_owned()),
                sub_delay_days: Some(0),
                dub_delay_days: Some(7),
                premiere_jp: Some(to_proto_timestamp(aired_from)),
                premiere_sub: None,
                premiere_dub: None,
                next_episode_at: Some(to_proto_timestamp(aired_from)),
                next_episode_number: Some(2),
            })
        );
        assert_eq!(
            anime.hiatus,
            Some(AnimeHiatus {
                reason: Some("Recap week".to_owned()),
                hiatus_from: Some("1998-06-01".to_owned()),
                hiatus_until: Some("1998-06-08".to_owned()),
            })
        );
    }

    #[test]
    fn build_get_anime_response_unknown_nested_lookup_text_returns_internal() {
        let mut unknown_season = build_anime_record("TV", "UNKNOWN");
        unknown_season.season = Some("AUTUMN".to_owned());
        let mut unknown_source_material = build_anime_record("TV", "UNKNOWN");
        unknown_source_material.source_material = Some("COMIC BOOK".to_owned());
        let mut unknown_status = build_anime_record("TV", "UNKNOWN");
        unknown_status.status = "AIRING".to_owned();
        let mut unknown_related_type = build_anime_record("TV", "UNKNOWN");
        unknown_related_type.related_anime = vec![VisibleAnimeRelation {
            relation_type: "SEQUEL".to_owned(),
            target_page_sources: Vec::new(),
            target_page_title: None,
            target_anime: Some(RelatedAnimeSummary {
                anime_id: Uuid::now_v7(),
                title: "Sequel".to_owned(),
                anime_type: "SERIES".to_owned(),
                status: "FINISHED".to_owned(),
                year: None,
                episode_count: 1,
            }),
        }];
        let mut unknown_related_status = build_anime_record("TV", "UNKNOWN");
        unknown_related_status.related_anime = vec![VisibleAnimeRelation {
            relation_type: "SEQUEL".to_owned(),
            target_page_sources: Vec::new(),
            target_page_title: None,
            target_anime: Some(RelatedAnimeSummary {
                anime_id: Uuid::now_v7(),
                title: "Sequel".to_owned(),
                anime_type: "TV".to_owned(),
                status: "AIRING".to_owned(),
                year: None,
                episode_count: 1,
            }),
        }];
        let source_material = AnimeSourceMaterialInput {
            relation_type: "ADAPTATION".to_owned(),
            title: "Manga".to_owned(),
            title_english: None,
            title_japanese: None,
            source_material_type: "MANGA".to_owned(),
            status: None,
            score: None,
            chapters: None,
            volumes: None,
            images: Vec::new(),
            xrefs: Vec::new(),
        };
        let mut unknown_material_type = build_anime_record("TV", "UNKNOWN");
        unknown_material_type.related_source_material = vec![AnimeSourceMaterialInput {
            source_material_type: "COMIC BOOK".to_owned(),
            ..source_material.clone()
        }];
        let mut unknown_material_status = build_anime_record("TV", "UNKNOWN");
        unknown_material_status.related_source_material = vec![AnimeSourceMaterialInput {
            status: Some("AIRING".to_owned()),
            ..source_material
        }];
        let mut unknown_company_role = build_anime_record("TV", "UNKNOWN");
        unknown_company_role.companies = vec![AnimeCompanyInput {
            name: "Sunrise".to_owned(),
            description: None,
            roles: vec!["ANIMATOR".to_owned()],
            xrefs: Vec::new(),
        }];
        for record in [
            unknown_season,
            unknown_source_material,
            unknown_status,
            unknown_related_type,
            unknown_related_status,
            unknown_material_type,
            unknown_material_status,
            unknown_company_role,
        ] {
            let status = build_get_anime_response(record).expect_err("unknown stored text");
            assert_eq!(status.code(), Code::Internal, "{}", status.message());
        }
    }

    #[test]
    fn build_list_episodes_response_unknown_character_role_returns_internal() {
        let episode = VisibleEpisodeRecord {
            episode_id: Uuid::now_v7(),
            anime_id: Uuid::now_v7(),
            version: 1,
            next_refresh_at: None,
            episode_number: 1,
            season_number: None,
            title: String::new(),
            title_japanese: None,
            title_romaji: None,
            synopsis: None,
            description: None,
            aired_at: None,
            duration_seconds: None,
            is_filler: false,
            is_recap: false,
            score: None,
            titles: vec![postgres_db::EpisodeTitleInput {
                language: "en".to_owned(),
                title: "Asteroid Blues".to_owned(),
            }],
            images: Vec::new(),
            streaming: Vec::new(),
            sources: Vec::new(),
            characters: Vec::new(),
        };
        let listed = build_list_episodes_response(vec![episode.clone()]).expect("list episodes");
        assert_eq!(
            listed.episodes[0].episode.as_ref().expect("episode").titles,
            BTreeMap::from([("en".to_owned(), "Asteroid Blues".to_owned())])
        );
        let status = build_list_episodes_response(vec![VisibleEpisodeRecord {
            characters: vec![VisibleEpisodeCharacter {
                name: None,
                role: "HERO".to_owned(),
                sources: Vec::new(),
            }],
            ..episode
        }])
        .expect_err("unknown role");
        assert_eq!(status.code(), Code::Internal);
    }

    #[test]
    fn build_get_character_response_unknown_ography_role_returns_internal() {
        let character = VisibleCharacterRecord {
            character_id: Uuid::now_v7(),
            version: 1,
            next_refresh_at: None,
            name: "Spike Spiegel".to_owned(),
            name_native: None,
            description: None,
            favorites: None,
            name_variations: Vec::new(),
            nicknames: Vec::new(),
            traits: Vec::new(),
            images: Vec::new(),
            attributes: BTreeMap::new(),
            spoilers: BTreeMap::new(),
            sources: Vec::new(),
            anime: Vec::new(),
            animeography: Vec::new(),
            mangaography: Vec::new(),
        };
        let unknown_entry = VisibleOgraphyEntry {
            title: None,
            role: "HERO".to_owned(),
            sources: Vec::new(),
        };
        for record in [
            VisibleCharacterRecord {
                animeography: vec![unknown_entry.clone()],
                ..character.clone()
            },
            VisibleCharacterRecord {
                mangaography: vec![unknown_entry],
                ..character
            },
        ] {
            let status = build_get_character_response(record).expect_err("unknown role");
            assert_eq!(status.code(), Code::Internal);
        }
    }
}
