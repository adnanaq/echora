//! Exhaustive protobuf-to-SQL mapping and canonical hashing for `Anime`.

use chrono::{DateTime, Utc};
use postgres_db::{
    ANIME_HASH_FORMAT, AnimeDetailsInput, AnimeImageInput, AnimeThemeInput, AnimeTitleInput,
    SaveAnimeInput,
};
use prost::Message as _;
use sha2::{Digest as _, Sha256};
use tonic::Status;
use uuid::Uuid;

use crate::anime_relations_mapping::build_relations_and_graph;
use crate::anime_schedule_and_stats_mapping::build_schedule_and_stats;
use crate::episode_mapping::build_episode_references;
use crate::lookup_text::{
    ANIME_RATING_LOOKUP_TEXTS, ANIME_SEASON_LOOKUP_TEXTS, ANIME_STATUS_LOOKUP_TEXTS,
    ANIME_TYPE_LOOKUP_TEXTS, SOURCE_MATERIAL_TYPE_LOOKUP_TEXTS, find_specified_lookup_text,
};
use crate::service::parse_page_sources;
use crate::service::postgres_service_proto::v1::SaveAnimeRequest;
use crate::service::shared_proto::v1::{
    Anime, AnimeImages, AnimeRating, AnimeSeason, AnimeStatus, AnimeType, EntityType,
    EpisodeReference, RelationClaim, SourceMaterialType, ThemeEntry,
};

/// Maps a `shared_proto.v1.Anime` and save metadata into [`SaveAnimeInput`].
///
/// Computes `merged_record_hash` over the full canonical `Anime` message, while
/// persisting scalar `anime` columns plus work names, classification, themes,
/// and images. Destructures every field of [`Anime`] without `..` and matches
/// every proto enum exhaustively without `_` so proto changes fail compilation
/// until handled.
///
/// # Errors
///
/// Returns [`Status::invalid_argument`] when `anime_id` is not a valid UUID,
/// required enums are unspecified or unknown, or numeric bounds are invalid.
#[expect(
    clippy::too_many_lines,
    reason = "Exhaustive destructuring of every Anime proto field without '..' is required by ECHO-98"
)]
/// Maps a `SaveAnime` request into the work save of `postgres_db`.
///
/// The merged record hash covers the work, its relation claims and its episode list.
///
/// # Errors
///
/// Returns `INVALID_ARGUMENT` for an invalid work ID, a missing work, an unknown enum value,
/// an invalid timestamp, an invalid episode reference, or an invalid page source.
pub fn build_save_anime_input(
    request: SaveAnimeRequest,
    traceparent: Option<String>,
) -> Result<SaveAnimeInput, Status> {
    let SaveAnimeRequest {
        anime_id: anime_id_text,
        expected_version,
        anime,
        merge_version,
        relation_claims,
        episodes,
        page_sources,
    } = request;
    let anime_id = Uuid::parse_str(&anime_id_text)
        .map_err(|err| Status::invalid_argument(format!("invalid anime_id UUID: {err}")))?;
    let anime = anime.ok_or_else(|| Status::invalid_argument("anime must be provided"))?;
    let page_sources = parse_page_sources(page_sources)?;

    let merged_record_hash = compute_save_record_hash(&anime, &relation_claims, &episodes);
    let mapped_episodes = build_episode_references(episodes)?;

    let Anime {
        background,
        country_of_origin,
        duration_seconds,
        episode_count,
        id: _,
        entity_type,
        month,
        is_nsfw,
        rating,
        season,
        source_material,
        status,
        synopsis,
        title,
        title_english,
        title_japanese,
        r#type,
        year,
        content_warnings,
        demographics,
        ending_themes,
        genres,
        opening_themes,
        related_anime,
        related_source_material,
        sources: _,
        streaming_sources,
        synonyms,
        tags,
        themes,
        trailers,
        aired_dates,
        broadcast,
        external_sources,
        images,
        score,
        production_staff: _,
        statistics,
        hiatus,
        titles,
        companies,
        partial_sources: _,
    } = anime;

    validate_entity_type(entity_type)?;
    let anime_type = map_anime_type(r#type)?.to_owned();
    let anime_status = map_anime_status(status)?.to_owned();
    let anime_rating = match rating {
        Some(rating_number) => map_anime_rating(rating_number)?.to_owned(),
        None => "UNKNOWN".to_owned(),
    };
    let anime_season = season.map(map_anime_season).transpose()?.map(str::to_owned);
    let anime_source_material = source_material
        .map(map_source_material_type)
        .transpose()?
        .map(str::to_owned);

    let release_year = year
        .map(|val| {
            i16::try_from(val).map_err(|_| Status::invalid_argument("year out of smallint range"))
        })
        .transpose()?;
    let release_month = month
        .map(|val| {
            i16::try_from(val).map_err(|_| Status::invalid_argument("month out of smallint range"))
        })
        .transpose()?;

    let (aired_from, aired_to) = match aired_dates {
        Some(dates) => (
            dates
                .aired_from
                .map(|proto_timestamp| parse_proto_timestamp("aired_from", proto_timestamp))
                .transpose()?,
            dates
                .aired_to
                .map(|proto_timestamp| parse_proto_timestamp("aired_to", proto_timestamp))
                .transpose()?,
        ),
        None => (None, None),
    };

    let (score_weighted, score_mean, score_median) = match score {
        Some(scores) => (
            scores.weighted.map(f64::from),
            scores.mean.map(f64::from),
            scores.median.map(f64::from),
        ),
        None => (None, None, None),
    };

    let details = build_anime_details(
        titles,
        synonyms,
        genres,
        tags,
        demographics,
        content_warnings,
        themes,
        images,
    );

    let schedule_and_stats = build_schedule_and_stats(
        opening_themes,
        ending_themes,
        trailers,
        external_sources,
        streaming_sources,
        statistics,
        broadcast,
        hiatus,
    )?;
    let relations_and_graph = build_relations_and_graph(
        related_anime,
        related_source_material,
        companies,
        relation_claims,
    )?;

    Ok(SaveAnimeInput {
        anime_id,
        expected_version,
        title,
        title_english,
        title_japanese,
        anime_type,
        status: anime_status,
        rating: anime_rating,
        season: anime_season,
        source_material: anime_source_material,
        year: release_year,
        month: release_month,
        episode_count,
        duration_seconds,
        is_nsfw,
        country_of_origin,
        synopsis,
        background,
        aired_from,
        aired_to,
        score_weighted,
        score_mean,
        score_median,
        merged_record_hash,
        hash_format: ANIME_HASH_FORMAT,
        merge_version,
        traceparent,
        page_sources,
        details,
        schedule_and_stats,
        relations_and_graph,
        episodes: mapped_episodes,
    })
}

#[expect(
    clippy::too_many_arguments,
    reason = "Maps the eight ECHO-100 detail collections into AnimeDetailsInput"
)]
/// Collects a work's titles, names, classification, themes and images for the save.
fn build_anime_details(
    titles: std::collections::BTreeMap<String, String>,
    synonyms: Vec<String>,
    genres: Vec<String>,
    tags: Vec<String>,
    demographics: Vec<String>,
    content_warnings: Vec<String>,
    themes: Vec<ThemeEntry>,
    images: Option<AnimeImages>,
) -> AnimeDetailsInput {
    let mapped_titles = titles
        .into_iter()
        .map(|(language, title)| AnimeTitleInput { language, title })
        .collect();
    let mapped_themes = themes
        .into_iter()
        .map(|ThemeEntry { name, description }| AnimeThemeInput { name, description })
        .collect();
    let mapped_images = match images {
        Some(AnimeImages {
            covers,
            posters,
            banners,
        }) => covers
            .into_iter()
            .map(|url| ("cover", url))
            .chain(posters.into_iter().map(|url| ("poster", url)))
            .chain(banners.into_iter().map(|url| ("banner", url)))
            .map(|(image_type, url)| AnimeImageInput {
                image_type: image_type.to_owned(),
                url,
            })
            .collect(),
        None => Vec::new(),
    };

    AnimeDetailsInput {
        titles: mapped_titles,
        synonyms,
        genres,
        tags,
        demographics,
        content_warnings,
        themes: mapped_themes,
        images: mapped_images,
    }
}

/// Computes the change hash of a work save as a SHA-256 hex digest.
///
/// The hash covers the canonical [`Anime`], its sorted [`RelationClaim`] list and its sorted
/// episode list, so a save whose hash matches the stored one changes nothing.
#[must_use]
pub fn compute_save_record_hash(
    anime: &Anime,
    relation_claims: &[RelationClaim],
    episodes: &[EpisodeReference],
) -> String {
    let mut canonical_anime = anime.clone();
    canonical_anime.content_warnings.sort();
    canonical_anime.demographics.sort();
    canonical_anime.genres.sort();
    canonical_anime.sources.sort();
    canonical_anime.partial_sources.sort();
    canonical_anime.synonyms.sort();
    canonical_anime.tags.sort();

    let mut canonical_claims = relation_claims.to_vec();
    canonical_claims.sort_by(|left, right| {
        let left_target = left.target_page.as_ref().map(|page| {
            (
                page.provider.as_str(),
                page.entity_type,
                page.address.as_str(),
            )
        });
        let right_target = right.target_page.as_ref().map(|page| {
            (
                page.provider.as_str(),
                page.entity_type,
                page.address.as_str(),
            )
        });
        (&left.provider, left.relation_type, left_target).cmp(&(
            &right.provider,
            right.relation_type,
            right_target,
        ))
    });

    let mut hasher = Sha256::new();
    let anime_bytes = canonical_anime.encode_to_vec();
    hasher.update(anime_bytes.len().to_be_bytes());
    hasher.update(&anime_bytes);
    for claim in &canonical_claims {
        let claim_bytes = claim.encode_to_vec();
        hasher.update(claim_bytes.len().to_be_bytes());
        hasher.update(&claim_bytes);
    }

    let mut canonical_episodes = episodes.to_vec();
    for episode in &mut canonical_episodes {
        episode.page_keys.sort_by(|left, right| {
            (&left.provider, left.entity_type, &left.address).cmp(&(
                &right.provider,
                right.entity_type,
                &right.address,
            ))
        });
        episode
            .list_entries
            .sort_by(|left, right| left.provider.cmp(&right.provider));
    }
    canonical_episodes.sort_by(|left, right| left.episode_id.cmp(&right.episode_id));
    for episode in &canonical_episodes {
        let episode_bytes = episode.encode_to_vec();
        hasher.update(episode_bytes.len().to_be_bytes());
        hasher.update(&episode_bytes);
    }
    let digest = hasher.finalize();
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(hex, "{byte:02x}");
    }
    hex
}

/// Converts a protobuf `Timestamp` field into a UTC timestamp.
///
/// # Errors
///
/// Returns `INVALID_ARGUMENT` naming `field_name` when the time is out of range.
pub(crate) fn parse_proto_timestamp(
    field_name: &str,
    proto_timestamp: prost_types::Timestamp,
) -> Result<DateTime<Utc>, Status> {
    let nanos = u32::try_from(proto_timestamp.nanos)
        .map_err(|_| Status::invalid_argument(format!("{field_name}.nanos is invalid")))?;
    DateTime::<Utc>::from_timestamp(proto_timestamp.seconds, nanos)
        .ok_or_else(|| Status::invalid_argument(format!("{field_name} is out of range")))
}

/// Checks that a work's entity type is `ANIME`.
///
/// # Errors
///
/// Returns `INVALID_ARGUMENT` for any other entity type.
fn validate_entity_type(enum_number: i32) -> Result<(), Status> {
    let Ok(entity_type) = EntityType::try_from(enum_number) else {
        return Err(Status::invalid_argument(format!(
            "unknown EntityType number {enum_number}"
        )));
    };
    match entity_type {
        EntityType::Unspecified | EntityType::Anime => Ok(()),
        EntityType::Character | EntityType::Episode => Err(Status::invalid_argument(
            "SaveAnime requires an ANIME entity_type",
        )),
    }
}

/// Converts an `AnimeType` enum number into its lookup text.
///
/// # Errors
///
/// Returns `INVALID_ARGUMENT` for an unspecified or unknown value.
fn map_anime_type(enum_number: i32) -> Result<&'static str, Status> {
    let Ok(anime_type) = AnimeType::try_from(enum_number) else {
        return Err(Status::invalid_argument(format!(
            "unknown AnimeType number {enum_number}"
        )));
    };
    if anime_type == AnimeType::Unspecified {
        return Err(Status::invalid_argument("anime.type must be specified"));
    }
    Ok(find_specified_lookup_text(
        ANIME_TYPE_LOOKUP_TEXTS,
        anime_type,
        "AnimeType",
    ))
}

/// Converts an `AnimeStatus` enum number into its lookup text.
///
/// # Errors
///
/// Returns `INVALID_ARGUMENT` for an unspecified or unknown value.
fn map_anime_status(enum_number: i32) -> Result<&'static str, Status> {
    let Ok(anime_status) = AnimeStatus::try_from(enum_number) else {
        return Err(Status::invalid_argument(format!(
            "unknown AnimeStatus number {enum_number}"
        )));
    };
    if anime_status == AnimeStatus::Unspecified {
        return Err(Status::invalid_argument("anime.status must be specified"));
    }
    Ok(find_specified_lookup_text(
        ANIME_STATUS_LOOKUP_TEXTS,
        anime_status,
        "AnimeStatus",
    ))
}

/// Converts an `AnimeRating` enum number into its lookup text.
///
/// # Errors
///
/// Returns `INVALID_ARGUMENT` for an unspecified or unknown value.
fn map_anime_rating(enum_number: i32) -> Result<&'static str, Status> {
    let Ok(anime_rating) = AnimeRating::try_from(enum_number) else {
        return Err(Status::invalid_argument(format!(
            "unknown AnimeRating number {enum_number}"
        )));
    };
    let known_rating = if anime_rating == AnimeRating::Unspecified {
        AnimeRating::Unknown
    } else {
        anime_rating
    };
    Ok(find_specified_lookup_text(
        ANIME_RATING_LOOKUP_TEXTS,
        known_rating,
        "AnimeRating",
    ))
}

/// Converts an `AnimeSeason` enum number into its lookup text.
///
/// # Errors
///
/// Returns `INVALID_ARGUMENT` for an unspecified or unknown value.
fn map_anime_season(enum_number: i32) -> Result<&'static str, Status> {
    let Ok(anime_season) = AnimeSeason::try_from(enum_number) else {
        return Err(Status::invalid_argument(format!(
            "unknown AnimeSeason number {enum_number}"
        )));
    };
    if anime_season == AnimeSeason::Unspecified {
        return Err(Status::invalid_argument(
            "anime.season must be specified when set",
        ));
    }
    Ok(find_specified_lookup_text(
        ANIME_SEASON_LOOKUP_TEXTS,
        anime_season,
        "AnimeSeason",
    ))
}

/// Converts a `SourceMaterialType` enum number into its lookup text.
///
/// # Errors
///
/// Returns `INVALID_ARGUMENT` for an unspecified or unknown value.
pub(crate) fn map_source_material_type(enum_number: i32) -> Result<&'static str, Status> {
    let Ok(source_material) = SourceMaterialType::try_from(enum_number) else {
        return Err(Status::invalid_argument(format!(
            "unknown SourceMaterialType number {enum_number}"
        )));
    };
    if source_material == SourceMaterialType::Unspecified {
        return Err(Status::invalid_argument(
            "anime.source_material must be specified when set",
        ));
    }
    Ok(find_specified_lookup_text(
        SOURCE_MATERIAL_TYPE_LOOKUP_TEXTS,
        source_material,
        "SourceMaterialType",
    ))
}

#[cfg(test)]
mod tests {
    use tonic::Code;

    use super::*;

    #[test]
    fn canonical_hash_is_order_independent_on_set_lists() {
        use crate::service::shared_proto::v1::{EntityType, EpisodeListEntry, PageKey};

        let mut first_anime = Anime {
            title: "Monster".to_owned(),
            r#type: AnimeType::Tv as i32,
            status: AnimeStatus::Finished as i32,
            genres: vec!["Thriller".to_owned(), "Drama".to_owned()],
            ..Anime::default()
        };
        let mut second_anime = first_anime.clone();
        second_anime.genres.reverse();

        let first_episodes = vec![EpisodeReference {
            episode_id: "01950000-0000-7000-8000-000000000001".to_owned(),
            episode_number: 1,
            title: "Ep 1".to_owned(),
            page_keys: vec![
                PageKey {
                    provider: "mal".to_owned(),
                    entity_type: EntityType::Episode as i32,
                    address: "21:1".to_owned(),
                },
                PageKey {
                    provider: "anilist".to_owned(),
                    entity_type: EntityType::Episode as i32,
                    address: "2".to_owned(),
                },
            ],
            list_entries: vec![
                EpisodeListEntry {
                    provider: "mal".to_owned(),
                    ..EpisodeListEntry::default()
                },
                EpisodeListEntry {
                    provider: "anilist".to_owned(),
                    ..EpisodeListEntry::default()
                },
            ],
        }];
        let mut second_episodes = first_episodes.clone();
        second_episodes[0].page_keys.reverse();
        second_episodes[0].list_entries.reverse();

        assert_eq!(
            compute_save_record_hash(&first_anime, &[], &first_episodes),
            compute_save_record_hash(&second_anime, &[], &second_episodes)
        );
        let relation_claims = vec![
            RelationClaim {
                provider: "mal".to_owned(),
                relation_type: 1,
                target_page: Some(PageKey {
                    provider: "mal".to_owned(),
                    entity_type: EntityType::Anime as i32,
                    address: "2".to_owned(),
                }),
            },
            RelationClaim {
                provider: "anilist".to_owned(),
                relation_type: 2,
                target_page: None,
            },
        ];
        let mut reversed_claims = relation_claims.clone();
        reversed_claims.reverse();
        assert_eq!(
            compute_save_record_hash(&first_anime, &relation_claims, &first_episodes),
            compute_save_record_hash(&first_anime, &reversed_claims, &first_episodes)
        );
        assert_ne!(
            compute_save_record_hash(&first_anime, &relation_claims, &first_episodes),
            compute_save_record_hash(&first_anime, &[], &first_episodes)
        );
        first_anime.title.push('!');
        assert_ne!(
            compute_save_record_hash(&first_anime, &[], &first_episodes),
            compute_save_record_hash(&second_anime, &[], &second_episodes)
        );
    }

    fn build_valid_anime() -> Anime {
        Anime {
            title: "Monster".to_owned(),
            r#type: AnimeType::Tv as i32,
            status: AnimeStatus::Finished as i32,
            ..Anime::default()
        }
    }

    #[test]
    fn build_save_anime_input_invalid_field_returns_invalid_argument() {
        use crate::service::shared_proto::v1::AiredDates;

        let invalid_id_status = build_save_anime_input(
            SaveAnimeRequest {
                anime_id: "not-a-uuid".to_owned(),
                expected_version: 1,
                anime: Some(build_valid_anime()),
                merge_version: 1,
                ..SaveAnimeRequest::default()
            },
            None,
        )
        .expect_err("an invalid work ID must be refused");
        assert_eq!(invalid_id_status.code(), Code::InvalidArgument);

        let breaking_changes: Vec<fn(&mut Anime)> = vec![
            |anime| anime.entity_type = 999,
            |anime| anime.entity_type = EntityType::Episode as i32,
            |anime| anime.r#type = 999,
            |anime| anime.r#type = AnimeType::Unspecified as i32,
            |anime| anime.status = 999,
            |anime| anime.status = AnimeStatus::Unspecified as i32,
            |anime| anime.rating = Some(999),
            |anime| anime.season = Some(999),
            |anime| anime.season = Some(AnimeSeason::Unspecified as i32),
            |anime| anime.source_material = Some(999),
            |anime| anime.source_material = Some(SourceMaterialType::Unspecified as i32),
            |anime| anime.year = Some(70_000),
            |anime| anime.month = Some(70_000),
            |anime| {
                anime.aired_dates = Some(AiredDates {
                    aired_from: Some(prost_types::Timestamp {
                        seconds: 0,
                        nanos: -1,
                    }),
                    aired_to: None,
                });
            },
            |anime| {
                anime.aired_dates = Some(AiredDates {
                    aired_from: None,
                    aired_to: Some(prost_types::Timestamp {
                        seconds: i64::MAX,
                        nanos: 0,
                    }),
                });
            },
        ];
        for break_rule in breaking_changes {
            let mut broken_anime = build_valid_anime();
            break_rule(&mut broken_anime);

            let status = build_save_anime_input(
                SaveAnimeRequest {
                    anime_id: Uuid::now_v7().to_string(),
                    expected_version: 1,
                    anime: Some(broken_anime),
                    merge_version: 1,
                    ..SaveAnimeRequest::default()
                },
                None,
            )
            .expect_err("an invalid work must be refused");

            assert_eq!(status.code(), Code::InvalidArgument);
        }
    }

    async fn assert_lookup_values_exist(
        pool: &sqlx::PgPool,
        lookup_table: &str,
        lookup_texts: Vec<&str>,
    ) {
        for lookup_text in lookup_texts {
            let is_stored: bool = sqlx::query_scalar(&format!(
                "SELECT EXISTS(SELECT 1 FROM {lookup_table} WHERE value = $1)"
            ))
            .bind(lookup_text)
            .fetch_one(pool)
            .await
            .expect("query the lookup table");
            assert!(is_stored, "{lookup_table} lacks {lookup_text:?}");
        }
    }

    fn map_every_enum_number(
        map_enum_number: fn(i32) -> Result<&'static str, Status>,
    ) -> Vec<&'static str> {
        (0..64)
            .filter_map(|enum_number| map_enum_number(enum_number).ok())
            .collect()
    }

    #[sqlx::test(migrations = "../postgres_db/migrations")]
    async fn lookup_text_every_proto_enum_value_exists_in_lookup_table(pool: sqlx::PgPool) {
        let type_texts = map_every_enum_number(map_anime_type);
        let status_texts = map_every_enum_number(map_anime_status);
        let rating_texts = map_every_enum_number(map_anime_rating);
        let season_texts = map_every_enum_number(map_anime_season);
        let source_material_texts = map_every_enum_number(map_source_material_type);

        assert_eq!(type_texts.len(), AnimeType::Other as usize);
        assert_eq!(status_texts.len(), AnimeStatus::Cancelled as usize);
        assert_eq!(season_texts.len(), AnimeSeason::Winter as usize);
        let accepted_entity_types = map_every_enum_number(|enum_number| {
            validate_entity_type(enum_number).map(|()| "ANIME")
        });
        assert_eq!(accepted_entity_types.len(), 2);
        assert_lookup_values_exist(&pool, "anime_type", type_texts).await;
        assert_lookup_values_exist(&pool, "anime_status", status_texts).await;
        assert_lookup_values_exist(&pool, "anime_rating", rating_texts).await;
        assert_lookup_values_exist(&pool, "anime_season", season_texts).await;
        assert_lookup_values_exist(&pool, "source_material_type", source_material_texts).await;
    }

    #[test]
    fn build_save_anime_input_maps_all_enums_dates_and_scores() {
        use std::collections::BTreeMap;

        use crate::service::shared_proto::v1::{
            AiredDates, AnimeImages, ScoreCalculations, ThemeEntry,
        };

        let mut titles = BTreeMap::new();
        titles.insert("de".to_owned(), "Monster".to_owned());

        let anime = Anime {
            title: "Monster".to_owned(),
            r#type: AnimeType::Tv as i32,
            status: AnimeStatus::Ongoing as i32,
            rating: Some(AnimeRating::Pg13 as i32),
            season: Some(AnimeSeason::Spring as i32),
            source_material: Some(SourceMaterialType::Koma4 as i32),
            year: Some(2004),
            month: Some(4),
            episode_count: 74,
            duration_seconds: Some(1440),
            is_nsfw: Some(false),
            country_of_origin: Some("JP".to_owned()),
            titles,
            synonyms: vec!["MONSTER".to_owned()],
            genres: vec!["Drama".to_owned(), "Mystery".to_owned()],
            tags: vec!["Crime".to_owned()],
            demographics: vec!["Seinen".to_owned()],
            content_warnings: vec!["Violence".to_owned()],
            themes: vec![ThemeEntry {
                name: "Psychological".to_owned(),
                description: Some("Mind games".to_owned()),
            }],
            images: Some(AnimeImages {
                covers: vec!["https://example.com/cover.jpg".to_owned()],
                posters: vec!["https://example.com/poster.jpg".to_owned()],
                banners: vec!["https://example.com/banner.jpg".to_owned()],
            }),
            aired_dates: Some(AiredDates {
                aired_from: Some(prost_types::Timestamp {
                    seconds: 1_081_296_000,
                    nanos: 0,
                }),
                aired_to: None,
            }),
            score: Some(ScoreCalculations {
                weighted: Some(8.5),
                mean: Some(8.4),
                median: Some(9.0),
            }),
            ..Anime::default()
        };
        let input = (build_save_anime_input(
            SaveAnimeRequest {
                anime_id: Uuid::now_v7().to_string(),
                expected_version: 2,
                anime: Some(anime),
                merge_version: 7,
                ..SaveAnimeRequest::default()
            },
            None,
        ))
        .expect("valid anime should map");
        assert_eq!(input.anime_type, "TV");
        assert_eq!(input.status, "ONGOING");
        assert_eq!(input.rating, "PG-13 - Teens 13 or older");
        assert_eq!(input.season.as_deref(), Some("SPRING"));
        assert_eq!(input.source_material.as_deref(), Some("4-KOMA"));
        assert_eq!(input.year, Some(2004));
        assert_eq!(input.month, Some(4));
        assert_eq!(input.score_weighted, Some(8.5));
        assert!(input.aired_from.is_some());
        assert_eq!(input.details.titles.len(), 1);
        assert_eq!(input.details.titles[0].language, "de");
        assert_eq!(input.details.synonyms, vec!["MONSTER"]);
        assert_eq!(input.details.genres, vec!["Drama", "Mystery"]);
        assert_eq!(input.details.tags, vec!["Crime"]);
        assert_eq!(input.details.demographics, vec!["Seinen"]);
        assert_eq!(input.details.content_warnings, vec!["Violence"]);
        assert_eq!(input.details.themes[0].name, "Psychological");
        assert_eq!(input.details.images.len(), 3);
        assert_eq!(input.details.images[0].image_type, "cover");
        assert_eq!(input.details.images[1].image_type, "poster");
        assert_eq!(input.details.images[2].image_type, "banner");
    }
}
