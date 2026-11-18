//! Exhaustive protobuf-to-SQL mapping and canonical hashing for `Anime`.

// Rust guideline compliant 2026-10-09

use chrono::{DateTime, Utc};
use postgres_db::{ANIME_HASH_FORMAT_V1, SaveAnimeInput};
use prost::Message as _;
use sha2::{Digest as _, Sha256};
use tonic::Status;
use uuid::Uuid;

use crate::service::shared_proto::v1::{
    Anime, AnimeRating, AnimeSeason, AnimeStatus, AnimeType, EntityType, SourceMaterialType,
};

/// Maps a `shared_proto.v1.Anime` and save metadata into [`SaveAnimeInput`].
///
/// Destructures every field of [`Anime`] without `..` and matches every proto
/// enum exhaustively without `_` so proto changes fail compilation until
/// handled.
///
/// # Errors
///
/// Returns [`Status::invalid_argument`] when `anime_id` is not a valid UUID,
/// required enums are unspecified or unknown, or numeric bounds are invalid.
#[expect(
    clippy::too_many_lines,
    reason = "Exhaustive destructuring of every Anime proto field without '..' is required by ECHO-98"
)]
pub fn build_save_anime_input(
    anime_id_str: &str,
    expected_version: i64,
    merge_version: i32,
    anime: Anime,
    traceparent: Option<String>,
) -> Result<SaveAnimeInput, Status> {
    let anime_id = Uuid::parse_str(anime_id_str)
        .map_err(|err| Status::invalid_argument(format!("invalid anime_id UUID: {err}")))?;

    let merged_record_hash = compute_anime_canonical_hash(&anime);

    let Anime {
        background,
        country_of_origin,
        duration,
        episode_count,
        id: _,
        entity_type,
        month,
        nsfw,
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
        content_warnings: _,
        demographics: _,
        ending_themes: _,
        genres: _,
        opening_themes: _,
        related_anime: _,
        related_source_material: _,
        sources: _,
        streaming_sources: _,
        synonyms: _,
        tags: _,
        themes: _,
        trailers: _,
        aired_dates,
        broadcast: _,
        external_sources: _,
        images: _,
        score,
        production_staff: _,
        statistics: _,
        hiatus: _,
        titles: _,
        companies: _,
        partial_sources: _,
    } = anime;

    validate_entity_type(entity_type)?;
    let anime_type = map_anime_type(r#type)?.to_owned();
    let status_str = map_anime_status(status)?.to_owned();
    let rating_str = match rating {
        Some(raw) => map_anime_rating(raw)?.to_owned(),
        None => "UNKNOWN".to_owned(),
    };
    let season_str = season.map(map_anime_season).transpose()?.map(str::to_owned);
    let source_material_str = source_material
        .map(map_source_material_type)
        .transpose()?
        .map(str::to_owned);

    let year_i16 = year
        .map(|val| {
            i16::try_from(val).map_err(|_| Status::invalid_argument("year out of smallint range"))
        })
        .transpose()?;
    let month_i16 = month
        .map(|val| {
            i16::try_from(val).map_err(|_| Status::invalid_argument("month out of smallint range"))
        })
        .transpose()?;

    let (aired_from, aired_to) = match aired_dates {
        Some(dates) => (
            dates
                .aired_from
                .map(|ts| parse_proto_timestamp("aired_from", ts))
                .transpose()?,
            dates
                .aired_to
                .map(|ts| parse_proto_timestamp("aired_to", ts))
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

    Ok(SaveAnimeInput {
        anime_id,
        expected_version,
        title,
        title_english,
        title_japanese,
        anime_type,
        status: status_str,
        rating: rating_str,
        season: season_str,
        source_material: source_material_str,
        year: year_i16,
        month: month_i16,
        episode_count,
        duration_seconds: duration,
        is_nsfw: nsfw,
        country_of_origin,
        synopsis,
        background,
        aired_from,
        aired_to,
        score_weighted,
        score_mean,
        score_median,
        merged_record_hash,
        hash_format: ANIME_HASH_FORMAT_V1,
        merge_version,
        traceparent,
    })
}

/// Computes a deterministic SHA-256 hex digest over a canonicalized [`Anime`].
#[must_use]
pub fn compute_anime_canonical_hash(anime: &Anime) -> String {
    let mut canonical = anime.clone();
    canonical.content_warnings.sort();
    canonical.demographics.sort();
    canonical.genres.sort();
    canonical.sources.sort();
    canonical.partial_sources.sort();
    canonical.synonyms.sort();
    canonical.tags.sort();

    let mut hasher = Sha256::new();
    hasher.update(canonical.encode_to_vec());
    let digest = hasher.finalize();
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(hex, "{byte:02x}");
    }
    hex
}

fn parse_proto_timestamp(
    field_name: &str,
    ts: prost_types::Timestamp,
) -> Result<DateTime<Utc>, Status> {
    let nanos = u32::try_from(ts.nanos)
        .map_err(|_| Status::invalid_argument(format!("{field_name}.nanos is invalid")))?;
    DateTime::<Utc>::from_timestamp(ts.seconds, nanos)
        .ok_or_else(|| Status::invalid_argument(format!("{field_name} is out of range")))
}

fn validate_entity_type(raw: i32) -> Result<(), Status> {
    let Ok(variant) = EntityType::try_from(raw) else {
        return Err(Status::invalid_argument(format!(
            "unknown EntityType number {raw}"
        )));
    };
    match variant {
        EntityType::Unspecified | EntityType::Anime => Ok(()),
        EntityType::Character | EntityType::Episode => Err(Status::invalid_argument(
            "SaveAnime requires an ANIME entity_type",
        )),
    }
}

fn map_anime_type(raw: i32) -> Result<&'static str, Status> {
    let Ok(variant) = AnimeType::try_from(raw) else {
        return Err(Status::invalid_argument(format!(
            "unknown AnimeType number {raw}"
        )));
    };
    match variant {
        AnimeType::Unspecified => Err(Status::invalid_argument("anime.type must be specified")),
        AnimeType::Movie => Ok("MOVIE"),
        AnimeType::Ona => Ok("ONA"),
        AnimeType::Ova => Ok("OVA"),
        AnimeType::Special => Ok("SPECIAL"),
        AnimeType::Tv => Ok("TV"),
        AnimeType::Unknown => Ok("UNKNOWN"),
        AnimeType::Music => Ok("MUSIC"),
        AnimeType::Pv => Ok("PV"),
        AnimeType::TvSpecial => Ok("TV SPECIAL"),
        AnimeType::TvShort => Ok("TV SHORT"),
        AnimeType::Cm => Ok("CM"),
        AnimeType::Other => Ok("OTHER"),
    }
}

fn map_anime_status(raw: i32) -> Result<&'static str, Status> {
    let Ok(variant) = AnimeStatus::try_from(raw) else {
        return Err(Status::invalid_argument(format!(
            "unknown AnimeStatus number {raw}"
        )));
    };
    match variant {
        AnimeStatus::Unspecified => Err(Status::invalid_argument("anime.status must be specified")),
        AnimeStatus::Finished => Ok("FINISHED"),
        AnimeStatus::Upcoming => Ok("UPCOMING"),
        AnimeStatus::Ongoing => Ok("ONGOING"),
        AnimeStatus::Unknown => Ok("UNKNOWN"),
        AnimeStatus::Cancelled => Ok("CANCELLED"),
    }
}

fn map_anime_rating(raw: i32) -> Result<&'static str, Status> {
    let Ok(variant) = AnimeRating::try_from(raw) else {
        return Err(Status::invalid_argument(format!(
            "unknown AnimeRating number {raw}"
        )));
    };
    match variant {
        AnimeRating::Unspecified | AnimeRating::Unknown => Ok("UNKNOWN"),
        AnimeRating::G => Ok("G - All Ages"),
        AnimeRating::Pg => Ok("PG - Children"),
        AnimeRating::Pg13 => Ok("PG-13 - Teens 13 or older"),
        AnimeRating::R => Ok("R - 17+ (violence & profanity)"),
        AnimeRating::Rplus => Ok("R+ - Mild Nudity"),
        AnimeRating::Rx => Ok("Rx - Hentai"),
    }
}

fn map_anime_season(raw: i32) -> Result<&'static str, Status> {
    let Ok(variant) = AnimeSeason::try_from(raw) else {
        return Err(Status::invalid_argument(format!(
            "unknown AnimeSeason number {raw}"
        )));
    };
    match variant {
        AnimeSeason::Unspecified => Err(Status::invalid_argument(
            "anime.season must be specified when set",
        )),
        AnimeSeason::Spring => Ok("SPRING"),
        AnimeSeason::Summer => Ok("SUMMER"),
        AnimeSeason::Fall => Ok("FALL"),
        AnimeSeason::Winter => Ok("WINTER"),
    }
}

fn map_source_material_type(raw: i32) -> Result<&'static str, Status> {
    let Ok(variant) = SourceMaterialType::try_from(raw) else {
        return Err(Status::invalid_argument(format!(
            "unknown SourceMaterialType number {raw}"
        )));
    };
    match variant {
        SourceMaterialType::Unspecified => Err(Status::invalid_argument(
            "anime.source_material must be specified when set",
        )),
        SourceMaterialType::Original => Ok("ORIGINAL"),
        SourceMaterialType::Manga => Ok("MANGA"),
        SourceMaterialType::LightNovel => Ok("LIGHT NOVEL"),
        SourceMaterialType::VisualNovel => Ok("VISUAL NOVEL"),
        SourceMaterialType::Game => Ok("GAME"),
        SourceMaterialType::WebManga => Ok("WEB MANGA"),
        SourceMaterialType::WebNovel => Ok("WEB NOVEL"),
        SourceMaterialType::Novel => Ok("NOVEL"),
        SourceMaterialType::OneShot => Ok("ONE SHOT"),
        SourceMaterialType::Doujinshi => Ok("DOUJINSHI"),
        SourceMaterialType::Koma4 => Ok("4-KOMA"),
        SourceMaterialType::MixedMedia => Ok("MIXED MEDIA"),
        SourceMaterialType::Music => Ok("MUSIC"),
        SourceMaterialType::Radio => Ok("RADIO"),
        SourceMaterialType::Book => Ok("BOOK"),
        SourceMaterialType::PictureBook => Ok("PICTURE BOOK"),
        SourceMaterialType::CardGame => Ok("CARD GAME"),
        SourceMaterialType::Other => Ok("OTHER"),
        SourceMaterialType::Unknown => Ok("UNKNOWN"),
        SourceMaterialType::Manhwa => Ok("MANHWA"),
        SourceMaterialType::Manhua => Ok("MANHUA"),
        SourceMaterialType::Comic => Ok("COMIC"),
        SourceMaterialType::LiveAction => Ok("LIVE ACTION"),
        SourceMaterialType::Illustration => Ok("ILLUSTRATION"),
        SourceMaterialType::WesternMedia => Ok("WESTERN MEDIA"),
        SourceMaterialType::Webtoon => Ok("WEBTOON"),
        SourceMaterialType::FairyTale => Ok("FAIRY TALE"),
        SourceMaterialType::ShortStory => Ok("SHORT STORY"),
    }
}

#[cfg(test)]
mod tests {
    use tonic::Code;

    use super::*;

    #[test]
    fn canonical_hash_is_order_independent_on_set_lists() {
        let mut a = Anime {
            title: "Monster".to_owned(),
            r#type: AnimeType::Tv as i32,
            status: AnimeStatus::Finished as i32,
            genres: vec!["Thriller".to_owned(), "Drama".to_owned()],
            ..Anime::default()
        };
        let mut b = a.clone();
        b.genres.reverse();
        assert_eq!(
            compute_anime_canonical_hash(&a),
            compute_anime_canonical_hash(&b)
        );
        a.title.push('!');
        assert_ne!(
            compute_anime_canonical_hash(&a),
            compute_anime_canonical_hash(&b)
        );
    }

    #[test]
    fn unknown_enum_numbers_return_invalid_argument() {
        let anime = Anime {
            title: "Monster".to_owned(),
            r#type: 999,
            status: AnimeStatus::Finished as i32,
            ..Anime::default()
        };
        let Err(err) = build_save_anime_input(&Uuid::now_v7().to_string(), 1, 1, anime, None)
        else {
            panic!("unknown enum must fail");
        };
        assert_eq!(err.code(), Code::InvalidArgument);
    }

    #[test]
    fn build_save_anime_input_maps_all_enums_dates_and_scores() {
        use crate::service::shared_proto::v1::{AiredDates, ScoreCalculations};

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
            duration: Some(1440),
            nsfw: Some(false),
            country_of_origin: Some("JP".to_owned()),
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
        let Ok(input) = build_save_anime_input(&Uuid::now_v7().to_string(), 2, 7, anime, None)
        else {
            panic!("valid anime should map");
        };
        assert_eq!(input.anime_type, "TV");
        assert_eq!(input.status, "ONGOING");
        assert_eq!(input.rating, "PG-13 - Teens 13 or older");
        assert_eq!(input.season.as_deref(), Some("SPRING"));
        assert_eq!(input.source_material.as_deref(), Some("4-KOMA"));
        assert_eq!(input.year, Some(2004));
        assert_eq!(input.month, Some(4));
        assert_eq!(input.score_weighted, Some(8.5));
        assert!(input.aired_from.is_some());
    }
}
