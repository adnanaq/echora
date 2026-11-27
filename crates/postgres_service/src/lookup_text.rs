//! The lookup text the database stores for each proto enum value, shared by saves and reads.
//!
//! The database keeps provider-facing words such as `TV SPECIAL` or `PG-13 - Teens 13 or older`,
//! which are not the proto value names, so each enum has one table used in both directions.

use tonic::Status;

use crate::service::shared_proto::v1::{
    AnimeRating, AnimeSeason, AnimeStatus, AnimeType, CharacterRole, CompanyRole,
    SourceMaterialType,
};

/// Lookup text of every specified `AnimeType` value.
pub(crate) const ANIME_TYPE_LOOKUP_TEXTS: &[(AnimeType, &str)] = &[
    (AnimeType::Movie, "MOVIE"),
    (AnimeType::Ona, "ONA"),
    (AnimeType::Ova, "OVA"),
    (AnimeType::Special, "SPECIAL"),
    (AnimeType::Tv, "TV"),
    (AnimeType::Unknown, "UNKNOWN"),
    (AnimeType::Music, "MUSIC"),
    (AnimeType::Pv, "PV"),
    (AnimeType::TvSpecial, "TV SPECIAL"),
    (AnimeType::TvShort, "TV SHORT"),
    (AnimeType::Cm, "CM"),
    (AnimeType::Other, "OTHER"),
];

/// Lookup text of every specified `AnimeStatus` value.
pub(crate) const ANIME_STATUS_LOOKUP_TEXTS: &[(AnimeStatus, &str)] = &[
    (AnimeStatus::Finished, "FINISHED"),
    (AnimeStatus::Upcoming, "UPCOMING"),
    (AnimeStatus::Ongoing, "ONGOING"),
    (AnimeStatus::Unknown, "UNKNOWN"),
    (AnimeStatus::Cancelled, "CANCELLED"),
];

/// Lookup text of every specified `AnimeRating` value.
pub(crate) const ANIME_RATING_LOOKUP_TEXTS: &[(AnimeRating, &str)] = &[
    (AnimeRating::Unknown, "UNKNOWN"),
    (AnimeRating::G, "G - All Ages"),
    (AnimeRating::Pg, "PG - Children"),
    (AnimeRating::Pg13, "PG-13 - Teens 13 or older"),
    (AnimeRating::R, "R - 17+ (violence & profanity)"),
    (AnimeRating::Rplus, "R+ - Mild Nudity"),
    (AnimeRating::Rx, "Rx - Hentai"),
];

/// Lookup text of every specified `AnimeSeason` value.
pub(crate) const ANIME_SEASON_LOOKUP_TEXTS: &[(AnimeSeason, &str)] = &[
    (AnimeSeason::Spring, "SPRING"),
    (AnimeSeason::Summer, "SUMMER"),
    (AnimeSeason::Fall, "FALL"),
    (AnimeSeason::Winter, "WINTER"),
];

/// Lookup text of every specified `SourceMaterialType` value.
pub(crate) const SOURCE_MATERIAL_TYPE_LOOKUP_TEXTS: &[(SourceMaterialType, &str)] = &[
    (SourceMaterialType::Original, "ORIGINAL"),
    (SourceMaterialType::Manga, "MANGA"),
    (SourceMaterialType::LightNovel, "LIGHT NOVEL"),
    (SourceMaterialType::VisualNovel, "VISUAL NOVEL"),
    (SourceMaterialType::Game, "GAME"),
    (SourceMaterialType::WebManga, "WEB MANGA"),
    (SourceMaterialType::WebNovel, "WEB NOVEL"),
    (SourceMaterialType::Novel, "NOVEL"),
    (SourceMaterialType::OneShot, "ONE SHOT"),
    (SourceMaterialType::Doujinshi, "DOUJINSHI"),
    (SourceMaterialType::Koma4, "4-KOMA"),
    (SourceMaterialType::MixedMedia, "MIXED MEDIA"),
    (SourceMaterialType::Music, "MUSIC"),
    (SourceMaterialType::Radio, "RADIO"),
    (SourceMaterialType::Book, "BOOK"),
    (SourceMaterialType::PictureBook, "PICTURE BOOK"),
    (SourceMaterialType::CardGame, "CARD GAME"),
    (SourceMaterialType::Other, "OTHER"),
    (SourceMaterialType::Unknown, "UNKNOWN"),
    (SourceMaterialType::Manhwa, "MANHWA"),
    (SourceMaterialType::Manhua, "MANHUA"),
    (SourceMaterialType::Comic, "COMIC"),
    (SourceMaterialType::LiveAction, "LIVE ACTION"),
    (SourceMaterialType::Illustration, "ILLUSTRATION"),
    (SourceMaterialType::WesternMedia, "WESTERN MEDIA"),
    (SourceMaterialType::Webtoon, "WEBTOON"),
    (SourceMaterialType::FairyTale, "FAIRY TALE"),
    (SourceMaterialType::ShortStory, "SHORT STORY"),
];

/// Lookup text of every specified `CharacterRole` value.
pub(crate) const CHARACTER_ROLE_LOOKUP_TEXTS: &[(CharacterRole, &str)] = &[
    (CharacterRole::Main, "MAIN"),
    (CharacterRole::Supporting, "SUPPORTING"),
    (CharacterRole::Background, "BACKGROUND"),
    (CharacterRole::Unknown, "UNKNOWN"),
];

/// Lookup text of every specified `CompanyRole` value.
pub(crate) const COMPANY_ROLE_LOOKUP_TEXTS: &[(CompanyRole, &str)] = &[
    (CompanyRole::Studio, "STUDIO"),
    (CompanyRole::Producer, "PRODUCER"),
    (CompanyRole::Licensor, "LICENSOR"),
    (CompanyRole::Unknown, "UNKNOWN"),
];

/// Finds the lookup text of `enum_value`, or `None` when the table leaves it out.
pub(crate) fn find_lookup_text<E: Copy + PartialEq>(
    lookup_texts: &[(E, &'static str)],
    enum_value: E,
) -> Option<&'static str> {
    lookup_texts
        .iter()
        .find(|(table_value, _)| *table_value == enum_value)
        .map(|(_, lookup_text)| *lookup_text)
}

/// Returns the lookup text a save stores for a specified enum value.
///
/// Every caller refuses the unspecified value first, and the tests check that each table lists
/// every specified value, so a value missing from its table is a programming error.
///
/// # Panics
///
/// Panics when `lookup_texts` has no text for `enum_value`.
pub(crate) fn find_specified_lookup_text<E: Copy + PartialEq + std::fmt::Debug>(
    lookup_texts: &[(E, &'static str)],
    enum_value: E,
    enum_name: &str,
) -> &'static str {
    find_lookup_text(lookup_texts, enum_value).unwrap_or_else(|| {
        panic!("{enum_name} {enum_value:?} has no stored lookup text; add it to its lookup table")
    })
}

/// Converts stored lookup text into its proto enum number.
///
/// # Errors
///
/// Returns `INTERNAL` when the stored text has no proto value, since every stored value came
/// through the same table and a missing one means the table and the lookup rows disagree.
pub(crate) fn read_enum_number<E: Copy + Into<i32>>(
    lookup_texts: &[(E, &'static str)],
    lookup_text: &str,
    field_name: &str,
) -> Result<i32, Status> {
    lookup_texts
        .iter()
        .find(|(_, table_text)| *table_text == lookup_text)
        .map(|(enum_value, _)| (*enum_value).into())
        .ok_or_else(|| {
            Status::internal(format!(
                "stored {field_name} {lookup_text:?} has no proto value"
            ))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_table_covers_enum<E>(lookup_texts: &[(E, &'static str)], enum_name: &str)
    where
        E: Copy + PartialEq + Into<i32> + TryFrom<i32> + std::fmt::Debug,
    {
        let specified_values: Vec<(i32, E)> = (1..256)
            .filter_map(|enum_number| {
                E::try_from(enum_number)
                    .ok()
                    .map(|enum_value| (enum_number, enum_value))
            })
            .collect();
        assert_eq!(
            lookup_texts.len(),
            specified_values.len(),
            "{enum_name} lists every specified value once"
        );
        for (enum_number, enum_value) in specified_values {
            let lookup_text = find_lookup_text(lookup_texts, enum_value).unwrap_or_default();
            assert_eq!(
                read_enum_number(lookup_texts, lookup_text, enum_name).ok(),
                Some(enum_number),
                "{enum_name} {enum_value:?} has lookup text that reads back as the same value"
            );
        }
    }

    #[test]
    fn lookup_text_tables_cover_every_specified_value_and_round_trip() {
        assert_table_covers_enum(ANIME_TYPE_LOOKUP_TEXTS, "AnimeType");
        assert_table_covers_enum(ANIME_STATUS_LOOKUP_TEXTS, "AnimeStatus");
        assert_table_covers_enum(ANIME_RATING_LOOKUP_TEXTS, "AnimeRating");
        assert_table_covers_enum(ANIME_SEASON_LOOKUP_TEXTS, "AnimeSeason");
        assert_table_covers_enum(SOURCE_MATERIAL_TYPE_LOOKUP_TEXTS, "SourceMaterialType");
        assert_table_covers_enum(CHARACTER_ROLE_LOOKUP_TEXTS, "CharacterRole");
        assert_table_covers_enum(COMPANY_ROLE_LOOKUP_TEXTS, "CompanyRole");
    }

    #[test]
    fn read_enum_number_unknown_lookup_text_returns_internal() {
        let status = read_enum_number(ANIME_TYPE_LOOKUP_TEXTS, "SERIES", "anime.type")
            .expect_err("unknown text must be refused");
        assert_eq!(status.code(), tonic::Code::Internal);
        assert!(status.message().contains("\"SERIES\""));
    }

    #[test]
    fn read_enum_number_proto_value_name_returns_internal() {
        let status = read_enum_number(ANIME_TYPE_LOOKUP_TEXTS, "ANIME_TYPE_TV", "anime.type")
            .expect_err("a proto value name is not lookup text");
        assert_eq!(status.code(), tonic::Code::Internal);
    }

    #[test]
    #[should_panic(expected = "AnimeType Unspecified has no stored lookup text")]
    fn find_specified_lookup_text_value_missing_from_table_panics() {
        find_specified_lookup_text(ANIME_TYPE_LOOKUP_TEXTS, AnimeType::Unspecified, "AnimeType");
    }
}
