//! The lookup text the database stores for each proto enum value, shared by saves and reads.
//!
//! The database keeps provider-facing words such as `TV SPECIAL` or `PG-13 - Teens 13 or older`,
//! which are not the proto value names, so each enum has one table used in both directions.
//! [`define_lookup_texts`] turns each table into a complete `match`, so a value added to the
//! proto fails the build until its table lists it.

use tonic::Status;

use crate::service::shared_proto::v1::{
    AnimeRating, AnimeSeason, AnimeStatus, AnimeType, CharacterRole, CompanyRole,
    SourceMaterialType,
};

/// A proto enum whose values the database stores as lookup text.
pub(crate) trait LookupText: Sized + Copy + Into<i32> {
    /// Finds the lookup text the database stores for this value.
    ///
    /// Returns `None` for the unspecified value, which the database never stores.
    fn find_lookup_text(self) -> Option<&'static str>;

    /// Finds the value whose stored lookup text is `lookup_text`.
    ///
    /// Returns `None` when no value stores that text.
    fn find_by_lookup_text(lookup_text: &str) -> Option<Self>;
}

/// Implements [`LookupText`] for a proto enum from its table of `Variant => "TEXT"` pairs.
///
/// The value-to-text `match` names every variant and has no `_` arm, so the build fails when
/// the proto gains a value the table leaves out. Two variants given the same text make the
/// text-to-value `match` unreachable, which the workspace lints refuse.
macro_rules! define_lookup_texts {
    ($enum_type:ident { $($variant:ident => $lookup_text:literal),+ $(,)? }) => {
        impl LookupText for $enum_type {
            fn find_lookup_text(self) -> Option<&'static str> {
                match self {
                    Self::Unspecified => None,
                    $(Self::$variant => Some($lookup_text),)+
                }
            }

            fn find_by_lookup_text(lookup_text: &str) -> Option<Self> {
                match lookup_text {
                    $($lookup_text => Some(Self::$variant),)+
                    _ => None,
                }
            }
        }
    };
}

define_lookup_texts!(AnimeType {
    Movie => "MOVIE",
    Ona => "ONA",
    Ova => "OVA",
    Special => "SPECIAL",
    Tv => "TV",
    Unknown => "UNKNOWN",
    Music => "MUSIC",
    Pv => "PV",
    TvSpecial => "TV SPECIAL",
    TvShort => "TV SHORT",
    Cm => "CM",
    Other => "OTHER",
});

define_lookup_texts!(AnimeStatus {
    Finished => "FINISHED",
    Upcoming => "UPCOMING",
    Ongoing => "ONGOING",
    Unknown => "UNKNOWN",
    Cancelled => "CANCELLED",
});

define_lookup_texts!(AnimeRating {
    Unknown => "UNKNOWN",
    G => "G - All Ages",
    Pg => "PG - Children",
    Pg13 => "PG-13 - Teens 13 or older",
    R => "R - 17+ (violence & profanity)",
    Rplus => "R+ - Mild Nudity",
    Rx => "Rx - Hentai",
});

define_lookup_texts!(AnimeSeason {
    Spring => "SPRING",
    Summer => "SUMMER",
    Fall => "FALL",
    Winter => "WINTER",
});

define_lookup_texts!(SourceMaterialType {
    Original => "ORIGINAL",
    Manga => "MANGA",
    LightNovel => "LIGHT NOVEL",
    VisualNovel => "VISUAL NOVEL",
    Game => "GAME",
    WebManga => "WEB MANGA",
    WebNovel => "WEB NOVEL",
    Novel => "NOVEL",
    OneShot => "ONE SHOT",
    Doujinshi => "DOUJINSHI",
    Koma4 => "4-KOMA",
    MixedMedia => "MIXED MEDIA",
    Music => "MUSIC",
    Radio => "RADIO",
    Book => "BOOK",
    PictureBook => "PICTURE BOOK",
    CardGame => "CARD GAME",
    Other => "OTHER",
    Unknown => "UNKNOWN",
    Manhwa => "MANHWA",
    Manhua => "MANHUA",
    Comic => "COMIC",
    LiveAction => "LIVE ACTION",
    Illustration => "ILLUSTRATION",
    WesternMedia => "WESTERN MEDIA",
    Webtoon => "WEBTOON",
    FairyTale => "FAIRY TALE",
    ShortStory => "SHORT STORY",
});

define_lookup_texts!(CharacterRole {
    Main => "MAIN",
    Supporting => "SUPPORTING",
    Background => "BACKGROUND",
    Unknown => "UNKNOWN",
});

define_lookup_texts!(CompanyRole {
    Studio => "STUDIO",
    Producer => "PRODUCER",
    Licensor => "LICENSOR",
    Unknown => "UNKNOWN",
});

/// Converts stored lookup text into its proto enum number.
///
/// # Errors
///
/// Returns `INTERNAL` when the stored text has no proto value, since every stored value came
/// through the same table and a missing one means the table and the lookup rows disagree.
pub(crate) fn read_enum_number<E: LookupText>(
    lookup_text: &str,
    field_name: &str,
) -> Result<i32, Status> {
    E::find_by_lookup_text(lookup_text)
        .map(Into::into)
        .ok_or_else(|| {
            Status::internal(format!(
                "stored {field_name} {lookup_text:?} has no proto value"
            ))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_lookup_texts_round_trip<E>(enum_name: &str)
    where
        E: LookupText + TryFrom<i32> + std::fmt::Debug,
    {
        for enum_number in 1..256 {
            let Ok(enum_value) = E::try_from(enum_number) else {
                continue;
            };
            let lookup_text = enum_value.find_lookup_text().unwrap_or_default();
            assert_eq!(
                read_enum_number::<E>(lookup_text, enum_name).ok(),
                Some(enum_number),
                "{enum_name} {enum_value:?} has lookup text that reads back as the same value"
            );
        }
    }

    #[test]
    fn find_lookup_text_every_specified_value_round_trips() {
        assert_lookup_texts_round_trip::<AnimeType>("AnimeType");
        assert_lookup_texts_round_trip::<AnimeStatus>("AnimeStatus");
        assert_lookup_texts_round_trip::<AnimeRating>("AnimeRating");
        assert_lookup_texts_round_trip::<AnimeSeason>("AnimeSeason");
        assert_lookup_texts_round_trip::<SourceMaterialType>("SourceMaterialType");
        assert_lookup_texts_round_trip::<CharacterRole>("CharacterRole");
        assert_lookup_texts_round_trip::<CompanyRole>("CompanyRole");
    }

    #[test]
    fn find_lookup_text_unspecified_value_returns_none() {
        assert_eq!(AnimeType::Unspecified.find_lookup_text(), None);
        assert_eq!(CharacterRole::Unspecified.find_lookup_text(), None);
    }

    #[test]
    fn read_enum_number_unknown_lookup_text_returns_internal() {
        let status = read_enum_number::<AnimeType>("SERIES", "anime.type")
            .expect_err("unknown text must be refused");
        assert_eq!(status.code(), tonic::Code::Internal);
        assert!(status.message().contains("\"SERIES\""));
    }

    #[test]
    fn read_enum_number_proto_value_name_returns_internal() {
        let status = read_enum_number::<AnimeType>("ANIME_TYPE_TV", "anime.type")
            .expect_err("a proto value name is not lookup text");
        assert_eq!(status.code(), tonic::Code::Internal);
    }
}
