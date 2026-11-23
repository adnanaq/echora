//! Protobuf-to-SQL mapping and canonical hashing for `Character`.
//!
//! A character's `animeography` names each work by its page URLs. Each URL becomes a work
//! page key, which the database joins to the work that holds the page, so a work stored
//! after the character still reaches it. `mangaography` URLs become source material
//! identifiers in the same way, which the database joins to `source_material_xref`.

use postgres_db::{
    CHARACTER_HASH_FORMAT, CharacterAppearanceInput, CharacterMangaographyInput, SaveCharacterInput,
};
use prost::Message as _;
use sha2::{Digest as _, Sha256};
use tonic::Status;
use uuid::Uuid;

use crate::anime_relations_mapping::{
    parse_anime_target_from_url, parse_source_material_target_from_url,
};
use crate::episode_mapping::map_character_role;
use crate::service::shared_proto::v1::{Character, Ography};

/// Maps a `SaveCharacterRequest` payload into [`SaveCharacterInput`].
///
/// `roles` and `anime_ids` are read through the works the character's pages name, so the
/// save ignores them, and so does the change hash.
///
/// # Errors
///
/// Returns [`Status::invalid_argument`] when `character_id` is not a valid UUID, a role is
/// unknown, or `favorites` does not fit the stored column.
pub fn build_save_character_input(
    character_id_text: &str,
    expected_version: i64,
    merge_version: i32,
    character: Character,
    traceparent: Option<String>,
) -> Result<SaveCharacterInput, Status> {
    let character_id = Uuid::parse_str(character_id_text)
        .map_err(|err| Status::invalid_argument(format!("invalid character_id UUID: {err}")))?;
    let merged_record_hash = compute_save_character_hash(&character);

    let Character {
        description,
        favorites,
        id: _,
        entity_type: _,
        name,
        name_native,
        roles: _,
        anime_ids: _,
        animeography,
        traits,
        images,
        mangaography,
        name_variations,
        nicknames,
        sources: _,
        voice_actors: _,
        attributes,
        spoilers,
    } = character;

    let favorites = favorites
        .map(i32::try_from)
        .transpose()
        .map_err(|_| Status::invalid_argument("favorites does not fit a 32-bit count"))?;

    Ok(SaveCharacterInput {
        character_id,
        expected_version,
        name,
        name_native,
        description,
        favorites,
        name_variations,
        nicknames,
        traits,
        images,
        attributes,
        spoilers,
        animeography: map_character_animeography(animeography)?,
        mangaography: map_character_mangaography(mangaography)?,
        merged_record_hash,
        hash_format: CHARACTER_HASH_FORMAT,
        merge_version,
        traceparent,
    })
}

/// Converts the works a character appears in into one appearance per work page URL.
///
/// A URL that names no provider's anime page is left out, and so is an entry with none.
///
/// # Errors
///
/// Returns `INVALID_ARGUMENT` for an unknown character role.
fn map_character_animeography(
    animeography: Vec<Ography>,
) -> Result<Vec<CharacterAppearanceInput>, Status> {
    let mut appearances = Vec::new();
    for Ography {
        title: _,
        role,
        sources,
    } in animeography
    {
        let mapped_role = map_character_role(role)?;
        for source_url in sources {
            if let Some((provider, address)) = parse_anime_target_from_url(&source_url) {
                appearances.push(CharacterAppearanceInput {
                    anime_page_provider: provider.to_owned(),
                    anime_page_address: address,
                    role: mapped_role.to_owned(),
                });
            }
        }
    }
    Ok(appearances)
}

/// Converts the manga and novels a character appears in into source material identifiers.
///
/// Each URL that names a provider's manga or novel page becomes one identifier with the
/// entry's role, and any other URL is left out.
///
/// # Errors
///
/// Returns `INVALID_ARGUMENT` for an unknown character role.
fn map_character_mangaography(
    mangaography: Vec<Ography>,
) -> Result<Vec<CharacterMangaographyInput>, Status> {
    let mut entries = Vec::new();
    for Ography {
        title: _,
        role,
        sources,
    } in mangaography
    {
        let mapped_role = map_character_role(role)?;
        for source_url in sources {
            if let Some((provider, external_id)) =
                parse_source_material_target_from_url(&source_url)
            {
                entries.push(CharacterMangaographyInput {
                    source_material_provider: provider.to_owned(),
                    source_material_external_id: external_id,
                    role: mapped_role.to_owned(),
                });
            }
        }
    }
    Ok(entries)
}

/// Computes a deterministic SHA-256 hex digest over a canonicalized [`Character`].
///
/// Lists are sorted, so the same record hashes the same whatever order the merge gave.
/// `roles` and `anime_ids` are cleared, since the save reads them from the works the
/// character's pages name and never stores them.
#[must_use]
pub fn compute_save_character_hash(character: &Character) -> String {
    let mut canonical_character = character.clone();
    canonical_character.roles.clear();
    canonical_character.anime_ids.clear();
    for list in [
        &mut canonical_character.traits,
        &mut canonical_character.images,
        &mut canonical_character.name_variations,
        &mut canonical_character.nicknames,
        &mut canonical_character.sources,
    ] {
        list.sort();
    }
    for ography_list in [
        &mut canonical_character.animeography,
        &mut canonical_character.mangaography,
    ] {
        for entry in ography_list.iter_mut() {
            entry.sources.sort();
        }
        ography_list.sort_by(|left, right| {
            (&left.title, left.role, &left.sources).cmp(&(&right.title, right.role, &right.sources))
        });
    }

    let encoded_bytes = canonical_character.encode_to_vec();
    let mut hasher = Sha256::new();
    hasher.update(encoded_bytes.len().to_be_bytes());
    hasher.update(&encoded_bytes);
    let digest = hasher.finalize();

    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(hex, "{byte:02x}");
    }
    hex
}

#[cfg(test)]
mod tests {
    use tonic::Code;

    use std::collections::BTreeMap;

    use super::*;
    use crate::service::shared_proto::v1::CharacterRole;

    fn build_ography(title: &str, role: CharacterRole, sources: &[&str]) -> Ography {
        Ography {
            title: title.to_owned(),
            role: role as i32,
            sources: sources.iter().map(|source| (*source).to_owned()).collect(),
        }
    }

    fn build_character() -> Character {
        Character {
            description: Some("Captain of the Straw Hats".to_owned()),
            favorites: Some(150_000),
            id: None,
            entity_type: 0,
            name: "Monkey D. Luffy".to_owned(),
            name_native: Some("モンキー・D・ルフィ".to_owned()),
            roles: vec![CharacterRole::Main as i32],
            anime_ids: vec!["0190e4c2-0000-7000-8000-000000000001".to_owned()],
            animeography: vec![
                build_ography(
                    "One Piece",
                    CharacterRole::Main,
                    &[
                        "https://myanimelist.net/anime/21/One_Piece",
                        "https://anilist.co/anime/21",
                        "https://myanimelist.net/manga/13",
                    ],
                ),
                build_ography(
                    "No page",
                    CharacterRole::Supporting,
                    &["https://example.com/x"],
                ),
            ],
            traits: vec!["Rubber".to_owned(), "Captain".to_owned()],
            images: vec!["https://cdn.example/luffy.jpg".to_owned()],
            mangaography: vec![
                build_ography(
                    "One Piece",
                    CharacterRole::Main,
                    &[
                        "https://myanimelist.net/manga/13/One_Piece",
                        "https://anilist.co/manga/30013",
                        "https://myanimelist.net/anime/21",
                    ],
                ),
                build_ography("No page", CharacterRole::Main, &["https://example.com/y"]),
            ],
            name_variations: vec!["Luffy".to_owned()],
            nicknames: vec!["Straw Hat".to_owned()],
            sources: vec!["https://myanimelist.net/character/40".to_owned()],
            voice_actors: Vec::new(),
            attributes: BTreeMap::from([("Age".to_owned(), "19".to_owned())]),
            spoilers: BTreeMap::from([("description".to_owned(), "Son of Dragon".to_owned())]),
        }
    }

    #[test]
    fn build_save_character_input_maps_fields_anime_pages_and_source_material() {
        let character_id = Uuid::now_v7();

        let input = build_save_character_input(
            &character_id.to_string(),
            3,
            2,
            build_character(),
            Some("00-trace".to_owned()),
        )
        .expect("map the character");

        assert_eq!(input.character_id, character_id);
        assert_eq!((input.expected_version, input.merge_version), (3, 2));
        assert_eq!(input.favorites, Some(150_000));
        assert_eq!(input.hash_format, CHARACTER_HASH_FORMAT);
        assert_eq!(input.traceparent.as_deref(), Some("00-trace"));
        assert_eq!(
            input.attributes,
            BTreeMap::from([("Age".to_owned(), "19".to_owned())])
        );
        assert_eq!(
            input.spoilers,
            BTreeMap::from([("description".to_owned(), "Son of Dragon".to_owned())])
        );
        assert_eq!(
            input.animeography,
            vec![
                CharacterAppearanceInput {
                    anime_page_provider: "mal".to_owned(),
                    anime_page_address: "21".to_owned(),
                    role: "MAIN".to_owned(),
                },
                CharacterAppearanceInput {
                    anime_page_provider: "anilist".to_owned(),
                    anime_page_address: "21".to_owned(),
                    role: "MAIN".to_owned(),
                },
            ]
        );
        assert_eq!(
            input.mangaography,
            vec![
                CharacterMangaographyInput {
                    source_material_provider: "mal".to_owned(),
                    source_material_external_id: "13".to_owned(),
                    role: "MAIN".to_owned(),
                },
                CharacterMangaographyInput {
                    source_material_provider: "anilist".to_owned(),
                    source_material_external_id: "30013".to_owned(),
                    role: "MAIN".to_owned(),
                },
            ]
        );
    }

    #[test]
    fn build_save_character_input_invalid_request_returns_invalid_argument() {
        let mut unspecified_role = build_character();
        unspecified_role.animeography[0].role = CharacterRole::Unspecified as i32;
        let mut unknown_manga_role = build_character();
        unknown_manga_role.mangaography[0].role = 99;
        let mut oversized_favorites = build_character();
        oversized_favorites.favorites = Some(i64::from(i32::MAX) + 1);
        let valid_id = Uuid::now_v7().to_string();

        for (character_id_text, character) in [
            ("not-a-uuid", build_character()),
            (valid_id.as_str(), unspecified_role),
            (valid_id.as_str(), unknown_manga_role),
            (valid_id.as_str(), oversized_favorites),
        ] {
            let status = build_save_character_input(character_id_text, 1, 1, character, None)
                .expect_err("the request is refused");
            assert_eq!(status.code(), Code::InvalidArgument, "{status:?}");
        }
    }

    #[test]
    fn compute_save_character_hash_ignores_order_and_read_fields_and_tracks_content() {
        let character = build_character();
        let mut reordered_character = character.clone();
        reordered_character.traits.reverse();
        reordered_character.animeography.reverse();
        reordered_character.animeography[1].sources.reverse();
        reordered_character.mangaography.reverse();
        reordered_character.roles = vec![CharacterRole::Background as i32];
        reordered_character.anime_ids.clear();
        let mut renamed_character = character.clone();
        renamed_character.name = "Luffy".to_owned();
        let mut recast_character = character.clone();
        recast_character.animeography[0].role = CharacterRole::Supporting as i32;

        let character_hash = compute_save_character_hash(&character);

        assert_eq!(character_hash.len(), 64);
        assert_eq!(
            character_hash,
            compute_save_character_hash(&reordered_character)
        );
        assert_ne!(
            character_hash,
            compute_save_character_hash(&renamed_character)
        );
        assert_ne!(
            character_hash,
            compute_save_character_hash(&recast_character)
        );
    }
}
