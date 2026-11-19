//! Protobuf-to-SQL mapping and canonical hashing for `Episode` and `EpisodeReference`.

// Rust guideline compliant 2026-10-09

use postgres_db::{
    EPISODE_HASH_FORMAT, EpisodeCharacterInput, EpisodeListEntryInput, EpisodeReferenceInput,
    EpisodeStreamingInput, EpisodeTitleInput, PageKeyInput, PageKind as DbPageKind,
    SaveEpisodeInput,
};
use prost::Message as _;
use sha2::{Digest as _, Sha256};
use tonic::Status;
use uuid::Uuid;

use crate::anime_mapping::parse_proto_timestamp;
use crate::anime_relations_mapping::{map_page_kind, parse_character_target_from_url};
use crate::service::shared_proto::v1::{
    CharacterRole, Episode, EpisodeCharacter, EpisodeListEntry, EpisodeReference, PageKey,
};

/// Maps repeated `EpisodeReference` proto messages into [`EpisodeReferenceInput`]s.
///
/// # Errors
///
/// Returns [`Status::invalid_argument`] when an `episode_id` is not a valid
/// UUID, a `page_key` is not `PageKind::Episode`, or a timestamp is invalid.
pub fn build_episode_references(
    episodes: Vec<EpisodeReference>,
) -> Result<Vec<EpisodeReferenceInput>, Status> {
    episodes
        .into_iter()
        .map(
            |EpisodeReference {
                 episode_id,
                 episode_number,
                 title,
                 page_keys,
                 list_entries,
             }| {
                let parsed_episode_id = Uuid::parse_str(&episode_id).map_err(|err| {
                    Status::invalid_argument(format!("invalid episode_id UUID: {err}"))
                })?;

                let mapped_page_keys = page_keys
                    .into_iter()
                    .map(
                        |PageKey {
                             provider,
                             page_kind,
                             address,
                         }| {
                            let mapped_kind = map_page_kind(page_kind)?;
                            if mapped_kind != DbPageKind::Episode {
                                return Err(Status::invalid_argument(
                                    "EpisodeReference.page_keys must use PageKind::Episode",
                                ));
                            }
                            Ok(PageKeyInput {
                                provider,
                                page_kind: mapped_kind,
                                address,
                            })
                        },
                    )
                    .collect::<Result<Vec<_>, Status>>()?;

                let mapped_list_entries = list_entries
                    .into_iter()
                    .map(
                        |EpisodeListEntry {
                             provider,
                             title: entry_title,
                             aired_at,
                             duration_seconds,
                             is_filler,
                             is_recap,
                         }| {
                            let parsed_aired_at = aired_at
                                .map(|ts| parse_proto_timestamp("list_entry.aired_at", ts))
                                .transpose()?;
                            Ok(EpisodeListEntryInput {
                                provider,
                                title: entry_title,
                                aired_at: parsed_aired_at,
                                duration_seconds,
                                is_filler,
                                is_recap,
                            })
                        },
                    )
                    .collect::<Result<Vec<_>, Status>>()?;

                Ok(EpisodeReferenceInput {
                    episode_id: parsed_episode_id,
                    episode_number,
                    title,
                    page_keys: mapped_page_keys,
                    list_entries: mapped_list_entries,
                })
            },
        )
        .collect()
}

/// Maps a [`SaveEpisodeRequest`] payload into [`SaveEpisodeInput`].
///
/// # Errors
///
/// Returns [`Status::invalid_argument`] if `episode_id` is not a valid UUID,
/// if any enum or timestamp is invalid, or if required fields are missing.
pub fn build_save_episode_input(
    episode_id_text: &str,
    expected_version: i64,
    merge_version: i32,
    episode: Episode,
    traceparent: Option<String>,
) -> Result<SaveEpisodeInput, Status> {
    let episode_id = Uuid::parse_str(episode_id_text)
        .map_err(|err| Status::invalid_argument(format!("invalid episode_id UUID: {err}")))?;

    let merged_record_hash = compute_save_episode_hash(&episode);

    let Episode {
        aired,
        anime_id: _,
        description,
        duration,
        episode_number,
        filler,
        id: _,
        entity_type: _,
        recap,
        score,
        season_number,
        synopsis,
        title,
        title_japanese,
        title_romaji,
        images,
        streaming,
        sources: _,
        characters,
        staff: _,
        titles,
    } = episode;

    let aired_at = aired
        .map(|ts| parse_proto_timestamp("episode.aired", ts))
        .transpose()?;

    let mapped_titles = titles
        .into_iter()
        .map(|(language, title_value)| EpisodeTitleInput {
            language,
            title: title_value,
        })
        .collect();

    let mapped_streaming = streaming
        .into_iter()
        .map(|(platform, url)| EpisodeStreamingInput { platform, url })
        .collect();

    let mapped_characters = map_episode_characters(characters)?;

    Ok(SaveEpisodeInput {
        episode_id,
        expected_version,
        episode_number,
        season_number,
        title,
        title_japanese,
        title_romaji,
        synopsis,
        description,
        aired_at,
        duration_seconds: duration,
        is_filler: filler,
        is_recap: recap,
        score: score.map(f64::from),
        titles: mapped_titles,
        images,
        streaming: mapped_streaming,
        characters: mapped_characters,
        merged_record_hash,
        hash_format: EPISODE_HASH_FORMAT,
        merge_version,
        traceparent,
    })
}

fn map_episode_characters(
    characters: Vec<EpisodeCharacter>,
) -> Result<Vec<EpisodeCharacterInput>, Status> {
    let mut mapped_characters = Vec::new();
    for EpisodeCharacter {
        name: _,
        role,
        sources,
        voice_actors: _,
    } in characters
    {
        let mapped_role = map_character_role(role)?.to_owned();
        for source_url in sources {
            if let Some((provider, address)) = parse_character_target_from_url(&source_url) {
                mapped_characters.push(EpisodeCharacterInput {
                    character_page_provider: provider.to_owned(),
                    character_page_address: address,
                    role: mapped_role.clone(),
                });
            }
        }
    }
    Ok(mapped_characters)
}

fn map_character_role(enum_number: i32) -> Result<&'static str, Status> {
    let Ok(role) = CharacterRole::try_from(enum_number) else {
        return Err(Status::invalid_argument(format!(
            "unknown CharacterRole number {enum_number}"
        )));
    };
    match role {
        CharacterRole::Unspecified => {
            Err(Status::invalid_argument("CharacterRole must be specified"))
        }
        CharacterRole::Main => Ok("MAIN"),
        CharacterRole::Supporting => Ok("SUPPORTING"),
        CharacterRole::Background => Ok("BACKGROUND"),
        CharacterRole::Unknown => Ok("UNKNOWN"),
    }
}

/// Computes a deterministic SHA-256 hex digest over a canonicalized [`Episode`].
#[must_use]
pub fn compute_save_episode_hash(episode: &Episode) -> String {
    let mut canonical_episode = episode.clone();
    canonical_episode.images.sort();
    canonical_episode.sources.sort();
    for character in &mut canonical_episode.characters {
        character.sources.sort();
    }
    canonical_episode.characters.sort_by(|left, right| {
        (&left.name, left.role, &left.sources).cmp(&(&right.name, right.role, &right.sources))
    });

    let mut hasher = Sha256::new();
    let encoded_bytes = canonical_episode.encode_to_vec();
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
    use std::collections::BTreeMap;

    use tonic::Code;

    use super::*;
    use crate::service::shared_proto::v1::PageKind;

    #[test]
    fn build_save_episode_input_maps_fields_and_is_order_independent() {
        let mut titles = BTreeMap::new();
        titles.insert("en".to_owned(), "Herr Dr. Tenma".to_owned());

        let mut streaming = BTreeMap::new();
        streaming.insert(
            "crunchyroll".to_owned(),
            "https://crunchyroll.com/watch/1".to_owned(),
        );

        let first_episode = Episode {
            episode_number: 1,
            season_number: Some(1),
            title: "Herr Dr. Tenma".to_owned(),
            duration: Some(1440),
            filler: false,
            recap: false,
            score: Some(8.6),
            images: vec![
                "https://example.com/a.jpg".to_owned(),
                "https://example.com/b.jpg".to_owned(),
            ],
            titles,
            streaming,
            characters: vec![EpisodeCharacter {
                name: "Kenzo Tenma".to_owned(),
                role: CharacterRole::Main as i32,
                sources: vec!["https://myanimelist.net/character/42".to_owned()],
                voice_actors: Vec::new(),
            }],
            ..Episode::default()
        };
        let mut second_episode = first_episode.clone();
        second_episode.images.reverse();

        assert_eq!(
            compute_save_episode_hash(&first_episode),
            compute_save_episode_hash(&second_episode)
        );

        let Ok(input) =
            build_save_episode_input(&Uuid::now_v7().to_string(), 1, 1, first_episode, None)
        else {
            panic!("valid episode should map");
        };
        assert_eq!(input.episode_number, 1);
        assert_eq!(input.duration_seconds, Some(1440));
        assert!(!input.is_filler);
        assert!(!input.is_recap);
        assert_eq!(input.titles.len(), 1);
        assert_eq!(input.streaming.len(), 1);
        assert_eq!(input.characters.len(), 1);
        assert_eq!(input.characters[0].character_page_provider, "mal");
        assert_eq!(input.characters[0].character_page_address, "42");
        assert_eq!(input.characters[0].role, "MAIN");
    }

    #[test]
    fn build_episode_references_rejects_non_episode_page_kind() {
        let Err(err) = build_episode_references(vec![EpisodeReference {
            episode_id: Uuid::now_v7().to_string(),
            episode_number: 1,
            title: "Ep 1".to_owned(),
            page_keys: vec![PageKey {
                provider: "mal".to_owned(),
                page_kind: PageKind::Anime as i32,
                address: "19".to_owned(),
            }],
            list_entries: Vec::new(),
        }]) else {
            panic!("non-episode page kind must fail");
        };
        assert_eq!(err.code(), Code::InvalidArgument);
    }
}
