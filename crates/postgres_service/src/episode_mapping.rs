//! Protobuf-to-SQL mapping and canonical hashing for `Episode` and `EpisodeReference`.

use postgres_db::{
    EPISODE_HASH_FORMAT, EntityType as DbEntityType, EpisodeCharacterInput, EpisodeListEntryInput,
    EpisodeReferenceInput, EpisodeStreamingInput, EpisodeTitleInput, PageSource, SaveEpisodeInput,
};
use prost::Message as _;
use sha2::{Digest as _, Sha256};
use tonic::Status;
use uuid::Uuid;

use crate::anime_mapping::parse_proto_timestamp;
use crate::anime_relations_mapping::{build_page_source_from_url, parse_character_target_from_url};
use crate::lookup_text::LookupText as _;
use crate::service::parse_page_sources;
use crate::service::postgres_service_proto::v1::SaveEpisodeRequest;
use crate::service::shared_proto::v1::{
    CharacterRole, Episode, EpisodeCharacter, EpisodeListEntry, EpisodeReference,
    PageSource as ProtoPageSource,
};
use crate::service::{map_db_error, to_proto_episode_list_entry, to_proto_page_key};

/// Maps repeated `EpisodeReference` proto messages into [`EpisodeReferenceInput`]s.
///
/// # Errors
///
/// Returns [`Status::invalid_argument`] when an `episode_id` is not a valid
/// UUID, a provider is unknown, a `page_key` is not `EntityType::Episode`, or a timestamp
/// is invalid.
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
                 page_sources,
                 list_entries,
             }| {
                let parsed_episode_id = Uuid::parse_str(&episode_id).map_err(|err| {
                    Status::invalid_argument(format!("invalid episode_id UUID: {err}"))
                })?;

                let mapped_page_sources = parse_page_sources(page_sources)?;
                if mapped_page_sources
                    .iter()
                    .any(|page_source| page_source.page_key.entity_type() != DbEntityType::Episode)
                {
                    return Err(Status::invalid_argument(
                        "EpisodeReference.page_sources must use EntityType::Episode",
                    ));
                }

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
                                .map(|proto_timestamp| {
                                    parse_proto_timestamp("list_entry.aired_at", proto_timestamp)
                                })
                                .transpose()?;
                            Ok(EpisodeListEntryInput {
                                provider: provider.parse().map_err(|err| map_db_error(&err))?,
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
                    page_sources: mapped_page_sources,
                    list_entries: mapped_list_entries,
                })
            },
        )
        .collect()
}

/// Converts a known episode of a work into its proto reference for `LoadAnime`.
///
/// The work's merge reads these to keep each episode's ID, matching by episode number or
/// page key, so the reference carries the episode's pages with their stored URLs and its
/// list entries.
pub fn to_proto_episode_reference(known_episode: EpisodeReferenceInput) -> EpisodeReference {
    let EpisodeReferenceInput {
        episode_id,
        episode_number,
        title,
        page_sources,
        list_entries,
    } = known_episode;
    EpisodeReference {
        episode_id: episode_id.to_string(),
        episode_number,
        title,
        page_sources: page_sources
            .into_iter()
            .map(
                |PageSource {
                     page_key,
                     source,
                     title,
                 }| ProtoPageSource {
                    page_key: Some(to_proto_page_key(&page_key)),
                    source: source.map(String::from).unwrap_or_default(),
                    title: title.unwrap_or_default(),
                },
            )
            .collect(),
        list_entries: list_entries
            .into_iter()
            .map(to_proto_episode_list_entry)
            .collect(),
    }
}

/// Maps a [`SaveEpisodeRequest`] payload into [`SaveEpisodeInput`].
///
/// # Errors
///
/// Returns [`Status::invalid_argument`] if `episode_id` is not a valid UUID, the episode is
/// missing, an enum or timestamp is invalid, or a page source has no valid page key.
pub fn build_save_episode_input(
    request: SaveEpisodeRequest,
    traceparent: Option<String>,
) -> Result<SaveEpisodeInput, Status> {
    let SaveEpisodeRequest {
        episode_id: episode_id_text,
        expected_version,
        episode,
        merge_version,
        page_sources,
    } = request;
    let episode_id = Uuid::parse_str(&episode_id_text)
        .map_err(|err| Status::invalid_argument(format!("invalid episode_id UUID: {err}")))?;
    let episode = episode.ok_or_else(|| Status::invalid_argument("episode must be provided"))?;
    let page_sources = parse_page_sources(page_sources)?;

    let merged_record_hash = compute_save_episode_hash(&episode);

    let Episode {
        aired_at,
        anime_id: _,
        description,
        duration_seconds,
        episode_number,
        is_filler,
        id: _,
        entity_type: _,
        is_recap,
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

    let aired_at = aired_at
        .map(|proto_timestamp| parse_proto_timestamp("episode.aired_at", proto_timestamp))
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
        .map(|(platform, source)| EpisodeStreamingInput { platform, source })
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
        duration_seconds,
        is_filler,
        is_recap,
        score: score.map(f64::from),
        titles: mapped_titles,
        images,
        streaming: mapped_streaming,
        characters: mapped_characters,
        merged_record_hash,
        hash_format: EPISODE_HASH_FORMAT,
        merge_version,
        traceparent,
        page_sources,
    })
}

/// Converts an episode's characters into one appearance per character page they name.
///
/// A character whose sources name no character page is left out.
///
/// # Errors
///
/// Returns `INVALID_ARGUMENT` for an unknown character role.
fn map_episode_characters(
    characters: Vec<EpisodeCharacter>,
) -> Result<Vec<EpisodeCharacterInput>, Status> {
    let mut mapped_characters = Vec::new();
    for EpisodeCharacter {
        name,
        role,
        sources,
        voice_actors: _,
    } in characters
    {
        let mapped_role = map_character_role(role)?.to_owned();
        for source_url in sources {
            if let Some(character_page_source) = build_page_source_from_url(
                &source_url,
                &name,
                DbEntityType::Character,
                parse_character_target_from_url,
            ) {
                mapped_characters.push(EpisodeCharacterInput {
                    character_page_source,
                    role: mapped_role.clone(),
                });
            }
        }
    }
    Ok(mapped_characters)
}

/// Converts a `CharacterRole` enum number into its lookup text.
///
/// # Errors
///
/// Returns `INVALID_ARGUMENT` for an unspecified or unknown value.
pub(crate) fn map_character_role(enum_number: i32) -> Result<&'static str, Status> {
    let Ok(role) = CharacterRole::try_from(enum_number) else {
        return Err(Status::invalid_argument(format!(
            "unknown CharacterRole number {enum_number}"
        )));
    };
    role.find_lookup_text()
        .ok_or_else(|| Status::invalid_argument("CharacterRole must be specified"))
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

    use postgres_db::{PageKey, Provider};
    use url::Url;

    use super::*;
    use crate::service::shared_proto::v1::{EntityType, PageKey as ProtoPageKey};

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "one episode covers every mapped field in two orders"
    )]
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
            duration_seconds: Some(1440),
            is_filler: false,
            is_recap: false,
            score: Some(8.6),
            images: vec![
                "https://example.com/a.jpg".to_owned(),
                "https://example.com/b.jpg".to_owned(),
            ],
            titles,
            streaming,
            characters: vec![
                EpisodeCharacter {
                    name: "Kenzo Tenma".to_owned(),
                    role: CharacterRole::Main as i32,
                    sources: vec!["https://myanimelist.net/character/42/Kenzo_Tenma".to_owned()],
                    voice_actors: Vec::new(),
                },
                EpisodeCharacter {
                    name: "Nina Fortner".to_owned(),
                    role: CharacterRole::Supporting as i32,
                    sources: vec!["https://myanimelist.net/character/43".to_owned()],
                    voice_actors: Vec::new(),
                },
                EpisodeCharacter {
                    name: "Passer-by".to_owned(),
                    role: CharacterRole::Background as i32,
                    sources: vec!["https://anilist.co/character/44".to_owned()],
                    voice_actors: Vec::new(),
                },
                EpisodeCharacter {
                    name: "Unnamed".to_owned(),
                    role: CharacterRole::Unknown as i32,
                    sources: vec!["https://example.com/not-a-character".to_owned()],
                    voice_actors: Vec::new(),
                },
            ],
            ..Episode::default()
        };
        let mut second_episode = first_episode.clone();
        second_episode.images.reverse();
        second_episode.characters.reverse();

        assert_eq!(
            compute_save_episode_hash(&first_episode),
            compute_save_episode_hash(&second_episode)
        );
        let mut renumbered_episode = first_episode.clone();
        renumbered_episode.episode_number += 1;
        let mut reseasoned_episode = first_episode.clone();
        reseasoned_episode.season_number = Some(2);
        assert_ne!(
            compute_save_episode_hash(&first_episode),
            compute_save_episode_hash(&renumbered_episode)
        );
        assert_ne!(
            compute_save_episode_hash(&first_episode),
            compute_save_episode_hash(&reseasoned_episode)
        );

        let input = build_save_episode_input(
            SaveEpisodeRequest {
                episode_id: Uuid::now_v7().to_string(),
                expected_version: 1,
                episode: Some(first_episode),
                merge_version: 1,
                page_sources: Vec::new(),
            },
            None,
        )
        .expect("valid episode should map");
        assert_eq!(input.episode_number, 1);
        assert_eq!(input.duration_seconds, Some(1440));
        assert!(!input.is_filler);
        assert!(!input.is_recap);
        assert_eq!(input.titles.len(), 1);
        assert_eq!(input.streaming.len(), 1);
        let character_pages: Vec<(&str, &str, &str, &str)> = input
            .characters
            .iter()
            .map(|character| {
                let page_source = &character.character_page_source;
                (
                    page_source.page_key.provider().as_str(),
                    page_source.page_key.address().as_str(),
                    page_source.source.as_ref().map_or("", Url::as_str),
                    character.role.as_str(),
                )
            })
            .collect();
        assert_eq!(
            character_pages,
            vec![
                (
                    "mal",
                    "42",
                    "https://myanimelist.net/character/42/Kenzo_Tenma",
                    "MAIN"
                ),
                (
                    "mal",
                    "43",
                    "https://myanimelist.net/character/43",
                    "SUPPORTING"
                ),
                (
                    "anilist",
                    "44",
                    "https://anilist.co/character/44",
                    "BACKGROUND"
                ),
            ]
        );
    }

    fn first_page_key(reference: &mut EpisodeReference) -> &mut ProtoPageKey {
        reference.page_sources[0]
            .page_key
            .as_mut()
            .expect("the valid reference names one page")
    }

    fn build_valid_reference() -> EpisodeReference {
        EpisodeReference {
            episode_id: Uuid::now_v7().to_string(),
            episode_number: 1,
            title: "Placeholder".to_owned(),
            page_sources: vec![ProtoPageSource {
                page_key: Some(ProtoPageKey {
                    provider: "mal".to_owned(),
                    entity_type: EntityType::Episode as i32,
                    address: "21:1".to_owned(),
                }),
                source: String::new(),
                title: String::new(),
            }],
            list_entries: vec![EpisodeListEntry {
                provider: "mal".to_owned(),
                ..EpisodeListEntry::default()
            }],
        }
    }

    #[test]
    fn build_episode_references_invalid_reference_returns_invalid_argument() {
        assert!(build_episode_references(vec![build_valid_reference()]).is_ok());
        let breaking_changes: Vec<fn(&mut EpisodeReference)> = vec![
            |reference| reference.episode_id = "not-a-uuid".to_owned(),
            |reference| first_page_key(reference).provider = "not_a_provider".to_owned(),
            |reference| first_page_key(reference).entity_type = EntityType::Anime as i32,
            |reference| first_page_key(reference).address = "not-numeric".to_owned(),
            |reference| reference.page_sources[0].page_key = None,
            |reference| reference.list_entries[0].provider = "not_a_provider".to_owned(),
            |reference| {
                reference.list_entries[0].aired_at = Some(prost_types::Timestamp {
                    seconds: i64::MAX,
                    nanos: 0,
                });
            },
        ];

        for break_rule in breaking_changes {
            let mut broken_reference = build_valid_reference();
            break_rule(&mut broken_reference);

            let status = build_episode_references(vec![broken_reference])
                .expect_err("an invalid reference must be refused");

            assert_eq!(status.code(), Code::InvalidArgument);
        }
    }

    #[test]
    fn build_save_episode_input_invalid_input_returns_invalid_argument() {
        let valid_episode = Episode {
            episode_number: 1,
            title: "Placeholder".to_owned(),
            ..Episode::default()
        };
        let invalid_cases = vec![
            ("not-a-uuid".to_owned(), valid_episode.clone()),
            (
                Uuid::now_v7().to_string(),
                Episode {
                    aired_at: Some(prost_types::Timestamp {
                        seconds: i64::MAX,
                        nanos: 0,
                    }),
                    ..valid_episode.clone()
                },
            ),
            (
                Uuid::now_v7().to_string(),
                Episode {
                    characters: vec![EpisodeCharacter {
                        role: 99,
                        ..EpisodeCharacter::default()
                    }],
                    ..valid_episode.clone()
                },
            ),
            (
                Uuid::now_v7().to_string(),
                Episode {
                    characters: vec![EpisodeCharacter {
                        role: CharacterRole::Unspecified as i32,
                        ..EpisodeCharacter::default()
                    }],
                    ..valid_episode
                },
            ),
        ];

        for (episode_id, episode) in invalid_cases {
            let status = build_save_episode_input(
                SaveEpisodeRequest {
                    episode_id,
                    expected_version: 1,
                    episode: Some(episode),
                    merge_version: 1,
                    page_sources: Vec::new(),
                },
                None,
            )
            .expect_err("an invalid episode must be refused");
            assert_eq!(status.code(), Code::InvalidArgument);
        }
        let missing_episode_status = build_save_episode_input(
            SaveEpisodeRequest {
                episode_id: Uuid::now_v7().to_string(),
                expected_version: 1,
                episode: None,
                merge_version: 1,
                page_sources: Vec::new(),
            },
            None,
        )
        .expect_err("a request without an episode is refused");
        assert_eq!(missing_episode_status.code(), Code::InvalidArgument);
    }

    #[test]
    fn to_proto_episode_reference_round_trips_through_build_episode_references() {
        let known_episode = EpisodeReferenceInput {
            episode_id: Uuid::now_v7(),
            episode_number: 3,
            title: "Placeholder episode 3".to_owned(),
            page_sources: vec![PageSource {
                page_key: PageKey::new(Provider::AniDb, DbEntityType::Episode, "3").unwrap(),
                source: Some(Url::parse("https://anidb.net/episode/3").expect("valid URL")),
                title: None,
            }],
            list_entries: vec![EpisodeListEntryInput {
                provider: Provider::Mal,
                title: Some("Placeholder episode 3".to_owned()),
                aired_at: chrono::DateTime::from_timestamp(1_700_000_000, 0),
                duration_seconds: Some(1440),
                is_filler: Some(true),
                is_recap: None,
            }],
        };

        let proto_reference = to_proto_episode_reference(known_episode.clone());
        let mapped_references = build_episode_references(vec![proto_reference]).unwrap();

        assert_eq!(mapped_references, vec![known_episode]);
    }
}
