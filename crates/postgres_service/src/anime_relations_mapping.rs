//! Protobuf-to-SQL mapping for anime relations, relation claims, source material,
//! and companies.

use std::borrow::Cow;
use std::collections::BTreeMap;

use postgres_db::{
    AnimeCompanyInput, AnimeRelationClaimInput, AnimeRelationInput, AnimeRelationsAndGraphInput,
    AnimeSourceMaterialInput, CompanyXrefInput, EntityType as DbEntityType, PageKey, PageSource,
    Provider, SourceMaterialXrefInput,
};
use tonic::Status;
use url::Url;

use crate::anime_mapping::map_source_material_type;
use crate::lookup_text::LookupText as _;
use crate::service::shared_proto::v1::{
    AnimeRelationType, AnimeStatus, CompanyEntry, CompanyRole, EntityType, PageKey as ProtoPageKey,
    RelatedAnimeList, RelatedSourceMaterial, RelatedSourceMaterialList, RelationClaim,
};
use crate::service::{map_db_error, parse_page_title};

/// Collects a work's relations, claims, source material and companies for the save.
///
/// # Errors
///
/// Returns `INVALID_ARGUMENT` for an unknown relation type, role or entity type, or a related
/// page whose provider or address is not valid.
pub(crate) fn build_relations_and_graph(
    related_anime: BTreeMap<String, RelatedAnimeList>,
    related_source_material: BTreeMap<String, RelatedSourceMaterialList>,
    companies: Vec<CompanyEntry>,
    relation_claims: Vec<RelationClaim>,
) -> Result<AnimeRelationsAndGraphInput, Status> {
    let mapped_relations = map_related_anime(related_anime)?;
    let mapped_claims = map_relation_claims(relation_claims)?;
    let mapped_source_materials = map_related_source_material(related_source_material)?;
    let mapped_companies = map_companies(companies)?;

    Ok(AnimeRelationsAndGraphInput {
        relations: mapped_relations,
        relation_claims: mapped_claims,
        source_materials: mapped_source_materials,
        companies: mapped_companies,
    })
}

/// Converts related works into relations keyed by the target's page.
///
/// # Errors
///
/// Returns `INVALID_ARGUMENT` for an unknown relation type.
fn map_related_anime(
    related_anime: BTreeMap<String, RelatedAnimeList>,
) -> Result<Vec<AnimeRelationInput>, Status> {
    let mut anime_relations = Vec::new();
    for (raw_relation_type, RelatedAnimeList { items }) in related_anime {
        let relation_type = normalize_anime_relation_type(&raw_relation_type)?;
        for item in items {
            for source_url in item.sources {
                if let Some(target_page_source) = build_page_source_from_url(
                    &source_url,
                    &item.title,
                    DbEntityType::Anime,
                    parse_anime_target_from_url,
                ) {
                    anime_relations.push(AnimeRelationInput {
                        relation_type: relation_type.to_owned(),
                        target_page_source,
                    });
                }
            }
        }
    }
    Ok(anime_relations)
}

/// Converts each provider's relation claims into claim inputs.
///
/// # Errors
///
/// Returns `INVALID_ARGUMENT` for an unknown provider, relation type or entity type, or a
/// target page that is missing or not valid.
fn map_relation_claims(claims: Vec<RelationClaim>) -> Result<Vec<AnimeRelationClaimInput>, Status> {
    claims
        .into_iter()
        .map(
            |RelationClaim {
                 provider,
                 relation_type,
                 target_page,
             }| {
                let claiming_provider: Provider =
                    provider.parse().map_err(|err| map_db_error(&err))?;
                let relation_type_text = map_anime_relation_type(relation_type)?.to_owned();
                let ProtoPageKey {
                    provider: target_provider,
                    entity_type,
                    address,
                } = target_page.ok_or_else(|| {
                    Status::invalid_argument("relation_claim requires target_page")
                })?;

                let parsed_target_provider: Provider =
                    target_provider.parse().map_err(|err| map_db_error(&err))?;
                let mapped_entity_type = map_entity_type(entity_type)?;
                Ok(AnimeRelationClaimInput {
                    provider: claiming_provider,
                    relation_type: relation_type_text,
                    target_page: PageKey::new(parsed_target_provider, mapped_entity_type, address)
                        .map_err(|err| map_db_error(&err))?,
                })
            },
        )
        .collect()
}

/// Converts related source material into source material inputs with cross-references.
///
/// # Errors
///
/// Returns `INVALID_ARGUMENT` for an unknown relation type, material type or status.
fn map_related_source_material(
    related_source_material: BTreeMap<String, RelatedSourceMaterialList>,
) -> Result<Vec<AnimeSourceMaterialInput>, Status> {
    let mut source_materials = Vec::new();
    for (raw_relation_type, RelatedSourceMaterialList { items }) in related_source_material {
        let relation_type = normalize_source_material_relation_type(&raw_relation_type)?.to_owned();
        for RelatedSourceMaterial {
            title,
            r#type,
            sources,
            status,
            score,
            images,
            chapters,
            volumes,
        } in items
        {
            let source_material_type = map_source_material_type(r#type)?.to_owned();
            let publication_status = status
                .map(map_anime_status_optional)
                .transpose()?
                .flatten()
                .map(str::to_owned);
            let xrefs: Vec<SourceMaterialXrefInput> = sources
                .into_iter()
                .filter_map(|source_url| build_source_material_xref_from_url(source_url, ""))
                .collect();
            if xrefs.is_empty() {
                continue;
            }
            source_materials.push(AnimeSourceMaterialInput {
                relation_type: relation_type.clone(),
                title,
                title_english: None,
                title_japanese: None,
                source_material_type,
                status: publication_status,
                score: score.map(f64::from),
                chapters,
                volumes,
                images,
                xrefs,
            });
        }
    }
    Ok(source_materials)
}

/// Converts a work's companies into company inputs with cross-references.
///
/// # Errors
///
/// Returns `INVALID_ARGUMENT` for an unknown company role.
fn map_companies(companies: Vec<CompanyEntry>) -> Result<Vec<AnimeCompanyInput>, Status> {
    let mut anime_companies = Vec::new();
    for CompanyEntry {
        name,
        description,
        sources,
        roles,
    } in companies
    {
        let mut company_roles = Vec::new();
        for role_num in roles {
            if let Some(role) = map_company_role(role_num)? {
                company_roles.push(role.to_owned());
            }
        }
        let xrefs: Vec<CompanyXrefInput> = sources
            .into_iter()
            .filter_map(|url| {
                let page_url = Url::parse(&url).ok()?;
                find_url_provider(&page_url).map(|provider| CompanyXrefInput {
                    provider: provider.to_owned(),
                    url,
                })
            })
            .collect();
        if company_roles.is_empty() || xrefs.is_empty() {
            continue;
        }
        anime_companies.push(AnimeCompanyInput {
            name,
            description,
            roles: company_roles,
            xrefs,
        });
    }
    Ok(anime_companies)
}

/// Converts a relation type name from the model's map key into its lookup text.
///
/// # Errors
///
/// Returns `INVALID_ARGUMENT` for a name no relation type has.
fn normalize_anime_relation_type(raw_text: &str) -> Result<&'static str, Status> {
    match raw_text.trim().to_ascii_uppercase().as_str() {
        "ADAPTATION" => Ok("ADAPTATION"),
        "ALTERNATIVE_VERSION" | "ALTERNATIVE VERSION" => Ok("ALTERNATIVE_VERSION"),
        "ALTERNATIVE_SETTING" | "ALTERNATIVE SETTING" => Ok("ALTERNATIVE_SETTING"),
        "CHARACTER" => Ok("CHARACTER"),
        "CROSSOVER" => Ok("CROSSOVER"),
        "FULL_STORY" | "FULL STORY" => Ok("FULL_STORY"),
        "OTHER" => Ok("OTHER"),
        "PARENT_STORY" | "PARENT STORY" => Ok("PARENT_STORY"),
        "PREQUEL" => Ok("PREQUEL"),
        "SAME_FRANCHISE" | "SAME FRANCHISE" => Ok("SAME_FRANCHISE"),
        "SAME_SETTING" | "SAME SETTING" => Ok("SAME_SETTING"),
        "SEQUEL" => Ok("SEQUEL"),
        "SIDE_STORY" | "SIDE STORY" => Ok("SIDE_STORY"),
        "SPIN_OFF" | "SPIN-OFF" | "SPIN OFF" => Ok("SPIN_OFF"),
        "SUMMARY" => Ok("SUMMARY"),
        _ => Err(Status::invalid_argument(format!(
            "unknown anime_relation_type {raw_text:?}"
        ))),
    }
}

/// Converts a source material relation name into its lookup text.
///
/// # Errors
///
/// Returns `INVALID_ARGUMENT` for a name no relation type has.
fn normalize_source_material_relation_type(raw_text: &str) -> Result<&'static str, Status> {
    match raw_text.trim().to_ascii_uppercase().as_str() {
        "ADAPTATION" => Ok("ADAPTATION"),
        "SOURCE" => Ok("SOURCE"),
        "ALTERNATIVE" => Ok("ALTERNATIVE"),
        "SPIN_OFF" | "SPIN-OFF" | "SPIN OFF" => Ok("SPIN_OFF"),
        "CHARACTER" => Ok("CHARACTER"),
        "SEQUEL" => Ok("SEQUEL"),
        "PREQUEL" => Ok("PREQUEL"),
        "PARENT_STORY" | "PARENT STORY" => Ok("PARENT_STORY"),
        "SIDE_STORY" | "SIDE STORY" => Ok("SIDE_STORY"),
        "OTHER" => Ok("OTHER"),
        _ => Err(Status::invalid_argument(format!(
            "unknown source_material_relation_type {raw_text:?}"
        ))),
    }
}

/// Converts an `AnimeRelationType` enum number into its lookup text.
///
/// # Errors
///
/// Returns `INVALID_ARGUMENT` for an unspecified or unknown value.
fn map_anime_relation_type(enum_number: i32) -> Result<&'static str, Status> {
    let Ok(relation_type) = AnimeRelationType::try_from(enum_number) else {
        return Err(Status::invalid_argument(format!(
            "unknown AnimeRelationType number {enum_number}"
        )));
    };
    match relation_type {
        AnimeRelationType::Unspecified => Err(Status::invalid_argument(
            "relation_claim.relation_type must be specified",
        )),
        AnimeRelationType::Adaptation => Ok("ADAPTATION"),
        AnimeRelationType::AlternativeVersion => Ok("ALTERNATIVE_VERSION"),
        AnimeRelationType::AlternativeSetting => Ok("ALTERNATIVE_SETTING"),
        AnimeRelationType::Character => Ok("CHARACTER"),
        AnimeRelationType::Crossover => Ok("CROSSOVER"),
        AnimeRelationType::FullStory => Ok("FULL_STORY"),
        AnimeRelationType::Other => Ok("OTHER"),
        AnimeRelationType::ParentStory => Ok("PARENT_STORY"),
        AnimeRelationType::Prequel => Ok("PREQUEL"),
        AnimeRelationType::SameFranchise => Ok("SAME_FRANCHISE"),
        AnimeRelationType::SameSetting => Ok("SAME_SETTING"),
        AnimeRelationType::Sequel => Ok("SEQUEL"),
        AnimeRelationType::SideStory => Ok("SIDE_STORY"),
        AnimeRelationType::SpinOff => Ok("SPIN_OFF"),
        AnimeRelationType::Summary => Ok("SUMMARY"),
    }
}

/// Converts an `EntityType` enum number into a `postgres_db` entity type.
///
/// # Errors
///
/// Returns `INVALID_ARGUMENT` for an unspecified or unknown value.
pub(crate) fn map_entity_type(enum_number: i32) -> Result<DbEntityType, Status> {
    let Ok(entity_type) = EntityType::try_from(enum_number) else {
        return Err(Status::invalid_argument(format!(
            "unknown EntityType number {enum_number}"
        )));
    };
    match entity_type {
        EntityType::Unspecified => Err(Status::invalid_argument(
            "relation_claim.target_page.entity_type must be specified",
        )),
        EntityType::Anime => Ok(DbEntityType::Anime),
        EntityType::Episode => Ok(DbEntityType::Episode),
        EntityType::Character => Ok(DbEntityType::Character),
    }
}

/// Converts a `CompanyRole` enum number into its lookup text, if it has one.
///
/// # Errors
///
/// Returns `INVALID_ARGUMENT` for an unknown value.
fn map_company_role(enum_number: i32) -> Result<Option<&'static str>, Status> {
    let Ok(role) = CompanyRole::try_from(enum_number) else {
        return Err(Status::invalid_argument(format!(
            "unknown CompanyRole number {enum_number}"
        )));
    };
    Ok(role.find_lookup_text())
}

/// Converts an optional `AnimeStatus` enum number into its lookup text.
///
/// # Errors
///
/// Returns `INVALID_ARGUMENT` for an unknown value.
fn map_anime_status_optional(enum_number: i32) -> Result<Option<&'static str>, Status> {
    let Ok(anime_status) = AnimeStatus::try_from(enum_number) else {
        return Err(Status::invalid_argument(format!(
            "unknown AnimeStatus number {enum_number}"
        )));
    };
    Ok(anime_status.find_lookup_text())
}

/// Finds the provider whose site a parsed `http` or `https` URL belongs to.
///
/// [`Url::parse`] has already lowercased the host, so only a leading `www.` is dropped.
fn find_url_provider(page_url: &Url) -> Option<&'static str> {
    if !matches!(page_url.scheme(), "http" | "https") {
        return None;
    }
    let host = page_url.host_str()?;
    match host.strip_prefix("www.").unwrap_or(host) {
        "myanimelist.net" => Some("mal"),
        "anilist.co" => Some("anilist"),
        "kitsu.app" | "kitsu.io" => Some("kitsu"),
        "anidb.net" => Some("anidb"),
        "anime-planet.com" => Some("anime_planet"),
        "anisearch.com" => Some("anisearch"),
        "animeschedule.net" => Some("animeschedule"),
        "cal.syoboi.jp" => Some("syoboi"),
        "animenewsnetwork.com" => Some("ann"),
        "livechart.me" => Some("livechart"),
        "simkl.com" => Some("simkl"),
        "animecountdown.com" => Some("animecountdown"),
        _ => None,
    }
}

/// Builds the page a URL names, of `entity_type`, with the parsed URL as its source.
///
/// `page_title` is the title the save's record gives the page's entity, kept so a read of a
/// page no stored entity holds yet shows a title; an empty title gives none. `parse_target`
/// finds the provider and address the URL names. A URL that names no page,
/// a page whose provider or address is not valid, or text that is not a URL gives `None`,
/// so it is left out like any other URL that names no such page.
pub(crate) fn build_page_source_from_url(
    source_url: &str,
    page_title: &str,
    entity_type: DbEntityType,
    parse_target: fn(&Url) -> Option<(&'static str, String)>,
) -> Option<PageSource> {
    let page_url = Url::parse(source_url).ok()?;
    let (provider, address) = parse_target(&page_url)?;
    let page_key = PageKey::new(provider.parse().ok()?, entity_type, address).ok()?;
    Some(PageSource {
        page_key,
        source: Some(page_url),
        title: parse_page_title(page_title),
    })
}

/// Builds the cross-reference a manga or novel page URL names, with the page's title.
///
/// An empty `page_title` gives none. A URL that names no manga or novel page, or text that
/// is not a URL, gives `None`.
pub(crate) fn build_source_material_xref_from_url(
    source_url: String,
    page_title: &str,
) -> Option<SourceMaterialXrefInput> {
    let page_url = Url::parse(&source_url).ok()?;
    let (provider, external_id) = parse_source_material_target_from_url(&page_url)?;
    Some(SourceMaterialXrefInput {
        provider: provider.to_owned(),
        external_id,
        url: source_url,
        title: parse_page_title(page_title),
    })
}

/// Finds the provider and work address an anime page URL names, refusing manga and novels.
pub(crate) fn parse_anime_target_from_url(page_url: &Url) -> Option<(&'static str, String)> {
    parse_provider_target_with_path_segments(page_url, &["anime"])
}

/// Finds the provider and identifier a manga or novel page URL names, refusing anime.
fn parse_source_material_target_from_url(page_url: &Url) -> Option<(&'static str, String)> {
    parse_provider_target_with_path_segments(page_url, &["manga", "light-novels"])
}

/// Finds the provider and character address a character page URL names.
///
/// Kitsu and Anime-Planet write the segment in the plural (`/characters/…`), the others in
/// the singular.
pub(crate) fn parse_character_target_from_url(page_url: &Url) -> Option<(&'static str, String)> {
    parse_provider_target_with_path_segments(page_url, &["character", "characters"])
}

/// Finds the provider and address a URL names, if its first path segment is allowed.
fn parse_provider_target_with_path_segments(
    page_url: &Url,
    allowed_segments: &[&str],
) -> Option<(&'static str, String)> {
    let provider = find_url_provider(page_url)?;
    let segments: Vec<&str> = page_url
        .path_segments()?
        .filter(|segment| !segment.is_empty())
        .collect();

    let raw_id: Cow<'_, str> = match provider {
        "mal" | "anilist" | "anisearch" | "anidb" | "kitsu" | "anime_planet" | "livechart"
        | "simkl" => {
            if segments.len() >= 2 {
                if allowed_segments.contains(&segments[0]) {
                    Cow::Borrowed(segments[1])
                } else {
                    return None;
                }
            } else {
                let first_segment = segments.first().copied()?;
                if matches!(
                    first_segment,
                    "anime" | "manga" | "light-novels" | "character" | "characters"
                ) {
                    return None;
                }
                Cow::Borrowed(first_segment)
            }
        }
        "animecountdown" => Cow::Borrowed(segments.first().copied()?),
        "ann" => page_url
            .query_pairs()
            .find_map(|(query_name, query_value)| (query_name == "id").then_some(query_value))?,
        _ => Cow::Borrowed(segments.last().copied()?),
    };

    let id: &str = match provider {
        // Numbered providers (including AniSearch `/anime/123,slug`) key by leading digits.
        "mal" | "anilist" | "anisearch" | "anidb" | "ann" | "livechart" | "simkl"
        | "animecountdown" => raw_id
            .split(|id_char: char| !id_char.is_ascii_digit())
            .next()
            .unwrap_or(""),
        _ => &raw_id,
    };

    if id.is_empty() {
        return None;
    }
    // Page keys hold lowercase addresses, as work matching builds them, so a slug written
    // with capitals still names the same stored page.
    Some((provider, id.to_lowercase()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_test_url(url_text: &str) -> Url {
        Url::parse(url_text).expect("valid URL")
    }
    use crate::service::shared_proto::v1::{RelatedAnime, SourceMaterialType};

    #[test]
    fn build_relations_and_graph_maps_relations_claims_sources_and_companies() {
        let mut related_anime = BTreeMap::new();
        related_anime.insert(
            "SPIN_OFF".to_owned(),
            RelatedAnimeList {
                items: vec![RelatedAnime {
                    title: "Spin Off".to_owned(),
                    sources: vec![
                        "https://myanimelist.net/anime/21/One_Piece".to_owned(),
                        "https://myanimelist.net/manga/13/One_Piece".to_owned(),
                    ],
                    ..RelatedAnime::default()
                }],
            },
        );

        let mut related_sources = BTreeMap::new();
        related_sources.insert(
            "ADAPTATION".to_owned(),
            RelatedSourceMaterialList {
                items: vec![RelatedSourceMaterial {
                    title: "Manga".to_owned(),
                    r#type: SourceMaterialType::Manga as i32,
                    sources: vec![
                        "https://myanimelist.net/manga/13/One_Piece".to_owned(),
                        "https://myanimelist.net/anime/21/One_Piece".to_owned(),
                    ],
                    status: Some(AnimeStatus::Ongoing as i32),
                    score: Some(9.2),
                    images: vec!["https://example.com/m.jpg".to_owned()],
                    chapters: Some(1100),
                    volumes: Some(108),
                }],
            },
        );

        let claims = vec![RelationClaim {
            provider: "mal".to_owned(),
            relation_type: AnimeRelationType::SpinOff as i32,
            target_page: Some(ProtoPageKey {
                provider: "mal".to_owned(),
                entity_type: EntityType::Anime as i32,
                address: "21".to_owned(),
            }),
        }];

        let companies = vec![CompanyEntry {
            name: "Toei Animation".to_owned(),
            description: None,
            sources: vec!["https://myanimelist.net/anime/producer/18/Toei_Animation".to_owned()],
            roles: vec![CompanyRole::Studio as i32],
        }];

        let mapped = build_relations_and_graph(related_anime, related_sources, companies, claims)
            .expect("valid ECHO-102 payload should map");
        assert_eq!(mapped.relations.len(), 1);
        assert_eq!(
            mapped.relations[0]
                .target_page_source
                .page_key
                .address()
                .as_str(),
            "21"
        );
        assert_eq!(
            mapped.relations[0]
                .target_page_source
                .source
                .as_ref()
                .map(Url::as_str),
            Some("https://myanimelist.net/anime/21/One_Piece")
        );
        assert_eq!(mapped.relation_claims.len(), 1);
        assert_eq!(mapped.source_materials.len(), 1);
        assert_eq!(mapped.source_materials[0].xrefs.len(), 1);
        assert_eq!(mapped.source_materials[0].xrefs[0].external_id, "13");
        assert_eq!(mapped.companies.len(), 1);
        assert_eq!(mapped.companies[0].roles, vec!["STUDIO"]);
    }

    #[test]
    fn parse_anime_and_source_material_targets_accept_only_their_path_segments() {
        assert_eq!(
            parse_anime_target_from_url(&parse_test_url(
                "https://myanimelist.net/anime/21/One_Piece"
            )),
            Some(("mal", "21".to_owned()))
        );
        assert_eq!(
            parse_anime_target_from_url(&parse_test_url("https://myanimelist.net/manga/13")),
            None
        );
        assert_eq!(
            parse_anime_target_from_url(&parse_test_url(
                "https://www.anime-planet.com/anime/One-Piece"
            )),
            Some(("anime_planet", "one-piece".to_owned()))
        );
        assert_eq!(
            parse_anime_target_from_url(&parse_test_url("https://kitsu.io/anime/One-Piece")),
            Some(("kitsu", "one-piece".to_owned()))
        );
        assert_eq!(
            parse_source_material_target_from_url(&parse_test_url(
                "https://myanimelist.net/manga/13/One_Piece"
            )),
            Some(("mal", "13".to_owned()))
        );
        assert_eq!(
            parse_source_material_target_from_url(&parse_test_url(
                "https://anilist.co/light-novels/7"
            )),
            Some(("anilist", "7".to_owned()))
        );
        assert_eq!(
            parse_source_material_target_from_url(&parse_test_url(
                "https://myanimelist.net/anime/21"
            )),
            None
        );
    }

    #[test]
    fn parse_target_from_url_character_and_work_segments_reject_each_other() {
        assert!(
            parse_anime_target_from_url(&parse_test_url("https://myanimelist.net/character/42"))
                .is_none()
        );
        assert!(
            parse_character_target_from_url(&parse_test_url("https://kitsu.app/anime/42"))
                .is_none()
        );
        assert_eq!(
            parse_character_target_from_url(&parse_test_url(
                "https://myanimelist.net/character/42"
            )),
            Some(("mal", "42".to_owned()))
        );
        assert_eq!(
            parse_character_target_from_url(&parse_test_url(
                "https://kitsu.io/characters/monkey-d-luffy"
            )),
            Some(("kitsu", "monkey-d-luffy".to_owned()))
        );
        assert_eq!(
            parse_character_target_from_url(&parse_test_url(
                "https://www.anime-planet.com/characters/brook"
            )),
            Some(("anime_planet", "brook".to_owned()))
        );
        assert!(
            parse_anime_target_from_url(&parse_test_url("https://kitsu.io/characters/brook"))
                .is_none()
        );
    }

    #[test]
    fn parse_anime_target_from_url_returns_provider_and_address() {
        for (url, expected_target) in [
            (
                "https://myanimelist.net/anime/21/One_Piece",
                Some(("mal", "21")),
            ),
            ("http://www.anilist.co/anime/21", Some(("anilist", "21"))),
            (
                "https://kitsu.app/anime/one-piece",
                Some(("kitsu", "one-piece")),
            ),
            ("https://anidb.net/anime/69", Some(("anidb", "69"))),
            ("https://anidb.net/69", Some(("anidb", "69"))),
            (
                "https://www.anime-planet.com/anime/one-piece",
                Some(("anime_planet", "one-piece")),
            ),
            (
                "https://www.anisearch.com/anime/2227,one-piece",
                Some(("anisearch", "2227")),
            ),
            (
                "https://animeschedule.net/anime/one-piece",
                Some(("animeschedule", "one-piece")),
            ),
            ("https://cal.syoboi.jp/tid/350", Some(("syoboi", "350"))),
            (
                "https://www.animenewsnetwork.com/encyclopedia/anime.php?id=836#top",
                Some(("ann", "836")),
            ),
            (
                "https://www.livechart.me/anime/321",
                Some(("livechart", "321")),
            ),
            ("https://simkl.com/anime/38636", Some(("simkl", "38636"))),
            (
                "https://animecountdown.com/1234/one-piece",
                Some(("animecountdown", "1234")),
            ),
            ("https://myanimelist.net/people/1", None),
            ("https://myanimelist.net/anime", None),
            ("https://example.com/anime/1", None),
            ("ftp://myanimelist.net/anime/1", None),
            (
                "https://www.animenewsnetwork.com/encyclopedia/anime.php",
                None,
            ),
            ("https://myanimelist.net/anime/abc", None),
        ] {
            let parsed_target = Url::parse(url)
                .ok()
                .and_then(|page_url| parse_anime_target_from_url(&page_url));
            assert_eq!(
                parsed_target
                    .as_ref()
                    .map(|(provider, address)| (*provider, address.as_str())),
                expected_target,
                "{url}"
            );
        }
    }

    #[test]
    fn normalize_relation_type_names_return_lookup_text() {
        for (relation_name, lookup_text) in [
            ("adaptation", "ADAPTATION"),
            ("Alternative Version", "ALTERNATIVE_VERSION"),
            ("ALTERNATIVE_SETTING", "ALTERNATIVE_SETTING"),
            ("character", "CHARACTER"),
            ("Crossover", "CROSSOVER"),
            ("full story", "FULL_STORY"),
            ("other", "OTHER"),
            ("Parent Story", "PARENT_STORY"),
            ("prequel", "PREQUEL"),
            ("same franchise", "SAME_FRANCHISE"),
            ("Same Setting", "SAME_SETTING"),
            ("sequel", "SEQUEL"),
            ("side story", "SIDE_STORY"),
            ("Spin-off", "SPIN_OFF"),
            ("summary", "SUMMARY"),
        ] {
            assert_eq!(
                normalize_anime_relation_type(relation_name).expect(relation_name),
                lookup_text
            );
        }
        for (relation_name, lookup_text) in [
            ("adaptation", "ADAPTATION"),
            ("source", "SOURCE"),
            ("Alternative", "ALTERNATIVE"),
            ("spin off", "SPIN_OFF"),
            ("character", "CHARACTER"),
            ("sequel", "SEQUEL"),
            ("prequel", "PREQUEL"),
            ("parent story", "PARENT_STORY"),
            ("Side Story", "SIDE_STORY"),
            ("other", "OTHER"),
        ] {
            assert_eq!(
                normalize_source_material_relation_type(relation_name).expect(relation_name),
                lookup_text
            );
        }
        assert!(normalize_anime_relation_type("remake").is_err());
        assert!(normalize_source_material_relation_type("remake").is_err());
    }

    #[test]
    fn map_entity_type_proto_numbers_return_entity_types() {
        assert_eq!(
            map_entity_type(EntityType::Anime as i32).ok(),
            Some(DbEntityType::Anime)
        );
        assert_eq!(
            map_entity_type(EntityType::Episode as i32).ok(),
            Some(DbEntityType::Episode)
        );
        assert_eq!(
            map_entity_type(EntityType::Character as i32).ok(),
            Some(DbEntityType::Character)
        );
        assert!(map_entity_type(EntityType::Unspecified as i32).is_err());
        assert!(map_entity_type(99).is_err());
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

    #[sqlx::test(migrations = "../postgres_db/migrations")]
    async fn lookup_text_every_relation_enum_value_exists_in_lookup_table(pool: sqlx::PgPool) {
        let relation_texts: Vec<&str> = (0..64)
            .filter_map(|enum_number| map_anime_relation_type(enum_number).ok())
            .collect();
        let company_role_texts: Vec<&str> = (0..64)
            .filter_map(|enum_number| map_company_role(enum_number).ok().flatten())
            .collect();
        let status_texts: Vec<&str> = (0..64)
            .filter_map(|enum_number| map_anime_status_optional(enum_number).ok().flatten())
            .collect();

        assert_eq!(relation_texts.len(), 15);
        assert_eq!(company_role_texts.len(), 4);
        assert_eq!(status_texts.len(), 5);
        assert!(map_anime_relation_type(AnimeRelationType::Unspecified as i32).is_err());
        assert_eq!(
            map_company_role(CompanyRole::Unspecified as i32).ok(),
            Some(None)
        );
        assert!(map_company_role(99).is_err());
        assert_eq!(
            map_anime_status_optional(AnimeStatus::Unspecified as i32).ok(),
            Some(None)
        );
        assert!(map_anime_status_optional(99).is_err());
        assert_lookup_values_exist(&pool, "anime_relation_type", relation_texts).await;
        assert_lookup_values_exist(&pool, "company_role", company_role_texts).await;
        assert_lookup_values_exist(&pool, "anime_status", status_texts).await;
    }

    #[test]
    fn build_relations_and_graph_unusable_entries_are_left_out() {
        let related_anime = BTreeMap::from([(
            "SEQUEL".to_owned(),
            RelatedAnimeList {
                items: vec![RelatedAnime {
                    sources: vec!["https://example.com/anime/2".to_owned()],
                    ..RelatedAnime::default()
                }],
            },
        )]);
        let related_sources = BTreeMap::from([(
            "ADAPTATION".to_owned(),
            RelatedSourceMaterialList {
                items: vec![RelatedSourceMaterial {
                    title: "Unlinked Manga".to_owned(),
                    r#type: SourceMaterialType::Manga as i32,
                    sources: vec!["https://example.com/manga/13".to_owned()],
                    ..RelatedSourceMaterial::default()
                }],
            },
        )]);
        let companies = vec![
            CompanyEntry {
                name: "Unrolled Studio".to_owned(),
                description: None,
                sources: vec!["https://myanimelist.net/anime/producer/1".to_owned()],
                roles: vec![CompanyRole::Unspecified as i32],
            },
            CompanyEntry {
                name: "Unlinked Studio".to_owned(),
                description: None,
                sources: vec!["https://example.com/studio/1".to_owned()],
                roles: vec![CompanyRole::Producer as i32],
            },
        ];

        let mapped =
            build_relations_and_graph(related_anime, related_sources, companies, Vec::new())
                .expect("unusable entries are dropped, not refused");

        assert_eq!(mapped.relations, Vec::new());
        assert_eq!(mapped.source_materials, Vec::new());
        assert_eq!(mapped.companies, Vec::new());
    }

    #[test]
    fn build_relations_and_graph_invalid_claim_returns_invalid_argument() {
        let valid_claim = RelationClaim {
            provider: "mal".to_owned(),
            relation_type: AnimeRelationType::Sequel as i32,
            target_page: Some(ProtoPageKey {
                provider: "mal".to_owned(),
                entity_type: EntityType::Anime as i32,
                address: "2".to_owned(),
            }),
        };
        let breaking_changes: Vec<fn(&mut RelationClaim)> = vec![
            |claim| claim.provider = "not_a_provider".to_owned(),
            |claim| claim.relation_type = 99,
            |claim| claim.target_page = None,
            |claim| {
                claim
                    .target_page
                    .as_mut()
                    .expect("valid target page")
                    .provider = "not_a_provider".to_owned();
            },
            |claim| {
                claim
                    .target_page
                    .as_mut()
                    .expect("valid target page")
                    .address = "not-numeric".to_owned();
            },
        ];
        for break_rule in breaking_changes {
            let mut broken_claim = valid_claim.clone();
            break_rule(&mut broken_claim);

            let status = build_relations_and_graph(
                BTreeMap::new(),
                BTreeMap::new(),
                Vec::new(),
                vec![broken_claim],
            )
            .expect_err("an invalid claim must be refused");

            assert_eq!(status.code(), tonic::Code::InvalidArgument);
        }
    }
}
