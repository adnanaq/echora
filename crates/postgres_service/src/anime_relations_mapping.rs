//! Protobuf-to-SQL mapping for anime relations, relation claims, source material,
//! and companies.

// Rust guideline compliant 2026-10-09

use std::collections::BTreeMap;

use postgres_db::{
    AnimeCompanyInput, AnimeRelationClaimInput, AnimeRelationInput, AnimeRelationsAndGraphInput,
    AnimeSourceMaterialInput, CompanyXrefInput, PageKeyInput, PageKind as DbPageKind,
    SourceMaterialXrefInput,
};
use tonic::Status;

use crate::anime_mapping::map_source_material_type;
use crate::service::shared_proto::v1::{
    AnimeRelationType, AnimeStatus, CompanyEntry, CompanyRole, PageKey, PageKind, RelatedAnimeList,
    RelatedSourceMaterial, RelatedSourceMaterialList, RelationClaim,
};

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

fn map_related_anime(
    related_anime: BTreeMap<String, RelatedAnimeList>,
) -> Result<Vec<AnimeRelationInput>, Status> {
    let mut anime_relations = Vec::new();
    for (raw_relation_type, RelatedAnimeList { items }) in related_anime {
        let relation_type = normalize_anime_relation_type(&raw_relation_type)?;
        for item in items {
            for source_url in item.sources {
                if let Some((provider, address)) = parse_provider_target_from_url(&source_url) {
                    anime_relations.push(AnimeRelationInput {
                        relation_type: relation_type.to_owned(),
                        to_page_provider: provider.to_owned(),
                        to_page_address: address,
                    });
                }
            }
        }
    }
    Ok(anime_relations)
}

fn map_relation_claims(claims: Vec<RelationClaim>) -> Result<Vec<AnimeRelationClaimInput>, Status> {
    claims
        .into_iter()
        .map(
            |RelationClaim {
                 provider,
                 relation_type,
                 target_page,
             }| {
                let relation_type_text = map_anime_relation_type(relation_type)?.to_owned();
                let PageKey {
                    provider: target_provider,
                    page_kind,
                    address,
                } = target_page.ok_or_else(|| {
                    Status::invalid_argument("relation_claim requires target_page")
                })?;
                let mapped_kind = map_page_kind(page_kind)?;
                Ok(AnimeRelationClaimInput {
                    provider,
                    relation_type: relation_type_text,
                    target_page: PageKeyInput {
                        provider: target_provider,
                        page_kind: mapped_kind,
                        address,
                    },
                })
            },
        )
        .collect()
}

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
                .filter_map(|url| {
                    parse_provider_target_from_url(&url).map(|(provider, external_id)| {
                        SourceMaterialXrefInput {
                            provider: provider.to_owned(),
                            external_id,
                            url,
                        }
                    })
                })
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
                parse_provider_from_url(&url).map(|provider| CompanyXrefInput {
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

pub(crate) fn map_page_kind(enum_number: i32) -> Result<DbPageKind, Status> {
    let Ok(page_kind) = PageKind::try_from(enum_number) else {
        return Err(Status::invalid_argument(format!(
            "unknown PageKind number {enum_number}"
        )));
    };
    match page_kind {
        PageKind::Unspecified => Err(Status::invalid_argument(
            "relation_claim.target_page.page_kind must be specified",
        )),
        PageKind::Anime => Ok(DbPageKind::Anime),
        PageKind::Episode => Ok(DbPageKind::Episode),
        PageKind::Character => Ok(DbPageKind::Character),
    }
}

fn map_company_role(enum_number: i32) -> Result<Option<&'static str>, Status> {
    let Ok(role) = CompanyRole::try_from(enum_number) else {
        return Err(Status::invalid_argument(format!(
            "unknown CompanyRole number {enum_number}"
        )));
    };
    match role {
        CompanyRole::Unspecified => Ok(None),
        CompanyRole::Studio => Ok(Some("STUDIO")),
        CompanyRole::Producer => Ok(Some("PRODUCER")),
        CompanyRole::Licensor => Ok(Some("LICENSOR")),
        CompanyRole::Unknown => Ok(Some("UNKNOWN")),
    }
}

fn map_anime_status_optional(enum_number: i32) -> Result<Option<&'static str>, Status> {
    let Ok(anime_status) = AnimeStatus::try_from(enum_number) else {
        return Err(Status::invalid_argument(format!(
            "unknown AnimeStatus number {enum_number}"
        )));
    };
    match anime_status {
        AnimeStatus::Unspecified => Ok(None),
        AnimeStatus::Finished => Ok(Some("FINISHED")),
        AnimeStatus::Upcoming => Ok(Some("UPCOMING")),
        AnimeStatus::Ongoing => Ok(Some("ONGOING")),
        AnimeStatus::Unknown => Ok(Some("UNKNOWN")),
        AnimeStatus::Cancelled => Ok(Some("CANCELLED")),
    }
}

fn parse_provider_from_url(url: &str) -> Option<&'static str> {
    let without_scheme = url
        .trim()
        .strip_prefix("https://")
        .or_else(|| url.trim().strip_prefix("http://"))?;
    let host_part = without_scheme.split('/').next()?;
    let host = host_part.strip_prefix("www.").unwrap_or(host_part);
    match host {
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

pub(crate) fn parse_provider_target_from_url(url: &str) -> Option<(&'static str, String)> {
    parse_provider_target_with_kinds(url, &["anime", "manga", "light-novels"])
}

pub(crate) fn parse_character_target_from_url(url: &str) -> Option<(&'static str, String)> {
    parse_provider_target_with_kinds(url, &["character"])
}

fn parse_provider_target_with_kinds(
    url: &str,
    allowed_segments: &[&str],
) -> Option<(&'static str, String)> {
    let provider = parse_provider_from_url(url)?;
    let without_scheme = url
        .trim()
        .strip_prefix("https://")
        .or_else(|| url.trim().strip_prefix("http://"))?;
    let path_and_query = without_scheme.split_once('/')?.1;
    let segments: Vec<&str> = path_and_query
        .split('?')
        .next()?
        .split('#')
        .next()?
        .split('/')
        .filter(|segment| !segment.is_empty())
        .collect();

    let raw_id = match provider {
        "mal" | "anilist" | "anisearch" | "anidb" | "kitsu" | "anime_planet" | "livechart"
        | "simkl" => {
            if segments.len() >= 2 && allowed_segments.contains(&segments[0]) {
                segments[1]
            } else {
                segments.first().copied()?
            }
        }
        "animecountdown" => segments.first().copied()?,
        "ann" => {
            let query = path_and_query.split_once('?')?.1;
            query
                .split('&')
                .find_map(|pair| pair.strip_prefix("id="))?
                .split('#')
                .next()?
        }
        _ => segments.last().copied()?,
    };

    let id = match provider {
        // Numbered providers (including AniSearch `/anime/123,slug`) key by leading digits.
        "mal" | "anilist" | "anisearch" | "anidb" | "ann" | "livechart" | "simkl"
        | "animecountdown" => raw_id
            .split(|ch: char| !ch.is_ascii_digit())
            .next()
            .unwrap_or(""),
        _ => raw_id,
    };

    if id.is_empty() {
        return None;
    }
    Some((provider, id.to_owned()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::service::shared_proto::v1::{RelatedAnime, SourceMaterialType};

    #[test]
    fn build_relations_and_graph_maps_relations_claims_sources_and_companies() {
        let mut related_anime = BTreeMap::new();
        related_anime.insert(
            "SPIN_OFF".to_owned(),
            RelatedAnimeList {
                items: vec![RelatedAnime {
                    title: "Spin Off".to_owned(),
                    sources: vec!["https://myanimelist.net/anime/21/One_Piece".to_owned()],
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
                    sources: vec!["https://myanimelist.net/manga/13/One_Piece".to_owned()],
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
            target_page: Some(PageKey {
                provider: "mal".to_owned(),
                page_kind: PageKind::Anime as i32,
                address: "21".to_owned(),
            }),
        }];

        let companies = vec![CompanyEntry {
            name: "Toei Animation".to_owned(),
            description: None,
            sources: vec!["https://myanimelist.net/anime/producer/18/Toei_Animation".to_owned()],
            roles: vec![CompanyRole::Studio as i32],
        }];

        let Ok(mapped) =
            build_relations_and_graph(related_anime, related_sources, companies, claims)
        else {
            panic!("valid ECHO-102 payload should map");
        };
        assert_eq!(mapped.relations.len(), 1);
        assert_eq!(mapped.relations[0].to_page_address, "21");
        assert_eq!(mapped.relation_claims.len(), 1);
        assert_eq!(mapped.source_materials.len(), 1);
        assert_eq!(mapped.source_materials[0].xrefs[0].external_id, "13");
        assert_eq!(mapped.companies.len(), 1);
        assert_eq!(mapped.companies[0].roles, vec!["STUDIO"]);
    }

    #[test]
    fn parse_provider_target_from_url_rejects_character_urls() {
        assert!(parse_provider_target_from_url("https://myanimelist.net/character/42").is_none());
        assert_eq!(
            parse_character_target_from_url("https://myanimelist.net/character/42"),
            Some(("mal", "42".to_owned()))
        );
    }
}
