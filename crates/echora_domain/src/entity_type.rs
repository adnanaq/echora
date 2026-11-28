//! The kinds of catalogue entity: works, characters and episodes.

use std::fmt;
use std::str::FromStr;

use crate::KeyError;

/// Type of catalogue entity, which is also what a provider page describes.
///
/// The variants follow the numbering of `shared_proto.v1.EntityType` (anime 1,
/// character 2, episode 3), and [`Ord`] sorts them in that order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize, serde::Serialize),
    serde(try_from = "String", into = "&'static str")
)]
pub enum EntityType {
    /// A work (anime).
    Anime,
    /// A character.
    Character,
    /// An episode of a work.
    Episode,
}

impl EntityType {
    /// Every entity type, in `shared_proto.v1.EntityType` number order.
    pub const ALL: [Self; 3] = [Self::Anime, Self::Character, Self::Episode];

    /// Returns the lookup table key for this entity type.
    ///
    /// # Examples
    ///
    /// ```
    /// use echora_domain::EntityType;
    ///
    /// assert_eq!(EntityType::Anime.as_str(), "anime");
    /// ```
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Anime => "anime",
            Self::Character => "character",
            Self::Episode => "episode",
        }
    }
}

impl fmt::Display for EntityType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl From<EntityType> for &'static str {
    fn from(entity_type: EntityType) -> Self {
        entity_type.as_str()
    }
}

impl FromStr for EntityType {
    type Err = KeyError;

    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        match raw {
            "anime" => Ok(Self::Anime),
            "character" => Ok(Self::Character),
            "episode" => Ok(Self::Episode),
            other => Err(KeyError::unknown_entity_type(other)),
        }
    }
}

impl TryFrom<String> for EntityType {
    type Error = KeyError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        value.parse()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entity_type_from_str_round_trips_every_lookup_value() {
        for entity_type in EntityType::ALL {
            let name = entity_type.as_str();
            assert_eq!(name.parse::<EntityType>().ok(), Some(entity_type));
            assert_eq!(
                EntityType::try_from(name.to_owned()).ok(),
                Some(entity_type)
            );
            assert_eq!(<&'static str>::from(entity_type), name);
            assert_eq!(entity_type.to_string(), name);
        }
    }

    #[test]
    fn entity_type_all_sorted_by_declaration_order() {
        let mut sorted_entity_types = EntityType::ALL;
        sorted_entity_types.sort_unstable();
        assert_eq!(sorted_entity_types, EntityType::ALL);
    }

    #[test]
    fn entity_type_from_str_unknown_name_returns_unknown_entity_type() {
        let error = "manga".parse::<EntityType>().unwrap_err();
        assert!(error.is_unknown_entity_type());
        assert!(!error.is_unknown_provider());
        assert!(error.to_string().starts_with("unknown entity type: manga"));
    }
}
