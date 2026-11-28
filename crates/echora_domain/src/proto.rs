//! Conversions between the domain types and the `shared_proto.v1` messages.
//!
//! Turning a domain value into a message cannot fail. Reading a message checks it the same
//! way the domain constructors do, in this order: the provider name, then the entity type,
//! then the address.

use echora_protos::shared_proto::v1::{EntityType as ProtoEntityType, PageKey as ProtoPageKey};

use crate::{EntityType, KeyError, PageKey, Provider};

impl From<EntityType> for ProtoEntityType {
    fn from(entity_type: EntityType) -> Self {
        match entity_type {
            EntityType::Anime => Self::Anime,
            EntityType::Character => Self::Character,
            EntityType::Episode => Self::Episode,
        }
    }
}

/// Reads a proto entity type, refusing `ENTITY_TYPE_UNSPECIFIED`.
impl TryFrom<ProtoEntityType> for EntityType {
    type Error = KeyError;

    fn try_from(proto_entity_type: ProtoEntityType) -> Result<Self, Self::Error> {
        match proto_entity_type {
            ProtoEntityType::Unspecified => Err(KeyError::unspecified_entity_type()),
            ProtoEntityType::Anime => Ok(Self::Anime),
            ProtoEntityType::Character => Ok(Self::Character),
            ProtoEntityType::Episode => Ok(Self::Episode),
        }
    }
}

/// Reads the enum number a proto message carries in its `entity_type` field.
///
/// A number the proto enum does not define is an unknown entity type, and `0` is
/// `ENTITY_TYPE_UNSPECIFIED`.
impl TryFrom<i32> for EntityType {
    type Error = KeyError;

    fn try_from(enum_number: i32) -> Result<Self, Self::Error> {
        ProtoEntityType::try_from(enum_number)
            .map_err(|_| KeyError::unknown_entity_type_number(enum_number))?
            .try_into()
    }
}

impl From<&PageKey> for ProtoPageKey {
    fn from(page_key: &PageKey) -> Self {
        Self {
            provider: page_key.provider().as_str().to_owned(),
            entity_type: ProtoEntityType::from(page_key.entity_type()).into(),
            address: page_key.address().as_str().to_owned(),
        }
    }
}

/// Reads a proto page key, checking its provider, entity type and address in that order.
impl TryFrom<ProtoPageKey> for PageKey {
    type Error = KeyError;

    fn try_from(proto_page_key: ProtoPageKey) -> Result<Self, Self::Error> {
        let provider: Provider = proto_page_key.provider.parse()?;
        let entity_type = EntityType::try_from(proto_page_key.entity_type)?;
        Self::new(provider, entity_type, proto_page_key.address)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proto_entity_type_numbers_match_entity_type_order() {
        for (position, entity_type) in EntityType::ALL.into_iter().enumerate() {
            let enum_number = ProtoEntityType::from(entity_type) as i32;
            assert_eq!(usize::try_from(enum_number).unwrap(), position + 1);
            assert_eq!(EntityType::try_from(enum_number).ok(), Some(entity_type));
        }
    }

    #[test]
    fn entity_type_try_from_unspecified_or_unknown_number_returns_distinct_errors() {
        let unspecified_error =
            EntityType::try_from(ProtoEntityType::Unspecified as i32).unwrap_err();
        assert!(unspecified_error.is_unspecified_entity_type());
        assert!(!unspecified_error.is_unknown_entity_type());

        let unknown_error = EntityType::try_from(99).unwrap_err();
        assert!(unknown_error.is_unknown_entity_type());
        assert!(!unknown_error.is_unspecified_entity_type());
        assert!(
            unknown_error
                .to_string()
                .starts_with("unknown EntityType number 99")
        );
    }

    #[test]
    fn page_key_proto_round_trips_every_entity_type() {
        for (entity_type, address) in [
            (EntityType::Anime, "40"),
            (EntityType::Character, "40"),
            (EntityType::Episode, "40:1"),
        ] {
            let page_key = PageKey::new(Provider::Mal, entity_type, address).unwrap();
            let proto_page_key = ProtoPageKey::from(&page_key);
            assert_eq!(proto_page_key.provider, "mal");
            assert_eq!(proto_page_key.address, address);
            assert_eq!(PageKey::try_from(proto_page_key).unwrap(), page_key);
        }
    }

    #[test]
    fn page_key_try_from_invalid_proto_reports_first_invalid_field() {
        let valid_proto_page_key = ProtoPageKey {
            provider: "mal".to_owned(),
            entity_type: ProtoEntityType::Anime as i32,
            address: "21".to_owned(),
        };

        let unknown_provider_error = PageKey::try_from(ProtoPageKey {
            provider: "myanimelist".to_owned(),
            entity_type: ProtoEntityType::Unspecified as i32,
            address: "bad".to_owned(),
        })
        .unwrap_err();
        assert!(unknown_provider_error.is_unknown_provider());

        let unspecified_error = PageKey::try_from(ProtoPageKey {
            entity_type: ProtoEntityType::Unspecified as i32,
            address: "bad".to_owned(),
            ..valid_proto_page_key.clone()
        })
        .unwrap_err();
        assert!(unspecified_error.is_unspecified_entity_type());

        let invalid_address_error = PageKey::try_from(ProtoPageKey {
            address: "bad".to_owned(),
            ..valid_proto_page_key
        })
        .unwrap_err();
        assert!(invalid_address_error.is_invalid_address());
    }
}
