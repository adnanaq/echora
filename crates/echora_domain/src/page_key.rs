//! Identifiers of one provider page: its provider, entity type and checked address.

use std::fmt;

use crate::{EntityType, KeyError, Provider};

/// Validated provider-specific page identifier or slug.
///
/// An address is only built together with the provider and entity type whose rules it
/// passed, so every `Address` is valid for the page it was built for.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Address(String);

impl Address {
    /// Creates a validated page address for a `provider` page describing `entity_type`.
    ///
    /// # Examples
    ///
    /// ```
    /// use echora_domain::{Address, EntityType, Provider};
    ///
    /// let address = Address::new(Provider::Mal, EntityType::Anime, "21")?;
    /// assert_eq!(address.as_str(), "21");
    /// # Ok::<(), echora_domain::KeyError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an invalid-address error when `address` does not match
    /// `provider`'s identifier rules for `entity_type`.
    pub fn new(
        provider: Provider,
        entity_type: EntityType,
        address: impl Into<String>,
    ) -> Result<Self, KeyError> {
        let address = address.into();
        provider.validate_address(entity_type, &address)?;
        Ok(Self(address))
    }

    /// Returns the address as the provider writes it.
    ///
    /// # Examples
    ///
    /// ```
    /// use echora_domain::{Address, EntityType, Provider};
    ///
    /// let address =
    ///     Address::new(Provider::Mal, EntityType::Anime, "21").expect("MAL addresses are numeric");
    /// assert_eq!(address.as_str(), "21");
    /// ```
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl AsRef<str> for Address {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for Address {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Identifies a provider page by provider, entity type, and address.
///
/// With the `serde` feature a page key reads and writes as an object with `provider`,
/// `entity_type` and `address` fields holding their lookup names, and reading one checks the
/// address the same way [`PageKey::new`] does.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize, serde::Serialize),
    serde(try_from = "PageKeyFields", into = "PageKeyFields")
)]
pub struct PageKey {
    provider: Provider,
    entity_type: EntityType,
    address: Address,
}

impl PageKey {
    /// Creates a validated page key for `provider`, `entity_type`, and `address`.
    ///
    /// # Examples
    ///
    /// ```
    /// use echora_domain::{EntityType, PageKey, Provider};
    ///
    /// let key = PageKey::new(Provider::Kitsu, EntityType::Anime, "cowboy-bebop")?;
    /// assert_eq!(key.provider(), Provider::Kitsu);
    /// assert_eq!(key.address().as_str(), "cowboy-bebop");
    /// # Ok::<(), echora_domain::KeyError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an invalid-address error when `address` fails `provider` validation.
    pub fn new(
        provider: Provider,
        entity_type: EntityType,
        address: impl Into<String>,
    ) -> Result<Self, KeyError> {
        let address = Address::new(provider, entity_type, address)?;
        Ok(Self {
            provider,
            entity_type,
            address,
        })
    }

    /// Returns the provider for this page key.
    #[must_use]
    pub const fn provider(&self) -> Provider {
        self.provider
    }

    /// Returns the entity type this page key's page describes.
    #[must_use]
    pub const fn entity_type(&self) -> EntityType {
        self.entity_type
    }

    /// Returns the validated address for this page key.
    #[must_use]
    pub const fn address(&self) -> &Address {
        &self.address
    }
}

impl fmt::Display for PageKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}:{}", self.provider, self.entity_type, self.address)
    }
}

/// Unchecked page key fields, the serde form of [`PageKey`].
#[cfg(feature = "serde")]
#[derive(serde::Deserialize, serde::Serialize)]
struct PageKeyFields {
    provider: Provider,
    entity_type: EntityType,
    address: String,
}

#[cfg(feature = "serde")]
impl TryFrom<PageKeyFields> for PageKey {
    type Error = KeyError;

    fn try_from(fields: PageKeyFields) -> Result<Self, Self::Error> {
        Self::new(fields.provider, fields.entity_type, fields.address)
    }
}

#[cfg(feature = "serde")]
impl From<PageKey> for PageKeyFields {
    fn from(page_key: PageKey) -> Self {
        Self {
            provider: page_key.provider,
            entity_type: page_key.entity_type,
            address: page_key.address.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_key_new_invalid_address_returns_invalid_address() {
        assert!(PageKey::new(Provider::Mal, EntityType::Anime, "21").is_ok());
        let error = PageKey::new(Provider::Mal, EntityType::Anime, "invalid").unwrap_err();
        assert!(error.is_invalid_address());
        assert_eq!(
            error.to_string().lines().next(),
            Some("provider mal requires a numeric address, got \"invalid\"")
        );
    }

    #[test]
    fn page_key_display_joins_provider_entity_type_and_address() {
        let page_key = PageKey::new(Provider::Mal, EntityType::Episode, "21:1").unwrap();
        assert_eq!(page_key.to_string(), "mal:episode:21:1");
    }

    #[cfg(feature = "serde")]
    #[test]
    fn page_key_serde_writes_lookup_names_and_checks_address() {
        let page_key =
            PageKey::new(Provider::AnimePlanet, EntityType::Anime, "cowboy-bebop").unwrap();
        let page_key_json = serde_json::to_value(&page_key).unwrap();
        assert_eq!(
            page_key_json,
            serde_json::json!({
                "provider": "anime_planet",
                "entity_type": "anime",
                "address": "cowboy-bebop",
            })
        );
        assert_eq!(
            serde_json::from_value::<PageKey>(page_key_json).unwrap(),
            page_key
        );

        for invalid_page_key_json in [
            serde_json::json!({"provider": "mal", "entity_type": "anime", "address": "bad"}),
            serde_json::json!({"provider": "myanimelist", "entity_type": "anime", "address": "1"}),
            serde_json::json!({"provider": "mal", "entity_type": "manga", "address": "1"}),
        ] {
            assert!(serde_json::from_value::<PageKey>(invalid_page_key_json).is_err());
        }
    }
}
