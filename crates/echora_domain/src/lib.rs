//! Domain types shared by Echora's Rust services.
//!
//! The catalogue's own concepts live here once, so the PostgreSQL Service, the Temporal
//! workflows and the Update Service agree on their names, rules and wire forms:
//!
//! - [`Provider`]: an external site whose anime pages the catalogue stores.
//! - [`EntityType`]: the kind of catalogue entity, a work, character or episode, which is
//!   also what a provider page describes.
//! - [`Address`] and [`PageKey`]: one provider page, whose address has passed its provider's
//!   rules for that entity type.
//! - [`KeyError`]: why a provider name, entity type or address was refused.
//!
//! Values are written everywhere by their lookup names (`mal`, `anime_planet`, `episode`),
//! the same names the PostgreSQL lookup tables and the protos use.
//!
//! # Features
//!
//! Each feature adds conversions for one consumer and changes nothing else:
//!
//! - `proto`: conversions to and from the `shared_proto.v1` `EntityType` and `PageKey`
//!   messages in `echora_protos`.
//! - `serde`: `Serialize` and `Deserialize`, used for Temporal payloads. A page key is an
//!   object with `provider`, `entity_type` and `address` fields, and reading one checks its
//!   address.
//! - `sqlx`: `sqlx` `Type` and `Decode` for reading PostgreSQL `text` lookup columns.
//!
//! # Examples
//!
//! ```
//! use echora_domain::{EntityType, PageKey, Provider};
//!
//! let page_key = PageKey::new(Provider::Mal, EntityType::Episode, "21:1")?;
//! assert_eq!(page_key.provider().as_str(), "mal");
//! assert!(PageKey::new(Provider::Mal, EntityType::Anime, "21:1").is_err());
//! # Ok::<(), echora_domain::KeyError>(())
//! ```

mod entity_type;
mod error;
mod page_key;
#[cfg(feature = "proto")]
mod proto;
mod provider;
#[cfg(feature = "sqlx")]
mod sql;

#[doc(inline)]
pub use entity_type::EntityType;
#[doc(inline)]
pub use error::KeyError;
#[doc(inline)]
pub use page_key::{Address, PageKey};
#[doc(inline)]
pub use provider::Provider;
