//! The error returned when a provider, entity type or page address is refused.

use std::backtrace::{Backtrace, BacktraceStatus};
use std::fmt;

/// Reports a provider name, entity type or page address that is not valid.
///
/// The `is_*` methods tell the cases apart, so a caller can answer each with its own
/// message or status code.
#[derive(Debug)]
pub struct KeyError {
    kind: KeyErrorKind,
    backtrace: Backtrace,
}

#[derive(Debug)]
enum KeyErrorKind {
    UnknownProvider(String),
    UnknownEntityType(String),
    #[cfg(feature = "proto")]
    UnknownEntityTypeNumber(i32),
    #[cfg(feature = "proto")]
    UnspecifiedEntityType,
    InvalidAddress(String),
}

impl KeyError {
    /// Creates an error of `kind`, capturing a backtrace when `RUST_BACKTRACE` asks for one.
    fn from_kind(kind: KeyErrorKind) -> Self {
        Self {
            kind,
            backtrace: Backtrace::capture(),
        }
    }

    /// Creates an error for a provider name no [`crate::Provider`] has.
    pub(crate) fn unknown_provider(name: &str) -> Self {
        Self::from_kind(KeyErrorKind::UnknownProvider(name.to_owned()))
    }

    /// Creates an error for an entity type name no [`crate::EntityType`] has.
    pub(crate) fn unknown_entity_type(name: &str) -> Self {
        Self::from_kind(KeyErrorKind::UnknownEntityType(name.to_owned()))
    }

    /// Creates an error for a `shared_proto.v1.EntityType` number the enum does not define.
    #[cfg(feature = "proto")]
    pub(crate) fn unknown_entity_type_number(enum_number: i32) -> Self {
        Self::from_kind(KeyErrorKind::UnknownEntityTypeNumber(enum_number))
    }

    /// Creates an error for `ENTITY_TYPE_UNSPECIFIED`, which names no entity type.
    #[cfg(feature = "proto")]
    pub(crate) fn unspecified_entity_type() -> Self {
        Self::from_kind(KeyErrorKind::UnspecifiedEntityType)
    }

    /// Creates an error for a page address its provider's rules refuse.
    pub(crate) fn invalid_address(message: impl Into<String>) -> Self {
        Self::from_kind(KeyErrorKind::InvalidAddress(message.into()))
    }

    /// Returns whether a provider name was not one of the known providers.
    ///
    /// # Examples
    ///
    /// ```
    /// use echora_domain::Provider;
    ///
    /// let error = "myanimelist".parse::<Provider>().unwrap_err();
    /// assert!(error.is_unknown_provider());
    /// ```
    #[must_use]
    pub fn is_unknown_provider(&self) -> bool {
        matches!(self.kind, KeyErrorKind::UnknownProvider(_))
    }

    /// Returns whether an entity type name or proto enum number was not a known one.
    ///
    /// `ENTITY_TYPE_UNSPECIFIED` is reported by [`Self::is_unspecified_entity_type`] instead.
    ///
    /// # Examples
    ///
    /// ```
    /// use echora_domain::EntityType;
    ///
    /// let error = "manga".parse::<EntityType>().unwrap_err();
    /// assert!(error.is_unknown_entity_type());
    /// ```
    #[must_use]
    pub fn is_unknown_entity_type(&self) -> bool {
        match self.kind {
            KeyErrorKind::UnknownEntityType(_) => true,
            #[cfg(feature = "proto")]
            KeyErrorKind::UnknownEntityTypeNumber(_) => true,
            #[cfg(feature = "proto")]
            KeyErrorKind::UnspecifiedEntityType => false,
            KeyErrorKind::UnknownProvider(_) | KeyErrorKind::InvalidAddress(_) => false,
        }
    }

    /// Returns whether a proto entity type was `ENTITY_TYPE_UNSPECIFIED`.
    #[must_use]
    pub fn is_unspecified_entity_type(&self) -> bool {
        #[cfg(feature = "proto")]
        {
            matches!(self.kind, KeyErrorKind::UnspecifiedEntityType)
        }
        #[cfg(not(feature = "proto"))]
        {
            false
        }
    }

    /// Returns whether a page address broke its provider's address rules.
    ///
    /// # Examples
    ///
    /// ```
    /// use echora_domain::{EntityType, PageKey, Provider};
    ///
    /// let error = PageKey::new(Provider::Mal, EntityType::Anime, "cowboy-bebop").unwrap_err();
    /// assert!(error.is_invalid_address());
    /// ```
    #[must_use]
    pub fn is_invalid_address(&self) -> bool {
        matches!(self.kind, KeyErrorKind::InvalidAddress(_))
    }
}

impl fmt::Display for KeyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            KeyErrorKind::UnknownProvider(name) => write!(formatter, "unknown provider {name:?}"),
            KeyErrorKind::UnknownEntityType(name) => {
                write!(formatter, "unknown entity type: {name}")
            }
            #[cfg(feature = "proto")]
            KeyErrorKind::UnknownEntityTypeNumber(enum_number) => {
                write!(formatter, "unknown EntityType number {enum_number}")
            }
            #[cfg(feature = "proto")]
            KeyErrorKind::UnspecifiedEntityType => {
                write!(formatter, "entity type must be specified")
            }
            KeyErrorKind::InvalidAddress(message) => formatter.write_str(message),
        }
        .and_then(|()| {
            if self.backtrace.status() == BacktraceStatus::Captured {
                write!(formatter, "\n\nbacktrace:\n{}", self.backtrace)
            } else {
                Ok(())
            }
        })
    }
}

impl std::error::Error for KeyError {}
