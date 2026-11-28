//! `sqlx` reading of providers and entity types from PostgreSQL `text` lookup columns.
//!
//! Queries bind these values with `as_str()`, so only reading needs `sqlx` support. A stored
//! name the enum does not know fails the row with `sqlx::Error::ColumnDecode`.

use sqlx::error::BoxDynError;
use sqlx::postgres::{PgTypeInfo, PgValueRef};
use sqlx::{Decode, Postgres, Type};

use crate::{EntityType, Provider};

/// Reads a [`Provider`] from the `text` name its `provider` lookup row holds.
impl Type<Postgres> for Provider {
    fn type_info() -> PgTypeInfo {
        <&str as Type<Postgres>>::type_info()
    }

    fn compatible(column_type: &PgTypeInfo) -> bool {
        <&str as Type<Postgres>>::compatible(column_type)
    }
}

/// Reads a stored provider name through [`Provider`]'s `FromStr`.
impl<'row> Decode<'row, Postgres> for Provider {
    fn decode(value: PgValueRef<'row>) -> Result<Self, BoxDynError> {
        Ok(<&str as Decode<Postgres>>::decode(value)?.parse()?)
    }
}

/// Reads an [`EntityType`] from the `text` value its `entity_type` lookup row holds.
impl Type<Postgres> for EntityType {
    fn type_info() -> PgTypeInfo {
        <&str as Type<Postgres>>::type_info()
    }

    fn compatible(column_type: &PgTypeInfo) -> bool {
        <&str as Type<Postgres>>::compatible(column_type)
    }
}

/// Reads a stored entity type through [`EntityType`]'s `FromStr`.
impl<'row> Decode<'row, Postgres> for EntityType {
    fn decode(value: PgValueRef<'row>) -> Result<Self, BoxDynError> {
        Ok(<&str as Decode<Postgres>>::decode(value)?.parse()?)
    }
}
