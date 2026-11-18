//! Error types for the `postgres_db` crate.

// Rust guideline compliant 2026-10-09

use std::fmt;

/// Represents failures from PostgreSQL connections, queries, and migrations.
#[derive(Debug)]
pub struct Error {
    kind: ErrorKind,
}

#[derive(Debug)]
enum ErrorKind {
    NotFound,
    VersionConflict(String),
    UniqueViolation(String),
    ForeignKeyViolation(String),
    Migration(sqlx::migrate::MigrateError),
    Database(sqlx::Error),
}

impl Error {
    /// Creates an optimistic concurrency or version conflict error with `message`.
    #[must_use]
    pub fn version_conflict(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::VersionConflict(message.into()),
        }
    }

    /// Returns `true` when a required row was not found.
    ///
    /// # Examples
    ///
    /// ```
    /// use postgres_db::Error;
    ///
    /// let err = Error::from(sqlx::Error::RowNotFound);
    /// assert!(err.is_not_found());
    /// ```
    #[must_use]
    pub fn is_not_found(&self) -> bool {
        matches!(self.kind, ErrorKind::NotFound)
    }

    /// Returns `true` when an optimistic concurrency or serialization conflict occurred.
    ///
    /// # Examples
    ///
    /// ```
    /// use postgres_db::Error;
    ///
    /// let err = Error::version_conflict("stale row version");
    /// assert!(err.is_version_conflict());
    /// ```
    #[must_use]
    pub fn is_version_conflict(&self) -> bool {
        matches!(self.kind, ErrorKind::VersionConflict(_))
    }

    /// Returns `true` when a unique constraint was violated.
    #[must_use]
    pub fn is_unique_violation(&self) -> bool {
        matches!(self.kind, ErrorKind::UniqueViolation(_))
    }

    /// Returns `true` when a foreign key constraint was violated.
    #[must_use]
    pub fn is_foreign_key_violation(&self) -> bool {
        matches!(self.kind, ErrorKind::ForeignKeyViolation(_))
    }

    /// Returns `true` when running embedded `sqlx` migrations failed.
    #[must_use]
    pub fn is_migration(&self) -> bool {
        matches!(self.kind, ErrorKind::Migration(_))
    }

    /// Returns `true` when an unclassified database driver error occurred.
    #[must_use]
    pub fn is_database(&self) -> bool {
        matches!(self.kind, ErrorKind::Database(_))
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            ErrorKind::NotFound => write!(formatter, "record not found"),
            ErrorKind::VersionConflict(message) => write!(formatter, "version conflict: {message}"),
            ErrorKind::UniqueViolation(message) => {
                write!(formatter, "unique constraint violation: {message}")
            }
            ErrorKind::ForeignKeyViolation(message) => {
                write!(formatter, "foreign key constraint violation: {message}")
            }
            ErrorKind::Migration(err) => write!(formatter, "migration failed: {err}"),
            ErrorKind::Database(err) => write!(formatter, "database error: {err}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.kind {
            ErrorKind::Migration(err) => Some(err),
            ErrorKind::Database(err) => Some(err),
            ErrorKind::NotFound
            | ErrorKind::VersionConflict(_)
            | ErrorKind::UniqueViolation(_)
            | ErrorKind::ForeignKeyViolation(_) => None,
        }
    }
}

impl From<sqlx::Error> for Error {
    fn from(err: sqlx::Error) -> Self {
        match &err {
            sqlx::Error::RowNotFound => Self {
                kind: ErrorKind::NotFound,
            },
            sqlx::Error::Database(db_err) => {
                let code = db_err.code();
                let kind = match code.as_deref() {
                    Some("23505") => ErrorKind::UniqueViolation(db_err.message().to_owned()),
                    Some("23503") => ErrorKind::ForeignKeyViolation(db_err.message().to_owned()),
                    Some("40001") => ErrorKind::VersionConflict(db_err.message().to_owned()),
                    _ => ErrorKind::Database(err),
                };
                Self { kind }
            }
            sqlx::Error::Configuration(_)
            | sqlx::Error::InvalidArgument(_)
            | sqlx::Error::Io(_)
            | sqlx::Error::Tls(_)
            | sqlx::Error::Protocol(_)
            | sqlx::Error::TypeNotFound { .. }
            | sqlx::Error::ColumnIndexOutOfBounds { .. }
            | sqlx::Error::ColumnNotFound(_)
            | sqlx::Error::ColumnDecode { .. }
            | sqlx::Error::Encode(_)
            | sqlx::Error::Decode(_)
            | sqlx::Error::AnyDriverError(_)
            | sqlx::Error::PoolTimedOut
            | sqlx::Error::PoolClosed
            | sqlx::Error::WorkerCrashed
            | sqlx::Error::Migrate(_)
            | sqlx::Error::InvalidSavePointStatement
            | sqlx::Error::BeginFailed
            | _ => Self {
                kind: ErrorKind::Database(err),
            },
        }
    }
}

impl From<sqlx::migrate::MigrateError> for Error {
    fn from(err: sqlx::migrate::MigrateError) -> Self {
        Self {
            kind: ErrorKind::Migration(err),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_kind_predicates_match_variants() {
        let not_found = Error::from(sqlx::Error::RowNotFound);
        assert!(not_found.is_not_found());
        assert!(!not_found.is_version_conflict());

        let conflict = Error::version_conflict("expected version 2, found 3");
        assert!(conflict.is_version_conflict());
        assert!(!conflict.is_not_found());
        assert!(conflict.to_string().contains("version conflict"));
    }
}
