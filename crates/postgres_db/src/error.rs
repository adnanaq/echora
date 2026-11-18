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
    InvalidArgument(String),
    VersionConflict(String),
    UniqueViolation(String),
    ForeignKeyViolation(String),
    Migration(sqlx::migrate::MigrateError),
    Database(sqlx::Error),
}

impl Error {
    /// Creates a not-found error.
    #[must_use]
    pub fn not_found() -> Self {
        Self {
            kind: ErrorKind::NotFound,
        }
    }

    /// Creates an invalid-argument error with `message`.
    #[must_use]
    pub fn invalid_argument(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::InvalidArgument(message.into()),
        }
    }

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

    /// Returns `true` when the caller passed an invalid argument.
    #[must_use]
    pub fn is_invalid_argument(&self) -> bool {
        matches!(self.kind, ErrorKind::InvalidArgument(_))
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
            ErrorKind::InvalidArgument(message) => write!(formatter, "invalid argument: {message}"),
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
            | ErrorKind::InvalidArgument(_)
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
    use std::borrow::Cow;
    use std::error::Error as _;

    use super::*;

    #[derive(Debug)]
    struct StubDatabaseError {
        code: Option<&'static str>,
        message: &'static str,
    }

    impl fmt::Display for StubDatabaseError {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str(self.message)
        }
    }

    impl std::error::Error for StubDatabaseError {}

    impl sqlx::error::DatabaseError for StubDatabaseError {
        fn message(&self) -> &str {
            self.message
        }

        fn code(&self) -> Option<Cow<'_, str>> {
            self.code.map(Cow::Borrowed)
        }

        fn as_error(&self) -> &(dyn std::error::Error + Send + Sync + 'static) {
            self
        }

        fn as_error_mut(&mut self) -> &mut (dyn std::error::Error + Send + Sync + 'static) {
            self
        }

        fn into_error(self: Box<Self>) -> Box<dyn std::error::Error + Send + Sync + 'static> {
            self
        }

        fn kind(&self) -> sqlx::error::ErrorKind {
            sqlx::error::ErrorKind::Other
        }
    }

    #[test]
    fn error_kind_predicates_match_variants() {
        let not_found = Error::from(sqlx::Error::RowNotFound);
        assert!(not_found.is_not_found());
        assert!(!not_found.is_version_conflict());
        assert!(not_found.source().is_none());
        assert_eq!(not_found.to_string(), "record not found");

        let conflict = Error::version_conflict("expected version 2, found 3");
        assert!(conflict.is_version_conflict());
        assert!(!conflict.is_not_found());
        assert!(conflict.to_string().contains("version conflict"));
        assert!(conflict.source().is_none());

        let unique = Error::from(sqlx::Error::Database(Box::new(StubDatabaseError {
            code: Some("23505"),
            message: "duplicate key",
        })));
        assert!(unique.is_unique_violation());
        assert!(unique.to_string().contains("unique constraint violation"));
        assert!(unique.source().is_none());

        let foreign_key = Error::from(sqlx::Error::Database(Box::new(StubDatabaseError {
            code: Some("23503"),
            message: "missing parent",
        })));
        assert!(foreign_key.is_foreign_key_violation());
        assert!(
            foreign_key
                .to_string()
                .contains("foreign key constraint violation")
        );
        assert!(foreign_key.source().is_none());

        let serialization = Error::from(sqlx::Error::Database(Box::new(StubDatabaseError {
            code: Some("40001"),
            message: "could not serialize access",
        })));
        assert!(serialization.is_version_conflict());
        assert!(serialization.to_string().contains("version conflict"));

        let other_db = Error::from(sqlx::Error::Database(Box::new(StubDatabaseError {
            code: Some("42P01"),
            message: "undefined table",
        })));
        assert!(other_db.is_database());
        assert!(other_db.source().is_some());

        let database = Error::from(sqlx::Error::PoolTimedOut);
        assert!(database.is_database());
        assert!(database.source().is_some());
        assert!(database.to_string().contains("database error"));

        let migration = Error::from(sqlx::migrate::MigrateError::VersionMissing(42));
        assert!(migration.is_migration());
        assert!(migration.source().is_some());
        assert!(migration.to_string().contains("migration failed"));
    }
}
