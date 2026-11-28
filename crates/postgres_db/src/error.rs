//! Error types for the `postgres_db` crate.

use std::backtrace::{Backtrace, BacktraceStatus};
use std::fmt;

/// Represents failures from PostgreSQL connections, queries, and migrations.
#[derive(Debug)]
pub struct Error {
    kind: ErrorKind,
    backtrace: Backtrace,
}

#[derive(Debug)]
enum ErrorKind {
    NotFound,
    InvalidArgument(String),
    InvalidKey(Box<echora_domain::KeyError>),
    VersionConflict(String),
    UniqueViolation(String),
    ForeignKeyViolation(String),
    Migration(sqlx::migrate::MigrateError),
    Database(sqlx::Error),
}

impl Error {
    /// Creates an error of `kind`, capturing a backtrace when `RUST_BACKTRACE` asks for one.
    fn from_kind(kind: ErrorKind) -> Self {
        Self {
            kind,
            backtrace: Backtrace::capture(),
        }
    }

    /// Creates a not-found error.
    ///
    /// # Examples
    ///
    /// ```
    /// use postgres_db::Error;
    ///
    /// let err = Error::not_found();
    /// assert!(err.is_not_found());
    /// assert!(!err.is_version_conflict());
    /// assert!(err.to_string().starts_with("record not found"));
    /// ```
    #[must_use]
    pub fn not_found() -> Self {
        Self::from_kind(ErrorKind::NotFound)
    }

    /// Creates an invalid-argument error with `message`.
    ///
    /// # Examples
    ///
    /// ```
    /// use postgres_db::Error;
    ///
    /// let err = Error::invalid_argument("title must not be empty");
    /// assert!(err.is_invalid_argument());
    /// assert!(!err.is_not_found());
    /// assert!(err.to_string().starts_with("invalid argument: title must not be empty"));
    /// ```
    #[must_use]
    pub fn invalid_argument(message: impl Into<String>) -> Self {
        Self::from_kind(ErrorKind::InvalidArgument(message.into()))
    }

    /// Creates an optimistic concurrency or version conflict error with `message`.
    ///
    /// # Examples
    ///
    /// ```
    /// use postgres_db::Error;
    ///
    /// let err = Error::version_conflict("expected version 1, found 2");
    /// assert!(err.is_version_conflict());
    /// assert!(!err.is_invalid_argument());
    /// assert!(err.to_string().starts_with("version conflict: expected version 1, found 2"));
    /// ```
    #[must_use]
    pub fn version_conflict(message: impl Into<String>) -> Self {
        Self::from_kind(ErrorKind::VersionConflict(message.into()))
    }

    /// Creates a unique-constraint error with `message`, as PostgreSQL would report it.
    ///
    /// Only callers' tests need this, since real ones come from the database driver.
    ///
    /// # Examples
    ///
    /// ```
    /// use postgres_db::Error;
    ///
    /// let err = Error::unique_violation("duplicate key value violates episode_number_uniq");
    /// assert!(err.is_unique_violation());
    /// assert!(!err.is_foreign_key_violation());
    /// assert!(err.to_string().starts_with(
    ///     "unique constraint violation: duplicate key value violates episode_number_uniq"
    /// ));
    /// ```
    #[cfg(feature = "test-util")]
    #[must_use]
    pub fn unique_violation(message: impl Into<String>) -> Self {
        Self::from_kind(ErrorKind::UniqueViolation(message.into()))
    }

    /// Creates a foreign-key error with `message`, as PostgreSQL would report it.
    ///
    /// Only callers' tests need this, since real ones come from the database driver.
    ///
    /// # Examples
    ///
    /// ```
    /// use postgres_db::Error;
    ///
    /// let err = Error::foreign_key_violation("insert violates provider_page_provider_fkey");
    /// assert!(err.is_foreign_key_violation());
    /// assert!(!err.is_unique_violation());
    /// assert!(err.to_string().starts_with(
    ///     "foreign key constraint violation: insert violates provider_page_provider_fkey"
    /// ));
    /// ```
    #[cfg(feature = "test-util")]
    #[must_use]
    pub fn foreign_key_violation(message: impl Into<String>) -> Self {
        Self::from_kind(ErrorKind::ForeignKeyViolation(message.into()))
    }

    /// Returns `true` when a required row was not found.
    ///
    /// # Examples
    ///
    /// ```
    /// use postgres_db::Error;
    ///
    /// let missing_row = Error::from(sqlx::Error::RowNotFound);
    /// assert!(missing_row.is_not_found());
    ///
    /// let stale_save = Error::version_conflict("stale row version");
    /// assert!(!stale_save.is_not_found());
    /// ```
    #[must_use]
    pub fn is_not_found(&self) -> bool {
        matches!(self.kind, ErrorKind::NotFound)
    }

    /// Returns `true` when the caller passed an invalid argument.
    ///
    /// Text PostgreSQL cannot store, such as a NUL byte, counts too, since it never succeeds.
    ///
    /// # Examples
    ///
    /// ```
    /// use postgres_db::Error;
    ///
    /// let bad_input = Error::invalid_argument("bad input");
    /// assert!(bad_input.is_invalid_argument());
    ///
    /// let missing_row = Error::not_found();
    /// assert!(!missing_row.is_invalid_argument());
    /// ```
    #[must_use]
    pub fn is_invalid_argument(&self) -> bool {
        matches!(
            self.kind,
            ErrorKind::InvalidArgument(_) | ErrorKind::InvalidKey(_)
        )
    }

    /// Returns `true` when an optimistic concurrency or serialization conflict occurred.
    ///
    /// # Examples
    ///
    /// ```
    /// use postgres_db::Error;
    ///
    /// let stale_save = Error::version_conflict("stale row version");
    /// assert!(stale_save.is_version_conflict());
    ///
    /// let missing_row = Error::not_found();
    /// assert!(!missing_row.is_version_conflict());
    /// ```
    #[must_use]
    pub fn is_version_conflict(&self) -> bool {
        matches!(self.kind, ErrorKind::VersionConflict(_))
    }

    /// Returns `true` when a unique constraint was violated.
    ///
    /// # Examples
    ///
    /// ```
    /// use postgres_db::Error;
    ///
    /// let missing_row = Error::not_found();
    /// assert!(!missing_row.is_unique_violation());
    ///
    /// let bad_input = Error::invalid_argument("duplicate episode number");
    /// assert!(!bad_input.is_unique_violation(), "only PostgreSQL's own check counts");
    /// ```
    #[must_use]
    pub fn is_unique_violation(&self) -> bool {
        matches!(self.kind, ErrorKind::UniqueViolation(_))
    }

    /// Returns `true` when a foreign key constraint was violated.
    ///
    /// # Examples
    ///
    /// ```
    /// use postgres_db::Error;
    ///
    /// let missing_row = Error::not_found();
    /// assert!(!missing_row.is_foreign_key_violation());
    ///
    /// let bad_input = Error::invalid_argument("unknown provider");
    /// assert!(!bad_input.is_foreign_key_violation(), "only PostgreSQL's own check counts");
    /// ```
    #[must_use]
    pub fn is_foreign_key_violation(&self) -> bool {
        matches!(self.kind, ErrorKind::ForeignKeyViolation(_))
    }

    /// Returns `true` when running embedded `sqlx` migrations failed.
    ///
    /// # Examples
    ///
    /// ```
    /// use postgres_db::Error;
    ///
    /// let err = Error::from(sqlx::migrate::MigrateError::VersionMissing(1));
    /// assert!(err.is_migration());
    /// assert!(!err.is_database());
    /// assert!(err.to_string().starts_with("migration failed: "));
    /// ```
    #[must_use]
    pub fn is_migration(&self) -> bool {
        matches!(self.kind, ErrorKind::Migration(_))
    }

    /// Returns `true` when an unclassified database driver error occurred.
    ///
    /// # Examples
    ///
    /// ```
    /// use postgres_db::Error;
    ///
    /// let err = Error::from(sqlx::Error::PoolTimedOut);
    /// assert!(err.is_database());
    /// assert!(!err.is_version_conflict());
    /// assert!(err.to_string().starts_with("database error: "));
    /// ```
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
            ErrorKind::InvalidKey(err) => write!(formatter, "invalid argument: {err}"),
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
        .and_then(|()| {
            if self.backtrace.status() == BacktraceStatus::Captured {
                write!(formatter, "\n\nbacktrace:\n{}", self.backtrace)
            } else {
                Ok(())
            }
        })
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.kind {
            ErrorKind::Migration(err) => Some(err),
            ErrorKind::Database(err) => Some(err),
            ErrorKind::InvalidKey(err) => Some(err.as_ref()),
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
            sqlx::Error::RowNotFound => Self::from_kind(ErrorKind::NotFound),
            sqlx::Error::Database(db_err) => {
                /// PostgreSQL SQLSTATE for `unique_violation`.
                const PG_UNIQUE_VIOLATION: &str = "23505";
                /// PostgreSQL SQLSTATE for `foreign_key_violation`.
                const PG_FOREIGN_KEY_VIOLATION: &str = "23503";
                /// PostgreSQL SQLSTATE for `serialization_failure` (REPEATABLE READ / SERIALIZABLE).
                const PG_SERIALIZATION_FAILURE: &str = "40001";
                /// PostgreSQL SQLSTATE for `character_not_in_repertoire`, such as a NUL byte in
                /// text. The same input always fails, so it is the caller's to fix, not to retry.
                const PG_CHARACTER_NOT_IN_REPERTOIRE: &str = "22021";

                let code = db_err.code();
                let kind = match code.as_deref() {
                    Some(PG_UNIQUE_VIOLATION) => {
                        ErrorKind::UniqueViolation(db_err.message().to_owned())
                    }
                    Some(PG_FOREIGN_KEY_VIOLATION) => {
                        ErrorKind::ForeignKeyViolation(db_err.message().to_owned())
                    }
                    Some(PG_SERIALIZATION_FAILURE) => {
                        ErrorKind::VersionConflict(db_err.message().to_owned())
                    }
                    Some(PG_CHARACTER_NOT_IN_REPERTOIRE) => {
                        ErrorKind::InvalidArgument(db_err.message().to_owned())
                    }
                    _ => ErrorKind::Database(err),
                };
                Self::from_kind(kind)
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
            | _ => Self::from_kind(ErrorKind::Database(err)),
        }
    }
}

impl From<sqlx::migrate::MigrateError> for Error {
    fn from(err: sqlx::migrate::MigrateError) -> Self {
        Self::from_kind(ErrorKind::Migration(err))
    }
}

/// Treats a refused provider, entity type or page address as an invalid argument.
impl From<echora_domain::KeyError> for Error {
    fn from(err: echora_domain::KeyError) -> Self {
        Self::from_kind(ErrorKind::InvalidKey(Box::new(err)))
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error as _;

    use super::*;

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

        let database = Error::from(sqlx::Error::PoolTimedOut);
        assert!(database.is_database());
        assert!(database.source().is_some());
        assert!(database.to_string().contains("database error"));

        let invalid_key = Error::from(
            "myanimelist"
                .parse::<echora_domain::Provider>()
                .unwrap_err(),
        );
        assert!(invalid_key.is_invalid_argument());
        assert!(invalid_key.source().is_some());
        assert!(
            invalid_key
                .to_string()
                .starts_with("invalid argument: unknown provider \"myanimelist\"")
        );

        let migration = Error::from(sqlx::migrate::MigrateError::VersionMissing(42));
        assert!(migration.is_migration());
        assert!(migration.source().is_some());
        assert!(migration.to_string().contains("migration failed"));
    }

    async fn raise_database_error(pool: &sqlx::PgPool, sqlstate: &str) -> Error {
        let raise_statement = format!(
            "DO $$ BEGIN RAISE EXCEPTION 'raised by test' USING ERRCODE = '{sqlstate}'; END $$"
        );
        Error::from(
            sqlx::query(&raise_statement)
                .execute(pool)
                .await
                .expect_err("the statement raises"),
        )
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn from_database_error_classifies_sqlstate(pool: sqlx::PgPool) {
        let unique = raise_database_error(&pool, "23505").await;
        assert!(unique.is_unique_violation());
        assert!(unique.to_string().contains("unique constraint violation"));
        assert!(unique.source().is_none());

        let foreign_key = raise_database_error(&pool, "23503").await;
        assert!(foreign_key.is_foreign_key_violation());
        assert!(
            foreign_key
                .to_string()
                .contains("foreign key constraint violation")
        );
        assert!(foreign_key.source().is_none());

        let serialization = raise_database_error(&pool, "40001").await;
        assert!(serialization.is_version_conflict());
        assert!(serialization.to_string().contains("version conflict"));

        let undefined_table = raise_database_error(&pool, "42P01").await;
        assert!(undefined_table.is_database());
        assert!(undefined_table.source().is_some());
    }

    #[test]
    fn display_captured_backtrace_follows_summary() {
        let traced_error = Error {
            kind: ErrorKind::NotFound,
            backtrace: Backtrace::force_capture(),
        };
        let untraced_error = Error {
            kind: ErrorKind::NotFound,
            backtrace: Backtrace::disabled(),
        };

        let traced_text = traced_error.to_string();

        assert!(traced_text.starts_with("record not found\n\nbacktrace:\n"));
        assert_eq!(untraced_error.to_string(), "record not found");
    }
}
