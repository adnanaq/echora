"""Verify that PostgreSQL lookup table migrations match canonical anime enums."""

import re
import sys
from enum import StrEnum
from pathlib import Path

from common.models.anime import (
    AnimeRating,
    AnimeRelationType,
    AnimeSeason,
    AnimeStatus,
    AnimeType,
    CharacterRole,
    CompanyRole,
    SourceMaterialRelationType,
    SourceMaterialType,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
MIGRATIONS_DIR = REPO_ROOT / "crates" / "postgres_db" / "migrations"

ENUM_TABLE_MAP: dict[str, type[StrEnum]] = {
    "anime_type": AnimeType,
    "anime_status": AnimeStatus,
    "anime_rating": AnimeRating,
    "anime_season": AnimeSeason,
    "source_material_type": SourceMaterialType,
    "character_role": CharacterRole,
    "company_role": CompanyRole,
    "anime_relation_type": AnimeRelationType,
    "source_material_relation_type": SourceMaterialRelationType,
}

_SQL_TOKEN_PATTERN = re.compile(
    r"'(?:[^']|'')*'|--[^\n]*|/\*.*?\*/",
    re.DOTALL,
)
_SQL_VALUE_BODY = r"(?:[^;']|'(?:[^']|'')*')*"
_STATEMENT_PATTERN = re.compile(
    rf"(?:"
    rf"INSERT\s+INTO\s+(?P<insert_table>[a-z_]+)(?:\s*\([^)]*\))?\s+VALUES\s*(?P<insert_values>{_SQL_VALUE_BODY})"
    rf"|DELETE\s+FROM\s+(?P<delete_table>[a-z_]+)(?:\s+WHERE\s+(?P<delete_where>{_SQL_VALUE_BODY}))?"
    rf"|TRUNCATE(?:\s+TABLE)?\s+(?P<truncate_table>[a-z_]+)"
    rf")\s*;",
    re.IGNORECASE | re.DOTALL,
)
_STRING_LITERAL_PATTERN = re.compile(r"'((?:[^']|'')*)'")


def strip_sql_comments(sql_text: str) -> str:
    """Remove line and block SQL comments while preserving string literals.

    Args:
        sql_text: Raw SQL migration file contents.

    Returns:
        SQL text with comments replaced by spaces and string literals intact.
    """

    def _replace_token(match: re.Match[str]) -> str:
        token = match.group(0)
        return token if token.startswith("'") else " "

    return _SQL_TOKEN_PATTERN.sub(_replace_token, sql_text)


def _parse_string_literals(sql_fragment: str) -> set[str]:
    """Decode single-quoted SQL string literals from a statement fragment.

    Args:
        sql_fragment: SQL fragment containing single-quoted string literals.

    Returns:
        Set of decoded string values with escaped quotes unescaped.
    """
    return {
        literal.replace("''", "'")
        for literal in _STRING_LITERAL_PATTERN.findall(sql_fragment)
    }


def extract_lookup_values(migrations_dir: Path) -> dict[str, set[str]]:
    """Compute final lookup values per table after replaying SQL migrations.

    Args:
        migrations_dir: Directory containing timestamped ``sqlx`` SQL files.

    Returns:
        Mapping from lookup table name to the set of active string values.
    """
    values_by_table: dict[str, set[str]] = {}
    for migration_path in sorted(migrations_dir.glob("*.sql")):
        cleaned_sql = strip_sql_comments(migration_path.read_text(encoding="utf-8"))
        for match in _STATEMENT_PATTERN.finditer(cleaned_sql):
            if match.group("insert_table"):
                table_name = match.group("insert_table").lower()
                values_by_table.setdefault(table_name, set()).update(
                    _parse_string_literals(match.group("insert_values"))
                )
            elif match.group("delete_table"):
                table_name = match.group("delete_table").lower()
                delete_where = match.group("delete_where")
                if delete_where is None:
                    values_by_table.setdefault(table_name, set()).clear()
                else:
                    values_by_table.setdefault(table_name, set()).difference_update(
                        _parse_string_literals(delete_where)
                    )
            elif match.group("truncate_table"):
                table_name = match.group("truncate_table").lower()
                values_by_table.setdefault(table_name, set()).clear()
    return values_by_table


def find_lookup_mismatches(
    values_by_table: dict[str, set[str]],
) -> list[str]:
    """Compare canonical ``anime.py`` enum values against migration seed rows.

    Args:
        values_by_table: Final lookup values parsed from migration SQL files.

    Returns:
        Human-readable mismatch descriptions; empty when every table matches.
    """
    mismatches: list[str] = []
    for table_name, enum_class in ENUM_TABLE_MAP.items():
        enum_values = {member.value for member in enum_class}
        migration_values = values_by_table.get(table_name, set())
        missing_in_migration = sorted(enum_values - migration_values)
        extra_in_migration = sorted(migration_values - enum_values)
        if missing_in_migration:
            mismatches.append(
                f"{table_name}: missing from migrations: {missing_in_migration}"
            )
        if extra_in_migration:
            mismatches.append(
                f"{table_name}: missing from anime.py enum: {extra_in_migration}"
            )
    return mismatches


def main() -> int:
    """Run the enum-to-lookup-table drift check and report any mismatches.

    Returns:
        Zero when all lookup tables match; one when drift is detected.
    """
    values_by_table = extract_lookup_values(MIGRATIONS_DIR)
    mismatches = find_lookup_mismatches(values_by_table)
    if mismatches:
        for mismatch in mismatches:
            print(f"ERROR: {mismatch}", file=sys.stderr)
        return 1
    print(f"All {len(ENUM_TABLE_MAP)} lookup tables match anime.py enums.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
