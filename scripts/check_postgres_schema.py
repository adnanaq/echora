"""Check that the PostgreSQL migrations match the canonical model in ``anime.py``.

Two checks run over the same migration files:

- Lookup values: each lookup table holds exactly the values of its enum.
- Stored field names: every stored field of ``Anime``, ``Episode`` and ``Character`` has the table or
  column the naming rule in ``docs/architecture/Database Schema.md`` gives it.

The naming rule:

- A scalar field is a column of the same name in the entity's table.
- A list or map field is a table ``<entity>_<field in singular>``; a list nested inside a
  stored object is a table ``<parent table>_<field in singular>``.
- A single nested object is columns ``<field>_<subfield>``, or the subfield's own name
  when it already names its field, or a one-to-one table ``<entity>_<field>``.
- A subfield may live in a shared table the link table references, such as
  ``external_source`` for a link's platform.

Fields the database does not store, or stores somewhere the rule cannot derive, are listed
in ``FIELD_EXCEPTIONS`` with the reason. An exception naming a field that no longer exists
fails the check too, so the list stays current.
"""

import re
import sys
import types
import typing
from enum import StrEnum
from pathlib import Path

from common.models.anime import (
    Anime,
    AnimeRating,
    AnimeRelationType,
    AnimeSeason,
    AnimeStatus,
    AnimeType,
    Character,
    CharacterRole,
    CompanyRole,
    EntityType,
    Episode,
    SourceMaterialRelationType,
    SourceMaterialType,
)
from pydantic import BaseModel

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
    "entity_type": EntityType,
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


ENTITY_MODELS: dict[str, type[BaseModel]] = {
    "anime": Anime,
    "episode": Episode,
    "character": Character,
}

FIELD_EXCEPTIONS: dict[str, str] = {
    "anime.entity_type": "read: always anime",
    "anime.production_staff": "people tables, planned last",
    "anime.related_anime.*": "read from the target work through the join",
    "anime.related_source_material.*": "stored in source_material and its cross-references",
    "anime.companies.sources": "stored in company_xref",
    "anime.companies.roles": "the role key column of anime_company, one row per role",
    "anime.broadcast.day": "typed into weekday by ECHO-88",
    "anime.broadcast.time": "typed into jp_time by ECHO-88",
    "episode.entity_type": "read: always episode",
    "episode.anime_id": "the episode row's foreign key to its work",
    "episode.characters.name": "read from the character",
    "episode.characters.sources": "stored as the character's page key",
    "episode.characters.voice_actors": "people tables, planned last",
    "episode.staff": "people tables, planned last",
    "character.entity_type": "read: always character",
    "character.roles": "read from character_animeography through the join to anime_source",
    "character.anime_ids": "read from character_animeography through the join to anime_source",
    "character.animeography.title": "read from the work the page joins to",
    "character.animeography.sources": "stored as the work's page key",
    "character.mangaography.sources": "matched to source material through source_material_xref",
    "character.voice_actors": "people tables, planned last",
}

_IRREGULAR_SINGULARS = {
    "statistics": "statistics",
    "streaming": "streaming",
    "staff": "staff",
    "companies": "company",
}
_CREATE_TABLE = re.compile(
    r"CREATE\s+TABLE\s+(?P<table>[a-z_]+)\s*\((?P<body>.*?)\n\);", re.DOTALL
)
_TABLE_CONSTRAINT_WORDS = ("CONSTRAINT", "PRIMARY", "UNIQUE", "FOREIGN", "CHECK")
_REFERENCES = re.compile(r"REFERENCES\s+(?P<table>[a-z_]+)\s*\(")


def make_singular(field_name: str) -> str:
    """Return the singular form of a model field name, as table names use it.

    Args:
        field_name: A list or map field name such as ``genres`` or ``companies``.

    Returns:
        The singular name, such as ``genre`` or ``company``.

    >>> make_singular("content_warnings")
    'content_warning'
    >>> make_singular("companies")
    'company'
    >>> make_singular("statistics")
    'statistics'
    """
    if field_name in _IRREGULAR_SINGULARS:
        return _IRREGULAR_SINGULARS[field_name]
    if field_name.endswith("ies"):
        return field_name[:-3] + "y"
    if field_name.endswith("s"):
        return field_name[:-1]
    return field_name


def read_schema(
    migrations_dir: Path,
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Read every table's columns and the tables it references from the migrations.

    Args:
        migrations_dir: Folder of the ``sqlx`` migration files.

    Returns:
        The columns of each table, and the tables each table references.
    """
    columns_by_table: dict[str, set[str]] = {}
    references_by_table: dict[str, set[str]] = {}
    for migration_path in sorted(migrations_dir.glob("*.sql")):
        migration_text = strip_sql_comments(migration_path.read_text(encoding="utf-8"))
        for table_match in _CREATE_TABLE.finditer(migration_text):
            table_name = table_match.group("table")
            body_lines = [
                line.strip() for line in table_match.group("body").splitlines()
            ]
            columns_by_table[table_name] = {
                line.split()[0]
                for line in body_lines
                if line and not line.startswith(_TABLE_CONSTRAINT_WORDS)
            }
            references_by_table[table_name] = set(
                _REFERENCES.findall(table_match.group("body"))
            )
    return columns_by_table, references_by_table


def unwrap_annotation(annotation: object) -> object:
    """Remove ``Optional`` and ``Annotated`` wrappers from a field annotation.

    Args:
        annotation: A Pydantic field annotation.

    Returns:
        The annotation without its optional and annotated wrappers.
    """
    while True:
        origin = typing.get_origin(annotation)
        if origin is typing.Annotated:
            annotation = typing.get_args(annotation)[0]
        elif origin in (typing.Union, types.UnionType):
            present_arguments = [
                argument
                for argument in typing.get_args(annotation)
                if argument is not type(None)
            ]
            annotation = present_arguments[0]
        else:
            return annotation


def is_model(annotation: object) -> typing.TypeIs[type[BaseModel]]:
    """Report whether an annotation is a Pydantic model class.

    Args:
        annotation: An unwrapped field annotation.

    Returns:
        True for a ``BaseModel`` subclass.
    """
    return isinstance(annotation, type) and issubclass(annotation, BaseModel)


class FieldNameCheck:
    """Collect the stored model fields that have no table or column the rule gives them."""

    def __init__(
        self,
        columns_by_table: dict[str, set[str]],
        references_by_table: dict[str, set[str]],
    ) -> None:
        """Hold the schema the check compares the model with.

        Args:
            columns_by_table: The columns of each table.
            references_by_table: The tables each table references.
        """
        self.columns_by_table = columns_by_table
        self.references_by_table = references_by_table
        self.missing_fields: list[str] = []
        self.used_exceptions: set[str] = set()

    def find_exception(self, field_path: str) -> str | None:
        """Return the exception that covers a field path, or None.

        Args:
            field_path: A dotted path such as ``anime.broadcast.day``.

        Returns:
            The matching key of ``FIELD_EXCEPTIONS``, or None.
        """
        if field_path in FIELD_EXCEPTIONS:
            return field_path
        for exception_path in FIELD_EXCEPTIONS:
            if exception_path.endswith(".*") and field_path.startswith(
                exception_path[:-1]
            ):
                return exception_path
        return None

    def has_column(self, table_name: str, column_names: tuple[str, ...]) -> bool:
        """Report whether a table, or a table it references, has one of the columns.

        Args:
            table_name: The table that should hold the field.
            column_names: The column names the rule accepts.

        Returns:
            True when the table or a referenced table has one of the columns.
        """
        candidate_tables = {table_name} | self.references_by_table.get(
            table_name, set()
        )
        return any(
            column_name in self.columns_by_table.get(candidate_table, set())
            for candidate_table in candidate_tables
            for column_name in column_names
        )

    def report_missing(self, field_path: str, expected_location: str) -> None:
        """Record a field with no stored place, unless an exception covers it.

        Args:
            field_path: The dotted path of the field.
            expected_location: Where the rule says the field is stored.
        """
        exception_path = self.find_exception(field_path)
        if exception_path is None:
            self.missing_fields.append(f"{field_path}: expected {expected_location}")
        else:
            self.used_exceptions.add(exception_path)

    def check_fields(
        self, model: type[BaseModel], field_path: str, table_name: str
    ) -> None:
        """Check every field of a model stored in one table.

        Args:
            model: The model whose fields are checked.
            field_path: The dotted path of the model, such as ``anime``.
            table_name: The table that holds the model's own values.
        """
        for field_name, field_info in model.model_fields.items():
            child_path = f"{field_path}.{field_name}"
            exception_path = self.find_exception(child_path)
            if exception_path is not None:
                self.used_exceptions.add(exception_path)
                continue
            annotation = unwrap_annotation(field_info.annotation)
            origin = typing.get_origin(annotation)
            if origin is list:
                self.check_list(child_path, table_name, field_name, annotation)
            elif origin is dict:
                self.check_map(child_path, table_name, field_name, annotation)
            elif is_model(annotation):
                self.check_nested_object(child_path, table_name, field_name, annotation)
            elif not self.has_column(table_name, (field_name,)):
                self.report_missing(child_path, f"column {table_name}.{field_name}")

    def check_list(
        self, field_path: str, table_name: str, field_name: str, annotation: object
    ) -> None:
        """Check a list field, stored in a table named after its singular form.

        Args:
            field_path: The dotted path of the field.
            table_name: The table of the model that holds the list.
            field_name: The list field's name.
            annotation: The unwrapped ``list[...]`` annotation.
        """
        list_table = f"{table_name}_{make_singular(field_name)}"
        if list_table not in self.columns_by_table:
            self.report_missing(field_path, f"table {list_table}")
            return
        item_annotation = unwrap_annotation(typing.get_args(annotation)[0])
        if is_model(item_annotation):
            self.check_fields(item_annotation, field_path, list_table)

    def check_map(
        self, field_path: str, table_name: str, field_name: str, annotation: object
    ) -> None:
        """Check a map field, stored in a table named after its singular form.

        Args:
            field_path: The dotted path of the field.
            table_name: The table of the model that holds the map.
            field_name: The map field's name.
            annotation: The unwrapped ``dict[...]`` annotation.
        """
        map_table = f"{table_name}_{make_singular(field_name)}"
        if map_table not in self.columns_by_table:
            self.report_missing(field_path, f"table {map_table}")
            return
        value_annotation = unwrap_annotation(typing.get_args(annotation)[1])
        if typing.get_origin(value_annotation) is list:
            value_annotation = unwrap_annotation(typing.get_args(value_annotation)[0])
        if is_model(value_annotation):
            self.check_fields(value_annotation, field_path, map_table)

    def check_nested_object(
        self,
        field_path: str,
        table_name: str,
        field_name: str,
        nested_model: type[BaseModel],
    ) -> None:
        """Check a single nested object, stored as columns or in a one-to-one table.

        - With a table ``<table>_<field>``, each subfield is a column there.
        - An object of lists, such as ``images``, is one table ``<table>_<field in singular>``.
        - Otherwise each subfield is a column ``<field>_<subfield>`` or ``<subfield>``.

        Args:
            field_path: The dotted path of the field.
            table_name: The table of the model that holds the object.
            field_name: The object field's name.
            nested_model: The nested model class.
        """
        own_table = f"{table_name}_{field_name}"
        if own_table in self.columns_by_table:
            self.check_fields(nested_model, field_path, own_table)
            return
        singular_table = f"{table_name}_{make_singular(field_name)}"
        if singular_table in self.columns_by_table:
            return
        for subfield_name in nested_model.model_fields:
            subfield_path = f"{field_path}.{subfield_name}"
            if not self.has_column(
                table_name, (f"{field_name}_{subfield_name}", subfield_name)
            ):
                self.report_missing(
                    subfield_path, f"column {table_name}.{field_name}_{subfield_name}"
                )


def find_naming_problems(migrations_dir: Path) -> list[str]:
    """List every stored model field without its table or column, and every stale exception.

    Args:
        migrations_dir: Folder of the ``sqlx`` migration files.

    Returns:
        One line per problem; empty when the schema follows the rule.
    """
    columns_by_table, references_by_table = read_schema(migrations_dir)
    field_check = FieldNameCheck(columns_by_table, references_by_table)
    for entity_table, entity_model in ENTITY_MODELS.items():
        field_check.check_fields(entity_model, entity_table, entity_table)
    stale_exceptions = [
        f"{exception_path}: exception names no stored field"
        for exception_path in sorted(
            set(FIELD_EXCEPTIONS) - field_check.used_exceptions
        )
    ]
    return field_check.missing_fields + stale_exceptions


def main() -> int:
    """Run both checks against the repository's migrations and report every problem.

    Returns:
        Zero when the lookup tables and stored field names match the model; one otherwise.
    """
    values_by_table = extract_lookup_values(MIGRATIONS_DIR)
    schema_problems = find_lookup_mismatches(values_by_table) + find_naming_problems(
        MIGRATIONS_DIR
    )
    if schema_problems:
        for schema_problem in schema_problems:
            print(f"ERROR: {schema_problem}", file=sys.stderr)
        return 1
    print(f"All {len(ENUM_TABLE_MAP)} lookup tables match anime.py enums.")
    print("Every stored model field follows the naming rule.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
