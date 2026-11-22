import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import check_postgres_schema  # noqa: E402
from check_postgres_schema import (  # noqa: E402
    ENUM_TABLE_MAP,
    MIGRATIONS_DIR,
    extract_lookup_values,
    find_lookup_mismatches,
    find_naming_problems,
    main,
    make_singular,
    strip_sql_comments,
)


def test_extract_lookup_values_handles_column_lists_quotes_and_comments(
    tmp_path: Path,
) -> None:
    sql_file = tmp_path / "001_test.sql"
    sql_file.write_text(
        """
        -- INSERT INTO anime_status VALUES ('UNKNOWN');
        /* INSERT INTO anime_status VALUES ('ONGOING'); */
        INSERT INTO anime_type (value) VALUES ('TV'), ('--literal'), ('O''Brien');
        INSERT INTO anime_status VALUES ('FINISHED');
        """,
        encoding="utf-8",
    )

    lookup_values = extract_lookup_values(tmp_path)

    assert lookup_values["anime_type"] == {"TV", "--literal", "O'Brien"}
    assert lookup_values["anime_status"] == {"FINISHED"}
    assert "--literal" in strip_sql_comments("SELECT '--literal'; -- comment")


def test_extract_lookup_values_applies_deletes_and_truncates_in_order(
    tmp_path: Path,
) -> None:
    (tmp_path / "001_seed.sql").write_text(
        """
        INSERT INTO anime_type VALUES ('TV'), ('MOVIE'), ('OVA');
        INSERT INTO anime_status VALUES ('FINISHED'), ('ONGOING');
        INSERT INTO anime_rating VALUES ('G'), ('PG');
        """,
        encoding="utf-8",
    )
    (tmp_path / "002_prune.sql").write_text(
        """
        DELETE FROM anime_type WHERE code = 'TV';
        DELETE FROM anime_type WHERE code IN ('OVA');
        INSERT INTO anime_type VALUES ('SPECIAL'), ('TV');
        DELETE FROM anime_type WHERE code = 'TV';
        DELETE FROM anime_status;
        TRUNCATE TABLE anime_rating;
        INSERT INTO anime_rating VALUES ('R');
        """,
        encoding="utf-8",
    )

    lookup_values = extract_lookup_values(tmp_path)

    assert lookup_values["anime_type"] == {"MOVIE", "SPECIAL"}
    assert lookup_values["anime_status"] == set()
    assert lookup_values["anime_rating"] == {"R"}


def test_find_lookup_mismatches_detects_missing_and_extra_values() -> None:
    complete = {
        table_name: {member.value for member in enum_class}
        for table_name, enum_class in ENUM_TABLE_MAP.items()
    }
    assert find_lookup_mismatches(complete) == []

    drifted = {table_name: set(values) for table_name, values in complete.items()}
    drifted["anime_status"].discard("UNKNOWN")
    drifted["anime_status"].add("STALE_VALUE")

    mismatches = find_lookup_mismatches(drifted)

    assert len(mismatches) == 2
    assert "missing from migrations: ['UNKNOWN']" in mismatches[0]
    assert "missing from anime.py enum: ['STALE_VALUE']" in mismatches[1]


def test_main_returns_zero_when_matching_and_one_on_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main() == 0
    assert "All 10 lookup tables match anime.py enums." in capsys.readouterr().out

    (tmp_path / "001_empty.sql").write_text("", encoding="utf-8")
    monkeypatch.setattr(check_postgres_schema, "MIGRATIONS_DIR", tmp_path)

    assert main() == 1
    assert "ERROR:" in capsys.readouterr().err


def _copy_migrations(tmp_path: Path) -> Path:
    migrations_copy = tmp_path / "migrations"
    shutil.copytree(MIGRATIONS_DIR, migrations_copy)
    return migrations_copy


def _replace_in_migrations(migrations_dir: Path, old_text: str, new_text: str) -> None:
    replaced_count = 0
    for migration_path in migrations_dir.glob("*.sql"):
        migration_text = migration_path.read_text()
        replaced_count += migration_text.count(old_text)
        migration_path.write_text(migration_text.replace(old_text, new_text))
    assert replaced_count > 0


@pytest.mark.parametrize(
    ("field_name", "singular_name"),
    [
        ("genres", "genre"),
        ("companies", "company"),
        ("contextual_ranks", "contextual_rank"),
        ("statistics", "statistics"),
        ("streaming", "streaming"),
        ("related_anime", "related_anime"),
    ],
)
def test_make_singular_field_name_returns_table_word(
    field_name: str, singular_name: str
) -> None:
    assert make_singular(field_name) == singular_name


def test_find_naming_problems_repository_migrations_returns_no_problems() -> None:
    assert find_naming_problems(MIGRATIONS_DIR) == []


def test_find_naming_problems_table_not_named_after_field_reports_field(
    tmp_path: Path,
) -> None:
    migrations_copy = _copy_migrations(tmp_path)
    _replace_in_migrations(
        migrations_copy, "anime_external_source", "anime_external_link"
    )

    assert "anime.external_sources: expected table anime_external_source" in (
        find_naming_problems(migrations_copy)
    )


def test_find_naming_problems_column_not_named_after_field_reports_field(
    tmp_path: Path,
) -> None:
    migrations_copy = _copy_migrations(tmp_path)
    _replace_in_migrations(migrations_copy, "  is_nsfw ", "  nsfw ")

    assert "anime.is_nsfw: expected column anime.is_nsfw" in find_naming_problems(
        migrations_copy
    )


def test_find_naming_problems_shared_table_column_counts_for_link_table(
    tmp_path: Path,
) -> None:
    migrations_copy = _copy_migrations(tmp_path)
    _replace_in_migrations(
        migrations_copy,
        "  thumbnail_source  text\n",
        "  thumbnail_url     text\n",
    )

    assert find_naming_problems(migrations_copy) == [
        "anime.trailers.thumbnail_source: expected column anime_trailer.thumbnail_source"
    ]


def test_find_naming_problems_nested_object_columns_follow_field_name(
    tmp_path: Path,
) -> None:
    migrations_copy = _copy_migrations(tmp_path)
    _replace_in_migrations(migrations_copy, "score_median", "median_score")

    assert find_naming_problems(migrations_copy) == [
        "anime.score.median: expected column anime.score_median"
    ]


def test_find_naming_problems_stale_exception_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        check_postgres_schema.FIELD_EXCEPTIONS, "anime.removed_field", "gone"
    )

    assert find_naming_problems(MIGRATIONS_DIR) == [
        "anime.removed_field: exception names no stored field"
    ]


def test_main_matching_schema_reports_both_checks(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main() == 0
    assert "follows the naming rule" in capsys.readouterr().out


def test_main_stale_exception_prints_error_and_returns_one(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setitem(
        check_postgres_schema.FIELD_EXCEPTIONS, "anime.removed_field", "gone"
    )

    assert main() == 1
    assert "ERROR: anime.removed_field" in capsys.readouterr().err
