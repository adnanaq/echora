import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import check_postgres_lookup_tables  # noqa: E402
from check_postgres_lookup_tables import (  # noqa: E402
    ENUM_TABLE_MAP,
    extract_lookup_values,
    find_lookup_mismatches,
    main,
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
    assert "All 9 lookup tables match anime.py enums." in capsys.readouterr().out

    (tmp_path / "001_empty.sql").write_text("", encoding="utf-8")
    monkeypatch.setattr(check_postgres_lookup_tables, "MIGRATIONS_DIR", tmp_path)

    assert main() == 1
    assert "ERROR:" in capsys.readouterr().err
