import logging

import pytest
from common.models.language_codes import (
    UNDETERMINED_LANGUAGE,
    standardize_country_code,
    standardize_language_code,
)


@pytest.mark.parametrize(
    ("provider_value", "language_code"),
    [
        ("ja", "ja"),
        ("ja_jp", "ja-JP"),
        ("pt_br", "pt-BR"),
        ("jpn", "ja"),
        ("cmn", "zh"),
        ("iw", "he"),
        ("zh-hant", "zh-Hant"),
        ("x-jat", "x-jat"),
        (" de ", "de"),
        ("und", "und"),
    ],
)
def test_standardize_language_code_tag_returns_standard_tag(
    provider_value: str, language_code: str
) -> None:
    assert standardize_language_code(provider_value) == language_code


@pytest.mark.parametrize(
    ("provider_value", "language_code"),
    [
        ("Japanese", "ja"),
        ("JAPANESE", "ja"),
        ("English", "en"),
        ("Mandarin", "zh"),
        ("Hebrew", "he"),
        ("Latin American Spanish", "es-419"),
    ],
)
def test_standardize_language_code_english_name_returns_tag(
    provider_value: str, language_code: str
) -> None:
    assert standardize_language_code(provider_value) == language_code


@pytest.mark.parametrize("provider_value", ["jp", "Not a language", ""])
def test_standardize_language_code_unknown_word_returns_undetermined_and_logs_warning(
    provider_value: str, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        language_code = standardize_language_code(provider_value)

    assert language_code == UNDETERMINED_LANGUAGE
    assert repr(provider_value) in caplog.text


def test_standardize_language_code_non_string_returns_value_unchanged() -> None:
    assert standardize_language_code(12) == 12


@pytest.mark.parametrize(
    ("provider_value", "country_code"), [("JP", "JP"), ("jp", "JP"), (" kr ", "KR")]
)
def test_standardize_country_code_iso_code_returns_upper_case_code(
    provider_value: str, country_code: str
) -> None:
    assert standardize_country_code(provider_value) == country_code


@pytest.mark.parametrize("provider_value", ["UK", "JPN", "419", "Japan"])
def test_standardize_country_code_unknown_code_returns_none_and_logs_warning(
    provider_value: str, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        country_code = standardize_country_code(provider_value)

    assert country_code is None
    assert repr(provider_value) in caplog.text


def test_standardize_country_code_empty_text_returns_none_without_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        country_code = standardize_country_code("  ")

    assert country_code is None
    assert caplog.text == ""


def test_standardize_country_code_non_string_returns_value_unchanged() -> None:
    assert standardize_country_code(81) == 81
