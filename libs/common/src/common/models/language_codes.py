"""Standard language and country codes for the canonical model's fields.

Every language field of the canonical model holds a BCP 47 language tag, and
every country field an ISO 3166-1 alpha-2 code, so one convention holds from the
mappers to the database, where each code is stored once.
"""

import logging
from typing import Annotated

import langcodes
from pydantic import BeforeValidator

logger = logging.getLogger(__name__)

# BCP 47's tag for a language that cannot be determined.
UNDETERMINED_LANGUAGE = "und"


def standardize_language_code(value: object) -> object:
    """Turn a provider's language tag or English language name into a standard BCP 47 tag.

    - A valid tag is put in its standard form: ``ja_jp`` becomes ``ja-JP``,
      ``jpn`` becomes ``ja`` and the macrolanguage member ``cmn`` becomes ``zh``.
    - An English language name becomes its tag: ``Japanese`` becomes ``ja``.
    - A word that is neither becomes ``und`` and logs a warning, so new provider
      words show up instead of being stored unchecked.

    A value that is not a string is returned unchanged, so the field's own type
    check reports it.

    Args:
        value: The language as the provider wrote it.

    Returns:
        The standard tag, or ``value`` when it is not a string.

    >>> standardize_language_code("ja_jp")
    'ja-JP'
    >>> standardize_language_code("Japanese")
    'ja'
    >>> standardize_language_code("cmn")
    'zh'
    >>> standardize_language_code("jp")
    'und'
    """
    if not isinstance(value, str):
        return value
    language_text = value.strip()
    tag_text = language_text.replace("_", "-")
    if tag_text and langcodes.tag_is_valid(tag_text):
        return langcodes.standardize_tag(tag_text, macro=True)
    if language_text:
        try:
            found_language = langcodes.find(language_text)
        except LookupError:
            pass
        else:
            return langcodes.standardize_tag(str(found_language), macro=True)
    logger.warning(f"No language code matches {value!r}; using {UNDETERMINED_LANGUAGE}")
    return UNDETERMINED_LANGUAGE


def standardize_country_code(value: object) -> object:
    """Turn a provider's country code into an upper-case ISO 3166-1 alpha-2 code.

    - A valid two-letter region code is upper-cased: ``jp`` becomes ``JP``.
    - Any other text becomes None and logs a warning, since an unknown country
      is no country at all.

    A value that is not a string is returned unchanged, so the field's own type
    check reports it.

    Args:
        value: The country code as the provider wrote it.

    Returns:
        The standard code, None for an unknown code, or ``value`` when it is not a string.

    >>> standardize_country_code("jp")
    'JP'
    >>> standardize_country_code("UK") is None
    True
    """
    if not isinstance(value, str):
        return value
    country_text = value.strip().upper()
    if not country_text:
        return None
    if (
        len(country_text) == 2
        and country_text.isalpha()
        and langcodes.Language.make(territory=country_text).is_valid()
    ):
        return country_text
    logger.warning(f"{value!r} is not an ISO 3166-1 alpha-2 country code; dropping it")
    return None


LanguageCode = Annotated[str, BeforeValidator(standardize_language_code)]
# Optional by itself, since an unknown code becomes None
CountryCode = Annotated[str | None, BeforeValidator(standardize_country_code)]
