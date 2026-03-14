"""Unit tests for mal_anime_crawler.py — schema structure and post-processing helpers.

These tests validate the text-anchor + regex extraction pattern using in-memory raw
dicts that mirror the shape of real extraction output. No network calls are made.
"""

from enrichment.crawlers.mal_crawler.mal_anime_crawler import (
    _build_anime_from_raw,
    _get_anime_schema,
)


# =============================================================================
# Schema structure — verifies the benchmark extraction pattern is in place
# =============================================================================


def test_schema_has_fields_list() -> None:
    """Anime schema must have a 'fields' list (XPath strategy requirement)."""
    schema = _get_anime_schema()
    assert "fields" in schema
    assert isinstance(schema["fields"], list)


def test_schema_external_sources_uses_text_anchor() -> None:
    """external_sources_raw must anchor on h2 label text, not a fragile CSS class."""
    schema = _get_anime_schema()
    ext_field = next(f for f in schema["fields"] if f["name"] == "external_sources_raw")
    selector = ext_field["selector"]
    assert "Available At" in selector or "Resources" in selector
    # Old broken class name must not be present
    assert "streaming-platform_links" not in selector


def test_schema_streaming_links_uses_text_anchor() -> None:
    """streaming_links_raw must anchor on 'Streaming Platforms' h2 text."""
    schema = _get_anime_schema()
    stream_field = next(f for f in schema["fields"] if f["name"] == "streaming_links_raw")
    selector = stream_field["selector"]
    assert "Streaming Platforms" in selector
    assert "broadcasts" in selector


def test_schema_external_sources_name_uses_caption_div() -> None:
    """external_sources_raw name sub-field must target the .caption div."""
    schema = _get_anime_schema()
    ext_field = next(f for f in schema["fields"] if f["name"] == "external_sources_raw")
    name_field = next(f for f in ext_field["fields"] if f["name"] == "name")
    assert "caption" in name_field["selector"]


def test_schema_streaming_links_name_uses_title_attribute() -> None:
    """streaming_links_raw name sub-field must read the anchor's title attribute."""
    schema = _get_anime_schema()
    stream_field = next(f for f in schema["fields"] if f["name"] == "streaming_links_raw")
    name_field = next(f for f in stream_field["fields"] if f["name"] == "name")
    assert name_field.get("attribute") == "title"


# =============================================================================
# _build_anime_from_raw — external and streaming link post-processing
# =============================================================================


def _build(extra: dict | None = None):
    """Helper: call _build_anime_from_raw with a minimal raw dict."""
    raw: dict = {"title": "One Piece"}
    if extra:
        raw.update(extra)
    return _build_anime_from_raw(
        raw,
        anime_id=21,
        url="https://myanimelist.net/anime/21/One_Piece",
        picture_urls=[],
    )


def test_build_external_sources_parsed() -> None:
    """external_sources_raw list is converted to MalExternalLink objects."""
    anime = _build({
        "external_sources_raw": [
            {"name": "Official Site", "source": "https://example.com"},
            {"name": "Twitter", "source": "https://twitter.com/onepieceanime"},
        ]
    })
    assert len(anime.external_sources) == 2
    assert anime.external_sources[0].name == "Official Site"
    assert anime.external_sources[1].name == "Twitter"


def test_build_streaming_links_parsed() -> None:
    """streaming_links_raw list is converted to MalExternalLink objects."""
    anime = _build({
        "streaming_links_raw": [
            {"name": "Crunchyroll", "source": "https://crunchyroll.com/one-piece"},
            {"name": "Netflix", "source": "https://netflix.com/title/80107103"},
        ]
    })
    assert len(anime.streaming) == 2
    assert anime.streaming[0].name == "Crunchyroll"
    assert anime.streaming[1].name == "Netflix"


def test_build_empty_link_name_skipped() -> None:
    """Link entries with empty name are skipped."""
    anime = _build({
        "external_sources_raw": [
            {"name": "", "source": "https://example.com"},
            {"name": "Valid Site", "source": "https://valid.com"},
        ]
    })
    assert len(anime.external_sources) == 1
    assert anime.external_sources[0].name == "Valid Site"


def test_build_empty_link_source_skipped() -> None:
    """Link entries with empty source are skipped."""
    anime = _build({
        "streaming_links_raw": [
            {"name": "Crunchyroll", "source": ""},
            {"name": "Netflix", "source": "https://netflix.com"},
        ]
    })
    assert len(anime.streaming) == 1
    assert anime.streaming[0].name == "Netflix"


def test_build_missing_link_fields_returns_empty_lists() -> None:
    """None or missing external/streaming link lists result in empty lists."""
    anime = _build()
    assert anime.external_sources == []
    assert anime.streaming == []


def test_build_both_link_types_populated_independently() -> None:
    """External and streaming links are populated independently."""
    anime = _build({
        "external_sources_raw": [
            {"name": "Official Site", "source": "https://example.com"},
        ],
        "streaming_links_raw": [
            {"name": "Crunchyroll", "source": "https://crunchyroll.com"},
        ],
    })
    assert len(anime.external_sources) == 1
    assert len(anime.streaming) == 1
    assert anime.external_sources[0].name != anime.streaming[0].name
