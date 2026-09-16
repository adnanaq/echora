"""Work identity resolution for relationship consolidation.

Providers each report a related work under their own URL, so two entries for the
same work share nothing a string comparison can use - romaji and English titles
of the same film overlap barely at all. An identity resolver answers the single
question that fixes this:

    "do these two URLs denote the same work?"

It supplies identity only. Relation type, format, title and every other field
still come from the providers; see ``relationship_merger``.

Resolvers are pluggable so the backing store can change without touching the
merge logic. ``resolve`` returns an **opaque string id** rather than anything
store-specific, so a future implementation backed by an owned database can
return its own identifiers and nothing downstream changes.

``None`` means "I do not know this URL", never "no match" - callers fall back to
fuzzy matching on that result, so a sparse resolver degrades gracefully instead
of regressing works it has not learned yet.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

# Providers decorate their URLs differently from the offline database, which
# stores the bare id: MAL appends a title slug as a path segment, AniSearch
# appends ",slug" to the id. The id is the identity; the decoration is not.
#
# Every pattern is anchored and captures the id explicitly. Substring or prefix
# comparison is not safe here - "/anime/466" is a prefix of "/anime/46633", and
# treating those as equal silently fuses two unrelated works.
_ID_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^https?://(?:www\.)?myanimelist\.net/(anime|manga)/(\d+)"), "mal"),
    (
        re.compile(r"^https?://(?:www\.)?anisearch\.com/(anime|manga)/(\d+)"),
        "anisearch",
    ),
    (re.compile(r"^https?://(?:www\.)?anilist\.co/(anime|manga)/(\d+)"), "anilist"),
    (re.compile(r"^https?://(?:www\.)?anidb\.net/(anime)/(\d+)"), "anidb"),
    (
        re.compile(r"^https?://(?:www\.)?kitsu\.(?:app|io)/(anime|manga)/([^/?#]+)"),
        "kitsu",
    ),
    (
        re.compile(r"^https?://(?:www\.)?anime-planet\.com/(anime|manga)/([^/?#]+)"),
        "animeplanet",
    ),
    (re.compile(r"^https?://(?:www\.)?livechart\.me/(anime)/(\d+)"), "livechart"),
    (re.compile(r"^https?://(?:www\.)?simkl\.com/(anime)/(\d+)"), "simkl"),
    (re.compile(r"^https?://(?:www\.)?animecountdown\.com/(\d+)()"), "animecountdown"),
)


def canonical_url_key(url: str) -> str:
    """Reduce a provider URL to the identity it denotes, dropping decoration.

    MAL's title slug and AniSearch's ``,slug`` suffix are presentation only, so
    ``/anime/466/One_Piece__Taose_Kaizoku_Ganzack`` and ``/anime/466`` are the
    same work and must produce the same key.

    Args:
        url: A provider URL.

    Returns:
        A ``provider:kind:id`` key when the URL is recognised, otherwise the
        lowercased URL with ``www.`` and any trailing slash removed.
    """
    for pattern, provider in _ID_PATTERNS:
        match = pattern.match(url)
        if match:
            kind, identifier = match.group(1), match.group(2)
            return f"{provider}:{kind}:{identifier.lower()}"
    return url.lower().replace("://www.", "://").rstrip("/")


@runtime_checkable
class WorkIdentityResolver(Protocol):
    """Maps a provider URL to a stable id for the work it denotes."""

    def resolve(self, url: str) -> str | None:
        """Return the work id for ``url``, or ``None`` when unknown.

        Args:
            url: A provider URL for a single work.

        Returns:
            An opaque, stable identifier shared by every URL denoting the same
            work, or ``None`` if this resolver cannot place the URL.
        """
        ...


class NullIdentityResolver:
    """Resolver that knows nothing, leaving grouping entirely to fuzzy matching.

    This is the default so that enabling identity resolution is an explicit
    choice, and so existing behaviour is unchanged when no resolver is supplied.
    """

    def resolve(self, url: str) -> str | None:
        """Return ``None`` for every URL.

        Args:
            url: Ignored.

        Returns:
            Always ``None``.
        """
        del url
        return None


class OfflineDatabaseResolver:
    """Resolver backed by the manami anime-offline-database.

    Each entry in that database lists one work's URL across every platform it
    appears on, which is precisely the cross-provider equivalence relation the
    merge needs. Note this uses each entry's ``sources`` field; the entry's
    ``relatedAnime`` field is unrelated to identity and is not read.

    Build from already-parsed entries where possible - the enrichment pipeline
    loads this database anyway, and re-reading 84 MB per merge is wasteful.
    """

    def __init__(self, entries: list[dict[str, Any]]) -> None:
        """Index every source URL in the supplied entries.

        Args:
            entries: The ``data`` list of an anime-offline-database document.
        """
        self._by_url: dict[str, str] = {}
        for entry in entries:
            sources = entry.get("sources") or []
            work_id = self._work_id(sources)
            for url in sources:
                self._by_url[canonical_url_key(url)] = work_id

    def __len__(self) -> int:
        """Return how many URLs are indexed."""
        return len(self._by_url)

    @staticmethod
    def _work_id(sources: list[str]) -> str:
        """Derive a release-stable id from an entry's own source URLs.

        The entry's position in the file is deliberately not used: the database
        is republished weekly, and any insertion shifts every later row, so a
        positional id would silently come to mean a different work.

        Args:
            sources: The entry's source URLs.

        Returns:
            An ``aod:<digest>`` identifier derived from the sorted URLs. The
            digest is for identity only, never security.
        """
        canonical = "|".join(sorted(canonical_url_key(url) for url in sources))
        return f"aod:{hashlib.blake2s(canonical.encode(), digest_size=8).hexdigest()}"

    @classmethod
    def from_file(cls, path: str | Path) -> OfflineDatabaseResolver:
        """Load and index an anime-offline-database JSON document.

        Args:
            path: Path to the database file.

        Returns:
            A resolver indexed over that document's entries.
        """
        with Path(path).open(encoding="utf-8") as handle:
            document = json.load(handle)
        return cls(document["data"])

    def resolve(self, url: str) -> str | None:
        """Return the work id for ``url``, or ``None`` when absent.

        Args:
            url: A provider URL for a single work.

        Returns:
            An ``aod:<digest>`` identifier, or ``None`` if the URL is not
            indexed by this database.
        """
        return self._by_url.get(canonical_url_key(url))
