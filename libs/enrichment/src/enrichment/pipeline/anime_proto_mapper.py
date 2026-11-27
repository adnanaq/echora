"""Conversion from canonical Anime dicts to protobuf messages and field overrides."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, date, datetime, time
from enum import Enum
from typing import Any

from google.protobuf.timestamp_pb2 import Timestamp  # ty: ignore[unresolved-import]
from shared_proto.v1 import anime_pb2

_ENTITY_TYPE_MAP: dict[str, anime_pb2.EntityType] = {
    "ANIME": anime_pb2.ENTITY_TYPE_ANIME,
    "CHARACTER": anime_pb2.ENTITY_TYPE_CHARACTER,
    "EPISODE": anime_pb2.ENTITY_TYPE_EPISODE,
}

_ANIME_STATUS_MAP: dict[str, anime_pb2.AnimeStatus] = {
    "FINISHED": anime_pb2.ANIME_STATUS_FINISHED,
    "UPCOMING": anime_pb2.ANIME_STATUS_UPCOMING,
    "ONGOING": anime_pb2.ANIME_STATUS_ONGOING,
    "UNKNOWN": anime_pb2.ANIME_STATUS_UNKNOWN,
    "CANCELLED": anime_pb2.ANIME_STATUS_CANCELLED,
}

_ANIME_TYPE_MAP: dict[str, anime_pb2.AnimeType] = {
    "MOVIE": anime_pb2.ANIME_TYPE_MOVIE,
    "ONA": anime_pb2.ANIME_TYPE_ONA,
    "OVA": anime_pb2.ANIME_TYPE_OVA,
    "SPECIAL": anime_pb2.ANIME_TYPE_SPECIAL,
    "TV": anime_pb2.ANIME_TYPE_TV,
    "UNKNOWN": anime_pb2.ANIME_TYPE_UNKNOWN,
    "MUSIC": anime_pb2.ANIME_TYPE_MUSIC,
    "PV": anime_pb2.ANIME_TYPE_PV,
    "TV_SPECIAL": anime_pb2.ANIME_TYPE_TV_SPECIAL,
    "TV_SHORT": anime_pb2.ANIME_TYPE_TV_SHORT,
    "CM": anime_pb2.ANIME_TYPE_CM,
    "OTHER": anime_pb2.ANIME_TYPE_OTHER,
}

_ANIME_RATING_MAP: dict[str, anime_pb2.AnimeRating] = {
    "G": anime_pb2.ANIME_RATING_G,
    "PG": anime_pb2.ANIME_RATING_PG,
    "PG13": anime_pb2.ANIME_RATING_PG13,
    "PG-13": anime_pb2.ANIME_RATING_PG13,
    "R": anime_pb2.ANIME_RATING_R,
    "RPLUS": anime_pb2.ANIME_RATING_RPLUS,
    "R+": anime_pb2.ANIME_RATING_RPLUS,
    "RX": anime_pb2.ANIME_RATING_RX,
    "UNKNOWN": anime_pb2.ANIME_RATING_UNKNOWN,
}

_ANIME_SEASON_MAP: dict[str, anime_pb2.AnimeSeason] = {
    "SPRING": anime_pb2.ANIME_SEASON_SPRING,
    "SUMMER": anime_pb2.ANIME_SEASON_SUMMER,
    "FALL": anime_pb2.ANIME_SEASON_FALL,
    "AUTUMN": anime_pb2.ANIME_SEASON_FALL,
    "WINTER": anime_pb2.ANIME_SEASON_WINTER,
}

_SOURCE_MATERIAL_MAP: dict[str, anime_pb2.SourceMaterialType] = {
    "ORIGINAL": anime_pb2.SOURCE_MATERIAL_TYPE_ORIGINAL,
    "MANGA": anime_pb2.SOURCE_MATERIAL_TYPE_MANGA,
    "LIGHT_NOVEL": anime_pb2.SOURCE_MATERIAL_TYPE_LIGHT_NOVEL,
    "VISUAL_NOVEL": anime_pb2.SOURCE_MATERIAL_TYPE_VISUAL_NOVEL,
    "GAME": anime_pb2.SOURCE_MATERIAL_TYPE_GAME,
    "WEB_MANGA": anime_pb2.SOURCE_MATERIAL_TYPE_WEB_MANGA,
    "WEB_NOVEL": anime_pb2.SOURCE_MATERIAL_TYPE_WEB_NOVEL,
    "NOVEL": anime_pb2.SOURCE_MATERIAL_TYPE_NOVEL,
    "ONE_SHOT": anime_pb2.SOURCE_MATERIAL_TYPE_ONE_SHOT,
    "DOUJINSHI": anime_pb2.SOURCE_MATERIAL_TYPE_DOUJINSHI,
    "KOMA_4": anime_pb2.SOURCE_MATERIAL_TYPE_KOMA_4,
    "4-KOMA": anime_pb2.SOURCE_MATERIAL_TYPE_KOMA_4,
    "MIXED_MEDIA": anime_pb2.SOURCE_MATERIAL_TYPE_MIXED_MEDIA,
    "MUSIC": anime_pb2.SOURCE_MATERIAL_TYPE_MUSIC,
    "RADIO": anime_pb2.SOURCE_MATERIAL_TYPE_RADIO,
    "BOOK": anime_pb2.SOURCE_MATERIAL_TYPE_BOOK,
    "PICTURE_BOOK": anime_pb2.SOURCE_MATERIAL_TYPE_PICTURE_BOOK,
    "CARD_GAME": anime_pb2.SOURCE_MATERIAL_TYPE_CARD_GAME,
    "OTHER": anime_pb2.SOURCE_MATERIAL_TYPE_OTHER,
    "UNKNOWN": anime_pb2.SOURCE_MATERIAL_TYPE_UNKNOWN,
    "MANHWA": anime_pb2.SOURCE_MATERIAL_TYPE_MANHWA,
    "MANHUA": anime_pb2.SOURCE_MATERIAL_TYPE_MANHUA,
    "COMIC": anime_pb2.SOURCE_MATERIAL_TYPE_COMIC,
    "LIVE_ACTION": anime_pb2.SOURCE_MATERIAL_TYPE_LIVE_ACTION,
    "ILLUSTRATION": anime_pb2.SOURCE_MATERIAL_TYPE_ILLUSTRATION,
    "WESTERN_MEDIA": anime_pb2.SOURCE_MATERIAL_TYPE_WESTERN_MEDIA,
    "WEBTOON": anime_pb2.SOURCE_MATERIAL_TYPE_WEBTOON,
    "FAIRY_TALE": anime_pb2.SOURCE_MATERIAL_TYPE_FAIRY_TALE,
    "SHORT_STORY": anime_pb2.SOURCE_MATERIAL_TYPE_SHORT_STORY,
}

_COMPANY_ROLE_MAP: dict[str, anime_pb2.CompanyRole] = {
    "STUDIO": anime_pb2.COMPANY_ROLE_STUDIO,
    "PRODUCER": anime_pb2.COMPANY_ROLE_PRODUCER,
    "LICENSOR": anime_pb2.COMPANY_ROLE_LICENSOR,
    "UNKNOWN": anime_pb2.COMPANY_ROLE_UNKNOWN,
}

ANIME_RELATION_TYPE_MAP: dict[str, anime_pb2.AnimeRelationType] = {
    "ADAPTATION": anime_pb2.ANIME_RELATION_TYPE_ADAPTATION,
    "ALTERNATIVE_VERSION": anime_pb2.ANIME_RELATION_TYPE_ALTERNATIVE_VERSION,
    "ALTERNATIVE_SETTING": anime_pb2.ANIME_RELATION_TYPE_ALTERNATIVE_SETTING,
    "CHARACTER": anime_pb2.ANIME_RELATION_TYPE_CHARACTER,
    "CROSSOVER": anime_pb2.ANIME_RELATION_TYPE_CROSSOVER,
    "FULL_STORY": anime_pb2.ANIME_RELATION_TYPE_FULL_STORY,
    "OTHER": anime_pb2.ANIME_RELATION_TYPE_OTHER,
    "PARENT_STORY": anime_pb2.ANIME_RELATION_TYPE_PARENT_STORY,
    "PREQUEL": anime_pb2.ANIME_RELATION_TYPE_PREQUEL,
    "SAME_FRANCHISE": anime_pb2.ANIME_RELATION_TYPE_SAME_FRANCHISE,
    "SAME_SETTING": anime_pb2.ANIME_RELATION_TYPE_SAME_SETTING,
    "SEQUEL": anime_pb2.ANIME_RELATION_TYPE_SEQUEL,
    "SIDE_STORY": anime_pb2.ANIME_RELATION_TYPE_SIDE_STORY,
    "SPIN_OFF": anime_pb2.ANIME_RELATION_TYPE_SPIN_OFF,
    "SUMMARY": anime_pb2.ANIME_RELATION_TYPE_SUMMARY,
}


def _enum_key(value: Any) -> str:
    """Normalize an enum member or string into an upper-case lookup key.

    Args:
        value: Raw enum or string value.

    Returns:
        Upper-case key string.
    """
    raw = value.value if isinstance(value, Enum) else str(value)
    return raw.strip().upper().replace(" ", "_")


def to_proto_timestamp(value: datetime | str | None) -> Timestamp | None:
    """Convert a datetime or ISO-8601 string into a protobuf Timestamp.

    Args:
        value: Datetime object or ISO-8601 timestamp string.

    Returns:
        Populated ``Timestamp``, or ``None`` when input is empty or invalid.
    """
    if value is None or value == "":
        return None
    if isinstance(value, str):
        normalized = value.strip()
        if normalized.endswith("Z"):
            normalized = f"{normalized[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return None
    elif isinstance(value, datetime):
        parsed = value
    else:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    ts = Timestamp()
    ts.FromDatetime(parsed.astimezone(UTC))
    return ts


def _format_time_value(value: time | str | None) -> str | None:
    """Format a time-of-day value as ``HH:MM:SS`` or ``HH:MM``.

    Args:
        value: Time object or string.

    Returns:
        Formatted time string, or ``None`` when unset.
    """
    if value is None or value == "":
        return None
    if isinstance(value, time):
        return value.strftime("%H:%M:%S")
    return str(value)


def _format_date_value(value: date | datetime | str | None) -> str | None:
    """Format a date value as an ISO-8601 calendar date string.

    Args:
        value: Date, datetime, or string.

    Returns:
        ISO date string, or ``None`` when unset.
    """
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _build_aired_dates(raw: dict[str, Any] | None) -> anime_pb2.AiredDates | None:
    """Build an ``AiredDates`` proto message from a canonical aired_dates dict.

    Args:
        raw: Canonical aired_dates dictionary.

    Returns:
        Populated ``AiredDates`` message, or ``None`` when empty.
    """
    if not raw:
        return None
    aired_from = to_proto_timestamp(raw.get("aired_from"))
    aired_to = to_proto_timestamp(raw.get("aired_to"))
    if aired_from is None and aired_to is None:
        return None
    msg = anime_pb2.AiredDates()
    if aired_from is not None:
        msg.aired_from.CopyFrom(aired_from)
    if aired_to is not None:
        msg.aired_to.CopyFrom(aired_to)
    return msg


def _build_broadcast(raw: dict[str, Any] | None) -> anime_pb2.Broadcast | None:
    """Build a ``Broadcast`` proto message from a canonical broadcast dict.

    Args:
        raw: Canonical broadcast dictionary.

    Returns:
        Populated ``Broadcast`` message, or ``None`` when empty.
    """
    if not raw:
        return None
    msg = anime_pb2.Broadcast()
    if raw.get("day") is not None:
        msg.day = int(raw["day"])
    if raw.get("timezone"):
        msg.timezone = str(raw["timezone"])
    for field in ("jp_time", "sub_time", "dub_time"):
        formatted = _format_time_value(raw.get(field))
        if formatted is not None:
            setattr(msg, field, formatted)
    for field in ("sub_delay_days", "dub_delay_days", "next_episode_number"):
        if raw.get(field) is not None:
            setattr(msg, field, int(raw[field]))
    for field in ("premiere_jp", "premiere_sub", "premiere_dub", "next_episode_at"):
        ts = to_proto_timestamp(raw.get(field))
        if ts is not None:
            getattr(msg, field).CopyFrom(ts)
    return msg


def _build_hiatus(raw: dict[str, Any] | None) -> anime_pb2.AnimeHiatus | None:
    """Build an ``AnimeHiatus`` proto message from a canonical hiatus dict.

    Args:
        raw: Canonical hiatus dictionary.

    Returns:
        Populated ``AnimeHiatus`` message, or ``None`` when empty.
    """
    if not raw:
        return None
    msg = anime_pb2.AnimeHiatus()
    if raw.get("reason"):
        msg.reason = str(raw["reason"])
    if (hiatus_from := _format_date_value(raw.get("hiatus_from"))) is not None:
        msg.hiatus_from = hiatus_from
    if (hiatus_until := _format_date_value(raw.get("hiatus_until"))) is not None:
        msg.hiatus_until = hiatus_until
    return msg


def _build_theme_songs(items: Sequence[dict[str, Any]]) -> list[anime_pb2.ThemeSong]:
    """Build sorted ``ThemeSong`` proto messages from canonical theme dicts.

    Args:
        items: Sequence of canonical theme song dictionaries.

    Returns:
        Sorted list of ``ThemeSong`` messages.
    """
    songs: list[anime_pb2.ThemeSong] = []
    for entry in items:
        song = anime_pb2.ThemeSong(title=str(entry.get("title") or ""))
        if entry.get("artist"):
            song.artist = str(entry["artist"])
        for ep_range in entry.get("episodes") or []:
            proto_range = anime_pb2.EpisodeRange(
                range_start=int(ep_range["range_start"])
            )
            if ep_range.get("range_end") is not None:
                proto_range.range_end = int(ep_range["range_end"])
            song.episodes.append(proto_range)
        songs.append(song)
    return songs


def _build_statistics(raw: dict[str, Any]) -> dict[str, anime_pb2.Statistics]:
    """Convert per-provider statistics dicts into protobuf ``Statistics`` maps.

    Args:
        raw: Mapping of provider slug to canonical statistics dictionary.

    Returns:
        Mapping of provider slug to ``anime_pb2.Statistics``.
    """
    mapped: dict[str, anime_pb2.Statistics] = {}
    for provider, stats in raw.items():
        if not isinstance(stats, dict):
            continue
        proto_stats = anime_pb2.Statistics()
        if stats.get("score") is not None:
            proto_stats.score = float(stats["score"])
        for int_field in ("scored_by", "rank", "popularity", "members", "favorites"):
            if stats.get(int_field) is not None:
                setattr(proto_stats, int_field, int(stats[int_field]))
        for rank_entry in stats.get("contextual_ranks") or []:
            ctx = anime_pb2.ContextualRank(
                rank=int(rank_entry["rank"]),
                context=str(rank_entry.get("context") or ""),
            )
            if rank_entry.get("format"):
                ctx.format = str(rank_entry["format"])
            if rank_entry.get("year") is not None:
                ctx.year = int(rank_entry["year"])
            if rank_entry.get("season"):
                ctx.season = str(rank_entry["season"])
            if rank_entry.get("is_all_time") is not None:
                ctx.is_all_time = bool(rank_entry["is_all_time"])
            proto_stats.contextual_ranks.append(ctx)
        mapped[provider] = proto_stats
    return mapped


def _populate_related_anime(
    msg: anime_pb2.Anime,
    related_anime: dict[Any, list[dict[str, Any]]],
) -> None:
    """Populate the ``related_anime`` map on ``Anime``.

    Args:
        msg: Target ``anime_pb2.Anime`` protobuf message.
        related_anime: Canonical ``related_anime`` mapping.
    """
    for rel_key, entries in related_anime.items():
        target_list = msg.related_anime[_enum_key(rel_key)]
        for entry in entries:
            item = anime_pb2.RelatedAnime(
                title=str(entry.get("title") or ""),
                type=_ANIME_TYPE_MAP.get(
                    _enum_key(entry.get("type") or "UNKNOWN"),
                    anime_pb2.ANIME_TYPE_UNKNOWN,
                ),
                sources=sorted(str(s) for s in (entry.get("sources") or []) if s),
                images=sorted(str(i) for i in (entry.get("images") or []) if i),
            )
            if entry.get("status"):
                item.status = _ANIME_STATUS_MAP.get(
                    _enum_key(entry["status"]), anime_pb2.ANIME_STATUS_UNKNOWN
                )
            if entry.get("year") is not None:
                item.year = int(entry["year"])
            if entry.get("score") is not None:
                item.score = float(entry["score"])
            if entry.get("episode_count") is not None:
                item.episode_count = int(entry["episode_count"])
            target_list.items.append(item)


def _populate_related_source_material(
    msg: anime_pb2.Anime,
    related_source_material: dict[Any, list[dict[str, Any]]],
) -> None:
    """Populate the ``related_source_material`` map on ``Anime``.

    Args:
        msg: Target ``anime_pb2.Anime`` protobuf message.
        related_source_material: Canonical ``related_source_material`` mapping.
    """
    for rel_key, entries in related_source_material.items():
        target_list = msg.related_source_material[_enum_key(rel_key)]
        for entry in entries:
            item = anime_pb2.RelatedSourceMaterial(
                title=str(entry.get("title") or ""),
                type=_SOURCE_MATERIAL_MAP.get(
                    _enum_key(entry.get("type") or "UNKNOWN"),
                    anime_pb2.SOURCE_MATERIAL_TYPE_UNKNOWN,
                ),
                sources=sorted(str(s) for s in (entry.get("sources") or []) if s),
                images=sorted(str(i) for i in (entry.get("images") or []) if i),
            )
            if entry.get("status"):
                item.status = _ANIME_STATUS_MAP.get(
                    _enum_key(entry["status"]), anime_pb2.ANIME_STATUS_UNKNOWN
                )
            if entry.get("score") is not None:
                item.score = float(entry["score"])
            if entry.get("chapters") is not None:
                item.chapters = int(entry["chapters"])
            if entry.get("volumes") is not None:
                item.volumes = int(entry["volumes"])
            target_list.items.append(item)


def _populate_media_and_links(msg: anime_pb2.Anime, record: dict[str, Any]) -> None:
    """Populate themes, trailers, external links, streaming, and companies.

    Args:
        msg: Target ``anime_pb2.Anime`` protobuf message.
        record: Canonical merged Anime dictionary.
    """
    msg.opening_themes.extend(_build_theme_songs(record.get("opening_themes") or []))
    msg.ending_themes.extend(_build_theme_songs(record.get("ending_themes") or []))
    for theme in record.get("themes") or []:
        entry = anime_pb2.ThemeEntry(name=str(theme.get("name") or ""))
        if theme.get("description"):
            entry.description = str(theme["description"])
        msg.themes.append(entry)
    for trailer in record.get("trailers") or []:
        entry = anime_pb2.TrailerEntry()
        for field in ("source", "title", "thumbnail_source"):
            if trailer.get(field):
                setattr(entry, field, str(trailer[field]))
        msg.trailers.append(entry)
    for ext in record.get("external_sources") or []:
        entry = anime_pb2.ExternalLink(
            platform=str(ext.get("platform") or ""),
            source=str(ext.get("source") or ""),
        )
        for field in ("label", "language"):
            if ext.get(field):
                setattr(entry, field, str(ext[field]))
        msg.external_sources.append(entry)
    for stream in record.get("streaming_sources") or []:
        entry = anime_pb2.StreamingEntry(
            platform=str(stream.get("platform") or ""),
            source=str(stream.get("source") or ""),
            subtitle_languages=sorted(
                str(lang) for lang in (stream.get("subtitle_languages") or []) if lang
            ),
        )
        if stream.get("region"):
            entry.region = str(stream["region"])
        for bool_field in ("is_free", "is_premium_required", "has_dub"):
            if stream.get(bool_field) is not None:
                setattr(entry, bool_field, bool(stream[bool_field]))
        msg.streaming_sources.append(entry)


def _populate_companies_and_nested(
    msg: anime_pb2.Anime, record: dict[str, Any]
) -> None:
    """Populate companies, images, schedule, score, titles, and statistics.

    Args:
        msg: Target ``anime_pb2.Anime`` protobuf message.
        record: Canonical merged Anime dictionary.
    """
    for comp in record.get("companies") or []:
        entry = anime_pb2.CompanyEntry(
            name=str(comp.get("name") or ""),
            sources=sorted(str(s) for s in (comp.get("sources") or []) if s),
            roles=sorted(
                _COMPANY_ROLE_MAP.get(_enum_key(role), anime_pb2.COMPANY_ROLE_UNKNOWN)
                for role in (comp.get("roles") or [])
            ),
        )
        if comp.get("description"):
            entry.description = str(comp["description"])
        msg.companies.append(entry)
    if images := record.get("images"):
        msg.images.covers.extend(str(u) for u in (images.get("covers") or []) if u)
        msg.images.posters.extend(str(u) for u in (images.get("posters") or []) if u)
        msg.images.banners.extend(str(u) for u in (images.get("banners") or []) if u)
    if aired_dates := _build_aired_dates(record.get("aired_dates")):
        msg.aired_dates.CopyFrom(aired_dates)
    if broadcast := _build_broadcast(record.get("broadcast")):
        msg.broadcast.CopyFrom(broadcast)
    if hiatus := _build_hiatus(record.get("hiatus")):
        msg.hiatus.CopyFrom(hiatus)
    if score := record.get("score"):
        for field in ("weighted", "mean", "median"):
            if score.get(field) is not None:
                setattr(msg.score, field, float(score[field]))
    for lang, title_val in (record.get("titles") or {}).items():
        if lang and title_val:
            msg.titles[str(lang)] = str(title_val)
    for prov, stats_msg in _build_statistics(record.get("statistics") or {}).items():
        msg.statistics[prov].CopyFrom(stats_msg)


def _populate_scalar_and_list_fields(
    msg: anime_pb2.Anime,
    record: dict[str, Any],
) -> None:
    """Populate optional scalar, enum, and repeated string fields on ``Anime``.

    Args:
        msg: Target ``anime_pb2.Anime`` protobuf message.
        record: Canonical merged Anime dictionary.
    """
    for str_field in (
        "background",
        "country_of_origin",
        "synopsis",
        "title_english",
        "title_japanese",
    ):
        if record.get(str_field):
            setattr(msg, str_field, str(record[str_field]))
    for int_field in ("duration_seconds", "month", "year"):
        if record.get(int_field) is not None:
            setattr(msg, int_field, int(record[int_field]))
    if record.get("is_nsfw") is not None:
        msg.is_nsfw = bool(record["is_nsfw"])
    if record.get("rating"):
        msg.rating = _ANIME_RATING_MAP.get(
            _enum_key(record["rating"]), anime_pb2.ANIME_RATING_UNKNOWN
        )
    if (
        record.get("season")
        and (season_val := _ANIME_SEASON_MAP.get(_enum_key(record["season"])))
        is not None
    ):
        msg.season = season_val
    if record.get("source_material"):
        msg.source_material = _SOURCE_MATERIAL_MAP.get(
            _enum_key(record["source_material"]),
            anime_pb2.SOURCE_MATERIAL_TYPE_UNKNOWN,
        )
    for list_field in (
        "content_warnings",
        "demographics",
        "genres",
        "sources",
        "partial_sources",
        "synonyms",
        "tags",
    ):
        getattr(msg, list_field).extend(
            str(item) for item in (record.get(list_field) or []) if item
        )


def build_anime_proto(record: dict[str, Any]) -> anime_pb2.Anime:
    """Convert a canonical merged Anime dictionary into a protobuf ``Anime`` message.

    Args:
        record: Canonical merged Anime dictionary.

    Returns:
        Populated ``shared_proto.v1.anime_pb2.Anime`` message.
    """
    msg = anime_pb2.Anime(
        id=str(record.get("id") or ""),
        entity_type=_ENTITY_TYPE_MAP.get(
            _enum_key(record.get("entity_type") or "ANIME"),
            anime_pb2.ENTITY_TYPE_ANIME,
        ),
        title=str(record.get("title") or ""),
        episode_count=int(record.get("episode_count") or 0),
        status=_ANIME_STATUS_MAP.get(
            _enum_key(record.get("status") or "UNKNOWN"),
            anime_pb2.ANIME_STATUS_UNKNOWN,
        ),
        type=_ANIME_TYPE_MAP.get(
            _enum_key(record.get("type") or "UNKNOWN"),
            anime_pb2.ANIME_TYPE_UNKNOWN,
        ),
    )
    _populate_scalar_and_list_fields(msg, record)
    _populate_media_and_links(msg, record)
    _populate_companies_and_nested(msg, record)
    _populate_related_anime(msg, record.get("related_anime") or {})
    _populate_related_source_material(msg, record.get("related_source_material") or {})
    sort_anime_proto_lists(msg)
    return msg


def sort_anime_proto_lists(anime: anime_pb2.Anime) -> None:
    """Sort every repeated field on ``Anime`` for deterministic change hashing.

    Args:
        anime: Target ``anime_pb2.Anime`` message mutated in place.
    """
    for scalar_list in (
        "content_warnings",
        "demographics",
        "genres",
        "sources",
        "partial_sources",
        "synonyms",
        "tags",
    ):
        items = sorted(getattr(anime, scalar_list))
        del getattr(anime, scalar_list)[:]
        getattr(anime, scalar_list).extend(items)

    anime.images.covers[:] = sorted(anime.images.covers)
    anime.images.posters[:] = sorted(anime.images.posters)
    anime.images.banners[:] = sorted(anime.images.banners)

    anime.opening_themes.sort(key=lambda t: (t.title, t.artist))
    anime.ending_themes.sort(key=lambda t: (t.title, t.artist))
    anime.themes.sort(key=lambda t: (t.name, t.description))
    anime.trailers.sort(key=lambda t: (t.source, t.title))
    anime.external_sources.sort(key=lambda e: (e.platform, e.source, e.label))
    anime.streaming_sources.sort(key=lambda s: (s.platform, s.source, s.region))
    anime.companies.sort(key=lambda c: c.name)
    for rel_list in anime.related_anime.values():
        rel_list.items.sort(key=lambda r: (r.title, tuple(r.sources)))
    for rel_list in anime.related_source_material.values():
        rel_list.items.sort(key=lambda r: (r.title, tuple(r.sources)))
