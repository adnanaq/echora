"""Kitsu → canonical model mapper.

Pure value normalization functions. No I/O, no side effects.

Maps validated Kitsu models to canonical Anime / Character / Episode field dicts.
"""

from collections.abc import Sequence
from typing import Any

from common.models.anime import (
    AiredDates,
    Anime,
    AnimeImages,
    AnimeStatus,
    AnimeType,
    Broadcast,
    Character,
    CharacterRole,
    CompanyEntry,
    Episode,
    ExternalLink,
    Ography,
    Statistics,
    ThemeEntry,
    TrailerEntry,
    VoiceActor,
)
from common.utils.datetime_utils import (
    determine_anime_season,
    determine_anime_year,
    normalize_to_utc,
)
from enrichment.sources.base.companies import companies_from_roles
from enrichment.sources.base.external_links import page_link
from enrichment.sources.base.page_fetcher import (
    collect_paginated_responses,
    index_responses_by_name,
    parse_json_response,
    require_json_object,
)
from enrichment.sources.kitsu.kitsu_models import (
    KitsuAnime,
    KitsuCategory,
    KitsuEpisode,
    KitsuGenre,
    KitsuMapping,
    KitsuMediaCharacter,
    KitsuProduction,
)
from enrichment.utils.text_utils import normalize_score
from shared_proto.v1.merge_pb2 import PageResponse


def _kitsu_page(media_type: str, kitsu_id: str) -> str:
    """Return Kitsu's numeric page address, the form the offline seed uses.

    Kitsu serves an anime or manga page under its number as well as its slug,
    so the number keeps one key per page across the seed, the mapper and the
    work matching library.

    >>> _kitsu_page("anime", "12")
    'https://kitsu.app/anime/12'
    """
    return f"https://kitsu.app/{media_type}/{kitsu_id}"


def anime_from_kitsu(anime: KitsuAnime) -> dict[str, Any]:
    """Normalize a KitsuAnime into canonical Anime field values.

    ``anime.genres`` and ``anime.themes`` must be populated by the helper
    before calling this function (fetched from ``/genres`` and ``/categories``).

    Args:
        anime: Validated Kitsu anime model with genres and themes attached.

    Returns:
        Dict of canonical field name → normalized value.
    """
    attrs = anime.attributes
    titles = attrs.titles

    title = attrs.canonicalTitle or titles.en_jp or titles.en or "Unknown"

    season = determine_anime_season(attrs.startDate)
    year = determine_anime_year(attrs.startDate)
    start_dt = normalize_to_utc(attrs.startDate)
    end_dt = normalize_to_utc(attrs.endDate)

    poster = attrs.posterImage
    cover = attrs.coverImage
    images = AnimeImages(
        posters=[poster.original] if poster and poster.original else [],
        covers=[cover.original] if cover and cover.original else [],
    )

    kitsu_score = (
        normalize_score(attrs.averageRating)
        if attrs.averageRating is not None
        else None
    )
    statistics: dict[str, Statistics] = {
        "kitsu": Statistics(
            score=kitsu_score,
            scored_by=attrs.scored_by,
            members=attrs.userCount,
            favorites=attrs.favoritesCount,
            popularity=attrs.popularityRank,
            rank=attrs.ratingRank,
        )
    }

    broadcast: Broadcast | None = None
    if attrs.nextRelease:
        broadcast = Broadcast(next_episode_at=normalize_to_utc(attrs.nextRelease))

    trailers: list[TrailerEntry] = []
    if attrs.youtubeVideoId:
        trailers.append(
            TrailerEntry(
                source=f"https://www.youtube.com/watch?v={attrs.youtubeVideoId}"
            )
        )

    aired_dates: AiredDates | None = None
    if start_dt or end_dt:
        aired_dates = AiredDates(aired_from=start_dt, aired_to=end_dt)

    rating = attrs.content_rating

    by_role: dict[str, list[CompanyEntry]] = {
        "studio": [],
        "producer": [],
        "licensor": [],
    }
    for production in anime.companies:
        credited = by_role.get(production.role or "")
        if credited is not None:
            credited.append(
                CompanyEntry(
                    name=production.name,
                    sources=(
                        [
                            f"https://kitsu.app/api/edge/producers/{production.company_id}"
                        ]
                        if production.company_id
                        else []
                    ),
                )
            )

    result = Anime(
        title=title,
        title_english=titles.en,
        title_japanese=titles.ja_jp,
        synopsis=attrs.synopsis,
        type=AnimeType(attrs.subtype),
        status=AnimeStatus(attrs.status),
        episode_count=attrs.episodeCount or 0,
        duration_seconds=attrs.episodeLength * 60 if attrs.episodeLength else None,
        year=year,
        season=season,
        is_nsfw=attrs.nsfw,
        rating=rating,
        genres=anime.genres,
        themes=anime.themes,
        synonyms=attrs.abbreviatedTitles,
        sources=[_kitsu_page("anime", anime.id)],
        external_sources=_external_sources_from_mappings(anime.mappings),
        images=images,
        statistics=statistics,
        broadcast=broadcast,
        trailers=trailers,
        aired_dates=aired_dates,
        companies=companies_from_roles(
            studios=by_role["studio"],
            producers=by_role["producer"],
            licensors=by_role["licensor"],
        ),
    )
    return result.model_dump(mode="json", exclude_none=True)


def _external_sources_from_mappings(mappings: list[KitsuMapping]) -> list[ExternalLink]:
    """Turn Kitsu's mappings into links to this anime's page on each site.

    Args:
        mappings: The mappings Kitsu holds for the anime.

    Returns:
        One link per distinct address, in Kitsu's order.

    Examples:
        >>> def mapping(site, external_id):
        ...     attributes = {"externalSite": site, "externalId": external_id}
        ...     return KitsuMapping.model_validate({"id": site, "attributes": attributes})
        >>> links = _external_sources_from_mappings(
        ...     [mapping("myanimelist/anime", "21"), mapping("thetvdb", "79099/3")]
        ... )
        >>> [(link.platform, link.source) for link in links]
        [('myanimelist', 'https://myanimelist.net/anime/21'), ('thetvdb', 'https://thetvdb.com/dereferrer/series/79099')]
    """
    links: dict[str, ExternalLink] = {}
    for mapping in mappings:
        platform, _, kind = (mapping.attributes.external_site or "").partition("/")
        external_id = mapping.attributes.external_id or ""
        if (platform, kind) == ("thetvdb", ""):
            # The id is "series/season number"; only the series has an address.
            external_id, kind = external_id.partition("/")[0], "series"
        if (
            link := page_link(platform, external_id, kind=kind)
        ) and link.source not in links:
            links[link.source] = link
    return list(links.values())


def character_from_kitsu(char: KitsuMediaCharacter) -> dict[str, Any]:
    """Normalize a KitsuMediaCharacter into canonical Character field values.

    ``char.character``, ``char.voices``, and ``char.animeography``
    must all be populated by the helper before calling this function.

    Args:
        char: Validated media character model with all sub-resources attached.

    Returns:
        Dict of canonical field name → normalized value.

    Raises:
        ValueError: If ``char.character`` is not resolved.
    """
    character = char.character
    if not character:
        raise ValueError(f"char {char.id} has no resolved character")  # noqa: TRY003

    attrs = character.attributes
    names = attrs.names

    sources: list[str] = (
        [f"https://kitsu.io/characters/{attrs.slug}"] if attrs.slug else []
    )
    if attrs.malId:
        sources.append(f"https://myanimelist.net/character/{attrs.malId}")

    voice_actors: list[VoiceActor] = []
    for voice in char.voices:
        person = voice.person
        if not person:
            continue
        p_attrs = person.attributes
        voice_actors.append(
            VoiceActor(
                name=p_attrs.name or "Unknown",
                language=voice.attributes.locale,
                image=p_attrs.image.original if p_attrs.image else None,
                biography=p_attrs.description,
                sources=[f"https://kitsu.app/people/{person.id}"],
            )
        )

    anime_entries: list[Ography] = []
    manga_entries: list[Ography] = []
    for entry in char.animeography:
        media = entry.media
        if not media:
            continue
        m_attrs = media.attributes
        role = CharacterRole(entry.attributes.role)
        ography = Ography(
            title=m_attrs.canonicalTitle or "Unknown",
            role=role,
            sources=[_kitsu_page(entry.media_type, media.id)]
            if entry.media_type
            else [],
        )
        if entry.media_type == "manga":
            manga_entries.append(ography)
        else:
            anime_entries.append(ography)

    result = Character(
        name=attrs.canonicalName or attrs.name or "Unknown",
        name_native=names.ja_jp,
        name_variations=attrs.otherNames,
        description=attrs.description,
        images=[attrs.image.original] if attrs.image and attrs.image.original else [],
        roles=[CharacterRole(char.attributes.role)],
        sources=sources,
        voice_actors=voice_actors,
        animeography=anime_entries,
        mangaography=manga_entries,
    )
    return result.model_dump(mode="json", exclude_none=True)


def episode_from_kitsu(
    ep: KitsuEpisode, anime_slug: str | None = None
) -> dict[str, Any]:
    """Normalize a KitsuEpisode into canonical Episode field values.

    Args:
        ep: Validated Kitsu episode model.
        anime_slug: Kitsu anime slug (e.g. ``"DAN-DA-DAN"``). When provided, the
            source URL is ``https://kitsu.app/anime/{slug}/episodes/{number}``;
            without it the episode has no source URL.

    Returns:
        Dict of canonical field name → normalized value.
    """
    attrs = ep.attributes
    titles = attrs.titles

    title = (
        attrs.canonicalTitle
        or titles.en_us
        or titles.en_jp
        or f"Episode {attrs.number or '?'}"
    )

    if anime_slug and attrs.number is not None:
        source = f"https://kitsu.app/anime/{anime_slug}/episodes/{attrs.number}"
    else:
        source = None

    result = Episode(
        title=title,
        title_japanese=titles.ja_jp,
        title_romaji=titles.en_jp,
        episode_number=attrs.number or 0,
        season_number=attrs.seasonNumber,
        synopsis=attrs.synopsis or attrs.description,
        aired_at=normalize_to_utc(attrs.airdate),
        duration_seconds=attrs.length * 60 if attrs.length else None,
        images=(
            [attrs.thumbnail.original]
            if attrs.thumbnail and attrs.thumbnail.original
            else []
        ),
        sources=[source] if source else [],
    )
    return result.model_dump(mode="json", exclude_none=True, exclude_unset=True)


def _extract_kitsu_productions(
    pages: Sequence[dict[str, Any]],
) -> list[KitsuProduction]:
    """Assemble deduplicated KitsuProduction entries from paginated JSON:API payloads.

    Args:
        pages: Decoded JSON:API payloads from ``productions_page_*`` responses.

    Returns:
        List of deduplicated KitsuProduction entries.
    """
    credited: dict[tuple[str | None, str | None], KitsuProduction] = {}
    for page in pages:
        company_names = {
            company["id"]: (company.get("attributes") or {}).get("name")
            for company in page.get("included", [])
            if company.get("type") == "producers"
        }
        for join_row in page.get("data", []):
            company_id = (
                (join_row.get("relationships") or {})
                .get("producer", {})
                .get("data", {})
                .get("id")
            )
            name = company_names.get(company_id)
            if not name:
                continue
            role = (join_row.get("attributes") or {}).get("role")
            credited.setdefault(
                (company_id, role),
                KitsuProduction(name=name, role=role, company_id=company_id),
            )
    return list(credited.values())


def map_kitsu_anime_responses(
    responses: Sequence[PageResponse],
) -> dict[str, Any]:
    """Map stored Kitsu anime page responses into a canonical Anime dict.

    Args:
        responses: Stored PageResponse sequence containing a ``main`` response.

    Returns:
        Dict of canonical Anime field values.

    Raises:
        ValueError: If the ``main`` response or its ``data`` resource is missing.
    """
    by_name = index_responses_by_name(responses)
    main_response = by_name.get("main")
    if main_response is None:
        error_message = "Kitsu anime responses missing required 'main' response"
        raise ValueError(error_message)
    main_payload = parse_json_response(main_response)
    anime_resource = main_payload.get("data")
    if not isinstance(anime_resource, dict):
        error_message = "Kitsu 'main' response missing 'data' object"
        raise TypeError(error_message)

    anime_model = KitsuAnime.model_validate(anime_resource)
    anime_model.genres = [
        name
        for page in collect_paginated_responses(responses, "genres_page_")
        for genre in page.get("data", [])
        if (name := KitsuGenre.model_validate(genre).attributes.name) is not None
    ]
    anime_model.themes = [
        ThemeEntry(
            name=category.attributes.title,
            description=category.attributes.description or None,
        )
        for page in collect_paginated_responses(responses, "categories_page_")
        for category in map(KitsuCategory.model_validate, page.get("data", []))
        if category.attributes.title
    ]
    anime_model.companies = _extract_kitsu_productions(
        collect_paginated_responses(responses, "productions_page_")
    )
    anime_model.mappings = [
        KitsuMapping.model_validate(mapping)
        for page in collect_paginated_responses(responses, "mappings_page_")
        for mapping in page.get("data", [])
    ]
    return anime_from_kitsu(anime_model)


def map_kitsu_anime_episode_responses(
    responses: Sequence[PageResponse],
) -> list[tuple[str, dict[str, Any]]]:
    """Map stored Kitsu anime episode responses into Kitsu IDs and Episode dicts.

    A Kitsu episode page is named by its Kitsu episode ID, which its URL does not
    hold, so each listed episode comes with that ID.

    Args:
        responses: Stored PageResponse sequence containing ``main`` and ``episodes_page_*``.

    Returns:
        One ``(kitsu_episode_id, canonical Episode dict)`` pair per listed episode.
    """
    by_name = index_responses_by_name(responses)
    anime_slug: str | None = None
    if main_response := by_name.get("main"):
        anime_data = parse_json_response(main_response).get("data")
        if isinstance(anime_data, dict):
            anime_slug = (anime_data.get("attributes") or {}).get("slug")

    listed_episodes = [
        KitsuEpisode.model_validate(ep_raw)
        for page in collect_paginated_responses(responses, "episodes_page_")
        for ep_raw in page.get("data", [])
    ]
    return [
        (episode.id, episode_from_kitsu(episode, anime_slug=anime_slug))
        for episode in listed_episodes
    ]


def map_kitsu_episode_responses(
    responses: Sequence[PageResponse],
) -> dict[str, Any]:
    """Map stored Kitsu single-episode JSON:API responses into a canonical Episode dict.

    The ``main`` response is ``/episodes/{id}?include=media``, so the work it belongs
    to is included and its slug gives the episode's URL.

    Args:
        responses: Stored PageResponse sequence containing a ``main`` JSON response.

    Returns:
        Dict of canonical Episode field values.

    Raises:
        ValueError: If the ``main`` response or its ``data`` payload is missing.
    """
    by_name = index_responses_by_name(responses)
    main_response = by_name.get("main")
    if main_response is None:
        error_message = "Kitsu episode responses missing required 'main' response"
        raise ValueError(error_message)
    payload = parse_json_response(main_response)
    ep_raw = payload.get("data")
    if not isinstance(ep_raw, dict):
        error_message = "Kitsu episode 'main' response missing 'data' object"
        raise TypeError(error_message)
    anime_slug = next(
        (
            require_json_object(
                included.get("attributes"), "Kitsu included 'attributes'"
            ).get("slug")
            for included in payload.get("included") or []
            if isinstance(included, dict) and included.get("type") == "anime"
        ),
        None,
    )
    return episode_from_kitsu(
        KitsuEpisode.model_validate(ep_raw), anime_slug=anime_slug
    )


def _extract_kitsu_character_ography(
    responses: Sequence[PageResponse],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract animeography and mangaography from Kitsu media-characters pages.

    Args:
        responses: Stored PageResponse sequence containing ``media_characters_page_*``.

    Returns:
        Tuple of ``(animeography, mangaography)`` normalized dicts.
    """
    animeography: list[dict[str, Any]] = []
    mangaography: list[dict[str, Any]] = []
    for mc_page in collect_paginated_responses(responses, "media_characters_page_"):
        included_media = {
            (item.get("type"), str(item.get("id"))): item
            for item in mc_page.get("included") or []
        }
        for item in mc_page.get("data") or []:
            rel_media = ((item.get("relationships") or {}).get("media") or {}).get(
                "data"
            ) or {}
            media_type = str(rel_media.get("type") or "anime")
            media_id = str(rel_media.get("id") or "")
            media_obj = included_media.get((media_type, media_id)) or {}
            media_title = (media_obj.get("attributes") or {}).get(
                "canonicalTitle"
            ) or ""
            if not media_id or not media_title:
                continue
            role = CharacterRole(
                str((item.get("attributes") or {}).get("role") or "unknown")
            )
            entry = Ography(
                title=media_title,
                role=role,
                sources=[_kitsu_page(media_type, media_id)],
            ).model_dump(mode="json", exclude_none=True)
            (mangaography if media_type == "manga" else animeography).append(entry)
    return animeography, mangaography


def _extract_kitsu_character_voices(
    responses: Sequence[PageResponse],
) -> list[dict[str, Any]]:
    """Extract voice actor entries from Kitsu ``voices_*`` responses.

    Args:
        responses: Stored PageResponse sequence containing ``voices_*`` responses.

    Returns:
        List of normalized ``VoiceActor`` dicts.
    """
    voice_actors: list[dict[str, Any]] = []
    for resp in responses:
        if not resp.request_name.startswith("voices_"):
            continue
        voice_payload = parse_json_response(resp)
        people = {
            str(item.get("id")): item
            for item in voice_payload.get("included") or []
            if item.get("type") == "people"
        }
        for casting in voice_payload.get("data") or []:
            person_ref = ((casting.get("relationships") or {}).get("person") or {}).get(
                "data"
            ) or {}
            person_id = str(person_ref.get("id") or "")
            person = people.get(person_id) or {}
            person_name = (person.get("attributes") or {}).get("name")
            if person_name:
                voice_actors.append(
                    VoiceActor(
                        name=person_name,
                        language=(casting.get("attributes") or {}).get("locale"),
                        sources=[f"https://kitsu.app/people/{person_id}"],
                    ).model_dump(mode="json", exclude_none=True)
                )
    return voice_actors


def map_kitsu_character_page_responses(
    responses: Sequence[PageResponse],
) -> dict[str, Any]:
    """Map stored Kitsu character page responses into a canonical Character dict.

    Args:
        responses: Stored PageResponse sequence containing ``main``,
            ``media_characters_page_*``, and optional ``voices_*_page_*`` items.

    Returns:
        Dict of canonical Character field values.

    Raises:
        ValueError: If the ``main`` response is missing.
        TypeError: If ``main`` does not contain a ``data`` object.
    """
    by_name = index_responses_by_name(responses)
    main_response = by_name.get("main")
    if main_response is None:
        error_message = "Kitsu character responses missing required 'main' response"
        raise ValueError(error_message)
    payload = parse_json_response(main_response)
    char_raw = payload.get("data")
    if not isinstance(char_raw, dict):
        error_message = "Kitsu character 'main' response missing 'data' object"
        raise TypeError(error_message)

    attrs = require_json_object(
        char_raw.get("attributes"), "Kitsu character 'attributes'"
    )
    names = require_json_object(attrs.get("names"), "Kitsu character 'names'")
    slug = attrs.get("slug") or char_raw.get("id")
    sources = [f"https://kitsu.app/characters/{slug}"] if slug else []
    image_obj = require_json_object(attrs.get("image"), "Kitsu character 'image'")
    images = [image_obj["original"]] if image_obj.get("original") else []
    animeography, mangaography = _extract_kitsu_character_ography(responses)

    char_model = Character(
        name=attrs.get("canonicalName") or attrs.get("name") or "",
        name_native=names.get("ja_jp"),
        name_variations=[str(v) for v in (attrs.get("otherNames") or []) if v],
        description=attrs.get("description"),
        images=images,
        sources=sources,
        animeography=animeography,
        mangaography=mangaography,
        voice_actors=_extract_kitsu_character_voices(responses),
    )
    return char_model.model_dump(mode="json", exclude_none=True)
