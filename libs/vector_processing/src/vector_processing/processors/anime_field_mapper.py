"""Extract and map anime domain data into semantic text.

This module provides the AnimeFieldMapper class, which converts raw model data
(Anime, Character, Episode) into optimized text strings suitable for vector
embedding. The mapper serves as the content strategist in the vector processing
pipeline, determining which fields are important for semantic search.
"""

import logging

from common.models.anime import Anime, Character, CompanyRole, Episode
from common.utils.datetime_utils import ENGLISH_MONTH_NAMES

logger = logging.getLogger(__name__)


class AnimeFieldMapper:
    """Data extractor and formatter for the semantic vector architecture.

    This class is responsible for:
        - Aggregating multiple fields into single semantic strings.
        - Applying domain-specific text normalization (e.g., expanding abbreviations).
        - Extracting image URLs for separate visual embedding points.

    The mapper knows WHAT information is important but delegates HOW to encode
    it to the TextProcessor and VisionProcessor classes.
    """

    def __init__(self) -> None:
        """Initialize the AnimeFieldMapper.

        Sets up the logger for tracking field extraction operations.
        """
        self.logger = logger

    def extract_anime_text(self, anime: Anime) -> str:
        """Extract comprehensive text content for an Anime Point.

        Aggregates semantic data from all available fields to ensure the
        anime point is discoverable via any metadata. Includes title, synopsis,
        genres, staff, and temporal information.

        Args:
            anime: The Anime model instance to extract text from.

        Returns:
            A formatted string containing all relevant text fields for embedding.
        """
        sections = [
            self._extract_anime_core_section(anime),
            self._extract_anime_genre_section(anime),
            self._extract_anime_metadata_section(anime),
            self._extract_anime_production_section(anime),
            *self._extract_anime_temporal_and_streaming_sections(anime),
        ]
        return " || ".join(filter(None, sections))

    def _extract_anime_core_section(self, anime: Anime) -> str:
        """Extract core title, synopsis, and background text for an anime.

        Args:
            anime: Anime model instance to extract core text from.

        Returns:
            Pipe-delimited core identity and story fields.
        """
        title_parts = [
            f"Title: {anime.title}" if anime.title else "",
            f"English: {anime.title_english}" if anime.title_english else "",
            f"Japanese: {anime.title_japanese}" if anime.title_japanese else "",
            f"Synonyms: {', '.join(anime.synonyms)}" if anime.synonyms else "",
            f"Story: {anime.synopsis}" if anime.synopsis else "",
            f"Production Background: {anime.background}" if anime.background else "",
        ]
        return " | ".join(filter(None, title_parts))

    def _extract_anime_genre_section(self, anime: Anime) -> str:
        """Extract genres, demographics, tags, and themes for an anime.

        Args:
            anime: Anime model instance to extract classification text from.

        Returns:
            Pipe-delimited classification fields with expanded demographic terms.
        """
        if not (anime.genres or anime.demographics or anime.tags or anime.themes):
            return ""
        genre_parts = []
        if anime.genres:
            genres = ", ".join(anime.genres)
            # Expand demographic terms for broader search coverage
            genres = genres.replace("Shounen", "Shonen (young male)")
            genres = genres.replace("Shoujo", "Shojo (young female)")
            genres = genres.replace("Seinen", "Seinen (adult male)")
            genres = genres.replace("Josei", "Josei (adult female)")
            genre_parts.append(f"Genres: {genres}")
        if anime.demographics:
            genre_parts.append(f"Demographics: {', '.join(anime.demographics)}")
        if anime.tags:
            genre_parts.append(f"Tags: {', '.join(anime.tags)}")
        theme_names = [t.name for t in anime.themes if hasattr(t, "name") and t.name]
        if theme_names:
            genre_parts.append(f"Themes: {', '.join(theme_names)}")
        return " | ".join(genre_parts)

    def _extract_anime_metadata_section(self, anime: Anime) -> str:
        """Extract format, status, year, and season metadata for an anime.

        Args:
            anime: Anime model instance to extract release metadata from.

        Returns:
            Pipe-delimited format and release metadata.
        """
        meta_parts = []
        if anime.type and anime.type.value != "UNKNOWN":
            meta_parts.append(f"Type: {anime.type.value}")
        if anime.status and anime.status.value != "UNKNOWN":
            meta_parts.append(f"Status: {anime.status.value}")
        if anime.year:
            meta_parts.append(f"Year: {anime.year}")
        if anime.season:
            meta_parts.append(f"Season: {anime.season.value}")
        return " | ".join(meta_parts)

    def _extract_anime_production_section(self, anime: Anime) -> str:
        """Extract studio, producer, and licensor credits for an anime.

        Args:
            anime: Anime model instance to extract production credits from.

        Returns:
            Pipe-delimited production credits grouped by company role.
        """
        # One company can hold several roles, so the company is listed under each.
        by_role: dict[CompanyRole, list[str]] = {}
        for company in anime.companies:
            if not company.name:
                continue
            for role in company.roles:
                by_role.setdefault(role, []).append(company.name)
        production_parts = [
            f"{label}: {', '.join(by_role[role])}"
            for role, label in (
                (CompanyRole.STUDIO, "Studios"),
                (CompanyRole.PRODUCER, "Producers"),
                (CompanyRole.LICENSOR, "Licensors"),
            )
            if by_role.get(role)
        ]
        return " | ".join(production_parts)

    def _extract_anime_temporal_and_streaming_sections(self, anime: Anime) -> list[str]:
        """Extract detailed airing date and streaming platform sections.

        Args:
            anime: Anime model instance to extract airing and streaming info from.

        Returns:
            List of non-empty temporal and streaming section strings.
        """
        sections: list[str] = []
        temp_parts = []
        if anime.aired_dates and anime.aired_dates.aired_from:
            temp_parts.append(f"Aired: {anime.aired_dates.aired_from}")
        if anime.month is not None and 1 <= anime.month <= 12:
            temp_parts.append(f"Month: {ENGLISH_MONTH_NAMES[anime.month - 1]}")
        if temp_parts:
            sections.append(" | ".join(temp_parts))

        if anime.streaming_sources:
            stream_parts = [
                f"Platform: {source.platform}"
                for source in anime.streaming_sources
                if source.platform
            ]
            if stream_parts:
                sections.append(" | ".join(stream_parts))
        return sections

    def extract_character_text(self, character: Character) -> str:
        """Extract optimized text for a Character Point.

        Combines character attributes including name, role, native name,
        nicknames, description, gender, and age into a searchable text string.

        Args:
            character: The Character model instance to extract text from.

        Returns:
            A pipe-delimited string of character attributes for embedding.
        """
        parts = [
            f"Name: {character.name}" if character.name else "",
            f"Roles: {', '.join(r.value for r in character.roles)}"
            if character.roles
            else "",
            f"Native Name: {character.name_native}" if character.name_native else "",
            f"Nicknames: {', '.join(character.nicknames)}"
            if character.nicknames
            else "",
            f"Background: {character.description}" if character.description else "",
            *(f"{k}: {v}" for k, v in character.attributes.items()),
        ]
        return " | ".join(filter(None, parts))

    def extract_episode_text(self, episode: Episode) -> str:
        """Extract optimized text for an Episode Point.

        Combines episode number, title, and synopsis. Also appends flags
        for filler or recap episodes when applicable.

        Args:
            episode: The Episode model instance to extract text from.

        Returns:
            A formatted string of episode information for embedding.
        """
        parts = [
            f"Episode {episode.episode_number}"
            if episode.episode_number is not None
            else "",
            f"Title: {episode.title}" if episode.title else "",
            f"Synopsis: {episode.synopsis}" if episode.synopsis else "",
        ]

        flags = []
        if episode.is_filler:
            flags.append("Filler")
        if episode.is_recap:
            flags.append("Recap")

        result = " | ".join(filter(None, parts))
        if flags:
            result += f" ({', '.join(flags)})"
        return result

    def extract_image_urls(self, anime: Anime) -> list[str]:
        """Extract all valid image URLs from an Anime for image point creation.

        Collects URLs from covers, posters, banners, and trailer thumbnails.
        Deduplicates and filters out None values.

        Args:
            anime: The Anime model instance to extract image URLs from.

        Returns:
            A deduplicated list of valid image URLs.
        """
        urls = []
        urls.extend(anime.images.covers)
        urls.extend(anime.images.posters)
        urls.extend(anime.images.banners)

        for trailer in anime.trailers:
            if trailer.thumbnail_source:
                urls.append(trailer.thumbnail_source)

        return list(dict.fromkeys(filter(None, urls)))

    def extract_character_image_urls(self, character: Character) -> list[str]:
        """Extract image URLs for a specific Character.

        Args:
            character: The Character model instance to extract image URLs from.

        Returns:
            A deduplicated list of valid character image URLs, or empty list if none.
        """
        if character.images:
            return list(dict.fromkeys(filter(None, character.images)))
        return []
