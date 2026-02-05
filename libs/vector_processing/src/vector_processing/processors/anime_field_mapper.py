"""AnimeFieldMapper for extracting and mapping anime data to semantic text.

This module provides the AnimeFieldMapper class which converts raw model data
(Anime, Character, Episode) into optimized text strings suitable for vector
embedding. It serves as the "Content Strategist" in the vector processing
pipeline, determining what information is important for semantic search.
"""

import logging

from common.models.anime import Anime, Character, Episode

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
        sections = []

        # 1. Core Content (Title + Synopsis)
        title_parts = [
            f"Title: {anime.title}" if anime.title else "",
            f"English: {anime.title_english}" if anime.title_english else "",
            f"Japanese: {anime.title_japanese}" if anime.title_japanese else "",
            f"Synonyms: {', '.join(anime.synonyms)}" if anime.synonyms else "",
            f"Story: {anime.synopsis}" if anime.synopsis else "",
            f"Production Background: {anime.background}" if anime.background else "",
        ]
        sections.append(" | ".join(filter(None, title_parts)))

        # 2. Genres & Themes (with expansions for better semantic matching)
        if anime.genres or anime.tags or anime.themes:
            genre_parts = []
            if anime.genres:
                genres = ", ".join(anime.genres)
                # Expand demographic terms for broader search coverage
                genres = genres.replace("Shounen", "Shonen (young male)")
                genres = genres.replace("Shoujo", "Shojo (young female)")
                genres = genres.replace("Seinen", "Seinen (adult male)")
                genres = genres.replace("Josei", "Josei (adult female)")
                genre_parts.append(f"Genres: {genres}")

            if anime.tags:
                genre_parts.append(f"Tags: {', '.join(anime.tags)}")

            theme_names = [
                t.name for t in anime.themes if hasattr(t, "name") and t.name
            ]
            if theme_names:
                genre_parts.append(f"Themes: {', '.join(theme_names)}")

            sections.append(" | ".join(genre_parts))

        # 3. Staff & Production
        if anime.staff_data:
            staff_parts = []
            # Studios
            studios = [s.name for s in anime.staff_data.studios if s.name]
            if studios:
                staff_parts.append(f"Studios: {', '.join(studios)}")

            # Key Staff (Directors/Composers)
            if anime.staff_data.production_staff:
                roles = anime.staff_data.production_staff.get_all_roles()
                for role, members in roles.items():
                    names = [m.name for m in members if hasattr(m, "name") and m.name]
                    if names:
                        role_name = role.replace("_", " ").title()
                        staff_parts.append(f"{role_name}: {', '.join(names)}")

            sections.append(" | ".join(staff_parts))

        # 4. Temporal & Status
        temp_parts = []
        if anime.aired_dates and anime.aired_dates.from_date:
            temp_parts.append(f"Aired: {anime.aired_dates.from_date}")
        if anime.month:
            temp_parts.append(f"Season: {anime.month}")
        sections.append(" | ".join(temp_parts))

        # 5. Streaming & Franchise
        if anime.streaming_info:
            stream_parts = []
            for s in anime.streaming_info:
                if s.platform:
                    stream_parts.append(f"Platform: {s.platform}")
            if stream_parts:
                sections.append(" | ".join(stream_parts))

        return " || ".join(filter(None, sections))

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
            f"Role: {character.role}" if character.role else "",
            f"Native Name: {character.name_native}" if character.name_native else "",
            f"Nicknames: {', '.join(character.nicknames)}"
            if character.nicknames
            else "",
            f"Background: {character.description}" if character.description else "",
            f"Gender: {character.gender}" if character.gender else "",
            f"Age: {character.age}" if character.age else "",
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
        if hasattr(episode, "filler") and episode.filler:
            flags.append("Filler")
        if hasattr(episode, "recap") and episode.recap:
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
        if anime.images:
            for category in ["covers", "posters", "banners"]:
                urls.extend(anime.images.get(category, []))

        for trailer in anime.trailers:
            if hasattr(trailer, "thumbnail_url") and trailer.thumbnail_url:
                urls.append(trailer.thumbnail_url)

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
