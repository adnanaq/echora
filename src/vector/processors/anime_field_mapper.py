# src/vector/anime_field_mapper.py
"""
AnimeFieldMapper - Extract and map anime data fields to 13-vector semantic architecture

Maps anime data from AnimeEntry models to appropriate text/visual embeddings
for each vector type. Implements the comprehensive field mapping strategy
defined in Phase 2.5 architecture with character image semantic separation.
"""

import logging
from typing import Any, Dict, List, Union

from src.models.anime import AnimeEntry

logger = logging.getLogger(__name__)


class AnimeFieldMapper:
    """
    Maps anime data fields to 11-vector semantic architecture.

    Extracts and processes anime data for embedding into:
    - 9 text vectors (BGE-M3, 1024-dim each) for semantic search
    - 2 visual vectors (OpenCLIP ViT-L/14, 768-dim each) for image search
      * image_vector: covers, posters, banners, trailer thumbnails
      * character_image_vector: character images for character identification
    """

    def __init__(self) -> None:
        """Initialize the anime field mapper."""
        self.logger = logger

    def map_anime_to_vectors(
        self, anime: AnimeEntry
    ) -> Dict[str, Union[str, List[str]]]:
        """
        Map complete anime entry to all 13 vectors.

        Args:
            anime: AnimeEntry model with comprehensive anime data

        Returns:
            Dict mapping vector names to their content for embedding
        """
        vector_data: Dict[str, Union[str, List[str]]] = {}

        # Text vectors (9)
        vector_data["title_vector"] = self._extract_title_content(anime)
        vector_data["character_vector"] = self._extract_character_content(anime)
        vector_data["genre_vector"] = self._extract_genre_content(anime)
        vector_data["staff_vector"] = self._extract_staff_content(anime)
        vector_data["temporal_vector"] = self._extract_temporal_content(anime)
        vector_data["streaming_vector"] = self._extract_streaming_content(anime)
        vector_data["related_vector"] = self._extract_related_content(anime)
        vector_data["franchise_vector"] = self._extract_franchise_content(anime)
        vector_data["episode_vector"] = self._extract_episode_content(anime)

        # Visual vectors (2)
        vector_data["image_vector"] = self._extract_image_content(anime)
        vector_data["character_image_vector"] = self._extract_character_image_content(
            anime
        )

        return vector_data

    # ============================================================================
    # TEXT VECTOR EXTRACTORS (BGE-M3, 1024-dim)
    # ============================================================================

    def _extract_title_content(self, anime: AnimeEntry) -> str:
        """Extract title, synopsis, background, and synonyms for semantic search."""
        content_parts = []

        # Primary titles
        if anime.title:
            content_parts.append(f"Title: {anime.title}")
        if anime.title_english:
            content_parts.append(f"English: {anime.title_english}")
        if anime.title_japanese:
            content_parts.append(f"Japanese: {anime.title_japanese}")

        # Alternative titles
        if anime.synonyms:
            content_parts.append(f"Synonyms: {', '.join(anime.synonyms)}")

        # Descriptive content
        if anime.synopsis:
            content_parts.append(f"Synopsis: {anime.synopsis}")
        if anime.background:
            content_parts.append(f"Background: {anime.background}")

        return " | ".join(content_parts)

    def _extract_character_content(self, anime: AnimeEntry) -> str:
        """Extract character information for semantic character search."""
        content_parts = []

        for char in anime.characters:
            char_info = []

            # Character name and role
            char_info.append(f"Name: {char.name}")
            if char.role:
                char_info.append(f"Role: {char.role}")

            # Name variations
            if char.name_variations:
                char_info.append(f"Variations: {', '.join(char.name_variations)}")
            if char.name_native:
                char_info.append(f"Native: {char.name_native}")
            if char.nicknames:
                char_info.append(f"Nicknames: {', '.join(char.nicknames)}")

            # Character details
            if char.description:
                char_info.append(f"Description: {char.description}")
            if char.age:
                char_info.append(f"Age: {char.age}")
            if char.gender:
                char_info.append(f"Gender: {char.gender}")

            if char_info:
                content_parts.append(" | ".join(char_info))

        return " || ".join(content_parts)

    def _extract_genre_content(self, anime: AnimeEntry) -> str:
        """Extract genres, tags, themes, demographics, and content warnings."""
        content_parts = []

        if anime.genres:
            content_parts.append(f"Genres: {', '.join(anime.genres)}")
        if anime.tags:
            content_parts.append(f"Tags: {', '.join(anime.tags)}")
        if anime.demographics:
            content_parts.append(f"Demographics: {', '.join(anime.demographics)}")
        if anime.content_warnings:
            content_parts.append(
                f"Content Warnings: {', '.join(anime.content_warnings)}"
            )

        # Theme descriptions
        theme_info = []
        for theme in anime.themes:
            if hasattr(theme, "name") and theme.name:
                theme_part = f"Theme: {theme.name}"
                if hasattr(theme, "description") and theme.description:
                    theme_part += f" - {theme.description}"
                theme_info.append(theme_part)
        if theme_info:
            content_parts.append(" | ".join(theme_info))

        return " | ".join(content_parts)


    def _extract_staff_content(self, anime: AnimeEntry) -> str:
        """Extract staff data including directors, composers, studios, voice actors."""
        content_parts = []

        # Extract staff information from staff_data StaffData object
        if anime.staff_data:
            # Production staff by role - NEW: Dynamic role extraction
            if anime.staff_data.production_staff:
                all_roles = anime.staff_data.production_staff.get_all_roles()
                for role_key, staff_members in all_roles.items():
                    staff_names = []
                    for member in staff_members:
                        # Handle both dict and object cases
                        if hasattr(member, 'name') and member.name:
                            staff_names.append(member.name)
                        elif isinstance(member, dict) and member.get('name'):
                            staff_names.append(member['name'])
                    if staff_names:
                        # Format role name for display (replace underscores, title case)
                        role_display = role_key.replace('_', ' ').title()
                        content_parts.append(f"{role_display}: {', '.join(staff_names)}")

            # Studios
            if anime.staff_data.studios:
                studio_names = [
                    studio.name for studio in anime.staff_data.studios if studio.name
                ]
                if studio_names:
                    content_parts.append(f"Studios: {', '.join(studio_names)}")

            # Producers
            if anime.staff_data.producers:
                producer_names = [
                    producer.name
                    for producer in anime.staff_data.producers
                    if producer.name
                ]
                if producer_names:
                    content_parts.append(f"Producers: {', '.join(producer_names)}")

            # Licensors
            if anime.staff_data.licensors:
                licensor_names = [
                    licensor.name
                    for licensor in anime.staff_data.licensors
                    if licensor.name
                ]
                if licensor_names:
                    content_parts.append(f"Licensors: {', '.join(licensor_names)}")

            # Voice actors
            if anime.staff_data.voice_actors and anime.staff_data.voice_actors.japanese:
                for va in anime.staff_data.voice_actors.japanese:
                    if va.name and va.character_assignments:
                        content_parts.append(
                            f"Voice Actor: {va.name} ({', '.join(va.character_assignments)})"
                        )

        return " | ".join(content_parts)

def _extract_temporal_content(self, anime: AnimeEntry) -> str:
        """Extract aired dates, anime season, broadcast, premiere dates."""
        content_parts = []

        # Aired dates
        if anime.aired_dates:
            if hasattr(anime.aired_dates, "from_date") and anime.aired_dates.from_date:
                content_parts.append(f"Aired From: {anime.aired_dates.from_date}")
            if hasattr(anime.aired_dates, "to_date") and anime.aired_dates.to_date:
                content_parts.append(f"Aired To: {anime.aired_dates.to_date}")

        # Broadcast information
        if hasattr(anime, "broadcast") and anime.broadcast:
            if hasattr(anime.broadcast, "day") and anime.broadcast.day:
                broadcast_info = f"Broadcast: {anime.broadcast.day}"
                if hasattr(anime.broadcast, "time") and anime.broadcast.time:
                    broadcast_info += f" at {anime.broadcast.time}"
                content_parts.append(broadcast_info)

        # Premiere month
        if anime.month:
            content_parts.append(f"Premiere Month: {anime.month}")

        # Broadcast schedule
        if anime.broadcast_schedule:
            if anime.broadcast_schedule.jpn_time:
                content_parts.append(f"Japan Time: {anime.broadcast_schedule.jpn_time}")
            if anime.broadcast_schedule.sub_time:
                content_parts.append(f"Sub Time: {anime.broadcast_schedule.sub_time}")
            if anime.broadcast_schedule.dub_time:
                content_parts.append(f"Dub Time: {anime.broadcast_schedule.dub_time}")

        # Delay information
        if anime.delay_information:
            if anime.delay_information.delayed_timetable:
                content_parts.append("Delayed Timetable: Yes")
            if anime.delay_information.delay_reason:
                content_parts.append(
                    f"Delay Reason: {anime.delay_information.delay_reason}"
                )

        # Premiere dates
        if anime.premiere_dates:
            if anime.premiere_dates.original:
                content_parts.append(
                    f"Original Premiere: {anime.premiere_dates.original}"
                )
            if anime.premiere_dates.sub:
                content_parts.append(f"Sub Premiere: {anime.premiere_dates.sub}")
            if anime.premiere_dates.dub:
                content_parts.append(f"Dub Premiere: {anime.premiere_dates.dub}")

        return " | ".join(content_parts)

    def _extract_streaming_content(self, anime: AnimeEntry) -> str:
        """Extract streaming platform information and licenses."""
        content_parts = []

        # Streaming platforms
        streaming_info = []
        for stream in anime.streaming_info:
            if stream.platform:
                stream_part = f"Platform: {stream.platform}"
                if stream.url:
                    stream_part += f" ({stream.url})"
                if stream.region:
                    stream_part += f" - Region: {stream.region}"
                if stream.free is not None:
                    stream_part += f" - Free: {stream.free}"
                streaming_info.append(stream_part)
        if streaming_info:
            content_parts.extend(streaming_info)

        # Streaming licenses
        if anime.streaming_licenses:
            content_parts.append(f"Licenses: {', '.join(anime.streaming_licenses)}")

        return " | ".join(content_parts)

    def _extract_related_content(self, anime: AnimeEntry) -> str:
        """Extract related anime and franchise connections."""
        content_parts = []

        # Related anime entries (anime-to-anime relationships)
        related_info = []
        for related in anime.related_anime:
            if hasattr(related, "title") and related.title:
                relation_type = getattr(related, "relation_type", "Other")

                if relation_type == "Sequel":
                    related_part = f"Followed by {related.title}"
                elif relation_type == "Prequel":
                    related_part = f"Preceded by {related.title}"
                elif relation_type == "Character":
                    related_part = f"Shares characters with {related.title}"
                elif relation_type in ["Parent Story", "Parent story"]:
                    related_part = f"Side story of {related.title}"
                elif relation_type in ["Side story", "Side Story"]:
                    related_part = f"Has side story {related.title}"
                elif relation_type == "Music Video":
                    related_part = f"Has music video {related.title}"
                elif relation_type == "Special":
                    related_part = f"Has special {related.title}"
                elif relation_type == "Movie":
                    related_part = f"Has movie {related.title}"
                elif relation_type == "ONA":
                    related_part = f"Has online series {related.title}"
                else:  # "Other" and unknown types
                    related_part = f"Related to {related.title}"

                related_info.append(related_part)
        if related_info:
            content_parts.extend(related_info)

        # Relations with URLs (anime-to-source material relationships)
        relation_info = []
        for relation in anime.relations:
            if hasattr(relation, "title") and relation.title:
                relation_type = getattr(relation, "relation_type", "Other")

                if relation_type == "Adaptation":
                    relation_part = f"Adapted into {relation.title}"
                elif relation_type == "Original Work":
                    relation_part = f"Based on {relation.title}"
                else:
                    relation_part = f"Related to {relation.title}"

                relation_info.append(relation_part)
        if relation_info:
            content_parts.extend(relation_info)

        return " | ".join(content_parts)

    def _extract_franchise_content(self, anime: AnimeEntry) -> str:
        """Extract trailers, opening themes, ending themes (multimedia content)."""
        content_parts = []

        # Trailers
        trailer_info = []
        for trailer in anime.trailers:
            if hasattr(trailer, "title") and trailer.title:
                trailer_part = f"Trailer: {trailer.title}"
                if hasattr(trailer, "url") and trailer.url:
                    trailer_part += f" ({trailer.url})"
                trailer_info.append(trailer_part)
        if trailer_info:
            content_parts.extend(trailer_info)

        # Opening themes
        opening_info = []
        for opening in anime.opening_themes:
            if hasattr(opening, "title") and opening.title:
                opening_part = f"Opening: {opening.title}"
                if hasattr(opening, "artist") and opening.artist:
                    opening_part += f" by {opening.artist}"
                opening_info.append(opening_part)
        if opening_info:
            content_parts.extend(opening_info)

        # Ending themes
        ending_info = []
        for ending in anime.ending_themes:
            if hasattr(ending, "title") and ending.title:
                ending_part = f"Ending: {ending.title}"
                if hasattr(ending, "artist") and ending.artist:
                    ending_part += f" by {ending.artist}"
                ending_info.append(ending_part)
        if ending_info:
            content_parts.extend(ending_info)

        return " | ".join(content_parts)

    def _extract_episode_content(self, anime: AnimeEntry) -> str:
        """Extract detailed episode information with all available fields and chunking for large series."""
        if not anime.episode_details:
            return ""

        # Constants for chunking strategy
        EPISODES_PER_CHUNK = 50  # Future-proof for rich episode data

        # Extract episode info with all available semantic fields
        episode_info = []
        for episode in anime.episode_details:
            ep_parts = []

            # Episode number
            if hasattr(episode, "episode_number") and episode.episode_number:
                ep_parts.append(f"Episode {episode.episode_number}")

            # Season context
            if hasattr(episode, "season_number") and episode.season_number:
                ep_parts.append(f"Season {episode.season_number}")

            # English title (if available and meaningful)
            if hasattr(episode, "title") and episode.title and episode.title.strip():
                title = episode.title.strip()
                # Skip only purely generic patterns like "Episode X"
                if not (title.startswith("Episode ") and title[8:].isdigit()):
                    ep_parts.append(title)

            # Japanese title (for cultural searches, if different from English)
            if (
                hasattr(episode, "title_japanese")
                and episode.title_japanese
                and episode.title_japanese.strip()
            ):
                japanese_title = episode.title_japanese.strip()
                # Only add if different from English title
                english_title = getattr(episode, "title", "")
                if japanese_title != english_title:
                    ep_parts.append(f"Japanese: {japanese_title}")

            # Synopsis (richest semantic content)
            if (
                hasattr(episode, "synopsis")
                and episode.synopsis
                and episode.synopsis.strip()
            ):
                synopsis = episode.synopsis.strip()
                ep_parts.append(synopsis)

            # Aired date (temporal context)
            if hasattr(episode, "aired") and episode.aired:
                aired_date = str(episode.aired)
                # Extract just the date part
                if "T" in aired_date:
                    aired_date = aired_date.split("T")[0]
                ep_parts.append(f"Aired: {aired_date}")

            # Duration (user-searchable temporal metadata)
            if hasattr(episode, "duration") and episode.duration:
                duration_seconds = episode.duration
                if duration_seconds > 0:
                    # Store in seconds to match data format
                    ep_parts.append(f"{duration_seconds} seconds")

            # Episode score (quality indicator for search)
            if hasattr(episode, "score") and episode.score is not None:
                # Format score for semantic search
                ep_parts.append(f"rated {episode.score}")

            # Build episode entry if we have any content
            if ep_parts:
                episode_entry = (
                    ": ".join(ep_parts) if len(ep_parts) > 1 else ep_parts[0]
                )

                # Add episode type flags
                type_flags = []
                if hasattr(episode, "filler") and episode.filler:
                    type_flags.append("Filler")
                if hasattr(episode, "recap") and episode.recap:
                    type_flags.append("Recap")

                if type_flags:
                    episode_entry += f" ({', '.join(type_flags)})"

                episode_info.append(episode_entry)

        if not episode_info:
            return ""

        # For small series (≤50 episodes), return directly
        if len(episode_info) <= EPISODES_PER_CHUNK:
            return " | ".join(episode_info)

        # For large series, chunk episodes for future hierarchical averaging
        chunks = []
        for i in range(0, len(episode_info), EPISODES_PER_CHUNK):
            chunk = episode_info[i : i + EPISODES_PER_CHUNK]
            chunk_content = " | ".join(chunk)
            chunks.append(chunk_content)

        # For now, join all chunks (hierarchical averaging to be implemented in embedding_manager)
        return " || CHUNK_SEPARATOR || ".join(chunks)


    # ============================================================================
    # VISUAL VECTOR EXTRACTORS (OpenCLIP ViT-L/14, 768-dim)
    # ============================================================================

    def _extract_image_content(self, anime: AnimeEntry) -> List[str]:
        """Extract general anime image URLs (covers, posters, banners, trailers) excluding character images."""
        image_urls = []

        # Process all images from unified images field (now simple URL strings)
        if hasattr(anime, "images") and anime.images:
            # Process covers (highest priority)
            if "covers" in anime.images and anime.images["covers"]:
                for cover_url in anime.images["covers"]:
                    if cover_url:  # Simple URL string
                        image_urls.append(cover_url)

            # Process posters (high quality promotional images)
            if "posters" in anime.images and anime.images["posters"]:
                for poster_url in anime.images["posters"]:
                    if poster_url:  # Simple URL string
                        image_urls.append(poster_url)

            # Process banners (additional visual content)
            if "banners" in anime.images and anime.images["banners"]:
                for banner_url in anime.images["banners"]:
                    if banner_url:  # Simple URL string
                        image_urls.append(banner_url)

            # Process any other image types in the images field
            for image_type, image_list in anime.images.items():
                if image_type not in ["covers", "posters", "banners"] and image_list:
                    for image_url in image_list:
                        if image_url:  # Simple URL string
                            image_urls.append(image_url)

        # Trailer thumbnails (promotional visual content)
        for trailer in anime.trailers:
            if hasattr(trailer, "thumbnail_url") and trailer.thumbnail_url:
                image_urls.append(trailer.thumbnail_url)

        # Remove duplicates while preserving order
        unique_urls = list(dict.fromkeys(image_urls))
        return unique_urls

    def _extract_character_image_content(self, anime: AnimeEntry) -> List[str]:
        """Extract character image URLs for character-specific visual embedding."""
        character_image_urls = []

        # Extract character images separately for character identification and recommendations
        for character in anime.characters:
            if character.images:
                # character.images is now List[str] with direct image URLs
                for image_url in character.images:
                    if image_url:
                        character_image_urls.append(image_url)

        # Remove duplicates while preserving order
        unique_urls = list(dict.fromkeys(character_image_urls))
        return unique_urls

    # ============================================================================
    # UTILITY METHODS
    # ============================================================================

    def get_vector_types(self) -> Dict[str, str]:
        """Get mapping of vector names to their types (text/visual)."""
        return {
            # Text vectors (BGE-M3, 1024-dim)
            "title_vector": "text",
            "character_vector": "text",
            "genre_vector": "text",
            "staff_vector": "text",
            "temporal_vector": "text",
            "streaming_vector": "text",
            "related_vector": "text",
            "franchise_vector": "text",
            "episode_vector": "text",
            # Visual vectors (OpenCLIP ViT-L/14, 768-dim)
            "image_vector": "visual",
            "character_image_vector": "visual",
        }

    def validate_mapping(self, vector_data: Dict[str, Any]) -> bool:
        """Validate that vector data contains all expected vectors."""
        expected_vectors = set(self.get_vector_types().keys())
        actual_vectors = set(vector_data.keys())

        missing_vectors = expected_vectors - actual_vectors
        if missing_vectors:
            self.logger.warning(f"Missing vectors: {missing_vectors}")
            return False

        return True

    def _extract_image_url(self, anime: AnimeEntry) -> str:
        """Extract the primary image URL for visual embedding using unified images field.

        Args:
            anime: AnimeEntry instance

        Returns:
            Primary image URL or empty string if not available
        """
        # Use unified images field with priority: covers -> posters -> banners
        if hasattr(anime, "images") and anime.images:
            # Priority 1: covers (best quality cover images)
            if "covers" in anime.images and anime.images["covers"]:
                for cover_url in anime.images["covers"]:
                    if cover_url:  # Simple URL string check
                        return cover_url

            # Priority 2: posters (good quality poster images)
            if "posters" in anime.images and anime.images["posters"]:
                for poster_url in anime.images["posters"]:
                    if poster_url:  # Simple URL string check
                        return poster_url

            # Priority 3: banners (fallback option)
            if "banners" in anime.images and anime.images["banners"]:
                for banner_url in anime.images["banners"]:
                    if banner_url:  # Simple URL string check
                        return banner_url

        return ""
