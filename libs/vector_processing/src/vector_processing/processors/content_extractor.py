"""Searchable content extraction from entity payloads.

This module provides entity-type-aware text extraction for search result
reranking. It knows about anime domain entities (anime, character, episode)
and extracts the most relevant fields for semantic matching.
"""

from typing import Any


class SearchableContentExtractor:
    """Extract searchable text from search result payloads.

    Entity-type-aware extraction for anime, character, episode entities.
    Pure logic layer with no dependencies on DB or ML models.

    This service supports reranking by providing clean, semantically rich text
    from structured payloads. Each entity type has optimized field extraction
    that prioritizes identity, semantic content, and contextual signals.
    """

    def extract_text(self, payload: dict[str, Any]) -> str:
        """Extract searchable text from payload based on entity_type.

        Args:
            payload: Document payload from search hit.

        Returns:
            Combined text content optimized for semantic matching.
        """
        entity_type = payload.get("entity_type", "anime")

        if entity_type == "anime":
            return self._extract_anime_text(payload)
        elif entity_type == "character":
            return self._extract_character_text(payload)
        elif entity_type == "episode":
            return self._extract_episode_text(payload)
        else:
            # Fallback for unknown types
            return str(payload.get("title") or payload.get("name", ""))

    def extract_batch(self, payloads: list[dict[str, Any]]) -> list[str]:
        """Extract text from multiple payloads efficiently.

        Args:
            payloads: List of document payloads.

        Returns:
            List of extracted text strings (same order as input).
        """
        return [self.extract_text(payload) for payload in payloads]

    def _extract_anime_text(self, payload: dict[str, Any]) -> str:
        """Extract text from anime entity for reranking.

        Priority fields:
        - title, title_english, title_japanese, synonyms (identity/aliases)
        - synopsis, background (primary semantic content)
        - genres, tags, demographics (contextual signals)
        - themes (thematic content)

        Args:
            payload: Anime entity payload.

        Returns:
            Space-separated text from priority fields.
        """
        parts = []

        # Primary identity (title and all variations)
        if title := payload.get("title"):
            parts.append(str(title))
        if title_english := payload.get("title_english"):
            parts.append(str(title_english))
        if title_japanese := payload.get("title_japanese"):
            parts.append(str(title_japanese))
        if synonyms := payload.get("synonyms"):
            if isinstance(synonyms, list):
                parts.append(" ".join(str(s) for s in synonyms))

        # Primary semantic content
        if synopsis := payload.get("synopsis"):
            parts.append(str(synopsis))
        if background := payload.get("background"):
            parts.append(str(background))

        # Contextual signals
        if genres := payload.get("genres"):
            if isinstance(genres, list):
                parts.append(" ".join(str(g) for g in genres))
        if tags := payload.get("tags"):
            if isinstance(tags, list):
                parts.append(" ".join(str(t) for t in tags))
        if demographics := payload.get("demographics"):
            if isinstance(demographics, list):
                parts.append(" ".join(str(d) for d in demographics))

        # Thematic content
        if themes := payload.get("themes"):
            if isinstance(themes, list):
                theme_texts = [
                    theme.get("name", "") for theme in themes if isinstance(theme, dict)
                ]
                parts.append(" ".join(theme_texts))

        return " ".join(parts)

    def _extract_character_text(self, payload: dict[str, Any]) -> str:
        """Extract text from character entity for reranking.

        Priority fields:
        - name, name_native, nicknames, name_variations (identity/aliases)
        - description (primary semantic content)
        - character_traits (personality/behavior tags)
        - role (character importance context)

        Args:
            payload: Character entity payload.

        Returns:
            Space-separated text from priority fields.
        """
        parts = []

        # Primary identity (name and all variations)
        if name := payload.get("name"):
            parts.append(str(name))
        if name_native := payload.get("name_native"):
            parts.append(str(name_native))
        if nicknames := payload.get("nicknames"):
            if isinstance(nicknames, list):
                parts.append(" ".join(str(n) for n in nicknames))
        if name_variations := payload.get("name_variations"):
            if isinstance(name_variations, list):
                parts.append(" ".join(str(v) for v in name_variations))

        # Primary semantic content
        if description := payload.get("description"):
            parts.append(str(description))

        # Semantic traits
        if traits := payload.get("character_traits"):
            if isinstance(traits, list):
                parts.append(" ".join(str(t) for t in traits))

        # Role context
        if role := payload.get("role"):
            parts.append(f"{role} character")

        return " ".join(parts)

    def _extract_episode_text(self, payload: dict[str, Any]) -> str:
        """Extract text from episode entity for reranking.

        Priority fields:
        - title, synopsis, description (primary semantic content)
        - title_japanese, title_romaji (alternative titles)

        Args:
            payload: Episode entity payload.

        Returns:
            Space-separated text from priority fields.
        """
        parts = []

        # Primary content
        if title := payload.get("title"):
            parts.append(str(title))
        if synopsis := payload.get("synopsis"):
            parts.append(str(synopsis))
        if description := payload.get("description"):
            parts.append(str(description))

        # Alternative titles
        if title_jp := payload.get("title_japanese"):
            parts.append(str(title_jp))
        if title_romaji := payload.get("title_romaji"):
            parts.append(str(title_romaji))

        return " ".join(parts)
