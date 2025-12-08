# Field Standardization Analysis

This document outlines a proposal for standardizing data fields across various anime data sources. The analysis is based on the JSON outputs from the following sources: anidb, anilist, anime-planet, animeschedule, jikan (MyAnimeList), and kitsu.

## Proposed Standardized Fields

### Core Anime Information

*   **Title**:
    *   **Standard Field**: `title` (main title), `title_english`, `title_japanese`, `synonyms` (array)
    *   **Sources**: `title` (object), `titles` (object), `canonicalTitle`, `names` (object)

*   **Type**:
    *   **Standard Field**: `type` (e.g., "TV", "Movie", "OVA")
    *   **Sources**: `type`, `format`, `showType`, `subtype`, `mediaTypes`

*   **Episodes**:
    *   **Standard Field**: `episodes`
    *   **Sources**: `episode_count`, `episodes`, `numberOfEpisodes`, `episodeCount`

*   **Status**:
    *   **Standard Field**: `status` (e.g., "Finished Airing", "Currently Airing")
    *   **Sources**: `status`

*   **Airing Dates**:
    *   **Standard Field**: `aired` (object with `from` and `to` ISO 8601 dates)
    *   **Sources**: `start_date`/`end_date`, `startDate`/`endDate`, `premier`, `aired`

*   **Season & Year**:
    *   **Standard Field**: `season`, `year`
    *   **Sources**: `season`, `seasonYear`, `year`, `animeSeason`

*   **Duration**:
    *   **Standard Field**: `duration_minutes`
    *   **Sources**: `duration`, `lengthMin`, `episodeLength`

*   **Age Rating**:
    *   **Standard Field**: `age_rating` (e.g., "PG-13")
    *   **Sources**: `rating`, `ageRating`, `ageRatingGuide`

### Descriptions

*   **Synopsis**:
    *   **Standard Field**: `synopsis`
    *   **Sources**: `description`, `synopsis`

*   **Background**:
    *   **Standard Field**: `background`
    *   **Sources**: `background`

### Media

*   **Images**:
    *   **Standard Field**: `images` (object with `poster`, `cover`, `banner` URLs)
    *   **Sources**: `picture`, `coverImage`, `bannerImage`, `posterImage`, `images`, `thumbnail`

*   **Trailer**:
    *   **Standard Field**: `trailer_url`
    *   **Sources**: `trailer`, `youtubeVideoId`

### Classification

*   **Genres, Tags, and Demographics**:
    *   **Standard Fields**: `genres`, `tags`, `demographics` (arrays of strings)
    *   **Sources**: `genres`, `tags`, `themes`, `demographics`

### Production

*   **Studios & Producers**:
    *   **Standard Fields**: `studios`, `producers` (arrays of strings)
    *   **Sources**: `studios`, `producers`, `licensors`

*   **Staff**:
    *   **Standard Field**: `staff` (array of objects with `name` and `role`)
    *   **Sources**: `creators`, `staff`, `director`, `creator`

### Scores and Statistics

*   **Score**:
    *   **Standard Field**: `score` (normalized to a common scale, e.g., out of 10)
    *   **Sources**: `ratings`, `averageScore`, `meanScore`, `rating`, `score`

*   **Statistics**:
    *   **Standard Field**: `statistics` (object for `popularity_rank`, `favorites_count`, etc.)
    *   **Sources**: `popularity`, `favourites`, `trending`, `userCount`, `favoritesCount`, `members`, `rank`

### Relationships and External Links

*   **Source Material**:
    *   **Standard Field**: `source_material` (e.g., "Manga")
    *   **Sources**: `source`, `sources`, `original work`

*   **Related Anime**:
    *   **Standard Field**: `related_anime` (array of objects with `relation_type` and `entry`)
    *   **Sources**: `relations`, `relatedAnime`

*   **External & Streaming Links**:
    *   **Standard Fields**: `external_links`, `streaming_links` (objects or arrays of objects)
    *   **Sources**: `externalLinks`, `websites`, `external`, `streamingEpisodes`, `streaming`
