"""MAL direct scraping crawler package.

Replaces Jikan (third-party MAL API) with direct crawl4ai-based scrapers.

Public API:
    fetch_mal_anime(mal_id)               → MalScrapedAnime | None
    fetch_mal_character_refs(mal_id)   → list[CharacterRef]          (mal_character_refs_crawler)
    fetch_mal_character(char_id)       → MalScrapedCharacter | None  (mal_character_crawler)
    fetch_mal_characters(char_ids)     → list[MalScrapedCharacter | None]  (mal_character_crawler)
    fetch_mal_episode(mal_id, ep_num)     → MalScrapedEpisode | None
"""
