"""MAL direct scraping crawler package.

Replaces Jikan (third-party MAL API) with direct crawl4ai-based scrapers.

Public API:
    fetch_mal_anime(mal_id)               → MalAnime | None
    fetch_mal_character_refs(mal_id)   → list[CharacterRef]          (mal_character_refs_crawler)
    fetch_mal_character(char_id)       → MalCharacter | None  (mal_character_crawler)
    fetch_mal_characters(char_ids)     → list[MalCharacter | None]  (mal_character_crawler)
    fetch_mal_episode(mal_id, ep_num)     → MalEpisode | None
"""
