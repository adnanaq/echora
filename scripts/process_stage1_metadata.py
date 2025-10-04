#!/usr/bin/env python3
"""
Comprehensive Stage 1 Metadata Extraction Script

This script implements the complete Stage 1 metadata extraction process according to
the prompt template in src/enrichment/prompts/stages/01_metadata_extraction.txt

Key Features:
- Foundation-first approach using offline database as base
- Multi-source integration from 6 APIs with intelligent deduplication
- Array field deduplication (genres, synonyms, demographics, etc.)
- Complex theme merging with conflict resolution
- Image organization by type (covers, posters, banners)
- External link normalization and deduplication
- Synopsis extraction with 6-level source hierarchy
- Schema compliance with AnimeEntry Pydantic model
"""

import json
import re
import sys
import os
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from datetime import datetime

# Project root for resolving paths (works from anywhere)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_offline_database(current_anime_file: str) -> Dict[str, Any]:
    """Load offline database entry as foundation."""
    try:
        with open(current_anime_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading offline database: {e}")
        return {}


def load_source_data(temp_dir: str) -> Dict[str, Dict[str, Any]]:
    """Load all external source data files."""
    sources = {}

    source_files = {
        'jikan': f'{temp_dir}/jikan.json',
        'animeschedule': f'{temp_dir}/animeschedule.json',
        'kitsu': f'{temp_dir}/kitsu.json',
        'anime_planet': f'{temp_dir}/anime_planet.json',
        'anilist': f'{temp_dir}/anilist.json',
        'anidb': f'{temp_dir}/anidb.json',
        'anisearch': f'{temp_dir}/anisearch.json'
    }

    for source_name, file_path in source_files.items():
        try:
            with open(file_path, 'r') as f:
                sources[source_name] = json.load(f)
                print(f"Loaded {source_name} data")
        except Exception as e:
            print(f"Warning: Could not load {source_name}: {e}")
            sources[source_name] = {}

    return sources


def normalize_string_for_comparison(text: str) -> str:
    """Normalize string for case-insensitive comparison."""
    if not text:
        return ""
    return text.lower().strip()


def deduplicate_array_field(offline_values: List[str], external_values: List[str]) -> List[str]:
    """
    Deduplicate array field values with offline database as foundation.

    Args:
        offline_values: Values from offline database
        external_values: Values from external sources

    Returns:
        Deduplicated list with offline values first, then unique external values
    """
    result = []
    seen = set()

    # Add offline values first
    for value in offline_values:
        if value and value.strip():
            normalized = normalize_string_for_comparison(value)
            if normalized not in seen:
                result.append(value.strip())
                seen.add(normalized)

    # Add unique external values
    for value in external_values:
        if value and value.strip():
            normalized = normalize_string_for_comparison(value)
            if normalized not in seen:
                result.append(value.strip())
                seen.add(normalized)

    return result


def merge_themes_intelligently(offline_themes: List[Dict], external_themes: List[Dict],
                             existing_genres: Set[str]) -> List[Dict]:
    """
    Merge themes from multiple sources with intelligent conflict resolution.

    Args:
        offline_themes: Themes from offline database
        external_themes: Themes from external sources (Jikan, Kitsu, AniList, AniDB)
        existing_genres: Set of existing genre names (case-insensitive)

    Returns:
        List of merged theme objects with proper deduplication
    """
    themes_by_name = {}

    # Process offline themes first
    for theme in offline_themes:
        if not theme or not theme.get('name'):
            continue
        name = theme['name'].strip()
        normalized_name = normalize_string_for_comparison(name)

        # Skip if matches existing genre (case-insensitive)
        if normalized_name in existing_genres:
            continue

        themes_by_name[normalized_name] = {
            'name': name,
            'description': theme.get('description')
        }

    # Process external themes with priority: Jikan → Kitsu → AniList → AniDB
    for theme in external_themes:
        if not theme or not theme.get('name'):
            continue

        name = theme['name'].strip()
        normalized_name = normalize_string_for_comparison(name)
        description = theme.get('description')

        # Skip if matches existing genre (case-insensitive)
        if normalized_name in existing_genres:
            continue

        if normalized_name in themes_by_name:
            # Theme exists - update description if current has null and new has description
            existing_theme = themes_by_name[normalized_name]
            if existing_theme['description'] is None and description:
                existing_theme['description'] = description
        else:
            # New theme - add it
            themes_by_name[normalized_name] = {
                'name': name,
                'description': description
            }

    # Return themes in insertion order
    return list(themes_by_name.values())


def organize_images_by_type(sources: Dict[str, Dict]) -> Dict[str, List[str]]:
    """
    Organize images by type from all sources with URL preprocessing.

    Returns:
        Dict with keys: 'covers', 'posters', 'banners' containing URL arrays
    """
    images = {
        'covers': [],
        'posters': [],
        'banners': []
    }

    all_urls = {
        'covers': set(),
        'posters': set(),
        'banners': set()
    }

    # Jikan images (covers)
    jikan = sources.get('jikan', {})
    if jikan.get('data', {}).get('images', {}).get('jpg', {}).get('large_image_url'):
        url = jikan['data']['images']['jpg']['large_image_url']
        if url not in all_urls['covers']:
            images['covers'].append(url)
            all_urls['covers'].add(url)

    # AnimSchedule images (covers)
    animeschedule = sources.get('animeschedule', {})
    if animeschedule.get('imageVersionRoute'):
        url = f"https://img.animeschedule.net/production/assets/public/img/{animeschedule['imageVersionRoute']}"
        if url not in all_urls['covers']:
            images['covers'].append(url)
            all_urls['covers'].add(url)

    # Kitsu images (posters and covers)
    kitsu = sources.get('kitsu', {})
    if kitsu.get('anime', {}).get('attributes'):
        attrs = kitsu['anime']['attributes']

        # Poster image
        if attrs.get('posterImage', {}).get('original'):
            url = attrs['posterImage']['original']
            if url not in all_urls['posters']:
                images['posters'].append(url)
                all_urls['posters'].add(url)

        # Cover image
        if attrs.get('coverImage', {}).get('original'):
            url = attrs['coverImage']['original']
            if url not in all_urls['covers']:
                images['covers'].append(url)
                all_urls['covers'].add(url)

    # Anime-Planet images (covers) - now flattened to top level
    anime_planet = sources.get('anime_planet', {})
    if anime_planet.get('image'):
        img_url = anime_planet['image']
        if img_url and img_url not in all_urls['covers']:
            images['covers'].append(img_url)
            all_urls['covers'].add(img_url)

    # Anime-Planet poster - now flattened to top level
    if anime_planet.get('poster'):
        img_url = anime_planet['poster']
        if img_url and img_url not in all_urls['posters']:
            images['posters'].append(img_url)
            all_urls['posters'].add(img_url)

    # AniList images (covers and banners)
    anilist = sources.get('anilist', {})

    # Cover image
    if anilist.get('coverImage', {}).get('extraLarge'):
        url = anilist['coverImage']['extraLarge']
        if url not in all_urls['covers']:
            images['covers'].append(url)
            all_urls['covers'].add(url)

    # Banner image
    if anilist.get('bannerImage'):
        url = anilist['bannerImage']
        if url not in all_urls['banners']:
            images['banners'].append(url)
            all_urls['banners'].add(url)

    # AniDB images (covers)
    anidb = sources.get('anidb', {})
    if anidb.get('picture'):
        url = f"https://cdn.anidb.net/images/main/{anidb['picture']}"
        if url not in all_urls['covers']:
            images['covers'].append(url)
            all_urls['covers'].add(url)

    # AniSearch images (covers and screenshots)
    anisearch = sources.get('anisearch', {})
    if anisearch.get('cover'):
        img_url = anisearch['cover']
        if img_url and img_url not in all_urls['covers']:
            images['covers'].append(img_url)
            all_urls['covers'].add(img_url)

    # AniSearch screenshots (posters)
    for screenshot_url in anisearch.get('screenshots', []):
        if screenshot_url and screenshot_url not in all_urls['posters']:
            images['posters'].append(screenshot_url)
            all_urls['posters'].add(screenshot_url)

    return images


def normalize_external_links(sources: Dict[str, Dict]) -> Dict[str, str]:
    """
    Extract and normalize external links from all sources.

    Returns:
        Dict with normalized lowercase keys and URLs as values
    """
    external_links = {}

    # Jikan external links
    jikan = sources.get('jikan', {})
    for link_type in ['external', 'streaming']:
        for link in jikan.get('data', {}).get(link_type, []):
            if link.get('name') and link.get('url'):
                key = normalize_string_for_comparison(link['name']).replace(' ', '')
                if key not in external_links:
                    external_links[key] = link['url']

    # AnimSchedule websites
    animeschedule = sources.get('animeschedule', {})
    websites = animeschedule.get('websites', {})
    if isinstance(websites, dict):
        for name, url in websites.items():
            if name and url and isinstance(url, str):
                key = normalize_string_for_comparison(name).replace(' ', '')
                if key not in external_links:
                    # Add https:// if not present
                    if not url.startswith('http'):
                        url = f"https://{url}"
                    external_links[key] = url

    # AniList external links
    anilist = sources.get('anilist', {})
    for link in anilist.get('externalLinks', []):
        if link.get('site') and link.get('url'):
            key = normalize_string_for_comparison(link['site']).replace(' ', '')
            if key not in external_links:
                external_links[key] = link['url']

    return external_links


def extract_synopsis_with_hierarchy(sources: Dict[str, Dict]) -> Optional[str]:
    """
    Extract synopsis using 6-level source hierarchy with cleanup.

    Priority: AniDB → Jikan → AnimSchedule → Kitsu → Anime-Planet → AniList
    """

    # 1. AniDB (highest priority)
    anidb = sources.get('anidb', {})
    if anidb.get('description'):
        synopsis = anidb['description']
        # Remove hyperlinks and clean markup
        synopsis = re.sub(r'\[([^\]]+)\]', r'\1', synopsis)  # [link text] → link text
        synopsis = re.sub(r'\[i\]([^\[]*)\[/i\]', r'\1', synopsis)  # [i]text[/i] → text
        if synopsis.strip():
            return synopsis.strip()

    # 2. Jikan (high priority)
    jikan = sources.get('jikan', {})
    if jikan.get('data', {}).get('synopsis'):
        synopsis = jikan['data']['synopsis']
        if synopsis.strip():
            return synopsis.strip()

    # 3. AnimSchedule (medium priority)
    animeschedule = sources.get('animeschedule', {})
    if animeschedule.get('description'):
        synopsis = animeschedule['description']
        # Convert HTML entities and remove <br> tags
        synopsis = synopsis.replace('&#39;', "'").replace('&#34;', '"')
        synopsis = re.sub(r'<br\s*/?>', ' ', synopsis)
        if synopsis.strip():
            return synopsis.strip()

    # 4. Kitsu (medium priority)
    kitsu = sources.get('kitsu', {})
    kitsu_desc = None
    if kitsu.get('data', {}).get('attributes'):
        attrs = kitsu['data']['attributes']
        kitsu_desc = attrs.get('synopsis') or attrs.get('description')
    if kitsu_desc and kitsu_desc.strip():
        return kitsu_desc.strip()

    # 5. Anime-Planet (lower priority) - now flattened to top level
    anime_planet = sources.get('anime_planet', {})
    if anime_planet.get('description'):
        synopsis = anime_planet['description']
        if synopsis.strip():
            return synopsis.strip()

    # 6. AniList (fallback)
    anilist = sources.get('anilist', {})
    if anilist.get('description'):
        synopsis = anilist['description']
        # Remove <br> tags
        synopsis = re.sub(r'<br\s*/?>', ' ', synopsis)
        if synopsis.strip():
            return synopsis.strip()

    return None


def extract_trailers_with_deduplication(sources: Dict[str, Dict]) -> List[Dict]:
    """
    Extract trailers from multiple sources with YouTube URL deduplication.
    """
    trailers = []
    seen_urls = set()

    # Jikan trailers
    jikan = sources.get('jikan', {})
    trailer = jikan.get('data', {}).get('trailer')
    if trailer and trailer.get('url'):
        url = trailer['url']
        if url not in seen_urls:
            trailers.append({
                'url': url,
                'thumbnail_url': trailer.get('images', {}).get('maximum_image_url'),
                'title': trailer.get('title', 'Official Trailer')
            })
            seen_urls.add(url)

    # AniList trailers
    anilist = sources.get('anilist', {})
    anilist_trailer = anilist.get('trailer')
    if anilist_trailer and anilist_trailer.get('id'):
        url = f"https://www.youtube.com/watch?v={anilist_trailer['id']}"
        if url not in seen_urls:
            trailers.append({
                'url': url,
                'thumbnail_url': anilist_trailer.get('thumbnail'),
                'title': f"{anilist_trailer.get('site', 'YouTube')} Trailer"
            })
            seen_urls.add(url)

    # Kitsu trailers
    kitsu = sources.get('kitsu', {})
    youtube_id = kitsu.get('data', {}).get('attributes', {}).get('youtubeVideoId')
    if youtube_id:
        url = f"https://www.youtube.com/watch?v={youtube_id}"
        if url not in seen_urls:
            trailers.append({
                'url': url,
                'thumbnail_url': None,
                'title': 'Official Trailer'
            })
            seen_urls.add(url)

    return trailers


def extract_genres_from_sources(sources: Dict[str, Dict]) -> List[str]:
    """Extract genres from Jikan + AnimSchedule + Anime-Planet + AniList."""
    all_genres = []

    # Jikan genres
    jikan = sources.get('jikan', {})
    for genre in jikan.get('data', {}).get('genres', []):
        if genre.get('name'):
            all_genres.append(genre['name'])

    # AnimSchedule genres
    animeschedule = sources.get('animeschedule', {})
    for genre in animeschedule.get('genres', []):
        if isinstance(genre, str):
            all_genres.append(genre)
        elif isinstance(genre, dict) and genre.get('name'):
            all_genres.append(genre['name'])

    # Anime-Planet genres (now flattened to top level)
    anime_planet = sources.get('anime_planet', {})
    for genre in anime_planet.get('genres', []):
        if isinstance(genre, str):
            all_genres.append(genre)
        elif isinstance(genre, dict) and genre.get('name'):
            all_genres.append(genre['name'])

    # AniList genres
    anilist = sources.get('anilist', {})
    for genre in anilist.get('genres', []):
        if isinstance(genre, str):
            all_genres.append(genre)

    return all_genres


def extract_themes_from_sources(sources: Dict[str, Dict]) -> List[Dict]:
    """Extract themes from all sources with proper priority ordering."""
    all_themes = []

    # Jikan themes (highest priority)
    jikan = sources.get('jikan', {})
    for theme in jikan.get('data', {}).get('themes', []):
        if theme.get('name'):
            all_themes.append({
                'name': theme['name'],
                'description': None  # Jikan themes typically don't have descriptions
            })

    # Kitsu categories (second priority)
    kitsu = sources.get('kitsu', {})
    categories = kitsu.get('categories', [])
    for category in categories:
        title = category.get('attributes', {}).get('title')
        if title:
            all_themes.append({
                'name': title,
                'description': category.get('attributes', {}).get('description')
            })

    # AniList tags (third priority)
    anilist = sources.get('anilist', {})
    for tag in anilist.get('tags', []):
        if tag.get('name'):
            all_themes.append({
                'name': tag['name'],
                'description': tag.get('description')
            })

    # AniDB tags (fourth priority)
    anidb = sources.get('anidb', {})
    for tag in anidb.get('tags', []):
        if isinstance(tag, dict) and tag.get('name'):
            all_themes.append({
                'name': tag['name'],
                'description': tag.get('description')
            })

    return all_themes


def extract_synonyms_from_sources(sources: Dict[str, Dict]) -> List[str]:
    """Extract synonyms from all sources following priority order."""
    all_synonyms = []

    # Jikan synonyms
    jikan = sources.get('jikan', {})
    for synonym in jikan.get('data', {}).get('titles', []):
        if synonym.get('type') == 'Synonym' and synonym.get('title'):
            all_synonyms.append(synonym['title'])

    # AniList synonyms
    anilist = sources.get('anilist', {})
    for synonym in anilist.get('synonyms', []):
        if synonym:
            all_synonyms.append(synonym)

    # Kitsu abbreviated titles
    kitsu = sources.get('kitsu', {})
    for title in kitsu.get('data', {}).get('attributes', {}).get('abbreviatedTitles', []):
        if title:
            all_synonyms.append(title)

    # AnimSchedule synonyms
    animeschedule = sources.get('animeschedule', {})
    for synonym in animeschedule.get('names', {}).get('synonyms', []):
        if synonym:
            all_synonyms.append(synonym)

    # AniDB synonyms
    anidb = sources.get('anidb', {})
    anidb_titles = anidb.get('titles', {})
    if isinstance(anidb_titles, dict):
        for synonym in anidb_titles.get('synonyms', []):
            if synonym:
                all_synonyms.append(synonym)

    return all_synonyms


def extract_tags_from_sources(sources: Dict[str, Dict]) -> List[str]:
    """Extract tags from external sources (currently none have simple string tags)."""
    # Tags are only available from offline database as simple strings
    # External sources have 'tags' that go into themes field with descriptions
    return []


def parse_theme_song_string(theme_string: str) -> Dict[str, Optional[str]]:
    """Parse Jikan theme song string into ThemeSong components."""
    import re

    # Pattern: "{num}: \"{title}\" by {artist} (eps {episodes})"
    # More robust pattern to handle artist names with Japanese text in parentheses
    pattern = r'\d+:\s*"([^"]+)"\s+by\s+(.+?)(?:\s+\(eps\s+([^)]+)\))?$'

    match = re.match(pattern, theme_string.strip())
    if match:
        title = match.group(1).strip()
        artist_raw = match.group(2).strip()
        episodes = match.group(3).strip() if match.group(3) else None

        # Keep the full artist name including Japanese names in parentheses
        artist = artist_raw.strip()

        return {
            'title': title,
            'artist': artist if artist else None,
            'episodes': episodes
        }

    # Fallback: try to extract title and artist without episode info
    fallback_pattern = r'\d+:\s*"([^"]+)"\s+by\s+(.+)$'
    fallback_match = re.match(fallback_pattern, theme_string.strip())
    if fallback_match:
        title = fallback_match.group(1).strip()
        artist_raw = fallback_match.group(2).strip()

        # Keep the full artist name including Japanese names in parentheses
        artist = artist_raw.strip()

        return {
            'title': title,
            'artist': artist if artist else None,
            'episodes': None
        }

    # Final fallback: try to extract at least the title
    if '"' in theme_string:
        title_match = re.search(r'"([^"]+)"', theme_string)
        if title_match:
            return {
                'title': title_match.group(1),
                'artist': None,
                'episodes': None
            }

    return {
        'title': theme_string.strip(),
        'artist': None,
        'episodes': None
    }


def extract_opening_themes(sources: Dict[str, Dict]) -> List[Dict[str, Optional[str]]]:
    """Extract opening themes from Jikan theme.openings array."""
    opening_themes = []

    jikan = sources.get('jikan', {})
    openings = jikan.get('data', {}).get('theme', {}).get('openings', [])

    for opening in openings:
        if opening:
            theme_song = parse_theme_song_string(opening)
            opening_themes.append(theme_song)

    return opening_themes


def extract_ending_themes(sources: Dict[str, Dict]) -> List[Dict[str, Optional[str]]]:
    """Extract ending themes from Jikan theme.endings array."""
    ending_themes = []

    jikan = sources.get('jikan', {})
    endings = jikan.get('data', {}).get('theme', {}).get('endings', [])

    for ending in endings:
        if ending:
            theme_song = parse_theme_song_string(ending)
            ending_themes.append(theme_song)

    return ending_themes


def cross_validate_with_offline(offline_data: Dict, sources: Dict[str, Dict], field: str) -> Any:
    """
    Cross-validate field with offline database using source hierarchy.

    Hierarchy: Offline DB → Jikan → AniList → Kitsu/AnimSchedule
    """
    offline_value = None

    # Extract offline value based on field
    if field == 'episodes':
        offline_value = offline_data.get('episodes')
    elif field == 'status':
        offline_value = offline_data.get('status')
    elif field == 'year':
        offline_value = offline_data.get('animeSeason', {}).get('year')
    elif field == 'season':
        offline_value = offline_data.get('animeSeason', {}).get('season')
    elif field == 'duration':
        duration_obj = offline_data.get('duration')
        if isinstance(duration_obj, dict):
            offline_value = duration_obj.get('value')
        else:
            offline_value = duration_obj

    # If offline value exists and is valid, use it
    if offline_value is not None:
        return offline_value

    # Otherwise follow source hierarchy
    source_hierarchy = ['jikan', 'anilist', 'kitsu', 'animeschedule']

    for source_name in source_hierarchy:
        source_data = sources.get(source_name, {})
        value = None

        if source_name == 'jikan':
            data = source_data.get('data', {})
            if field == 'episodes':
                value = data.get('episodes')
            elif field == 'status':
                value = data.get('status')
            elif field == 'year':
                aired = data.get('aired', {})
                if aired.get('from'):
                    try:
                        year = datetime.fromisoformat(aired['from'].replace('Z', '+00:00')).year
                        value = year
                    except:
                        pass
            elif field == 'season':
                value = data.get('season')
            elif field == 'duration':
                value = data.get('duration')  # Duration in seconds

        elif source_name == 'anilist':
            media = source_data.get('data', {}).get('Media', {})
            if field == 'episodes':
                value = media.get('episodes')
            elif field == 'status':
                value = media.get('status')
            elif field == 'year':
                start_date = media.get('startDate')
                if start_date and start_date.get('year'):
                    value = start_date['year']
            elif field == 'season':
                value = media.get('season')
            elif field == 'duration':
                value = media.get('duration') * 60 if media.get('duration') else None  # Convert minutes to seconds

        if value is not None:
            return value

    return offline_value


def process_stage1_metadata(current_anime_file: str, temp_dir: str) -> Dict[str, Any]:
    """
    Main function to process Stage 1 metadata extraction with comprehensive multi-source integration.
    """
    print("=== STAGE 1 METADATA EXTRACTION (COMPREHENSIVE) ===\n")

    # Load offline database as foundation
    print("Loading offline database as foundation...")
    offline_data = load_offline_database(current_anime_file)
    if not offline_data:
        print("Error: Could not load offline database")
        return {}

    # Load all external source data
    print("Loading external source data...")
    sources = load_source_data(temp_dir)

    # Start building output with offline foundation
    output = {}

    # =====================================================================
    # SCALAR FIELDS (alphabetical)
    # =====================================================================

    # Background from Jikan
    jikan_data = sources.get('jikan', {}).get('data', {})
    output['background'] = jikan_data.get('background')

    # Episodes (cross-validated)
    output['episodes'] = cross_validate_with_offline(offline_data, sources, 'episodes')

    # Month from AnimSchedule
    animeschedule_data = sources.get('animeschedule', {})
    output['month'] = animeschedule_data.get('month')

    # NSFW from Kitsu
    kitsu_data = sources.get('kitsu', {}).get('anime', {}).get('attributes', {})
    output['nsfw'] = kitsu_data.get('nsfw')

    # Rating from Jikan
    output['rating'] = jikan_data.get('rating')

    # Season (cross-validated)
    output['season'] = cross_validate_with_offline(offline_data, sources, 'season')

    # Source material from Jikan, fallback to AniList
    source_material = jikan_data.get('source')
    if not source_material:
        anilist_media = sources.get('anilist', {})
        source_material = anilist_media.get('source')
    # Convert to uppercase for enum compliance
    output['source_material'] = source_material.upper() if source_material else None

    # Status (cross-validated)
    output['status'] = cross_validate_with_offline(offline_data, sources, 'status')

    # Synopsis with 6-level hierarchy
    output['synopsis'] = extract_synopsis_with_hierarchy(sources)

    # Title from offline database
    output['title'] = offline_data.get('title')

    # English and Japanese titles from Jikan
    titles = jikan_data.get('titles', [])
    output['title_english'] = None
    output['title_japanese'] = None

    for title in titles:
        if title.get('type') == 'English' and not output['title_english']:
            output['title_english'] = title.get('title')
        elif title.get('type') == 'Japanese' and not output['title_japanese']:
            output['title_japanese'] = title.get('title')

    # Fallback to Anime-Planet for Japanese title if not found in Jikan
    if not output['title_japanese']:
        anime_planet = sources.get('anime_planet', {})
        if anime_planet.get('title_japanese'):
            output['title_japanese'] = anime_planet['title_japanese']

    # Type from offline database
    output['type'] = offline_data.get('type')

    # Year (cross-validated)
    output['year'] = cross_validate_with_offline(offline_data, sources, 'year')

    # =====================================================================
    # ARRAY FIELDS (alphabetical)
    # =====================================================================

    # Content warnings (inferred from rating)
    content_warnings = []
    if output.get('rating'):
        rating = output['rating'].lower()
        if 'violence' in rating:
            content_warnings.append('Violence')
        if 'profanity' in rating:
            content_warnings.append('Strong Language')
        if 'nudity' in rating:
            content_warnings.append('Nudity')
    output['content_warnings'] = content_warnings

    # Demographics from Jikan only
    demographics = []
    for demo in jikan_data.get('demographics', []):
        if demo.get('name'):
            demographics.append(demo['name'])
    output['demographics'] = demographics

    # Ending themes from Jikan
    output['ending_themes'] = extract_ending_themes(sources)

    # Genres with multi-source integration and deduplication
    offline_genres = []  # No genres in offline database typically
    external_genres = extract_genres_from_sources(sources)
    output['genres'] = deduplicate_array_field(offline_genres, external_genres)

    # Opening themes from Jikan
    output['opening_themes'] = extract_opening_themes(sources)

    # Synonyms with multi-source integration and deduplication
    offline_synonyms = offline_data.get('synonyms', [])
    external_synonyms = extract_synonyms_from_sources(sources)

    # Exclude main titles from synonyms
    main_titles = [
        output.get('title'),
        output.get('title_english'),
        output.get('title_japanese')
    ]
    main_titles = [t for t in main_titles if t]
    main_titles_normalized = {normalize_string_for_comparison(t) for t in main_titles}

    # Filter out main titles from external synonyms
    filtered_external_synonyms = []
    for synonym in external_synonyms:
        if normalize_string_for_comparison(synonym) not in main_titles_normalized:
            filtered_external_synonyms.append(synonym)

    output['synonyms'] = deduplicate_array_field(offline_synonyms, filtered_external_synonyms)

    # Tags from offline database and all sources
    offline_tags = offline_data.get('tags', [])
    external_tags = extract_tags_from_sources(sources)
    output['tags'] = deduplicate_array_field(offline_tags, external_tags)

    # Themes with intelligent multi-source merging
    offline_themes = []  # No themes in offline database typically
    external_themes = extract_themes_from_sources(sources)
    existing_genres_set = {normalize_string_for_comparison(g) for g in output['genres']}
    output['themes'] = merge_themes_intelligently(offline_themes, external_themes, existing_genres_set)

    # Trailers with deduplication
    output['trailers'] = extract_trailers_with_deduplication(sources)

    # =====================================================================
    # OBJECT/DICT FIELDS (alphabetical)
    # =====================================================================

    # Aired dates from Jikan
    aired = jikan_data.get('aired', {})
    if aired:
        output['aired_dates'] = {
            'from': aired.get('from'),
            'to': aired.get('to'),
            'string': aired.get('string')
        }
    else:
        output['aired_dates'] = None

    # Broadcast from Jikan
    broadcast = jikan_data.get('broadcast', {})
    if broadcast:
        output['broadcast'] = {
            'day': broadcast.get('day'),
            'time': broadcast.get('time'),
            'timezone': broadcast.get('timezone')
        }
    else:
        output['broadcast'] = None

    # Broadcast schedule from AnimSchedule
    if any(animeschedule_data.get(k) for k in ['jpnTime', 'subTime', 'dubTime']):
        output['broadcast_schedule'] = {
            'jpn_time': animeschedule_data.get('jpnTime'),
            'sub_time': animeschedule_data.get('subTime'),
            'dub_time': animeschedule_data.get('dubTime')
        }
    else:
        output['broadcast_schedule'] = None

    # Delay information from AnimSchedule
    delay_fields = ['delayedTimetable', 'delayedFrom', 'delayedUntil', 'delayedDesc']
    if any(animeschedule_data.get(k) for k in delay_fields):
        output['delay_information'] = {
            'delayed_timetable': animeschedule_data.get('delayedTimetable', False),
            'delayed_from': animeschedule_data.get('delayedFrom'),
            'delayed_until': animeschedule_data.get('delayedUntil'),
            'delay_reason': animeschedule_data.get('delayedDesc')
        }
    else:
        output['delay_information'] = None

    # Duration (cross-validated)
    output['duration'] = cross_validate_with_offline(offline_data, sources, 'duration')


    # External links with normalization and deduplication
    output['external_links'] = normalize_external_links(sources)

    # Images organized by type
    output['images'] = organize_images_by_type(sources)

    # Premiere dates from AnimSchedule
    premiere_fields = ['premier', 'subPremier', 'dubPremier']
    if any(animeschedule_data.get(k) for k in premiere_fields):
        output['premiere_dates'] = {
            'original': animeschedule_data.get('premier'),
            'sub': animeschedule_data.get('subPremier'),
            'dub': animeschedule_data.get('dubPremier')
        }
    else:
        output['premiere_dates'] = None

    # Clean up None values in nested objects
    for key, value in output.items():
        if isinstance(value, dict) and all(v is None for v in value.values()):
            output[key] = None

    return output


def auto_detect_temp_dir():
    """Auto-detect the temp directory based on available directories."""
    temp_base = 'temp'
    if not os.path.exists(temp_base):
        print(f"Error: {temp_base} directory not found")
        sys.exit(1)

    # Look for directories in temp/
    temp_dirs = [d for d in os.listdir(temp_base) if os.path.isdir(os.path.join(temp_base, d))]

    if not temp_dirs:
        print(f"Error: No anime directories found in {temp_base}/")
        sys.exit(1)

    if len(temp_dirs) == 1:
        detected_dir = os.path.join(temp_base, temp_dirs[0])
        print(f"Auto-detected temp directory: {detected_dir}")
        return detected_dir
    else:
        print(f"Multiple temp directories found: {temp_dirs}")
        print("Please specify which one to use with --temp-dir argument")
        sys.exit(1)


def auto_detect_current_anime_file(temp_dir: str):
    """Auto-detect current anime file following consistent directory structure."""
    # First, check if current_anime.json exists in the provided temp_dir
    current_anime_in_dir = os.path.join(temp_dir, 'current_anime.json')
    if os.path.exists(current_anime_in_dir):
        print(f"Found current anime file: {current_anime_in_dir}")
        return current_anime_in_dir

    # Fallback: Look for current_anime.json files in agent directories under temp/
    temp_base = 'temp'
    if os.path.exists(temp_base):
        agent_dirs = []
        for item in os.listdir(temp_base):
            item_path = os.path.join(temp_base, item)
            if os.path.isdir(item_path):
                current_anime_file = os.path.join(item_path, 'current_anime.json')
                if os.path.exists(current_anime_file):
                    agent_dirs.append((item, current_anime_file))

        if agent_dirs:
            if len(agent_dirs) == 1:
                dir_name, detected_file = agent_dirs[0]
                print(f"Auto-detected current anime file: {detected_file}")
                return detected_file
            else:
                print(f"Multiple agent directories with current_anime.json found: {[d[0] for d in agent_dirs]}")
                print("Please specify which one to use with --current-anime argument")
                sys.exit(1)

    print(f"Error: No current anime file found. Looked in {temp_dir}/ and agent directories in temp/")
    sys.exit(1)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Process Stage 1: Metadata extraction with multi-source integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python process_stage1_metadata.py One_agent1                         # Process One_agent1 directory
  python process_stage1_metadata.py Dandadan_agent1                    # Process Dandadan_agent1 directory
  python process_stage1_metadata.py One_agent1 --temp-dir custom_temp # Use custom temp directory
        """
    )
    parser.add_argument(
        "agent_id",
        nargs="?",
        help="Agent directory name to process (e.g., One_agent1, Dandadan_agent1)"
    )
    parser.add_argument(
        "--temp-dir",
        default="temp",
        help="Temporary directory path (default: temp)"
    )
    parser.add_argument(
        "--current-anime",
        type=str,
        help="Override current anime JSON file path (optional, for backward compatibility)"
    )

    args = parser.parse_args()

    # Determine temp_dir and construct full path (relative to project root)
    if args.agent_id:
        # New pattern: agent_id + temp_dir
        temp_base = Path(args.temp_dir)
        if not temp_base.is_absolute():
            temp_base = PROJECT_ROOT / temp_base
        temp_dir = str(temp_base / args.agent_id)
        if not os.path.exists(temp_dir):
            print(f"Error: Directory '{temp_dir}' does not exist")
            sys.exit(1)
    elif args.current_anime:
        # Legacy pattern: derive from file path
        temp_dir = os.path.dirname(args.current_anime)
        print(f"Derived temp directory from file path: {temp_dir}")
    else:
        # Fallback: auto-detect
        temp_dir = auto_detect_temp_dir()

    # Determine current_anime_file
    if args.current_anime:
        current_anime_file = args.current_anime
        if not os.path.exists(current_anime_file):
            print(f"Error: Specified current anime file '{current_anime_file}' does not exist")
            sys.exit(1)
    else:
        current_anime_file = auto_detect_current_anime_file(temp_dir)

    print(f"Processing with:")
    print(f"  Current anime file: {current_anime_file}")
    print(f"  Temp directory: {temp_dir}")

    # Process Stage 1 metadata
    result = process_stage1_metadata(current_anime_file, temp_dir)

    if result:
        # Save to output file
        output_file = f'{temp_dir}/stage1_metadata.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"\nStage 1 metadata extraction complete!")
        print(f"Output saved: {output_file}")
        print(f"Summary:")
        print(f"   - Title: {result.get('title', 'N/A')}")
        print(f"   - Genres: {len(result.get('genres', []))} entries")
        print(f"   - Synonyms: {len(result.get('synonyms', []))} entries")
        print(f"   - Tags: {len(result.get('tags', []))} entries")
        print(f"   - Themes: {len(result.get('themes', []))} entries")
        print(f"   - Opening themes: {len(result.get('opening_themes', []))} entries")
        print(f"   - Ending themes: {len(result.get('ending_themes', []))} entries")
        print(f"   - Images: {sum(len(urls) for urls in result.get('images', {}).values())} URLs")
        print(f"   - External links: {len(result.get('external_links', {}))} links")
        print(f"   - Synopsis: {'Yes' if result.get('synopsis') else 'No'}")
    else:
        print("Stage 1 metadata extraction failed")
        sys.exit(1)


if __name__ == "__main__":
    main()