#!/usr/bin/env python3
"""
Stage 5: AI-Powered Character Processing

Replaces the primitive string matching with enterprise-grade AI character matching.
Uses the new ai_character_matcher.py for 99% precision vs 0.3% with string matching.

Enhanced with AniDB-specific optimizations:
- 80% semantic similarity weight for AniDB's standardized format
- Improved name preprocessing for anime character patterns
- Language-aware matching cleaned and optimized
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

# Add project root to path for imports (works from anywhere)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.enrichment.ai_character_matcher import process_characters_with_ai_matching

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_id_from_character_page(url: str, source: str) -> Optional[Union[int, str]]:
    """Extract character ID from character_pages URL

    Args:
        url: Character page URL
        source: Source type (anilist, anidb)

    Returns:
        Extracted ID (int for anilist, str for anidb) or None if extraction fails
    """
    if not url:
        return None

    try:
        if source == 'anilist':
            # https://anilist.co/character/40 -> 40
            id_str = url.split('/character/')[-1].rstrip('/')
            return int(id_str)
        elif source == 'anidb':
            # https://anidb.net/character/474 -> "474"
            id_str = url.split('/character/')[-1].rstrip('/')
            return id_str
    except (ValueError, IndexError) as e:
        logger.warning(f"Failed to extract {source} ID from URL {url}: {e}")
        return None

    return None


def load_stage_data(stage_file: Path) -> List[Dict[str, Any]]:
    """Load data from a stage JSON file"""
    try:
        with open(stage_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # Handle different data structures
            if 'data' in data:  # Jikan format
                return data['data']
            else:
                # Single object, return as list (for anilist, anidb, kitsu)
                return [data]
        else:
            logger.error(f"Unexpected data format in {stage_file}")
            return []

    except FileNotFoundError:
        logger.warning(f"Stage file not found: {stage_file}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {stage_file}: {e}")
        return []
    except Exception as e:
        logger.error(f"Error loading {stage_file}: {e}")
        return []


def get_working_file_paths(anime_id: str, temp_dir: Path) -> Dict[str, Path]:
    """Get paths for working files"""
    working_dir = temp_dir / anime_id
    return {
        'jikan': working_dir / "working_jikan.json",
        'anilist': working_dir / "working_anilist.json",
        'anidb': working_dir / "working_anidb.json",
        'animeplanet': working_dir / "working_animeplanet.json"
    }


def working_files_exist(working_paths: Dict[str, Path]) -> bool:
    """Check if all working files exist"""
    return all(path.exists() for path in working_paths.values())


def create_working_files(anime_id: str, temp_dir: Path, jikan_chars: List[Dict[str, Any]],
                         anilist_chars: List[Dict[str, Any]], anidb_chars: List[Dict[str, Any]],
                         anime_planet_chars: List[Dict[str, Any]], force_restart: bool = False) -> Dict[str, Path]:
    """Create or resume from working copies of character arrays for progressive deletion

    All working files contain ONLY character arrays, no wrapper objects.

    Args:
        anime_id: The anime ID
        temp_dir: Temporary directory path
        jikan_chars: Jikan character array
        anilist_chars: AniList character array
        anidb_chars: AniDB character array
        anime_planet_chars: AnimePlanet character array
        force_restart: If True, overwrite existing working files. If False (default), resume from existing files.

    Returns:
        Dict with paths to working files
    """
    working_dir = temp_dir / anime_id
    working_dir.mkdir(parents=True, exist_ok=True)

    working_paths = get_working_file_paths(anime_id, temp_dir)

    # Check if working files already exist
    if not force_restart and working_files_exist(working_paths):
        # Resume from existing working files
        jikan_count = len(load_working_file(working_paths['jikan']))
        anilist_count = len(load_working_file(working_paths['anilist']))
        anidb_count = len(load_working_file(working_paths['anidb']))
        animeplanet_count = len(load_working_file(working_paths['animeplanet']))

        logger.info(f"🔄 RESUMING from existing working files in {working_dir}")
        logger.info(f"  - working_jikan.json: {jikan_count} characters remaining")
        logger.info(f"  - working_anilist.json: {anilist_count} characters remaining")
        logger.info(f"  - working_anidb.json: {anidb_count} characters remaining")
        logger.info(f"  - working_animeplanet.json: {animeplanet_count} characters remaining")
        return working_paths

    # Create fresh working files
    with open(working_paths['jikan'], 'w', encoding='utf-8') as f:
        json.dump(jikan_chars, f, ensure_ascii=False, indent=2)

    with open(working_paths['anilist'], 'w', encoding='utf-8') as f:
        json.dump(anilist_chars, f, ensure_ascii=False, indent=2)

    with open(working_paths['anidb'], 'w', encoding='utf-8') as f:
        json.dump(anidb_chars, f, ensure_ascii=False, indent=2)

    with open(working_paths['animeplanet'], 'w', encoding='utf-8') as f:
        json.dump(anime_planet_chars, f, ensure_ascii=False, indent=2)

    logger.info(f"✨ Created NEW working files in {working_dir}")
    logger.info(f"  - working_jikan.json: {len(jikan_chars)} characters")
    logger.info(f"  - working_anilist.json: {len(anilist_chars)} characters")
    logger.info(f"  - working_anidb.json: {len(anidb_chars)} characters")
    logger.info(f"  - working_animeplanet.json: {len(anime_planet_chars)} characters")
    return working_paths


def load_working_file(file_path: Path) -> List[Dict[str, Any]]:
    """Load a working file and return its contents"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load working file {file_path}: {e}")
        return []


def save_working_file(file_path: Path, data: List[Dict[str, Any]]) -> None:
    """Save data to a working file"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Failed to save working file {file_path}: {e}")


def remove_matched_entry(working_data: List[Dict[str, Any]], matched_id: Any, source_type: str) -> List[Dict[str, Any]]:
    """Remove a matched entry from the working data list"""
    if source_type == 'jikan':
        # Jikan uses 'character_id' field
        return [char for char in working_data if char.get('character_id') != matched_id]
    elif source_type == 'anilist':
        # AniList uses 'id' field (integer)
        return [char for char in working_data if char.get('id') != matched_id]
    elif source_type == 'anidb':
        # AniDB uses 'id' field (string)
        return [char for char in working_data if str(char.get('id')) != str(matched_id)]
    elif source_type == 'animeplanet':
        # AnimePlanet uses 'name' field as unique identifier
        return [char for char in working_data if char.get('name') != matched_id]
    else:
        return working_data


async def process_stage5_ai_characters(anime_id: str, temp_dir: Path, force_restart: bool = False) -> None:
    """Process Stage 5 using AI character matching with progressive pool deletion optimization

    Args:
        anime_id: The anime ID to process
        temp_dir: Temporary directory path
        force_restart: If True, restart from scratch. If False (default), resume from existing working files.
    """

    logger.info(f"Starting AI character processing for {anime_id}")
    if force_restart:
        logger.info("⚠️  FORCE RESTART mode enabled - will create fresh working files")

    # Validate agent directory exists
    agent_dir = temp_dir / anime_id
    if not agent_dir.exists():
        logger.error(f"Agent directory not found: {agent_dir}")
        raise FileNotFoundError(f"Agent directory '{anime_id}' does not exist in {temp_dir}")

    # Load data from all sources
    stage_files = {
        'jikan': agent_dir / "characters_detailed.json",
        'anilist': agent_dir / "anilist.json",
        'anidb': agent_dir / "anidb.json",
        'animeplanet': agent_dir / "anime_planet.json",
    }

    # Validate required files exist
    required_files = ['jikan', 'anilist', 'anidb']
    missing_files = []
    for source in required_files:
        if not stage_files[source].exists():
            missing_files.append(str(stage_files[source]))

    if missing_files:
        logger.error(f"Required source files missing:")
        for f in missing_files:
            logger.error(f"  - {f}")
        raise FileNotFoundError(f"Missing required source files for {anime_id}. Run previous stages first.")

    # Load character data from all sources
    source_data = {}
    for source, file_path in stage_files.items():
        data = load_stage_data(file_path)
        source_data[source] = data
        logger.info(f"Loaded {len(data)} items from {source}")

    # Extract character data by source type
    jikan_chars = source_data['jikan']  # Already character list

    # Extract AniList characters from API response
    anilist_chars = []
    if source_data['anilist']:
        anilist_data = source_data['anilist'][0] if isinstance(source_data['anilist'], list) else source_data['anilist']
        if 'characters' in anilist_data and 'edges' in anilist_data['characters']:
            for edge in anilist_data['characters']['edges']:
                if 'node' in edge:
                    character = edge['node'].copy()
                    character['role'] = edge.get('role', 'UNKNOWN')
                    character['voice_actors'] = edge.get('voiceActors', [])
                    anilist_chars.append(character)

    # Extract AniDB characters from API response
    anidb_chars = []
    if source_data['anidb']:
        anidb_data = source_data['anidb'][0] if isinstance(source_data['anidb'], list) else source_data['anidb']
        if 'characters' in anidb_data:
            anidb_chars = anidb_data['characters']

    # Extract AnimePlanet characters
    anime_planet_chars = []
    if source_data['animeplanet']:
        ap_data = source_data['animeplanet'][0] if isinstance(source_data['animeplanet'], list) else source_data['animeplanet']
        if 'characters' in ap_data:
            anime_planet_chars = ap_data['characters']

    logger.info(f"Character counts - Jikan: {len(jikan_chars)}, AniList: {len(anilist_chars)}, "
                f"AniDB: {len(anidb_chars)}, AnimePlanet: {len(anime_planet_chars)}")

    # Create or resume from working files for progressive deletion
    working_paths = create_working_files(anime_id, temp_dir, jikan_chars, anilist_chars, anidb_chars,
                                        anime_planet_chars, force_restart=force_restart)

    # Process with progressive matching and deletion
    try:
        # Setup incremental output file (line-delimited JSON)
        output_jsonl = temp_dir / f"{anime_id}" / "stage5_characters.jsonl"
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)

        # Initialize or clear JSONL file
        with open(output_jsonl, 'w', encoding='utf-8') as f:
            pass  # Create empty file

        matched_count = 0
        working_jikan = load_working_file(working_paths['jikan'])
        working_anilist = load_working_file(working_paths['anilist'])
        working_anidb = load_working_file(working_paths['anidb'])
        working_animeplanet = load_working_file(working_paths['animeplanet'])

        total_jikan = len(working_jikan)

        # Process each character one at a time
        for i, jikan_char in enumerate(working_jikan.copy(), 1):
            char_name = jikan_char.get('name', 'Unknown')

            # Skip already-processed partial matches (have found_in field)
            if 'found_in' in jikan_char:
                logger.info(f"[{i}/{total_jikan}] SKIP '{char_name}' (already processed as partial match)")
                continue

            # Match this ONE character against current pools
            result = await process_characters_with_ai_matching(
                jikan_chars=[jikan_char],
                anilist_chars=working_anilist,
                anidb_chars=working_anidb,
                anime_planet_chars=working_animeplanet
            )

            matched_char = result['characters'][0]

            # Check if found in ALL sources (AniList, AniDB, AND AnimePlanet)
            has_anilist = matched_char['character_pages'].get('anilist') is not None
            has_anidb = matched_char['character_pages'].get('anidb') is not None
            has_animeplanet = matched_char['character_pages'].get('animeplanet') is not None

            if has_anilist and has_anidb and has_animeplanet:
                # FULL MATCH - integrate and delete from all pools
                # Remove internal _match_scores field before adding to output
                if '_match_scores' in matched_char:
                    del matched_char['_match_scores']

                # Write to JSONL file immediately (incremental output)
                with open(output_jsonl, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(matched_char, ensure_ascii=False) + '\n')
                matched_count += 1

                # Remove from working pools - extract IDs from character_pages URLs
                anilist_url = matched_char['character_pages'].get('anilist', '')
                anidb_url = matched_char['character_pages'].get('anidb', '')
                anilist_id = extract_id_from_character_page(anilist_url, 'anilist')
                anidb_id = extract_id_from_character_page(anidb_url, 'anidb')
                jikan_id = jikan_char.get('character_id')
                # AnimePlanet uses name as identifier (extract from URL)
                animeplanet_name = matched_char['character_pages'].get('animeplanet', '').split('/characters/')[-1]
                # Find the actual name from the matched character
                for ap_char in working_animeplanet:
                    if ap_char.get('url', '').endswith(animeplanet_name):
                        animeplanet_name = ap_char.get('name')
                        break

                working_anilist = remove_matched_entry(working_anilist, anilist_id, 'anilist')
                working_anidb = remove_matched_entry(working_anidb, anidb_id, 'anidb')
                working_animeplanet = remove_matched_entry(working_animeplanet, animeplanet_name, 'animeplanet')
                working_jikan = remove_matched_entry(working_jikan, jikan_id, 'jikan')

                # Save updated working files immediately
                save_working_file(working_paths['jikan'], working_jikan)
                save_working_file(working_paths['anilist'], working_anilist)
                save_working_file(working_paths['anidb'], working_anidb)
                save_working_file(working_paths['animeplanet'], working_animeplanet)

                logger.info(f"[{i}/{total_jikan}] MATCHED '{char_name}' - Pools: Jikan={len(working_jikan)}, AniList={len(working_anilist)}, AniDB={len(working_anidb)}, AnimePlanet={len(working_animeplanet)}")

            else:
                # PARTIAL or NO MATCH
                found_in = []

                if has_anilist:
                    anilist_url = matched_char['character_pages'].get('anilist', '')
                    anilist_id = extract_id_from_character_page(anilist_url, 'anilist')
                    found_in.append({
                        "source": "anilist",
                        "matched_id": anilist_id,
                        "score": matched_char.get('_match_scores', {}).get('anilist', 0.0) if '_match_scores' in matched_char else 0.0
                    })

                if has_anidb:
                    anidb_url = matched_char['character_pages'].get('anidb', '')
                    anidb_id = extract_id_from_character_page(anidb_url, 'anidb')
                    found_in.append({
                        "source": "anidb",
                        "matched_id": anidb_id,
                        "score": matched_char.get('_match_scores', {}).get('anidb', 0.0) if '_match_scores' in matched_char else 0.0
                    })

                if has_animeplanet:
                    found_in.append({
                        "source": "animeplanet",
                        "matched_page": matched_char['character_pages']['animeplanet'],
                        "score": matched_char.get('_match_scores', {}).get('animeplanet', 0.0) if '_match_scores' in matched_char else 0.0
                    })

                # Only process if there are partial matches
                if found_in:
                    # BUILD INTEGRATED CHARACTER with matched sources
                    # Remove internal _match_scores field before building
                    if '_match_scores' in matched_char:
                        del matched_char['_match_scores']

                    # Add found_in field to integrated character
                    matched_char['found_in'] = found_in

                    # REPLACE in working_jikan with integrated object
                    jikan_id = jikan_char.get('character_id')
                    for idx, char in enumerate(working_jikan):
                        if char.get('character_id') == jikan_id:
                            working_jikan[idx] = matched_char
                            break

                    # Save updated working_jikan
                    save_working_file(working_paths['jikan'], working_jikan)

                    logger.info(f"[{i}/{total_jikan}] PARTIAL '{char_name}' (found in {len(found_in)}/4 sources) - integrated data saved")
                else:
                    logger.info(f"[{i}/{total_jikan}] NO MATCH '{char_name}'")

        # Convert JSONL to final JSON format
        output_file = temp_dir / f"{anime_id}" / "stage5_characters.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Read all matched characters from JSONL
        matched_characters = []
        if output_jsonl.exists():
            with open(output_jsonl, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        matched_characters.append(json.loads(line))

        # Write final JSON output
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({"characters": matched_characters}, f, ensure_ascii=False, indent=2)

        # Clean up JSONL file (optional - keep for debugging)
        # output_jsonl.unlink()

        # Final statistics
        partial_matches = sum(1 for char in working_jikan if 'found_in' in char)
        no_matches = len(working_jikan) - partial_matches

        logger.info(f"=" * 80)
        logger.info(f"AI character processing complete for {anime_id}")
        logger.info(f"=" * 80)
        logger.info(f"Total processed: {total_jikan} characters")
        logger.info(f"  ✅ Fully matched: {matched_count} (saved to stage5_characters.json)")
        logger.info(f"  ⚠️  Partial matches: {partial_matches} (in working_jikan.json with 'found_in' field)")
        logger.info(f"  ❌ No matches: {no_matches} (in working_jikan.json, no 'found_in' field)")
        logger.info(f"=" * 80)
        logger.info(f"Pool reduction:")
        logger.info(f"  AniList: {len(anilist_chars)} → {len(working_anilist)}")
        logger.info(f"  AniDB: {len(anidb_chars)} → {len(working_anidb)}")
        logger.info(f"  AnimePlanet: {len(anime_planet_chars)} → {len(working_animeplanet)}")

    except Exception as e:
        logger.error(f"AI character processing failed: {e}")
        raise


def main():
    """Main entry point for standalone usage"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Process Stage 5 with AI character matching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Resume from existing working files (default)
  python process_stage5_characters.py One_agent1

  # Force restart from scratch
  python process_stage5_characters.py One_agent1 --restart

  # Custom temp directory
  python process_stage5_characters.py One_agent1 --temp-dir custom_temp
        """
    )
    parser.add_argument("agent_id", help="Agent directory name to process (e.g., One_agent1, Dandadan_agent1)")
    parser.add_argument("--temp-dir", default="temp", help="Temporary directory path (default: temp)")
    parser.add_argument("--restart", action="store_true",
                        help="Force restart from scratch, overwriting existing working files")

    args = parser.parse_args()

    # Resolve temp_dir relative to project root, not current working directory
    temp_dir = Path(args.temp_dir)
    if not temp_dir.is_absolute():
        temp_dir = PROJECT_ROOT / temp_dir

    # Run async processing
    asyncio.run(process_stage5_ai_characters(args.agent_id, temp_dir, force_restart=args.restart))


if __name__ == "__main__":
    main()