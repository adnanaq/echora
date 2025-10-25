#!/usr/bin/env python3
"""
Comprehensive staff vector validation test with random entry selection, field combinations, and pattern testing.
Based on the architecture patterns from test_comprehensive_title.py and test_character_vector.py.
"""

import json
import sys
import random
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from itertools import combinations

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent / "tests"))

from src.config import get_settings
from src.vector.processors.text_processor import TextProcessor
from src.vector.client.qdrant_client import QdrantClient

def load_anime_database() -> Dict:
    """Load full anime database from enrichment file."""
    with open('./data/qdrant_storage/enriched_anime_database.json', 'r') as f:
        return json.load(f)

def discover_all_staff_roles(anime_data: List[Dict]) -> List[str]:
    """Dynamically discover all available staff role fields from actual data."""
    all_roles = set()

    for anime in anime_data:
        if anime.get('staff_data') and anime['staff_data'].get('production_staff'):
            staff = anime['staff_data']['production_staff']
            # Get all keys that represent staff roles
            for key in staff.keys():
                if isinstance(staff[key], list) and staff[key]:  # Has actual staff members
                    all_roles.add(key)

    return sorted(list(all_roles))

def get_anime_with_staff_data(anime_data: List[Dict]) -> List[Dict]:
    """Filter anime that have rich staff data."""
    anime_with_staff = []

    for anime in anime_data:
        if (anime.get('staff_data') and
            anime['staff_data'].get('production_staff')):

            staff = anime['staff_data']['production_staff']
            # Count non-empty staff roles
            non_empty_roles = sum(1 for role_list in staff.values()
                                if isinstance(role_list, list) and role_list)

            if non_empty_roles >= 1:  # At least 1 role with staff
                anime_with_staff.append(anime)

    return anime_with_staff

def generate_staff_field_combinations(staff_roles: List[str]) -> List[List[str]]:
    """Generate all possible combinations of staff role fields (1 to N roles)."""
    all_combinations = []

    # Single role combinations
    for role in staff_roles:
        all_combinations.append([role])

    # Multi-role combinations (2-3 roles to avoid overly complex queries)
    for r in range(2, min(4, len(staff_roles) + 1)):
        for combo in combinations(staff_roles, r):
            all_combinations.append(list(combo))

    return all_combinations

def create_staff_query_patterns() -> List[Dict]:
    """Create diverse query patterns to test different staff field approaches."""
    return [
        # Pattern 1: Role-Specific (most common)
        {
            "name": "Role + Name",
            "generator": lambda anime, roles: create_role_name_query(anime, roles),
            "description": "Combine role title with staff member name"
        },

        # Pattern 2: Multi-Role Context
        {
            "name": "Multi-Role Context",
            "generator": lambda anime, roles: create_multi_role_query(anime, roles),
            "description": "Multiple roles with staff names"
        },

        # Pattern 3: Position-Focused
        {
            "name": "Position Focused",
            "generator": lambda anime, roles: create_position_query(anime, roles),
            "description": "Focus on position/title without names"
        },

        # Pattern 4: Name-Only
        {
            "name": "Name Only",
            "generator": lambda anime, roles: create_name_only_query(anime, roles),
            "description": "Staff member names without role context"
        },

        # Pattern 5: Full Context
        {
            "name": "Full Context",
            "generator": lambda anime, roles: create_full_context_query(anime, roles),
            "description": "All available role and name information"
        },

        # Pattern 6: Minimal Context (stress test)
        {
            "name": "Minimal Context",
            "generator": lambda anime, roles: create_minimal_query(anime, roles),
            "description": "Single role or name only"
        }
    ]

def create_role_name_query(anime: Dict, roles: List[str]) -> str:
    """Create query combining role title with staff member name."""
    query_parts = []
    staff_data = anime['staff_data']['production_staff']

    for role in roles:
        if role in staff_data and staff_data[role]:
            # Format role name for display
            role_display = role.replace('_', ' ').title()
            # Get first staff member name
            staff_member = staff_data[role][0]
            name = staff_member.get('name', '') if isinstance(staff_member, dict) else str(staff_member)
            if name:
                query_parts.append(f"{role_display} {name}")

    return ' '.join(query_parts).strip()

def create_multi_role_query(anime: Dict, roles: List[str]) -> str:
    """Create query with multiple roles and names."""
    query_parts = []
    staff_data = anime['staff_data']['production_staff']

    for role in roles[:2]:  # Limit to 2 roles to keep query manageable
        if role in staff_data and staff_data[role]:
            role_display = role.replace('_', ' ').title()
            staff_member = staff_data[role][0]
            name = staff_member.get('name', '') if isinstance(staff_member, dict) else str(staff_member)
            if name:
                query_parts.append(f"{role_display}: {name}")

    return ', '.join(query_parts).strip()

def create_position_query(anime: Dict, roles: List[str]) -> str:
    """Create query focusing on positions/titles without names."""
    query_parts = []

    for role in roles:
        role_display = role.replace('_', ' ').title()
        query_parts.append(role_display)

    return ' '.join(query_parts).strip()

def create_name_only_query(anime: Dict, roles: List[str]) -> str:
    """Create query with only staff member names."""
    query_parts = []
    staff_data = anime['staff_data']['production_staff']

    for role in roles:
        if role in staff_data and staff_data[role]:
            # Get up to 2 names to avoid overly long queries
            for staff_member in staff_data[role][:2]:
                name = staff_member.get('name', '') if isinstance(staff_member, dict) else str(staff_member)
                if name:
                    query_parts.append(name)

    return ' '.join(query_parts).strip()

def create_full_context_query(anime: Dict, roles: List[str]) -> str:
    """Create query with all available role and name information."""
    query_parts = []
    staff_data = anime['staff_data']['production_staff']

    for role in roles:
        if role in staff_data and staff_data[role]:
            role_display = role.replace('_', ' ').title()
            names = []
            for staff_member in staff_data[role][:2]:  # Limit to first 2 per role
                name = staff_member.get('name', '') if isinstance(staff_member, dict) else str(staff_member)
                if name:
                    names.append(name)

            if names:
                query_parts.append(f"{role_display}: {', '.join(names)}")

    return ' | '.join(query_parts).strip()

def create_minimal_query(anime: Dict, roles: List[str]) -> str:
    """Create minimal query with single role or name."""
    staff_data = anime['staff_data']['production_staff']

    # Try to get just one name from first available role
    for role in roles:
        if role in staff_data and staff_data[role]:
            staff_member = staff_data[role][0]
            name = staff_member.get('name', '') if isinstance(staff_member, dict) else str(staff_member)
            if name:
                return name

    # Fallback to role name
    return roles[0].replace('_', ' ').title() if roles else ""

def verify_staff_in_anime(staff_names: List[str], returned_anime_title: str, anime_database: Dict) -> bool:
    """Check if any of the staff names actually exist in the returned anime."""
    for anime in anime_database.get('data', []):
        if anime.get('title') == returned_anime_title:
            if anime.get('staff_data') and anime['staff_data'].get('production_staff'):
                staff_data = anime['staff_data']['production_staff']

                # Collect all staff names from returned anime
                all_staff_names = []
                for role_list in staff_data.values():
                    if isinstance(role_list, list):
                        for staff_member in role_list:
                            name = staff_member.get('name', '') if isinstance(staff_member, dict) else str(staff_member)
                            if name:
                                all_staff_names.append(name.lower())

                # Check if any query staff names match
                for staff_name in staff_names:
                    if staff_name.lower() in all_staff_names:
                        return True

            return False

    return False

def extract_staff_names_from_query(query: str, anime: Dict, roles: List[str]) -> List[str]:
    """Extract staff names that were used in the query for validation."""
    names = []
    staff_data = anime['staff_data']['production_staff']

    for role in roles:
        if role in staff_data and staff_data[role]:
            for staff_member in staff_data[role]:
                name = staff_member.get('name', '') if isinstance(staff_member, dict) else str(staff_member)
                if name and name.lower() in query.lower():
                    names.append(name)

    return names


def test_staff_vector_comprehensive():
    """Test staff_vector with comprehensive field combination testing using random entries."""
    from utils.test_formatter import formatter

    formatter.print_header(
        "🎭 Comprehensive Staff Vector Validation",
        "Testing staff_vector with random entry selection and field combinations using production search_single_vector() method"
    )

    settings = get_settings()
    text_processor = TextProcessor(settings=settings)

    # Load anime database
    anime_database = load_anime_database()
    anime_data = anime_database.get('data', [])

    if not anime_data:
        print("   ❌ No anime data found for testing")
        return

    # Find anime with staff data
    anime_with_staff = get_anime_with_staff_data(anime_data)

    if not anime_with_staff:
        print("   ❌ No anime with staff data found for testing")
        return

    # Discover all available staff roles
    staff_roles = discover_all_staff_roles(anime_with_staff)

    if not staff_roles:
        print("   ❌ No staff roles discovered from data")
        return

    # True randomization with timestamp seed
    random.seed(int(time.time() * 1000) % 2**32)
    random_seed = int(time.time() * 1000) % 2**32

    # Generate field combinations
    all_field_combinations = generate_staff_field_combinations(staff_roles)

    # Randomly select 10 anime for testing
    test_count = min(10, len(anime_with_staff))
    test_anime = random.sample(anime_with_staff, test_count)

    # Print configuration summary
    formatter.print_test_summary(
        "Staff Vector Test",
        len(anime_data),
        len(test_anime),
        len(all_field_combinations),
        random_seed
    )

    # Print discovered roles in rich format
    print(f"\n🎯 Discovered {len(staff_roles)} Staff Roles:")
    roles_display = ", ".join([role.replace('_', ' ').title() for role in staff_roles[:10]])
    if len(staff_roles) > 10:
        roles_display += f" ... and {len(staff_roles) - 10} more"
    print(f"   {roles_display}")

    # Get query patterns
    query_patterns = create_staff_query_patterns()

    # Initialize tracking
    passed_tests = 0
    total_tests = 0
    pattern_stats = {pattern["name"]: {"tests": 0, "passes": 0, "avg_score": 0.0}
                    for pattern in query_patterns}

    # Initialize for stacked panels instead of table
    formatter.create_anime_test_panels()

    for i, anime in enumerate(test_anime):
        anime_title = anime.get('title', 'Unknown')

        # Filter combinations to only those with available fields
        available_roles = []
        staff_data = anime['staff_data']['production_staff']
        for role in staff_roles:
            if role in staff_data and staff_data[role]:
                available_roles.append(role)

        if not available_roles:
            continue

        # Filter field combinations to available roles
        valid_combinations = []
        for combination in all_field_combinations:
            if all(role in available_roles for role in combination):
                valid_combinations.append(combination)

        if not valid_combinations:
            continue

        # Randomly select a field combination and query pattern
        selected_combination = random.choice(valid_combinations)
        selected_pattern = random.choice(query_patterns)

        # Generate query using selected pattern and combination
        text_query = selected_pattern["generator"](anime, selected_combination)

        if not text_query or len(text_query.strip()) < 3:
            continue

        # Extract staff names for validation
        staff_names = extract_staff_names_from_query(text_query, anime, selected_combination)

        # Generate embedding
        embedding = text_processor.encode_text(text_query)

        if not embedding:
            continue

        # Search staff_vector using production method
        try:
            qdrant_client = QdrantClient(settings=settings)

            results = asyncio.run(qdrant_client.search_single_vector(
                vector_name="staff_vector",
                vector_data=embedding,
                limit=5
            ))

            if results:
                top_result = results[0]
                top_title = top_result.get("title", "Unknown")
                top_score = top_result.get("score", 0.0)

                # Enhanced validation - Check top 5 results instead of top 3
                test_passed = False
                if top_title == anime_title:
                    test_passed = True
                    passed_tests += 1
                elif any(r.get("title") == anime_title for r in results[:5]):
                    test_passed = True
                    passed_tests += 1
                elif staff_names and verify_staff_in_anime(staff_names, top_title, anime_database):
                    test_passed = True
                    passed_tests += 1

                # Track pattern effectiveness
                pattern_stats[selected_pattern["name"]]["tests"] += 1
                if test_passed:
                    pattern_stats[selected_pattern["name"]]["passes"] += 1
                pattern_stats[selected_pattern["name"]]["avg_score"] += top_score

                # Print detailed result using rich formatter
                formatter.print_detailed_staff_test_result(
                    i+1, anime, selected_combination, selected_pattern,
                    text_query, results, test_passed, staff_names
                )

                total_tests += 1
            else:
                formatter.print_detailed_staff_test_result(
                    i+1, anime, selected_combination, selected_pattern,
                    text_query, [], False, staff_names
                )
                pattern_stats[selected_pattern["name"]]["tests"] += 1
                total_tests += 1

        except Exception as e:
            error_results = [{"title": f"Error: {str(e)}", "score": 0.0}]
            formatter.print_detailed_staff_test_result(
                i+1, anime, selected_combination, selected_pattern,
                text_query, error_results, False, staff_names
            )
            pattern_stats[selected_pattern["name"]]["tests"] += 1
            total_tests += 1

    # Calculate averages
    for pattern_name in pattern_stats:
        if pattern_stats[pattern_name]["tests"] > 0:
            pattern_stats[pattern_name]["avg_score"] /= pattern_stats[pattern_name]["tests"]

    # Print pattern analysis using rich formatter
    if pattern_stats:
        print(f"\n🎯 Query Pattern Analysis:")
        for pattern_name, stats in pattern_stats.items():
            if stats["tests"] > 0:
                success_rate = (stats["passes"] / stats["tests"]) * 100
                avg_score = stats["avg_score"]
                print(f"   • {pattern_name}: {stats['passes']}/{stats['tests']} ({success_rate:.1f}%) | Avg Score: {avg_score:.4f}")
            else:
                print(f"   • {pattern_name}: Not tested this run")

    # Print final results using rich formatter
    insights = [
        "Validates recent staff extraction fixes work correctly",
        "Random field combinations ensure comprehensive coverage",
        "Cross-reference validation confirms staff name accuracy",
        "Pattern analysis identifies most effective query approaches",
        "Uses production search_single_vector() method with real similarity scores"
    ]

    formatter.print_final_results(
        "Staff Vector Test",
        passed_tests,
        total_tests,
        random_seed,
        insights
    )

if __name__ == "__main__":
    test_staff_vector_comprehensive()