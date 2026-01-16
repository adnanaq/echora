#!/usr/bin/env python3
"""
Enrichment Database Validation Script

Validates all entries in the enriched anime database for:
1. Core field requirements (8 required fields always present)
2. Empty field compliance (omit empty collections/objects)
3. Voice actor structure validation (proper nested format)
4. Schema compliance using Pydantic models

Usage:
    python validate_enrichment_database.py --validate-only
    python validate_enrichment_database.py --report
    python validate_enrichment_database.py --fix --backup
"""

import argparse
import json
import logging
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from common.models.anime import (
        AnimeRecord as _AnimeRecord,
    )
    from common.models.anime import (
        Character as _Character,
    )
    from common.models.anime import (
        SimpleVoiceActor as _SimpleVoiceActor,
    )
    from pydantic import ValidationError as _ValidationError

    PYDANTIC_AVAILABLE = True
    AnimeRecord: type[_AnimeRecord] | None = _AnimeRecord
    Character: type[_Character] | None = _Character
    SimpleVoiceActor: type[_SimpleVoiceActor] | None = _SimpleVoiceActor
    ValidationError: type[_ValidationError] | None = _ValidationError
except ImportError:
    print(
        "Warning: Could not import Pydantic models. Schema validation will be limited."
    )
    PYDANTIC_AVAILABLE = False
    AnimeRecord = None
    Character = None
    SimpleVoiceActor = None
    ValidationError = None

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ValidationIssue:
    """Individual validation issue"""

    issue_type: str
    field_path: str
    description: str
    severity: str  # 'error', 'warning', 'info'
    suggested_fix: str | None = None


@dataclass
class ValidationResult:
    """Validation result for a single anime entry"""

    anime_title: str
    anime_index: int
    is_valid: bool
    issues: list[ValidationIssue]

    @property
    def error_count(self) -> int:
        return len([i for i in self.issues if i.severity == "error"])

    @property
    def warning_count(self) -> int:
        return len([i for i in self.issues if i.severity == "warning"])


class EnrichmentValidator:
    """Validates enriched anime database entries"""

    # Fields that must ALWAYS be present (even if empty)
    REQUIRED_FIELDS = {
        "sources",
        "title",
        "type",
        "episodes",
        "status",
        "episode_details",
        "statistics",
        "enrichment_metadata",
    }

    # Collections that should be OMITTED when empty
    OMIT_EMPTY_COLLECTIONS = {
        "characters",
        "themes",
        "genres",
        "demographics",
        "streaming_info",
        "opening_themes",
        "ending_themes",
        "relations",
        "related_anime",
        "content_warnings",
        "licensors",
        "synonyms",
        "tags",
        "trailers",
    }

    # Objects that should be OMITTED when empty
    OMIT_EMPTY_OBJECTS = {
        "images",
        "external_links",
        "staff_data",
        "aired_dates",
        "broadcast",
        "broadcast_schedule",
        "duration",
        "premiere_dates",
        "score",
        "delay_information",
        "episode_overrides",
        "popularity_trends",
        "character_pages",
    }

    # Scalar fields that should be OMITTED when null/empty
    OMIT_IF_NULL = {
        "synopsis",
        "picture",
        "thumbnail",
        "title_english",
        "title_japanese",
        "rating",
        "source_material",
        "background",
        "month",
        "nsfw",
        "year",
        "season",
    }

    def __init__(self) -> None:
        self.validation_results: list[ValidationResult] = []

    def validate_database(self, file_path: str) -> dict[str, Any]:
        """Validate entire enriched database"""
        logger.info(f"Starting validation of {file_path}")

        try:
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load database file: {e}")
            return {"error": str(e)}

        if "data" not in data:
            logger.error("Database file missing 'data' field")
            return {"error": "Invalid database structure - missing 'data' field"}

        anime_entries = data["data"]
        logger.info(f"Found {len(anime_entries)} anime entries to validate")

        # Validate each entry
        self.validation_results = []
        for index, entry in enumerate(anime_entries):
            result = self.validate_entry(entry, index)
            self.validation_results.append(result)

        # Generate summary report
        return self.generate_report()

    def validate_entry(self, entry: dict[str, Any], index: int) -> ValidationResult:
        """Validate a single anime entry"""
        title = entry.get("title", f"Entry #{index}")
        issues: list[ValidationIssue] = []

        # 1. Core field validation
        issues.extend(self.check_required_fields(entry))

        # 2. Empty field compliance
        issues.extend(self.check_empty_field_compliance(entry))

        # 3. Voice actor structure validation
        issues.extend(self.validate_voice_actors(entry))

        # 4. Schema validation (if Pydantic available)
        if AnimeRecord is not None:
            issues.extend(self.validate_schema_compliance(entry))

        is_valid = len([i for i in issues if i.severity == "error"]) == 0

        return ValidationResult(
            anime_title=title, anime_index=index, is_valid=is_valid, issues=issues
        )

    def check_required_fields(self, entry: dict[str, Any]) -> list[ValidationIssue]:
        """Check that all required fields are present"""
        issues = []

        for field in self.REQUIRED_FIELDS:
            if field not in entry:
                issues.append(
                    ValidationIssue(
                        issue_type="missing_required_field",
                        field_path=field,
                        description=f"Required field '{field}' is missing",
                        severity="error",
                        suggested_fix=f"Add '{field}' field with appropriate default value",
                    )
                )
            elif entry[field] is None and field not in [
                "statistics",
                "enrichment_metadata",
            ]:
                issues.append(
                    ValidationIssue(
                        issue_type="null_required_field",
                        field_path=field,
                        description=f"Required field '{field}' is null",
                        severity="error",
                        suggested_fix=f"Set '{field}' to appropriate non-null value",
                    )
                )

        return issues

    def check_empty_field_compliance(
        self, entry: dict[str, Any]
    ) -> list[ValidationIssue]:
        """Check that empty fields are handled correctly"""
        issues = []

        # Check for empty collections that should be omitted
        for field in self.OMIT_EMPTY_COLLECTIONS:
            if field in entry:
                value = entry[field]
                if isinstance(value, list) and len(value) == 0:
                    issues.append(
                        ValidationIssue(
                            issue_type="empty_collection_present",
                            field_path=field,
                            description=f"Empty collection '{field}' should be omitted",
                            severity="warning",
                            suggested_fix=f"Remove '{field}' field when empty",
                        )
                    )

        # Check for empty objects that should be omitted
        for field in self.OMIT_EMPTY_OBJECTS:
            if field in entry:
                value = entry[field]
                if isinstance(value, dict) and self._is_empty_object(value):
                    issues.append(
                        ValidationIssue(
                            issue_type="empty_object_present",
                            field_path=field,
                            description=f"Empty object '{field}' should be omitted",
                            severity="warning",
                            suggested_fix=f"Remove '{field}' field when empty",
                        )
                    )

        # Check for null fields that should be omitted
        for field in self.OMIT_IF_NULL:
            if field in entry and (entry[field] is None or entry[field] == ""):
                issues.append(
                    ValidationIssue(
                        issue_type="null_field_present",
                        field_path=field,
                        description=f"Null/empty field '{field}' should be omitted",
                        severity="info",
                        suggested_fix=f"Remove '{field}' field when null/empty",
                    )
                )

        # Check character-level empty fields
        if "characters" in entry and isinstance(entry["characters"], list):
            for i, char in enumerate(entry["characters"]):
                if isinstance(char, dict):
                    issues.extend(self._check_character_empty_fields(char, i))

        # Check staff_data nested empty fields
        if "staff_data" in entry and isinstance(entry["staff_data"], dict):
            issues.extend(self._check_staff_data_empty_fields(entry["staff_data"]))

        return issues

    def _check_character_empty_fields(
        self, char: dict[str, Any], char_index: int
    ) -> list[ValidationIssue]:
        """Check for empty fields within character objects"""
        issues = []
        char_name = char.get("name", f"Character {char_index}")

        # Character fields that should be omitted when empty
        char_empty_collections = {
            "name_variations",
            "nicknames",
            "images",
            "character_traits",
        }
        char_empty_objects = {"character_ids", "character_pages"}
        char_empty_scalars = {
            "name_native",
            "description",
            "age",
            "gender",
            "hair_color",
            "eye_color",
            "favorites",
        }

        # Check empty collections
        for field in char_empty_collections:
            if (
                field in char
                and isinstance(char[field], list)
                and len(char[field]) == 0
            ):
                issues.append(
                    ValidationIssue(
                        issue_type="empty_character_collection",
                        field_path=f"characters[{char_index}].{field}",
                        description=f"Empty collection '{field}' in character '{char_name}' should be omitted",
                        severity="warning",
                        suggested_fix=f"Remove '{field}' field from character when empty",
                    )
                )

        # Check empty objects
        for field in char_empty_objects:
            if (
                field in char
                and isinstance(char[field], dict)
                and self._is_empty_object(char[field])
            ):
                issues.append(
                    ValidationIssue(
                        issue_type="empty_character_object",
                        field_path=f"characters[{char_index}].{field}",
                        description=f"Empty object '{field}' in character '{char_name}' should be omitted",
                        severity="warning",
                        suggested_fix=f"Remove '{field}' field from character when empty",
                    )
                )

        # Check null/empty scalars
        for field in char_empty_scalars:
            if field in char and (char[field] is None or char[field] == ""):
                issues.append(
                    ValidationIssue(
                        issue_type="empty_character_scalar",
                        field_path=f"characters[{char_index}].{field}",
                        description=f"Null/empty field '{field}' in character '{char_name}' should be omitted",
                        severity="info",
                        suggested_fix=f"Remove '{field}' field from character when null/empty",
                    )
                )

        return issues

    def _check_staff_data_empty_fields(
        self, staff_data: dict[str, Any]
    ) -> list[ValidationIssue]:
        """Check for empty fields within staff_data structure"""
        issues = []

        # Check top-level staff_data empty collections
        staff_empty_collections = {"studios", "producers", "licensors"}
        for field in staff_empty_collections:
            if (
                field in staff_data
                and isinstance(staff_data[field], list)
                and len(staff_data[field]) == 0
            ):
                issues.append(
                    ValidationIssue(
                        issue_type="empty_staff_collection",
                        field_path=f"staff_data.{field}",
                        description=f"Empty collection '{field}' in staff_data should be omitted",
                        severity="warning",
                        suggested_fix=f"Remove '{field}' field from staff_data when empty",
                    )
                )

        # Check production_staff nested empty collections
        if "production_staff" in staff_data and isinstance(
            staff_data["production_staff"], dict
        ):
            production_staff = staff_data["production_staff"]

            # All possible production staff roles that could be empty
            production_roles = {
                "directors",
                "music_composers",
                "character_designers",
                "series_writers",
                "animation_directors",
                "original_creators",
                "art_directors",
                "sound_directors",
                "producers",
                "writers",
                "composers",
                "designers",
                "animators",
            }

            for role in production_roles:
                if (
                    role in production_staff
                    and isinstance(production_staff[role], list)
                    and len(production_staff[role]) == 0
                ):
                    issues.append(
                        ValidationIssue(
                            issue_type="empty_production_staff_role",
                            field_path=f"staff_data.production_staff.{role}",
                            description=f"Empty production staff role '{role}' should be omitted",
                            severity="warning",
                            suggested_fix=f"Remove '{role}' field from production_staff when empty",
                        )
                    )

        return issues

    def validate_voice_actors(self, entry: dict[str, Any]) -> list[ValidationIssue]:
        """Validate voice actor structure"""
        issues = []

        # Check staff_data voice_actors structure
        if "staff_data" in entry and "voice_actors" in entry["staff_data"]:
            va_data = entry["staff_data"]["voice_actors"]

            # Check for malformed array structure
            if isinstance(va_data, list):
                issues.append(
                    ValidationIssue(
                        issue_type="malformed_voice_actors",
                        field_path="staff_data.voice_actors",
                        description="Voice actors should be object {japanese: []} not array []",
                        severity="error",
                        suggested_fix="Convert to {japanese: []} structure or omit if empty",
                    )
                )
            elif isinstance(va_data, dict):
                # Check if all language arrays are empty
                if self._is_empty_object(va_data):
                    issues.append(
                        ValidationIssue(
                            issue_type="empty_voice_actors_object",
                            field_path="staff_data.voice_actors",
                            description="Empty voice_actors object should be omitted",
                            severity="warning",
                            suggested_fix="Remove voice_actors field from staff_data",
                        )
                    )

        # Check character voice_actors
        if "characters" in entry:
            for i, char in enumerate(entry["characters"]):
                if "voice_actors" in char:
                    va_list = char["voice_actors"]
                    if isinstance(va_list, list) and len(va_list) == 0:
                        issues.append(
                            ValidationIssue(
                                issue_type="empty_character_voice_actors",
                                field_path=f"characters[{i}].voice_actors",
                                description=f"Empty voice_actors for character '{char.get('name', 'Unknown')}' should be omitted",
                                severity="warning",
                                suggested_fix="Remove voice_actors field from character",
                            )
                        )
                    elif isinstance(va_list, list):
                        # Validate individual voice actor entries
                        for j, va in enumerate(va_list):
                            if (
                                not isinstance(va, dict)
                                or "name" not in va
                                or "language" not in va
                            ):
                                issues.append(
                                    ValidationIssue(
                                        issue_type="invalid_voice_actor_structure",
                                        field_path=f"characters[{i}].voice_actors[{j}]",
                                        description="Voice actor must have 'name' and 'language' fields",
                                        severity="error",
                                        suggested_fix="Ensure voice actor has {name: str, language: str} structure",
                                    )
                                )

        return issues

    def validate_schema_compliance(
        self, entry: dict[str, Any]
    ) -> list[ValidationIssue]:
        """Validate entry against Pydantic schema"""
        issues = []

        # Skip schema validation if Pydantic models are not available
        if not PYDANTIC_AVAILABLE or AnimeRecord is None or ValidationError is None:
            return issues

        try:
            # Attempt to validate with AnimeRecord model
            AnimeRecord(**entry)
        except ValidationError as e:
            for error in e.errors():
                field_path = ".".join(str(x) for x in error["loc"])
                issues.append(
                    ValidationIssue(
                        issue_type="schema_validation_error",
                        field_path=field_path,
                        description=f"Schema validation error: {error['msg']}",
                        severity="error",
                        suggested_fix="Fix field to match expected schema type/format",
                    )
                )
        except Exception as e:
            issues.append(
                ValidationIssue(
                    issue_type="validation_exception",
                    field_path="root",
                    description=f"Validation exception: {str(e)}",
                    severity="warning",
                    suggested_fix="Review entry structure for unexpected format",
                )
            )

        return issues

    def _is_empty_object(self, obj: dict[str, Any]) -> bool:
        """Check if object is effectively empty"""
        if not obj:
            return True

        # Recursively check if all values are empty
        for value in obj.values():
            if isinstance(value, dict) and not self._is_empty_object(value):
                return False
            elif isinstance(value, list) and len(value) > 0:
                return False
            elif value not in [None, "", [], {}]:
                return False

        return True

    def generate_report(self) -> dict[str, Any]:
        """Generate comprehensive validation report"""
        total_entries = len(self.validation_results)
        valid_entries = len([r for r in self.validation_results if r.is_valid])

        # Count issues by type
        issue_counts: dict[str, int] = {}
        severity_counts = {"error": 0, "warning": 0, "info": 0}

        for result in self.validation_results:
            for issue in result.issues:
                issue_counts[issue.issue_type] = (
                    issue_counts.get(issue.issue_type, 0) + 1
                )
                severity_counts[issue.severity] += 1

        # Identify most problematic entries
        problematic_entries = sorted(
            self.validation_results, key=lambda r: r.error_count, reverse=True
        )[:5]

        report = {
            "validation_summary": {
                "total_entries": total_entries,
                "valid_entries": valid_entries,
                "invalid_entries": total_entries - valid_entries,
                "compliance_percentage": round(
                    (valid_entries / total_entries) * 100, 2
                ),
                "validation_timestamp": datetime.now().isoformat(),
            },
            "issue_statistics": {
                "by_type": issue_counts,
                "by_severity": severity_counts,
                "total_issues": sum(severity_counts.values()),
            },
            "most_problematic_entries": [
                {
                    "title": entry.anime_title,
                    "index": entry.anime_index,
                    "error_count": entry.error_count,
                    "warning_count": entry.warning_count,
                    "issues": [
                        {
                            "type": issue.issue_type,
                            "field": issue.field_path,
                            "description": issue.description,
                            "severity": issue.severity,
                        }
                        for issue in entry.issues[:3]  # Top 3 issues
                    ],
                }
                for entry in problematic_entries
                if entry.error_count > 0
            ],
            "detailed_results": [
                {
                    "title": result.anime_title,
                    "index": result.anime_index,
                    "is_valid": result.is_valid,
                    "error_count": result.error_count,
                    "warning_count": result.warning_count,
                    "issues": [
                        {
                            "type": issue.issue_type,
                            "field": issue.field_path,
                            "description": issue.description,
                            "severity": issue.severity,
                            "suggested_fix": issue.suggested_fix,
                        }
                        for issue in result.issues
                    ],
                }
                for result in self.validation_results
            ],
        }

        return report

    def auto_fix_database(self, file_path: str, backup: bool = True) -> dict[str, Any]:
        """Auto-fix common issues in the database"""
        logger.info(f"Starting auto-fix for {file_path}")

        # Create backup if requested
        if backup:
            backup_path = (
                f"{file_path}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            with (
                open(file_path, encoding="utf-8") as original,
                open(backup_path, "w", encoding="utf-8") as backup_file,
            ):
                backup_file.write(original.read())
            logger.info(f"Created backup: {backup_path}")

        # Load database
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        fixed_count = 0
        fix_summary = []

        # Fix each entry
        for index, entry in enumerate(data["data"]):
            fixed_entry, fixes = self.auto_fix_entry(entry)
            data["data"][index] = fixed_entry
            if fixes:
                fixed_count += 1
                fix_summary.extend(fixes)

        # Save fixed database
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"Auto-fix complete. Fixed {fixed_count} entries.")

        return {
            "fixed_entries": fixed_count,
            "total_fixes": len(fix_summary),
            "fixes_applied": fix_summary,
            "backup_created": backup_path if backup else None,
        }

    def auto_fix_entry(self, entry: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        """Auto-fix a single entry"""
        fixed_entry = deepcopy(entry)
        fixes_applied = []

        # Fix malformed voice_actors structure
        if "staff_data" in fixed_entry and "voice_actors" in fixed_entry["staff_data"]:
            va_data = fixed_entry["staff_data"]["voice_actors"]
            if isinstance(va_data, list):
                if len(va_data) == 0:
                    # Remove empty array
                    del fixed_entry["staff_data"]["voice_actors"]
                    fixes_applied.append(
                        "Removed empty voice_actors array from staff_data"
                    )
                else:
                    # Convert to proper structure
                    fixed_entry["staff_data"]["voice_actors"] = {"japanese": va_data}
                    fixes_applied.append(
                        "Converted voice_actors array to proper object structure"
                    )
            elif isinstance(va_data, dict) and self._is_empty_object(va_data):
                # Remove empty voice_actors object
                del fixed_entry["staff_data"]["voice_actors"]
                fixes_applied.append(
                    "Removed empty voice_actors object from staff_data"
                )

        # Remove empty collections that should be omitted
        for field in self.OMIT_EMPTY_COLLECTIONS:
            if field in fixed_entry:
                value = fixed_entry[field]
                if isinstance(value, list) and len(value) == 0:
                    del fixed_entry[field]
                    fixes_applied.append(f"Removed empty collection: {field}")

        # Remove empty objects that should be omitted
        for field in self.OMIT_EMPTY_OBJECTS:
            if field in fixed_entry:
                value = fixed_entry[field]
                if isinstance(value, dict) and self._is_empty_object(value):
                    del fixed_entry[field]
                    fixes_applied.append(f"Removed empty object: {field}")

        # Remove null fields that should be omitted
        for field in self.OMIT_IF_NULL:
            if field in fixed_entry and (
                fixed_entry[field] is None or fixed_entry[field] == ""
            ):
                del fixed_entry[field]
                fixes_applied.append(f"Removed null/empty field: {field}")

        # Clean up character-level empty fields
        if "characters" in fixed_entry:
            for char in fixed_entry["characters"]:
                char_name = char.get("name", "Unknown")

                # Remove empty voice_actors
                if (
                    "voice_actors" in char
                    and isinstance(char["voice_actors"], list)
                    and len(char["voice_actors"]) == 0
                ):
                    del char["voice_actors"]
                    fixes_applied.append(
                        f"Removed empty voice_actors from character: {char_name}"
                    )

                # Remove empty collections
                for field in {
                    "name_variations",
                    "nicknames",
                    "images",
                    "character_traits",
                }:
                    if (
                        field in char
                        and isinstance(char[field], list)
                        and len(char[field]) == 0
                    ):
                        del char[field]
                        fixes_applied.append(
                            f"Removed empty {field} from character: {char_name}"
                        )

                # Remove empty objects
                for field in ["character_ids", "character_pages"]:
                    if (
                        field in char
                        and isinstance(char[field], dict)
                        and self._is_empty_object(char[field])
                    ):
                        del char[field]
                        fixes_applied.append(
                            f"Removed empty {field} from character: {char_name}"
                        )

                # Remove null/empty scalars
                for field in [
                    "name_native",
                    "description",
                    "age",
                    "gender",
                    "hair_color",
                    "eye_color",
                    "favorites",
                ]:
                    if field in char and (char[field] is None or char[field] == ""):
                        del char[field]
                        fixes_applied.append(
                            f"Removed null/empty {field} from character: {char_name}"
                        )

        # Clean up staff_data nested empty fields
        if "staff_data" in fixed_entry and isinstance(fixed_entry["staff_data"], dict):
            staff_data = fixed_entry["staff_data"]

            # Remove empty collections from top-level staff_data
            for field in ["studios", "producers", "licensors"]:
                if (
                    field in staff_data
                    and isinstance(staff_data[field], list)
                    and len(staff_data[field]) == 0
                ):
                    del staff_data[field]
                    fixes_applied.append(f"Removed empty {field} from staff_data")

            # Remove empty collections from production_staff
            if "production_staff" in staff_data and isinstance(
                staff_data["production_staff"], dict
            ):
                production_staff = staff_data["production_staff"]
                production_roles = {
                    "directors",
                    "music_composers",
                    "character_designers",
                    "series_writers",
                    "animation_directors",
                    "original_creators",
                    "art_directors",
                    "sound_directors",
                    "producers",
                    "writers",
                    "composers",
                    "designers",
                    "animators",
                }

                roles_to_remove = []
                for role in production_roles:
                    if (
                        role in production_staff
                        and isinstance(production_staff[role], list)
                        and len(production_staff[role]) == 0
                    ):
                        roles_to_remove.append(role)

                for role in roles_to_remove:
                    del production_staff[role]
                    fixes_applied.append(f"Removed empty {role} from production_staff")

        return fixed_entry, fixes_applied


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate enriched anime database")
    parser.add_argument(
        "--file",
        default="data/enriched_anime_database.json",
        help="Path to enriched database file",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate, do not generate detailed report",
    )
    parser.add_argument(
        "--report", action="store_true", help="Generate detailed validation report"
    )
    parser.add_argument("--fix", action="store_true", help="Auto-fix common issues")
    parser.add_argument(
        "--backup",
        action="store_true",
        default=True,
        help="Create backup before fixing (default: True)",
    )
    parser.add_argument("--output", help="Output file for validation report")

    args = parser.parse_args()

    # Resolve file path
    file_path = Path(args.file)
    if not file_path.is_absolute():
        file_path = Path(__file__).parent / file_path

    if not file_path.exists():
        logger.error(f"Database file not found: {file_path}")
        return 1

    validator = EnrichmentValidator()

    # Auto-fix if requested
    if args.fix:
        fix_result = validator.auto_fix_database(str(file_path), args.backup)
        print("Auto-fix completed:")
        print(f"  Fixed entries: {fix_result['fixed_entries']}")
        print(f"  Total fixes: {fix_result['total_fixes']}")
        if fix_result.get("backup_created"):
            print(f"  Backup created: {fix_result['backup_created']}")
        print()

    # Validate database
    report = validator.validate_database(str(file_path))

    if "error" in report:
        logger.error(f"Validation failed: {report['error']}")
        return 1

    # Print summary
    summary = report["validation_summary"]
    print("Validation Summary:")
    print(f"  Total entries: {summary['total_entries']}")
    print(f"  Valid entries: {summary['valid_entries']}")
    print(f"  Invalid entries: {summary['invalid_entries']}")
    print(f"  Compliance: {summary['compliance_percentage']}%")
    print()

    # Print issue statistics
    issue_stats = report["issue_statistics"]
    print("Issue Statistics:")
    print(f"  Errors: {issue_stats['by_severity']['error']}")
    print(f"  Warnings: {issue_stats['by_severity']['warning']}")
    print(f"  Info: {issue_stats['by_severity']['info']}")
    print(f"  Total issues: {issue_stats['total_issues']}")
    print()

    # Show most problematic entries if any
    if report["most_problematic_entries"]:
        print("Most Problematic Entries:")
        for entry in report["most_problematic_entries"][:3]:
            print(f"  '{entry['title']}' (Index: {entry['index']})")
            print(
                f"    Errors: {entry['error_count']}, Warnings: {entry['warning_count']}"
            )
            for issue in entry["issues"]:
                print(f"    - {issue['severity'].upper()}: {issue['description']}")
        print()

    # Generate detailed report if requested
    if args.report or args.output:
        output_file = (
            args.output
            or f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Detailed report saved to: {output_file}")

    return 0 if summary["invalid_entries"] == 0 else 1


if __name__ == "__main__":
    exit(main())
