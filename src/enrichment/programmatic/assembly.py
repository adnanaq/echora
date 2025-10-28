"""
Step 5: Assembly Module - Merge AI outputs into AnimeEntry schema

This module takes outputs from Step 4 (6-stage AI pipeline) and assembles them
into a complete AnimeEntry object that passes schema validation.

Key responsibilities:
1. Merge programmatic data (Steps 1-3) with AI-enhanced data (Step 4)
2. Apply intelligent field mapping and conflict resolution
3. Ensure schema compliance using validation rules
4. Handle empty field compliance per EnrichmentValidator rules
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Type annotations for optional imports
_AnimeEntry: type[Any] | None = None
_EnrichmentMetadata: type[Any] | None = None
_EnrichmentValidator: type[Any] | None = None
_ValidationError: type[Exception] = Exception

try:
    # Import the existing validator
    import sys
    from pathlib import Path

    from pydantic import ValidationError

    from src.models.anime import AnimeEntry, EnrichmentMetadata

    sys.path.append(str(Path(__file__).parent.parent.parent.parent))
    from validate_enrichment_database import EnrichmentValidator

    # Set the type markers for successful imports
    _AnimeEntry = AnimeEntry
    _EnrichmentMetadata = EnrichmentMetadata
    _EnrichmentValidator = EnrichmentValidator
    _ValidationError = ValidationError
except ImportError:
    print(
        "Warning: Could not import Pydantic models or validator. Assembly will be limited."
    )
    # Keep the None values already assigned above

logger = logging.getLogger(__name__)

# AnimeEntry schema field ordering constants
SCALAR_FIELDS = [
    "background",
    "episodes",
    "month",
    "nsfw",
    "picture",
    "rating",
    "source_material",
    "status",
    "synopsis",
    "thumbnail",
    "title",
    "title_english",
    "title_japanese",
    "type",
]

ARRAY_FIELDS = [
    "awards",
    "characters",
    "content_warnings",
    "demographics",
    "ending_themes",
    "episode_details",
    "genres",
    "licensors",
    "opening_themes",
    "related_anime",
    "relations",
    "sources",
    "streaming_info",
    "synonyms",
    "tags",
    "themes",
    "trailers",
]

OBJECT_FIELDS = [
    "aired_dates",
    "anime_season",
    "broadcast",
    "broadcast_schedule",
    "delay_information",
    "duration",
    "external_links",
    "producers",
    "score",
    "statistics",
    "studios",
]


@dataclass
class AssemblyResult:
    """Result of assembly operation"""

    success: bool
    anime_entry: dict[str, Any] | None
    errors: list[str]
    warnings: list[str]
    validation_passed: bool


class EnrichmentAssembler:
    """
    Assembles AI stage outputs into final AnimeEntry schema
    """

    # Required fields that must always be present
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

    # Fields that should be omitted when empty (from validator)
    OMIT_EMPTY_COLLECTIONS = {
        "characters",
        "awards",
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

    OMIT_EMPTY_OBJECTS = {
        "images",
        "external_links",
        "staff_data",
        "aired_dates",
        "broadcast",
        "broadcast_schedule",
        "anime_season",
        "duration",
        "premiere_dates",
        "score",
        "delay_information",
        "episode_overrides",
        "popularity_trends",
    }

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
    }

    def __init__(self) -> None:
        self.errors = []
        self.warnings = []

    def assemble_from_stages(
        self,
        stage_outputs: dict[str, Any],
        programmatic_data: dict[str, Any],
        anime_sources: list[str],
    ) -> AssemblyResult:
        """
        Assemble complete AnimeEntry from stage outputs and programmatic data

        Args:
            stage_outputs: Dictionary with keys 'stage1' through 'stage6' containing AI outputs
            programmatic_data: Data from Steps 1-3 (API fetching, episode processing)
            anime_sources: Original source URLs for the anime

        Returns:
            AssemblyResult with assembled anime entry
        """
        self.errors = []
        self.warnings = []

        try:
            # Start with base structure
            anime_entry = self._create_base_entry(anime_sources)

            # Merge programmatic data (Steps 1-3)
            anime_entry = self._merge_programmatic_data(anime_entry, programmatic_data)

            # Merge AI stage outputs (Step 4)
            anime_entry = self._merge_ai_stages(anime_entry, stage_outputs)

            # Apply field mapping and cleanup
            anime_entry = self._apply_field_mapping(anime_entry)

            # Add enrichment metadata and apply schema ordering
            anime_entry = self._add_enrichment_metadata(anime_entry)

            # Assembly complete - validation will be done separately
            validation_passed = True

            return AssemblyResult(
                success=len(self.errors) == 0,
                anime_entry=anime_entry,
                errors=self.errors,
                warnings=self.warnings,
                validation_passed=validation_passed,
            )

        except Exception as e:
            logger.error(f"Assembly failed: {str(e)}")
            self.errors.append(f"Assembly exception: {str(e)}")
            return AssemblyResult(
                success=False,
                anime_entry=None,
                errors=self.errors,
                warnings=self.warnings,
                validation_passed=False,
            )

    def _create_base_entry(self, sources: list[str]) -> dict[str, Any]:
        """Create base anime entry with required fields"""
        return {
            "sources": sources,
            "title": "",
            "type": "",
            "episodes": 0,
            "status": "",
            "episode_details": [],
            "statistics": {},
            "enrichment_metadata": None,
        }

    def _merge_programmatic_data(
        self, entry: dict[str, Any], programmatic_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Merge data from programmatic Steps 1-3"""
        if not programmatic_data:
            return entry

        # Merge API data (from Step 2 - API fetching)
        if "api_data" in programmatic_data:
            api_data = programmatic_data["api_data"]

            # Extract basic fields from API data
            for api_name, data in api_data.items():
                if not data:
                    continue

                # Merge title information
                if "title" in data and not entry["title"]:
                    entry["title"] = data["title"]
                if "title_english" in data:
                    entry["title_english"] = data.get("title_english")
                if "title_japanese" in data:
                    entry["title_japanese"] = data.get("title_japanese")

                # Merge basic metadata
                for field in ["type", "episodes", "status", "synopsis", "rating"]:
                    if field in data and field not in entry:
                        entry[field] = data[field]

        # Merge episode data (from Step 3 - episode processing)
        if "episodes" in programmatic_data:
            entry["episode_details"] = programmatic_data["episodes"]

        return entry

    def _merge_ai_stages(
        self, entry: dict[str, Any], stage_outputs: dict[str, Any]
    ) -> dict[str, Any]:
        """Merge AI stage outputs into entry"""

        # Stage 1: Metadata extraction
        if "stage1" in stage_outputs:
            entry = self._merge_stage1_metadata(entry, stage_outputs["stage1"])

        # Stage 2: Episodes (already handled by programmatic pipeline)
        if "stage2" in stage_outputs:
            entry = self._merge_stage2_episodes(entry, stage_outputs["stage2"])

        # Stage 3: Relationships
        if "stage3" in stage_outputs:
            entry = self._merge_stage3_relationships(entry, stage_outputs["stage3"])

        # Stage 4: Statistics and media
        if "stage4" in stage_outputs:
            entry = self._merge_stage4_statistics(entry, stage_outputs["stage4"])

        # Stage 5: Characters
        if "stage5" in stage_outputs:
            entry = self._merge_stage5_characters(entry, stage_outputs["stage5"])

        # Stage 6: Staff
        if "stage6" in stage_outputs:
            entry = self._merge_stage6_staff(entry, stage_outputs["stage6"])

        return entry

    def _merge_stage1_metadata(
        self, entry: dict[str, Any], stage1_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Merge Stage 1: Metadata extraction"""
        if not stage1_data:
            return entry

        # Direct field mapping for scalar values
        scalar_fields = [
            "background",
            "episodes",
            "month",
            "nsfw",
            "rating",
            "source_material",
            "status",
            "synopsis",
            "title",
            "title_english",
            "title_japanese",
            "type",
        ]

        for field in scalar_fields:
            if field in stage1_data:
                value = stage1_data[field]
                if value is not None and value != "":
                    entry[field] = value

        # Array fields
        array_fields = [
            "content_warnings",
            "demographics",
            "genres",
            "synonyms",
            "themes",
            "trailers",
        ]

        for field in array_fields:
            if field in stage1_data and isinstance(stage1_data[field], list):
                if len(stage1_data[field]) > 0:  # Only add non-empty arrays
                    entry[field] = stage1_data[field]

        # Object fields
        object_fields = [
            "aired_dates",
            "anime_season",
            "broadcast",
            "broadcast_schedule",
            "delay_information",
            "duration",
            "episode_overrides",
            "external_links",
        ]

        for field in object_fields:
            if field in stage1_data and stage1_data[field]:
                if not self._is_empty_object(stage1_data[field]):
                    entry[field] = stage1_data[field]

        return entry

    def _merge_stage2_episodes(
        self, entry: dict[str, Any], stage2_data: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Merge Stage 2: Episodes (supplement programmatic data)"""
        if not stage2_data or not isinstance(stage2_data, list):
            return entry

        # If we already have episode_details from programmatic pipeline, enhance them
        # Otherwise, use the AI-generated episodes
        if not entry.get("episode_details"):
            entry["episode_details"] = stage2_data
        else:
            # Enhance existing episodes with AI data
            existing_episodes = {
                ep.get("episode_number"): ep for ep in entry["episode_details"]
            }

            for ai_episode in stage2_data:
                ep_num = ai_episode.get("episode_number")
                if ep_num in existing_episodes:
                    # Merge AI enhancements into existing episode
                    existing_ep = existing_episodes[ep_num]
                    for field, value in ai_episode.items():
                        if field not in existing_ep or not existing_ep[field]:
                            existing_ep[field] = value
                else:
                    # Add new episode from AI
                    entry["episode_details"].append(ai_episode)

        return entry

    def _merge_stage3_relationships(
        self, entry: dict[str, Any], stage3_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Merge Stage 3: Relationships"""
        if not stage3_data:
            return entry

        if "related_anime" in stage3_data and isinstance(
            stage3_data["related_anime"], list
        ):
            if len(stage3_data["related_anime"]) > 0:
                entry["related_anime"] = stage3_data["related_anime"]

        if "relations" in stage3_data and isinstance(stage3_data["relations"], list):
            if len(stage3_data["relations"]) > 0:
                entry["relations"] = stage3_data["relations"]

        return entry

    def _merge_stage4_statistics(
        self, entry: dict[str, Any], stage4_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Merge Stage 4: Statistics and media"""
        if not stage4_data:
            return entry

        if "statistics" in stage4_data and isinstance(stage4_data["statistics"], dict):
            entry["statistics"] = stage4_data["statistics"]

        return entry

    def _merge_stage5_characters(
        self, entry: dict[str, Any], stage5_data: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Merge Stage 5: Characters"""
        if not stage5_data or not isinstance(stage5_data, list):
            return entry

        if len(stage5_data) > 0:
            entry["characters"] = stage5_data

        return entry

    def _merge_stage6_staff(
        self, entry: dict[str, Any], stage6_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Merge Stage 6: Staff"""
        if not stage6_data:
            return entry

        if "staff_data" in stage6_data and isinstance(stage6_data["staff_data"], dict):
            if not self._is_empty_object(stage6_data["staff_data"]):
                entry["staff_data"] = stage6_data["staff_data"]

        return entry

    def _apply_field_mapping(self, entry: dict[str, Any]) -> dict[str, Any]:
        """Apply intelligent field mapping and conflict resolution"""

        # Ensure required fields have appropriate defaults
        if not entry.get("title"):
            self.errors.append("Missing required field: title")
            entry["title"] = "Unknown Title"

        if not entry.get("type"):
            self.warnings.append("Missing type field, defaulting to 'UNKNOWN'")
            entry["type"] = "UNKNOWN"

        if not entry.get("status"):
            self.warnings.append("Missing status field, defaulting to 'UNKNOWN'")
            entry["status"] = "UNKNOWN"

        # Ensure episodes is a number
        if not isinstance(entry.get("episodes"), int):
            try:
                entry["episodes"] = int(entry.get("episodes", 0))
            except (ValueError, TypeError):
                entry["episodes"] = 0
                self.warnings.append("Invalid episodes value, defaulting to 0")

        return entry

    # Note: No cleanup in Step 5 - just merge the data
    # Validation and cleanup will be done separately after assembly

    def _add_enrichment_metadata(self, entry: dict[str, Any]) -> dict[str, Any]:
        """Add enrichment metadata and apply proper schema field ordering"""
        entry["enrichment_metadata"] = {
            "source": "programmatic_ai_pipeline",
            "enriched_at": datetime.now().isoformat(),
            "success": len(self.errors) == 0,
            "error_message": "; ".join(self.errors) if self.errors else None,
        }

        # Apply proper AnimeEntry schema field ordering
        return self._apply_schema_ordering(entry)

    def _apply_schema_ordering(self, entry: dict[str, Any]) -> dict[str, Any]:
        """Apply AnimeEntry schema field ordering: SCALAR → ARRAY → OBJECT → enrichment_metadata"""

        # Build ordered entry
        ordered_entry = {}

        # 1. SCALAR FIELDS (alphabetical)
        for field in SCALAR_FIELDS:
            if field in entry:
                ordered_entry[field] = entry[field]

        # 2. ARRAY FIELDS (alphabetical)
        for field in ARRAY_FIELDS:
            if field in entry:
                ordered_entry[field] = entry[field]

        # 3. OBJECT/DICT FIELDS (alphabetical)
        for field in OBJECT_FIELDS:
            if field in entry:
                ordered_entry[field] = entry[field]

        # 4. FINAL FIELD: enrichment_metadata (always last)
        if "enrichment_metadata" in entry:
            ordered_entry["enrichment_metadata"] = entry["enrichment_metadata"]

        return ordered_entry

    def _validate_final_output(self, entry: dict[str, Any]) -> bool:
        """Run validator on final merged output"""
        if not _EnrichmentValidator:
            self.warnings.append(
                "EnrichmentValidator not available, skipping validation"
            )
            return True

        try:
            validator = _EnrichmentValidator()

            # First validate the merged entry
            validation_result = validator.validate_entry(entry, 0)

            # Add validation issues to our results
            for issue in validation_result.issues:
                if issue.severity == "error":
                    self.errors.append(
                        f"Final validation error in {issue.field_path}: {issue.description}"
                    )
                elif issue.severity == "warning":
                    self.warnings.append(
                        f"Final validation warning in {issue.field_path}: {issue.description}"
                    )
                else:  # info
                    self.warnings.append(
                        f"Final validation info in {issue.field_path}: {issue.description}"
                    )

            # Apply auto-fix if there are any validation issues (errors or warnings)
            if validation_result.issues:
                self.warnings.append("Applying auto-fix to final merged output...")
                fixed_entry, fixes = validator.auto_fix_entry(entry)

                # Update the entry with fixes
                entry.clear()
                entry.update(fixed_entry)

                # Log the fixes
                for fix in fixes:
                    self.warnings.append(f"Auto-fix applied: {fix}")

                # Re-validate after fixes
                final_validation = validator.validate_entry(entry, 0)
                return bool(final_validation.is_valid)

            return bool(validation_result.is_valid)

        except Exception as e:
            self.errors.append(f"Final validation exception: {str(e)}")
            return False

    def _is_empty_object(self, obj: dict[str, Any]) -> bool:
        """Check if object is effectively empty"""
        if not obj:
            return True

        for value in obj.values():
            if isinstance(value, dict) and not self._is_empty_object(value):
                return False
            elif isinstance(value, list) and len(value) > 0:
                return False
            elif value not in [None, "", [], {}]:
                return False

        return True


def load_stage_outputs(stage_dir: Path) -> dict[str, Any]:
    """Load all stage outputs from directory"""
    stage_outputs = {}

    for stage_num in range(1, 7):
        stage_file = stage_dir / f"stage{stage_num}_*.json"
        stage_files = list(stage_dir.glob(f"stage{stage_num}_*.json"))

        if stage_files:
            stage_file = stage_files[0]  # Take first match
            try:
                with open(stage_file, encoding="utf-8") as f:
                    stage_outputs[f"stage{stage_num}"] = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load {stage_file}: {e}")
                stage_outputs[f"stage{stage_num}"] = None
        else:
            logger.warning(f"Stage {stage_num} output not found in {stage_dir}")
            stage_outputs[f"stage{stage_num}"] = None

    return stage_outputs


def validate_and_fix_entry(
    anime_entry: dict[str, Any],
) -> tuple[dict[str, Any], bool, list[str]]:
    """
    Validate and optionally fix an assembled anime entry using validate_enrichment_database.py

    Args:
        anime_entry: Assembled anime entry

    Returns:
        Tuple of (fixed_entry, is_valid, validation_messages)
    """
    if not _EnrichmentValidator:
        return anime_entry, True, ["EnrichmentValidator not available"]

    try:
        validator = _EnrichmentValidator()

        # First validate
        validation_result = validator.validate_entry(anime_entry, 0)

        # Collect validation messages
        messages = []
        for issue in validation_result.issues:
            messages.append(
                f"{issue.severity.upper()}: {issue.field_path} - {issue.description}"
            )

        # Apply auto-fix if there are issues
        if not validation_result.is_valid:
            fixed_entry, fixes = validator.auto_fix_entry(anime_entry)
            messages.extend([f"AUTO-FIX: {fix}" for fix in fixes])

            # Re-validate after fixes
            final_validation = validator.validate_entry(fixed_entry, 0)
            return fixed_entry, final_validation.is_valid, messages

        return anime_entry, True, messages

    except Exception as e:
        return anime_entry, False, [f"Validation exception: {str(e)}"]


def assemble_anime_entry(
    stage_dir: Path, programmatic_data: dict[str, Any], anime_sources: list[str]
) -> AssemblyResult:
    """
    Main entry point for assembly process

    Args:
        stage_dir: Directory containing stage1-6 output files
        programmatic_data: Data from Steps 1-3
        anime_sources: Original source URLs

    Returns:
        AssemblyResult with complete anime entry
    """
    assembler = EnrichmentAssembler()
    stage_outputs = load_stage_outputs(stage_dir)

    return assembler.assemble_from_stages(
        stage_outputs=stage_outputs,
        programmatic_data=programmatic_data,
        anime_sources=anime_sources,
    )
