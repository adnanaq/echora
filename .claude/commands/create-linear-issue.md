Read the file at: $1

Analyze the codebase to understand what changes were made. Check git status and recent commits to identify modified files.

Classify the change type and assign appropriate label: Feature, Bug, Refactor, Research, or Chore.

Based on the analysis:

1. Determine if this represents a single cohesive change or multiple separate issues
2. If multiple issues are needed, list them and ask for user approval before proceeding
3. Only proceed after user confirms the issue breakdown

Generate a concise, descriptive title based on the changes made (this will be passed as the `title` parameter, separate from description).

Fill the template sections with actual implementation details based on code changes. Use template headings exactly as provided - do NOT add or modify sections:

- What: Brief description of what was implemented/changed
- Why: Reason and impact
- Implementation Details: Files created/modified, dependencies added/updated
- Acceptance Criteria: What the implementation achieves
- Testing/Validation: Test files, coverage, test cases
- Related Issues/Dependencies: Dependencies and related issues

Format the description with:

- Use `###` for section headings (as in template)
- Use `backticks` for code, file paths, library versions
- Use **bold** for library names only (e.g., **crawl4ai**)
- Testing/Validation items MUST be markdown checkboxes `[ ]`
- Do NOT add bold anywhere except library names
- Do NOT use colons after prefixes like "Dependency:" or "Conflict:" - write naturally without labels
- Do NOT use plain bullet points for Testing/Validation - they MUST be checkboxes

Bad example: **Dependency:** `crawl4ai>=0.7.4` (browser automation)
Good example: `crawl4ai>=0.7.4` requires `chardet>=5.2.0`

Show user the title and formatted description first for review, then create the issue in Linear using `mcp__linear-server__create_issue` with:

- title: Generated title
- team: "$2"
- description: Formatted template content
- labels: One of [Feature, Bug, Refactor, Research, Chore]

Example:

Title: Integrate AniSearch Crawlers into API Helper System

Description:

### What

Integration of AniSearch crawlers into enrichment pipeline

### Why

Replace legacy scraper with robust browser automation for reliable data extraction

### Implementation Details

- Created `src/enrichment/api_helpers/anisearch_helper.py` with `AniSearchEnrichmentHelper` class
- Integrated `anisearch_anime_crawler`, `anisearch_episode_crawler`, `anisearch_character_crawler`
- Added specific exception handling (`AttributeError`, `TypeError`, `ValueError`)

### Acceptance Criteria

- Helper successfully fetches and aggregates anime, episode, and character data
- Graceful degradation if optional data fails

### Testing / Validation

- [ ] Test suite at `tests/enrichment/api_helpers/test_anisearch_helper.py` with 38 cases
- [ ] 100% code coverage (100 statements, 0 missed)
- [ ] Integration test with Dandadan (ID: 18878) passed

### Related Issues / Dependencies

- `crawl4ai>=0.7.4` requires `chardet>=5.2.0`
- `animeplanet_helper.py` reference pattern
