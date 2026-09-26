Analyze what the current branch changes. Compare it against `main`, not just the latest commits: `git log main..HEAD` for the commits and `git diff main...HEAD --stat` for the files, plus `git status` for uncommitted work. The description must match what is in the branch, using the files' current names.

If $ARGUMENTS is an existing issue ID (e.g. `ECHO-28`), this is an update: fetch it with `mcp__linear-server__get_issue`, keep its title unless the branch no longer matches it, and take the type from its label. Otherwise $ARGUMENTS is the team, and a new issue is created.

For a new issue, classify the change type and assign appropriate label: Feature, Bug, Refactor, Research, or Chore.

Based on the analysis:

1. Determine if this represents a single cohesive change or multiple separate issues
2. If multiple issues are needed, list them and ask for user approval before proceeding
3. Only proceed after user confirms the issue breakdown

Load the matching template from Linear, where the templates live:

1. List the issue templates with `mcp__linear-server__list_templates` (type `issue`)
2. Pick the one named after the type (the existing issue's label when updating, the classified type otherwise): Feature Template, Refactor Template, Chore Template, Bug Template or Research Template
3. Read its sections with `mcp__linear-server__get_template`; the description in its content is the structure to fill

Generate a concise, descriptive title based on the changes made (this will be passed as the `title` parameter, separate from description).

Fill the template sections with actual implementation details based on code changes. CRITICAL: Use the loaded template's exact section headings and structure - do NOT modify section names or add/remove sections

Format the description with:

- Use `###` for section headings exactly as they appear in the loaded template
- Preserve all section names including suffixes such as "(optional)"; replace the template's bracketed placeholder text with real content
- Use `backticks` for code, file paths, library versions
- Use **bold** for library names only (e.g., **crawl4ai**)
- All checklist items MUST be markdown checkboxes `[ ]`
- List items can have nested sublists when needed (e.g., file changes with sub-bullets for modifications)
- Do NOT add bold anywhere except library names
- Do NOT use colons after prefixes like "Dependency:" or "Conflict:" - write naturally without labels

Bad example: **Dependency:** `crawl4ai>=0.7.4` (browser automation)
Good example: `crawl4ai>=0.7.4` requires `chardet>=5.2.0`

Nested list example for Implementation Notes:

- Created `src/enrichment/api_helpers/anisearch_helper.py`
  - Added `AniSearchEnrichmentHelper` class
  - Integrated anime, episode, and character crawlers
  - Implemented error handling for `AttributeError`, `TypeError`, `ValueError`
- Modified `CLAUDE.md`
  - Added Stage 4 documentation section
  - Updated usage examples

Show user the title and formatted description first for review.

When updating, save it with `mcp__linear-server__save_issue` passing the issue `id` and the new `description`; leave the label as it is. When creating:

1. Get team ID using `mcp__linear-server__list_teams` to find the team matching "$ARGUMENTS"
2. Create the issue in Linear using `mcp__linear-server__save_issue` with:
   - title: Generated title
   - team: Team ID (NOT team name - use the id field from list_teams)
   - description: Formatted template content
   - labels: One of [Feature, Bug, Refactor, Research, Chore]

Feature template example:

Title: Integrate AniSearch Crawlers into API Helper System

Description:

### What

Integration of AniSearch crawlers into enrichment pipeline

### Why

Replace legacy scraper with robust browser automation for reliable data extraction

### Acceptance Criteria

- [ ] Helper successfully fetches and aggregates anime, episode, and character data
- [ ] Graceful degradation if optional data fails

### Testing / Validation

- [ ] Test suite at `tests/enrichment/api_helpers/test_anisearch_helper.py` with 38 cases
- [ ] 100% code coverage (100 statements, 0 missed)
- [ ] Integration test with Dandadan (ID: 18878) passed

### Implementation Notes (optional)

- Created `src/enrichment/api_helpers/anisearch_helper.py` with `AniSearchEnrichmentHelper` class
- Integrated `anisearch_anime_crawler`, `anisearch_episode_crawler`, `anisearch_character_crawler`
- Added specific exception handling (`AttributeError`, `TypeError`, `ValueError`)

### Related Issues / Dependencies (optional)

- `crawl4ai>=0.7.4` requires `chardet>=5.2.0`
- `animeplanet_helper.py` reference pattern
