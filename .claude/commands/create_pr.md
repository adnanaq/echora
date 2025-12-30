Create a pull request based on Linear issue $ARGUMENTS.

1. Fetch issue using `mcp__linear-server__get_issue` with ID $ARGUMENTS
2. Extract issue title and label
3. Determine commit type from label: Feature → feat, Bug → fix, Refactor → refactor, Research → docs, Chore → chore
4. Check git status and recent commits to understand what changed
5. Generate branch name: `<type>/<issue-id>-<slugified-title>` (lowercase, hyphens)
6. Generate PR title: `<type>(<issue-id>): <Title>`
7. Generate PR body based on actual code changes:
   - Summary of what was changed
   - List of modified files
   - Key implementation details
   - Testing information
8. Do NOT add emojis or co-author info
9. Show the user the complete `gh pr create` command with all generated content

Example:
Issue ECHO-10: "Add AniSearch crawler", label "Feature"
Branch: `feat/ECHO-10-add-anisearch-crawler`
Title: `feat(ECHO-10): Add AniSearch crawler`
Body: Generated from git diff and changes analysis
