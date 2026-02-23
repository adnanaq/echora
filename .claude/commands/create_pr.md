Create a pull request based on Linear issue $ARGUMENTS.

1. Fetch issue using `mcp__linear-server__get_issue` with ID $ARGUMENTS
2. Extract issue title and label
3. Determine commit type from label: Feature → feat, Bug → fix, Refactor → refactor, Research → docs, Chore → chore
4. Check git status and recent commits to understand what changed
5. Generate branch name: `<type>/<issue-id>-<slugified-title>` (lowercase, hyphens)
6. Generate PR title: `<type>(<issue-id>): <Title>`
7. Generate PR body by filling in `.github/PULL_REQUEST_TEMPLATE.md`:
   - **Type of Change**: check the box(es) matching the commit types present
   - **Summary**: brief description of what the PR does and why
   - **Changes**: detailed bullet list of every meaningful change made
   - **Testing**: how the changes were tested (unit tests, manual, integration, etc.)
   - **Related Issues**: `Closes <issue-id>` using the Linear issue ID
   - **Breaking Changes**: describe any breaking changes; "None" if absent
   - **Questions for Reviewers**: flag anything needing extra attention; omit section if none
   - **Checklist**: check all items that apply based on the actual work done
8. Do NOT add emojis or co-author info
9. Show the user the complete `gh pr create` command with all generated content

Example:
Issue ECHO-10: "Add AniSearch crawler", label "Feature"
Branch: `feat/ECHO-10-add-anisearch-crawler`
Title: `feat(ECHO-10): Add AniSearch crawler`
Body: PULL_REQUEST_TEMPLATE.md filled in from git diff and changes analysis
