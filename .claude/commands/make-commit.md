---
description: Generate and create a conventional commit message
---

You are helping create a semantic versioning commit following Conventional Commits format.

**Instructions:**

1. Run `git status` and `git diff --cached` to see staged changes
2. If nothing is staged, show `git status` and ask what to stage
3. **Analyze changes by type** - Group changes into logical commits:
   - **CRITICAL**: If changes contain MULTIPLE types (feat + fix, feat + chore, etc.), they MUST be split into separate commits
   - Identify distinct change types: feat, fix, docs, style, refactor, test, chore, perf, ci, build, revert
   - Group related files by their semantic purpose and type
   - **Never mix different types in one commit** (e.g., don't combine feat + fix)

4. **For EACH logical group**, determine:
   - **Type**: feat, fix, docs, style, refactor, test, chore, perf, ci, build, revert
   - **Scope** (optional): component/module affected
   - **Subject**: Concise description (imperative mood, max 50 chars, no period)
   - **Body** (optional): Why the change was made (wrap at 72 chars)
   - **Breaking Change** (if applicable): Note in footer with "BREAKING CHANGE:"
   - **Files**: Which specific files belong to this commit
   - **IMPORTANT**: No emojis in commit messages
   - **IMPORTANT**: No Co-Authored-By or AI generation footers

5. **Present commit plan** to user:
   ```
   Commit Plan (X commits required):

   Commit 1 (feat):
   - file1.py
   - file2.py
   Subject: add new feature X

   Commit 2 (fix):
   - file3.py
   Subject: resolve bug in Y

   Commit 3 (docs):
   - README.md
   Subject: update documentation for Z
   ```

6. **After user approval**, execute commits in sequence:
   - Unstage all: `git reset HEAD`
   - For each commit: stage specific files → commit → repeat
   - Format: `git add <files> && git commit -m "..." -m "..." -m "..."`
   - **NEVER add**: Emojis, "Generated with Claude Code", Co-Authored-By, or AI attribution

**Version Impact Guide:**
- `feat:` → Minor version bump (0.X.0)
- `fix:` → Patch version bump (0.0.X)
- `BREAKING CHANGE:` → Major version bump (X.0.0)
- Others → No version bump

**Examples:**

**Single type commit:**
```
feat: add character similarity search endpoint

Implements cosine similarity search for character vectors
using Qdrant's vector search capabilities.

Closes #42
```

**Multiple types (SPLIT INTO SEPARATE COMMITS):**

**BAD** - Mixed types in one commit:
```
INCORRECT: feat: add search endpoint and fix timeout bug
```

**GOOD** - Split into logical commits:
```
Commit 1:
feat: add character similarity search endpoint

Commit 2:
fix(api): resolve timeout in multimodal search
```

**Real-world splitting example:**

If staged changes include:
- New API endpoint (feat)
- Bug fix in existing endpoint (fix)
- Updated README (docs)
- Moved scripts to scripts/ folder (chore)

Then create 4 separate commits:
1. `feat(api): add character similarity endpoint`
2. `fix(api): resolve connection pool exhaustion`
3. `docs: update API usage examples in README`
4. `chore: reorganize scripts into scripts/ directory`

**Arguments** (optional):
- `$ARGUMENTS`: Custom context or message hint

If arguments provided: "$ARGUMENTS"
Use this as additional context when generating the commit message.
