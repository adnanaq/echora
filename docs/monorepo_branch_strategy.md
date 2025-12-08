# Monorepo Branch Strategy

## Current State

- **Base Branch**: `refactor/ECHO-31-refactor-qdrantclient-to-async-factory-pattern-with-dependency-injection`
- **New Branch**: `chore/ECHO-32-initialize-pants-build-system-for-monorepo-migration`
- **Strategy**: Option A - Branch now, rebase later

## Branch Creation

```bash
# Create new branch from current ECHO-31 branch
git checkout refactor/ECHO-31-refactor-qdrantclient-to-async-factory-pattern-with-dependency-injection
git checkout -b chore/ECHO-32-initialize-pants-build-system-for-monorepo-migration
```

## Development Phase

Work on Pants initialization and monorepo setup commits on `chore/ECHO-32-initialize-pants-build-system-for-monorepo-migration`.

**Note**: This branch currently contains all commits from ECHO-31 branch. This is temporary.

## Rebase Phase (AFTER ECHO-31 MERGES TO MAIN)

**CRITICAL**: Once `refactor/ECHO-31-refactor-qdrantclient-to-async-factory-pattern-with-dependency-injection` is merged to `main`, execute this rebase:

```bash
# Update main
git checkout main
git pull origin main

# Rebase monorepo branch onto main, excluding ECHO-31 commits
git checkout chore/ECHO-32-initialize-pants-build-system-for-monorepo-migration
git rebase --onto main refactor/ECHO-31-refactor-qdrantclient-to-async-factory-pattern-with-dependency-injection chore/ECHO-32-initialize-pants-build-system-for-monorepo-migration
```

**What this does**: Replays ONLY the monorepo-specific commits on top of `main`, excluding all ECHO-31 commits (which are already in `main`).

## Result

Clean history with:
- `main` branch commits
- Only monorepo-specific commits on top

## Related Issues

- Linear Issue: ECHO-32 (Initialize Pants build system for monorepo migration)
- Base Issue: ECHO-31 (QdrantClient refactoring)
