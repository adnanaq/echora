# CCIP Location & Dependency Conflict Resolution Plan

## Problem Statement

**Issue 1**: `dghs-imgutils==0.19.0` (required for CCIP) conflicts with `qdrant-client[fastembed]` (numpy<2 vs numpy>=2.1.0)

**Issue 2**: CCIP currently in `libs/vector_processing/` but monorepo plan says it should move to `libs/enrichment/`

**Goal**:

1. Determine correct location for CCIP in monorepo structure
2. Resolve dependency conflict using Pants multiple resolves
3. Create `libs/enrichment` library with proper structure

---

## Phase 1: Architectural Decision - CCIP Location

### 1.1 Current State

**CCIP Current Location:**

- `libs/vector_processing/src/vector_processing/legacy_ccips.py`

**Usage:**

- `src/enrichment/ai_character_matcher.py` imports: `from vector_processing.legacy_ccips import LegacyCCIPS`
- Used in Stage 5 character matching pipeline

**Monorepo Plan Directive** (MONOREPO_MIGRATION_PLAN.md line 184):

> "Move the `ccips` logic from `libs/vector_processing` into `libs/enrichment` to break the dependency"

**Problem:**

- `libs/enrichment` does not exist yet
- Needs to be created as part of monorepo migration (Phase 2)

### 1.2 Recommended Structure

**Create `libs/enrichment` with CCIP:**

```
libs/enrichment/
├── src/enrichment/
│   ├── __init__.py
│   ├── BUILD
│   ├── ai_character_matcher.py         # Moved from src/enrichment/
│   ├── similarity/                      # NEW: Similarity metrics module
│   │   ├── __init__.py
│   │   ├── BUILD
│   │   └── ccip.py                     # Moved from libs/vector_processing/legacy_ccips.py
│   ├── api_helpers/                     # Move from src/enrichment/
│   ├── crawlers/                        # Move from src/enrichment/
│   ├── scrapers/                        # Move from src/enrichment/
│   └── prompts/                         # Move from src/enrichment/
└── tests/
    ├── BUILD
    └── unit/
```

**Rationale:**

1. **Architectural Correctness**: CCIP is enrichment-specific (anime character similarity), not general vector processing
2. **Dependency Isolation**: Breaks circular dependency between enrichment and vector_processing
3. **Monorepo Alignment**: Follows MONOREPO_MIGRATION_PLAN.md directive
4. **Future Extensibility**: `similarity/` module can hold other enrichment-specific similarity metrics

### 1.3 Import Path Changes

**Before:**

```python
from vector_processing.legacy_ccips import LegacyCCIPS
```

**After:**

```python
from enrichment.similarity.ccip import CCIP  # Renamed LegacyCCIPS → CCIP
```

---

## Phase 2: Dependency Conflict Resolution

### 2.1 Exact Conflict (Verified with `uv sync`)

```
dghs-imgutils==0.19.0
  └─> numpy<2                      ❌ REQUIRES numpy < 2

qdrant-client[fastembed]==1.14.3
  └─> numpy>=2.1.0                 ❌ REQUIRES numpy >= 2.1.0
```

**Root Cause**: Incompatible `numpy` version constraints

### 2.2 Multiple Resolves Strategy

**Proposed resolves:**

```toml
[python.resolves]
resolves = {
    default = "python-default.lock",      # FastAPI, Qdrant, vector processing
    enrichment = "python-enrichment.lock" # Enrichment with CCIP (dghs-imgutils)
}
```

**Simplified from 3 to 2 resolves:**

- **default**: Core app, API, vector processing (qdrant-client with numpy>=2.1)
- **enrichment**: All enrichment code including CCIP (dghs-imgutils with numpy<2)
  - Includes `ai_character_matcher.py`, `crawlers/`, `api_helpers/`, etc.
  - Both CCIP and crawl4ai can coexist here (no conflict between them)

**Why 2 resolves instead of 3?**

- CCIP and crawl4ai don't conflict with each other (chardet conflict was in older version)
- Both can use numpy<2 (crawl4ai doesn't require numpy>=2.1)
- Simpler architecture with less cross-resolve communication

### 2.3 Package Assignment

| Resolve        | Packages                         | Location                                                                                |
| -------------- | -------------------------------- | --------------------------------------------------------------------------------------- |
| **default**    | Core app, API, vector processing | `src/main.py`, `src/api/`, `libs/vector_processing/`, `libs/qdrant_db/`, `libs/common/` |
| **enrichment** | All enrichment pipeline code     | `libs/enrichment/src/enrichment/**` (includes CCIP, crawlers, matchers)                 |

---

## Phase 3: Implementation Plan

### 3.1 Create `libs/enrichment` Structure

**Step 1: Create directories**

```bash
mkdir -p libs/enrichment/src/enrichment/similarity
mkdir -p libs/enrichment/tests/unit
```

**Step 2: Create BUILD files**

`libs/enrichment/src/enrichment/BUILD`:

```python
python_sources(
    resolve="enrichment",
    dependencies=[
        "//:reqs#dghs-imgutils",
        "//:reqs#crawl4ai",
        "libs/common/src/common:common",
    ],
)
```

`libs/enrichment/src/enrichment/similarity/BUILD`:

```python
python_sources(
    resolve="enrichment",
    dependencies=[
        "//:reqs#dghs-imgutils",
        "libs/common/src/common:common",
    ],
)
```

### 3.2 Move CCIP to `libs/enrichment`

**File movement:**

```bash
# Move and rename
mv libs/vector_processing/src/vector_processing/legacy_ccips.py \
   libs/enrichment/src/enrichment/similarity/ccip.py
```

**Update imports in `ccip.py`:**

```python
# Change class name
class CCIP:  # Was: LegacyCCIPS
    """Character Comparison Image Processing for anime character similarity."""
```

### 3.3 Move Enrichment Code from `src/` to `libs/`

**Move entire enrichment module:**

```bash
mv src/enrichment/* libs/enrichment/src/enrichment/
```

**Files to move:**

- `ai_character_matcher.py`
- `api_helpers/`
- `crawlers/`
- `scrapers/`
- `prompts/`
- `programmatic/`

### 3.4 Configure Pants Multiple Resolves

**File**: `pants.toml`

```toml
[python]
interpreter_constraints = ["==3.12.*"]
enable_resolves = true

[python.resolves]
resolves = {
    default = "python-default.lock",
    enrichment = "python-enrichment.lock"
}

[python.resolves_to_interpreter_constraints]
default = ["==3.12.*"]
enrichment = ["==3.12.*"]
```

### 3.5 Update `pyproject.toml`

**Add enrichment dependencies to optional-dependencies:**

```toml
[project.optional-dependencies]
enrichment = [
    "dghs-imgutils==0.19.0",
    "crawl4ai>=0.7.4",
    # Other enrichment-specific deps
]
```

### 3.6 Update Import Statements

**Files to update:**

1. `libs/enrichment/src/enrichment/ai_character_matcher.py`:

   ```python
   # Before
   from vector_processing.legacy_ccips import LegacyCCIPS

   # After
   from enrichment.similarity.ccip import CCIP
   ```

2. `scripts/process_stage5_characters.py` (if it imports CCIP):
   ```python
   from enrichment.similarity.ccip import CCIP
   from enrichment.ai_character_matcher import CharacterMatcher
   ```

### 3.7 Generate Lockfiles

```bash
./pants generate-lockfiles
```

Creates:

- `python-default.lock` (with numpy>=2.1.0)
- `python-enrichment.lock` (with numpy<2)

### 3.8 Create Executable for Stage 5 (Optional)

If Stage 5 needs to be run independently:

`apps/lambdas/process_stage5_characters/BUILD`:

```python
pex_binary(
    name="stage5",
    entry_point="scripts.process_stage5_characters:main",
    resolve="enrichment",
    dependencies=[
        "libs/enrichment/src/enrichment:enrichment",
    ],
)
```

---

## Phase 4: Testing & Validation

### 4.1 Validate Structure

```bash
# Check enrichment library structure
./pants list libs/enrichment::

# Check dependencies
./pants dependencies libs/enrichment/src/enrichment:enrichment
```

### 4.2 Validate Resolves

```bash
# Check no cross-resolve dependencies
./pants peek libs/enrichment/src/enrichment:enrichment
./pants peek libs/vector_processing/src/vector_processing:vector_processing

# Verify numpy versions in lockfiles
grep "numpy" python-default.lock       # Should show numpy>=2.1.0
grep "numpy" python-enrichment.lock    # Should show numpy<2
```

### 4.3 Test CCIP Functionality

```bash
# Test enrichment library independently
./pants test libs/enrichment::

# Test Stage 5 with CCIP
python scripts/process_stage5_characters.py Test_Agent --restart
```

---

## Phase 5: Cross-Resolve Communication

### 5.1 API Integration (If Needed)

**Scenario**: If default resolve (API) needs to call CCIP functionality

**Solution**: Subprocess execution via Stage 5 script

```python
# In API code (default resolve)
import subprocess
result = subprocess.run(
    ["python", "scripts/process_stage5_characters.py", agent_id],
    capture_output=True
)
```

**Why this works:**

- API (default resolve) doesn't directly import CCIP
- Stage 5 script uses enrichment resolve
- Communication via command-line interface

### 5.2 Enrichment Pipeline

**No cross-resolve communication needed:**

- Enrichment pipeline runs entirely in `enrichment` resolve
- All dependencies (CCIP, crawlers, matchers) available
- No need to call default resolve code

---

## Phase 6: Alternative Approaches (If Issues Arise)

### Alternative 1: Keep CCIP in `vector_processing`

**Pros:**

- Less file movement
- Existing imports work

**Cons:**

- ❌ Violates monorepo plan
- ❌ Circular dependency (enrichment ← vector_processing ← enrichment)
- ❌ CCIP not architecturally correct location

**Verdict:** Not recommended

### Alternative 2: Separate CCIP into Own Library

**Structure:**

```
libs/similarity_metrics/
└── src/similarity_metrics/
    └── ccip.py
```

**Pros:**

- Clean separation
- Reusable across projects

**Cons:**

- More complex (3rd library)
- Overkill for single use case

**Verdict:** Future consideration, not now

---

## Critical Files Reference

### Files to Create

1. `libs/enrichment/src/enrichment/__init__.py`
2. `libs/enrichment/src/enrichment/BUILD`
3. `libs/enrichment/src/enrichment/similarity/__init__.py`
4. `libs/enrichment/src/enrichment/similarity/BUILD`
5. `libs/enrichment/src/enrichment/similarity/ccip.py` (moved + renamed)
6. `libs/enrichment/tests/BUILD`
7. `python-enrichment.lock` (generated)

### Files to Move

1. `src/enrichment/ai_character_matcher.py` → `libs/enrichment/src/enrichment/`
2. `src/enrichment/api_helpers/` → `libs/enrichment/src/enrichment/`
3. `src/enrichment/crawlers/` → `libs/enrichment/src/enrichment/`
4. `src/enrichment/scrapers/` → `libs/enrichment/src/enrichment/`
5. `src/enrichment/prompts/` → `libs/enrichment/src/enrichment/`
6. `src/enrichment/programmatic/` → `libs/enrichment/src/enrichment/`
7. `libs/vector_processing/src/vector_processing/legacy_ccips.py` → `libs/enrichment/src/enrichment/similarity/ccip.py`

### Files to Modify

1. `pants.toml` - Add enrichment resolve
2. `pyproject.toml` - Add enrichment optional-dependencies
3. `libs/enrichment/src/enrichment/ai_character_matcher.py` - Update CCIP import
4. `scripts/process_stage5_characters.py` - Update imports (if needed)
5. Any other files importing from `src/enrichment/` or `legacy_ccips`

### Files to Delete (After Migration)

1. `src/enrichment/` directory (now empty)
2. Consider: `libs/vector_processing/src/vector_processing/legacy_ccips.py` backup

---

## Summary

**Architectural Decision:**

- Create `libs/enrichment` library
- Move CCIP to `libs/enrichment/src/enrichment/similarity/ccip.py`
- Move all `src/enrichment/` to `libs/enrichment/src/enrichment/`

**Dependency Resolution:**

- Use 2 Pants resolves (default + enrichment)
- Enrichment resolve includes both CCIP and crawl4ai (no conflict)
- Default resolve has qdrant-client with numpy>=2.1

**Benefits:**

1. ✅ Follows monorepo plan directive
2. ✅ Correct architectural placement
3. ✅ Breaks circular dependency
4. ✅ Resolves numpy conflict
5. ✅ Simplifies to 2 resolves instead of 3
6. ✅ All enrichment code isolated together

**Next Steps:**

1. Create `libs/enrichment` structure
2. Move CCIP and enrichment code
3. Configure Pants resolves
4. Update imports
5. Generate lockfiles
6. Test functionality
