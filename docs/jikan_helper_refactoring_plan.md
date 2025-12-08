# Jikan Helper Refactoring Plan

**Status**: Planned for future PR
**Priority**: Medium
**Effort**: 6-8 hours
**Created**: 2025-01-21

## Executive Summary

The current `JikanDetailedFetcher` class violates the Single Responsibility Principle by combining episode and character fetching into a single class with a type discriminator parameter. This creates:

- **Architectural inconsistency**: Other helpers (AniSearch, AnimePlanet) use composition with separate methods
- **Code duplication**: ~70 lines duplicated between `fetch_episode_detail()` and `fetch_character_detail()`
- **Maintenance complexity**: Branching on `data_type` in 4 locations throughout the code
- **Testing overhead**: Must test both types for every feature (rate limiting, caching, retries, batching)

**Recommended Solution**: Refactor to composition pattern with separate methods (`fetch_episodes()`, `fetch_characters()`, `fetch_anime()`), matching the architecture of other enrichment helpers.

---

## Current Architecture Problems

### 1. Single Responsibility Principle Violation

**Location**: `src/enrichment/api_helpers/jikan_helper.py:25-397`

The `JikanDetailedFetcher` class has **two reasons to change**:
1. Changes to episode fetching logic (API endpoints, field mapping, parsing)
2. Changes to character fetching logic (different endpoints, different fields)

**Evidence**:
```python
class JikanDetailedFetcher:
    def __init__(self, anime_id: str, data_type: str, session: Optional[Any] = None):
        self.data_type = data_type  # 'episodes' or 'characters'
        # Type discriminator is code smell

    async def fetch_detailed_data(self, input_file: str, output_file: str) -> None:
        # Branching on type in 4 locations:
        if self.data_type == "episodes":
            # Episode-specific logic (lines 291-306)
        else:  # characters
            # Character-specific logic (lines 307-345)
```

### 2. Code Duplication

**Duplicated code between methods** (~70 lines):

- HTTP request boilerplate (lines 101-156 vs 173-227)
- 429 retry logic (identical in both methods)
- Cache hit detection (`from_cache` attribute check)
- Error handling (identical try-except structure)
- Rate limiting hooks (identical `_record_network_request` calls)

**Example duplication**:
```python
# fetch_episode_detail (lines 125-142)
from_cache = (
    isinstance(getattr(response, "from_cache", None), bool)
    and response.from_cache
)
if response.status == 200:
    data = await response.json()
    await self._record_network_request(from_cache)
    return {...}
elif response.status == 429:
    if retry_count >= 3:
        return None
    await asyncio.sleep(5)
    return await self.fetch_episode_detail(...)

# fetch_character_detail (lines 190-207) - IDENTICAL LOGIC
from_cache = (...)  # Same code
if response.status == 200: # Same structure
    ...
elif response.status == 429:  # Same retry logic
    ...
```

### 3. Architectural Inconsistency

**Other helpers use composition**, not type discrimination:

#### AniSearch Helper Pattern
```python
# src/enrichment/api_helpers/anisearch_helper.py
class AniSearchEnrichmentHelper:
    async def fetch_anime_data(self, anisearch_id: int) -> Dict
    async def fetch_episode_data(self, anisearch_id: int) -> List[Dict]
    async def fetch_character_data(self, anisearch_id: int) -> Dict

    async def fetch_all_data(self, anisearch_id: int) -> Dict:
        # Composition: call individual methods
        return {
            "anime": await self.fetch_anime_data(anisearch_id),
            "episodes": await self.fetch_episode_data(anisearch_id),
            "characters": await self.fetch_character_data(anisearch_id),
        }
```

#### AnimePlanet Helper Pattern
```python
# src/enrichment/api_helpers/animeplanet_helper.py
class AnimePlanetEnrichmentHelper:
    async def fetch_anime_data(self, anime_slug: str) -> Dict
    async def fetch_character_data(self, anime_slug: str) -> Dict

    # Separate methods, no type parameter
```

**Jikan is the only helper** with a type discriminator parameter in the constructor.

### 4. Maintenance Complexity

**Branching on `data_type` appears in 4 locations**:

1. **Input data parsing** (lines 291-311)
2. **Item processing loop** (lines 338-345)
3. **Sorting results** (lines 380-387)
4. **Logging messages** (lines 335, 352, 358, 367, 377, 383, 387)

**Risk**: When adding features or fixing bugs, developers must remember to update **all 4 branches**, increasing the risk of inconsistencies.

### 5. Testing Complexity

**Test file**: `tests/api_helpers/test_jikan_helper.py` (1485 lines)

Tests must cover:
- Both data types for every feature (rate limiting, caching, retries, batching)
- Combinations of parameters (cache hit for episodes, cache miss for characters)
- Edge cases doubled (empty lists for both, all failures for both)

**Example**: Rate limiting tests duplicated for episodes and characters.

---

## Why Episodes and Characters Are Combined

### Shared Infrastructure (95% overlap)

Both data types share:

1. **Rate Limiting** (Jikan API: 3 req/sec, 60 req/min)
   - Same rate limits for all Jikan endpoints
   - Same cache-aware rate limiting logic
   - Methods: `respect_rate_limits()`, `_record_network_request()`

2. **Session Management**
   - Both use cached aiohttp session from `http_cache_manager`
   - Ownership tracking (`_owns_session` flag)
   - Context manager protocol (`__aenter__`, `__aexit__`, `close()`)

3. **Batch Processing & Progress Tracking**
   - Identical batch file append logic
   - Same progress file resume capability (`.progress` files)
   - Same batch size (50 items)

4. **HTTP Request Pattern**
   - Same retry logic for 429 errors (3 max retries, 5s wait)
   - Same cache hit detection (`from_cache` attribute)
   - Same error handling

### What's Different (5% divergence)

1. **API Endpoint URLs**:
   - Episodes: `https://api.jikan.moe/v4/anime/{anime_id}/episodes/{episode_id}`
   - Characters: `https://api.jikan.moe/v4/characters/{character_id}`

2. **Input Data Structure**:
   - Episodes: List of episode IDs or episode count
   - Characters: List of character objects with nested `character.mal_id`

3. **Response Field Mapping**:
   - Episodes: 10 fields (title, aired, score, filler, recap, synopsis, etc.)
   - Characters: 8 fields (name, nicknames, about, images, role, voice_actors, etc.)

4. **Sorting Key**:
   - Episodes: `episode_number`
   - Characters: `character_id`

**Analysis**: The 5% difference doesn't justify the architectural complexity of type discrimination. Composition with shared infrastructure is a better fit.

---

## Recommended Design: Composition Pattern

### Architecture Overview

**Goal**: Match the pattern used by AniSearch and AnimePlanet helpers while preserving shared infrastructure.

### Proposed API

```python
class JikanEnrichmentHelper:
    """
    Fetches data from Jikan API with proper rate limiting and caching.
    Supports anime metadata, episodes, and characters.
    """

    def __init__(self, anime_id: str, session: Optional[Any] = None):
        """
        Initialize Jikan helper for a specific anime.

        Args:
            anime_id: MyAnimeList anime ID
            session: Optional pre-existing aiohttp session (for connection pooling)
        """
        self.anime_id = anime_id
        self._owns_session = session is None
        self.session = session or _cache_manager.get_aiohttp_session("jikan")

        # Shared rate limiting state
        self.request_count = 0
        self.start_time = time.time()
        self.max_requests_per_second = 3
        self.max_requests_per_minute = 60

    # Shared infrastructure methods
    async def respect_rate_limits(self) -> None:
        """Ensure we don't exceed Jikan API rate limits."""
        # Existing logic (lines 47-79)

    async def _record_network_request(self, from_cache: bool) -> None:
        """Increment counters for real network hits and apply pacing."""
        # Existing logic (lines 81-85)

    async def _fetch_with_retry(
        self,
        url: str,
        retry_count: int = 0
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch URL with automatic retry on 429 errors.

        This extracts the common HTTP request logic shared by all fetch methods.
        """
        try:
            async with self.session.get(url) as response:
                from_cache = (
                    isinstance(getattr(response, "from_cache", None), bool)
                    and response.from_cache
                )

                if response.status == 200:
                    data = await response.json()
                    await self._record_network_request(from_cache)
                    return data

                elif response.status == 429:
                    if retry_count >= 3:
                        logger.warning(f"Max retries reached for {url}")
                        return None

                    logger.info(f"Rate limit hit, waiting (attempt {retry_count + 1}/3)...")
                    await asyncio.sleep(5)
                    await self._record_network_request(from_cache)
                    return await self._fetch_with_retry(url, retry_count + 1)

                else:
                    logger.error(f"HTTP {response.status} for {url}")
                    await self._record_network_request(from_cache)
                    return None

        except Exception as e:
            logger.error(f"Request failed for {url}: {e}")
            return None

    # Public API methods (composition)

    async def fetch_anime(self) -> Optional[Dict[str, Any]]:
        """
        Fetch anime metadata from /anime/{id}/full endpoint.

        Returns:
            Anime data dict or None if failed
        """
        url = f"https://api.jikan.moe/v4/anime/{self.anime_id}/full"
        response_data = await self._fetch_with_retry(url)

        if response_data and "data" in response_data:
            return response_data["data"]
        return None

    async def fetch_episodes(
        self,
        episode_count: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch detailed episode data.

        Args:
            episode_count: Number of episodes to fetch. If None, fetches from anime metadata.

        Returns:
            List of episode dicts with detailed information
        """
        if episode_count is None:
            anime_data = await self.fetch_anime()
            if not anime_data:
                return []
            episode_count = anime_data.get("episodes", 0)

        if episode_count == 0:
            return []

        episodes = []
        for episode_id in range(1, episode_count + 1):
            episode_detail = await self._fetch_episode_detail(episode_id)
            if episode_detail:
                episodes.append(episode_detail)

        return episodes

    async def fetch_characters(self) -> List[Dict[str, Any]]:
        """
        Fetch detailed character data.

        Returns:
            List of character dicts with detailed information
        """
        # First get character list
        url = f"https://api.jikan.moe/v4/anime/{self.anime_id}/characters"
        characters_basic = await self._fetch_with_retry(url)

        if not characters_basic or "data" not in characters_basic:
            return []

        # Then fetch details for each character
        characters = []
        for char_data in characters_basic["data"]:
            character_detail = await self._fetch_character_detail(char_data)
            if character_detail:
                characters.append(character_detail)

        return characters

    async def fetch_all_data(self) -> Dict[str, Any]:
        """
        Fetch all data (anime, episodes, characters) in sequence.

        Note: Sequential execution respects Jikan's 3 req/sec limit.

        Returns:
            Dict with keys: 'anime', 'episodes', 'characters'
        """
        anime_data = await self.fetch_anime()
        episodes_data = await self.fetch_episodes()
        characters_data = await self.fetch_characters()

        return {
            "anime": anime_data,
            "episodes": episodes_data,
            "characters": characters_data,
        }

    # Private helper methods

    async def _fetch_episode_detail(self, episode_id: int) -> Optional[Dict[str, Any]]:
        """Fetch single episode detail."""
        url = f"https://api.jikan.moe/v4/anime/{self.anime_id}/episodes/{episode_id}"
        response_data = await self._fetch_with_retry(url)

        if response_data and "data" in response_data:
            episode = response_data["data"]
            return {
                "episode_number": episode_id,
                "url": episode.get("url"),
                "title": episode.get("title"),
                "title_japanese": episode.get("title_japanese"),
                "title_romaji": episode.get("title_romaji"),
                "aired": episode.get("aired"),
                "score": episode.get("score"),
                "filler": episode.get("filler", False),
                "recap": episode.get("recap", False),
                "duration": episode.get("duration"),
                "synopsis": episode.get("synopsis"),
            }
        return None

    async def _fetch_character_detail(
        self,
        character_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Fetch single character detail."""
        character_id = character_data["character"]["mal_id"]
        url = f"https://api.jikan.moe/v4/characters/{character_id}"
        response_data = await self._fetch_with_retry(url)

        if response_data and "data" in response_data:
            char = response_data["data"]
            return {
                "character_id": character_id,
                "url": char.get("url"),
                "name": char.get("name"),
                "name_kanji": char.get("name_kanji"),
                "nicknames": char.get("nicknames", []),
                "about": char.get("about"),
                "images": char.get("images", {}),
                "favorites": char.get("favorites"),
                "role": character_data.get("role"),
                "voice_actors": character_data.get("voice_actors", []),
            }
        return None

    # Context manager protocol (unchanged)
    async def close(self) -> None:
        """Close the underlying HTTP session if we created it."""
        if self._owns_session and self.session:
            await self.session.close()

    async def __aenter__(self) -> "JikanEnrichmentHelper":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        await self.close()
        return False
```

### Key Design Decisions

1. **Composition over Type Discrimination**
   - Separate methods: `fetch_anime()`, `fetch_episodes()`, `fetch_characters()`
   - No `data_type` parameter in constructor
   - Caller decides what to fetch

2. **Shared Infrastructure Preserved**
   - All methods use shared `_fetch_with_retry()` (eliminates duplication)
   - Shared rate limiting state (request_count, start_time)
   - Shared session management

3. **Consistent with Other Helpers**
   - Matches AniSearch pattern (separate methods + fetch_all_data())
   - Matches AnimePlanet pattern (no type parameter)

4. **Backward Compatibility**
   - Keep old `JikanDetailedFetcher` as deprecated wrapper
   - Or update callsites directly (only 2 files: api_fetcher.py, CLI)

---

## Comparison: Before vs After

### Before (Current)

```python
# Usage in ParallelAPIFetcher
episode_fetcher = JikanDetailedFetcher(mal_id, "episodes", session=self.jikan_session)
await episode_fetcher.fetch_detailed_data("episodes_input.json", "episodes_output.json")

character_fetcher = JikanDetailedFetcher(mal_id, "characters", session=self.jikan_session)
await character_fetcher.fetch_detailed_data("characters_input.json", "characters_output.json")

# CLI Usage
python -m src.enrichment.api_helpers.jikan_helper episodes 21 input.json output.json
```

**Problems**:
- Two separate instances for same anime
- Type discriminator parameter
- File-based I/O baked into API

### After (Proposed)

```python
# Usage in ParallelAPIFetcher
async with JikanEnrichmentHelper(mal_id, session=self.jikan_session) as helper:
    anime_data = await helper.fetch_anime()
    episodes_data = await helper.fetch_episodes()
    characters_data = await helper.fetch_characters()

    # Or fetch all at once
    all_data = await helper.fetch_all_data()

# CLI Usage
python -m src.enrichment.api_helpers.jikan_helper anime 21 --output anime.json
python -m src.enrichment.api_helpers.jikan_helper episodes 21 --output episodes.json
python -m src.enrichment.api_helpers.jikan_helper characters 21 --output characters.json
```

**Benefits**:
- Single instance per anime
- Clearer intent (method name indicates what's fetched)
- File I/O separated from data fetching logic
- Easier to fetch just what you need

---

## Adding Anime Metadata Fetching

### Current Gap

**Problem**: Anime metadata is currently fetched in `ParallelAPIFetcher._fetch_jikan_complete()` (lines 189-198), but there's **no standalone way** to fetch just anime metadata via:
- CLI script
- Programmatic import
- JikanEnrichmentHelper class

**Current workaround**: Must use `ParallelAPIFetcher` even if you only need anime metadata.

### Proposed Solution

Add `fetch_anime()` method to the new `JikanEnrichmentHelper` class (see design above).

#### CLI Support

```bash
# Fetch only anime metadata
python -m src.enrichment.api_helpers.jikan_helper anime 21 --output anime.json

# Fetch only episodes
python -m src.enrichment.api_helpers.jikan_helper episodes 21 --output episodes.json

# Fetch only characters
python -m src.enrichment.api_helpers.jikan_helper characters 21 --output characters.json

# Fetch all (anime + episodes + characters)
python -m src.enrichment.api_helpers.jikan_helper all 21 --output-dir temp/
```

#### Programmatic Usage

```python
from src.enrichment.api_helpers.jikan_helper import JikanEnrichmentHelper

async with JikanEnrichmentHelper("21") as helper:
    # Fetch just anime metadata
    anime_data = await helper.fetch_anime()

    # Or fetch selectively
    episodes = await helper.fetch_episodes()

    # Or fetch everything
    all_data = await helper.fetch_all_data()
```

### Endpoint Details

**Jikan Anime Full Endpoint**: `https://api.jikan.moe/v4/anime/{mal_id}/full`

**Returns**:
- All anime metadata (title, synopsis, genres, studios, etc.)
- Statistics (score, rank, popularity, members)
- Relations (sequels, prequels, side stories)
- Streaming links
- **Does NOT include**: Detailed episodes or character bios (separate endpoints)

**Rate Limiting**: Same 3 req/sec, 60 req/min limits apply

---

## Migration Strategy

### Phase 1: Implement New API (Non-Breaking)

1. Create `JikanEnrichmentHelper` class with composition pattern
2. Add `fetch_anime()`, `fetch_episodes()`, `fetch_characters()` methods
3. Extract shared `_fetch_with_retry()` method
4. Add comprehensive tests for new API
5. Keep old `JikanDetailedFetcher` class untouched

**Status**: Both old and new APIs coexist

### Phase 2: Update Callsites (Optional)

**Option A: Update ParallelAPIFetcher**

```python
# Before
episode_fetcher = JikanDetailedFetcher(mal_id, "episodes", session=self.jikan_session)
await episode_fetcher.fetch_detailed_data(input_file, output_file)

# After
async with JikanEnrichmentHelper(mal_id, session=self.jikan_session) as helper:
    episodes = await helper.fetch_episodes()
    # Write to file if needed
    with open(output_file, "w") as f:
        json.dump(episodes, f, indent=2)
```

**Option B: Keep Backward Compatibility**

Create wrapper function in old `JikanDetailedFetcher` that delegates to new API:

```python
# Deprecated wrapper for backward compatibility
class JikanDetailedFetcher:
    def __init__(self, anime_id: str, data_type: str, session: Optional[Any] = None):
        warnings.warn(
            "JikanDetailedFetcher is deprecated. Use JikanEnrichmentHelper instead.",
            DeprecationWarning,
            stacklevel=2
        )
        self.helper = JikanEnrichmentHelper(anime_id, session)
        self.data_type = data_type

    async def fetch_detailed_data(self, input_file: str, output_file: str) -> None:
        if self.data_type == "episodes":
            data = await self.helper.fetch_episodes()
        elif self.data_type == "characters":
            data = await self.helper.fetch_characters()
        else:
            raise ValueError(f"Invalid data_type: {self.data_type}")

        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)
```

### Phase 3: Deprecate Old API (Future)

1. Add deprecation warnings to `JikanDetailedFetcher`
2. Update documentation to recommend new API
3. Create migration guide
4. Remove old API in next major version

---

## Implementation Checklist

### Core Implementation

- [ ] Create `JikanEnrichmentHelper` class in `src/enrichment/api_helpers/jikan_helper.py`
- [ ] Implement `__init__()` with anime_id and session parameters
- [ ] Extract shared `_fetch_with_retry()` method from duplicate code
- [ ] Implement `fetch_anime()` method (new capability)
- [ ] Implement `fetch_episodes()` method (refactored from old code)
- [ ] Implement `fetch_characters()` method (refactored from old code)
- [ ] Implement `fetch_all_data()` composition method
- [ ] Implement private `_fetch_episode_detail()` helper
- [ ] Implement private `_fetch_character_detail()` helper
- [ ] Preserve rate limiting methods (`respect_rate_limits`, `_record_network_request`)
- [ ] Preserve context manager protocol (`__aenter__`, `__aexit__`, `close`)

### CLI Updates

- [ ] Update `main()` function to support new commands: `anime`, `episodes`, `characters`, `all`
- [ ] Update argparse to handle new subcommands
- [ ] Add `--output` and `--output-dir` options
- [ ] Update help text and examples
- [ ] Keep backward compatibility for old CLI usage (optional)

### Callsite Updates

- [ ] Update `ParallelAPIFetcher._fetch_jikan_complete()` to use new API
- [ ] Remove file-based I/O from ParallelAPIFetcher (data returned directly)
- [ ] Update error handling for new return types

### Tests

- [ ] Create `tests/api_helpers/test_jikan_enrichment_helper.py`
- [ ] Test `fetch_anime()` method
  - Success case (anime metadata returned)
  - 404 case (anime not found)
  - 429 retry logic
  - Cache hit detection
- [ ] Test `fetch_episodes()` method
  - Multiple episodes fetched sequentially
  - Rate limiting applied correctly
  - Empty episode list handling
- [ ] Test `fetch_characters()` method
  - Character list fetched first
  - Details fetched for each character
  - Empty character list handling
- [ ] Test `fetch_all_data()` method
  - All three methods called in sequence
  - Partial failures handled gracefully
- [ ] Test `_fetch_with_retry()` shared logic
  - 429 retry (max 3 attempts)
  - Non-429 errors don't retry
  - Cache hits skip rate limiting
- [ ] Test context manager protocol
  - Session closed on exit (if owned)
  - Session not closed if provided
- [ ] Test CLI
  - Each subcommand (anime, episodes, characters, all)
  - File output works correctly
- [ ] Integration tests with real Jikan API (optional)

### Documentation

- [ ] Update `src/enrichment/README.md` with new API examples
- [ ] Document `fetch_anime()` capability
- [ ] Update dual-usage pattern section
- [ ] Add migration guide from old API to new API
- [ ] Update docstrings for all public methods

### Backward Compatibility (Optional)

- [ ] Create deprecated wrapper `JikanDetailedFetcher` class
- [ ] Add deprecation warnings
- [ ] Update tests to suppress deprecation warnings
- [ ] Document deprecation timeline

---

## Test Strategy

### Unit Tests

**File**: `tests/api_helpers/test_jikan_enrichment_helper.py`

**Scope**: Test individual methods in isolation with mocked HTTP responses

**Key Test Cases**:

1. **Rate Limiting**
   - Verify 0.5s delay between requests
   - Verify request counter resets every 60 seconds
   - Verify cache hits skip rate limiting

2. **Retry Logic**
   - 429 errors trigger retry (max 3 attempts, 5s wait)
   - Non-429 errors don't retry
   - Successful retry returns data

3. **Session Management**
   - Session created if not provided
   - Session closed on context manager exit (if owned)
   - Session not closed if provided (borrowed)

4. **Data Fetching**
   - `fetch_anime()` returns correct fields
   - `fetch_episodes()` returns list of episodes
   - `fetch_characters()` returns list of characters
   - `fetch_all_data()` returns combined dict

5. **Error Handling**
   - 404 returns None
   - Network errors handled gracefully
   - Invalid JSON handled gracefully

### Integration Tests

**File**: `tests/api_helpers/integration/test_jikan_helper_integration.py`

**Scope**: Test with real Jikan API (optional, slow)

**Key Test Cases**:

1. Fetch real anime data (e.g., MAL ID 21 = One Piece)
2. Verify rate limiting works with real API
3. Verify caching reduces requests
4. Verify retry on actual 429 errors

### CLI Tests

**File**: `tests/api_helpers/test_jikan_helper_cli.py`

**Scope**: Test CLI argument parsing and file output

**Key Test Cases**:

1. Each subcommand produces correct file output
2. Invalid arguments show helpful error messages
3. Help text displays correctly

---

## Success Criteria

### Functional Requirements

- [ ] Can fetch anime metadata standalone (new capability)
- [ ] Can fetch episodes (existing capability preserved)
- [ ] Can fetch characters (existing capability preserved)
- [ ] Can fetch all data in one call
- [ ] Rate limiting works correctly (no 429 errors under normal usage)
- [ ] Caching works (cache hits skip rate limiting)
- [ ] CLI works for all subcommands

### Non-Functional Requirements

- [ ] All existing tests pass (no regressions)
- [ ] New tests achieve >90% coverage
- [ ] No code duplication between fetch methods
- [ ] Consistent with other enrichment helpers (AniSearch, AnimePlanet)
- [ ] Documentation updated and clear
- [ ] Performance equivalent or better than old API

### Code Quality

- [ ] Passes mypy type checking
- [ ] Passes pytest with no warnings
- [ ] Code follows project style guide
- [ ] All public methods have docstrings
- [ ] Complex logic has inline comments

---

## Impact Assessment

### Files Changed

| File | Lines Changed | Risk | Notes |
|------|---------------|------|-------|
| `src/enrichment/api_helpers/jikan_helper.py` | +200, -100 | Medium | Core refactoring |
| `src/enrichment/programmatic/api_fetcher.py` | ±30 | Low | Update callsite |
| `tests/api_helpers/test_jikan_enrichment_helper.py` | +500 | Low | New tests |
| `tests/api_helpers/test_jikan_helper.py` | -200 | Low | Remove duplicated tests |
| `src/enrichment/README.md` | +50 | Low | Documentation update |

**Total Estimated LOC**: +450, -300 (net +150 lines due to reduced duplication)

### Breaking Changes

**None** if using backward compatibility wrapper.

**If updating callsites directly**:
- `ParallelAPIFetcher._fetch_jikan_complete()` needs minor updates
- CLI usage remains the same (can add new subcommands without breaking old usage)

### Performance Impact

**Neutral to positive**:
- Shared `_fetch_with_retry()` reduces code paths
- Rate limiting behavior unchanged
- Caching behavior unchanged
- Sequential execution (not parallel) preserves same throughput

---

## Future Enhancements

### 1. Batch Fetching with Progress Tracking

The current design fetches items sequentially without progress tracking or resume capability. Consider adding:

```python
async def fetch_episodes_batch(
    self,
    episode_count: int,
    batch_size: int = 50,
    progress_callback: Optional[Callable] = None
) -> List[Dict[str, Any]]:
    """Fetch episodes with batch processing and progress tracking."""
    # Implementation similar to old fetch_detailed_data()
```

### 2. Parallel Fetching (with Rate Limiting)

The new design fetches sequentially. For large series (1000+ episodes), consider parallel fetching with a semaphore:

```python
async def fetch_episodes_parallel(
    self,
    episode_count: int,
    max_concurrent: int = 3  # Respects 3 req/sec limit
) -> List[Dict[str, Any]]:
    """Fetch episodes in parallel with rate limiting."""
    semaphore = asyncio.Semaphore(max_concurrent)
    # Implementation with semaphore-controlled parallelism
```

### 3. Incremental Updates

Add methods to fetch only new episodes/characters since last fetch:

```python
async def fetch_new_episodes(
    self,
    last_episode_number: int
) -> List[Dict[str, Any]]:
    """Fetch only episodes after the specified episode number."""
```

### 4. Selective Field Fetching

Allow caller to specify which fields to fetch (reduce bandwidth):

```python
async def fetch_episodes(
    self,
    fields: Optional[List[str]] = None  # e.g., ["title", "aired", "synopsis"]
) -> List[Dict[str, Any]]:
    """Fetch episodes with only specified fields."""
```

---

## References

### Related Files

- `src/enrichment/api_helpers/jikan_helper.py` - Current implementation
- `src/enrichment/api_helpers/anisearch_helper.py` - Reference pattern (composition)
- `src/enrichment/api_helpers/animeplanet_helper.py` - Reference pattern (composition)
- `src/enrichment/programmatic/api_fetcher.py` - Main callsite
- `tests/api_helpers/test_jikan_helper.py` - Current tests

### External Documentation

- [Jikan API v4 Documentation](https://docs.api.jikan.moe/)
- [Jikan API Rate Limits](https://docs.api.jikan.moe/#section/Information/Rate-Limiting)
- Jikan Endpoints:
  - `/anime/{id}/full` - Complete anime metadata
  - `/anime/{id}/episodes/{episode_id}` - Detailed episode data
  - `/characters/{id}` - Detailed character data

### Design Patterns

- **Composition over Inheritance**: Favor object composition over class inheritance
- **Single Responsibility Principle**: A class should have only one reason to change
- **Open-Closed Principle**: Open for extension, closed for modification
- **Strategy Pattern**: Define family of algorithms, encapsulate each, make them interchangeable

---

## Appendix: Full Code Example

See **Recommended Design: Composition Pattern** section for complete implementation.

---

**Document Version**: 1.0
**Last Updated**: 2025-01-21
**Next Review**: After PR implementation
