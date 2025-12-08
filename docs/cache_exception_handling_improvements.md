# Cache Manager: Exception Handling Improvements

## Status
**Deferred to future PR** - Documented 2025-11-26

## Overview
This document combines two related exception handling improvements for the cache manager:
1. **Custom Exception Classes** - What we RAISE (improving error signaling)
2. **Narrow Exception Handling** - What we CATCH (improving error handling)

---

## Part 1: Custom Exception Classes (What We Raise)

### Review Comment Source
From code review of `src/cache_manager/async_redis_storage.py:92-93`:
> Optional: Consider using a custom exception class for validation errors.
>
> The static analysis hint flags the inline error message. While not critical, extracting validation error messages to exception class attributes improves maintainability when error messages need localization or standardization across the codebase.

### Current State

#### Validation Errors Using Standard ValueError
The `async_redis_storage.py` file currently has 3 validation errors using standard `ValueError`:

1. **Line 93** - Constructor TTL validation:
   ```python
   if default_ttl is not None and default_ttl < 0:
       raise ValueError("default_ttl must be non-negative")
   ```

2. **Line 364** - Entry ID mismatch validation:
   ```python
   if current_entry.id != complete_entry.id:
       raise ValueError("Entry ID mismatch")
   ```

3. **Lines 512-514** - Request metadata TTL validation:
   ```python
   if ttl_float < 0:
       raise ValueError(
           f"TTL must be non-negative, got {ttl_float}"
       )
   ```

#### No Custom Exceptions Exist
- No `exceptions.py` module exists in `src/cache_manager/`
- No custom exception classes found in broader `src/` codebase
- Codebase currently uses standard Python exceptions throughout

---

## Part 2: Narrow Exception Handling (What We Catch)

### Review Comment Source
From code review of `src/cache_manager/manager.py`:
> Broad exception handling is acceptable here but may need explicit justification
>
> Several blocks (_init_redis_storage, old‑client cleanup in _get_or_create_redis_client, async cache init, and close_async) catch bare Exception and log, which is reasonable for a best‑effort cache layer but will trip BLE001. If you want stricter lint compliance, consider narrowing to RedisError/aiohttp/OSError where applicable, or add comments/# noqa: BLE001 to indicate that dropping unexpected errors is deliberate in these non‑critical paths.

### Current State

#### Broad Exception Handling (BLE001 Violations)
The `manager.py` file has 4 locations catching bare `Exception`:

1. **Line 67** - `_init_redis_storage` method:
   ```python
   except (ValueError, Exception) as e:
       logger.warning(
           f"Redis configuration failed: {e}. "
           "Async (aiohttp) requests will not be cached on Redis."
       )
   ```

2. **Line 103** - Old client cleanup in `_get_or_create_redis_client`:
   ```python
   except Exception as close_error:
       logger.debug(
           "Failed to close previous Redis client: %s", close_error
       )
   ```

3. **Line 187** - Async cache initialization in `get_aiohttp_session`:
   ```python
   except Exception as e:
       logger.warning(f"Failed to initialize async cache: {e}")
       return aiohttp.ClientSession(**session_kwargs)
   ```

4. **Line 211** - `close_async` method:
   ```python
   except Exception as e:
       logger.warning(f"Error closing async Redis client: {e}")
   ```

#### What's Missing
- ❌ No `# noqa: BLE001` comments to suppress linting warnings
- ❌ No explanatory comments justifying broad exception handling
- ❌ No narrowing to specific exception types

---

## Proposed Implementation

### 1. Create Exception Module

Create `src/cache_manager/exceptions.py` with custom exception hierarchy:

```python
"""Custom exceptions for cache manager module."""

from typing import Optional


class CacheError(Exception):
    """Base exception for cache-related errors."""
    pass


class CacheValidationError(CacheError):
    """Raised when cache configuration or data validation fails."""
    pass


class CacheTTLError(CacheValidationError):
    """Raised when TTL validation fails."""

    def __init__(self, message: str, ttl_value: Optional[float] = None):
        super().__init__(message)
        self.ttl_value = ttl_value


class CacheEntryError(CacheError):
    """Raised when cache entry operations fail."""
    pass


class CacheEntryIDMismatchError(CacheEntryError):
    """Raised when entry ID doesn't match expected value."""

    def __init__(self, expected_id: str, actual_id: str):
        super().__init__(f"Entry ID mismatch: expected {expected_id}, got {actual_id}")
        self.expected_id = expected_id
        self.actual_id = actual_id


class CacheConfigurationError(CacheValidationError):
    """Raised when cache configuration is invalid or incomplete."""
    pass


class CacheConnectionError(CacheError):
    """Raised when cache connection operations fail."""
    pass
```

### 2. Update async_redis_storage.py - Custom Exceptions

**Import section:**
```python
from .exceptions import CacheTTLError, CacheEntryIDMismatchError
```

**Line 93 update:**
```python
if default_ttl is not None and default_ttl < 0:
    raise CacheTTLError("default_ttl must be non-negative", ttl_value=default_ttl)
```

**Line 364 update:**
```python
if current_entry.id != complete_entry.id:
    raise CacheEntryIDMismatchError(
        expected_id=str(current_entry.id),
        actual_id=str(complete_entry.id)
    )
```

**Lines 512-514 update:**
```python
if ttl_float < 0:
    raise CacheTTLError(
        f"TTL must be non-negative, got {ttl_float}",
        ttl_value=ttl_float
    )
```

### 3. Update manager.py - Narrow Exception Handling

**Import section:**
```python
import redis.exceptions as redis_exceptions
import aiohttp
from .exceptions import CacheConfigurationError, CacheConnectionError, CacheTTLError
```

**Option A: Narrow to Specific Exceptions (Recommended)**

**Line 67 update (_init_redis_storage):**
```python
except (ValueError, CacheTTLError, CacheConfigurationError) as e:
    logger.warning(
        f"Redis configuration failed: {e}. "
        "Async (aiohttp) requests will not be cached on Redis."
    )
```

**Line 103 update (old client cleanup):**
```python
except (redis_exceptions.RedisError, OSError, asyncio.CancelledError) as close_error:
    logger.debug(
        "Failed to close previous Redis client: %s", close_error
    )
```

**Line 187 update (async cache init):**
```python
except (
    ImportError,
    redis_exceptions.RedisError,
    aiohttp.ClientError,
    CacheConnectionError,
) as e:
    logger.warning(f"Failed to initialize async cache: {e}")
    return aiohttp.ClientSession(**session_kwargs)
```

**Line 211 update (close_async):**
```python
except (redis_exceptions.RedisError, OSError, asyncio.CancelledError) as e:
    logger.warning(f"Error closing async Redis client: {e}")
```

**Option B: Add noqa Comments (If broad catching is justified)**

If keeping broad exception handling for best-effort behavior:

```python
except Exception as e:  # noqa: BLE001 - Best-effort cache, suppress all errors
    logger.warning(f"Failed to initialize async cache: {e}")
    return aiohttp.ClientSession(**session_kwargs)
```

### 4. Update Tests

**Update test assertions in `tests/cache_manager/test_async_redis_storage.py`:**
```python
# Before:
with pytest.raises(ValueError, match="default_ttl must be non-negative"):
    storage = AsyncRedisStorage(default_ttl=-1)

# After:
from src.cache_manager.exceptions import CacheTTLError

with pytest.raises(CacheTTLError, match="default_ttl must be non-negative"):
    storage = AsyncRedisStorage(default_ttl=-1)
```

**Update test assertions in `tests/cache_manager/test_manager.py`:**
```python
# If using Option A (narrow exceptions), verify specific exceptions are raised
# If using Option B (noqa comments), tests remain unchanged
```

---

## Benefits

### Custom Exception Classes Benefits
1. **Type Safety**: Callers can catch specific exception types instead of generic `ValueError`
2. **Structured Error Data**: Exception attributes provide programmatic access to error context
3. **Better Error Messages**: Centralized exception classes ensure consistent error formatting
4. **Localization Ready**: Error messages can be externalized for i18n support
5. **API Clarity**: Custom exceptions serve as documentation of possible error conditions
6. **Testing**: More precise test assertions with specific exception types

### Narrow Exception Handling Benefits
1. **Lint Compliance**: Eliminates BLE001 warnings from static analysis
2. **Better Debugging**: Only expected exceptions are suppressed, unexpected ones bubble up
3. **Code Documentation**: Explicit exception types document what can go wrong
4. **Fail Fast**: Unexpected errors surface immediately instead of being silently logged
5. **Maintenance**: Clear intent when errors are deliberately suppressed vs. overlooked

### Combined Benefits
When custom exceptions are raised AND specifically caught:
- Complete error flow visibility (what's raised → what's caught)
- Type-safe error handling throughout the cache layer
- Clear separation between expected failures (logged) and unexpected bugs (raised)

---

## Implementation Checklist

### Phase 1: Custom Exception Classes
- [ ] Create `src/cache_manager/exceptions.py` with exception hierarchy
- [ ] Add comprehensive docstrings to all exception classes
- [ ] Add type hints to exception constructors
- [ ] Update `async_redis_storage.py` to raise custom exceptions
- [ ] Search for and update other cache_manager files raising ValueError
- [ ] Update test assertions in `tests/cache_manager/test_async_redis_storage.py`
- [ ] Run tests to verify ValueError → custom exception migration

### Phase 2: Narrow Exception Handling
- [ ] Decide: Option A (narrow exceptions) or Option B (noqa comments)
- [ ] If Option A: Research specific exception types from dependencies
  - [ ] Identify redis-py exception types
  - [ ] Identify aiohttp exception types
  - [ ] Identify asyncio exception types
- [ ] Update `manager.py` exception handling (all 4 locations)
- [ ] Add explanatory comments for each catch block
- [ ] Update tests if exception types changed
- [ ] Verify BLE001 linting warnings are resolved

### Phase 3: Integration & Verification
- [ ] Run full test suite: `uv run pytest tests/cache_manager/`
- [ ] Run mypy type checking: `uv run mypy --strict src/cache_manager/`
- [ ] Run linting: `uv run ruff check src/cache_manager/`
- [ ] Test error scenarios manually (invalid config, connection failures, etc.)
- [ ] Review all logging to ensure error context is preserved
- [ ] Update documentation if exception contracts changed

### Phase 4: Documentation
- [ ] Add exception hierarchy diagram to docstrings
- [ ] Document exception handling strategy in cache_manager README
- [ ] Add examples of proper exception handling for cache users
- [ ] Update any API documentation referencing exceptions

---

## Scope Considerations

### Minimal Scope (Recommended for First PR)
**Part 1:**
- Only update the 3 validation errors in `async_redis_storage.py`
- Create minimal exception hierarchy (just what's needed)

**Part 2:**
- Only update the 4 catch blocks in `manager.py`
- Use Option B (noqa comments) if narrowing is too complex initially

### Extended Scope (Future Iteration)
**Part 1:**
- Survey all cache_manager modules for validation errors
- Create comprehensive exception hierarchy
- Consider base exceptions for entire src/ codebase

**Part 2:**
- Audit all exception handling across cache_manager
- Establish exception handling patterns/guidelines
- Consider error recovery strategies beyond logging

---

## Exception Flow Examples

### Before (Current State)
```python
# async_redis_storage.py
if default_ttl < 0:
    raise ValueError("default_ttl must be non-negative")  # Generic exception

# manager.py
try:
    storage = AsyncRedisStorage(default_ttl=config.ttl)
except Exception as e:  # BLE001: Catches everything!
    logger.warning(f"Failed: {e}")
```

### After (Improved State)
```python
# async_redis_storage.py
if default_ttl < 0:
    raise CacheTTLError("default_ttl must be non-negative", ttl_value=default_ttl)

# manager.py
try:
    storage = AsyncRedisStorage(default_ttl=config.ttl)
except (CacheTTLError, CacheConfigurationError) as e:  # Specific, intentional
    logger.warning(f"Cache config failed: {e}")
    # Unexpected errors (bugs) still propagate up!
```

---

## Related Files

### Files to Modify
- `src/cache_manager/exceptions.py` - **NEW** exception module
- `src/cache_manager/async_redis_storage.py` - Raise custom exceptions
- `src/cache_manager/manager.py` - Narrow exception catching
- `tests/cache_manager/test_async_redis_storage.py` - Update test assertions
- `tests/cache_manager/test_manager.py` - Update/add exception handling tests

### Files to Review
- `src/cache_manager/config.py` - May use Pydantic validation already
- `src/cache_manager/aiohttp_adapter.py` - May have exception handling patterns
- `src/cache_manager/result_cache.py` - May raise/catch exceptions

---

## Dependencies to Research

### For Narrow Exception Handling
- **redis-py**: `redis.exceptions.RedisError`, `redis.exceptions.ConnectionError`, etc.
- **aiohttp**: `aiohttp.ClientError`, `aiohttp.ClientConnectionError`, etc.
- **asyncio**: `asyncio.CancelledError`, `asyncio.TimeoutError`
- **Python stdlib**: `OSError`, `IOError`, `ConnectionError`

---

## References

- Original review comments: PR review for AVS-29 Redis HTTP caching
- Python Exception Best Practices: https://docs.python.org/3/tutorial/errors.html#user-defined-exceptions
- Effective Python Item 87: Define a Root Exception to Insulate Callers from APIs
- PEP 8 Exception Naming: https://peps.python.org/pep-0008/#exception-names
- BLE001 Linting Rule: https://docs.astral.sh/ruff/rules/blind-except/
- redis-py exceptions: https://redis-py.readthedocs.io/en/stable/exceptions.html
- aiohttp exceptions: https://docs.aiohttp.org/en/stable/client_reference.html#exceptions
