# Timezone Consistency Analysis Report

**Date**: 2025-10-10
**Analyzed**: 31 JSON files across 3 agent directories (One Piece, Naruto, Dandadan)
**Status**: Analysis Complete - Implementation Pending

---

## Executive Summary

Found **critical timezone inconsistencies** across API sources with different priorities based on anime status:
- **Ongoing anime** (One Piece): Critical - affects real-time scheduling
- **Finished anime** (Naruto, Dandadan): Medium - historical data consistency

**Target Standard**: All datetimes must be stored in **UTC (Coordinated Universal Time)**

---

## Critical Findings

### 1. Episode Air Dates - JST Timezone (+09:00)

**Source**: `episodes_detailed.json` (all agents)
**Problem**: All episode air dates use Japan Standard Time (JST, UTC+09:00) instead of UTC

**Examples**:
```json
// One Piece (ongoing - 1100+ episodes)
"aired": "1999-10-20T00:00:00+09:00"  // ❌ JST

// Naruto (finished - 220 episodes)
"aired": "2002-10-03T00:00:00+09:00"  // ❌ JST

// Dandadan (finished - 12 episodes)
"aired": "2024-10-04T00:00:00+09:00"  // ❌ JST
```

**Impact**:
- 9-hour offset from UTC
- Air date "2024-10-04 00:00 JST" = "2024-10-03 15:00 UTC"
- Causes date boundary issues and sorting problems
- All times are midnight (00:00:00) - not actual broadcast times, just air dates

**Required Fix**: Convert all JST datetimes to UTC

---

### 2. Kitsu API - Mixed Timezones

**Source**: `kitsu.json` (all agents)
**Problem**: Kitsu uses both UTC (Z) and JST (+09:00) in the same file

**Examples**:
```json
// UTC timestamps (metadata)
"createdAt": "2013-02-20T16:00:25.722Z"           // ✅ UTC
"updatedAt": "2025-09-27T19:40:23.914Z"           // ✅ UTC

// JST timestamp (CRITICAL for ongoing anime)
"nextRelease": "2025-09-28T09:30:00.000+09:00"    // ❌ JST - actual broadcast time

// Date-only (no timezone)
"airdate": "1999-10-20"                           // ⚠️ NO TZ
"startDate": "1999-10-20"                         // ⚠️ NO TZ
```

**Impact for Ongoing Anime (One Piece)**:
- Next episode scheduled for **09:30 JST** = **00:30 UTC previous day**
- 9-hour timezone difference causes scheduling errors
- Critical for real-time episode scheduling features

**Required Fix**: Convert `nextRelease` to UTC, document date-only interpretation

---

### 3. Time Precision Analysis

**Sources with Exact Times** (Hour:Minute:Second):

| Source | Field | Precision | Timezone | Use Case |
|--------|-------|-----------|----------|----------|
| **AniList** | `nextAiringEpisode.airingAt` | Seconds | ✅ UTC (unix) | ⭐ Real-time scheduling |
| **Kitsu** | `nextRelease` | Milliseconds | ❌ JST | ⭐ Broadcast time |
| episodes_detailed | `aired` | Seconds | ❌ JST | Date only (00:00:00) |

**Sources with Date-Only** (No Time):

| Source | Field | Format | Timezone |
|--------|-------|--------|----------|
| **AniDB** | `startdate`, `enddate` | YYYY-MM-DD | ❌ None |
| **Anime-Planet** | `start_date`, `end_date` | YYYY-MM-DD | ❌ None |
| **AniSearch** | `start_date`, `end_date` | YYYY-MM-DD | ❌ None |
| **Kitsu** | `airdate`, `startDate` | YYYY-MM-DD | ❌ None |

**Most Accurate for Real-Time Scheduling**:
1. 🥇 **AniList** `nextAiringEpisode.airingAt` - Unix timestamp, UTC, second precision
2. 🥇 **Kitsu** `nextRelease` - ISO 8601, JST, millisecond precision (actual broadcast time)

---

## UTC-Compliant Sources

### ✅ AnimSchedule
```json
{
  "premier": "1999-10-20T00:00:00Z",          // ✅ UTC
  "dubPremier": "2020-11-24T00:00:00Z",       // ✅ UTC
  "delayedFrom": "2025-08-31T00:00:00Z"       // ✅ UTC
}
```
- **Format**: ISO 8601 with `Z` suffix
- **Status**: FULLY COMPLIANT
- **Note**: Null values represented as `0001-01-01T00:00:00Z`

### ✅ Jikan/MyAnimeList
```json
{
  "aired": {
    "from": "1999-10-20T00:00:00+00:00",     // ✅ UTC
    "to": null
  }
}
```
- **Format**: ISO 8601 with `+00:00` offset
- **Status**: FULLY COMPLIANT

### ✅ AniList
```json
{
  "updatedAt": 1759429593,                    // ✅ UTC (unix)
  "nextAiringEpisode": {
    "episode": 1146,
    "airingAt": 1759673760,                   // ✅ UTC (unix)
    "timeUntilAiring": 243184
  }
}
```
- **Format**: Unix timestamp (seconds since epoch)
- **Status**: FULLY COMPLIANT
- **Note**: Unix timestamps are inherently UTC

**Example Conversion**:
- `airingAt: 1759673760` = `2025-10-05 14:16:00 UTC`

---

## Priority Matrix

| Issue | Affects | Priority | Reason |
|-------|---------|----------|--------|
| Kitsu `nextRelease` JST | Ongoing anime | 🔴 **CRITICAL** | Scheduling accuracy for current episodes |
| Episode air dates JST | All anime | 🟡 **MEDIUM** | Historical accuracy, global consistency |
| Date-only formats | All APIs | 🟢 **LOW** | Interpretation documented, not time-critical |
| Kitsu mixed timezones | All anime | 🟡 **MEDIUM** | Internal consistency |

---

## Timezone Distribution

| Timezone | Count | APIs |
|----------|-------|------|
| **UTC (Z or +00:00)** | 4 | AnimSchedule, Jikan, AniList (unix), Kitsu (partial) |
| **JST (+09:00)** | 2 | episodes_detailed.json, Kitsu nextRelease |
| **Date-only (ambiguous)** | 4 | AniDB, Anime-Planet, AniSearch, Kitsu (episodes) |
| **Mixed (UTC + JST)** | 1 | Kitsu |

---

## Required Fixes

### Priority 1: Critical (Ongoing Anime)

#### Fix 1: Convert Kitsu nextRelease from JST to UTC

**Files**: `kitsu.json` (all agents with ongoing anime)

```python
def normalize_kitsu_next_release(kitsu_data: dict) -> dict:
    """Convert Kitsu nextRelease from JST to UTC for ongoing anime."""
    from datetime import datetime, timezone

    if "nextRelease" in kitsu_data.get("anime", {}).get("attributes", {}):
        next_release = kitsu_data["anime"]["attributes"]["nextRelease"]
        if next_release and "+09:00" in next_release:
            # Parse JST datetime and convert to UTC
            dt = datetime.fromisoformat(next_release)
            utc_dt = dt.astimezone(timezone.utc)
            kitsu_data["anime"]["attributes"]["nextRelease"] = utc_dt.isoformat()

    return kitsu_data
```

**Example**:
```
Before: "2025-09-28T09:30:00.000+09:00" (JST - 9:30 AM Japan)
After:  "2025-09-28T00:30:00.000+00:00" (UTC - 12:30 AM)
         ↑ Same moment in time, different timezone representation
```

---

### Priority 2: Medium (All Anime - Consistency)

#### Fix 2: Convert Episode Air Dates from JST to UTC

**Files**: `episodes_detailed.json` (all agents)

```python
def normalize_episode_air_dates(episodes: list) -> list:
    """Convert all episode air dates from JST to UTC."""
    from datetime import datetime, timezone

    for episode in episodes:
        if "aired" in episode and episode["aired"] and "+09:00" in episode["aired"]:
            dt = datetime.fromisoformat(episode["aired"])
            utc_dt = dt.astimezone(timezone.utc)
            episode["aired"] = utc_dt.isoformat()

    return episodes
```

**Example**:
```
Before: "2024-10-04T00:00:00+09:00" (midnight JST)
After:  "2024-10-03T15:00:00+00:00" (3 PM UTC previous day)
         ↑ Date boundary shift - episode aired on Oct 3 in UTC
```

**Impact**:
- Historical air dates become UTC-consistent
- Date boundaries shift by 9 hours
- Sorting and filtering work correctly across timezones

---

#### Fix 3: Standardize Date-Only Formats to UTC

**Files**: `anidb.json`, `anime_planet.json`, `anisearch.json`, `kitsu.json`

```python
def normalize_date_only_to_utc(date_str: str, is_air_date: bool = True) -> str:
    """Convert date-only string to ISO 8601 UTC datetime.

    Args:
        date_str: Date in YYYY-MM-DD format
        is_air_date: If True, assumes midnight JST and converts to UTC
                     If False, assumes midnight UTC

    Returns:
        ISO 8601 datetime string in UTC
    """
    from datetime import datetime, timezone, timedelta

    if is_air_date:
        # For anime air dates: assume midnight JST, convert to UTC
        jst_datetime = f"{date_str}T00:00:00+09:00"
        dt = datetime.fromisoformat(jst_datetime)
        utc_dt = dt.astimezone(timezone.utc)
        return utc_dt.isoformat()
    else:
        # For generic dates: assume midnight UTC
        return f"{date_str}T00:00:00+00:00"
```

**Examples**:
```
Air date:     "2024-10-04" → "2024-10-03T15:00:00+00:00" (midnight JST → 3 PM UTC prev day)
Generic date: "2024-10-04" → "2024-10-04T00:00:00+00:00" (midnight UTC)
```

---

### Priority 3: Low (Documentation)

#### Fix 4: Document Timezone Conventions

Create `docs/timezone_conventions.md`:

```markdown
# Timezone Conventions

## Standard
- **All datetimes MUST be stored in UTC**
- **Format**: ISO 8601 with explicit timezone (`Z` or `+00:00`)
- **Unix timestamps**: Already UTC (no conversion needed)

## Source-Specific Rules

### Date-Only Formats
- **Anime air dates**: Assume midnight JST (UTC+9), convert to UTC
- **Generic dates**: Assume midnight UTC
- **Examples**:
  - "2024-10-04" (air date) = "2024-10-04T00:00:00+09:00" JST = "2024-10-03T15:00:00Z" UTC
  - "2024-10-04" (created) = "2024-10-04T00:00:00Z" UTC

### API Timezone Matrix
| API | Timezone | Conversion Needed |
|-----|----------|-------------------|
| AnimSchedule | UTC (Z) | ❌ No |
| Jikan | UTC (+00:00) | ❌ No |
| AniList | UTC (unix) | ❌ No |
| Kitsu nextRelease | JST (+09:00) | ✅ Yes |
| episodes_detailed | JST (+09:00) | ✅ Yes |
| AniDB, Anime-Planet, AniSearch | None (date-only) | ✅ Yes |

## Presentation Layer
- Store in UTC, convert to user's timezone only when displaying
- Never store user's local timezone in database
- Use ISO 8601 for all datetime serialization
```

---

## Implementation Plan

### Phase 1: Immediate Fixes (Week 1)
1. ✅ Create `src/enrichment/utils/datetime_normalizer.py` utility module
2. ✅ Implement JST → UTC conversion function
3. ✅ Add timezone validation to data ingestion pipeline
4. ✅ Convert Kitsu `nextRelease` for ongoing anime
5. ✅ Add tests for timezone conversion edge cases

### Phase 2: Data Normalization (Week 2)
1. ✅ Convert `episodes_detailed.json` air dates to UTC (all agents)
2. ✅ Convert date-only formats to UTC with documented rules
3. ✅ Add air date vs. generic date detection logic
4. ✅ Update all affected JSON files

### Phase 3: Validation & Testing (Week 3)
1. ✅ Add timezone validation tests
2. ✅ Verify all datetimes are UTC after conversion
3. ✅ Test edge cases (date boundaries, leap years)
4. ✅ Update documentation and API specs

---

## Validation Checklist

After implementation, verify:
- [ ] All episode air dates are in UTC (`Z` or `+00:00`)
- [ ] No JST (+09:00) timestamps remain (except as source data backup)
- [ ] Kitsu `nextRelease` is UTC for ongoing anime
- [ ] Date-only formats documented with interpretation rules
- [ ] Unix timestamps unchanged (already UTC)
- [ ] Data sorting works correctly across timezones
- [ ] Date boundary logic handles UTC correctly
- [ ] Real-time scheduling features use UTC
- [ ] Presentation layer converts to user's local timezone

---

## Critical Warnings

1. **DON'T convert Unix timestamps** - they're already UTC
2. **DON'T change AnimSchedule, Jikan, or AniList data** - already UTC-compliant
3. **DO preserve original data** - create backup before conversion
4. **DO document assumptions** - air date = JST midnight, generic = UTC midnight
5. **DO test date boundaries** - October 3 15:00 UTC vs October 4 00:00 JST
6. **DO handle null values** - check for null before conversion
7. **DO maintain timezone info** - always include `Z` or `+00:00` in output

---

## Example Conversions

### Kitsu nextRelease (Ongoing Anime)
```
Source:    "nextRelease": "2025-09-28T09:30:00.000+09:00"
Converted: "nextRelease": "2025-09-28T00:30:00.000+00:00"
```

### Episode Air Date
```
Source:    "aired": "1999-10-20T00:00:00+09:00"
Converted: "aired": "1999-10-19T15:00:00+00:00"
```

### Date-Only Air Date
```
Source:    "startdate": "2024-10-04"
Converted: "startdate": "2024-10-03T15:00:00+00:00"
```

---

## Technical Notes

### JST to UTC Conversion
- Japan Standard Time (JST) = UTC+9
- No daylight saving time in Japan
- Subtract 9 hours to convert JST to UTC
- October 4, 2024 00:00 JST = October 3, 2024 15:00 UTC

### Unix Timestamp Handling
- Unix timestamps represent seconds since January 1, 1970 00:00:00 UTC
- Always in UTC by definition
- No conversion needed
- Example: `1759673760` = `2025-10-05 14:16:00 UTC`

### ISO 8601 Format
- Standard: `YYYY-MM-DDTHH:mm:ss.sss+HH:MM`
- UTC suffix: `Z` (equivalent to `+00:00`)
- Examples:
  - `2024-10-04T00:00:00Z` (UTC with Z)
  - `2024-10-04T00:00:00+00:00` (UTC with offset)
  - `2024-10-04T00:00:00+09:00` (JST with offset)

---

## Files Affected

### Critical (Ongoing Anime)
- `temp/One_agent1/kitsu.json` - nextRelease field
- `temp/One_agent1/episodes_detailed.json` - all episode air dates

### Medium (All Anime)
- `temp/*/episodes_detailed.json` - all agents
- `temp/*/anidb.json` - date-only fields
- `temp/*/anime_planet.json` - date-only fields
- `temp/*/anisearch.json` - date-only fields
- `temp/*/kitsu.json` - airdate, startDate fields

### No Changes Required
- `temp/*/animeschedule.json` - already UTC
- `temp/*/jikan.json` - already UTC
- `temp/*/anilist.json` - already UTC (unix timestamps)

---

## References

- ISO 8601: https://en.wikipedia.org/wiki/ISO_8601
- Unix Time: https://en.wikipedia.org/wiki/Unix_time
- Japan Standard Time: https://en.wikipedia.org/wiki/Japan_Standard_Time
- Python datetime: https://docs.python.org/3/library/datetime.html
- AniList API: https://anilist.gitbook.io/anilist-apiv2-docs/
- Kitsu API: https://kitsu.docs.apiary.io/
- Jikan API: https://docs.api.jikan.moe/

---

**Status**: Analysis complete, implementation pending
**Next Step**: Create `datetime_normalizer.py` utility module
**Owner**: TBD
**Deadline**: TBD
